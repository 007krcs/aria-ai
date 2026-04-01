"""
ARIA Vision OCR
===============
OCR powered by local vision LLMs (LLaVA, moondream, llama3.2-vision).
No EasyOCR. No Tesseract. No external dependency beyond Ollama.

Why this is BETTER than EasyOCR:
- EasyOCR pattern-matches character shapes
- Vision LLMs UNDERSTAND the image — reads tables, handwriting,
  rotated text, mixed languages, diagrams with text, screenshots
- Returns structured data (JSON) not just raw characters
- Can answer questions about image content, not just extract text
- Works with any language the base model knows
- Understands context (invoice → extracts fields, not just words)

Models (all free via Ollama):
  ollama pull moondream          # 1.8GB — fast, good for simple text
  ollama pull llava:7b           # 4.7GB — better quality, tables, mixed content
  ollama pull llama3.2-vision    # 2.0GB — best for Hindi/multilingual

Install: just `ollama pull moondream` — nothing else needed
"""

import base64
import json
import re
import time
import requests
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()


class VisionOCR:
    """
    Multimodal OCR using local Ollama vision models.
    Reads any image — scanned docs, screenshots, photos, handwriting.
    """

    # Preferred models in order — first available wins
    VISION_MODELS = [
        "llama3.2-vision",
        "llava:13b",
        "llava:7b",
        "llava",
        "moondream",
        "bakllava",
    ]

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url   = base_url
        self._model     = None

    # ── Model selection ───────────────────────────────────────────────────────

    @property
    def model(self) -> Optional[str]:
        if self._model:
            return self._model
        self._model = self._detect_vision_model()
        return self._model

    def _detect_vision_model(self) -> Optional[str]:
        """Find the best available vision model in Ollama."""
        try:
            r       = requests.get(f"{self.base_url}/api/tags", timeout=4)
            installed = [m["name"] for m in r.json().get("models", [])]
            for preferred in self.VISION_MODELS:
                for inst in installed:
                    if preferred.lower().split(":")[0] in inst.lower():
                        console.print(f"  [green]Vision OCR:[/] using {inst}")
                        return inst
            console.print(
                "  [yellow]No vision model found. Run: ollama pull moondream[/]"
            )
            return None
        except Exception:
            return None

    # ── Core: image to text ───────────────────────────────────────────────────

    def image_to_text(self, image_source: str, prompt: str = None) -> dict:
        """
        Extract all text from an image.

        image_source: file path OR URL OR base64 string
        Returns: {text, language, confidence, model_used, method}
        """
        if not self.model:
            return {"text": "", "error": "No vision model installed. Run: ollama pull moondream"}

        b64 = self._to_base64(image_source)
        if not b64:
            return {"text": "", "error": "Could not load image"}

        ocr_prompt = prompt or (
            "Extract ALL text from this image exactly as it appears. "
            "Preserve formatting, line breaks, and structure. "
            "If there are multiple columns, read left to right, top to bottom. "
            "Return only the extracted text, nothing else."
        )

        t0     = time.time()
        result = self._call_vision(b64, ocr_prompt)
        ms     = int((time.time() - t0) * 1000)

        if result:
            lang = self._detect_language(result)
            return {
                "text":        result.strip(),
                "language":    lang,
                "model_used":  self.model,
                "latency_ms":  ms,
                "method":      "vision_llm",
                "confidence":  0.90,
            }
        return {"text": "", "error": "Vision model returned empty response"}

    def extract_structured(self, image_source: str, doc_type: str = "auto") -> dict:
        """
        Extract structured data from a document image.
        Much smarter than plain OCR — understands document semantics.

        doc_type: "invoice", "receipt", "form", "table", "card", "auto"
        Returns structured JSON extracted from the document.
        """
        if not self.model:
            return {"error": "No vision model installed"}

        b64 = self._to_base64(image_source)
        if not b64:
            return {"error": "Could not load image"}

        type_prompts = {
            "invoice": (
                "Extract all invoice data as JSON: "
                "{invoice_number, date, vendor, items:[{description,qty,price}], total, tax}"
            ),
            "receipt": (
                "Extract receipt data as JSON: "
                "{store, date, items:[{name,price}], subtotal, tax, total}"
            ),
            "form": (
                "Extract all form fields and their values as JSON: "
                "{field_name: field_value}"
            ),
            "table": (
                "Extract this table as JSON array of objects. "
                "Use column headers as keys."
            ),
            "card": (
                "Extract business card info as JSON: "
                "{name, title, company, email, phone, address, website}"
            ),
            "auto": (
                "Analyse this image. Extract all text and structure as JSON. "
                "Detect the document type and extract relevant fields. "
                "Return: {doc_type, extracted_data, raw_text}"
            ),
        }

        prompt = type_prompts.get(doc_type, type_prompts["auto"])
        prompt += "\nReturn ONLY valid JSON, no explanation."

        raw = self._call_vision(b64, prompt)
        if not raw:
            return {"error": "No response from vision model"}

        # Parse JSON from response
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"```json?\s*", "", raw)
                raw = re.sub(r"```\s*", "", raw)
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            # Return raw if JSON parse fails
            return {"raw_text": raw, "parse_error": "Could not parse as JSON"}

    def describe_image(self, image_source: str) -> str:
        """Get a full description of what's in an image."""
        if not self.model:
            return "No vision model installed"
        b64 = self._to_base64(image_source)
        if not b64:
            return "Could not load image"
        return self._call_vision(
            b64,
            "Describe this image in detail. Include all text, objects, people, "
            "colors, layout, and any important information visible."
        ) or ""

    def answer_about_image(self, image_source: str, question: str) -> str:
        """Answer any question about an image."""
        if not self.model:
            return "No vision model installed"
        b64 = self._to_base64(image_source)
        if not b64:
            return "Could not load image"
        return self._call_vision(b64, question) or ""

    def scan_screenshot(self, screenshot_path: str) -> dict:
        """
        Optimised for screenshots — reads UI text, menus, notifications, error messages.
        Useful for the system assistant feature.
        """
        prompt = (
            "This is a screenshot. Extract: "
            "1) All visible text exactly as shown "
            "2) Any error messages or warnings "
            "3) The main application/window visible "
            "4) Any important UI elements\n"
            "Format as JSON: {text, errors, app_name, ui_elements}"
        )
        return self.extract_structured(screenshot_path, "auto")

    # ── Batch processing ──────────────────────────────────────────────────────

    def scan_pdf_pages(self, pdf_path: str) -> list[dict]:
        """
        Convert PDF pages to images and run vision OCR on each.
        Far better than text extraction for scanned PDFs.
        """
        try:
            import fitz  # pymupdf
        except ImportError:
            return [{"error": "pip install pymupdf for PDF processing"}]

        results = []
        try:
            doc = fitz.open(pdf_path)
            console.print(f"  [dim]Vision OCR: scanning {len(doc)} pages...[/]")
            for page_num, page in enumerate(doc):
                # Render page to image at 200 DPI
                pix    = page.get_pixmap(dpi=200)
                b64    = base64.b64encode(pix.tobytes("png")).decode()
                result = self.image_to_text(f"data:image/png;base64,{b64}")
                result["page"] = page_num + 1
                results.append(result)
                console.print(f"  [dim]  Page {page_num+1}/{len(doc)}: {len(result.get('text',''))} chars[/]")
            doc.close()
        except Exception as e:
            results.append({"error": str(e)})

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_vision(self, b64_image: str, prompt: str) -> Optional[str]:
        """Make a multimodal call to Ollama vision model."""
        # Strip data URL prefix if present
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]

        payload = {
            "model":  self.model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {"temperature": 0.05, "num_predict": 1024},
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            console.print(f"  [yellow]Vision API error: {r.status_code}[/]")
            return None
        except Exception as e:
            console.print(f"  [yellow]Vision call failed: {e}[/]")
            return None

    def _to_base64(self, source: str) -> Optional[str]:
        """Convert file path, URL, or existing base64 to base64 string."""
        # Already base64
        if source.startswith("data:image") or (len(source) > 100 and "/" not in source[:50]):
            return source

        # File path
        if not source.startswith("http"):
            try:
                return base64.b64encode(Path(source).read_bytes()).decode()
            except Exception as e:
                console.print(f"  [yellow]Image load error: {e}[/]")
                return None

        # URL
        try:
            r = requests.get(source, timeout=10)
            return base64.b64encode(r.content).decode()
        except Exception as e:
            console.print(f"  [yellow]Image URL error: {e}[/]")
            return None

    def _detect_language(self, text: str) -> str:
        try:
            from langdetect import detect
            return detect(text[:300])
        except Exception:
            return "unknown"

    def is_available(self) -> bool:
        return self.model is not None

    def status(self) -> dict:
        return {
            "available":   self.is_available(),
            "model":       self.model,
            "preferred":   self.VISION_MODELS,
            "install_cmd": "ollama pull moondream" if not self.is_available() else None,
        }
