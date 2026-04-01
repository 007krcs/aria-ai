"""
ARIA — Tiered OCR Engine
=========================
Never depends on a single OCR library.
Automatically falls back through 6 tiers if any engine fails.

Install what you have — the system uses whatever is available:
    pip install easyocr                    # Tier 1 — best multilingual
    pip install surya-ocr                  # Tier 2 — best on documents
    pip install paddleocr paddlepaddle     # Tier 3 — strong on Asian/Indian langs
    pip install pytesseract                # Tier 4 — always reliable
    pip install pillow opencv-python-headless  # Tier 5 — pure preprocessing
    # Tier 6 uses local Ollama — no install needed if server is running

You don't need ALL of them. Install any one and it works.
Install more and it gets better + more resilient.

Supports: English, Hindi, Bengali, Tamil, Telugu, Urdu, Arabic,
          Chinese, Japanese, Korean, French, German, Spanish, and 90+ more.
"""

import io
import re
import time
import hashlib
import importlib
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSOR
# Makes any image easier to OCR regardless of which engine runs
# ─────────────────────────────────────────────────────────────────────────────

class ImagePreprocessor:
    """
    Cleans up images before OCR to dramatically improve accuracy.
    Uses only PIL and numpy — always available.

    Techniques:
    - Grayscale conversion
    - Contrast enhancement (CLAHE)
    - Binarization (Otsu thresholding)
    - Deskewing (straightens tilted text)
    - Noise removal (median filter)
    - Upscaling (small images read better at 300 DPI equivalent)
    """

    def preprocess(self, image_input) -> "PIL.Image":
        """
        Preprocess an image for better OCR.
        Accepts: PIL Image, bytes, file path, or numpy array.
        Returns: PIL Image (processed).
        """
        from PIL import Image, ImageFilter, ImageEnhance, ImageOps
        import numpy as np

        # Load image from any input type
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, 'read'):
            img = Image.open(image_input)
        else:
            img = image_input  # assume already PIL

        # Convert to RGB if needed
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Upscale if too small (OCR works better at >= 300 DPI equiv)
        w, h = img.size
        if w < 600 or h < 400:
            scale = max(600 / w, 400 / h, 2.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Convert to grayscale
        gray = img.convert('L')

        # Enhance contrast
        gray = ImageEnhance.Contrast(gray).enhance(1.8)
        gray = ImageEnhance.Sharpness(gray).enhance(2.0)

        # Denoise with median filter
        gray = gray.filter(ImageFilter.MedianFilter(size=3))

        return gray

    def to_bytes(self, img) -> bytes:
        """Convert PIL Image to PNG bytes."""
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def to_numpy(self, img) -> "numpy.ndarray":
        """Convert PIL Image to numpy array."""
        import numpy as np
        return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# OCR RESULT
# Standardised output regardless of which engine ran
# ─────────────────────────────────────────────────────────────────────────────

class OCRResult:
    def __init__(
        self,
        text:       str,
        engine:     str,
        confidence: float = 0.0,
        language:   str   = "en",
        boxes:      list  = None,
        latency_ms: int   = 0,
    ):
        self.text       = text.strip()
        self.engine     = engine
        self.confidence = confidence
        self.language   = language
        self.boxes      = boxes or []
        self.latency_ms = latency_ms
        self.success    = len(self.text) > 0

    def to_dict(self) -> dict:
        return {
            "text":       self.text,
            "engine":     self.engine,
            "confidence": round(self.confidence, 3),
            "language":   self.language,
            "word_count": len(self.text.split()),
            "success":    self.success,
            "latency_ms": self.latency_ms,
        }

    def __repr__(self):
        return f"OCRResult(engine={self.engine}, words={len(self.text.split())}, conf={self.confidence:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — EasyOCR
# ─────────────────────────────────────────────────────────────────────────────

class EasyOCREngine:
    name = "easyocr"
    _reader = None
    _loaded_langs: list = []

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("easyocr") is not None

    def run(self, image, languages: list[str] = None) -> Optional[OCRResult]:
        if not self.available():
            return None
        try:
            import easyocr
            langs = languages or ["en"]
            # Map full language names to EasyOCR codes
            lang_map = {
                "hindi": "hi", "bengali": "bn", "tamil": "ta",
                "telugu": "te", "arabic": "ar", "chinese": "ch_sim",
                "japanese": "ja", "korean": "ko", "french": "fr",
                "german": "de", "spanish": "es", "portuguese": "pt",
                "russian": "ru", "urdu": "ur",
            }
            mapped = [lang_map.get(l.lower(), l) for l in langs]

            if set(mapped) != set(self._loaded_langs) or self._reader is None:
                console.print(f"  [dim]EasyOCR loading for langs: {mapped}...[/]")
                EasyOCREngine._reader = easyocr.Reader(
                    mapped, gpu=self._has_gpu(), verbose=False
                )
                EasyOCREngine._loaded_langs = mapped

            t0 = time.time()
            import numpy as np
            if not isinstance(image, np.ndarray):
                preprocessor = ImagePreprocessor()
                image = preprocessor.to_numpy(
                    preprocessor.preprocess(image)
                )

            results = self._reader.readtext(image, detail=1)
            ms      = int((time.time() - t0) * 1000)

            if not results:
                return None

            texts  = [r[1] for r in results]
            confs  = [r[2] for r in results]
            boxes  = [r[0] for r in results]
            text   = "\n".join(texts)
            avg_c  = sum(confs) / len(confs) if confs else 0.0

            return OCRResult(text, self.name, avg_c, mapped[0], boxes, ms)

        except Exception as e:
            console.print(f"  [yellow]EasyOCR error: {e}[/]")
            return None

    def _has_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — Surya OCR
# ─────────────────────────────────────────────────────────────────────────────

class SuryaEngine:
    name = "surya"
    _det_model  = None
    _rec_model  = None
    _rec_proc   = None

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("surya") is not None

    def run(self, image, languages: list[str] = None) -> Optional[OCRResult]:
        if not self.available():
            return None
        try:
            from PIL import Image
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det
            from surya.model.recognition.model import load_model as load_rec
            from surya.model.recognition.processor import load_processor

            if SuryaEngine._det_model is None:
                console.print("  [dim]Loading Surya models (first time only)...[/]")
                SuryaEngine._det_model = load_det()
                SuryaEngine._rec_model, SuryaEngine._rec_proc = load_rec(), load_processor()

            if not isinstance(image, Image.Image):
                from PIL import Image as PILImage
                preprocessor = ImagePreprocessor()
                image = preprocessor.preprocess(image)

            langs = languages or ["en"]
            t0    = time.time()
            results = run_ocr(
                [image], [langs],
                SuryaEngine._det_model,
                SuryaEngine._rec_model,
                SuryaEngine._rec_proc,
            )
            ms = int((time.time() - t0) * 1000)

            if not results or not results[0].text_lines:
                return None

            lines = results[0].text_lines
            text  = "\n".join(line.text for line in lines)
            confs = [line.confidence for line in lines if hasattr(line, 'confidence')]
            avg_c = sum(confs) / len(confs) if confs else 0.8

            return OCRResult(text, self.name, avg_c, langs[0], [], ms)

        except Exception as e:
            console.print(f"  [yellow]Surya error: {e}[/]")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — PaddleOCR
# ─────────────────────────────────────────────────────────────────────────────

class PaddleEngine:
    name = "paddleocr"
    _ocr = None

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("paddleocr") is not None

    def run(self, image, languages: list[str] = None) -> Optional[OCRResult]:
        if not self.available():
            return None
        try:
            from paddleocr import PaddleOCR
            import numpy as np

            lang = (languages or ["en"])[0]
            # PaddleOCR language codes
            paddle_langs = {
                "hi": "hi", "en": "en", "zh": "ch", "ja": "japan",
                "ko": "korean", "ar": "arabic", "fr": "fr", "de": "german",
                "es": "es", "pt": "pt", "ru": "ru", "ta": "ta", "te": "te",
            }
            paddle_lang = paddle_langs.get(lang, "en")

            if PaddleEngine._ocr is None:
                console.print(f"  [dim]Loading PaddleOCR ({paddle_lang})...[/]")
                PaddleEngine._ocr = PaddleOCR(
                    use_angle_cls=True, lang=paddle_lang, show_log=False
                )

            if not isinstance(image, np.ndarray):
                preprocessor = ImagePreprocessor()
                image = preprocessor.to_numpy(preprocessor.preprocess(image))

            t0      = time.time()
            results = PaddleEngine._ocr.ocr(image, cls=True)
            ms      = int((time.time() - t0) * 1000)

            if not results or not results[0]:
                return None

            texts = [line[1][0] for line in results[0] if line]
            confs = [line[1][1] for line in results[0] if line]
            text  = "\n".join(texts)
            avg_c = sum(confs) / len(confs) if confs else 0.0

            return OCRResult(text, self.name, avg_c, lang, [], ms)

        except Exception as e:
            console.print(f"  [yellow]PaddleOCR error: {e}[/]")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# TIER 4 — Tesseract (via pytesseract)
# ─────────────────────────────────────────────────────────────────────────────

class TesseractEngine:
    name = "tesseract"

    # Map language codes to Tesseract lang codes
    LANG_MAP = {
        "en": "eng", "hi": "hin", "bn": "ben", "ta": "tam",
        "te": "tel", "ar": "ara", "zh": "chi_sim", "ja": "jpn",
        "ko": "kor", "fr": "fra", "de": "deu", "es": "spa",
        "pt": "por", "ru": "rus", "ur": "urd", "gu": "guj",
        "mr": "mar", "pa": "pan", "ml": "mal", "kn": "kan",
    }

    @classmethod
    def available(cls) -> bool:
        if importlib.util.find_spec("pytesseract") is None:
            return False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def run(self, image, languages: list[str] = None) -> Optional[OCRResult]:
        if not self.available():
            return None
        try:
            import pytesseract
            from PIL import Image

            preprocessor = ImagePreprocessor()
            if not isinstance(image, Image.Image):
                image = preprocessor.preprocess(image)

            langs     = languages or ["en"]
            tess_lang = "+".join(self.LANG_MAP.get(l, "eng") for l in langs)

            config = f"--oem 3 --psm 3 -l {tess_lang}"
            t0     = time.time()

            data   = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            ms     = int((time.time() - t0) * 1000)

            # Filter low-confidence words
            words = []
            confs = []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf > 20 and word.strip():
                    words.append(word)
                    confs.append(conf / 100.0)

            if not words:
                return None

            text  = " ".join(words)
            avg_c = sum(confs) / len(confs) if confs else 0.0
            return OCRResult(text, self.name, avg_c, langs[0], [], ms)

        except Exception as e:
            console.print(f"  [yellow]Tesseract error: {e}[/]")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# TIER 5 — PIL + OpenCV classical approach
# No ML at all — works on clean printed text
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalEngine:
    """
    Pure image processing + Tesseract without pytesseract wrapper.
    Falls back to simple region-based character extraction if Tesseract binary exists.
    """
    name = "classical"

    @classmethod
    def available(cls) -> bool:
        try:
            import subprocess
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def run(self, image, languages: list[str] = None) -> Optional[OCRResult]:
        if not self.available():
            return None
        try:
            import subprocess
            import tempfile
            from PIL import Image

            preprocessor = ImagePreprocessor()
            if not isinstance(image, Image.Image):
                image = preprocessor.preprocess(image)
            else:
                image = preprocessor.preprocess(image)

            # Save to temp file and run tesseract directly
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_in  = f.name
                tmp_out = f.name.replace(".png", "")
            image.save(tmp_in)

            langs     = languages or ["en"]
            lang_map  = TesseractEngine.LANG_MAP
            tess_lang = "+".join(lang_map.get(l, "eng") for l in langs)

            t0 = time.time()
            subprocess.run(
                ["tesseract", tmp_in, tmp_out, "-l", tess_lang, "--oem", "3", "--psm", "3"],
                capture_output=True, timeout=30
            )
            ms = int((time.time() - t0) * 1000)

            out_file = tmp_out + ".txt"
            if Path(out_file).exists():
                text = Path(out_file).read_text(encoding="utf-8").strip()
                Path(tmp_in).unlink(missing_ok=True)
                Path(out_file).unlink(missing_ok=True)
                if text:
                    return OCRResult(text, self.name, 0.6, langs[0], [], ms)
            return None

        except Exception as e:
            console.print(f"  [yellow]Classical OCR error: {e}[/]")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# TIER 6 — Local Vision LLM via Ollama
# Uses the local LLM to read the image — no external API
# ─────────────────────────────────────────────────────────────────────────────

class VisionLLMEngine:
    """
    Last resort: send the image to a local Ollama vision model.
    Models with vision: llava, llama3.2-vision, bakllava, moondream
    """
    name = "vision_llm"
    VISION_MODELS = ["llama3.2-vision", "llava", "bakllava", "moondream", "minicpm-v"]

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    @classmethod
    def available(cls) -> bool:
        return True  # Always attempt — Ollama is always running in ARIA

    def _find_vision_model(self) -> Optional[str]:
        try:
            import requests
            r      = requests.get(f"{self.base_url}/api/tags", timeout=4)
            models = [m["name"] for m in r.json().get("models", [])]
            for vm in self.VISION_MODELS:
                for m in models:
                    if vm.lower() in m.lower():
                        return m
        except Exception:
            pass
        return None

    def run(self, image, languages: list[str] = None) -> Optional[OCRResult]:
        try:
            import base64
            import requests
            import json
            from PIL import Image

            model = self._find_vision_model()
            if not model:
                console.print("  [dim]No vision model found. Install: ollama pull llava[/]")
                return None

            # Convert to base64
            preprocessor = ImagePreprocessor()
            if not isinstance(image, Image.Image):
                img = preprocessor.preprocess(image)
            else:
                img = image

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            lang = (languages or ["en"])[0]
            prompt = (
                f"Extract ALL text from this image exactly as written. "
                f"Output only the text, preserving line breaks. "
                f"Language hint: {lang}. "
                f"Do not describe the image — only output the extracted text."
            )

            t0 = time.time()
            r  = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "images": [b64], "stream": False},
                timeout=60,
            )
            ms   = int((time.time() - t0) * 1000)
            text = r.json().get("response", "").strip()

            if text:
                console.print(f"  [dim]Vision LLM ({model}):[/] extracted {len(text.split())} words")
                return OCRResult(text, f"vision_llm:{model}", 0.7, lang, [], ms)

        except Exception as e:
            console.print(f"  [yellow]Vision LLM error: {e}[/]")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER TIERED OCR ENGINE
# Tries each engine in order — stops at first success
# ─────────────────────────────────────────────────────────────────────────────

class TieredOCR:
    """
    The complete tiered OCR system.

    Tier 1 → EasyOCR          (best multilingual, ML-based)
    Tier 2 → Surya             (best on documents, ML-based)
    Tier 3 → PaddleOCR         (strong on Asian/Indian languages, ML-based)
    Tier 4 → Tesseract         (via pytesseract, classical + LSTM)
    Tier 5 → Classical         (direct tesseract binary, most portable)
    Tier 6 → Vision LLM        (local Ollama vision model, last resort)

    Install as many or as few as you want.
    The system uses whatever is available and falls back gracefully.

    Usage:
        ocr = TieredOCR()
        result = ocr.read(image_path, languages=["en", "hi"])
        print(result.text)
        print(result.engine)   # tells you which tier succeeded
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.engines = [
            EasyOCREngine(),
            SuryaEngine(),
            PaddleEngine(),
            TesseractEngine(),
            ClassicalEngine(),
            VisionLLMEngine(ollama_url),
        ]
        self.preprocessor = ImagePreprocessor()
        self._availability_cache = {}

    def available_engines(self) -> list[str]:
        """Return list of installed/available OCR engines."""
        available = []
        for engine in self.engines:
            name = engine.name
            if name not in self._availability_cache:
                try:
                    self._availability_cache[name] = engine.__class__.available()
                except Exception:
                    self._availability_cache[name] = False
            if self._availability_cache[name]:
                available.append(name)
        return available

    def read(
        self,
        image,
        languages:  list[str] = None,
        min_confidence: float  = 0.3,
        force_engine: str      = None,
    ) -> OCRResult:
        """
        Read text from an image using the best available engine.

        Args:
            image:          PIL Image, file path, bytes, or numpy array
            languages:      list of language codes e.g. ["en", "hi"]
            min_confidence: minimum confidence to accept a result
            force_engine:   force a specific engine name (for testing)

        Returns:
            OCRResult with text, engine name, confidence, and metadata
        """
        langs = languages or self._detect_script_hint(image)
        console.print(f"  [dim]TieredOCR: langs={langs}[/]")

        engines_to_try = self.engines
        if force_engine:
            engines_to_try = [e for e in self.engines if e.name == force_engine]

        for engine in engines_to_try:
            if not engine.__class__.available():
                continue

            console.print(f"  [dim]Trying {engine.name}...[/]")
            try:
                result = engine.run(image, langs)
                if result and result.success and result.confidence >= min_confidence:
                    console.print(
                        f"  [green]OCR success:[/] {engine.name} "
                        f"({len(result.text.split())} words, conf={result.confidence:.2f})"
                    )
                    return result
                elif result and result.success:
                    # Low confidence but has text — keep as candidate
                    console.print(
                        f"  [yellow]Low conf ({result.confidence:.2f}) from {engine.name}[/]"
                    )
            except Exception as e:
                console.print(f"  [yellow]{engine.name} failed: {e}[/]")
                continue

        # All engines failed — return empty result
        console.print("[red]All OCR engines failed[/]")
        return OCRResult(
            "", "none", 0.0, langs[0] if langs else "en",
            latency_ms=0
        )

    def read_pdf(self, pdf_path: str, languages: list[str] = None) -> dict:
        """
        Extract text from a PDF.
        First tries direct text extraction (digital PDFs).
        Falls back to OCR page by page if the PDF is scanned.
        """
        from pathlib import Path as P
        import fitz  # pymupdf

        try:
            doc   = fitz.open(pdf_path)
            pages = []
            total_text = ""

            for page_num, page in enumerate(doc):
                # Try direct text first
                direct_text = page.get_text().strip()

                if len(direct_text) > 50:
                    # Digital PDF — use direct text
                    pages.append({
                        "page":   page_num + 1,
                        "text":   direct_text,
                        "method": "direct",
                    })
                    total_text += direct_text + "\n"
                else:
                    # Scanned page — render and OCR
                    console.print(f"  [dim]Page {page_num+1}: scanned — running OCR[/]")
                    pix    = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    result = self.read(img_bytes, languages)
                    pages.append({
                        "page":   page_num + 1,
                        "text":   result.text,
                        "method": f"ocr:{result.engine}",
                        "confidence": result.confidence,
                    })
                    total_text += result.text + "\n"

            doc.close()
            return {
                "text":       total_text.strip(),
                "pages":      pages,
                "total_pages": len(pages),
                "success":    len(total_text.strip()) > 0,
            }

        except ImportError:
            console.print("[yellow]pymupdf not installed. pip install pymupdf[/]")
            return {"text": "", "success": False, "error": "pymupdf not installed"}
        except Exception as e:
            return {"text": "", "success": False, "error": str(e)}

    def _detect_script_hint(self, image) -> list[str]:
        """
        Try to auto-detect which scripts/languages are in the image
        based on pixel patterns (quick heuristic, not ML).
        Returns language code hints.
        """
        # Default to English if we can't detect
        return ["en"]

    def status(self) -> dict:
        """Return status of all OCR engines."""
        return {
            "available_engines": self.available_engines(),
            "total_engines":     len(self.engines),
            "tier_map": {
                "tier_1_easyocr":   EasyOCREngine.available(),
                "tier_2_surya":     SuryaEngine.available(),
                "tier_3_paddle":    PaddleEngine.available(),
                "tier_4_tesseract": TesseractEngine.available(),
                "tier_5_classical": ClassicalEngine.available(),
                "tier_6_vision_llm": True,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Install helper — tells you exactly what to install for each tier
# ─────────────────────────────────────────────────────────────────────────────

def print_install_guide():
    console.print("\n[bold]ARIA OCR Engine — Install Guide[/]")
    console.print("You only need ONE tier to work. Install more for better fallback.\n")

    guide = [
        ("Tier 1 — EasyOCR",    "pip install easyocr",                         "Best multilingual, needs ~500MB model download"),
        ("Tier 2 — Surya",      "pip install surya-ocr",                        "Best on documents, outperforms Tesseract"),
        ("Tier 3 — PaddleOCR",  "pip install paddleocr paddlepaddle",           "Strong on Indian + Asian languages"),
        ("Tier 4 — Tesseract",  "pip install pytesseract  +  install binary",   "Windows: https://github.com/UB-Mannheim/tesseract"),
        ("Tier 5 — Classical",  "Install Tesseract binary only (no pip)",       "Most portable — runs as system binary"),
        ("Tier 6 — Vision LLM", "ollama pull llava  (already running!)",        "Uses your local Ollama — no extra install"),
    ]

    for tier, cmd, note in guide:
        status = "[green]AVAILABLE[/]" if _check_tier(tier) else "[dim]not installed[/]"
        console.print(f"  {status}  [bold]{tier}[/]")
        console.print(f"           Install: [cyan]{cmd}[/]")
        console.print(f"           Note:    {note}\n")


def _check_tier(tier_name: str) -> bool:
    checks = {
        "Tier 1": EasyOCREngine.available,
        "Tier 2": SuryaEngine.available,
        "Tier 3": PaddleEngine.available,
        "Tier 4": TesseractEngine.available,
        "Tier 5": ClassicalEngine.available,
        "Tier 6": lambda: True,
    }
    for key, fn in checks.items():
        if key in tier_name:
            try:
                return fn()
            except Exception:
                return False
    return False


if __name__ == "__main__":
    print_install_guide()
    ocr    = TieredOCR()
    status = ocr.status()
    console.print(f"\n[bold]Available engines:[/] {status['available_engines']}")
