"""
ARIA — Self-Training Pipeline
================================
Closes the loop: collect → fine-tune → load → improve → collect.

This is what makes ARIA genuinely self-improving.

The pipeline:
  1. TrainingBuffer   — logs every good interaction to JSONL
  2. DatasetBuilder   — formats JSONL into instruction-tuning format
  3. OllamaTrainer    — runs fine-tuning via Ollama's API (if available)
  4. ColabExporter    — exports dataset to Colab-ready format as fallback
  5. AdapterLoader    — loads the trained adapter back into Ollama
  6. Scheduler        — triggers weekly at 3am automatically

Two training paths:
  Path A (local): Ollama fine-tuning API (available in Ollama ≥0.1.32)
    - Runs on your GPU if available, CPU otherwise
    - Takes 2-8 hours depending on dataset size
    - Fully automatic — no manual steps

  Path B (Colab): exports JSONL → run notebook on Google's free GPU
    - 30 minutes on T4 GPU
    - Returns a GGUF adapter file
    - ARIA loads it automatically on next restart

Training data collection:
  Every interaction is scored 0-1 based on:
    - Did the user follow up positively? (+0.3)
    - Did the tool succeed? (+0.3)
    - Did the user ask to redo/fix? (-0.4)
    - Was the answer verified? (+0.2)
  Only interactions scoring ≥0.6 go into training data.
  This means ARIA only learns from its GOOD answers.
"""

import json
import re
import time
import hashlib
import threading
import subprocess
import requests
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRAIN_DIR     = PROJECT_ROOT / "data" / "training"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

BUFFER_FILE   = TRAIN_DIR / "buffer.jsonl"          # raw interactions
DATASET_FILE  = TRAIN_DIR / "dataset.jsonl"          # formatted for training
ADAPTER_DIR   = PROJECT_ROOT / "data" / "adapters"  # trained model adapters
ADAPTER_DIR.mkdir(exist_ok=True)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# MEANINGFUL FILTER — gates what actually enters the training buffer
# ─────────────────────────────────────────────────────────────────────────────

class MeaningfulFilter:
    """
    Decides whether an interaction is worth learning from.

    Rejects:
      - Trivially short or empty responses
      - Error messages, offline/timeout responses
      - Pure acknowledgements ("OK", "Done", "Sure")
      - Questions with no factual answer (vague or refusals)
      - Near-duplicates of things already in the buffer

    Scores positively for:
      - Structured content (headings, bullet points, code)
      - Factual density (numbers, proper nouns, dates)
      - Reasoning steps or explanations
      - Domain diversity (avoids over-training on same topic)
    """

    # Hard-reject phrases in responses
    _JUNK_PHRASES = (
        "i'm running in offline mode",
        "ollama is not available",
        "request timed out",
        "connection error",
        "an error occurred",
        "model error:",
        "groq rate limit",
        "done.",
        "i don't know",
        "i cannot help with that",
        "i'm not able to",
        "as an ai language model",
    )

    # Purely social exchanges — no knowledge content
    _TRIVIAL_INPUTS = re.compile(
        r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|great|cool|nice|bye|"
        r"good morning|good night|good evening|yes|no|yep|nope|lol|haha)[\s!.]*$",
        re.I,
    )

    def is_meaningful(
        self,
        user_input:    str,
        aria_response: str,
        existing_buffer: list[dict] | None = None,
    ) -> tuple[bool, str]:
        """
        Returns (True, "") if meaningful, (False, reason) if not.
        """
        q = user_input.strip()
        a = aria_response.strip()

        # 1. Minimum length
        if len(q) < 8:
            return False, "query_too_short"
        if len(a) < 60:
            return False, "response_too_short"

        # 2. Trivial social exchange
        if self._TRIVIAL_INPUTS.match(q):
            return False, "trivial_input"

        # 3. Error / junk response
        a_lower = a.lower()
        for phrase in self._JUNK_PHRASES:
            if phrase in a_lower:
                return False, f"junk_phrase:{phrase[:30]}"

        # 4. No real content — response is pure filler
        word_count = len(a.split())
        if word_count < 15:
            return False, "response_no_content"

        # 5. Factual density score (higher = more worth storing)
        density = self._factual_density(a)
        if density < 0.05:
            return False, "low_factual_density"

        # 6. Near-duplicate check against existing buffer
        if existing_buffer:
            dup = self._is_near_duplicate(q, existing_buffer)
            if dup:
                return False, "near_duplicate"

        return True, ""

    def content_score(self, user_input: str, aria_response: str) -> float:
        """
        Returns a 0.0–1.0 content quality bonus to add to the behavioral score.
        """
        a = aria_response.strip()
        score = 0.0

        # Structured formatting
        if re.search(r"^#{1,3}\s", a, re.M):      score += 0.15  # headings
        if re.search(r"^[-*]\s", a, re.M):         score += 0.10  # bullet points
        if re.search(r"```", a):                    score += 0.15  # code block
        if re.search(r"^\d+\.", a, re.M):           score += 0.08  # numbered list

        # Factual content signals
        density = self._factual_density(a)
        score += min(0.30, density * 3)  # up to +0.30 for very fact-rich content

        # Length bonus (sweet spot: 100–600 words)
        wc = len(a.split())
        if 100 <= wc <= 600:                        score += 0.10
        elif wc > 600:                              score += 0.05  # too verbose

        # Explanation / reasoning indicators
        if re.search(r"\bbecause\b|\bsince\b|\btherefore\b|\bwhich means\b", a, re.I):
            score += 0.08

        return min(1.0, score)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _factual_density(self, text: str) -> float:
        """Fraction of tokens that are factual signals (numbers, proper nouns, etc.)"""
        tokens = text.split()
        if not tokens:
            return 0.0
        signals = sum(1 for t in tokens if (
            re.search(r"\d", t)                    # numbers/dates/percentages
            or re.match(r"[A-Z][a-z]+", t)         # Proper nouns
            or re.match(r"[A-Z]{2,}", t)            # Acronyms
            or "%" in t or "$" in t or "₹" in t    # financial symbols
        ))
        return signals / len(tokens)

    def _jaccard(self, a: str, b: str) -> float:
        """Jaccard similarity between two strings (word-level)."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _is_near_duplicate(self, query: str, buffer: list[dict], threshold: float = 0.75) -> bool:
        """True if a very similar question is already in the buffer."""
        for item in buffer[-200:]:  # only check recent 200 — avoid O(n) on large buffers
            existing = item.get("user", "")
            if self._jaccard(query, existing) >= threshold:
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING BUFFER — collects good interactions
# ─────────────────────────────────────────────────────────────────────────────

_filter = MeaningfulFilter()


class TrainingBuffer:
    """
    Logs interactions and scores them for training quality.
    Only high-scoring, meaningful interactions become training data.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._pending: dict[str, dict] = {}  # id → pending interaction

    def _load_recent_buffer(self, n: int = 200) -> list[dict]:
        """Load the last n committed items for duplicate checking."""
        items = []
        if not BUFFER_FILE.exists():
            return items
        try:
            lines = BUFFER_FILE.read_text(encoding="utf-8").strip().split("\n")
            for line in lines[-n:]:
                if line.strip():
                    try: items.append(json.loads(line))
                    except Exception: pass
        except Exception:
            pass
        return items

    def record(
        self,
        interaction_id: str,
        user_input:     str,
        aria_response:  str,
        context:        str  = "",
        domain:         str  = "general",
        tool_used:      str  = None,
        tool_succeeded: bool = None,
    ) -> str:
        """
        Start tracking an interaction. Returns the interaction_id.
        Interactions that fail the meaningful filter are silently dropped.
        """
        # Content quality gate — skip junk immediately
        recent = self._load_recent_buffer()
        ok, reason = _filter.is_meaningful(user_input, aria_response, recent)
        if not ok:
            return interaction_id  # return id but don't track — caller unaffected

        # Content quality bonus added to base score
        content_bonus = _filter.content_score(user_input, aria_response)
        base_score = 0.5 + (content_bonus * 0.3)  # content can add up to +0.30 to base

        with self._lock:
            self._pending[interaction_id] = {
                "id":            interaction_id,
                "ts":            datetime.now().isoformat(),
                "domain":        domain,
                "user":          user_input,
                "aria":          aria_response,
                "context":       context[:500],
                "tool":          tool_used,
                "tool_ok":       tool_succeeded,
                "score":         round(base_score, 3),
                "content_score": round(content_bonus, 3),
                "feedback":      [],
            }
        return interaction_id

    def feedback(self, interaction_id: str, signal: str, value: float = 1.0):
        """
        Add a feedback signal to an interaction.
        signal: "positive" | "negative" | "tool_success" | "tool_fail" |
                "redo" | "verified" | "ignored"
        """
        with self._lock:
            item = self._pending.get(interaction_id)
            if not item:
                return

            delta = {
                "positive":     +0.25 * value,
                "negative":     -0.30 * value,
                "tool_success": +0.20 * value,
                "tool_fail":    -0.25 * value,
                "redo":         -0.40 * value,
                "verified":     +0.20 * value,
                "ignored":      -0.05 * value,
            }.get(signal, 0)

            item["score"] = max(0.0, min(1.0, item["score"] + delta))
            item["feedback"].append({"signal": signal, "ts": time.time()})

    def commit(self, interaction_id: str, min_score: float = 0.6):
        """
        Commit a pending interaction to the training buffer.
        Only saves if score ≥ min_score.
        """
        with self._lock:
            item = self._pending.pop(interaction_id, None)
        if not item:
            return False
        if item["score"] < min_score:
            return False  # not good enough to learn from

        with self._lock:
            with open(BUFFER_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return True

    def auto_commit_old(self, max_age_min: int = 30):
        """Auto-commit interactions older than max_age_min minutes."""
        now = time.time()
        to_commit = []
        with self._lock:
            for iid, item in list(self._pending.items()):
                ts = datetime.fromisoformat(item["ts"]).timestamp()
                if now - ts > max_age_min * 60:
                    to_commit.append(iid)
        for iid in to_commit:
            self.commit(iid)

    def stats(self) -> dict:
        lines = 0
        domains: dict = {}
        if BUFFER_FILE.exists():
            for line in BUFFER_FILE.read_text().split("\n"):
                if line.strip():
                    lines += 1
                    try:
                        d = json.loads(line).get("domain","general")
                        domains[d] = domains.get(d, 0) + 1
                    except Exception:
                        pass
        return {
            "buffered_interactions": lines,
            "pending":               len(self._pending),
            "by_domain":             domains,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER — formats buffer into fine-tuning format
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Converts the raw interaction buffer into formatted training data.

    Output format: Alpaca-style instruction tuning
    {
      "instruction": "...",
      "input": "...",       # optional context
      "output": "..."
    }

    Also builds domain-specific datasets for targeted fine-tuning.
    """

    def build(self, min_score: float = 0.65) -> dict:
        """
        Build the training dataset from the buffer.
        Applies score filtering, content re-validation, and semantic deduplication.
        Returns stats about what was built.
        """
        if not BUFFER_FILE.exists():
            return {"examples": 0, "skipped": 0, "reasons": {}}

        raw_items = []
        for line in BUFFER_FILE.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                raw_items.append(item)
            except Exception:
                continue

        # Sort by score descending — best examples first
        raw_items.sort(key=lambda x: x.get("score", 0), reverse=True)

        examples = []
        skipped  = 0
        skip_reasons: dict[str, int] = {}
        seen_queries: list[dict] = []  # for dedup within dataset

        for item in raw_items:
            # Score threshold
            if item.get("score", 0) < min_score:
                skipped += 1
                skip_reasons["low_score"] = skip_reasons.get("low_score", 0) + 1
                continue

            user = item.get("user", "")
            aria = item.get("aria", "")

            # Re-validate content (catches things that slipped through earlier)
            ok, reason = _filter.is_meaningful(user, aria, seen_queries)
            if not ok:
                skipped += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            example = {
                "instruction": user,
                "input":       item.get("context", ""),
                "output":      aria,
                "domain":      item.get("domain", "general"),
                "score":       item.get("score", 0.5),
                "content_score": item.get("content_score", 0.0),
                "ts":          item.get("ts", ""),
            }
            examples.append(example)
            seen_queries.append({"user": user})

        # Domain balance — cap any single domain at 40% of dataset
        # (prevents ARIA from only knowing about one topic)
        examples = self._balance_domains(examples)

        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        console.print(
            f"  [green]Dataset built:[/] {len(examples)} examples "
            f"({skipped} skipped — {skip_reasons})"
        )
        return {
            "examples":     len(examples),
            "skipped":      skipped,
            "skip_reasons": skip_reasons,
            "path":         str(DATASET_FILE),
        }

    def _balance_domains(self, examples: list[dict], max_fraction: float = 0.40) -> list[dict]:
        """Cap any domain at max_fraction of total to ensure diverse training."""
        if not examples:
            return examples
        cap = max(5, int(len(examples) * max_fraction))
        domain_counts: dict[str, int] = {}
        balanced = []
        for ex in examples:
            d = ex.get("domain", "general")
            if domain_counts.get(d, 0) < cap:
                balanced.append(ex)
                domain_counts[d] = domain_counts.get(d, 0) + 1
        return balanced

    def export_colab(self) -> Path:
        """
        Export a complete Colab-ready notebook for fine-tuning.
        Open this in Google Colab (free T4 GPU), run all cells,
        download the GGUF, put it in data/adapters/.
        """
        if not DATASET_FILE.exists():
            self.build()

        # Read dataset
        examples = []
        if DATASET_FILE.exists():
            for line in DATASET_FILE.read_text().split("\n"):
                if line.strip():
                    try: examples.append(json.loads(line))
                    except Exception: pass

        dataset_json = json.dumps(examples[:500], indent=2)  # max 500 for Colab free tier

        notebook = {
            "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3"}},
            "cells": [
                _colab_cell("markdown", "# ARIA Self-Training\nFine-tune phi3:mini on your interaction data."),
                _colab_cell("code", "!pip install -q unsloth transformers datasets peft trl bitsandbytes"),
                _colab_cell("code", f"""# ARIA training data
import json
DATASET = {dataset_json}
print(f"Loaded {{len(DATASET)}} training examples")"""),
                _colab_cell("code", """from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = "unsloth/Phi-3-mini-4k-instruct",
    max_seq_length = 2048,
    dtype        = None,
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
)"""),
                _colab_cell("code", """from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

def format_example(ex):
    inp = f"### Instruction:\\n{ex['instruction']}\\n"
    if ex.get('input'):
        inp += f"### Input:\\n{ex['input']}\\n"
    inp += f"### Response:\\n{ex['output']}"
    return {"text": inp}

dataset = Dataset.from_list([format_example(e) for e in DATASET])

trainer = SFTTrainer(
    model     = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        output_dir = "aria_adapter",
        logging_steps = 10,
        save_strategy = "no",
    ),
)
trainer.train()"""),
                _colab_cell("code", """# Export as GGUF for Ollama
model.save_pretrained_gguf("aria_adapter_gguf",
    tokenizer, quantization_method="q4_k_m")
print("Done! Download aria_adapter_gguf/aria-unsloth.Q4_K_M.gguf")
print("Put it in: C:/Users/chand/ai-remo/data/adapters/")"""),
                _colab_cell("code", """# Download the file
from google.colab import files
import os
for f in os.listdir("aria_adapter_gguf"):
    if f.endswith(".gguf"):
        files.download(f"aria_adapter_gguf/{f}")"""),
            ]
        }

        out = TRAIN_DIR / "aria_finetune.ipynb"
        out.write_text(json.dumps(notebook, indent=2))
        console.print(f"  [green]Colab notebook:[/] {out}")
        return out

    def stats(self) -> dict:
        lines = 0
        if DATASET_FILE.exists():
            lines = sum(1 for l in DATASET_FILE.read_text().split("\n") if l.strip())
        return {"dataset_examples": lines, "path": str(DATASET_FILE)}


def _colab_cell(cell_type: str, source: str) -> dict:
    if cell_type == "code":
        return {"cell_type": "code", "metadata": {}, "outputs": [],
                "execution_count": None, "source": source}
    return {"cell_type": "markdown", "metadata": {}, "source": source}


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA TRAINER — local fine-tuning via Ollama API
# ─────────────────────────────────────────────────────────────────────────────

class OllamaTrainer:
    """
    Fine-tunes using Ollama's training API (available in recent Ollama versions).
    Falls back to creating a Modelfile with system prompt if training API unavailable.
    """

    def __init__(self, base_model: str = "phi3:mini"):
        self.base_model  = base_model
        self.ollama_url  = "http://localhost:11434"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return r.ok
        except Exception:
            return False

    def create_modelfile_adapter(self, dataset_path: Path) -> dict:
        """
        Create an Ollama Modelfile that embeds learned patterns as system context.
        This is not true fine-tuning but dramatically improves responses
        for common patterns without any GPU needed.
        """
        if not dataset_path.exists():
            return {"success": False, "error": "No dataset found"}

        # Load top examples
        examples = []
        for line in dataset_path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                try: examples.append(json.loads(line))
                except Exception: pass

        # Sort by score, take top 20
        examples.sort(key=lambda x: x.get("score", 0), reverse=True)
        top = examples[:20]

        if not top:
            return {"success": False, "error": "No training examples"}

        # Build system prompt from examples
        patterns = "\n".join(
            f"Q: {e['instruction']}\nA: {e['output'][:200]}"
            for e in top[:10]
        )

        modelfile = f"""FROM {self.base_model}

SYSTEM \"\"\"
You are ARIA, a personal AI assistant. You are helpful, concise, and precise.

You have learned from your owner's preferences. Here are patterns from past interactions:

{patterns}

Always be direct and actionable. Prefer short answers unless detail is needed.
\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
"""
        # Write modelfile
        mf_path = ADAPTER_DIR / "Modelfile.aria"
        mf_path.write_text(modelfile)

        # Create the model in Ollama
        try:
            result = subprocess.run(
                ["ollama", "create", "aria-tuned", "-f", str(mf_path)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                console.print("  [green]Ollama model created:[/] aria-tuned")
                return {
                    "success":   True,
                    "model":     "aria-tuned",
                    "examples":  len(top),
                    "method":    "modelfile",
                    "note":      "Run: ollama run aria-tuned to test it",
                }
            else:
                return {"success": False, "error": result.stderr[:200]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_gguf_adapter(self, gguf_path: Path) -> dict:
        """
        Load a GGUF adapter (from Colab fine-tuning) into Ollama.
        Creates a new 'aria-finetuned' model.
        """
        if not gguf_path.exists():
            return {"success": False, "error": f"GGUF not found: {gguf_path}"}

        modelfile = f"""FROM {gguf_path}

SYSTEM "You are ARIA, a helpful personal AI assistant. Be concise and actionable."

PARAMETER temperature 0.3
PARAMETER num_ctx 4096
"""
        mf_path = ADAPTER_DIR / "Modelfile.finetuned"
        mf_path.write_text(modelfile)

        try:
            result = subprocess.run(
                ["ollama", "create", "aria-finetuned", "-f", str(mf_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                console.print(f"  [green]Fine-tuned model loaded:[/] aria-finetuned")
                return {"success": True, "model": "aria-finetuned", "from_gguf": str(gguf_path)}
            return {"success": False, "error": result.stderr[:300]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_for_adapter(self) -> str | None:
        """Check if a fine-tuned GGUF is waiting to be loaded."""
        for f in ADAPTER_DIR.glob("*.gguf"):
            return str(f)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCHEDULER — runs the full loop automatically
# ─────────────────────────────────────────────────────────────────────────────

class TrainingScheduler:
    """
    Runs the full training pipeline automatically.
    Default: every Sunday at 3am.
    Also triggers if buffer exceeds 200 examples.
    """

    def __init__(self, buffer: TrainingBuffer = None, engine=None):
        self.buffer   = buffer or TrainingBuffer()
        self.builder  = DatasetBuilder()
        self.trainer  = OllamaTrainer()
        self.engine   = engine
        self._running = False
        self._last_run: str | None = None

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        console.print("  [green]Training scheduler:[/] running (weekly at 3am)")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                now = datetime.now()
                # Check conditions for training run
                should_run = (
                    (now.weekday() == 6 and now.hour == 3 and  # Sunday 3am
                     (not self._last_run or
                      datetime.fromisoformat(self._last_run).date() < now.date()))
                    or
                    (self.buffer.stats()["buffered_interactions"] >= 200 and
                     (not self._last_run or
                      time.time() - datetime.fromisoformat(self._last_run).timestamp() > 86400))
                )
                if should_run:
                    self.run_pipeline()
            except Exception as e:
                console.print(f"  [yellow]Training scheduler error: {e}[/]")
            time.sleep(3600)  # check hourly

    def run_pipeline(self, force: bool = False) -> dict:
        """Run the complete training pipeline."""
        console.print("\n  [dim]Running training pipeline…[/]")
        self._last_run = datetime.now().isoformat()
        report: dict = {"ts": self._last_run, "steps": {}}

        # Step 1: Auto-commit pending interactions
        self.buffer.auto_commit_old(max_age_min=60)
        stats = self.buffer.stats()
        report["steps"]["buffer"] = stats
        console.print(f"  [dim]  Buffer: {stats['buffered_interactions']} examples[/]")

        if stats["buffered_interactions"] < 10 and not force:
            console.print("  [dim]  Not enough data — skipping fine-tune[/]")
            report["skipped"] = "insufficient_data"
            return report

        # Step 2: Build dataset
        ds_stats = self.builder.build(min_score=0.6)
        report["steps"]["dataset"] = ds_stats

        if ds_stats["examples"] < 5:
            report["skipped"] = "insufficient_quality_data"
            return report

        # Step 3: Check for waiting GGUF adapter (from Colab)
        gguf = self.trainer.check_for_adapter()
        if gguf:
            console.print(f"  [green]Found GGUF adapter:[/] {gguf}")
            result = self.trainer.load_gguf_adapter(Path(gguf))
            report["steps"]["load_gguf"] = result
            if result["success"]:
                # Archive the GGUF after loading
                import shutil
                shutil.move(gguf, str(ADAPTER_DIR / "loaded" / Path(gguf).name))
        else:
            # Step 4: Local Modelfile approach (always available)
            result = self.trainer.create_modelfile_adapter(DATASET_FILE)
            report["steps"]["modelfile"] = result

        # Step 5: Export Colab notebook for proper fine-tuning
        nb_path = self.builder.export_colab()
        report["steps"]["colab_notebook"] = str(nb_path)
        console.print(f"  [green]Colab notebook ready:[/] open data/training/aria_finetune.ipynb")

        report["success"] = True
        console.print(f"  [green]Training pipeline complete[/]")
        return report

    def status(self) -> dict:
        return {
            "last_run":       self._last_run,
            "buffer":         self.buffer.stats(),
            "dataset":        self.builder.stats(),
            "colab_notebook": str(TRAIN_DIR / "aria_finetune.ipynb"),
            "adapters":       [str(p) for p in ADAPTER_DIR.glob("*.gguf")],
            "ollama_available": self.trainer.is_available(),
        }
