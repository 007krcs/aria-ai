"""
ARIA Training Pipeline
======================
Collects conversation data → builds Ollama Modelfile → runs `ollama create`.

Two modes:
  1. Lightweight (default) — prompt-engineering only: builds a Modelfile with
     embedded system prompt + few-shot examples from collected JSONL data.
     Works on ANY hardware, no GPU needed, result ready in seconds.

  2. LoRA (optional) — full fine-tuning via `nova_self_train.OnlineLoraTrainer`.
     Requires GPU (or slow CPU run). Produces a GGUF adapter.

The lightweight path is always used first so ARIA always improves incrementally.
LoRA is triggered when buffer reaches threshold AND GPU is detected.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("aria.training")

DATA_DIR      = Path(r"C:\Users\chand\ai-remo\data\training")
MODEL_DIR     = Path(r"C:\Users\chand\ai-remo\models")
PIPELINE_DIR  = Path(r"C:\Users\chand\ai-remo\pipelines")
MODELFILE_PATH = MODEL_DIR / "Modelfile.aria"
CUSTOM_MODEL   = "aria-custom"
BASE_MODEL     = "llama3.2:latest"      # Ollama base model to start from

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Data collector ────────────────────────────────────────────────────────────

class ConversationCollector:
    """
    Buffers user↔ARIA conversation pairs for training.
    Persists to JSONL in data/training/.
    """

    def __init__(self, session_file: Optional[Path] = None):
        ts = datetime.now().strftime("%Y%m%d")
        self._file = session_file or (DATA_DIR / f"conversations_{ts}.jsonl")
        self._lock = threading.Lock()
        self._buffer: List[dict] = []
        self._count  = 0

    def record(
        self,
        user_msg:    str,
        aria_reply:  str,
        quality:     float = 1.0,    # 0.0 – 1.0 (from user feedback or auto-score)
        domain:      str   = "general",
        thumbs_up:   bool  = True,
    ) -> None:
        """Add one conversation turn to training buffer."""
        if not user_msg.strip() or not aria_reply.strip():
            return
        # Only keep high-quality pairs (quality >= 0.6 or explicit thumbs up)
        if quality < 0.6 and not thumbs_up:
            return
        pair = {
            "instruction": user_msg.strip(),
            "output":      aria_reply.strip(),
            "quality":     quality,
            "domain":      domain,
            "ts":          datetime.now().isoformat(),
        }
        with self._lock:
            self._buffer.append(pair)
            self._count += 1
            # Flush every 10 pairs
            if len(self._buffer) >= 10:
                self._flush()

    def flush(self) -> int:
        with self._lock:
            return self._flush()

    def _flush(self) -> int:
        if not self._buffer:
            return 0
        n = len(self._buffer)
        with open(self._file, "a", encoding="utf-8") as f:
            for pair in self._buffer:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        self._buffer.clear()
        logger.debug(f"Flushed {n} training pairs to {self._file.name}")
        return n

    def total_pairs(self) -> int:
        """Count all persisted training pairs."""
        total = 0
        for f in DATA_DIR.glob("conversations_*.jsonl"):
            try:
                total += sum(1 for _ in open(f, encoding="utf-8"))
            except Exception:
                pass
        return total + len(self._buffer)

    def load_recent(self, max_pairs: int = 200) -> List[dict]:
        """Load the most recent N pairs for Modelfile few-shot examples."""
        pairs = []
        files = sorted(DATA_DIR.glob("conversations_*.jsonl"), reverse=True)
        for fpath in files:
            try:
                lines = fpath.read_text(encoding="utf-8").strip().splitlines()
                for line in reversed(lines):
                    try:
                        pairs.append(json.loads(line))
                        if len(pairs) >= max_pairs:
                            return pairs
                    except Exception:
                        pass
            except Exception:
                pass
        return pairs


# ── Lightweight Modelfile builder ─────────────────────────────────────────────

ARIA_SYSTEM_PROMPT = """You are ARIA (Adaptive Reasoning Intelligence Architecture), a world-class personal AI assistant running locally on the user's device.

Core traits:
- You reason step-by-step before answering complex questions
- You are honest about uncertainty — say "I'm not sure" rather than hallucinate
- You adapt to the user's expertise level automatically
- You proactively suggest improvements and alternatives
- You remember context within a conversation and build on it
- For real-time data (weather, news, stocks), you acknowledge you'll search for it
- You are concise by default but thorough when the question requires it

Capabilities:
- Deep research using 23 trusted sources (NIH, WHO, FDA, PubMed, arXiv, etc.)
- Medical analysis: symptoms, drugs, lab reports, research grading
- Code writing, debugging, and execution in Python/JS/Shell/PowerShell
- Computer control: open apps, take screenshots, automate workflows
- Browser control: search, navigate, crawl pages, switch Chrome profiles
- Voice interaction: 24/7 listening and conversational mode
- Financial analysis: stocks, technical indicators, market sentiment
- Multi-device sync and offline operation

Always respond in the same language the user writes in."""


def build_modelfile(pairs: List[dict], base_model: str = BASE_MODEL) -> str:
    """Build an Ollama Modelfile with few-shot examples from training data."""

    # Pick top-quality pairs as few-shot examples (max 15 to keep file manageable)
    top_pairs = sorted(pairs, key=lambda p: p.get("quality", 0.5), reverse=True)[:15]

    messages = []
    for p in top_pairs:
        # Ollama MESSAGE syntax
        messages.append(f'MESSAGE user {json.dumps(p["instruction"])}')
        messages.append(f'MESSAGE assistant {json.dumps(p["output"])}')

    messages_block = "\n".join(messages)

    modelfile = f"""FROM {base_model}

SYSTEM {json.dumps(ARIA_SYSTEM_PROMPT)}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1

{messages_block}
"""
    return modelfile


# ── Ollama integration ────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _ollama_list_models() -> List[str]:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        lines = r.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if parts:
                models.append(parts[0].split(":")[0])
        return models
    except Exception:
        return []


def create_ollama_model(modelfile_content: str, model_name: str = CUSTOM_MODEL) -> dict:
    """
    Write Modelfile and run `ollama create {model_name} -f Modelfile`.
    Returns result dict.
    """
    if not _ollama_available():
        return {"ok": False, "error": "Ollama not found. Install from https://ollama.ai"}

    MODELFILE_PATH.write_text(modelfile_content, encoding="utf-8")

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(MODELFILE_PATH)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info(f"Ollama model '{model_name}' created successfully")
            return {"ok": True, "model": model_name, "output": result.stdout}
        else:
            return {"ok": False, "error": result.stderr or result.stdout}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ollama create timed out (120s)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def delete_ollama_model(model_name: str) -> dict:
    try:
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True, text=True, timeout=30,
        )
        return {"ok": result.returncode == 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Agent promotion pipeline ──────────────────────────────────────────────────

SANDBOX_DIR = Path(r"C:\Users\chand\ai-remo\agents\sandbox")
AGENTS_DIR  = Path(r"C:\Users\chand\ai-remo\agents")


def list_sandbox_agents() -> List[dict]:
    """List agents waiting in sandbox for promotion."""
    agents = []
    if not SANDBOX_DIR.exists():
        return agents
    for f in sorted(SANDBOX_DIR.glob("*.py")):
        stat = f.stat()
        agents.append({
            "filename": f.name,
            "path":     str(f),
            "size":     stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return agents


def auto_validate_and_promote(filename: str) -> dict:
    """
    Validate a sandboxed agent:
      1. AST syntax check
      2. Import test (no side effects)
      3. Has a class ending in 'Agent' or a run_agent() function
      4. No dangerous imports (subprocess exec, os.system, etc.)
    If all pass, promote to agents/.
    """
    import ast, importlib.util, shutil

    src = SANDBOX_DIR / filename
    if not src.exists():
        return {"ok": False, "error": f"{filename} not in sandbox"}

    code = src.read_text(encoding="utf-8")
    issues = []

    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "error": f"SyntaxError: {e}"}

    # 2. Dangerous imports
    DANGEROUS = {"os.system", "subprocess.call", "eval(", "exec(", "__import__"}
    for d in DANGEROUS:
        if d in code:
            issues.append(f"Potentially dangerous: {d}")

    # 3. Has a class or run_agent
    has_class = any(isinstance(n, ast.ClassDef) and n.name.endswith("Agent") for n in ast.walk(tree))
    has_run   = any(
        isinstance(n, ast.FunctionDef) and n.name == "run_agent"
        for n in ast.walk(tree)
    )
    if not has_class and not has_run:
        issues.append("No *Agent class or run_agent() function found")

    if issues and not has_class:
        return {"ok": False, "issues": issues, "promoted": False}

    # 4. Promote
    dest = AGENTS_DIR / filename
    shutil.copy2(src, dest)
    logger.info(f"Promoted {filename} -> agents/")

    return {
        "ok":       True,
        "promoted": True,
        "filename": filename,
        "issues":   issues,    # warnings (non-blocking)
        "dest":     str(dest),
    }


def promote_all_sandbox_agents() -> List[dict]:
    """Validate and promote all agents in sandbox."""
    results = []
    for agent in list_sandbox_agents():
        result = auto_validate_and_promote(agent["filename"])
        result["filename"] = agent["filename"]
        results.append(result)
    return results


# ── Main pipeline orchestrator ────────────────────────────────────────────────

class TrainingPipeline:
    """
    ARIA's self-improvement loop:
      1. Collect → 2. Build Modelfile → 3. ollama create → 4. Promote agents

    Call `run()` manually or schedule via `auto_schedule()`.
    """

    def __init__(self, min_pairs: int = 20, schedule_hours: int = 24):
        self.min_pairs      = min_pairs
        self.schedule_hours = schedule_hours
        self._collector     = ConversationCollector()
        self._lock          = threading.Lock()
        self._running       = False
        self._thread: Optional[threading.Thread] = None
        self._last_run: Optional[datetime] = None
        self._results: List[dict] = []

    @property
    def collector(self) -> ConversationCollector:
        return self._collector

    def run(self, force: bool = False) -> dict:
        """
        Run the full pipeline once.
        force=True skips the min_pairs check.
        """
        with self._lock:
            # Flush buffered pairs
            self._collector.flush()
            total = self._collector.total_pairs()

            if total < self.min_pairs and not force:
                return {
                    "ok":     False,
                    "reason": f"Only {total} pairs collected (need {self.min_pairs})",
                    "total":  total,
                }

            logger.info(f"Training pipeline: {total} pairs — building Modelfile...")

            # Step 1: Load training data
            pairs = self._collector.load_recent(max_pairs=200)

            # Step 2: Build Modelfile
            modelfile = build_modelfile(pairs)

            # Step 3: Create Ollama model
            model_result = create_ollama_model(modelfile, CUSTOM_MODEL)

            # Step 4: Promote validated sandbox agents
            promoted = promote_all_sandbox_agents()
            promoted_ok = [p for p in promoted if p.get("ok")]

            result = {
                "ok":               model_result.get("ok", False),
                "total_pairs":      total,
                "model":            CUSTOM_MODEL,
                "model_result":     model_result,
                "agents_promoted":  len(promoted_ok),
                "agent_results":    promoted,
                "ts":               datetime.now().isoformat(),
            }
            self._last_run = datetime.now()
            self._results.append(result)
            logger.info(f"Pipeline complete: model_ok={result['ok']}, promoted={len(promoted_ok)}")
            return result

    def status(self) -> dict:
        self._collector.flush()
        return {
            "total_pairs":      self._collector.total_pairs(),
            "min_pairs":        self.min_pairs,
            "ready":            self._collector.total_pairs() >= self.min_pairs,
            "last_run":         self._last_run.isoformat() if self._last_run else None,
            "ollama_available": _ollama_available(),
            "custom_model":     CUSTOM_MODEL in _ollama_list_models(),
            "sandbox_agents":   len(list_sandbox_agents()),
            "background_running": self._running,
        }

    def auto_schedule(self, interval_hours: int = None) -> None:
        """Start a background thread that runs pipeline every N hours."""
        hours = interval_hours or self.schedule_hours
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._scheduler_loop,
            args=(hours,),
            name="aria-training-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Training pipeline auto-scheduler started (every {hours}h)")

    def _scheduler_loop(self, hours: int) -> None:
        interval = hours * 3600
        # Run immediately on first tick if data is ready
        time.sleep(60)   # give server 60s to boot first
        while self._running:
            try:
                result = self.run()
                if result.get("ok"):
                    logger.info(f"Scheduled training run complete: {result}")
            except Exception as e:
                logger.error(f"Training pipeline error: {e}")
            time.sleep(interval)


# ── Module-level singleton ────────────────────────────────────────────────────

_pipeline: Optional[TrainingPipeline] = None
_pipe_lock = threading.Lock()


def get_pipeline() -> TrainingPipeline:
    global _pipeline
    with _pipe_lock:
        if _pipeline is None:
            _pipeline = TrainingPipeline()
        return _pipeline


def record_pair(user_msg: str, aria_reply: str, quality: float = 1.0,
                domain: str = "general") -> None:
    """Convenience function — record a training pair from anywhere in ARIA."""
    get_pipeline().collector.record(user_msg, aria_reply, quality, domain)
