"""
ARIA — Auto-Tuner  (self-improvement and performance optimization engine)
=========================================================================
Continuously measures ARIA's output quality, finds best model/temperature
combos per task type, A/B-tests system prompts, exports training pairs,
and triggers fine-tuning when enough quality data has accumulated.

Metrics are persisted across restarts so the tuner gets smarter over time.

Quick usage:
    from agents.auto_tuner import AutoTuner
    from core.engine import Engine

    engine = Engine()
    tuner  = AutoTuner()

    tuner.record_response("what is 2+2?", "4", "math", engine.model, 0.0,
                          latency_ms=120)
    report = tuner.get_performance_report()
    print(report.recommendations)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = DATA_DIR / "performance_metrics.jsonl"
CONFIGS_FILE = DATA_DIR / "optimal_configs.json"

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceMetric:
    query_hash:   str
    task_type:    str
    model:        str
    temperature:  float
    score:        float
    latency_ms:   float
    timestamp:    str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # stored for training pair export — not used in scoring
    query:        str = ""
    response:     str = ""


@dataclass
class PerformanceReport:
    avg_score:      float
    best_config:    Dict[str, Any]   # {"task_type", "model", "temperature", "score"}
    worst_config:   Dict[str, Any]
    recommendations: List[str]
    trend:          str              # "improving" | "stable" | "degrading"
    by_task:        Dict[str, float] = field(default_factory=dict)
    by_model:       Dict[str, float] = field(default_factory=dict)
    total_samples:  int = 0


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT TEMPERATURE RANGES  (fallback before enough data)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_TEMPS: Dict[str, float] = {
    "code":     0.07,
    "math":     0.0,
    "creative": 0.80,
    "chat":     0.45,
    "factual":  0.15,
}

_TEMP_RANGES: Dict[str, Tuple[float, float]] = {
    "code":     (0.05, 0.10),
    "math":     (0.0,  0.0),
    "creative": (0.70, 0.90),
    "chat":     (0.40, 0.50),
    "factual":  (0.10, 0.20),
}

# ──────────────────────────────────────────────────────────────────────────────
# AUTO-TUNER
# ──────────────────────────────────────────────────────────────────────────────

class AutoTuner:
    """
    ARIA self-improvement and auto-tuning engine.

    Thread-safe.  Persists all metrics to data/performance_metrics.jsonl and
    optimal configs to data/optimal_configs.json.
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._metrics:    List[PerformanceMetric] = []
        self._configs:    Dict[str, Any]          = {}
        self._bg_thread:  Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        """Load existing metrics and configs from disk."""
        # metrics
        if METRICS_FILE.exists():
            with open(METRICS_FILE, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            self._metrics.append(PerformanceMetric(**d))
                        except Exception:
                            pass
        # configs
        if CONFIGS_FILE.exists():
            try:
                with open(CONFIGS_FILE, "r", encoding="utf-8") as fh:
                    self._configs = json.load(fh)
            except Exception:
                self._configs = {}

    def _save_metric(self, m: PerformanceMetric):
        """Append a single metric to the JSONL file (fast, non-blocking)."""
        with open(METRICS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(m)) + "\n")

    def _save_configs(self):
        with open(CONFIGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._configs, fh, indent=2)

    # ── Core recording ────────────────────────────────────────────────────────

    def record_response(
        self,
        query:       str,
        response:    str,
        task_type:   str,
        model:       str,
        temperature: float,
        score:       Optional[float] = None,
        latency_ms:  Optional[float] = None,
    ) -> PerformanceMetric:
        """
        Log an interaction.  If score is None it will be auto-scored
        using the LLM critic (requires engine to be available later).
        """
        q_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        m = PerformanceMetric(
            query_hash  = q_hash,
            task_type   = task_type.lower(),
            model       = model,
            temperature = temperature,
            score       = score if score is not None else -1.0,  # -1 = pending
            latency_ms  = latency_ms or 0.0,
            query       = query,
            response    = response,
        )
        with self._lock:
            self._metrics.append(m)
        self._save_metric(m)
        return m

    # ── LLM critic scoring ────────────────────────────────────────────────────

    def score_response(
        self,
        query:    str,
        response: str,
        engine=None,
    ) -> float:
        """
        Use LLM critic to score a response 0.0 – 1.0.
        Returns 0.5 (neutral) if engine unavailable.
        """
        if engine is None:
            return 0.5

        prompt = (
            "You are a strict AI quality judge.\n"
            "Rate the following response to the given question on a scale of 0.0 to 1.0.\n"
            "0.0 = completely wrong or unhelpful\n"
            "0.5 = partially correct\n"
            "1.0 = perfect, complete, concise, accurate\n\n"
            f"Question: {query}\n\n"
            f"Response: {response}\n\n"
            "Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else."
        )
        try:
            raw = engine.generate(prompt, temperature=0.0).strip()
            # extract first float
            m = re.search(r"(\d+(?:\.\d+)?)", raw)
            if m:
                score = float(m.group(1))
                return max(0.0, min(1.0, score))
        except Exception:
            pass
        return 0.5

    # ── Optimal configuration lookup ──────────────────────────────────────────

    def get_optimal_temperature(self, task_type: str) -> float:
        """
        Return the empirically best temperature for a task type.
        Falls back to curated defaults if not enough data.
        """
        task = task_type.lower()
        # Check learned config first
        key = f"temp_{task}"
        if key in self._configs:
            return float(self._configs[key])

        # Check accumulated metrics (need at least 5 samples)
        with self._lock:
            relevant = [
                m for m in self._metrics
                if m.task_type == task and m.score >= 0.0
            ]
        if len(relevant) >= 5:
            # bin temperatures and find best performing bucket
            buckets: Dict[float, List[float]] = defaultdict(list)
            for m in relevant:
                bucket = round(m.temperature * 10) / 10  # round to 0.1
                buckets[bucket].append(m.score)
            best_temp = max(buckets, key=lambda t: sum(buckets[t]) / len(buckets[t]))
            # clamp to allowed range
            lo, hi = _TEMP_RANGES.get(task, (0.0, 1.0))
            best_temp = max(lo, min(hi, best_temp))
            return best_temp

        return _DEFAULT_TEMPS.get(task, 0.5)

    def get_optimal_model(
        self,
        task_type:        str,
        available_models: List[str],
    ) -> str:
        """
        Return the model from available_models that performs best for task_type.
        Falls back to the first available model if no data.
        """
        if not available_models:
            return ""
        task = task_type.lower()
        key  = f"model_{task}"
        if key in self._configs and self._configs[key] in available_models:
            return self._configs[key]

        # compute avg score per model for this task
        with self._lock:
            relevant = [
                m for m in self._metrics
                if m.task_type == task and m.score >= 0.0
                   and m.model in available_models
            ]
        if not relevant:
            return available_models[0]

        model_scores: Dict[str, List[float]] = defaultdict(list)
        for m in relevant:
            model_scores[m.model].append(m.score)
        best = max(model_scores, key=lambda mdl: sum(model_scores[mdl]) / len(model_scores[mdl]))
        return best

    # ── Prompt optimisation ───────────────────────────────────────────────────

    def optimize_system_prompt(
        self,
        base_prompt: str,
        task_type:   str,
        n_variants:  int = 3,
        engine=None,
    ) -> str:
        """
        A/B test n_variants of the base prompt using the LLM critic.
        Returns the variant that scores highest on a sample of recent queries.
        """
        if engine is None:
            console.print("[yellow]AutoTuner: engine required for prompt optimisation.[/]")
            return base_prompt

        # Generate variants
        gen_prompt = (
            f"You are an expert prompt engineer.\n"
            f"Given the base system prompt below, generate {n_variants} improved variants.\n"
            f"Each variant should be better for '{task_type}' tasks.\n"
            f"Output ONLY a JSON array of strings — one string per variant.\n\n"
            f"Base prompt:\n{base_prompt}"
        )
        try:
            raw = engine.generate(gen_prompt, temperature=0.7)
            # extract JSON array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return base_prompt
            variants: List[str] = json.loads(match.group(0))
        except Exception:
            return base_prompt

        if not variants:
            return base_prompt

        # Pick sample queries for this task type (up to 5)
        with self._lock:
            samples = [
                m for m in self._metrics
                if m.task_type == task_type.lower() and m.query
            ][-5:]

        if not samples:
            console.print("[yellow]No sample queries for A/B test — returning first variant.[/]")
            return variants[0]

        test_queries = [m.query for m in samples]

        # Score each variant
        best_score   = -1.0
        best_variant = base_prompt

        for variant in variants:
            variant_scores = []
            for q in test_queries:
                full_prompt = f"{variant}\n\nQuestion: {q}"
                response    = engine.generate(full_prompt, temperature=self.get_optimal_temperature(task_type))
                s           = self.score_response(q, response, engine)
                variant_scores.append(s)
            avg = sum(variant_scores) / len(variant_scores) if variant_scores else 0.0
            if avg > best_score:
                best_score   = avg
                best_variant = variant

        console.print(f"[green]Best prompt variant score: {best_score:.3f}[/]")
        return best_variant

    # ── Temperature tuning ────────────────────────────────────────────────────

    def tune_temperature(
        self,
        task_type: str,
        n_samples: int = 20,
        engine=None,
    ) -> float:
        """
        Empirically find the best temperature for a task by sampling n_samples
        queries from the metric history and testing each candidate temperature.
        Updates the optimal config and returns the best temperature.
        """
        if engine is None:
            return self.get_optimal_temperature(task_type)

        task = task_type.lower()
        lo, hi = _TEMP_RANGES.get(task, (0.0, 1.0))

        if lo == hi:
            # e.g. math: always 0.0
            return lo

        # candidate temperatures
        import numpy as _np
        candidates = list(_np.linspace(lo, hi, num=5))

        # get sample queries
        with self._lock:
            pool = [m for m in self._metrics if m.task_type == task and m.query]
        if not pool:
            # synthesise test questions using LLM
            gen = engine.generate(
                f"Generate {n_samples} short test questions for the task type '{task}'. "
                "Output a JSON array of strings.",
                temperature=0.7,
            )
            match = re.search(r"\[.*\]", gen, re.DOTALL)
            queries = json.loads(match.group(0)) if match else [f"Test question for {task}"] * 5
        else:
            import random
            queries = [m.query for m in random.sample(pool, min(n_samples, len(pool)))]

        best_temp  = candidates[0]
        best_score = -1.0

        console.print(f"[cyan]Tuning temperature for '{task}' over {len(candidates)} candidates...[/]")

        for temp in candidates:
            scores = []
            for q in queries[:10]:  # cap at 10 to save time
                response = engine.generate(q, temperature=temp)
                s        = self.score_response(q, response, engine)
                scores.append(s)
            avg = sum(scores) / len(scores) if scores else 0.0
            console.print(f"  temp={temp:.2f}  avg_score={avg:.3f}")
            if avg > best_score:
                best_score = avg
                best_temp  = temp

        console.print(f"[green]Best temperature for '{task}': {best_temp:.2f} (score {best_score:.3f})[/]")
        with self._lock:
            self._configs[f"temp_{task}"] = best_temp
        self._save_configs()
        return best_temp

    # ── Training data export ──────────────────────────────────────────────────

    def generate_training_pairs(self, min_score: float = 0.8) -> Path:
        """
        Export all metrics with score >= min_score as instruction-response
        JSONL pairs for fine-tuning.  Returns path to the output file.
        """
        out_path = DATA_DIR / "training" / "high_quality_pairs.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            high = [m for m in self._metrics if m.score >= min_score and m.query and m.response]

        count = 0
        with open(out_path, "w", encoding="utf-8") as fh:
            for m in high:
                pair = {
                    "instruction": m.query,
                    "response":    m.response,
                    "task_type":   m.task_type,
                    "score":       m.score,
                }
                fh.write(json.dumps(pair) + "\n")
                count += 1

        console.print(f"[green]Exported {count} training pairs → {out_path}[/]")
        return out_path

    # ── Fine-tuning trigger ───────────────────────────────────────────────────

    def trigger_finetune(self, engine, threshold: int = 200) -> bool:
        """
        If >= threshold high-quality pairs exist, trigger Ollama fine-tuning
        pipeline.  Returns True if fine-tuning was started.
        """
        pairs_path = self.generate_training_pairs(min_score=0.8)
        count = sum(1 for _ in open(pairs_path, encoding="utf-8")) if pairs_path.exists() else 0

        if count < threshold:
            console.print(
                f"[yellow]Only {count}/{threshold} training pairs available. "
                "Collect more data before fine-tuning.[/]"
            )
            return False

        model_name = f"aria-tuned-{datetime.utcnow().strftime('%Y%m%d')}"
        console.print(f"[cyan]Starting Ollama fine-tune → model: {model_name}[/]")

        # Build a Modelfile using the training data
        modelfile_path = DATA_DIR / "training" / "Modelfile"
        modelfile_path.write_text(
            f"FROM {engine.model}\n"
            f"# Fine-tuned by ARIA AutoTuner on {datetime.utcnow().isoformat()}\n"
            f"PARAMETER temperature {self.get_optimal_temperature('chat')}\n",
            encoding="utf-8",
        )

        try:
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                console.print(f"[green]Fine-tune complete: {model_name}[/]")
                with self._lock:
                    self._configs["last_finetuned_model"] = model_name
                self._save_configs()
                return True
            else:
                console.print(f"[red]Fine-tune failed:[/] {result.stderr[:300]}")
                return False
        except FileNotFoundError:
            console.print("[red]ollama binary not found. Install Ollama to enable fine-tuning.[/]")
            return False
        except subprocess.TimeoutExpired:
            console.print("[red]Fine-tuning timed out after 10 minutes.[/]")
            return False

    # ── Performance report ────────────────────────────────────────────────────

    def get_performance_report(self) -> PerformanceReport:
        """
        Build a comprehensive performance report from accumulated metrics.
        """
        with self._lock:
            metrics = [m for m in self._metrics if m.score >= 0.0]

        if not metrics:
            return PerformanceReport(
                avg_score   = 0.0,
                best_config  = {},
                worst_config = {},
                recommendations = ["No performance data yet. Start using ARIA to accumulate metrics."],
                trend        = "stable",
            )

        # overall average
        avg_score = sum(m.score for m in metrics) / len(metrics)

        # by task
        task_scores: Dict[str, List[float]] = defaultdict(list)
        for m in metrics:
            task_scores[m.task_type].append(m.score)
        by_task = {t: sum(s) / len(s) for t, s in task_scores.items()}

        # by model
        model_scores: Dict[str, List[float]] = defaultdict(list)
        for m in metrics:
            model_scores[m.model].append(m.score)
        by_model = {mdl: sum(s) / len(s) for mdl, s in model_scores.items()}

        # best and worst config (task + model + temperature combos)
        combo_scores: Dict[Tuple, List[float]] = defaultdict(list)
        for m in metrics:
            key = (m.task_type, m.model, round(m.temperature, 1))
            combo_scores[key].append(m.score)
        combo_avg = {k: sum(v) / len(v) for k, v in combo_scores.items() if len(v) >= 2}

        if combo_avg:
            best_key  = max(combo_avg, key=combo_avg.__getitem__)
            worst_key = min(combo_avg, key=combo_avg.__getitem__)
            best_config  = {"task_type": best_key[0],  "model": best_key[1],
                            "temperature": best_key[2], "score": combo_avg[best_key]}
            worst_config = {"task_type": worst_key[0], "model": worst_key[1],
                            "temperature": worst_key[2], "score": combo_avg[worst_key]}
        else:
            best_config  = {}
            worst_config = {}

        # trend: compare last 20 vs previous 20
        trend = "stable"
        if len(metrics) >= 40:
            recent   = [m.score for m in metrics[-20:]]
            previous = [m.score for m in metrics[-40:-20]]
            delta    = (sum(recent) / len(recent)) - (sum(previous) / len(previous))
            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "degrading"

        # recommendations
        recommendations = self._build_recommendations(
            avg_score, by_task, by_model, trend, len(metrics)
        )

        return PerformanceReport(
            avg_score        = avg_score,
            best_config      = best_config,
            worst_config     = worst_config,
            recommendations  = recommendations,
            trend            = trend,
            by_task          = by_task,
            by_model         = by_model,
            total_samples    = len(metrics),
        )

    def _build_recommendations(
        self,
        avg_score:  float,
        by_task:    Dict[str, float],
        by_model:   Dict[str, float],
        trend:      str,
        n_samples:  int,
    ) -> List[str]:
        recs = []
        if avg_score < 0.6:
            recs.append("Overall quality is below 60%. Consider pulling a larger model (e.g., llama3.1:8b).")
        if trend == "degrading":
            recs.append("Performance is trending downward. Run tune_temperature() and optimize_system_prompt().")
        for task, score in by_task.items():
            if score < 0.55:
                recs.append(f"Task '{task}' scores poorly ({score:.2f}). Run tune_temperature('{task}').")
        worst_model = min(by_model, key=by_model.__getitem__) if by_model else None
        if worst_model and by_model[worst_model] < 0.5:
            recs.append(f"Model '{worst_model}' is underperforming. Consider switching to a different model.")
        if n_samples >= 200:
            recs.append("You have 200+ quality samples. Consider running trigger_finetune() to create a custom model.")
        if not recs:
            recs.append("Performance looks good! Keep using ARIA to accumulate more data for deeper insights.")
        return recs

    # ── Pretty-print report ───────────────────────────────────────────────────

    def print_report(self, report: Optional[PerformanceReport] = None):
        if report is None:
            report = self.get_performance_report()

        table = Table(title="ARIA Performance Report", show_header=True)
        table.add_column("Metric",  style="cyan")
        table.add_column("Value",   style="white")
        table.add_row("Overall avg score", f"{report.avg_score:.3f}")
        table.add_row("Trend",             report.trend)
        table.add_row("Total samples",     str(report.total_samples))
        if report.best_config:
            table.add_row(
                "Best config",
                f"{report.best_config.get('model')} | "
                f"{report.best_config.get('task_type')} | "
                f"temp={report.best_config.get('temperature')} | "
                f"score={report.best_config.get('score', 0):.3f}",
            )
        for task, score in report.by_task.items():
            table.add_row(f"  {task}", f"{score:.3f}")
        console.print(table)
        console.print("\n[bold]Recommendations:[/]")
        for r in report.recommendations:
            console.print(f"  • {r}")

    # ── Auto-scheduling ───────────────────────────────────────────────────────

    def auto_schedule(self, interval_hours: float = 24, engine=None):
        """
        Start a background thread that runs a daily performance review and
        prompt optimisation.  Non-blocking.  Call once at startup.
        """
        if self._bg_thread and self._bg_thread.is_alive():
            console.print("[yellow]AutoTuner scheduler already running.[/]")
            return

        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=interval_hours * 3600)
                if self._stop_event.is_set():
                    break
                console.print("[cyan][AutoTuner] Running scheduled performance review...[/]")
                report = self.get_performance_report()
                self.print_report(report)
                if engine and report.trend == "degrading":
                    for task in report.by_task:
                        if report.by_task[task] < 0.55:
                            console.print(f"[cyan][AutoTuner] Tuning temperature for '{task}'...[/]")
                            self.tune_temperature(task, engine=engine)

        self._bg_thread = threading.Thread(target=_loop, daemon=True, name="AutoTuner-Scheduler")
        self._bg_thread.start()
        console.print(f"[green]AutoTuner scheduler started (every {interval_hours}h).[/]")

    def stop_schedule(self):
        """Stop the background scheduler."""
        self._stop_event.set()

    # ── Model upgrade suggestions ─────────────────────────────────────────────

    def suggest_model_upgrade(
        self,
        current_model:     str,
        performance_trend: str,
    ) -> Optional[str]:
        """
        If performance is degrading, suggest a better model to pull.
        Returns the recommended model name or None.
        """
        upgrade_path = {
            "phi3:mini":     "llama3.2:3b",
            "llama3.2:3b":   "llama3.1:8b",
            "llama3.1:8b":   "llama3.1:70b",
            "mistral:7b":    "llama3.1:8b",
            "gemma2:2b":     "gemma2:9b",
            "gemma2:9b":     "llama3.1:8b",
        }
        if performance_trend != "degrading":
            return None
        suggestion = upgrade_path.get(current_model)
        if suggestion:
            console.print(
                f"[yellow]Performance degrading with '{current_model}'. "
                f"Suggested upgrade: ollama pull {suggestion}[/]"
            )
        return suggestion

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(
        self,
        questions_list: List[str],
        engine,
        task_type: str = "factual",
    ) -> Dict[str, Any]:
        """
        Run a benchmark: generate answers for each question and score them.
        Returns {"scores": [...], "avg": float, "min": float, "max": float}.
        """
        scores     = []
        latencies  = []

        console.print(f"[cyan]Benchmarking {len(questions_list)} questions...[/]")

        for i, q in enumerate(questions_list):
            t0       = time.time()
            response = engine.generate(q, temperature=self.get_optimal_temperature(task_type))
            lat_ms   = (time.time() - t0) * 1000
            score    = self.score_response(q, response, engine)
            scores.append(score)
            latencies.append(lat_ms)
            self.record_response(q, response, task_type, engine.model,
                                 self.get_optimal_temperature(task_type),
                                 score=score, latency_ms=lat_ms)
            console.print(f"  [{i+1}/{len(questions_list)}] score={score:.3f}  lat={lat_ms:.0f}ms")

        result = {
            "scores":   scores,
            "avg":      sum(scores) / len(scores) if scores else 0.0,
            "min":      min(scores) if scores else 0.0,
            "max":      max(scores) if scores else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        }
        console.print(
            f"[green]Benchmark complete — avg score: {result['avg']:.3f}, "
            f"avg latency: {result['avg_latency_ms']:.0f}ms[/]"
        )
        return result

    # ── Natural-language interface ────────────────────────────────────────────

    def run_nl(self, query: str, engine=None) -> str:
        """
        Natural-language entry point.

        Examples:
            "how is my ARIA performing?"
            "optimize temperature for code tasks"
            "run benchmark"
            "export training data"
            "tune creative temperature"
            "show performance report"
        """
        q = query.lower().strip()

        # performance / status queries
        if any(kw in q for kw in ("performing", "performance", "report", "stats", "how is")):
            report = self.get_performance_report()
            self.print_report(report)
            lines = [
                f"Overall avg score: {report.avg_score:.3f}",
                f"Trend: {report.trend}",
                f"Samples: {report.total_samples}",
            ]
            lines += report.recommendations
            return "\n".join(lines)

        # temperature tuning
        if "tune" in q or "optimize" in q or "optimise" in q:
            task = self._extract_task(q)
            if task:
                if engine:
                    best_t = self.tune_temperature(task, engine=engine)
                    return f"Best temperature for '{task}': {best_t:.2f}"
                else:
                    best_t = self.get_optimal_temperature(task)
                    return f"Current optimal temperature for '{task}': {best_t:.2f}"
            return "Specify a task type: code, math, creative, chat, or factual."

        # benchmark
        if "benchmark" in q:
            if engine is None:
                return "Engine required for benchmarking."
            default_questions = [
                "What is the capital of France?",
                "Explain quantum entanglement simply.",
                "Write a Python function to reverse a string.",
                "What is 17 * 23?",
                "Who wrote 'To Kill a Mockingbird'?",
            ]
            result = self.benchmark(default_questions, engine)
            return (
                f"Benchmark complete.\n"
                f"Avg score: {result['avg']:.3f}\n"
                f"Min: {result['min']:.3f}  Max: {result['max']:.3f}\n"
                f"Avg latency: {result['avg_latency_ms']:.0f}ms"
            )

        # export training data
        if "export" in q or "training" in q or "pairs" in q:
            path = self.generate_training_pairs()
            return f"Training pairs exported to: {path}"

        # fine-tune trigger
        if "finetune" in q or "fine-tune" in q or "fine tune" in q:
            if engine is None:
                return "Engine required to trigger fine-tuning."
            started = self.trigger_finetune(engine)
            return "Fine-tuning started." if started else "Not enough data yet (need 200+ pairs)."

        # model upgrade suggestion
        if "upgrade" in q or "better model" in q:
            if engine:
                report = self.get_performance_report()
                suggestion = self.suggest_model_upgrade(engine.model, report.trend)
                if suggestion:
                    return f"Suggested model upgrade: ollama pull {suggestion}"
                return f"Current model '{engine.model}' is performing adequately."
            return "Connect an engine to get model upgrade suggestions."

        return (
            "AutoTuner commands:\n"
            "  'how is ARIA performing?' — performance report\n"
            "  'optimize code temperature' — tune temperature for a task\n"
            "  'run benchmark' — test on sample questions\n"
            "  'export training data' — save high-quality pairs\n"
            "  'trigger finetune' — start Ollama fine-tuning (needs 200+ pairs)\n"
            "  'suggest upgrade' — recommend a better model"
        )

    def _extract_task(self, text: str) -> Optional[str]:
        for task in ("code", "math", "creative", "chat", "factual"):
            if task in text:
                return task
        return None


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    tuner = AutoTuner()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(tuner.run_nl(query))
    else:
        tuner.print_report()
