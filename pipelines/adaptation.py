"""
ARIA — Self-adaptation engine (The Transformer Moment)
This is what makes the car build other cars.

Runs on a schedule (every 6 hours by default).
Steps:
  1. Read failure patterns from the logger
  2. Try DSPy prompt auto-tuning first (cheap)
  3. If not enough → collect failures → trigger LoRA fine-tune → register new agent

Everything here is fully automated. No human needed after initial setup.
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table

from core.config import (
    CONFIDENCE_THRESHOLD, FAILURE_COUNT_THRESHOLD,
    ADAPTATION_CHECK_HOURS, MODELS_DIR, DEFAULT_MODEL
)
from tools.logger import Logger
from core.engine import Engine
from core.memory import Memory

console = Console()


class AdaptationEngine:
    """
    Monitors agent performance and adapts automatically.
    
    The loop:
        monitor() → detect failure patterns
        → try prompt tuning first (DSPy — fast, free, no GPU)
        → if still failing → spawn_specialist() (LoRA fine-tune)
        → register new agent → update routing
    """

    def __init__(self, logger: Logger, engine: Engine, memory: Memory):
        self.logger  = logger
        self.engine  = engine
        self.memory  = memory
        self.tuned_prompts: dict[str, str] = {}  # domain → best system prompt found

    # ── Main adaptation cycle ──────────────────────────────────────────────────

    def run_cycle(self) -> dict:
        """
        Run one full adaptation cycle. Call this on a schedule.
        Returns summary of what was done.
        """
        console.rule("[bold]Adaptation cycle starting[/]")
        summary = {"checked": 0, "tuned": 0, "spawned": 0, "timestamp": datetime.now().isoformat()}

        patterns = self.logger.get_failure_patterns(
            hours=ADAPTATION_CHECK_HOURS * 4,
            min_count=FAILURE_COUNT_THRESHOLD // 2,
        )

        if not patterns:
            console.print("[green]All agents performing well. No adaptation needed.[/]")
            return summary

        console.print(f"[yellow]{len(patterns)} domains with performance issues detected[/]")
        summary["checked"] = len(patterns)

        for pattern in patterns:
            intent    = pattern["intent"]
            avg_conf  = pattern["avg_conf"]
            count     = pattern["total"]

            console.print(f"\n  Domain: [bold]{intent}[/] | avg confidence: {avg_conf:.2f} | samples: {count}")

            # ── Step 1: Try prompt tuning first ──────────────────────────────
            improved = self._tune_prompt(intent)
            if improved:
                summary["tuned"] += 1
                console.print(f"  [green]Prompt tuning improved {intent}[/]")
                continue

            # ── Step 2: If we have enough failures, spawn a specialist ────────
            failures = self.logger.get_failures_for_domain(intent, hours=ADAPTATION_CHECK_HOURS * 4)
            if len(failures) >= FAILURE_COUNT_THRESHOLD:
                spawned = self._spawn_specialist(intent, failures)
                if spawned:
                    summary["spawned"] += 1
            else:
                console.print(
                    f"  [dim]Not enough failures yet ({len(failures)}/{FAILURE_COUNT_THRESHOLD}) "
                    f"to justify spawning. Collecting more data...[/]"
                )

        self._print_summary(summary)
        return summary

    # ── Prompt auto-tuning (DSPy) ─────────────────────────────────────────────

    def _tune_prompt(self, domain: str) -> bool:
        """
        Use DSPy to automatically rewrite the system prompt for a domain.
        Tries different prompt variants, scores them, keeps the best.
        
        This is the lightweight fix — no GPU, no training, just better instructions.
        Returns True if an improvement was found.
        """
        console.print(f"  [dim]Attempting prompt tuning for '{domain}'...[/]")

        failures = self.logger.get_failures_for_domain(domain, hours=48)
        if len(failures) < 5:
            return False

        # Generate candidate prompt variants using the engine itself
        prompt_generation_prompt = (
            f"You are improving an AI assistant's system prompt for the domain: '{domain}'.\n\n"
            f"These are example queries it failed on:\n"
            + "\n".join(f"- {f['query']}" for f in failures[:5])
            + "\n\nWrite 3 different improved system prompts that would handle these better."
            + "\nRespond with JSON: {\"prompts\": [\"prompt1\", \"prompt2\", \"prompt3\"]}"
        )

        result  = self.engine.generate_json(prompt_generation_prompt)
        prompts = result.get("prompts", [])

        if not prompts:
            return False

        # Score each candidate prompt on the failure examples
        best_prompt = None
        best_score  = self.tuned_prompts.get(f"{domain}_score", 0.0)

        for candidate in prompts:
            scores = []
            for fail in failures[:8]:
                test_answer = self.engine.generate(
                    fail["query"], system=candidate, temperature=0.1
                )
                score = self.engine.score(fail["query"], test_answer)
                scores.append(score)
            avg = sum(scores) / len(scores) if scores else 0.0
            if avg > best_score:
                best_score  = avg
                best_prompt = candidate

        if best_prompt and best_score > 0.6:
            self.tuned_prompts[domain]            = best_prompt
            self.tuned_prompts[f"{domain}_score"] = best_score
            self._save_tuned_prompts()
            console.print(f"  [green]New prompt for '{domain}' — avg score: {best_score:.2f}[/]")
            return True

        return False

    def _save_tuned_prompts(self):
        path = Path(MODELS_DIR) / "tuned_prompts.json"
        path.write_text(json.dumps(self.tuned_prompts, indent=2))

    def load_tuned_prompts(self):
        path = Path(MODELS_DIR) / "tuned_prompts.json"
        if path.exists():
            self.tuned_prompts = json.loads(path.read_text())
            console.print(f"[dim]Loaded {len(self.tuned_prompts)//2} tuned prompts[/]")

    def get_tuned_prompt(self, domain: str) -> str | None:
        return self.tuned_prompts.get(domain)

    # ── Specialist agent spawning ──────────────────────────────────────────────

    def _spawn_specialist(self, domain: str, failures: list[dict]) -> bool:
        """
        THE TRANSFORMER MOMENT.
        
        1. Generate synthetic training data from failures
        2. Write a LoRA fine-tune script
        3. Save the dataset
        4. Print instructions for running the fine-tune on Colab (free GPU)
        5. Register the future agent in the logger
        
        NOTE: The actual GPU training runs externally (Colab free tier).
        This function prepares everything and gives you the one-command to run.
        We cannot run multi-hour GPU training in this process — that's honest.
        What we CAN do: automate every other step.
        """
        console.print(f"\n  [bold yellow]Spawning specialist for domain: '{domain}'[/]")

        # ── Step 1: Generate training data ────────────────────────────────────
        training_data = self._generate_training_data(domain, failures)
        if not training_data:
            console.print("  [red]Could not generate training data[/]")
            return False

        # ── Step 2: Save dataset ──────────────────────────────────────────────
        dataset_path = Path(MODELS_DIR) / f"{domain}_training_data.jsonl"
        with open(dataset_path, "w") as f:
            for example in training_data:
                f.write(json.dumps(example) + "\n")
        console.print(f"  [green]Training data saved:[/] {dataset_path} ({len(training_data)} examples)")

        # ── Step 3: Write the fine-tune script ────────────────────────────────
        script_path = self._write_finetune_script(domain, dataset_path)
        console.print(f"  [green]Fine-tune script ready:[/] {script_path}")

        # ── Step 4: Register pending agent ────────────────────────────────────
        adapter_name = f"aria_{domain}_specialist"
        self.logger.register_agent(
            agent_name=adapter_name,
            domain=domain,
            model=f"{DEFAULT_MODEL}+{adapter_name}",
            adapter_path=str(Path(MODELS_DIR) / adapter_name),
        )

        # ── Step 5: Print human-readable instructions ──────────────────────────
        console.print(f"""
  [bold]Next step — run this fine-tune on Colab (free GPU):[/]
  
  1. Upload [cyan]{dataset_path}[/] to Google Colab
  2. Upload [cyan]{script_path}[/] to Google Colab  
  3. Run the script (takes ~1-2 hrs on T4 GPU)
  4. Download the adapter folder: [cyan]aria_{domain}_specialist/[/]
  5. Place it in: [cyan]{MODELS_DIR}/[/]
  6. Run: [cyan]ollama create aria_{domain} -f {MODELS_DIR}/{domain}.Modelfile[/]
  7. ARIA will auto-detect and route {domain} queries to the new specialist
  
  Colab link: https://colab.research.google.com/
        """)

        return True

    def _generate_training_data(self, domain: str, failures: list[dict]) -> list[dict]:
        """
        Generate synthetic Q&A pairs for fine-tuning.
        Uses the existing engine to create better answers for failed queries.
        """
        training_data = []

        console.print(f"  [dim]Generating {len(failures)} training examples...[/]")

        for fail in failures[:50]:  # cap at 50
            query = fail["query"]

            # Generate a better answer with a strong prompt
            better_answer = self.engine.generate(
                query,
                system=(
                    f"You are an expert assistant specializing in {domain}. "
                    f"Provide a thorough, accurate, well-structured answer. "
                    f"Be specific and cite your reasoning."
                ),
                temperature=0.2,
            )

            if better_answer and len(better_answer) > 30:
                training_data.append({
                    "instruction": query,
                    "output":      better_answer,
                    "domain":      domain,
                })

        # Also generate extra synthetic examples
        synthetic_prompt = (
            f"Generate 10 diverse question-answer pairs about: {domain}\n"
            f"Focus on common questions that need deep expertise.\n"
            f"Respond with JSON: {{\"pairs\": [{{\"q\": \"...\", \"a\": \"...\"}}]}}"
        )
        synth = self.engine.generate_json(synthetic_prompt)
        for pair in synth.get("pairs", []):
            if pair.get("q") and pair.get("a"):
                training_data.append({
                    "instruction": pair["q"],
                    "output":      pair["a"],
                    "domain":      domain,
                })

        return training_data

    def _write_finetune_script(self, domain: str, dataset_path: Path) -> Path:
        """Write the LoRA fine-tune Python script (runs on Colab free GPU)."""
        script = f'''"""
Auto-generated LoRA fine-tune script for domain: {domain}
Run this on Google Colab (free T4 GPU — 15GB VRAM).

Instructions:
    1. Runtime > Change runtime type > T4 GPU
    2. Run all cells
    3. Download the adapter folder when done
"""

# Install dependencies
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "peft", "bitsandbytes",
                "accelerate", "datasets", "trl"], check=True)

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # or microsoft/Phi-3-mini-4k-instruct
ADAPTER_NAME = "aria_{domain}_specialist"
DOMAIN       = "{domain}"
DATASET_FILE = "{dataset_path.name}"

# ── Load dataset ─────────────────────────────────────────────────────────────
def load_data():
    rows = []
    with open(DATASET_FILE) as f:
        for line in f:
            item = json.loads(line)
            rows.append({{
                "text": (
                    f"<|system|>You are an expert in {{DOMAIN}}.<|end|>\\n"
                    f"<|user|>{{item[\'instruction\']}}<|end|>\\n"
                    f"<|assistant|>{{item[\'output\']}}<|end|>"
                )
            }})
    return Dataset.from_list(rows)

dataset = load_data()
print(f"Loaded {{len(dataset)}} training examples")

# ── Load model in 4-bit (fits in free Colab 15GB VRAM) ───────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)

# ── LoRA config — trains only 0.5% of parameters ─────────────────────────────
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # should be ~0.5%

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=f"./{{ADAPTER_NAME}}",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    ),
)
trainer.train()
trainer.save_model(f"./{{ADAPTER_NAME}}")
print(f"\\nDone! Adapter saved to ./{{ADAPTER_NAME}}/")
print("Download this folder and place it in your ARIA models/ directory.")
'''
        script_path = Path(MODELS_DIR) / f"finetune_{domain}.py"
        script_path.write_text(script)
        return script_path

    # ── Stats display ──────────────────────────────────────────────────────────

    def _print_summary(self, summary: dict):
        table = Table(title="Adaptation Cycle Summary")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Domains checked",   str(summary["checked"]))
        table.add_row("Prompts tuned",     str(summary["tuned"]))
        table.add_row("Specialists queued",str(summary["spawned"]))
        table.add_row("Timestamp",         summary["timestamp"])
        console.print(table)

    def show_performance_report(self):
        """Print a full performance report from the logs."""
        stats = self.logger.get_stats()
        console.rule("[bold]ARIA Performance Report[/]")
        for k, v in stats.items():
            console.print(f"  {k}: [bold]{v}[/]")
