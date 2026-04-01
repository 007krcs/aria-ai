"""
ARIA Auto-Trainer
==================
Watches the training DB. When enough new data accumulates,
automatically triggers: build_dataset → finetune → convert → activate.

This is the continuous learning loop that makes ARIA improve over time.

Triggers:
  • NEW_EXAMPLES_THRESHOLD  — N new examples since last training
  • QUALITY_THRESHOLD       — minimum average confidence score
  • MAX_DAYS_BETWEEN_TRAINS — force retrain even if threshold not met

Usage:
    # Check status and trigger if ready:
    python training/auto_train.py

    # Run as daemon (checks every hour):
    python training/auto_train.py --daemon

    # Force retrain now:
    python training/auto_train.py --force

    # Activate a specific model:
    python training/auto_train.py --activate aria-custom
"""

import sys
import json
import time
import sqlite3
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DB       = PROJECT_ROOT / "logs" / "aria_logs.db"
STATE_FILE    = PROJECT_ROOT / "data" / "training" / "auto_train_state.json"
MODELS_DIR    = PROJECT_ROOT / "data" / "models"

# ── Thresholds ────────────────────────────────────────────────────────────────
NEW_EXAMPLES_THRESHOLD   = 50      # retrain after N new conversations
QUALITY_THRESHOLD        = 0.65    # minimum average confidence
MAX_DAYS_BETWEEN_TRAINS  = 7       # force weekly retrain regardless
MIN_EXAMPLES_EVER        = 30      # never train on less than this total
CHECK_INTERVAL_HOURS     = 1       # daemon check frequency


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "last_train_ts": None,
        "last_example_count": 0,
        "trained_versions": [],
        "active_model": None,
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_db_stats() -> dict:
    """Get current state of training data."""
    if not LOGS_DB.exists():
        return {"total": 0, "avg_confidence": 0.0, "new_since_last": 0}

    conn = sqlite3.connect(LOGS_DB)
    cur  = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    avg_q = cur.execute(
        "SELECT AVG(confidence) FROM training_examples WHERE confidence > 0"
    ).fetchone()[0] or 0.0

    conn.close()
    return {"total": total, "avg_confidence": round(avg_q, 3)}


def should_train(state: dict, stats: dict, force: bool = False) -> tuple[bool, str]:
    """Decide whether to trigger fine-tuning."""
    if force:
        return True, "forced"

    if stats["total"] < MIN_EXAMPLES_EVER:
        return False, f"not enough data ({stats['total']}/{MIN_EXAMPLES_EVER} examples)"

    if stats["avg_confidence"] < QUALITY_THRESHOLD:
        return False, f"quality too low ({stats['avg_confidence']:.2f} < {QUALITY_THRESHOLD})"

    new_count = stats["total"] - state.get("last_example_count", 0)
    if new_count >= NEW_EXAMPLES_THRESHOLD:
        return True, f"{new_count} new examples (threshold={NEW_EXAMPLES_THRESHOLD})"

    if state.get("last_train_ts"):
        last = datetime.fromisoformat(state["last_train_ts"])
        days_since = (datetime.now() - last).days
        if days_since >= MAX_DAYS_BETWEEN_TRAINS:
            return True, f"{days_since} days since last training"

    if state.get("last_train_ts") is None and stats["total"] >= MIN_EXAMPLES_EVER:
        return True, "first training run"

    return False, f"only {new_count} new examples (need {NEW_EXAMPLES_THRESHOLD})"


def run_pipeline(use_gpu: bool = False, model_name: str = "aria-custom") -> bool:
    """Run the full training pipeline."""
    print("\n" + "="*55)
    print("  ARIA CONTINUOUS LEARNING — TRAINING PIPELINE")
    print("="*55)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*55 + "\n")

    scripts = PROJECT_ROOT / "training"

    # Step 1: Build dataset
    print("━━ Step 1/3: Building dataset ━━━━━━━━━━━━━━━━━━━━━━━")
    ret = subprocess.run([sys.executable, str(scripts / "build_dataset.py")])
    if ret.returncode != 0:
        print("[error] Dataset build failed")
        return False

    # Step 2: Fine-tune
    print("\n━━ Step 2/3: Fine-tuning ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    cmd = [sys.executable, str(scripts / "finetune.py")]
    if use_gpu:
        cmd.append("--gpu")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("[error] Fine-tuning failed")
        return False

    # Step 3: Convert + register
    print("\n━━ Step 3/3: Converting to GGUF + Ollama ━━━━━━━━━━━")
    lora_path = str(MODELS_DIR / "aria-lora")
    ret = subprocess.run([
        sys.executable, str(scripts / "convert_gguf.py"),
        "--lora", lora_path,
        "--name", model_name,
        "--activate",
    ])
    if ret.returncode != 0:
        print("[error] Conversion failed")
        return False

    print(f"\n{'='*55}")
    print(f"  ✓ TRAINING COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Model: {model_name}")
    print(f"  Restart server.py to activate the new model.")
    print(f"{'='*55}\n")
    return True


def status():
    """Print current training status."""
    state = load_state()
    stats = get_db_stats()
    new   = stats["total"] - state.get("last_example_count", 0)
    ready, reason = should_train(state, stats)

    print("\n-- ARIA Training Status --------------------------------------")
    print(f"  Total examples    : {stats['total']}")
    print(f"  Avg quality       : {stats['avg_confidence']:.2f}")
    print(f"  New since training: {new}")
    print(f"  Active model      : {state.get('active_model', 'none (using Ollama default)')}")
    print(f"  Last trained      : {state.get('last_train_ts', 'never')}")
    print(f"  Training needed   : {'YES — ' + reason if ready else 'No — ' + reason}")
    print(f"  Versions trained  : {len(state.get('trained_versions', []))}")
    print("-------------------------------------------------------------\n")
    return ready


def activate(model_name: str):
    """Switch ARIA to a specific model."""
    env_path = PROJECT_ROOT / ".env"
    env_content = env_path.read_text() if env_path.exists() else ""
    if "ARIA_DEFAULT_MODEL" in env_content:
        lines = [f"ARIA_DEFAULT_MODEL={model_name}" if l.startswith("ARIA_DEFAULT_MODEL") else l
                 for l in env_content.splitlines()]
        env_path.write_text("\n".join(lines) + "\n")
    else:
        with open(env_path, "a") as f:
            f.write(f"\nARIA_DEFAULT_MODEL={model_name}\n")

    state = load_state()
    state["active_model"] = model_name
    save_state(state)
    print(f"[✓] ARIA will use model: {model_name}")
    print(f"    Restart server.py to apply.")


def check_and_train(force: bool = False, use_gpu: bool = False, model_name: str = "aria-custom"):
    state = load_state()
    stats = get_db_stats()
    ready, reason = should_train(state, stats, force)

    print(f"[check] {stats['total']} examples | quality={stats['avg_confidence']:.2f} | {reason}")

    if not ready:
        return False

    print(f"[!] Training triggered: {reason}")
    success = run_pipeline(use_gpu, model_name)

    if success:
        state["last_train_ts"]     = datetime.now().isoformat()
        state["last_example_count"] = stats["total"]
        state["active_model"]       = model_name
        state["trained_versions"].append({
            "ts":      datetime.now().isoformat(),
            "model":   model_name,
            "examples": stats["total"],
        })
        save_state(state)

    return success


def daemon(use_gpu: bool = False, model_name: str = "aria-custom"):
    """Run as background daemon, checking periodically."""
    print(f"[daemon] Auto-trainer running. Checks every {CHECK_INTERVAL_HOURS}h.")
    print(f"         Trigger: {NEW_EXAMPLES_THRESHOLD} new examples or {MAX_DAYS_BETWEEN_TRAINS} days")
    print(f"         Press Ctrl+C to stop\n")
    while True:
        try:
            check_and_train(use_gpu=use_gpu, model_name=model_name)
            next_check = datetime.now() + timedelta(hours=CHECK_INTERVAL_HOURS)
            print(f"[daemon] Next check: {next_check.strftime('%H:%M')}")
            time.sleep(CHECK_INTERVAL_HOURS * 3600)
        except KeyboardInterrupt:
            print("\n[daemon] Stopped.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIA auto-training controller")
    parser.add_argument("--force",    action="store_true", help="Force training now")
    parser.add_argument("--daemon",   action="store_true", help="Run as background daemon")
    parser.add_argument("--status",   action="store_true", help="Show training status")
    parser.add_argument("--activate", type=str, default=None, help="Activate a model by name")
    parser.add_argument("--gpu",      action="store_true", help="Use GPU for training")
    parser.add_argument("--name",     type=str, default="aria-custom", help="Model name in Ollama")
    args = parser.parse_args()

    if args.activate:
        activate(args.activate)
    elif args.status:
        status()
    elif args.daemon:
        daemon(args.gpu, args.name)
    else:
        check_and_train(args.force, args.gpu, args.name)
