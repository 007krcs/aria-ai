"""
ARIA Dataset Builder
====================
Exports ARIA's collected interactions into ChatML fine-tuning format.

Sources:
  1. logs/aria_logs.db → training_examples (chat Q&A pairs)
  2. data/conversations.db → messages (full conversation threads)
  3. data/training/*.jsonl → previously exported JSONL

Output: data/training/aria_dataset_YYYYMMDD.jsonl
Format: ChatML (messages list — works with Qwen2.5, Llama3, Phi3)

Usage:
    python training/build_dataset.py
    python training/build_dataset.py --min-quality 0.7 --output custom.jsonl
"""

import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
LOGS_DB  = PROJECT_ROOT / "logs"  / "aria_logs.db"
CONV_DB  = PROJECT_ROOT / "data"  / "conversations.db"
OUT_DIR  = PROJECT_ROOT / "data"  / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ARIA system identity — injected into every training example
ARIA_SYSTEM = (
    "You are ARIA (Adaptive Reasoning Intelligence Assistant), a personal AI assistant "
    "running locally on this device. You were built by and for the person you are talking to. "
    "You are NOT from Microsoft, NOT Cortana, NOT Copilot, NOT any cloud service. "
    "You run fully offline and privately. "
    "Be warm, concise, helpful, and direct. Respond naturally in the same language the user uses."
)


def load_training_examples(min_quality: float = 0.65) -> list[dict]:
    """Load Q&A pairs from training_examples table."""
    if not LOGS_DB.exists():
        print(f"[warn] {LOGS_DB} not found — skipping training_examples")
        return []

    conn = sqlite3.connect(LOGS_DB)
    cur  = conn.cursor()
    rows = cur.execute("""
        SELECT question, answer, domain, confidence
        FROM training_examples
        WHERE confidence >= ?
          AND LENGTH(answer) > 30
          AND LENGTH(question) > 5
        ORDER BY ts ASC
    """, (min_quality,)).fetchall()
    conn.close()

    examples = []
    for question, answer, domain, confidence in rows:
        # Skip examples where ARIA claims to be Microsoft (bad training signal)
        bad_phrases = ["developed by microsoft", "i'm microsoft", "microsoft ai",
                       "i am microsoft", "built by microsoft"]
        if any(p in answer.lower() for p in bad_phrases):
            continue
        examples.append({
            "messages": [
                {"role": "system",    "content": ARIA_SYSTEM},
                {"role": "user",      "content": question.strip()},
                {"role": "assistant", "content": answer.strip()},
            ],
            "_meta": {"domain": domain, "confidence": confidence, "source": "training_examples"},
        })
    print(f"[✓] training_examples: {len(examples)} examples (min_quality={min_quality})")
    return examples


def load_conversations(min_turns: int = 2) -> list[dict]:
    """
    Load multi-turn conversation threads from conversations.db.
    Groups messages by session, builds multi-turn ChatML examples.
    """
    if not CONV_DB.exists():
        print(f"[warn] {CONV_DB} not found — skipping conversations")
        return []

    conn = sqlite3.connect(CONV_DB)
    cur  = conn.cursor()
    rows = cur.execute("""
        SELECT session, role, text, ts
        FROM messages
        WHERE LENGTH(text) > 10
        ORDER BY session, ts ASC
    """).fetchall()
    conn.close()

    # Group by session
    sessions: dict[str, list] = {}
    for session, role, text, ts in rows:
        sessions.setdefault(session, []).append((role, text))

    examples = []
    for session_id, turns in sessions.items():
        if len(turns) < min_turns * 2:
            continue  # skip very short sessions

        # Build sliding window — each window of N turns is one training example
        window = 6   # 3 user + 3 aria turns per example
        messages = [{"role": "system", "content": ARIA_SYSTEM}]

        for role, text in turns:
            mapped_role = "assistant" if role == "aria" else "user"
            messages.append({"role": mapped_role, "content": text.strip()})

            # Every time we have a complete user+assistant pair, emit an example
            if (mapped_role == "assistant"
                    and len(messages) >= 3           # system + at least 1 pair
                    and len(messages) <= window + 1):
                # Skip if assistant says it's from Microsoft
                bad = ["developed by microsoft", "i'm microsoft", "microsoft ai"]
                if any(p in text.lower() for p in bad):
                    continue
                examples.append({
                    "messages": list(messages),
                    "_meta": {"session": session_id, "source": "conversation"},
                })

    print(f"[✓] conversations: {len(examples)} multi-turn examples from {len(sessions)} sessions")
    return examples


def load_previous_jsonl() -> list[dict]:
    """Load any previously exported JSONL files to avoid losing old data."""
    examples = []
    for f in OUT_DIR.glob("aria_dataset_*.jsonl"):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if "messages" in obj:
                        examples.append(obj)
    print(f"[✓] previous JSONL exports: {len(examples)} examples")
    return examples


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove duplicate examples by hashing the user message."""
    seen = set()
    unique = []
    for ex in examples:
        msgs = ex["messages"]
        user_msgs = " ".join(m["content"] for m in msgs if m["role"] == "user")
        key = hash(user_msgs[:200])
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def split_train_val(examples: list[dict], val_ratio: float = 0.1):
    """Split into train / validation sets."""
    import random
    random.shuffle(examples)
    split = max(1, int(len(examples) * val_ratio))
    return examples[split:], examples[:split]


def build(min_quality: float = 0.65, output: str = None):
    print("\n── ARIA Dataset Builder ─────────────────────────────")

    all_examples = []
    all_examples.extend(load_training_examples(min_quality))
    all_examples.extend(load_conversations())
    # Don't include old JSONL by default to avoid stale bad data
    # all_examples.extend(load_previous_jsonl())

    all_examples = deduplicate(all_examples)
    print(f"\n[total] {len(all_examples)} unique examples after dedup")

    if len(all_examples) < 10:
        print("[warn] Less than 10 examples — collect more data before fine-tuning")
        print("       Keep using ARIA via chat and voice to accumulate training data")
        return None

    train, val = split_train_val(all_examples)
    print(f"[split] train={len(train)}  val={len(val)}")

    # Write output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_train = OUT_DIR / (output or f"aria_dataset_{timestamp}.jsonl")
    out_val   = OUT_DIR / f"aria_val_{timestamp}.jsonl"

    with open(out_train, "w", encoding="utf-8") as f:
        for ex in train:
            # Strip _meta before writing (not needed by trainer)
            obj = {"messages": ex["messages"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(out_val, "w", encoding="utf-8") as f:
        for ex in val:
            obj = {"messages": ex["messages"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[✓] Saved: {out_train}")
    print(f"[✓] Saved: {out_val}")
    print("─────────────────────────────────────────────────────\n")
    return str(out_train), str(out_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ARIA fine-tuning dataset")
    parser.add_argument("--min-quality", type=float, default=0.65)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    build(args.min_quality, args.output)
