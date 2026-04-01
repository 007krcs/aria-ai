"""
ARIA Fine-Tuning Script
========================
QLoRA fine-tune on Qwen2.5-3B-Instruct.

AUTO-DETECTS device:
  GPU (CUDA)  → uses Unsloth for 2x speed (recommended)
  CPU only    → uses standard HuggingFace transformers + PEFT (slow but works)
  No GPU?     → use --colab to export a ready-to-run Google Colab notebook (free T4 GPU)

Why Qwen2.5-3B:
  - Native multilingual (Hindi, Telugu, Tamil, 30+ languages)
  - 3B params = runs in 6GB VRAM (or CPU with patience)
  - Better instruction following than phi3:mini
  - ~2GB as GGUF Q4

Requirements (already installed):
  pip install trl peft accelerate datasets transformers

For GPU acceleration (optional):
  pip install unsloth

Usage:
  python training/finetune.py                    # auto-detect device
  python training/finetune.py --gpu              # force GPU mode
  python training/finetune.py --colab            # export Colab notebook
  python training/finetune.py --dataset path.jsonl
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
LORA_R       = 8          # lower rank for CPU (less memory)
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 1024       # shorter for CPU speed
BATCH_SIZE   = 1
GRAD_ACC     = 8          # effective batch = 8
EPOCHS       = 2          # fewer epochs on CPU
LR           = 2e-4
OUTPUT_NAME  = "aria-lora"


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {name} — {vram:.1f} GB VRAM")
            return "cuda"
    except Exception:
        pass
    print("[CPU] No GPU detected — using CPU (slow but works)")
    return "cpu"


def has_unsloth() -> bool:
    try:
        import unsloth  # noqa
        return True
    except Exception:
        return False


def load_dataset_file(path: str):
    from datasets import Dataset
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_chatml(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train_gpu_unsloth(dataset_path: str, val_path: str = None):
    """Fast path: GPU + Unsloth (2x faster, recommended)."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"[1/5] Loading {BASE_MODEL} with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True, dtype=None,
    )
    print(f"[2/5] Attaching LoRA adapters (r={LORA_R})...")
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", use_gradient_checkpointing=True, random_state=42,
    )
    return _run_trainer(model, tokenizer, dataset_path, val_path, fp16=True)


def train_cpu_peft(dataset_path: str, val_path: str = None):
    """CPU fallback: standard HuggingFace transformers + PEFT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch

    print(f"\n[CPU mode] Fine-tuning on CPU. Estimated time: 2-4 hours for 30 examples.")
    print(f"           Tip: Run with --colab to get a free T4 GPU on Google Colab instead.\n")

    print(f"[1/5] Loading {BASE_MODEL} (float32 for CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"[2/5] Attaching LoRA adapters (r={LORA_R})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable: {n_params/1e6:.1f}M params")

    return _run_trainer(model, tokenizer, dataset_path, val_path, fp16=False)


def _run_trainer(model, tokenizer, dataset_path, val_path, fp16=False):
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"[3/5] Loading dataset: {dataset_path}")
    train_ds = load_dataset_file(dataset_path)
    train_ds = train_ds.map(lambda ex: format_chatml(ex, tokenizer), remove_columns=["messages"])
    print(f"   Train: {len(train_ds)} examples")

    eval_ds = None
    if val_path and Path(val_path).exists():
        eval_ds = load_dataset_file(val_path)
        eval_ds = eval_ds.map(lambda ex: format_chatml(ex, tokenizer), remove_columns=["messages"])
        print(f"   Val:   {len(eval_ds)} examples")

    lora_output = str(OUT_DIR / OUTPUT_NAME)
    args = TrainingArguments(
        output_dir=lora_output,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        warmup_ratio=0.05,
        learning_rate=LR,
        fp16=fp16,
        bf16=False,
        logging_steps=5,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=42,
    )

    print("[4/5] Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=args,
    )
    trainer.train()

    print("[5/5] Saving LoRA adapter...")
    model.save_pretrained(lora_output)
    tokenizer.save_pretrained(lora_output)
    print(f"\n[OK] LoRA adapter saved: {lora_output}")
    print(f"     Next: python training/convert_gguf.py")
    return lora_output


def export_colab(dataset_path: str):
    """
    Export a Google Colab notebook that trains on free T4 GPU.
    Upload this notebook to colab.research.google.com
    """
    import json as _json
    import base64 as _b64

    dataset_content = Path(dataset_path).read_text(encoding="utf-8") if Path(dataset_path).exists() else ""
    # Base64-encode the dataset so control characters, backslashes and quotes
    # in the JSONL don't corrupt the Python source embedded in the notebook.
    dataset_b64 = _b64.b64encode(dataset_content.encode("utf-8")).decode("ascii")
    n_examples   = len([l for l in dataset_content.splitlines() if l.strip()])

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "accelerator": "GPU"},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# ARIA Fine-Tuning on Google Colab (Free T4 GPU)\n",
                "Runtime > Change runtime type > T4 GPU\n",
                "Then Run All (Ctrl+F9)"
            ]},
            {"cell_type": "code", "metadata": {}, "source": [
                "!pip install unsloth trl peft accelerate datasets -q"
            ], "outputs": [], "execution_count": None},
            {"cell_type": "code", "metadata": {}, "source": [
                "import base64, json\n",
                f"# Dataset: {n_examples} examples (base64-encoded to avoid control-char issues)\n",
                f"_b64 = '{dataset_b64}'\n",
                "dataset_jsonl = base64.b64decode(_b64).decode('utf-8')\n",
                "with open('aria_dataset.jsonl', 'w', encoding='utf-8') as f:\n",
                "    f.write(dataset_jsonl)\n",
                f"print(f'Loaded {n_examples} training examples')"
            ], "outputs": [], "execution_count": None},
            {"cell_type": "code", "metadata": {}, "source": [
                "from unsloth import FastLanguageModel\n",
                "from datasets import Dataset\n",
                "from trl import SFTTrainer\n",
                "from transformers import TrainingArguments\n",
                "\n",
                f"BASE_MODEL = '{BASE_MODEL}'\n",
                "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                f"    model_name=BASE_MODEL, max_seq_length={MAX_SEQ_LEN},\n",
                "    load_in_4bit=True,\n",
                ")\n",
                "model = FastLanguageModel.get_peft_model(\n",
                f"    model, r={LORA_R}, lora_alpha={LORA_ALPHA},\n",
                "    target_modules=['q_proj','k_proj','v_proj','o_proj',\n",
                "                    'gate_proj','up_proj','down_proj'],\n",
                "    bias='none', use_gradient_checkpointing=True,\n",
                ")\n",
                "\n",
                "examples = [json.loads(l) for l in open('aria_dataset.jsonl') if l.strip()]\n",
                "ds = Dataset.from_list(examples)\n",
                "ds = ds.map(lambda ex: {'text': tokenizer.apply_chat_template(\n",
                "    ex['messages'], tokenize=False, add_generation_prompt=False)},\n",
                "    remove_columns=['messages'])\n",
                "\n",
                "trainer = SFTTrainer(\n",
                "    model=model, tokenizer=tokenizer, train_dataset=ds,\n",
                f"    dataset_text_field='text', max_seq_length={MAX_SEQ_LEN},\n",
                f"    args=TrainingArguments(output_dir='aria-lora', num_train_epochs={EPOCHS},\n",
                f"        per_device_train_batch_size=2, gradient_accumulation_steps=4,\n",
                "        learning_rate=2e-4, fp16=True, logging_steps=5,\n",
                "        save_strategy='epoch', report_to='none'),\n",
                ")\n",
                "trainer.train()\n",
                "model.save_pretrained('aria-lora')\n",
                "tokenizer.save_pretrained('aria-lora')\n",
                "print('Training complete!')"
            ], "outputs": [], "execution_count": None},
            {"cell_type": "code", "metadata": {}, "source": [
                "# Save LoRA adapter as GGUF and download\n",
                "model.save_pretrained_gguf('aria-gguf', tokenizer, quantization_method='q4_k_m')\n",
                "from google.colab import files\n",
                "import os\n",
                "for f in os.listdir('aria-gguf'):\n",
                "    if f.endswith('.gguf'):\n",
                "        files.download(f'aria-gguf/{f}')\n",
                "        print(f'Download started: {f}')"
            ], "outputs": [], "execution_count": None},
            {"cell_type": "markdown", "metadata": {}, "source": [
                "## After download:\n",
                "1. Move the .gguf file to `C:/Users/chand/ai-remo/data/models/gguf/`\n",
                "2. Run: `python training/register_gguf.py --file aria-custom-q4.gguf`\n",
                "3. Restart server.py — ARIA will use your fine-tuned model"
            ]},
        ]
    }

    out = PROJECT_ROOT / "training" / "aria_colab_finetune.ipynb"
    with open(out, "w", encoding="utf-8") as f:
        _json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Colab notebook saved: {out}")
    print(f"\nNext steps:")
    print(f"  1. Go to https://colab.research.google.com")
    print(f"  2. File > Upload notebook > select: {out.name}")
    print(f"  3. Runtime > Change runtime type > T4 GPU (free)")
    print(f"  4. Run All (Ctrl+F9) — trains in ~15 minutes")
    print(f"  5. GGUF file downloads automatically — place in data/models/gguf/")
    return str(out)


def auto_find_dataset():
    training_dir = PROJECT_ROOT / "data" / "training"
    files = sorted(training_dir.glob("aria_dataset_*.jsonl"))
    if not files:
        print("[error] No dataset found. Run: python training/build_dataset.py first")
        sys.exit(1)
    val_stem = str(files[-1]).replace("aria_dataset_", "aria_val_")
    val = val_stem if Path(val_stem).exists() else None
    return str(files[-1]), val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ARIA")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--val",     type=str, default=None)
    parser.add_argument("--gpu",     action="store_true", help="Force GPU/Unsloth mode")
    parser.add_argument("--cpu",     action="store_true", help="Force CPU mode")
    parser.add_argument("--colab",   action="store_true", help="Export Colab notebook instead")
    args = parser.parse_args()

    dataset, val = args.dataset or auto_find_dataset()[0], args.val
    if not args.dataset:
        dataset, val = auto_find_dataset()

    print(f"\n{'='*55}")
    print(f"  ARIA Fine-Tuning")
    print(f"  Dataset : {Path(dataset).name}")
    print(f"  Model   : {BASE_MODEL}")
    print(f"{'='*55}\n")

    if args.colab:
        export_colab(dataset)
        sys.exit(0)

    device = detect_device()
    if (device == "cuda" or args.gpu) and not args.cpu:
        if has_unsloth():
            print("[mode] GPU + Unsloth (fastest)")
            train_gpu_unsloth(dataset, val)
        else:
            print("[mode] GPU + standard PEFT (unsloth not installed)")
            train_cpu_peft(dataset, val)
    else:
        print("[mode] CPU + standard PEFT")
        train_cpu_peft(dataset, val)
