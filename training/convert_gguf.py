"""
ARIA Model Converter
=====================
Merges LoRA adapters into base model → converts to GGUF → registers in Ollama.

Pipeline:
  1. Load base model + LoRA adapter
  2. Merge adapter weights into base model
  3. Save as HuggingFace format
  4. Convert to GGUF using llama.cpp
  5. Quantize (Q4_K_M = best quality/speed tradeoff)
  6. Create Ollama Modelfile with ARIA system prompt
  7. Register as "aria-custom" in Ollama

Usage:
    python training/convert_gguf.py
    python training/convert_gguf.py --lora data/models/aria-lora --name aria-v2
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR   = PROJECT_ROOT / "data" / "models"
GGUF_DIR     = MODELS_DIR / "gguf"
GGUF_DIR.mkdir(parents=True, exist_ok=True)

ARIA_SYSTEM = """You are ARIA (Adaptive Reasoning Intelligence Assistant), a personal AI assistant running locally on this device.
You were built by and for the person you are talking to.
You are NOT from Microsoft, NOT Cortana, NOT Copilot, NOT any cloud service.
You run fully offline and privately on this machine.
Be warm, concise, helpful, and direct.
Respond naturally in the same language the user uses.
For voice responses: keep answers to 2-3 natural sentences."""


def merge_lora(lora_path: str) -> str:
    """Merge LoRA adapter into base model and save as HF format."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[error] unsloth not installed. Run: pip install unsloth")
        sys.exit(1)

    merged_path = str(MODELS_DIR / "aria-merged")
    print(f"[1/4] Merging LoRA from {lora_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = lora_path,
        max_seq_length = 2048,
        load_in_4bit   = False,  # load full precision for merging
    )

    # Merge LoRA weights into base model
    model = model.merge_and_unload()
    model.save_pretrained(merged_path, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(merged_path)
    print(f"[✓] Merged model saved to: {merged_path}")
    return merged_path


def convert_to_gguf(merged_path: str, model_name: str) -> str:
    """Convert HuggingFace model to GGUF using llama.cpp."""
    gguf_path = str(GGUF_DIR / f"{model_name}.gguf")
    q_path    = str(GGUF_DIR / f"{model_name}-q4.gguf")

    print(f"\n[2/4] Converting to GGUF...")

    # Try to find llama.cpp convert script
    convert_scripts = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("C:/llama.cpp/convert_hf_to_gguf.py"),
        Path("llama.cpp/convert_hf_to_gguf.py"),
    ]
    convert_script = next((s for s in convert_scripts if s.exists()), None)

    if not convert_script:
        print("[warn] llama.cpp not found. Downloading...")
        ret = subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(Path.home() / "llama.cpp"),
        ], capture_output=True)
        if ret.returncode != 0:
            print("[error] Could not clone llama.cpp. Install manually from https://github.com/ggerganov/llama.cpp")
            print("        Then run: python llama.cpp/convert_hf_to_gguf.py <merged_path> --outfile <output.gguf>")
            return None
        convert_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"

    # Install llama.cpp Python deps
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(convert_script.parent / "requirements" / "requirements-convert_hf_to_gguf.txt"),
                    "-q"], check=False)

    # Convert to GGUF (F16 first)
    ret = subprocess.run([
        sys.executable, str(convert_script),
        merged_path,
        "--outfile", gguf_path,
        "--outtype", "f16",
    ])
    if ret.returncode != 0:
        print(f"[error] GGUF conversion failed")
        return None
    print(f"[✓] GGUF (F16) saved: {gguf_path}")

    # Quantize to Q4_K_M (best balance of quality and speed)
    print(f"[3/4] Quantizing to Q4_K_M...")
    quantize_bin = Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = Path.home() / "llama.cpp" / "quantize"

    if quantize_bin.exists():
        ret = subprocess.run([str(quantize_bin), gguf_path, q_path, "Q4_K_M"])
        if ret.returncode == 0:
            print(f"[✓] Quantized model: {q_path}")
            # Clean up F16 version (large)
            Path(gguf_path).unlink(missing_ok=True)
            return q_path
        else:
            print("[warn] Quantization failed, using F16 version")
            return gguf_path
    else:
        print("[warn] llama-quantize not found. Build llama.cpp with: cd ~/llama.cpp && mkdir build && cd build && cmake .. && cmake --build .")
        print(f"       Using unquantized F16 model: {gguf_path}")
        return gguf_path


def register_ollama(gguf_path: str, model_name: str) -> bool:
    """Create Ollama Modelfile and register the model."""
    modelfile_path = GGUF_DIR / f"{model_name}.Modelfile"
    modelfile_content = f"""# ARIA Custom Fine-tuned Model
FROM {gguf_path}

SYSTEM \"\"\"{ARIA_SYSTEM}\"\"\"

PARAMETER temperature 0.5
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
    modelfile_path.write_text(modelfile_content)
    print(f"\n[4/4] Registering in Ollama as '{model_name}'...")

    ret = subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)],
                         capture_output=True, text=True)
    if ret.returncode == 0:
        print(f"[✓] Model '{model_name}' registered in Ollama!")
        print(f"\n    To use: ollama run {model_name}")
        print(f"    To set as ARIA default: edit ARIA_MODEL in .env or core/config.py")
        print(f"\n    Or let ARIA auto-switch by running:")
        print(f"    python training/auto_train.py --activate {model_name}")
        return True
    else:
        print(f"[error] Ollama registration failed: {ret.stderr}")
        print(f"        Modelfile saved at: {modelfile_path}")
        print(f"        Run manually: ollama create {model_name} -f {modelfile_path}")
        return False


def update_aria_model(model_name: str):
    """Update ARIA's active model in the environment config."""
    env_path = PROJECT_ROOT / ".env"
    env_content = env_path.read_text() if env_path.exists() else ""

    if "ARIA_DEFAULT_MODEL" in env_content:
        lines = env_content.splitlines()
        lines = [f"ARIA_DEFAULT_MODEL={model_name}" if l.startswith("ARIA_DEFAULT_MODEL") else l
                 for l in lines]
        env_path.write_text("\n".join(lines) + "\n")
    else:
        with open(env_path, "a") as f:
            f.write(f"\nARIA_DEFAULT_MODEL={model_name}\n")

    print(f"[✓] ARIA default model set to: {model_name}")
    print(f"    Restart server.py to activate the new model")


def run(lora_path: str, model_name: str = "aria-custom", activate: bool = False):
    print(f"\n── ARIA Model Converter ─────────────────────────────")
    merged = merge_lora(lora_path)
    gguf   = convert_to_gguf(merged, model_name)
    if not gguf:
        return False
    ok = register_ollama(gguf, model_name)
    if ok and activate:
        update_aria_model(model_name)
    print(f"─────────────────────────────────────────────────────\n")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA adapter to Ollama GGUF model")
    parser.add_argument("--lora", type=str, default=str(MODELS_DIR / "aria-lora"),
                        help="Path to LoRA adapter directory")
    parser.add_argument("--name", type=str, default="aria-custom",
                        help="Ollama model name")
    parser.add_argument("--activate", action="store_true",
                        help="Set as ARIA's default model after conversion")
    args = parser.parse_args()

    if not Path(args.lora).exists():
        print(f"[error] LoRA path not found: {args.lora}")
        print(f"        Run: python training/finetune.py first")
        sys.exit(1)

    run(args.lora, args.name, args.activate)
