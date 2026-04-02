"""
ARIA — Central configuration  (v3 — adaptive)
==============================================
All settings live here. Adaptive values auto-tune to your hardware.
Override any setting via environment variable or .env file.
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Suppress irrelevant HuggingFace / transformers warnings ─────────────────
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*Unrecognized keys.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ── Optional HF token (higher download rate limits — free at huggingface.co) ─
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

# ── Project paths ────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
LOGS_DIR   = ROOT / "logs"
MODELS_DIR = ROOT / "models"

for d in (DATA_DIR, LOGS_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)

# ── Adaptive defaults (overridden at runtime by core.adaptive) ───────────────
# These are the fallback values used before the adaptive profile is built.
# In practice, core.engine and core.memory read from core.adaptive first.

def _adaptive_model() -> str:
    """Best model for available RAM — used as default before adaptive profile loads."""
    try:
        import psutil
        gb = psutil.virtual_memory().available / (1024 ** 3)
        if gb < 2:   return "phi3:mini"
        if gb < 4:   return "phi3:mini"
        if gb < 8:   return "llama3.2:3b"
        if gb < 16:  return "llama3.1:8b"
        return "llama3.1:70b"
    except Exception:
        return "phi3:mini"

def _adaptive_tokens() -> int:
    try:
        import psutil
        gb = psutil.virtual_memory().available / (1024 ** 3)
        if gb < 2:  return 256
        if gb < 4:  return 512
        if gb < 8:  return 1024
        return 2048
    except Exception:
        return 512

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL")   or _adaptive_model()
FAST_MODEL      = os.getenv("FAST_MODEL",   "phi3:mini")
DEEP_MODEL      = os.getenv("DEEP_MODEL")   or DEFAULT_MODEL
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", str(_adaptive_tokens())))

# ── Embeddings ────────────────────────────────────────────────────────────────
# Multilingual MiniLM covers 50+ languages on CPU; falls back to MiniLM-L6
# on very low RAM devices (adaptive manager overrides this at runtime).
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# ── ChromaDB (vector memory) ──────────────────────────────────────────────────
CHROMA_PATH       = str(DATA_DIR / "chroma_db")
CHROMA_COLLECTION = "aria_memory"
MEMORY_TTL_DAYS   = int(os.getenv("MEMORY_TTL_DAYS", "90"))  # 0 = never expire

# ── RAG ───────────────────────────────────────────────────────────────────────
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",     "400"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP",  "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
MIN_SIMILARITY  = float(os.getenv("MIN_SIMILARITY", "0.45"))

# ── Agent system ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD    = float(os.getenv("CONFIDENCE_THRESHOLD",    "0.72"))
FAILURE_COUNT_THRESHOLD = int(os.getenv("FAILURE_COUNT_THRESHOLD",   "15"))
ADAPTATION_CHECK_HOURS  = int(os.getenv("ADAPTATION_CHECK_HOURS",    "6"))

# ── Logging (SQLite) ──────────────────────────────────────────────────────────
LOG_DB_PATH = str(LOGS_DIR / "aria_logs.db")

# ── API server ────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
# Railway sets PORT; also accept API_PORT for local override
API_PORT = int(os.getenv("PORT") or os.getenv("API_PORT", "8000"))

# ── Web search (DuckDuckGo — free, no key) ───────────────────────────────────
SEARCH_BACKEND     = "duckduckgo"
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

# ── Voice ─────────────────────────────────────────────────────────────────────
VOICE_DEFAULT      = os.getenv("VOICE_DEFAULT", "en-IN-NeerjaNeural")
VOICE_HINDI        = os.getenv("VOICE_HINDI",   "hi-IN-SwaraNeural")
WHISPER_MODEL      = os.getenv("WHISPER_MODEL", "auto")   # auto = adaptive
VAD_THRESHOLD      = float(os.getenv("VAD_THRESHOLD", "0.45"))

# ── Device control ────────────────────────────────────────────────────────────
ADB_AUTO_CONNECT   = os.getenv("ADB_AUTO_CONNECT", "").strip()   # IP:PORT
HA_URL             = os.getenv("HA_URL", "").strip()              # Home Assistant
HA_TOKEN           = os.getenv("HA_TOKEN", "").strip()
HUE_IP             = os.getenv("HUE_IP", "").strip()             # Philips Hue
HUE_USER           = os.getenv("HUE_USER", "").strip()

# ── Neural Agent Network (neuromorphic inter-agent communication) ─────────────
HEBBIAN_ALPHA        = float(os.getenv("HEBBIAN_ALPHA",        "0.10"))  # learning rate
SYNAPSE_DECAY_RATE   = float(os.getenv("SYNAPSE_DECAY_RATE",   "0.01"))  # decay per minute
SYNAPSE_BASELINE_W   = float(os.getenv("SYNAPSE_BASELINE_W",   "0.50"))  # initial weight
SYNAPSE_MIN_W        = float(os.getenv("SYNAPSE_MIN_W",        "0.05"))  # floor weight
LATERAL_INHIBIT_CONF = float(os.getenv("LATERAL_INHIBIT_CONF", "0.85"))  # inhibition threshold
WAVE1_TIMEOUT_S      = float(os.getenv("WAVE1_TIMEOUT_S",      "6.0"))   # fast wave timeout
WAVE2_TIMEOUT_S      = float(os.getenv("WAVE2_TIMEOUT_S",      "10.0"))  # reasoning wave timeout
SIGNAL_TTL_S         = float(os.getenv("SIGNAL_TTL_S",         "30.0"))  # signal expiry
NEURAL_WEIGHTS_PATH  = DATA_DIR / "synaptic_weights.json"                 # persisted Hebbian weights
CONSENSUS_N          = int(os.getenv("CONSENSUS_N",            "2"))      # min agents for consensus
CONSENSUS_SIM        = float(os.getenv("CONSENSUS_SIM",        "0.35"))   # Jaccard threshold

# ── LLM Provider ─────────────────────────────────────────────────────────────
# Priority order (auto): groq → openai → anthropic → gemini → together → ollama
# "auto"      → first available provider with a key set
# "groq"      → Groq (free tier, fastest)
# "openai"    → OpenAI GPT-4o
# "anthropic" → Claude 3.5 Sonnet
# "gemini"    → Google Gemini
# "together"  → Together.ai (free models)
# "ollama"    → Local Ollama (fallback, no key needed)
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "auto")

# ── Groq (free — 14,400 req/day) — console.groq.com ─────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "").strip()
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")

# ── OpenAI — platform.openai.com ─────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY",  "").strip()
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL",    "gpt-4o-mini")

# ── Anthropic Claude — console.anthropic.com ─────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL",   "claude-haiku-4-5-20251001")

# ── Google Gemini — aistudio.google.com (free) ───────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",  "").strip()
GEMINI_MODEL    = os.getenv("GEMINI_MODEL",    "gemini-1.5-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# ── Together.ai (free models) — api.together.ai ──────────────────────────────
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_MODEL   = os.getenv("TOGETHER_MODEL",  "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

# ── Ollama (local fallback — no key needed) ───────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Security ──────────────────────────────────────────────────────────────────
JWT_SECRET_PATH    = str(DATA_DIR / ".jwt_secret")
MAX_REQUEST_BYTES  = int(os.getenv("MAX_REQUEST_BYTES", str(50 * 1024 * 1024)))  # 50 MB
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() != "false"
