# ARIA — Adaptive Reasoning Intelligence Architecture

Your personal AI assistant. Runs 100% locally. No API keys. No cloud. No monthly cost.

---

## Complete step-by-step installation

### Prerequisites (install these first)

**Python 3.11**
- Windows: https://www.python.org/downloads/ — tick "Add Python to PATH"
- Mac: `brew install python@3.11`
- Linux: `sudo apt install python3.11 python3.11-venv`

Verify: `python --version` → must show 3.10 or higher

---

### Run the installer — does everything

```bash
cd aria
python install.py
```

The installer runs 10 steps:
```
STEP 1  Check Python version
STEP 2  Create .venv (virtual environment)
STEP 3  Install all Python packages
STEP 4  Install Ollama (local AI engine)
STEP 5  Download AI models (phi3:mini + moondream)
STEP 6  Install Playwright browser
STEP 7  Create folders and .env config
STEP 8  Self-test — verify everything works
STEP 9  Auto-start at login (optional, you choose)
STEP 10 Launch ARIA
```

Time: 10–20 minutes (mostly downloading ~4 GB of AI models)

---

### Manual steps (if installer fails on any step)

```bash
# Step 1 — Virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # Mac/Linux

# Step 2 — Python packages
pip install fastapi uvicorn requests python-dotenv rich pydantic
pip install sentence-transformers chromadb langdetect
pip install pymupdf python-docx openpyxl python-pptx beautifulsoup4 lxml trafilatura
pip install ddgs playwright z3-solver sympy
pip install flet pillow pystray plyer keyboard pyautogui pyperclip
pip install pip-audit youtube-transcript-api

# Step 3 — Playwright browser
playwright install chromium

# Step 4 — Ollama  (download from https://ollama.com — install like any app)

# Step 5 — AI models
ollama pull phi3:mini      # 1.8 GB — main model
ollama pull moondream      # 1.8 GB — reads images/screenshots

# Step 6 — Config
cp .env.example .env
```

---

## Start ARIA

```bash
python run.py              # server + opens browser (easiest)
python run.py --app        # server + desktop app + system tray
python server.py           # server only
```

---

## Access from any device

| Device | URL |
|--------|-----|
| This computer | http://localhost:8000 |
| Phone / tablet (same WiFi) | http://YOUR-PC-IP:8000 |
| NOVA reasoning | http://localhost:8000/nova |
| Learning dashboard | http://localhost:8000/learn |
| API docs | http://localhost:8000/docs |

Find your IP: `ipconfig` (Windows) · `ifconfig` (Mac/Linux)

---

## Hotkeys (work from any app)

| Hotkey | Action |
|--------|--------|
| Alt+Space | Open ARIA |
| Alt+Shift+S | Screenshot OCR |
| Alt+Shift+A | Analyse clipboard |
| Alt+Shift+Q | Quick question |

---

## OCR — reads any image (6 tiers, auto-fallback)

| Tier | Engine | How to add |
|------|--------|-----------|
| 1 | EasyOCR | `pip install easyocr` |
| 2 | Surya | `pip install surya-ocr` |
| 3 | PaddleOCR | `pip install paddleocr paddlepaddle` |
| 4 | Tesseract | `pip install pytesseract` + install binary |
| 5 | Classical | Tesseract binary only |
| 6 | Vision LLM | `ollama pull moondream` (already installed) |

You only need one. Install more for better fallback.
Check: http://localhost:8000/api/ocr/status

---

## Troubleshooting

`ollama not found` → install from https://ollama.com, restart terminal

`No module named X` → activate venv first, then `pip install X`

`Connection refused :8000` → run `python server.py` first

OCR not working → `ollama pull moondream` always works as Tier 6

Too slow → use phi3:mini (1.8GB), it's tuned for low-RAM systems
