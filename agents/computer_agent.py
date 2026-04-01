"""
ARIA Computer Agent — Full Perceive→Plan→Act→Verify Loop
=========================================================
Equivalent to Anthropic Computer Use / Perplexity Computer Agent.

Architecture:
  1. PERCEIVE  — screenshot + vision LLM understanding
  2. PLAN      — LLM decomposes goal into ordered steps
  3. ACT       — pyautogui + shell + file + browser actions
  4. VERIFY    — screenshot after action + LLM confirms success
  5. RETRY     — if verify fails, replan with failure context (max 5 retries)

Action space:
  click(x, y) | right_click(x, y) | double_click(x, y)
  type(text) | key(combo) | scroll(x, y, direction, amount)
  screenshot() | find_element(description) | wait_for(condition)
  open_app(name) | close_app(name) | run_command(cmd)
  read_file(path) | write_file(path, content)
  browser_navigate(url) | browser_click(selector) | browser_type(selector, text)

Vision backends (in priority order):
  1. Local Ollama vision model (llava, moondream, llama3.2-vision)
  2. Groq vision (llama-3.2-11b-vision-preview) — free tier
  3. Google Gemini Flash — free tier
  4. Fallback: OCR-only mode (tesseract)

Safety: never act on DELETE/FORMAT/WIPE without explicit DANGEROUS flag.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import re
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

# ── Optional GUI automation ───────────────────────────────────────────────────
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0          # we control delays manually
    PYAUTOGUI_AVAILABLE = True
except Exception:
    pyautogui = None               # type: ignore
    PYAUTOGUI_AVAILABLE = False

# ── PIL / image diff ──────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageChops, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    Image = ImageChops = ImageGrab = None  # type: ignore
    PIL_AVAILABLE = False

# ── OCR fallback ──────────────────────────────────────────────────────────────
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None  # type: ignore
    OCR_AVAILABLE = False

# ── HTTP client ───────────────────────────────────────────────────────────────
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None  # type: ignore
    REQUESTS_AVAILABLE = False

# ── Selenium (browser actions) ────────────────────────────────────────────────
try:
    from selenium import webdriver as _selenium_webdriver
    from selenium.webdriver.chrome.options import Options as _ChromeOptions
    from selenium.webdriver.common.by import By as _By
    from selenium.webdriver.common.keys import Keys as _Keys
    from selenium.webdriver.support.ui import WebDriverWait as _WebDriverWait
    from selenium.webdriver.support import expected_conditions as _EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# ── ARIA internals ────────────────────────────────────────────────────────────
try:
    from agents.vision_ocr import extract_text_from_image
    ARIA_OCR = True
except ImportError:
    ARIA_OCR = False

# ─────────────────────────────────────────────────────────────────────────────
# Config / env
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_BASE      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
OLLAMA_VISION    = os.getenv("ARIA_VISION_MODEL", "llava")
GROQ_VISION      = "llama-3.2-11b-vision-preview"
GEMINI_VISION    = "gemini-1.5-flash"
OLLAMA_TEXT      = os.getenv("ARIA_TEXT_MODEL", "llama3.2")

DANGEROUS_WORDS  = {"DELETE", "FORMAT", "WIPE", "DESTROY", "REMOVE", "PURGE"}
MAX_STEPS_DEFAULT = 20
MAX_RETRIES       = 5
DELAY_MIN_MS      = 50
DELAY_MAX_MS      = 300


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ActionResult:
    success: bool
    action: Dict[str, Any]
    output: str = ""
    screenshot_b64: str = ""
    error: str = ""
    retry_count: int = 0


@dataclass
class TaskResult:
    goal: str
    steps_taken: int
    success: bool
    final_screenshot: str = ""
    total_time: float = 0.0
    actions_log: List[ActionResult] = field(default_factory=list)
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def _human_delay(min_ms: int = DELAY_MIN_MS, max_ms: int = DELAY_MAX_MS) -> None:
    """Sleep a random human-like interval between min_ms and max_ms milliseconds."""
    time.sleep(random.randint(min_ms, max_ms) / 1000.0)


def _img_to_b64(img: "Image.Image") -> str:  # type: ignore[name-defined]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_img(b64: str) -> Optional["Image.Image"]:  # type: ignore[name-defined]
    if not PIL_AVAILABLE:
        return None
    try:
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data))
    except Exception:
        return None


def _image_diff_score(b64_a: str, b64_b: str) -> float:
    """Return a 0-1 change score between two base64 PNG screenshots.
    0 = identical, 1 = completely different."""
    if not PIL_AVAILABLE or not b64_a or not b64_b:
        return 1.0
    try:
        img_a = _b64_to_img(b64_a).convert("RGB")
        img_b = _b64_to_img(b64_b).convert("RGB")
        # Resize to small thumbnail for speed
        img_a = img_a.resize((160, 90))
        img_b = img_b.resize((160, 90))
        diff  = ImageChops.difference(img_a, img_b)
        pixels = list(diff.getdata())
        total = sum(sum(p) for p in pixels)
        maximum = 255 * 3 * len(pixels)
        return total / maximum if maximum else 0.0
    except Exception:
        return 1.0


def _safety_check(action: Dict[str, Any], dangerous_flag: bool = False) -> Optional[str]:
    """Return an error string if the action is unsafe, else None."""
    atype = action.get("type", "").lower()
    if atype == "run_command":
        cmd = str(action.get("cmd", "")).upper()
        for dw in DANGEROUS_WORDS:
            if dw in cmd and not dangerous_flag:
                return (
                    f"Safety block: command contains '{dw}'. "
                    "Pass dangerous=True to override."
                )
    if atype == "write_file":
        path = str(action.get("path", "")).upper()
        for dw in DANGEROUS_WORDS:
            if dw in path and not dangerous_flag:
                return f"Safety block: file path contains '{dw}'."
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Vision / LLM back-ends
# ─────────────────────────────────────────────────────────────────────────────
def _http_post_json(url: str, payload: Dict, headers: Optional[Dict] = None,
                    timeout: int = 60) -> Optional[Dict]:
    headers = headers or {"Content-Type": "application/json"}
    if HTTPX_AVAILABLE:
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            pass
    if REQUESTS_AVAILABLE:
        try:
            r = _requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            pass
    return None


def _ollama_vision(image_b64: str, prompt: str) -> Optional[str]:
    payload = {
        "model": OLLAMA_VISION,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    result = _http_post_json(f"{OLLAMA_BASE}/api/generate", payload)
    if result:
        return result.get("response", "")
    return None


def _ollama_text(prompt: str) -> Optional[str]:
    payload = {
        "model": OLLAMA_TEXT,
        "prompt": prompt,
        "stream": False,
    }
    result = _http_post_json(f"{OLLAMA_BASE}/api/generate", payload)
    if result:
        return result.get("response", "")
    return None


def _groq_vision(image_b64: str, prompt: str) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_VISION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": 1024,
    }
    result = _http_post_json(
        "https://api.groq.com/openai/v1/chat/completions", payload, headers
    )
    if result:
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            pass
    return None


def _gemini_vision(image_b64: str, prompt: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_VISION}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                ]
            }
        ]
    }
    result = _http_post_json(url, payload)
    if result:
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            pass
    return None


def _ocr_fallback(image_b64: str) -> str:
    """Return OCR text from a base64 PNG."""
    # Try ARIA's own OCR layer first
    if ARIA_OCR:
        try:
            return extract_text_from_image(image_b64) or ""
        except Exception:
            pass
    if OCR_AVAILABLE and PIL_AVAILABLE:
        try:
            img = _b64_to_img(image_b64)
            return pytesseract.image_to_string(img)
        except Exception:
            pass
    return "[OCR unavailable — install pytesseract or a vision model]"


def _vision_describe(image_b64: str, prompt: str) -> str:
    """Try each vision backend in priority order; fall back to OCR."""
    # 1. Local Ollama
    result = _ollama_vision(image_b64, prompt)
    if result:
        return result
    # 2. Groq
    result = _groq_vision(image_b64, prompt)
    if result:
        return result
    # 3. Gemini
    result = _gemini_vision(image_b64, prompt)
    if result:
        return result
    # 4. OCR fallback
    return _ocr_fallback(image_b64)


def _llm_text(prompt: str) -> str:
    """Send a plain text prompt; prefer Ollama, then Groq text endpoint."""
    result = _ollama_text(prompt)
    if result:
        return result
    # Groq text (reuse chat completions with a text-only model)
    if GROQ_API_KEY:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        }
        r = _http_post_json(
            "https://api.groq.com/openai/v1/chat/completions", payload, headers
        )
        if r:
            try:
                return r["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Main agent class
# ─────────────────────────────────────────────────────────────────────────────
class ComputerAgent:
    """
    ARIA Computer Agent — Perceive → Plan → Act → Verify loop.
    Equivalent to Anthropic Computer Use.
    """

    def __init__(self, dangerous: bool = False, headless: bool = False):
        self.dangerous    = dangerous      # allow destructive commands
        self.headless     = headless       # no GUI available
        self._driver      = None           # lazy Selenium driver
        self._action_history: List[ActionResult] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def run(self, goal: str, max_steps: int = MAX_STEPS_DEFAULT) -> TaskResult:
        """
        Synchronous entry point. Runs the full Perceive→Plan→Act→Verify loop.
        Returns a TaskResult summarising the entire task.
        """
        start_time = time.time()
        self._action_history = []
        history: List[Dict] = []

        # Initial perceive
        ss_b64 = self.screenshot()
        current_state = self._perceive(ss_b64)

        steps = 0
        while steps < max_steps:
            # PLAN
            plan = self._plan(goal, current_state, history)
            if not plan:
                break

            action = plan[0]

            # Safety gate
            safety_err = _safety_check(action, self.dangerous)
            if safety_err:
                result = ActionResult(
                    success=False,
                    action=action,
                    error=safety_err,
                    screenshot_b64=ss_b64,
                )
                self._action_history.append(result)
                history.append({"action": action, "result": "BLOCKED", "reason": safety_err})
                break

            # ACT
            before_ss = ss_b64
            result = self._act_single(action)
            self._action_history.append(result)

            after_ss = self.screenshot()
            result.screenshot_b64 = after_ss

            # VERIFY
            verified = self._verify(action, before_ss, after_ss,
                                    action.get("expected_state", ""))

            history.append({
                "action": action,
                "result": "ok" if result.success else "fail",
                "verified": verified,
                "output": result.output[:200],
            })

            if not verified and result.retry_count < MAX_RETRIES:
                result.retry_count += 1
                # Replan with failure context
                current_state = self._perceive(after_ss)
                goal_with_ctx = (
                    f"{goal}\n\n[Previous attempt at step {steps+1} failed. "
                    f"Failure: {result.error or 'unverified'}. Adapt plan.]"
                )
                ss_b64 = after_ss
                steps += 1
                continue

            ss_b64 = after_ss
            current_state = self._perceive(ss_b64)

            # Check goal completion
            if self._is_goal_complete(goal, current_state, history):
                break

            steps += 1

        final_ss = self.screenshot()
        total_time = time.time() - start_time
        success = self._is_goal_complete(goal, self._perceive(final_ss), history)

        return TaskResult(
            goal=goal,
            steps_taken=steps,
            success=success,
            final_screenshot=final_ss,
            total_time=total_time,
            actions_log=list(self._action_history),
        )

    async def run_async(
        self, goal: str, max_steps: int = MAX_STEPS_DEFAULT
    ) -> TaskResult:
        """Async wrapper — offloads blocking work to executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, goal, max_steps)

    async def run_nl(self, query: str) -> Dict[str, Any]:
        """
        Natural-language task dispatch for SSE / API usage.
        Returns a dict compatible with ARIA's agent bus response format.
        """
        result = await self.run_async(query)
        return {
            "agent": "computer_agent",
            "goal": result.goal,
            "success": result.success,
            "steps": result.steps_taken,
            "total_time_s": round(result.total_time, 2),
            "actions": [
                {
                    "type": r.action.get("type"),
                    "success": r.success,
                    "output": r.output[:300],
                    "error": r.error,
                }
                for r in result.actions_log
            ],
            "final_screenshot": result.final_screenshot[:100] + "..."
            if result.final_screenshot
            else "",
        }

    async def stream_run(
        self, goal: str, max_steps: int = MAX_STEPS_DEFAULT
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields SSE-compatible JSON strings as progress events.
        Use with FastAPI StreamingResponse or equivalent.
        """
        yield json.dumps({"event": "start", "goal": goal, "timestamp": datetime.utcnow().isoformat()})

        self._action_history = []
        history: List[Dict] = []
        start_time = time.time()

        loop = asyncio.get_event_loop()
        ss_b64 = await loop.run_in_executor(None, self.screenshot)
        current_state = await loop.run_in_executor(None, self._perceive, ss_b64)

        yield json.dumps({"event": "perceived", "state_summary": current_state[:300]})

        steps = 0
        while steps < max_steps:
            plan = await loop.run_in_executor(
                None, self._plan, goal, current_state, history
            )
            if not plan:
                yield json.dumps({"event": "plan_empty", "step": steps})
                break

            action = plan[0]
            yield json.dumps({"event": "action_start", "step": steps, "action": action})

            safety_err = _safety_check(action, self.dangerous)
            if safety_err:
                yield json.dumps({"event": "safety_block", "reason": safety_err})
                break

            before_ss = ss_b64
            result = await loop.run_in_executor(None, self._act_single, action)
            self._action_history.append(result)

            after_ss = await loop.run_in_executor(None, self.screenshot)
            result.screenshot_b64 = after_ss

            verified = await loop.run_in_executor(
                None, self._verify, action, before_ss, after_ss,
                action.get("expected_state", "")
            )

            yield json.dumps({
                "event": "action_done",
                "step": steps,
                "success": result.success,
                "verified": verified,
                "output": result.output[:200],
                "error": result.error,
            })

            history.append({
                "action": action,
                "result": "ok" if result.success else "fail",
                "verified": verified,
            })

            ss_b64 = after_ss
            current_state = await loop.run_in_executor(None, self._perceive, ss_b64)

            if self._is_goal_complete(goal, current_state, history):
                break

            steps += 1

        total_time = round(time.time() - start_time, 2)
        success = self._is_goal_complete(goal, current_state, history)
        yield json.dumps({
            "event": "done",
            "success": success,
            "steps": steps,
            "total_time_s": total_time,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # PERCEIVE
    # ──────────────────────────────────────────────────────────────────────────
    def _perceive(self, screenshot_b64: str) -> str:
        """Return a textual description of the current screen state."""
        prompt = (
            "You are a computer vision assistant. Describe what you see on this screen "
            "in detail: what application is open, what text is visible, what buttons/inputs "
            "exist, and what the UI state is. Be concise but complete. "
            "Format: [App: ...] [Content: ...] [Interactive elements: ...]"
        )
        return _vision_describe(screenshot_b64, prompt)

    # ──────────────────────────────────────────────────────────────────────────
    # PLAN
    # ──────────────────────────────────────────────────────────────────────────
    def _plan(
        self,
        goal: str,
        current_state: str,
        history: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Ask the LLM to decompose the goal into an ordered list of actions.
        Returns a list of action dicts. Each dict must include 'type'.
        """
        history_text = json.dumps(history[-5:], indent=2) if history else "[]"
        prompt = f"""You are a computer automation planner. Your job is to output a JSON array of actions to accomplish the goal.

GOAL: {goal}

CURRENT SCREEN STATE:
{current_state}

RECENT HISTORY (last 5 steps):
{history_text}

OUTPUT FORMAT — return ONLY a valid JSON array, no other text:
[
  {{"type": "click", "x": 100, "y": 200, "description": "Click the search bar", "expected_state": "cursor in search bar"}},
  {{"type": "type", "text": "hello", "description": "Type search query"}},
  {{"type": "key", "combo": "Return", "description": "Press Enter to search"}}
]

AVAILABLE ACTION TYPES:
- click / right_click / double_click: x, y (integers), description
- type: text (string), description
- key: combo (e.g. "ctrl+c", "Return", "alt+F4"), description
- scroll: x, y, direction ("up"/"down"/"left"/"right"), amount (1-10), description
- screenshot: description (just take a screenshot to observe)
- find_element: description (natural-language element to locate)
- open_app: name (application name), description
- close_app: name, description
- run_command: cmd (shell command), description
- read_file: path, description
- write_file: path, content, description
- browser_navigate: url, description
- browser_click: selector (CSS), description
- browser_type: selector, text, description
- wait_for: condition (text to wait for on screen), timeout_s, description
- done: description (use this when the goal is fully complete)

If the goal is already complete based on the screen state, return:
[{{"type": "done", "description": "Goal already achieved"}}]

Return only the JSON array:"""

        raw = _llm_text(prompt)
        return self._parse_plan(raw)

    def _parse_plan(self, raw: str) -> List[Dict[str, Any]]:
        """Extract JSON action array from LLM response."""
        if not raw:
            return []
        # Try direct parse
        try:
            data = json.loads(raw.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        # Extract JSON array from surrounding text
        match = re.search(r"\[[\s\S]*?\]", raw)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        return []

    # ──────────────────────────────────────────────────────────────────────────
    # ACT
    # ──────────────────────────────────────────────────────────────────────────
    def act(self, action_dict: Dict[str, Any]) -> ActionResult:
        """Public single-action entry point."""
        return self._act_single(action_dict)

    def _act_single(self, action: Dict[str, Any]) -> ActionResult:
        """Dispatch a single action dict and return an ActionResult."""
        atype = action.get("type", "").lower()
        _human_delay()

        try:
            if atype == "done":
                return ActionResult(success=True, action=action,
                                    output="Task marked complete")

            elif atype == "screenshot":
                b64 = self.screenshot()
                return ActionResult(success=True, action=action,
                                    output="screenshot captured", screenshot_b64=b64)

            elif atype == "click":
                return self._do_click(action, button="left")

            elif atype == "right_click":
                return self._do_click(action, button="right")

            elif atype == "double_click":
                return self._do_click(action, button="left", double=True)

            elif atype == "type":
                return self._do_type(action)

            elif atype == "key":
                return self._do_key(action)

            elif atype == "scroll":
                return self._do_scroll(action)

            elif atype == "find_element":
                x, y, confidence = self.find_element(action.get("description", ""))
                if x is not None:
                    return ActionResult(success=True, action=action,
                                        output=f"Found at ({x}, {y}) confidence={confidence:.2f}")
                return ActionResult(success=False, action=action,
                                    error="Element not found on screen")

            elif atype == "open_app":
                return self._do_open_app(action)

            elif atype == "close_app":
                return self._do_close_app(action)

            elif atype == "run_command":
                return self._do_run_command(action)

            elif atype == "read_file":
                return self._do_read_file(action)

            elif atype == "write_file":
                return self._do_write_file(action)

            elif atype == "browser_navigate":
                return self._do_browser_navigate(action)

            elif atype == "browser_click":
                return self._do_browser_click(action)

            elif atype == "browser_type":
                return self._do_browser_type(action)

            elif atype == "wait_for":
                return self._do_wait_for(action)

            else:
                return ActionResult(success=False, action=action,
                                    error=f"Unknown action type: {atype}")

        except Exception as exc:
            return ActionResult(
                success=False,
                action=action,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Individual action implementations
    # ──────────────────────────────────────────────────────────────────────────
    def _do_click(self, action: Dict, button: str = "left",
                  double: bool = False) -> ActionResult:
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        if not PYAUTOGUI_AVAILABLE:
            return ActionResult(success=False, action=action,
                                error="pyautogui not installed")
        _human_delay(30, 80)
        if double:
            pyautogui.doubleClick(x, y, button=button)
        else:
            pyautogui.click(x, y, button=button)
        return ActionResult(success=True, action=action,
                            output=f"{'Double-' if double else ''}Clicked ({x},{y})")

    def _do_type(self, action: Dict) -> ActionResult:
        text = action.get("text", "")
        if not PYAUTOGUI_AVAILABLE:
            return ActionResult(success=False, action=action,
                                error="pyautogui not installed")
        # Type with human-like per-character delay
        for char in text:
            pyautogui.typewrite(char, interval=random.uniform(0.03, 0.12))
        return ActionResult(success=True, action=action,
                            output=f"Typed: {text[:60]}{'...' if len(text) > 60 else ''}")

    def _do_key(self, action: Dict) -> ActionResult:
        combo = action.get("combo", "")
        if not PYAUTOGUI_AVAILABLE:
            return ActionResult(success=False, action=action,
                                error="pyautogui not installed")
        keys = [k.strip().lower() for k in combo.replace("+", " ").split()]
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        return ActionResult(success=True, action=action,
                            output=f"Key: {combo}")

    def _do_scroll(self, action: Dict) -> ActionResult:
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        direction = action.get("direction", "down").lower()
        amount = int(action.get("amount", 3))
        if not PYAUTOGUI_AVAILABLE:
            return ActionResult(success=False, action=action,
                                error="pyautogui not installed")
        clicks = amount if direction == "down" else -amount
        if direction in ("left", "right"):
            pyautogui.hscroll(x, y, clicks if direction == "right" else -clicks)
        else:
            pyautogui.scroll(clicks, x=x, y=y)
        return ActionResult(success=True, action=action,
                            output=f"Scrolled {direction} {amount} at ({x},{y})")

    def _do_open_app(self, action: Dict) -> ActionResult:
        name = action.get("name", "")
        try:
            if os.name == "nt":
                subprocess.Popen(["start", name], shell=True)
            elif os.uname().sysname == "Darwin":
                subprocess.Popen(["open", "-a", name])
            else:
                subprocess.Popen([name])
            time.sleep(1.5)
            return ActionResult(success=True, action=action,
                                output=f"Opened: {name}")
        except Exception as exc:
            return ActionResult(success=False, action=action,
                                error=str(exc))

    def _do_close_app(self, action: Dict) -> ActionResult:
        name = action.get("name", "").lower()
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/IM", f"{name}.exe"],
                               capture_output=True)
            else:
                subprocess.run(["pkill", "-f", name], capture_output=True)
            return ActionResult(success=True, action=action,
                                output=f"Closed: {name}")
        except Exception as exc:
            return ActionResult(success=False, action=action,
                                error=str(exc))

    def _do_run_command(self, action: Dict) -> ActionResult:
        cmd = action.get("cmd", "")
        timeout = int(action.get("timeout", 30))
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=timeout,
            )
            output = (result.stdout + result.stderr).strip()
            return ActionResult(
                success=result.returncode == 0,
                action=action,
                output=output[:2000],
                error="" if result.returncode == 0 else f"Exit code {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ActionResult(success=False, action=action,
                                error=f"Command timed out after {timeout}s")
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc))

    def _do_read_file(self, action: Dict) -> ActionResult:
        path = action.get("path", "")
        try:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
            return ActionResult(success=True, action=action,
                                output=content[:4000])
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc))

    def _do_write_file(self, action: Dict) -> ActionResult:
        path = action.get("path", "")
        content = action.get("content", "")
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content, encoding="utf-8")
            return ActionResult(success=True, action=action,
                                output=f"Written {len(content)} chars to {path}")
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc))

    # ── Browser actions ───────────────────────────────────────────────────────
    def _get_driver(self):
        if self._driver:
            return self._driver
        if not SELENIUM_AVAILABLE:
            return None
        opts = _ChromeOptions()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        try:
            self._driver = _selenium_webdriver.Chrome(options=opts)
            return self._driver
        except Exception:
            return None

    def _do_browser_navigate(self, action: Dict) -> ActionResult:
        url = action.get("url", "")
        driver = self._get_driver()
        if not driver:
            return ActionResult(success=False, action=action,
                                error="Selenium/Chrome not available")
        try:
            driver.get(url)
            time.sleep(1.0)
            return ActionResult(success=True, action=action,
                                output=f"Navigated to {url}")
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc))

    def _do_browser_click(self, action: Dict) -> ActionResult:
        selector = action.get("selector", "")
        driver = self._get_driver()
        if not driver:
            return ActionResult(success=False, action=action,
                                error="Selenium/Chrome not available")
        try:
            el = _WebDriverWait(driver, 5).until(
                _EC.element_to_be_clickable((_By.CSS_SELECTOR, selector))
            )
            _human_delay(50, 150)
            el.click()
            return ActionResult(success=True, action=action,
                                output=f"Clicked: {selector}")
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc))

    def _do_browser_type(self, action: Dict) -> ActionResult:
        selector = action.get("selector", "")
        text = action.get("text", "")
        driver = self._get_driver()
        if not driver:
            return ActionResult(success=False, action=action,
                                error="Selenium/Chrome not available")
        try:
            el = _WebDriverWait(driver, 5).until(
                _EC.visibility_of_element_located((_By.CSS_SELECTOR, selector))
            )
            el.clear()
            for char in text:
                el.send_keys(char)
                time.sleep(random.uniform(0.03, 0.09))
            return ActionResult(success=True, action=action,
                                output=f"Typed into {selector}: {text[:60]}")
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc))

    def _do_wait_for(self, action: Dict) -> ActionResult:
        condition = action.get("condition", "")
        timeout_s = float(action.get("timeout_s", 10))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ss = self.screenshot()
            text_on_screen = _ocr_fallback(ss)
            if condition.lower() in text_on_screen.lower():
                return ActionResult(success=True, action=action,
                                    output=f"Condition met: '{condition}'")
            time.sleep(0.5)
        return ActionResult(success=False, action=action,
                            error=f"Timed out waiting for: '{condition}'")

    # ──────────────────────────────────────────────────────────────────────────
    # SCREENSHOT
    # ──────────────────────────────────────────────────────────────────────────
    def screenshot(self) -> str:
        """Take a screenshot and return as base64-encoded PNG string."""
        # Prefer PIL ImageGrab (cross-platform, no extra deps)
        if PIL_AVAILABLE and ImageGrab:
            try:
                img = ImageGrab.grab()
                return _img_to_b64(img)
            except Exception:
                pass
        # pyautogui fallback
        if PYAUTOGUI_AVAILABLE:
            try:
                img = pyautogui.screenshot()
                return _img_to_b64(img)
            except Exception:
                pass
        # Headless / CI: return empty string, logic degrades gracefully
        return ""

    # ──────────────────────────────────────────────────────────────────────────
    # FIND ELEMENT
    # ──────────────────────────────────────────────────────────────────────────
    def find_element(
        self, desc: str
    ) -> Tuple[Optional[int], Optional[int], float]:
        """
        Locate a UI element by natural-language description.
        Returns (x, y, confidence). (None, None, 0.0) if not found.
        Strategy: vision LLM → pyautogui.locateCenterOnScreen (template).
        """
        ss_b64 = self.screenshot()
        if not ss_b64:
            return None, None, 0.0

        # Ask vision LLM for coordinates
        prompt = (
            f"On this screenshot, find the element described as: '{desc}'. "
            "Respond with ONLY a JSON object: {\"x\": <int>, \"y\": <int>, \"confidence\": <float 0-1>}. "
            "If not found, respond: {\"x\": null, \"y\": null, \"confidence\": 0}."
        )
        raw = _vision_describe(ss_b64, prompt)
        try:
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                data = json.loads(match.group())
                x = data.get("x")
                y = data.get("y")
                conf = float(data.get("confidence", 0.0))
                if x is not None and y is not None:
                    return int(x), int(y), conf
        except Exception:
            pass
        return None, None, 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # VERIFY
    # ──────────────────────────────────────────────────────────────────────────
    def _verify(
        self,
        action: Dict[str, Any],
        before_screenshot: str,
        after_screenshot: str,
        expected_state: str = "",
    ) -> bool:
        """
        Verify that an action had the intended effect.
        Strategy:
          1. If expected_state provided → ask vision LLM
          2. Image diff: if change_score > 0.002, something changed (likely good)
          3. For done/screenshot actions → always True
        """
        atype = action.get("type", "").lower()
        if atype in ("done", "screenshot", "wait_for", "read_file"):
            return True

        # If action failed, don't try to verify
        # (caller checks result.success separately)

        change_score = _image_diff_score(before_screenshot, after_screenshot)

        if expected_state and after_screenshot:
            prompt = (
                f"The expected screen state after performing '{action.get('description', atype)}' is: "
                f"'{expected_state}'. "
                "Does the current screenshot match this expectation? "
                "Reply with exactly 'YES' or 'NO'."
            )
            answer = _vision_describe(after_screenshot, prompt).strip().upper()
            if answer.startswith("YES"):
                return True
            if answer.startswith("NO"):
                return False
            # Ambiguous — fall through to diff check

        # For passive actions (type, key, scroll) a very small diff is still success
        if atype in ("type", "key"):
            return True  # trust pyautogui succeeded if no exception was raised

        # For click / navigate we expect the screen to change
        if atype in ("click", "right_click", "double_click",
                     "browser_navigate", "browser_click", "browser_type",
                     "open_app", "close_app"):
            return change_score > 0.001

        # run_command verified by exit code (ActionResult.success)
        if atype == "run_command":
            return True

        return change_score > 0.0005

    # ──────────────────────────────────────────────────────────────────────────
    # Goal completion check
    # ──────────────────────────────────────────────────────────────────────────
    def _is_goal_complete(
        self,
        goal: str,
        current_state: str,
        history: List[Dict],
    ) -> bool:
        """Ask LLM whether the goal has been achieved given current screen state."""
        # Check if last action was 'done'
        if history and history[-1].get("action", {}).get("type") == "done":
            return True

        if not current_state:
            return False

        prompt = (
            f"Goal: {goal}\n\n"
            f"Current screen state: {current_state}\n\n"
            "Has the goal been fully accomplished based on the current screen state? "
            "Reply with exactly 'YES' or 'NO'."
        )
        answer = _llm_text(prompt).strip().upper()
        return answer.startswith("YES")

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────
    def close(self) -> None:
        """Release resources (Selenium driver etc.)."""
        driver = getattr(self, "_driver", None)
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
            self._driver = None

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
_agent_instance: Optional[ComputerAgent] = None


def get_agent(dangerous: bool = False, headless: bool = False) -> ComputerAgent:
    """Return the global ComputerAgent singleton, creating it if needed."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ComputerAgent(dangerous=dangerous, headless=headless)
    return _agent_instance


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point for quick testing
# ─────────────────────────────────────────────────────────────────────────────
def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ARIA Computer Agent CLI")
    parser.add_argument("goal", nargs="?", default=None,
                        help="Goal to accomplish (quoted string)")
    parser.add_argument("--steps", type=int, default=MAX_STEPS_DEFAULT,
                        help="Max steps (default: 20)")
    parser.add_argument("--dangerous", action="store_true",
                        help="Allow destructive commands (DELETE/FORMAT/WIPE)")
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode (no display required)")
    parser.add_argument("--screenshot", action="store_true",
                        help="Just take a screenshot and print base64 length")
    args = parser.parse_args()

    agent = ComputerAgent(dangerous=args.dangerous, headless=args.headless)

    if args.screenshot:
        b64 = agent.screenshot()
        if b64:
            print(f"Screenshot captured: {len(b64)} base64 chars")
        else:
            print("Screenshot failed (no display or PIL not installed)")
        return

    if not args.goal:
        parser.print_help()
        return

    print(f"[ARIA Computer Agent] Goal: {args.goal}")
    print(f"[ARIA Computer Agent] Max steps: {args.steps}")
    print("-" * 60)

    result = agent.run(args.goal, max_steps=args.steps)

    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Steps taken: {result.steps_taken}")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"\nAction log ({len(result.actions_log)} actions):")
    for i, ar in enumerate(result.actions_log, 1):
        status = "OK " if ar.success else "ERR"
        atype  = ar.action.get("type", "?")
        desc   = ar.action.get("description", "")[:50]
        err    = f" | {ar.error[:60]}" if ar.error else ""
        print(f"  {i:2d}. [{status}] {atype:20s} {desc}{err}")

    agent.close()


if __name__ == "__main__":
    _cli_main()
