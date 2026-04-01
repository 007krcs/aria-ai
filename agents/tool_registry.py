"""
ARIA — Universal Tool Registry
================================
The single most important file in ARIA's self-sufficiency.

Every capability ARIA has is registered as a tool here.
Tools are pure functions. No model needed to call them.
The model only decides WHICH tool — and only when patterns fail.

Design principle:
  1. Pattern match intent → tool name          (0ms, no model)
  2. Tool registry → get tool function         (0ms)
  3. Arg extractor → fill required args        (0ms, regex)
  4. Execute tool → result                     (actual action)
  5. Log result → improve pattern              (learning)

Model is only called at step 3 if arg extraction fails.
That means 80%+ of commands never touch the model at all.

IoT compatibility:
  Every tool has an `iot_safe` flag.
  On resource-constrained devices (Raspberry Pi, ESP32),
  only iot_safe tools are loaded. The full registry is not imported.
  IoT agent talks to ARIA server for any non-safe tool.
"""

import re
import os
import sys
import json
import time
import platform
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()
PLATFORM     = platform.system()   # Windows / Darwin / Linux

TOOLS_USAGE_FILE = PROJECT_ROOT / "data" / "tools_usage.json"
TOOLS_USAGE_FILE.parent.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    """
    A registered capability ARIA can execute.

    name:         unique identifier, snake_case
    description:  what this tool does (used for matching + docs)
    triggers:     list of phrases/patterns that activate this tool
    function:     the actual Python callable
    args:         required argument names
    arg_patterns: regex to extract each arg from natural language
    returns:      what the function returns (for chaining)
    iot_safe:     can run on Raspberry Pi / ARM with no model
    os_support:   list of OS names this works on
    needs_network: requires internet connection
    needs_device:  requires physical device (phone, camera, etc.)
    category:     grouping for UI
    """
    name:          str
    description:   str
    triggers:      list[str]
    function:      Callable
    args:          list[str]            = field(default_factory=list)
    arg_patterns:  dict[str, str]       = field(default_factory=dict)
    returns:       str                  = "dict"
    iot_safe:      bool                 = True
    os_support:    list[str]            = field(default_factory=lambda: ["Windows","Darwin","Linux"])
    needs_network: bool                 = False
    needs_device:  bool                 = False
    category:      str                  = "general"
    usage_count:   int                  = 0
    success_count: int                  = 0
    avg_latency_ms: float               = 0.0

    def matches(self, text: str) -> float:
        """
        Score how well this tool matches the input text.
        Returns 0.0-1.0. Used for routing without the model.
        """
        text_lower = text.lower()
        score      = 0.0
        for trigger in self.triggers:
            if trigger.lower() in text_lower:
                # Longer trigger = more specific = higher score
                score = max(score, len(trigger) / max(len(text_lower), 1))
        return min(score * 3, 1.0)

    def extract_args(self, text: str) -> dict:
        """Extract required arguments from natural language using regex."""
        extracted = {}
        for arg_name, pattern in self.arg_patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                extracted[arg_name] = m.group(1).strip()
        return extracted

    def record_use(self, success: bool, latency_ms: float):
        self.usage_count   += 1
        if success:
            self.success_count += 1
        self.avg_latency_ms = (
            0.8 * self.avg_latency_ms + 0.2 * latency_ms
        ) if self.avg_latency_ms else latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry of all ARIA capabilities.
    Zero model needed for lookup or execution.

    Usage:
        registry = ToolRegistry()
        tool, args = registry.route("open youtube on chrome")
        result = tool.function(**args)
    """

    def __init__(self, iot_mode: bool = False):
        self.iot_mode = iot_mode
        self._tools:  dict[str, Tool] = {}
        self._lock    = threading.Lock()
        self._load_usage_stats()

    def register(self, tool: Tool):
        """Register a tool. IoT mode skips non-iot_safe tools."""
        if self.iot_mode and not tool.iot_safe:
            return
        if PLATFORM not in tool.os_support and "All" not in tool.os_support:
            return
        with self._lock:
            self._tools[tool.name] = tool

    def route(self, text: str, top_k: int = 3) -> list[tuple[Tool, dict, float]]:
        """
        Route natural language to the best matching tools.
        Returns [(tool, extracted_args, confidence)] sorted by confidence.
        No model needed.
        """
        text    = text.strip()
        results = []

        with self._lock:
            for tool in self._tools.values():
                score = tool.matches(text)
                if score > 0.05:
                    args  = tool.extract_args(text)
                    results.append((tool, args, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def execute(
        self,
        tool_name: str,
        args:      dict,
        timeout:   float = 30.0,
    ) -> dict:
        """
        Execute a tool by name with given args.
        Handles timeout, error capture, and usage logging.
        """
        tool = self.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool not found: {tool_name}"}

        t0      = time.time()
        result  = {}
        success = False

        try:
            # Run with timeout in a thread
            result_holder = [None]
            error_holder  = [None]

            def run():
                try:
                    result_holder[0] = tool.function(**args)
                except Exception as e:
                    error_holder[0] = str(e)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                result  = {"success": False, "error": f"Tool timed out after {timeout}s"}
            elif error_holder[0]:
                result  = {"success": False, "error": error_holder[0]}
            else:
                result  = result_holder[0] or {}
                success = result.get("success", True)

        except Exception as e:
            result = {"success": False, "error": str(e)}

        ms = int((time.time() - t0) * 1000)
        tool.record_use(success, ms)
        self._save_usage_stats()

        return {**result, "_tool": tool_name, "_ms": ms, "_success": success}

    def smart_execute(
        self,
        text:    str,
        engine=  None,
        timeout: float = 30.0,
    ) -> dict:
        """
        Full pipeline: route text → execute best tool.
        Uses model ONLY if arg extraction fails.
        """
        candidates = self.route(text)
        if not candidates:
            return {"success": False, "error": "No matching tool", "text": text}

        tool, args, confidence = candidates[0]
        console.print(
            f"  [dim]Tool: {tool.name} (conf={confidence:.2f})[/]"
        )

        # Check if all required args are present
        missing = [a for a in tool.args if a not in args]

        if missing and engine:
            # Use model only to extract missing args
            json_template = ", ".join('"' + m + '": "..."' for m in missing)
            prompt = (
                f"Extract these values from the text:\n"
                f"Text: {text}\n"
                f"Values needed: {missing}\n"
                f"Return JSON only: {{{json_template}}}"
            )
            try:
                raw     = engine.generate(prompt, temperature=0.0)
                raw     = re.sub(r"```\w*\n?|```", "", raw).strip()
                llm_args = json.loads(raw)
                args.update(llm_args)
            except Exception:
                pass

        return self.execute(tool.name, args, timeout)

    def list_tools(self, category: str = None) -> list[dict]:
        with self._lock:
            tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return [{
            "name":        t.name,
            "description": t.description,
            "category":    t.category,
            "iot_safe":    t.iot_safe,
            "usage_count": t.usage_count,
            "success_rate": round(t.success_count / t.usage_count * 100, 1)
                           if t.usage_count else 0,
            "avg_latency_ms": round(t.avg_latency_ms, 0),
        } for t in tools]

    def stats(self) -> dict:
        with self._lock:
            tools = list(self._tools.values())
        total     = sum(t.usage_count for t in tools)
        successes = sum(t.success_count for t in tools)
        return {
            "total_tools":    len(tools),
            "total_executions": total,
            "success_rate":   round(successes/total*100, 1) if total else 0,
            "top_tools":      sorted(
                [{"name": t.name, "uses": t.usage_count} for t in tools],
                key=lambda x: x["uses"], reverse=True
            )[:5],
            "categories":     list(set(t.category for t in tools)),
            "iot_tools":      sum(1 for t in tools if t.iot_safe),
        }

    def _load_usage_stats(self):
        try:
            if TOOLS_USAGE_FILE.exists():
                data = json.loads(TOOLS_USAGE_FILE.read_text())
                for name, stats in data.items():
                    if name in self._tools:
                        self._tools[name].usage_count   = stats.get("uses", 0)
                        self._tools[name].success_count = stats.get("successes", 0)
        except Exception:
            pass

    def _save_usage_stats(self):
        try:
            data = {
                name: {
                    "uses":      t.usage_count,
                    "successes": t.success_count,
                    "avg_ms":    round(t.avg_latency_ms, 1),
                }
                for name, t in self._tools.items()
                if t.usage_count > 0
            }
            TOOLS_USAGE_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN TOOLS
# These work on every OS with no model, no internet, no external service
# ─────────────────────────────────────────────────────────────────────────────

def _open_url(url: str) -> dict:
    import webbrowser
    webbrowser.open(url)
    return {"success": True, "url": url}


def _open_app_windows(app: str) -> dict:
    APP_MAP = {
        "chrome":         r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "notepad":        "notepad.exe",
        "calculator":     "calc.exe",
        "explorer":       "explorer.exe",
        "terminal":       "wt.exe",
        "cmd":            "cmd.exe",
        "powershell":     "powershell.exe",
        "paint":          "mspaint.exe",
        "word":           "WINWORD.EXE",
        "excel":          "EXCEL.EXE",
        "outlook":        "OUTLOOK.EXE",
        "teams":          "Teams.exe",
        "spotify":        "Spotify.exe",
        "vscode":         "Code.exe",
        "vs code":        "Code.exe",
    }
    cmd = APP_MAP.get(app.lower().strip(), app)
    try:
        subprocess.Popen(cmd, shell=True)
        return {"success": True, "app": app, "cmd": cmd}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _open_app_mac(app: str) -> dict:
    result = subprocess.run(["open", "-a", app], capture_output=True, text=True, timeout=10)
    return {"success": result.returncode == 0, "app": app, "error": result.stderr}


def _open_app_linux(app: str) -> dict:
    try:
        subprocess.Popen([app], shell=True)
        return {"success": True, "app": app}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _open_app(app: str) -> dict:
    if PLATFORM == "Windows": return _open_app_windows(app)
    if PLATFORM == "Darwin":  return _open_app_mac(app)
    return _open_app_linux(app)


def _close_app(app: str) -> dict:
    if PLATFORM == "Windows":
        result = subprocess.run(
            ["taskkill", "/F", "/IM", f"{app}.exe"],
            capture_output=True, text=True
        )
        return {"success": result.returncode == 0, "app": app}
    elif PLATFORM == "Darwin":
        result = subprocess.run(["pkill", "-x", app], capture_output=True)
        return {"success": result.returncode == 0}
    else:
        result = subprocess.run(["pkill", "-f", app], capture_output=True)
        return {"success": result.returncode == 0}


def _search_google(query: str) -> dict:
    import urllib.parse
    url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    return _open_url(url)


def _search_youtube(query: str) -> dict:
    import urllib.parse
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    return _open_url(url)


def _play_youtube(query: str) -> dict:
    """Search YouTube and open first result."""
    import urllib.parse, requests
    try:
        r = requests.get(
            f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        # Extract first video ID
        m = re.search(r'"videoId":"([^"]{11})"', r.text)
        if m:
            url = f"https://www.youtube.com/watch?v={m.group(1)}"
            return _open_url(url)
    except Exception:
        pass
    return _search_youtube(query)


def _send_email(to: str, subject: str = "", body: str = "") -> dict:
    import urllib.parse
    url = f"mailto:{to}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
    return _open_url(url)


def _set_timer(minutes: float) -> dict:
    """Set a countdown timer that fires a desktop notification."""
    def fire():
        time.sleep(minutes * 60)
        _notify("ARIA Timer", f"{minutes:.0f} minute timer complete")
    threading.Thread(target=fire, daemon=True).start()
    return {"success": True, "minutes": minutes, "fires_at": datetime.now().strftime("%H:%M")}


def _notify(title: str, message: str) -> dict:
    try:
        from plyer import notification
        notification.notify(title=f"ARIA — {title}", message=message[:200], timeout=6)
        return {"success": True}
    except ImportError:
        if PLATFORM == "Darwin":
            subprocess.run(["osascript","-e",f'display notification "{message}" with title "ARIA — {title}"'])
        elif PLATFORM == "Windows":
            subprocess.run([
                "powershell","-command",
                f'New-BurntToastNotification -Text "ARIA — {title}","{message}"'
            ], capture_output=True)
        else:
            subprocess.run(["notify-send", f"ARIA — {title}", message], capture_output=True)
        return {"success": True}


def _get_time() -> dict:
    now = datetime.now()
    return {
        "success": True,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%A, %d %B %Y"),
        "timestamp": now.isoformat(),
    }


def _get_weather(city: str) -> dict:
    try:
        import requests
        r = requests.get(
            f"https://wttr.in/{city}?format=j1",
            headers={"User-Agent": "ARIA/1.0"},
            timeout=8,
        )
        d    = r.json()["current_condition"][0]
        temp = d.get("temp_C","?")
        desc = d.get("weatherDesc",[{}])[0].get("value","?")
        return {
            "success":    True,
            "city":       city,
            "temp_c":     temp,
            "description": desc,
            "humidity":   d.get("humidity","?"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_stock_price(symbol: str) -> dict:
    try:
        import requests
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        d = r.json()
        meta  = d["chart"]["result"][0]["meta"]
        price = meta["regularMarketPrice"]
        prev  = meta.get("previousClose", price)
        change= round(price - prev, 2)
        return {
            "success": True,
            "symbol":  symbol.upper(),
            "price":   round(price, 2),
            "change":  change,
            "currency": meta.get("currency","INR"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _take_screenshot() -> dict:
    try:
        import pyautogui
        path = str(PROJECT_ROOT / "data" / "screenshots" /
                   f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pyautogui.screenshot(path)
        return {"success": True, "path": path}
    except ImportError:
        return {"success": False, "error": "pip install pyautogui"}


def _type_text(text: str) -> dict:
    try:
        import pyautogui
        time.sleep(0.5)
        pyautogui.write(text, interval=0.02)
        return {"success": True, "typed": text}
    except ImportError:
        return {"success": False, "error": "pip install pyautogui"}


def _read_clipboard() -> dict:
    try:
        import pyperclip
        text = pyperclip.paste()
        return {"success": True, "text": text, "length": len(text)}
    except ImportError:
        return {"success": False, "error": "pip install pyperclip"}


def _write_clipboard(text: str) -> dict:
    try:
        import pyperclip
        pyperclip.copy(text)
        return {"success": True, "copied": text[:50]}
    except ImportError:
        return {"success": False, "error": "pip install pyperclip"}


def _run_terminal_command(command: str) -> dict:
    """Run a shell command safely. Blocked dangerous commands."""
    BLOCKED = ["rm -rf", "del /f", "format", "shutdown", "reboot",
               "dd if=", "mkfs", ":(){:|:&};:"]
    for bad in BLOCKED:
        if bad in command.lower():
            return {"success": False, "error": f"Blocked dangerous command: {bad}"}
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return {
            "success":   result.returncode == 0,
            "output":    result.stdout[:2000],
            "error":     result.stderr[:500],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _search_files(query: str, directory: str = "~") -> dict:
    """Find files matching a pattern."""
    base = Path(directory).expanduser()
    found = list(base.rglob(f"*{query}*"))[:20]
    return {
        "success": True,
        "query":   query,
        "results": [str(f) for f in found],
        "count":   len(found),
    }


def _read_file(path: str) -> dict:
    """Read a text file."""
    try:
        content = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
        return {"success": True, "path": path, "content": content[:5000],
                "size": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _write_file(path: str, content: str) -> dict:
    """Write content to a file."""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(p), "bytes": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_system_info() -> dict:
    try:
        import psutil
        return {
            "success":    True,
            "platform":   PLATFORM,
            "cpu_pct":    psutil.cpu_percent(interval=0.1),
            "ram_pct":    psutil.virtual_memory().percent,
            "ram_free_gb": round(psutil.virtual_memory().available / 1e9, 2),
            "disk_pct":   psutil.disk_usage("/").percent,
            "uptime_h":   round((time.time() - psutil.boot_time()) / 3600, 1),
        }
    except ImportError:
        return {"success": True, "platform": PLATFORM}


def _list_running_apps() -> dict:
    try:
        import psutil
        apps = []
        for p in psutil.process_iter(["pid","name","cpu_percent","memory_percent"]):
            try:
                info = p.info
                if info["cpu_percent"] > 0 or info["memory_percent"] > 0.1:
                    apps.append({
                        "pid":    info["pid"],
                        "name":   info["name"],
                        "cpu":    round(info["cpu_percent"], 1),
                        "mem":    round(info["memory_percent"], 1),
                    })
            except Exception:
                pass
        apps.sort(key=lambda x: x["cpu"], reverse=True)
        return {"success": True, "apps": apps[:20]}
    except ImportError:
        return {"success": False, "error": "pip install psutil"}


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY: build the default registry with all built-in tools
# ─────────────────────────────────────────────────────────────────────────────

def build_default_registry(iot_mode: bool = False) -> ToolRegistry:
    """Build and return the fully populated tool registry."""
    reg = ToolRegistry(iot_mode=iot_mode)

    ALL_OS = ["Windows","Darwin","Linux"]

    tools = [

        # ── App control ──────────────────────────────────────────────────────
        Tool(
            name="open_app",
            description="Open any application on the device",
            triggers=["open ","launch ","start ","run "],
            function=_open_app,
            args=["app"],
            arg_patterns={"app": r"(?:open|launch|start|run)\s+(.+?)(?:\s+app|\s+application|$)"},
            iot_safe=True, os_support=ALL_OS, category="apps",
        ),
        Tool(
            name="close_app",
            description="Close a running application",
            triggers=["close ","quit ","kill ","exit "],
            function=_close_app,
            args=["app"],
            arg_patterns={"app": r"(?:close|quit|kill|exit)\s+(.+?)(?:\s+app|\s+application|$)"},
            iot_safe=True, os_support=ALL_OS, category="apps",
        ),

        # ── Browser / web ────────────────────────────────────────────────────
        Tool(
            name="open_url",
            description="Open a URL in the browser",
            triggers=["open https://","open http://","go to ","visit ","browse to "],
            function=_open_url,
            args=["url"],
            arg_patterns={"url": r"(https?://\S+)"},
            iot_safe=False, os_support=ALL_OS, category="browser",
            needs_network=True,
        ),
        Tool(
            name="search_google",
            description="Search Google for anything",
            triggers=["search ","google ","look up ","find ","search on google"],
            function=_search_google,
            args=["query"],
            arg_patterns={"query": r"(?:search|google|look up|find)\s+(?:for\s+)?(.+)"},
            iot_safe=False, os_support=ALL_OS, category="browser",
            needs_network=True,
        ),
        Tool(
            name="play_youtube",
            description="Play a video on YouTube",
            triggers=["play on youtube","watch on youtube","youtube ","open video","play video"],
            function=_play_youtube,
            args=["query"],
            arg_patterns={"query": r"(?:play|watch|open|search)\s+(.+?)(?:\s+on youtube|\s+video|$)"},
            iot_safe=False, os_support=ALL_OS, category="media",
            needs_network=True,
        ),
        Tool(
            name="search_youtube",
            description="Search YouTube",
            triggers=["search youtube","youtube search"],
            function=_search_youtube,
            args=["query"],
            arg_patterns={"query": r"(?:search youtube|youtube search)\s+(?:for\s+)?(.+)"},
            iot_safe=False, os_support=ALL_OS, category="media",
            needs_network=True,
        ),

        # ── Communication ────────────────────────────────────────────────────
        Tool(
            name="send_email",
            description="Compose an email",
            triggers=["send email","send mail","email to","write email","compose email"],
            function=_send_email,
            args=["to"],
            arg_patterns={
                "to":      r"(?:send email|email|mail)\s+(?:to\s+)?(.+?)(?:\s+subject|\s+about|\s+saying|$)",
                "subject": r"subject[:]\s*(.+?)(?:\s+body|\s+saying|$)",
                "body":    r"(?:body|saying|that|message)\s*[:]\s*(.+)",
            },
            iot_safe=False, os_support=ALL_OS, category="communication",
        ),

        # ── Information ──────────────────────────────────────────────────────
        Tool(
            name="get_time",
            description="Get the current time and date",
            triggers=["what time","current time","what date","what day","today's date"],
            function=_get_time,
            args=[],
            iot_safe=True, os_support=ALL_OS, category="info",
        ),
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            triggers=["weather in ","weather at ","what's the weather","temperature in"],
            function=_get_weather,
            args=["city"],
            arg_patterns={"city": r"weather\s+(?:in|at|for)\s+(.+?)(?:\s+today|\s+now|$)"},
            iot_safe=False, os_support=ALL_OS, category="info",
            needs_network=True,
        ),
        Tool(
            name="get_stock_price",
            description="Get real-time stock price",
            triggers=["stock price","share price","price of ","how much is ","stock of"],
            function=_get_stock_price,
            args=["symbol"],
            arg_patterns={"symbol": r"(?:stock|share|price of|price for)\s+(?:of\s+)?([A-Z]{1,5}|[A-Za-z]+)"},
            iot_safe=False, os_support=ALL_OS, category="finance",
            needs_network=True,
        ),

        # ── System ───────────────────────────────────────────────────────────
        Tool(
            name="take_screenshot",
            description="Take a screenshot of the screen",
            triggers=["screenshot","take screenshot","capture screen","screen capture"],
            function=_take_screenshot,
            args=[],
            iot_safe=False, os_support=ALL_OS, category="system",
        ),
        Tool(
            name="type_text",
            description="Type text at the current cursor position",
            triggers=["type ","write ","input "],
            function=_type_text,
            args=["text"],
            arg_patterns={"text": r"(?:type|write|input)\s+(?:text\s+)?[\"']?(.+?)[\"']?$"},
            iot_safe=False, os_support=ALL_OS, category="system",
        ),
        Tool(
            name="get_system_info",
            description="Get system CPU, RAM, disk usage",
            triggers=["system info","cpu usage","memory usage","ram usage","disk usage","system status"],
            function=_get_system_info,
            args=[],
            iot_safe=True, os_support=ALL_OS, category="system",
        ),
        Tool(
            name="list_running_apps",
            description="List all running applications",
            triggers=["running apps","running applications","what is running","list processes","active apps"],
            function=_list_running_apps,
            args=[],
            iot_safe=False, os_support=ALL_OS, category="system",
        ),
        Tool(
            name="read_clipboard",
            description="Read text from clipboard",
            triggers=["read clipboard","what's in clipboard","clipboard content","paste content"],
            function=_read_clipboard,
            args=[],
            iot_safe=False, os_support=ALL_OS, category="system",
        ),
        Tool(
            name="write_clipboard",
            description="Copy text to clipboard",
            triggers=["copy to clipboard","write to clipboard"],
            function=_write_clipboard,
            args=["text"],
            arg_patterns={"text": r"copy\s+[\"']?(.+?)[\"']?\s+to clipboard"},
            iot_safe=False, os_support=ALL_OS, category="system",
        ),
        Tool(
            name="run_command",
            description="Run a terminal/shell command",
            triggers=["run command","execute command","run in terminal","shell command"],
            function=_run_terminal_command,
            args=["command"],
            arg_patterns={"command": r"(?:run|execute)\s+(?:command\s+)?[\"']?(.+?)[\"']?$"},
            iot_safe=True, os_support=ALL_OS, category="system",
        ),

        # ── Files ────────────────────────────────────────────────────────────
        Tool(
            name="search_files",
            description="Find files on the computer",
            triggers=["find file","search file","locate file","where is "],
            function=_search_files,
            args=["query"],
            arg_patterns={
                "query": r"(?:find|search|locate)\s+(?:file\s+)?[\"']?(.+?)[\"']?(?:\s+in|\s+under|$)",
                "directory": r"(?:in|under)\s+(.+?)$",
            },
            iot_safe=True, os_support=ALL_OS, category="files",
        ),
        Tool(
            name="read_file",
            description="Read the contents of a file",
            triggers=["read file","open file","show file","show contents of"],
            function=_read_file,
            args=["path"],
            arg_patterns={"path": r"(?:read|open|show)\s+(?:file\s+)?[\"']?(.+?)[\"']?$"},
            iot_safe=True, os_support=ALL_OS, category="files",
        ),

        # ── Notifications ────────────────────────────────────────────────────
        Tool(
            name="notify",
            description="Send a desktop notification",
            triggers=["notify me","send notification","show notification","alert me"],
            function=_notify,
            args=["title","message"],
            arg_patterns={
                "title":   r"(?:title|about)\s*[:]\s*(.+?)(?:\s+message|$)",
                "message": r"(?:message|saying|that)\s*[:]\s*(.+)",
            },
            iot_safe=True, os_support=ALL_OS, category="system",
        ),
        Tool(
            name="set_timer",
            description="Set a countdown timer",
            triggers=["set timer","timer for","start timer","countdown"],
            function=_set_timer,
            args=["minutes"],
            arg_patterns={"minutes": r"(?:timer|timer for|countdown)\s+(\d+(?:\.\d+)?)\s*(?:minute|min|m)?"},
            iot_safe=True, os_support=ALL_OS, category="productivity",
        ),
    ]

    for tool in tools:
        reg.register(tool)

    console.print(f"  [green]Tool registry:[/] {len(reg._tools)} tools loaded"
                  + (" (IoT mode)" if iot_mode else ""))
    return reg
