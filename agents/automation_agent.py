"""
ARIA Automation Agent
Multi-step workflow automation — macros, workflows, scheduling, NL execution.
Dependencies: pyautogui, pynput, pywin32, requests, psutil (optional)
Storage:
  Workflows -> C:\\Users\\chand\\ai-remo\\data\\workflows\\workflows.json
  Macros    -> C:\\Users\\chand\\ai-remo\\data\\workflows\\macros.json
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional GUI automation imports
# ---------------------------------------------------------------------------
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    _PYAUTOGUI = True
except ImportError:
    _PYAUTOGUI = False

try:
    import pynput.mouse as _pmouse
    import pynput.keyboard as _pkbd
    _PYNPUT = True
except ImportError:
    _PYNPUT = False

try:
    import win32gui
    import win32con
    import win32process
    import win32api
    _WIN32 = True
except ImportError:
    _WIN32 = False

try:
    import ctypes
    _CTYPES = True
except ImportError:
    _CTYPES = False

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(r"C:\Users\chand\ai-remo\data\workflows")
WORKFLOWS_FILE = DATA_DIR / "workflows.json"
MACROS_FILE    = DATA_DIR / "macros.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _ok(result: str, data: Any = None) -> Dict:
    return {"ok": True, "result": result, "data": data}


def _err(result: str, data: Any = None) -> Dict:
    return {"ok": False, "result": result, "data": data}


# ---------------------------------------------------------------------------
# Macro recording / replay
# ---------------------------------------------------------------------------

_recording: Dict = {"active": False, "events": [], "thread": None}


def record_macro(name: str, duration: int = 30) -> Dict:
    """
    Record mouse and keyboard events for `duration` seconds, then save.
    Uses pynput if available; falls back to a placeholder on missing dep.
    """
    try:
        if not name:
            return _err("Macro name cannot be empty")
        if not _PYNPUT:
            return _err(
                "pynput not installed. Install with: pip install pynput. "
                "Cannot record macros without pynput."
            )

        events: List[Dict] = []
        start_time = time.time()
        stop_event = threading.Event()

        def _on_move(x, y):
            events.append({"t": round(time.time() - start_time, 3), "type": "move", "x": x, "y": y})

        def _on_click(x, y, button, pressed):
            events.append({
                "t": round(time.time() - start_time, 3),
                "type": "click",
                "x": x, "y": y,
                "button": str(button),
                "pressed": pressed,
            })

        def _on_scroll(x, y, dx, dy):
            events.append({"t": round(time.time() - start_time, 3), "type": "scroll", "x": x, "y": y, "dx": dx, "dy": dy})

        def _on_key_press(key):
            try:
                char = key.char
            except AttributeError:
                char = str(key)
            events.append({"t": round(time.time() - start_time, 3), "type": "key_press", "key": char})

        def _on_key_release(key):
            try:
                char = key.char
            except AttributeError:
                char = str(key)
            events.append({"t": round(time.time() - start_time, 3), "type": "key_release", "key": char})
            if key == _pkbd.Key.esc:
                stop_event.set()
                return False

        mouse_listener = _pmouse.Listener(
            on_move=_on_move, on_click=_on_click, on_scroll=_on_scroll
        )
        kbd_listener = _pkbd.Listener(
            on_press=_on_key_press, on_release=_on_key_release
        )

        mouse_listener.start()
        kbd_listener.start()

        # Wait up to `duration` seconds or until Esc is pressed
        stop_event.wait(timeout=duration)
        mouse_listener.stop()
        kbd_listener.stop()

        macros = _load_json(MACROS_FILE)
        macros[name] = {
            "name": name,
            "events": events,
            "duration": round(time.time() - start_time, 2),
            "recorded_at": datetime.now().isoformat(),
        }
        _save_json(MACROS_FILE, macros)
        return _ok(f"Macro '{name}' recorded ({len(events)} events, {macros[name]['duration']}s)")
    except Exception as e:
        return _err(f"record_macro error: {e}\n{traceback.format_exc()}")


def play_macro(name: str) -> Dict:
    """Replay a recorded macro."""
    try:
        macros = _load_json(MACROS_FILE)
        if name not in macros:
            return _err(f"Macro '{name}' not found")
        if not _PYNPUT:
            return _err("pynput not installed; cannot replay macros")

        events = macros[name]["events"]
        if not events:
            return _err(f"Macro '{name}' has no events")

        mouse_ctrl = _pmouse.Controller()
        kbd_ctrl = _pkbd.Controller()

        prev_t = 0.0
        for ev in events:
            delay = ev["t"] - prev_t
            if delay > 0:
                time.sleep(delay)
            prev_t = ev["t"]

            etype = ev.get("type")
            if etype == "move":
                mouse_ctrl.position = (ev["x"], ev["y"])
            elif etype == "click":
                button = _pmouse.Button.left if "left" in ev.get("button", "") else _pmouse.Button.right
                mouse_ctrl.position = (ev["x"], ev["y"])
                if ev.get("pressed"):
                    mouse_ctrl.press(button)
                else:
                    mouse_ctrl.release(button)
            elif etype == "scroll":
                mouse_ctrl.scroll(ev.get("dx", 0), ev.get("dy", 0))
            elif etype == "key_press":
                key_str = ev.get("key", "")
                try:
                    key = _pkbd.KeyCode.from_char(key_str) if len(key_str) == 1 else getattr(_pkbd.Key, key_str.replace("Key.", ""), None)
                    if key:
                        kbd_ctrl.press(key)
                except Exception:
                    pass
            elif etype == "key_release":
                key_str = ev.get("key", "")
                try:
                    key = _pkbd.KeyCode.from_char(key_str) if len(key_str) == 1 else getattr(_pkbd.Key, key_str.replace("Key.", ""), None)
                    if key:
                        kbd_ctrl.release(key)
                except Exception:
                    pass

        return _ok(f"Macro '{name}' replayed ({len(events)} events)")
    except Exception as e:
        return _err(f"play_macro error: {e}")


def list_macros() -> Dict:
    """List all saved macros."""
    try:
        macros = _load_json(MACROS_FILE)
        if not macros:
            return _ok("No macros saved", {"macros": []})
        items = [
            {
                "name": k,
                "events": len(v.get("events", [])),
                "duration": v.get("duration", 0),
                "recorded_at": v.get("recorded_at", ""),
            }
            for k, v in macros.items()
        ]
        lines = "\n".join(
            f"  {i['name']} ({i['events']} events, {i['duration']}s)" for i in items
        )
        return _ok(f"Macros:\n{lines}", {"macros": items})
    except Exception as e:
        return _err(f"list_macros error: {e}")


def delete_macro(name: str) -> Dict:
    """Delete a saved macro."""
    try:
        macros = _load_json(MACROS_FILE)
        if name not in macros:
            return _err(f"Macro '{name}' not found")
        del macros[name]
        _save_json(MACROS_FILE, macros)
        return _ok(f"Macro '{name}' deleted")
    except Exception as e:
        return _err(f"delete_macro error: {e}")


# ---------------------------------------------------------------------------
# Workflow CRUD
# ---------------------------------------------------------------------------

def create_workflow(name: str, steps: List[Dict]) -> Dict:
    """Create a named workflow from a steps list."""
    try:
        if not name:
            return _err("Workflow name cannot be empty")
        if not isinstance(steps, list) or not steps:
            return _err("Steps must be a non-empty list")
        workflows = _load_json(WORKFLOWS_FILE)
        workflows[name] = {
            "name": name,
            "steps": steps,
            "created_at": datetime.now().isoformat(),
            "schedule": None,
        }
        _save_json(WORKFLOWS_FILE, workflows)
        return _ok(f"Workflow '{name}' created with {len(steps)} step(s)")
    except Exception as e:
        return _err(f"create_workflow error: {e}")


def list_workflows() -> Dict:
    """List all saved workflows."""
    try:
        workflows = _load_json(WORKFLOWS_FILE)
        if not workflows:
            return _ok("No workflows saved", {"workflows": []})
        items = [
            {
                "name": k,
                "steps": len(v.get("steps", [])),
                "created_at": v.get("created_at", ""),
                "schedule": v.get("schedule"),
            }
            for k, v in workflows.items()
        ]
        lines = "\n".join(
            f"  {i['name']} ({i['steps']} steps)"
            + (f" [scheduled: {i['schedule']}]" if i["schedule"] else "")
            for i in items
        )
        return _ok(f"Workflows:\n{lines}", {"workflows": items})
    except Exception as e:
        return _err(f"list_workflows error: {e}")


def run_workflow(name: str) -> Dict:
    """Execute a saved workflow by name."""
    try:
        workflows = _load_json(WORKFLOWS_FILE)
        if name not in workflows:
            return _err(f"Workflow '{name}' not found")
        steps = workflows[name].get("steps", [])
        return execute_steps(steps)
    except Exception as e:
        return _err(f"run_workflow error: {e}")


def schedule_workflow(name: str, at: str) -> Dict:
    """
    Schedule a workflow.
    `at` examples: "14:30", "every 1h", "every 30m", "every day at 09:00"
    Uses Windows Task Scheduler via schtasks for persistent scheduling,
    or starts a background thread for in-process scheduling.
    """
    try:
        workflows = _load_json(WORKFLOWS_FILE)
        if name not in workflows:
            return _err(f"Workflow '{name}' not found")

        workflows[name]["schedule"] = at
        _save_json(WORKFLOWS_FILE, workflows)

        # Parse schedule and start background thread scheduler
        at_lower = at.lower().strip()

        interval_sec = None
        specific_time = None

        # "every Xh" / "every Xm" / "every Xs"
        m = re.match(r"every\s+(\d+)\s*(h|hour|m|min|minute|s|sec|second)", at_lower)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            if unit.startswith("h"):
                interval_sec = n * 3600
            elif unit.startswith("m"):
                interval_sec = n * 60
            else:
                interval_sec = n

        # "HH:MM" one-shot or daily
        m2 = re.match(r"(\d{1,2}):(\d{2})", at_lower)
        if m2 and not interval_sec:
            specific_time = (int(m2.group(1)), int(m2.group(2)))

        def _runner():
            if interval_sec:
                while True:
                    time.sleep(interval_sec)
                    try:
                        run_workflow(name)
                    except Exception:
                        pass
            elif specific_time:
                hh, mm = specific_time
                while True:
                    now = datetime.now()
                    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    if target <= now:
                        target += timedelta(days=1)
                    time.sleep((target - datetime.now()).total_seconds())
                    try:
                        run_workflow(name)
                    except Exception:
                        pass
                    time.sleep(60)

        if interval_sec or specific_time:
            t = threading.Thread(target=_runner, daemon=True, name=f"sched_{name}")
            t.start()
            return _ok(f"Workflow '{name}' scheduled: {at} (background thread started)")
        else:
            return _ok(f"Schedule '{at}' saved for '{name}' (run manually with run_workflow)")
    except Exception as e:
        return _err(f"schedule_workflow error: {e}")


# ---------------------------------------------------------------------------
# Step execution engine
# ---------------------------------------------------------------------------

def execute_steps(steps: List[Dict]) -> Dict:
    """
    Execute a list of action steps.
    Supported actions:
      open, close, type, click, hotkey, wait, screenshot,
      read_screen, write_file, read_file, run_command,
      scroll, focus, notify
    """
    try:
        if not isinstance(steps, list):
            return _err("Steps must be a list")
        results = []
        for i, step in enumerate(steps):
            action = step.get("action", "").lower()
            try:
                res = _execute_single_step(step)
                results.append({"step": i + 1, "action": action, "result": res})
                if not res.get("ok", True):
                    # Non-fatal: log and continue unless step has "required": true
                    if step.get("required", False):
                        break
            except Exception as step_err:
                results.append({"step": i + 1, "action": action, "result": _err(str(step_err))})
                if step.get("required", False):
                    break

        failed = [r for r in results if not r["result"].get("ok", True)]
        summary = f"Executed {len(results)} step(s), {len(failed)} failed"
        return _ok(summary, {"steps_executed": results, "failed_count": len(failed)})
    except Exception as e:
        return _err(f"execute_steps error: {e}\n{traceback.format_exc()}")


def _execute_single_step(step: Dict) -> Dict:
    """Dispatch a single step dict to its handler."""
    action = step.get("action", "").lower()

    # --- open ---
    if action == "open":
        target = step.get("target", "")
        if not target:
            return _err("open: no target specified")
        try:
            os.startfile(target)
            return _ok(f"Opened: {target}")
        except Exception:
            result = subprocess.Popen(
                target, shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            )
            return _ok(f"Launched: {target} (PID {result.pid})")

    # --- close ---
    elif action == "close":
        target = step.get("target", "")
        if not target:
            return _err("close: no target specified")
        killed = []
        try:
            import psutil
            for proc in psutil.process_iter(["name", "pid"]):
                if target.lower() in proc.info["name"].lower():
                    proc.terminate()
                    killed.append(proc.info["name"])
        except ImportError:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", f"{target}*"],
                capture_output=True, text=True
            )
            return _ok(result.stdout.strip() or "kill sent")
        if killed:
            return _ok(f"Closed: {', '.join(killed)}")
        return _err(f"No process matching '{target}' found")

    # --- type ---
    elif action == "type":
        text = step.get("text", "")
        delay = step.get("delay", 0.05)
        if not _PYAUTOGUI:
            return _err("pyautogui not installed; cannot type")
        pyautogui.write(text, interval=delay)
        return _ok(f"Typed: {text[:60]}{'...' if len(text) > 60 else ''}")

    # --- click ---
    elif action == "click":
        x = step.get("x")
        y = step.get("y")
        button = step.get("button", "left")
        clicks = step.get("clicks", 1)
        if x is None or y is None:
            return _err("click: x and y required")
        if not _PYAUTOGUI:
            return _err("pyautogui not installed")
        pyautogui.click(x, y, clicks=clicks, button=button)
        return _ok(f"Clicked ({x}, {y}) [{button}] x{clicks}")

    # --- hotkey ---
    elif action == "hotkey":
        keys = step.get("keys", step.get("key", ""))
        if not keys:
            return _err("hotkey: no keys specified")
        if not _PYAUTOGUI:
            return _err("pyautogui not installed")
        if isinstance(keys, str):
            key_list = [k.strip() for k in re.split(r"[+,\s]+", keys) if k.strip()]
        else:
            key_list = keys
        pyautogui.hotkey(*key_list)
        return _ok(f"Hotkey: {'+'.join(key_list)}")

    # --- wait ---
    elif action == "wait":
        seconds = float(step.get("seconds", step.get("duration", 1)))
        time.sleep(seconds)
        return _ok(f"Waited {seconds}s")

    # --- screenshot ---
    elif action == "screenshot":
        out_path = step.get("path", str(Path(r"C:\Users\chand\ai-remo\data\screenshots") / f"auto_{int(time.time())}.png"))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if not _PYAUTOGUI:
            return _err("pyautogui not installed; cannot take screenshot")
        img = pyautogui.screenshot()
        img.save(out_path)
        return _ok(f"Screenshot saved: {out_path}", {"path": out_path})

    # --- read_screen ---
    elif action == "read_screen":
        region = step.get("region")  # (x, y, w, h) or None for full screen
        if not _PYAUTOGUI:
            return _err("pyautogui not installed")
        if region:
            img = pyautogui.screenshot(region=tuple(region))
        else:
            img = pyautogui.screenshot()
        # Try OCR via pytesseract if available
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
            return _ok(f"Screen text:\n{text[:1000]}", {"text": text})
        except ImportError:
            tmp = Path(r"C:\Users\chand\ai-remo\data\screenshots") / f"read_screen_{int(time.time())}.png"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(tmp))
            return _ok(f"Screen captured (pytesseract not installed; saved to {tmp})", {"path": str(tmp)})

    # --- write_file ---
    elif action == "write_file":
        path = step.get("path", "")
        content = step.get("content", "")
        if not path:
            return _err("write_file: no path specified")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        mode = step.get("mode", "w")
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return _ok(f"Written to: {path}")

    # --- read_file ---
    elif action == "read_file":
        path = step.get("path", "")
        if not path:
            return _err("read_file: no path specified")
        if not os.path.exists(path):
            return _err(f"File not found: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return _ok(f"Read {len(content)} chars from {path}", {"content": content, "path": path})

    # --- run_command ---
    elif action == "run_command":
        cmd = step.get("command", step.get("cmd", ""))
        if not cmd:
            return _err("run_command: no command specified")
        timeout = int(step.get("timeout", 30))
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return _err(f"Command failed (rc={result.returncode}): {output[:500]}", {"rc": result.returncode, "output": output})
        return _ok(f"Command output: {output[:500]}", {"rc": result.returncode, "output": output})

    # --- scroll ---
    elif action == "scroll":
        x = step.get("x")
        y = step.get("y")
        amount = int(step.get("amount", 3))
        direction = step.get("direction", "down")
        if not _PYAUTOGUI:
            return _err("pyautogui not installed")
        if direction == "up":
            amount = abs(amount)
        elif direction == "down":
            amount = -abs(amount)
        if x is not None and y is not None:
            pyautogui.scroll(amount, x=x, y=y)
        else:
            pyautogui.scroll(amount)
        return _ok(f"Scrolled {direction} by {abs(amount)}")

    # --- focus ---
    elif action == "focus":
        target = step.get("target", "")
        if not target:
            return _err("focus: no target window title specified")
        if _WIN32:
            hwnd = win32gui.FindWindow(None, target)
            if not hwnd:
                # Partial match
                matches = []
                def _cb(h, _):
                    t = win32gui.GetWindowText(h)
                    if target.lower() in t.lower():
                        matches.append(h)
                win32gui.EnumWindows(_cb, None)
                hwnd = matches[0] if matches else 0
            if hwnd:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                return _ok(f"Focused window: {target}")
            return _err(f"Window not found: {target}")
        elif _PYAUTOGUI:
            # best-effort via pyautogui (Windows only)
            try:
                wins = pyautogui.getWindowsWithTitle(target)
                if wins:
                    wins[0].activate()
                    return _ok(f"Focused: {target}")
            except Exception:
                pass
        return _err(f"Cannot focus window (win32 not available): {target}")

    # --- notify ---
    elif action == "notify":
        title = step.get("title", "ARIA")
        message = step.get("message", step.get("text", ""))
        duration = int(step.get("duration", 5))
        # Try Windows Toast via win10toast or plyer
        try:
            from win10toast import ToastNotifier
            ToastNotifier().show_toast(title, message, duration=duration, threaded=True)
            return _ok(f"Notification sent: {message}")
        except ImportError:
            pass
        try:
            import plyer.notification as pn
            pn.notify(title=title, message=message, timeout=duration)
            return _ok(f"Notification sent: {message}")
        except ImportError:
            pass
        # Final fallback: msg command
        subprocess.Popen(
            ["msg", "*", f"{title}: {message}"],
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        return _ok(f"Notification (msg): {message}")

    else:
        return _err(f"Unknown action: '{action}'")


# ---------------------------------------------------------------------------
# Natural language execution
# ---------------------------------------------------------------------------

def execute_nl(instruction: str) -> Dict:
    """
    Parse a natural language instruction and execute the appropriate function.
    Examples:
      "create a workflow to open chrome and search for news"
      "run my morning_routine workflow"
      "record a macro called login"
      "play macro login"
      "list workflows"
      "set volume to 50"
      "ping google.com"
    """
    try:
        instr = instruction.strip().lower()

        # --- Record macro ---
        m = re.search(r"record\s+(?:a\s+)?macro\s+(?:called\s+|named\s+)?['\"]?(\w+)['\"]?", instr)
        if m:
            name = m.group(1)
            dur_m = re.search(r"(\d+)\s*(?:seconds?|secs?|s\b)", instr)
            duration = int(dur_m.group(1)) if dur_m else 30
            return record_macro(name, duration)

        # --- Play macro ---
        m = re.search(r"(?:play|replay|run)\s+(?:macro\s+)?['\"]?(\w+)['\"]?\s*macro", instr)
        if not m:
            m = re.search(r"macro\s+['\"]?(\w+)['\"]?", instr)
        if m and ("play" in instr or "replay" in instr or "run" in instr):
            name = m.group(1)
            return play_macro(name)

        # --- List macros ---
        if re.search(r"list\s+macros?", instr):
            return list_macros()

        # --- Delete macro ---
        m = re.search(r"delete\s+(?:macro\s+)?['\"]?(\w+)['\"]?", instr)
        if m and "macro" in instr:
            return delete_macro(m.group(1))

        # --- Run workflow ---
        m = re.search(r"run\s+(?:(?:my|the)\s+)?(?:workflow\s+)?['\"]?([\w_]+)['\"]?\s+workflow", instr)
        if not m:
            m = re.search(r"run\s+workflow\s+['\"]?([\w_]+)['\"]?", instr)
        if not m:
            m = re.search(r"(?:execute|start)\s+(?:workflow\s+)?['\"]?([\w_]+)['\"]?", instr)
        if m and ("workflow" in instr or "run" in instr):
            return run_workflow(m.group(1))

        # --- List workflows ---
        if re.search(r"list\s+workflows?", instr):
            return list_workflows()

        # --- Create workflow from NL ---
        if "create" in instr and "workflow" in instr:
            name_m = re.search(r"(?:called|named)\s+['\"]?([\w_]+)['\"]?", instr)
            wf_name = name_m.group(1) if name_m else f"wf_{int(time.time())}"
            steps = _nl_to_steps(instruction)
            return create_workflow(wf_name, steps)

        # --- Schedule workflow ---
        if "schedule" in instr and "workflow" in instr:
            wf_m = re.search(r"workflow\s+['\"]?([\w_]+)['\"]?", instr)
            at_m = re.search(r"(?:at|every)\s+([\d:hms\s]+)", instr)
            if wf_m and at_m:
                return schedule_workflow(wf_m.group(1), at_m.group(0))

        # --- Execute ad-hoc steps from NL ---
        steps = _nl_to_steps(instruction)
        if steps:
            return execute_steps(steps)

        return _err(f"Could not parse instruction: '{instruction}'")
    except Exception as e:
        return _err(f"execute_nl error: {e}\n{traceback.format_exc()}")


def _nl_to_steps(instruction: str) -> List[Dict]:
    """
    Convert a natural language instruction to a list of step dicts.
    Handles common patterns like:
      - "open chrome"
      - "type hello world"
      - "press ctrl+c"
      - "wait 2 seconds"
      - "click at 500 300"
      - "run command dir"
      - "search for news" -> opens browser with search
    """
    steps = []
    instr = instruction.lower()

    # open <app/url>
    m = re.search(r"open\s+([a-z0-9_\-./:\\]+)", instr)
    if m:
        target = m.group(1)
        # Common app name mappings
        app_map = {
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
            "notepad": "notepad.exe",
            "explorer": "explorer.exe",
            "calculator": "calc.exe",
            "cmd": "cmd.exe",
            "terminal": "wt.exe",
            "vscode": "code",
            "word": "winword.exe",
            "excel": "excel.exe",
        }
        steps.append({"action": "open", "target": app_map.get(target, target)})

    # search for <query>
    m = re.search(r"search\s+for\s+(.+?)(?:\s+in\s+\w+)?$", instr)
    if m:
        query = m.group(1).strip().replace(" ", "+")
        steps.append({"action": "open", "target": f"https://www.google.com/search?q={query}"})

    # type <text>
    m = re.search(r'type\s+["\']?(.+?)["\']?$', instruction, re.IGNORECASE)
    if m:
        steps.append({"action": "type", "text": m.group(1)})

    # press / hotkey <keys>
    m = re.search(r"(?:press|hotkey)\s+([\w+\s]+)", instr)
    if m:
        steps.append({"action": "hotkey", "keys": m.group(1).strip()})

    # wait <n> seconds
    m = re.search(r"wait\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)", instr)
    if m:
        steps.append({"action": "wait", "seconds": float(m.group(1))})

    # click at <x> <y>
    m = re.search(r"click\s+at\s+(\d+)\s+(\d+)", instr)
    if m:
        steps.append({"action": "click", "x": int(m.group(1)), "y": int(m.group(2))})

    # take screenshot
    if "screenshot" in instr or "take screenshot" in instr:
        steps.append({"action": "screenshot"})

    # run command <cmd>
    m = re.search(r"run\s+command\s+(.+)$", instruction, re.IGNORECASE)
    if m:
        steps.append({"action": "run_command", "command": m.group(1)})

    # notify <message>
    m = re.search(r"(?:notify|alert|show notification)\s+(.+)$", instr)
    if m:
        steps.append({"action": "notify", "message": m.group(1)})

    return steps


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== list_macros ===")
    print(json.dumps(list_macros(), indent=2))

    print("\n=== list_workflows ===")
    print(json.dumps(list_workflows(), indent=2))

    print("\n=== create_workflow test ===")
    r = create_workflow("test_wf", [
        {"action": "notify", "message": "ARIA workflow test started"},
        {"action": "wait", "seconds": 1},
        {"action": "notify", "message": "ARIA workflow test done"},
    ])
    print(json.dumps(r, indent=2))

    print("\n=== run_workflow test_wf ===")
    print(json.dumps(run_workflow("test_wf"), indent=2))

    print("\n=== execute_nl: open notepad ===")
    print(json.dumps(execute_nl("open notepad"), indent=2))

    print("\n=== execute_nl: create a workflow called demo to open chrome ===")
    print(json.dumps(execute_nl("create a workflow called demo to open chrome"), indent=2))
