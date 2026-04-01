"""
ARIA — IoT Agent + Self-Improvement Engine
==========================================
IoT Agent: ultra-lightweight ARIA for Raspberry Pi, ARM devices, ESP32.
Self-Improvement Engine: nightly improvement cycle — patches, learns, gets smarter.

IoT install (Raspberry Pi):
    pip install requests pyttsx3 sounddevice scipy
    python agents/iot_agent.py --server http://192.168.1.X:8000 --voice
"""

import os
import re
import sys
import json
import time
import socket
import requests
import threading
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# IOT AGENT
# ─────────────────────────────────────────────────────────────────────────────

class IoTAgent:
    """
    Minimal ARIA agent for IoT and embedded devices.
    No LLM required. Uses tool registry + server delegation.
    Memory footprint: ~20MB base, ~50MB with voice.
    """

    def __init__(self, server_url="http://localhost:8000",
                 device_name=None, voice=False, mqtt_host=None):
        self.server      = server_url.rstrip("/")
        self.device_name = device_name or socket.gethostname()
        self.voice_en    = voice
        self.mqtt_host   = mqtt_host
        self._running    = False

        # Load ONLY IoT-safe tools — no model, no heavy imports
        from agents.tool_registry import build_default_registry
        self.registry = build_default_registry(iot_mode=True)

        print(f"[ARIA IoT] Device: {self.device_name}")
        print(f"[ARIA IoT] Server: {self.server}")
        print(f"[ARIA IoT] Tools:  {len(self.registry._tools)} loaded (IoT-safe)")

    def start(self):
        self._running = True
        threading.Thread(target=self._http_listener, daemon=True).start()
        print("[ARIA IoT] HTTP: http://0.0.0.0:8001/action")
        if self.voice_en:
            threading.Thread(target=self._voice_listener, daemon=True).start()
        if self.mqtt_host:
            threading.Thread(target=self._mqtt_listener, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        print("[ARIA IoT] Ready")
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self._running = False

    def execute(self, text: str) -> dict:
        """Try local tools first, delegate to server if needed."""
        candidates = self.registry.route(text)
        if candidates:
            tool, args, conf = candidates[0]
            if conf > 0.3:
                print(f"[ARIA IoT] Local: {tool.name} ({conf:.2f})")
                return self.registry.execute(tool.name, args)
        return self._delegate(text)

    def _delegate(self, text: str) -> dict:
        try:
            r = requests.post(
                f"{self.server}/api/action",
                json={"command": text, "device": self.device_name},
                timeout=30,
            )
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e), "offline": True}

    def _http_listener(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        agent = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/action":
                    n = int(self.headers.get("Content-Length", 0))
                    b = json.loads(self.rfile.read(n)) if n else {}
                    r = agent.execute(b.get("command", ""))
                    self.send_response(200)
                    self.send_header("Content-Type","application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(r).encode())

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type","application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "device": agent.device_name,
                        "tools":  len(agent.registry._tools),
                        "server": agent.server,
                    }).encode())

            def log_message(self, *a): pass

        HTTPServer(("0.0.0.0", 8001), H).serve_forever()

    def _voice_listener(self):
        try:
            import sounddevice as sd, scipy.io.wavfile as wav, numpy as np
            import whisper, tempfile
            model = whisper.load_model("tiny")   # 39MB — works on Pi 4
            print("[ARIA IoT] Voice: Whisper tiny active")
            RATE = 16000
            while self._running:
                audio = sd.rec(int(3*RATE), samplerate=RATE, channels=1, dtype=np.float32)
                sd.wait()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav.write(f.name, RATE, (audio*32767).astype(np.int16))
                    text = model.transcribe(f.name)["text"].lower().strip()
                if "aria" in text:
                    self._speak("Yes?")
                    audio2 = sd.rec(int(6*RATE), samplerate=RATE, channels=1, dtype=np.float32)
                    sd.wait()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav.write(f.name, RATE, (audio2*32767).astype(np.int16))
                        cmd = model.transcribe(f.name)["text"].strip()
                    if cmd:
                        result = self.execute(cmd)
                        self._speak(str(result.get("message","Done"))[:100])
        except ImportError:
            print("[ARIA IoT] Voice needs: pip install openai-whisper sounddevice scipy")

    def _speak(self, text: str):
        try:
            import pyttsx3
            e = pyttsx3.init(); e.say(text); e.runAndWait()
        except Exception:
            print(f"[ARIA IoT] Say: {text}")

    def _mqtt_listener(self):
        try:
            import paho.mqtt.client as mqtt
            def on_msg(c, u, m):
                try:
                    cmd = json.loads(m.payload).get("command","")
                    if cmd:
                        r = self.execute(cmd)
                        c.publish("aria/result", json.dumps(r))
                except Exception: pass
            c = mqtt.Client()
            c.on_message = on_msg
            c.connect(self.mqtt_host, 1883, 60)
            c.subscribe("aria/command")
            print(f"[ARIA IoT] MQTT: {self.mqtt_host}")
            c.loop_forever()
        except ImportError:
            print("[ARIA IoT] pip install paho-mqtt for MQTT")

    def _heartbeat(self):
        while self._running:
            try:
                requests.post(f"{self.server}/api/device/heartbeat", json={
                    "device": self.device_name,
                    "ts":     datetime.now().isoformat(),
                    "tools":  len(self.registry._tools),
                    "ip":     socket.gethostbyname(socket.gethostname()),
                }, timeout=5)
            except Exception: pass
            time.sleep(60)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-IMPROVEMENT ENGINE
# Runs nightly — patches, learns, tests, reports
# ─────────────────────────────────────────────────────────────────────────────

class SelfImprovementEngine:
    """
    ARIA's nightly self-improvement cycle.

    1. Analyse tool failure rates → suggest fixes
    2. Check for new/better Ollama models → auto-download
    3. Run dependency security audit → flag vulnerabilities
    4. Run self-test suite → verify health
    5. Generate improvement report → log + notify
    6. LLM-generated suggestions for next iteration

    Schedule: runs automatically at 3am via apscheduler
    Manual:   python agents/iot_agent.py --improve
    """

    def __init__(self, registry=None, engine=None, notifier=None):
        self.registry = registry
        self.engine   = engine
        self.notifier = notifier
        self.log_path = PROJECT_ROOT / "logs" / "self_improvement.jsonl"
        self.log_path.parent.mkdir(exist_ok=True)

    def run_cycle(self) -> dict:
        print("[Self-Improvement] Starting nightly cycle...")
        report = {
            "ts":             datetime.now().isoformat(),
            "tool_issues":    self._analyse_tools(),
            "new_models":     self._check_models(),
            "security":       self._security_scan(),
            "test_results":   self._self_test(),
            "llm_suggestions":"",
        }

        if self.engine and (report["tool_issues"] or report["security"]):
            report["llm_suggestions"] = self._llm_suggest(report)

        # Notify user if issues found
        issues = len(report["tool_issues"]) + len(report["security"])
        if self.notifier and issues > 0:
            self.notifier.notify(
                "ARIA Self-Improvement",
                f"Found {issues} issues to address. Check logs."
            )

        # Save report
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(report) + "\n")
        except Exception:
            pass

        print(f"[Self-Improvement] Done. Issues: {issues}, Tests: {report['test_results']}")
        return report

    def _analyse_tools(self) -> list[str]:
        issues = []
        if not self.registry:
            return issues
        for tool in self.registry._tools.values():
            if tool.usage_count >= 5:
                fr = 1 - (tool.success_count / max(tool.usage_count, 1))
                if fr > 0.3:
                    issues.append(f"'{tool.name}': {fr*100:.0f}% failure ({tool.usage_count} uses)")
                if tool.avg_latency_ms > 8000:
                    issues.append(f"'{tool.name}': slow ({tool.avg_latency_ms:.0f}ms avg)")
        return issues

    def _check_models(self) -> list[str]:
        want    = ["llama3.2","phi3:mini","nomic-embed-text","moondream"]
        missing = []
        try:
            r         = requests.get("http://localhost:11434/api/tags", timeout=3)
            installed = [m["name"] for m in r.json().get("models",[])]
            missing   = [m for m in want if not any(m.split(":")[0] in i for i in installed)]
        except Exception:
            pass
        return missing

    def _security_scan(self) -> list[str]:
        vulns = []
        try:
            r = subprocess.run(
                ["pip-audit","--format=json","--progress-spinner=off"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode in (0,1):
                for dep in json.loads(r.stdout).get("dependencies",[]):
                    for v in dep.get("vulns",[]):
                        vulns.append(f"{dep['name']} {dep['version']}: {v['id']}")
        except Exception:
            pass
        return vulns

    def _self_test(self) -> dict:
        results = {}

        # Tool registry
        try:
            from agents.tool_registry import build_default_registry
            reg  = build_default_registry()
            hits = reg.route("what time is it")
            results["tool_registry"] = "ok" if hits else "no_matches"
        except Exception as e:
            results["tool_registry"] = f"error:{e}"

        # ARIA server
        try:
            r = requests.get("http://localhost:8000/api/health", timeout=3)
            results["server"] = "ok" if r.ok else f"http_{r.status_code}"
        except Exception:
            results["server"] = "offline"

        # Ollama
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            results["ollama"] = f"ok:{len(r.json().get('models',[]))}models"
        except Exception:
            results["ollama"] = "offline"

        # ChromaDB
        try:
            import chromadb
            c    = chromadb.PersistentClient(path=str(PROJECT_ROOT/"data"/"chroma_db"))
            cols = c.list_collections()
            results["chromadb"] = f"ok:{len(cols)}cols"
        except Exception as e:
            results["chromadb"] = f"error:{e}"

        return results

    def _llm_suggest(self, report: dict) -> str:
        if not self.engine:
            return ""
        prompt = (
            f"ARIA self-improvement report:\n"
            f"Tool issues: {report['tool_issues'][:3]}\n"
            f"Security: {report['security'][:3]}\n"
            f"Tests: {report['test_results']}\n\n"
            f"Give 3 specific, actionable fixes:"
        )
        try:
            return self.engine.generate(prompt, temperature=0.2)
        except Exception:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ARIA IoT Agent / Self-Improvement")
    p.add_argument("--server",  default="http://localhost:8000")
    p.add_argument("--name",    default=None)
    p.add_argument("--voice",   action="store_true")
    p.add_argument("--mqtt",    default=None)
    p.add_argument("--improve", action="store_true")
    args = p.parse_args()

    if args.improve:
        eng = SelfImprovementEngine()
        print(json.dumps(eng.run_cycle(), indent=2))
        sys.exit(0)

    IoTAgent(args.server, args.name, args.voice, args.mqtt).start()
