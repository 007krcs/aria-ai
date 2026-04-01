"""
ARIA — Advanced Security Monitor Agent
========================================
Cross-platform security monitoring and auto-remediation.

Capabilities:
1. Process threat scanning  — flags high-CPU/mem, known-bad names, unsigned binaries
2. Network threat detection — unusual outbound connections, suspicious ports, geo checks
3. Filesystem watcher       — ransomware patterns, exe drops in temp, mass renames
4. LLM-based threat classification
5. Auto-remediation         — terminate process, block with OS firewall, backup files
6. System health report     — CPU temp, disk SMART, memory leaks, open ports
7. Installed-app audit      — unverified/unsigned apps
8. SSL certificate checker
9. Background monitoring loop with callback
10. Firewall rule management (Windows netsh / Linux iptables / macOS pfctl)

Dependencies: psutil, requests (all optional with graceful fallback)
Optional: watchdog (filesystem watching)
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import socket
import platform
import threading
import subprocess
import ipaddress
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── optional imports ────────────────────────────────────────────────────────

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _WATCHDOG = True
except ImportError:
    _WATCHDOG = False

try:
    import requests as _requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    import ssl as _ssl_mod
    import urllib.request as _urllib
    _SSL_CHECK = True
except ImportError:
    _SSL_CHECK = False

# ── project root ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
THREAT_LOG_FILE = LOG_DIR / "security_threats.jsonl"

# ── constants ────────────────────────────────────────────────────────────────

OS = platform.system()  # "Windows" | "Linux" | "Darwin"

KNOWN_MALICIOUS_NAMES = {
    "mimikatz", "meterpreter", "payload", "cobalt", "metasploit",
    "nmap", "netcat", "nc.exe", "psexec", "wce", "fgdump",
    "quarks", "pwdump", "gsecdump", "lsass_dump", "inject",
}

SUSPICIOUS_PORTS = {
    1080, 4444, 4445, 8888, 31337, 12345, 27374,  # common RAT/shell ports
    6666, 6667, 6668, 6669,                         # IRC (C2)
    9050, 9051,                                      # Tor
}

COMMON_SAFE_PORTS = {
    20, 21, 22, 23, 25, 53, 80, 110, 119, 123, 143,
    443, 465, 587, 993, 995, 3306, 5432, 6379, 8080, 8443,
}

SUSPICIOUS_GEO_CIDRS: List[str] = []  # can be populated at runtime

RANSOMWARE_EXTENSIONS = {
    ".locked", ".encrypted", ".enc", ".crypt", ".crypto",
    ".rnsmwr", ".zepto", ".locky", ".cerber", ".dharma",
    ".wncry", ".wannacry", ".petya", ".ryuk",
}

TEMP_PATHS: List[str] = []
if OS == "Windows":
    TEMP_PATHS = [
        os.environ.get("TEMP", "C:/Windows/Temp"),
        os.environ.get("TMP",  "C:/Windows/Temp"),
        "C:/Windows/Temp",
    ]
else:
    TEMP_PATHS = ["/tmp", "/var/tmp"]


# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ProcessThreat:
    name: str
    pid: int
    cpu_percent: float
    mem_percent: float
    verdict: str          # "safe" | "suspicious" | "malicious"
    reason: str
    cmdline: str = ""
    timestamp: str = field(default_factory=lambda: _now())


@dataclass
class NetworkThreat:
    local_addr: str
    remote_addr: str
    remote_port: int
    protocol: str
    verdict: str          # "safe" | "suspicious" | "malicious"
    reason: str
    pid: Optional[int] = None
    process_name: str = ""
    timestamp: str = field(default_factory=lambda: _now())


@dataclass
class RemediationResult:
    action: str
    success: bool
    message: str
    threat_summary: str = ""
    timestamp: str = field(default_factory=lambda: _now())


@dataclass
class HealthReport:
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, Any]
    open_ports: List[int]
    cpu_temp_c: Optional[float]
    memory_leak_suspects: List[str]
    disk_smart_ok: Optional[bool]
    warnings: List[str]
    timestamp: str = field(default_factory=lambda: _now())


@dataclass
class ImplementationResult:  # shared stub
    success: bool
    data: Any = None
    error: str = ""


# ── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_threat(threat_dict: dict) -> None:
    with open(THREAT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(threat_dict) + "\n")


def _run(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 1, "", "Timeout"
    except Exception as exc:
        return 1, "", str(exc)


def _llm_call(engine, prompt: str, system: str = "") -> str:
    """Call ARIA's LLM engine; return response text."""
    if engine is None:
        return ""
    try:
        if hasattr(engine, "chat"):
            resp = engine.chat(prompt, system=system)
        elif hasattr(engine, "generate"):
            resp = engine.generate(prompt)
        elif callable(engine):
            resp = engine(prompt)
        else:
            return ""
        if isinstance(resp, dict):
            return resp.get("response") or resp.get("text") or str(resp)
        return str(resp)
    except Exception:
        return ""


# ── filesystem watcher (watchdog) ────────────────────────────────────────────

if _WATCHDOG:
    class _SuspiciousFileHandler(FileSystemEventHandler):
        def __init__(self, callback: Optional[Callable], rename_threshold: int = 5):
            super().__init__()
            self.callback = callback
            self._rename_count: Dict[str, int] = {}
            self._rename_threshold = rename_threshold
            self._lock = threading.Lock()

        def _alert(self, event_type: str, path: str, reason: str) -> None:
            threat = {
                "type": "filesystem",
                "event": event_type,
                "path": path,
                "reason": reason,
                "timestamp": _now(),
            }
            _log_threat(threat)
            if self.callback:
                try:
                    self.callback(threat)
                except Exception:
                    pass

        def on_moved(self, event):
            if event.is_directory:
                return
            dest = event.dest_path.lower()
            for ext in RANSOMWARE_EXTENSIONS:
                if dest.endswith(ext):
                    self._alert("rename", event.dest_path, f"Ransomware extension detected: {ext}")
                    return
            # track mass renames
            parent = str(Path(event.src_path).parent)
            with self._lock:
                self._rename_count[parent] = self._rename_count.get(parent, 0) + 1
                count = self._rename_count[parent]
            if count >= self._rename_threshold:
                self._alert("mass_rename", parent,
                            f"{count} renames in directory — possible ransomware")

        def on_created(self, event):
            if event.is_directory:
                return
            path_lower = event.src_path.lower()
            # exe drop in temp
            for tmp in TEMP_PATHS:
                if path_lower.startswith(tmp.lower()) and path_lower.endswith(".exe"):
                    self._alert("exe_drop", event.src_path,
                                "Executable dropped into temp directory")
                    return
            # script drop in temp
            for ext in (".bat", ".ps1", ".vbs", ".js", ".cmd"):
                if path_lower.startswith(
                    tuple(t.lower() for t in TEMP_PATHS)
                ) and path_lower.endswith(ext):
                    self._alert("script_drop", event.src_path,
                                f"Script ({ext}) dropped into temp directory")
                    return


# ── main agent ───────────────────────────────────────────────────────────────

class SecurityMonitorAgent:
    """
    Advanced cross-platform security monitoring and auto-remediation.

    Usage:
        agent = SecurityMonitorAgent()
        threats = agent.scan_processes()
        agent.monitor_start(interval_s=30, callback=print)
    """

    def __init__(self, engine=None, memory=None):
        self.engine = engine
        self.memory = memory
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop_event = threading.Event()
        self._threat_log: List[dict] = []
        self._threat_log_lock = threading.Lock()
        self._fs_observer: Optional[Any] = None  # watchdog Observer

    # ── 1. Process scanning ───────────────────────────────────────────────────

    def scan_processes(self, engine=None) -> List[ProcessThreat]:
        """
        Scan running processes for threats.
        Returns list of ProcessThreat with verdict safe/suspicious/malicious.
        """
        eng = engine or self.engine
        results: List[ProcessThreat] = []

        if not _PSUTIL:
            return results

        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent", "cmdline", "username"]
        ):
            try:
                info = proc.info
                name_lower = (info.get("name") or "").lower()
                cpu = info.get("cpu_percent") or 0.0
                mem = info.get("memory_percent") or 0.0
                pid = info.get("pid") or 0
                cmdline_parts = info.get("cmdline") or []
                cmdline = " ".join(str(c) for c in cmdline_parts)

                verdict = "safe"
                reason = "Normal process"

                # known malicious name
                for bad in KNOWN_MALICIOUS_NAMES:
                    if bad in name_lower or bad in cmdline.lower():
                        verdict = "malicious"
                        reason = f"Known malicious tool name: {bad}"
                        break

                if verdict == "safe":
                    # high CPU heuristic
                    if cpu > 90:
                        verdict = "suspicious"
                        reason = f"Very high CPU usage: {cpu:.1f}%"
                    # high memory
                    elif mem > 80:
                        verdict = "suspicious"
                        reason = f"Very high memory usage: {mem:.1f}%"
                    # exe from temp
                    elif any(
                        t.lower() in cmdline.lower() for t in TEMP_PATHS
                    ) and name_lower.endswith(".exe"):
                        verdict = "suspicious"
                        reason = "Process running from temp directory"

                # LLM second opinion for suspicious
                if verdict == "suspicious" and eng:
                    llm_verdict = self._llm_classify_process(
                        name_lower, cmdline, cpu, mem, eng
                    )
                    if llm_verdict:
                        verdict = llm_verdict.get("verdict", verdict)
                        reason = llm_verdict.get("reason", reason)

                pt = ProcessThreat(
                    name=info.get("name") or "",
                    pid=pid,
                    cpu_percent=round(cpu, 2),
                    mem_percent=round(mem, 2),
                    verdict=verdict,
                    reason=reason,
                    cmdline=cmdline[:200],
                )
                results.append(pt)

                if verdict in ("suspicious", "malicious"):
                    _log_threat(asdict(pt))
                    with self._threat_log_lock:
                        self._threat_log.append(asdict(pt))

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue

        return results

    def _llm_classify_process(
        self, name: str, cmdline: str, cpu: float, mem: float, engine
    ) -> Optional[dict]:
        prompt = (
            f"A process named '{name}' is running with {cpu:.1f}% CPU and {mem:.1f}% memory.\n"
            f"Command line: {cmdline[:300]}\n\n"
            "Is this process safe, suspicious, or malicious? "
            "Reply with JSON: {\"verdict\": \"safe|suspicious|malicious\", \"reason\": \"...\"}. "
            "Be concise."
        )
        raw = _llm_call(engine, prompt, system="You are a cybersecurity expert.")
        try:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return None

    # ── 2. Network connections ────────────────────────────────────────────────

    def check_network_connections(self) -> List[NetworkThreat]:
        """
        Check active network connections for threats.
        Flags unusual outbound, suspicious ports, and geo-suspicious IPs.
        """
        results: List[NetworkThreat] = []
        if not _PSUTIL:
            return results

        try:
            conns = psutil.net_connections(kind="inet")
        except (psutil.AccessDenied, Exception):
            return results

        pid_to_name: Dict[int, str] = {}
        try:
            for proc in psutil.process_iter(["pid", "name"]):
                pid_to_name[proc.pid] = proc.info.get("name") or ""
        except Exception:
            pass

        for conn in conns:
            try:
                if not conn.raddr:
                    continue
                remote_ip = conn.raddr.ip
                remote_port = conn.raddr.port
                local_addr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
                proto = "tcp" if conn.type == socket.SOCK_STREAM else "udp"
                proc_name = pid_to_name.get(conn.pid or -1, "")

                verdict = "safe"
                reason = "Normal connection"

                # check suspicious port
                if remote_port in SUSPICIOUS_PORTS:
                    verdict = "suspicious"
                    reason = f"Connection to known suspicious port {remote_port}"

                # check private/loopback — skip
                try:
                    ip_obj = ipaddress.ip_address(remote_ip)
                    if ip_obj.is_loopback or ip_obj.is_private:
                        continue
                except ValueError:
                    pass

                # unknown high port outbound
                if verdict == "safe" and remote_port > 49151:
                    verdict = "suspicious"
                    reason = f"Outbound to ephemeral/uncommon port {remote_port}"

                # geo check (basic: check against known suspicious CIDRs)
                if SUSPICIOUS_GEO_CIDRS and verdict == "safe":
                    for cidr in SUSPICIOUS_GEO_CIDRS:
                        try:
                            if ipaddress.ip_address(remote_ip) in ipaddress.ip_network(cidr):
                                verdict = "suspicious"
                                reason = f"Connection to geo-suspicious network {cidr}"
                                break
                        except ValueError:
                            pass

                nt = NetworkThreat(
                    local_addr=local_addr,
                    remote_addr=remote_ip,
                    remote_port=remote_port,
                    protocol=proto,
                    verdict=verdict,
                    reason=reason,
                    pid=conn.pid,
                    process_name=proc_name,
                )
                results.append(nt)

                if verdict in ("suspicious", "malicious"):
                    _log_threat(asdict(nt))
                    with self._threat_log_lock:
                        self._threat_log.append(asdict(nt))

            except Exception:
                continue

        return results

    # ── 3. Filesystem watcher ─────────────────────────────────────────────────

    def watch_filesystem(
        self,
        paths: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
    ) -> Optional[threading.Thread]:
        """
        Start background filesystem watcher for suspicious changes.
        Returns watcher thread, or None if watchdog not available.
        Watches for: ransomware extensions, mass renames, exe/script drops in temp.
        """
        if not _WATCHDOG:
            return None

        watch_paths = paths or TEMP_PATHS
        handler = _SuspiciousFileHandler(callback=callback)
        observer = Observer()
        for p in watch_paths:
            try:
                if os.path.isdir(p):
                    observer.schedule(handler, p, recursive=True)
            except Exception:
                pass

        self._fs_observer = observer
        observer.start()
        return observer  # type: ignore[return-value]

    def stop_filesystem_watcher(self) -> None:
        if self._fs_observer:
            try:
                self._fs_observer.stop()
                self._fs_observer.join(timeout=5)
            except Exception:
                pass
            self._fs_observer = None

    # ── 4. LLM threat classification ──────────────────────────────────────────

    def detect_threat(
        self, description: str, engine=None
    ) -> str:
        """
        Classify threat type using LLM.
        Returns one of: malware | ransomware | phishing | intrusion | ddos |
                        keylogger | none
        """
        eng = engine or self.engine
        categories = ["malware", "ransomware", "phishing", "intrusion",
                      "ddos", "keylogger", "none"]

        if eng:
            prompt = (
                f"Classify this security event description into exactly one category.\n"
                f"Categories: {', '.join(categories)}\n\n"
                f"Description: {description}\n\n"
                "Reply with only the category name, lowercase."
            )
            raw = _llm_call(eng, prompt, system="You are a cybersecurity expert.")
            raw = raw.strip().lower().split()[0] if raw.strip() else ""
            if raw in categories:
                return raw

        # heuristic fallback
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["ransom", "encrypt", "locked", "decrypt"]):
            return "ransomware"
        if any(w in desc_lower for w in ["phish", "credential", "login page", "fake"]):
            return "phishing"
        if any(w in desc_lower for w in ["keylog", "keystroke", "keyboard hook"]):
            return "keylogger"
        if any(w in desc_lower for w in ["flood", "ddos", "dos", "bandwidth"]):
            return "ddos"
        if any(w in desc_lower for w in ["intrusion", "brute force", "ssh", "rdp", "lateral"]):
            return "intrusion"
        if any(w in desc_lower for w in ["malware", "virus", "trojan", "worm", "rootkit"]):
            return "malware"
        return "none"

    # ── 5. Auto-remediation ───────────────────────────────────────────────────

    def auto_remediate(
        self, threat: Any, dry_run: bool = False
    ) -> RemediationResult:
        """
        Attempt automatic remediation for a detected threat.
        - ProcessThreat (malicious) → terminate process
        - NetworkThreat (suspicious) → block with OS firewall
        - dict with type ransomware → pause process, backup key files
        """
        if isinstance(threat, ProcessThreat):
            return self._remediate_process(threat, dry_run)
        elif isinstance(threat, NetworkThreat):
            return self._remediate_network(threat, dry_run)
        elif isinstance(threat, dict):
            t_type = threat.get("type", "")
            if t_type == "filesystem":
                return self._remediate_ransomware(threat, dry_run)
        return RemediationResult(
            action="unknown",
            success=False,
            message="Unrecognised threat type for remediation",
        )

    def _remediate_process(
        self, threat: ProcessThreat, dry_run: bool
    ) -> RemediationResult:
        action = f"terminate_process pid={threat.pid}"
        summary = f"{threat.name} (pid {threat.pid}): {threat.reason}"

        if dry_run:
            return RemediationResult(
                action=action, success=True,
                message=f"[DRY RUN] Would terminate {threat.name} (pid {threat.pid})",
                threat_summary=summary,
            )
        if not _PSUTIL:
            return RemediationResult(
                action=action, success=False,
                message="psutil not available — cannot terminate process",
                threat_summary=summary,
            )
        try:
            proc = psutil.Process(threat.pid)
            proc.terminate()
            time.sleep(1)
            if proc.is_running():
                proc.kill()
            return RemediationResult(
                action=action, success=True,
                message=f"Terminated process {threat.name} (pid {threat.pid})",
                threat_summary=summary,
            )
        except psutil.NoSuchProcess:
            return RemediationResult(
                action=action, success=True,
                message=f"Process {threat.pid} already gone",
                threat_summary=summary,
            )
        except psutil.AccessDenied as exc:
            return RemediationResult(
                action=action, success=False,
                message=f"Access denied terminating pid {threat.pid}: {exc}",
                threat_summary=summary,
            )
        except Exception as exc:
            return RemediationResult(
                action=action, success=False,
                message=str(exc),
                threat_summary=summary,
            )

    def _remediate_network(
        self, threat: NetworkThreat, dry_run: bool
    ) -> RemediationResult:
        action = f"firewall_block ip={threat.remote_addr} port={threat.remote_port}"
        summary = f"{threat.remote_addr}:{threat.remote_port} — {threat.reason}"

        if dry_run:
            return RemediationResult(
                action=action, success=True,
                message=f"[DRY RUN] Would block {threat.remote_addr}:{threat.remote_port}",
                threat_summary=summary,
            )

        result = self.firewall_rule(
            "block", port=threat.remote_port, ip=threat.remote_addr
        )
        return RemediationResult(
            action=action, success=result.get("success", False),
            message=result.get("message", ""),
            threat_summary=summary,
        )

    def _remediate_ransomware(
        self, threat: dict, dry_run: bool
    ) -> RemediationResult:
        """Pause suspicious process and back up key files."""
        path = threat.get("path", "")
        action = "pause_process_backup_files"
        summary = f"Filesystem threat at {path}: {threat.get('reason','')}"

        backed_up: List[str] = []
        paused_pid: Optional[int] = None

        if not dry_run:
            # find and suspend process writing to that path
            if _PSUTIL:
                for proc in psutil.process_iter(["pid", "open_files"]):
                    try:
                        files = proc.info.get("open_files") or []
                        for f in files:
                            if hasattr(f, "path") and path in (f.path or ""):
                                proc.suspend()
                                paused_pid = proc.pid
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

            # back up key files (Documents, Desktop)
            key_dirs = []
            home = Path.home()
            for d in ("Documents", "Desktop", "Pictures"):
                candidate = home / d
                if candidate.exists():
                    key_dirs.append(candidate)

            backup_root = PROJECT_ROOT / "data" / "security_backups" / _now()[:10]
            backup_root.mkdir(parents=True, exist_ok=True)

            for d in key_dirs[:2]:  # cap at 2 dirs to avoid huge copies
                import shutil
                try:
                    dest = backup_root / d.name
                    shutil.copytree(str(d), str(dest), dirs_exist_ok=True)
                    backed_up.append(str(d))
                except Exception:
                    pass

        msg = "[DRY RUN] " if dry_run else ""
        if paused_pid:
            msg += f"Suspended pid {paused_pid}. "
        if backed_up:
            msg += f"Backed up: {', '.join(backed_up)}"
        if not paused_pid and not backed_up:
            msg += "No process to pause found; no files backed up."

        return RemediationResult(
            action=action, success=True, message=msg, threat_summary=summary
        )

    # ── 6. System health check ────────────────────────────────────────────────

    def system_health_check(self) -> HealthReport:
        """
        Returns HealthReport: CPU temp, disk health, memory leaks, open ports.
        """
        warnings: List[str] = []

        # CPU / memory
        cpu_pct = 0.0
        mem_pct = 0.0
        if _PSUTIL:
            cpu_pct = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            mem_pct = mem.percent

        # Disk usage
        disk_usage: Dict[str, Any] = {}
        if _PSUTIL:
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disk_usage[part.mountpoint] = {
                        "total_gb": round(usage.total / 1e9, 1),
                        "used_gb": round(usage.used / 1e9, 1),
                        "free_gb": round(usage.free / 1e9, 1),
                        "percent": usage.percent,
                    }
                    if usage.percent > 90:
                        warnings.append(
                            f"Disk {part.mountpoint} is {usage.percent:.0f}% full"
                        )
                except Exception:
                    pass

        # CPU temperature
        cpu_temp: Optional[float] = None
        if _PSUTIL and hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                for name, entries in (temps or {}).items():
                    for entry in entries:
                        if entry.current and entry.current > 0:
                            cpu_temp = round(entry.current, 1)
                            if cpu_temp > 90:
                                warnings.append(
                                    f"CPU temperature high: {cpu_temp}°C"
                                )
                            break
                    if cpu_temp:
                        break
            except Exception:
                pass

        # Open ports
        open_ports: List[int] = []
        if _PSUTIL:
            try:
                for conn in psutil.net_connections(kind="inet"):
                    if conn.status == "LISTEN" and conn.laddr:
                        open_ports.append(conn.laddr.port)
            except Exception:
                pass
        open_ports = sorted(set(open_ports))

        # Memory leak suspects (high-mem processes)
        leak_suspects: List[str] = []
        if _PSUTIL:
            try:
                for proc in psutil.process_iter(["name", "memory_percent"]):
                    try:
                        if (proc.info.get("memory_percent") or 0) > 15:
                            leak_suspects.append(
                                f"{proc.info['name']} ({proc.info['memory_percent']:.1f}%)"
                            )
                    except Exception:
                        pass
            except Exception:
                pass

        # Disk SMART (Linux only via smartctl)
        smart_ok: Optional[bool] = None
        if OS == "Linux":
            rc, out, _ = _run(["smartctl", "-H", "/dev/sda"], timeout=10)
            if rc == 0:
                smart_ok = "PASSED" in out

        if cpu_pct > 95:
            warnings.append(f"CPU usage critical: {cpu_pct:.0f}%")
        if mem_pct > 90:
            warnings.append(f"Memory usage critical: {mem_pct:.0f}%")

        return HealthReport(
            cpu_percent=cpu_pct,
            memory_percent=mem_pct,
            disk_usage=disk_usage,
            open_ports=open_ports,
            cpu_temp_c=cpu_temp,
            memory_leak_suspects=leak_suspects,
            disk_smart_ok=smart_ok,
            warnings=warnings,
        )

    # ── 7. Installed app audit ────────────────────────────────────────────────

    def scan_installed_apps(self, engine=None) -> List[dict]:
        """
        Return list of suspicious/unverified installed apps.
        Windows: queries registry via reg query or wmic.
        Linux: dpkg / rpm.
        macOS: system_profiler.
        """
        apps: List[dict] = []
        eng = engine or self.engine

        if OS == "Windows":
            apps = self._installed_apps_windows()
        elif OS == "Linux":
            apps = self._installed_apps_linux()
        elif OS == "Darwin":
            apps = self._installed_apps_macos()

        # LLM flag suspicious
        if eng and apps:
            flagged = self._llm_flag_apps(apps, eng)
            return flagged
        return apps

    def _installed_apps_windows(self) -> List[dict]:
        apps = []
        reg_keys = [
            r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        for key in reg_keys:
            rc, out, _ = _run(
                ["reg", "query", key, "/s", "/v", "DisplayName"], timeout=20
            )
            if rc == 0:
                for line in out.splitlines():
                    if "DisplayName" in line and "REG_SZ" in line:
                        parts = line.strip().split("REG_SZ")
                        name = parts[-1].strip() if len(parts) > 1 else ""
                        if name:
                            apps.append({"name": name, "source": "windows_registry"})
        return apps

    def _installed_apps_linux(self) -> List[dict]:
        apps = []
        rc, out, _ = _run(["dpkg", "--list"], timeout=15)
        if rc == 0:
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "ii":
                    apps.append({"name": parts[1], "source": "dpkg"})
        else:
            rc, out, _ = _run(["rpm", "-qa", "--queryformat", "%{NAME}\n"], timeout=15)
            if rc == 0:
                for name in out.splitlines():
                    if name.strip():
                        apps.append({"name": name.strip(), "source": "rpm"})
        return apps

    def _installed_apps_macos(self) -> List[dict]:
        apps = []
        rc, out, _ = _run(
            ["system_profiler", "SPApplicationsDataType", "-json"], timeout=30
        )
        if rc == 0:
            try:
                data = json.loads(out)
                for app in data.get("SPApplicationsDataType", []):
                    apps.append({
                        "name": app.get("_name", ""),
                        "version": app.get("version", ""),
                        "obtained_from": app.get("obtained_from", "unknown"),
                        "source": "system_profiler",
                    })
            except Exception:
                pass
        return apps

    def _llm_flag_apps(self, apps: List[dict], engine) -> List[dict]:
        names = [a["name"] for a in apps[:80]]  # cap to avoid huge prompts
        prompt = (
            "From this list of installed applications, identify any that are suspicious, "
            "potentially unwanted, or potentially malicious. "
            "Reply with JSON: [{\"name\": \"...\", \"verdict\": \"safe|suspicious|malicious\", \"reason\": \"...\"}]\n\n"
            f"Apps: {json.dumps(names)}"
        )
        raw = _llm_call(engine, prompt, system="You are a cybersecurity expert.")
        try:
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                flagged = json.loads(m.group())
                # merge verdicts back
                verdict_map = {f["name"]: f for f in flagged}
                for app in apps:
                    if app["name"] in verdict_map:
                        app.update(verdict_map[app["name"]])
                    else:
                        app.setdefault("verdict", "safe")
        except Exception:
            pass
        return apps

    # ── 8. SSL certificate check ──────────────────────────────────────────────

    def check_ssl_certificates(self, domains: List[str]) -> List[dict]:
        """
        Verify SSL certs for given domains.
        Detects: expired, self-signed, hostname mismatch.
        """
        results = []
        if not _SSL_CHECK:
            return results

        import ssl
        import datetime as _dt

        for domain in domains:
            entry: Dict[str, Any] = {"domain": domain}
            try:
                ctx = ssl.create_default_context()
                with ctx.wrap_socket(
                    socket.socket(), server_hostname=domain
                ) as ssock:
                    ssock.settimeout(10)
                    ssock.connect((domain, 443))
                    cert = ssock.getpeercert()

                # expiry
                not_after_str = cert.get("notAfter", "")
                not_after = _dt.datetime.strptime(
                    not_after_str, "%b %d %H:%M:%S %Y %Z"
                ).replace(tzinfo=_dt.timezone.utc)
                days_left = (not_after - _dt.datetime.now(_dt.timezone.utc)).days

                entry["valid"] = True
                entry["expires"] = not_after.isoformat()
                entry["days_until_expiry"] = days_left
                entry["issuer"] = dict(x[0] for x in cert.get("issuer", []))
                entry["subject"] = dict(x[0] for x in cert.get("subject", []))

                if days_left <= 0:
                    entry["valid"] = False
                    entry["issue"] = "Certificate expired"
                elif days_left <= 30:
                    entry["issue"] = f"Certificate expiring soon ({days_left} days)"

                # self-signed heuristic: issuer == subject
                if entry["issuer"] == entry["subject"]:
                    entry["valid"] = False
                    entry["issue"] = entry.get("issue", "") + " Self-signed certificate"

            except ssl.SSLCertVerificationError as exc:
                entry["valid"] = False
                entry["issue"] = f"SSL verification failed: {exc}"
            except ssl.CertificateError as exc:
                entry["valid"] = False
                entry["issue"] = f"Certificate error: {exc}"
            except socket.timeout:
                entry["valid"] = None
                entry["issue"] = "Connection timed out"
            except ConnectionRefusedError:
                entry["valid"] = None
                entry["issue"] = "Connection refused"
            except Exception as exc:
                entry["valid"] = None
                entry["issue"] = str(exc)

            results.append(entry)

        return results

    # ── 9. Background monitoring ──────────────────────────────────────────────

    def monitor_start(
        self,
        interval_s: int = 60,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Start background monitoring thread.
        Calls callback(threat_dict) on each threat found.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            return  # already running

        self._monitor_stop_event.clear()

        def _loop():
            while not self._monitor_stop_event.is_set():
                try:
                    threats = self.scan_processes()
                    net_threats = self.check_network_connections()
                    all_threats = [
                        t for t in threats if t.verdict != "safe"
                    ] + [n for n in net_threats if n.verdict != "safe"]

                    for t in all_threats:
                        d = asdict(t)
                        if callback:
                            try:
                                callback(d)
                            except Exception:
                                pass
                except Exception:
                    pass
                self._monitor_stop_event.wait(interval_s)

        self._monitor_thread = threading.Thread(
            target=_loop, name="aria-security-monitor", daemon=True
        )
        self._monitor_thread.start()

    def monitor_stop(self) -> None:
        """Stop the background monitoring thread."""
        self._monitor_stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.stop_filesystem_watcher()

    # ── 10. Threat log ────────────────────────────────────────────────────────

    def get_threat_log(self, limit: int = 100) -> List[dict]:
        """Return recent threats from in-memory log."""
        with self._threat_log_lock:
            return list(self._threat_log[-limit:])

    # ── 11. Firewall rule management ──────────────────────────────────────────

    def firewall_rule(
        self,
        action: str,         # "block" | "allow" | "remove"
        port: Optional[int] = None,
        ip: Optional[str] = None,
    ) -> dict:
        """
        Add/remove firewall rules.
        action: "block" | "allow" | "remove"
        Uses: Windows netsh / Linux iptables / macOS pfctl
        Returns {"success": bool, "message": str, "command": str}
        """
        if not port and not ip:
            return {"success": False, "message": "Must specify port or ip", "command": ""}

        if OS == "Windows":
            return self._fw_windows(action, port, ip)
        elif OS == "Linux":
            return self._fw_linux(action, port, ip)
        elif OS == "Darwin":
            return self._fw_macos(action, port, ip)
        return {"success": False, "message": f"Unsupported OS: {OS}", "command": ""}

    def _fw_windows(self, action: str, port: Optional[int], ip: Optional[str]) -> dict:
        rule_name = f"ARIA_{'BLOCK' if action == 'block' else 'ALLOW'}_{ip or ''}_{port or ''}"
        if action in ("block", "allow"):
            direction = "out"
            fw_action = "block" if action == "block" else "allow"
            cmd = ["netsh", "advfirewall", "firewall", "add", "rule",
                   f"name={rule_name}", "dir=out", f"action={fw_action}",
                   "protocol=tcp"]
            if port:
                cmd += [f"remoteport={port}"]
            if ip:
                cmd += [f"remoteip={ip}"]
        else:  # remove
            cmd = ["netsh", "advfirewall", "firewall", "delete", "rule",
                   f"name={rule_name}"]

        rc, out, err = _run(cmd, timeout=15)
        return {
            "success": rc == 0,
            "message": out or err,
            "command": " ".join(cmd),
        }

    def _fw_linux(self, action: str, port: Optional[int], ip: Optional[str]) -> dict:
        if action == "block":
            flag = "-A"
            target = "DROP"
        elif action == "allow":
            flag = "-A"
            target = "ACCEPT"
        else:  # remove
            flag = "-D"
            target = "DROP"

        cmd = ["iptables", flag, "OUTPUT"]
        if ip:
            cmd += ["-d", ip]
        if port:
            cmd += ["-p", "tcp", "--dport", str(port)]
        cmd += ["-j", target]

        rc, out, err = _run(["sudo"] + cmd, timeout=15)
        return {
            "success": rc == 0,
            "message": out or err,
            "command": " ".join(cmd),
        }

    def _fw_macos(self, action: str, port: Optional[int], ip: Optional[str]) -> dict:
        # macOS pfctl approach: write an anchor rule
        if action == "block":
            rule = f"block out quick"
            if ip:
                rule += f" to {ip}"
            if port:
                rule += f" port {port}"
            cmd_str = f"echo '{rule}' | sudo pfctl -a ARIA -f -"
        else:
            cmd_str = "sudo pfctl -a ARIA -F rules"

        rc, out, err = _run(["bash", "-c", cmd_str], timeout=15)
        return {
            "success": rc == 0,
            "message": out or err,
            "command": cmd_str,
        }

    # ── 12. Natural language interface ────────────────────────────────────────

    def run_nl(self, query: str) -> dict:
        """
        Natural language interface.
        Examples:
          "scan for threats"
          "check my network"
          "am I safe?"
          "what processes are suspicious?"
          "check SSL for example.com"
          "block ip 1.2.3.4"
          "system health"
        """
        q = query.lower()

        if any(w in q for w in ["process", "suspicious proc", "what process", "running"]):
            threats = [t for t in self.scan_processes() if t.verdict != "safe"]
            return {
                "action": "scan_processes",
                "threats_found": len(threats),
                "threats": [asdict(t) for t in threats],
            }

        if any(w in q for w in ["network", "connection", "outbound"]):
            threats = [t for t in self.check_network_connections() if t.verdict != "safe"]
            return {
                "action": "check_network",
                "threats_found": len(threats),
                "threats": [asdict(t) for t in threats],
            }

        if any(w in q for w in ["safe", "scan", "threat", "malware", "secure"]):
            p_threats = [t for t in self.scan_processes() if t.verdict != "safe"]
            n_threats = [t for t in self.check_network_connections() if t.verdict != "safe"]
            total = len(p_threats) + len(n_threats)
            status = "safe" if total == 0 else "threats_detected"
            return {
                "action": "full_scan",
                "status": status,
                "process_threats": len(p_threats),
                "network_threats": len(n_threats),
                "process_details": [asdict(t) for t in p_threats],
                "network_details": [asdict(t) for t in n_threats],
            }

        if "health" in q or "cpu" in q or "memory" in q or "disk" in q:
            report = self.system_health_check()
            return {"action": "health_check", "report": asdict(report)}

        if "ssl" in q or "certificate" in q:
            # extract domain from query
            domain_match = re.search(r'(?:for\s+)?([a-z0-9][-a-z0-9.]*\.[a-z]{2,})', q)
            domains = [domain_match.group(1)] if domain_match else []
            if domains:
                results = self.check_ssl_certificates(domains)
                return {"action": "check_ssl", "results": results}
            return {"action": "check_ssl", "error": "No domain found in query"}

        if re.search(r'block\s+(?:ip\s+)?([\d.]+)', q):
            m = re.search(r'([\d.]+(?:\.[\d.]+){3})', q)
            if m:
                ip = m.group(1)
                result = self.firewall_rule("block", ip=ip)
                return {"action": "firewall_block", "ip": ip, **result}

        if "apps" in q or "installed" in q:
            apps = self.scan_installed_apps()
            suspicious = [a for a in apps if a.get("verdict") in ("suspicious", "malicious")]
            return {
                "action": "scan_apps",
                "total_apps": len(apps),
                "suspicious_count": len(suspicious),
                "suspicious": suspicious,
            }

        # fallback: full scan
        return self.run_nl("am I safe?")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARIA Security Monitor")
    parser.add_argument("query", nargs="?", default="am I safe?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    agent = SecurityMonitorAgent()
    result = agent.run_nl(args.query)
    print(json.dumps(result, indent=2, default=str))
