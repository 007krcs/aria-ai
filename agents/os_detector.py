"""
ARIA OS Detector Agent
======================
Comprehensive OS detection and profiling agent.

Detects: Windows, macOS, Linux (Ubuntu, Debian, Arch, Fedora, etc.),
         Android (via ADB), iOS (via libimobiledevice/ideviceinfo),
         WSL (Windows Subsystem for Linux).

Profiles: kernel version, uptime, CPU arch, RAM, GPU, admin/root status,
          shell type, package managers, installed languages, Docker,
          virtualization.

Provides:
  - OSProfile  — dataclass with all detection results
  - OsDetectorAgent.detect()         — full profile
  - OsDetectorAgent.get_shell()      — best shell for this OS
  - OsDetectorAgent.get_terminal_cmd(cmd) — correct shell invocation
  - OsDetectorAgent.is_admin()       — elevated privilege check
  - OsDetectorAgent.connect_remote() — SSH via paramiko
  - OsDetectorAgent.connect_android()— ADB interface
  - OsDetectorAgent.run_nl(query)    — natural-language queries
"""

from __future__ import annotations

import os
import sys
import re
import time
import platform
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS — graceful fallback if not installed
# ─────────────────────────────────────────────────────────────────────────────

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import paramiko
    _PARAMIKO = True
except ImportError:
    _PARAMIKO = False


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OSProfile:
    """Complete operating system profile for ARIA."""

    # Core OS identity
    os_name: str = "unknown"          # "windows", "linux", "macos", "android", "ios"
    os_distro: str = ""               # e.g. "Ubuntu 22.04", "Fedora 38"
    os_version: str = ""              # e.g. "10.0.19045", "22.04.3"
    kernel_version: str = ""          # e.g. "5.15.0-91-generic"
    is_wsl: bool = False              # True when running inside WSL
    wsl_version: int = 0              # 1 or 2 when is_wsl is True

    # Hardware
    cpu_arch: str = ""                # "x86_64", "arm64", "aarch64", etc.
    cpu_model: str = ""               # human-readable CPU name
    cpu_count: int = 0                # logical CPU count
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_info: List[str] = field(default_factory=list)

    # Runtime context
    is_admin: bool = False
    is_virtual: bool = False          # True inside VM or container
    virtual_platform: str = ""        # "docker", "vmware", "virtualbox", "wsl", etc.
    hostname: str = ""
    username: str = ""
    home_dir: str = ""

    # Shell / terminal
    shell: str = ""                   # "bash", "zsh", "powershell", "cmd", etc.
    shell_path: str = ""
    uptime_seconds: float = 0.0

    # Package managers present on this system
    package_managers: List[str] = field(default_factory=list)
    # e.g. ["apt", "snap", "flatpak"]

    # Developer tools
    languages: Dict[str, str] = field(default_factory=dict)
    # e.g. {"python": "3.11.4", "node": "20.1.0", "rust": "1.70.0"}
    docker_installed: bool = False
    docker_running: bool = False

    # Raw extras
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AndroidDevice:
    """Represents an ADB-connected Android device."""
    device_id: str
    model: str = ""
    android_version: str = ""
    sdk_level: str = ""
    is_rooted: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: List[str] | str, shell: bool = False, timeout: int = 8) -> str:
    """Run a command, return stdout stripped, or empty string on failure."""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _which(name: str) -> Optional[str]:
    """Return full path of a binary, or None."""
    return shutil.which(name)


def _read_file(path: str) -> str:
    """Read a text file, return content or empty string."""
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read().strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# OS DETECTOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class OsDetectorAgent:
    """
    ARIA OS detection and profiling agent.

    Usage:
        agent = OsDetectorAgent()
        profile = agent.detect()
        print(profile.os_name, profile.ram_total_gb)
    """

    def __init__(self) -> None:
        self._cached_profile: Optional[OSProfile] = None

    # ── public API ──────────────────────────────────────────────────────────

    def detect(self, refresh: bool = False) -> OSProfile:
        """
        Run full OS profiling. Cached after first call unless refresh=True.

        Returns
        -------
        OSProfile
            Fully populated profile for the current system.
        """
        if self._cached_profile and not refresh:
            return self._cached_profile

        profile = OSProfile()
        self._detect_os_name(profile)
        self._detect_kernel(profile)
        self._detect_wsl(profile)
        self._detect_hardware(profile)
        self._detect_admin(profile)
        self._detect_virtualization(profile)
        self._detect_shell(profile)
        self._detect_uptime(profile)
        self._detect_package_managers(profile)
        self._detect_languages(profile)
        self._detect_docker(profile)
        self._detect_gpu(profile)
        self._detect_identity(profile)

        self._cached_profile = profile
        return profile

    def get_shell(self) -> str:
        """
        Return the best interactive shell available for this OS.

        Returns
        -------
        str
            Shell name: "powershell", "bash", "zsh", "sh", "cmd".
        """
        profile = self.detect()
        if profile.os_name == "windows" and not profile.is_wsl:
            if _which("pwsh"):
                return "pwsh"
            if _which("powershell"):
                return "powershell"
            return "cmd"
        # Linux / macOS / WSL
        if _which("zsh"):
            return "zsh"
        if _which("bash"):
            return "bash"
        return "sh"

    def get_terminal_cmd(self, command: str) -> str:
        """
        Wrap *command* in the appropriate shell invocation for the current OS.

        Parameters
        ----------
        command : str
            The command or script to run.

        Returns
        -------
        str
            Full shell invocation string ready to pass to subprocess.
        """
        profile = self.detect()
        if profile.os_name == "windows" and not profile.is_wsl:
            shell = self.get_shell()
            if shell in ("pwsh", "powershell"):
                escaped = command.replace('"', '\\"')
                return f'{shell} -NoProfile -NonInteractive -Command "{escaped}"'
            # cmd.exe
            return f'cmd /c "{command}"'
        # POSIX
        shell = self.get_shell()
        return f'{shell} -c "{command}"'

    def is_admin(self) -> bool:
        """
        Check whether the current process has elevated/root/admin privileges.

        Returns
        -------
        bool
        """
        return self._check_admin()

    def connect_remote(
        self,
        host: str,
        port: int = 22,
        user: str = "",
        key_path: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Any:
        """
        Open an SSH connection to a remote host using paramiko.

        Parameters
        ----------
        host : str
        port : int
        user : str
        key_path : str, optional
            Path to a private key file.
        password : str, optional
            Password (used when key_path is None).

        Returns
        -------
        paramiko.SSHClient or None
            Connected client, or None if paramiko is unavailable / connection fails.
        """
        if not _PARAMIKO:
            print("[OsDetector] paramiko not installed — pip install paramiko")
            return None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            connect_kwargs: Dict[str, Any] = {
                "hostname": host,
                "port": port,
                "username": user or os.getlogin(),
            }
            if key_path:
                connect_kwargs["key_filename"] = key_path
            elif password:
                connect_kwargs["password"] = password
            client.connect(**connect_kwargs)
            return client
        except Exception as exc:
            print(f"[OsDetector] SSH connection failed: {exc}")
            return None

    def connect_android(self, device_id: Optional[str] = None) -> Optional[AndroidDevice]:
        """
        Connect to an Android device via ADB and return its profile.

        Parameters
        ----------
        device_id : str, optional
            Specific device serial. Uses first available if omitted.

        Returns
        -------
        AndroidDevice or None
        """
        adb = _which("adb")
        if not adb:
            print("[OsDetector] adb not found — install Android platform-tools")
            return None

        # List devices
        raw = _run([adb, "devices"])
        lines = [ln for ln in raw.splitlines()[1:] if ln.strip() and "offline" not in ln]
        if not lines:
            print("[OsDetector] No ADB devices connected")
            return None

        # Pick device
        if device_id is None:
            device_id = lines[0].split()[0]

        def _adb_prop(prop: str) -> str:
            return _run([adb, "-s", device_id, "shell", "getprop", prop])

        model = _adb_prop("ro.product.model")
        android_version = _adb_prop("ro.build.version.release")
        sdk_level = _adb_prop("ro.build.version.sdk")

        # Rough root check
        su_check = _run([adb, "-s", device_id, "shell", "su", "-c", "id"])
        is_rooted = "uid=0" in su_check

        return AndroidDevice(
            device_id=device_id,
            model=model,
            android_version=android_version,
            sdk_level=sdk_level,
            is_rooted=is_rooted,
        )

    def run_nl(self, query: str) -> str:
        """
        Natural-language interface to the OS detector.

        Recognised intents (case-insensitive):
          "what OS am I running?"
          "am I admin?" / "am I root?" / "do I have admin?"
          "what shell am I using?"
          "what CPU do I have?"
          "how much RAM do I have?"
          "is docker running?"
          "what package managers are available?"
          "what languages are installed?"
          "am I in WSL?"
          "what GPU do I have?"
          "what is my hostname?"
          "how long has this system been running?" / "uptime"

        Parameters
        ----------
        query : str
            Plain-English question.

        Returns
        -------
        str
            Human-readable answer.
        """
        q = query.lower().strip()
        p = self.detect()

        if any(kw in q for kw in ("os", "operating system", "platform", "distro")):
            distro = f" ({p.os_distro})" if p.os_distro else ""
            wsl_note = " [running inside WSL]" if p.is_wsl else ""
            return f"{p.os_name.title()}{distro}, version {p.os_version}{wsl_note}"

        if any(kw in q for kw in ("admin", "root", "elevated", "sudo", "privilege")):
            status = "Yes, you have elevated/admin privileges." if p.is_admin else "No, you are running as a standard user."
            return status

        if "shell" in q:
            return f"Current shell: {p.shell} ({p.shell_path})"

        if "cpu" in q or "processor" in q or "arch" in q:
            return f"{p.cpu_model} — {p.cpu_arch}, {p.cpu_count} logical cores"

        if "ram" in q or "memory" in q:
            return (
                f"Total RAM: {p.ram_total_gb:.1f} GB | "
                f"Available: {p.ram_available_gb:.1f} GB"
            )

        if "docker" in q:
            if not p.docker_installed:
                return "Docker is not installed on this system."
            running = "running" if p.docker_running else "installed but NOT running"
            return f"Docker is {running}."

        if any(kw in q for kw in ("package manager", "apt", "brew", "winget", "choco")):
            if not p.package_managers:
                return "No recognised package managers found."
            return "Available package managers: " + ", ".join(p.package_managers)

        if any(kw in q for kw in ("language", "python", "node", "java", "rust", "go")):
            if not p.languages:
                return "No recognised language runtimes detected."
            parts = [f"{lang} {ver}" for lang, ver in p.languages.items()]
            return "Installed language runtimes: " + ", ".join(parts)

        if "wsl" in q:
            if p.is_wsl:
                return f"Yes, you are running inside WSL (version {p.wsl_version})."
            return "No, you are not running inside WSL."

        if "gpu" in q or "graphics" in q:
            if not p.gpu_info:
                return "No GPU information found."
            return "GPU(s): " + "; ".join(p.gpu_info)

        if "hostname" in q or "computer name" in q or "machine name" in q:
            return f"Hostname: {p.hostname}"

        if "uptime" in q or "running" in q or "how long" in q:
            secs = p.uptime_seconds
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            return f"System uptime: {h}h {m}m"

        if "virtual" in q or "vm" in q or "container" in q:
            if p.is_virtual:
                return f"Yes, running inside a virtual environment: {p.virtual_platform}"
            return "No, this appears to be bare-metal hardware."

        if "username" in q or "user" in q or "who am i" in q:
            return f"Username: {p.username}"

        return (
            f"OS: {p.os_name.title()} {p.os_distro} {p.os_version} | "
            f"Arch: {p.cpu_arch} | RAM: {p.ram_total_gb:.1f} GB | "
            f"Admin: {p.is_admin} | Shell: {p.shell}"
        )

    # ── internal detection methods ───────────────────────────────────────────

    def _detect_os_name(self, p: OSProfile) -> None:
        """Populate os_name, os_distro, os_version."""
        system = platform.system().lower()

        if system == "windows":
            p.os_name = "windows"
            p.os_version = platform.version()
            release = platform.release()
            p.os_distro = f"Windows {release}"
            return

        if system == "darwin":
            p.os_name = "macos"
            mac_ver = platform.mac_ver()[0]
            p.os_version = mac_ver
            p.os_distro = f"macOS {mac_ver}"
            return

        if system == "linux":
            p.os_name = "linux"
            p.os_version = platform.release()

            # Try /etc/os-release for distro name
            os_release = _read_file("/etc/os-release")
            if os_release:
                name_match = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', os_release, re.M)
                if name_match:
                    p.os_distro = name_match.group(1)
                    return
            # Fallback: lsb_release
            lsb = _run(["lsb_release", "-d"])
            if lsb:
                p.os_distro = lsb.replace("Description:", "").strip()
                return
            p.os_distro = "Linux"
            return

        # Android via ADB (running from desktop connected to device)
        if _which("adb"):
            devices = _run(["adb", "devices"])
            if "device" in devices and len(devices.splitlines()) > 1:
                p.os_name = "android"
                return

        p.os_name = system or "unknown"

    def _detect_kernel(self, p: OSProfile) -> None:
        """Populate kernel_version."""
        try:
            p.kernel_version = platform.release()
            if p.os_name == "windows":
                p.kernel_version = platform.version()
        except Exception:
            p.kernel_version = ""

    def _detect_wsl(self, p: OSProfile) -> None:
        """Detect WSL 1 / WSL 2."""
        if p.os_name != "linux":
            return

        # Check /proc/version for Microsoft string
        proc_version = _read_file("/proc/version")
        if "microsoft" in proc_version.lower() or "wsl" in proc_version.lower():
            p.is_wsl = True
            # Distinguish WSL1 vs WSL2: WSL2 has its own kernel, WSL1 shares Windows kernel
            # /proc/sys/fs/binfmt_misc/WSLInterop only exists in WSL1
            if os.path.exists("/proc/sys/kernel/osrelease"):
                osrelease = _read_file("/proc/sys/kernel/osrelease")
                p.wsl_version = 2 if "microsoft" in osrelease.lower() else 1
            else:
                p.wsl_version = 1
            p.virtual_platform = f"WSL{p.wsl_version}"
            p.is_virtual = True
            return

        # Also check WSL_DISTRO_NAME env var
        if os.environ.get("WSL_DISTRO_NAME"):
            p.is_wsl = True
            p.wsl_version = 2
            p.virtual_platform = "WSL2"
            p.is_virtual = True

    def _detect_hardware(self, p: OSProfile) -> None:
        """Populate CPU arch, CPU model, core count, RAM."""
        p.cpu_arch = platform.machine() or _run(["uname", "-m"])

        # CPU model
        if p.os_name == "windows":
            p.cpu_model = platform.processor()
        elif p.os_name == "macos":
            p.cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        else:
            # Linux
            cpuinfo = _read_file("/proc/cpuinfo")
            match = re.search(r"model name\s*:\s*(.+)", cpuinfo, re.I)
            p.cpu_model = match.group(1).strip() if match else platform.processor()

        # Core count
        if _PSUTIL:
            p.cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 0
        else:
            p.cpu_count = os.cpu_count() or 0

        # RAM
        if _PSUTIL:
            mem = psutil.virtual_memory()
            p.ram_total_gb = round(mem.total / (1024 ** 3), 2)
            p.ram_available_gb = round(mem.available / (1024 ** 3), 2)
        else:
            p.ram_total_gb = self._fallback_ram_total()
            p.ram_available_gb = 0.0

    def _fallback_ram_total(self) -> float:
        """RAM total without psutil."""
        if os.name == "nt":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return round(stat.ullTotalPhys / (1024 ** 3), 2)
            except Exception:
                pass
        # Linux
        meminfo = _read_file("/proc/meminfo")
        match = re.search(r"MemTotal:\s+(\d+)", meminfo)
        if match:
            return round(int(match.group(1)) / (1024 ** 2), 2)
        return 0.0

    def _detect_gpu(self, p: OSProfile) -> None:
        """Populate gpu_info list."""
        gpus: List[str] = []

        if p.os_name == "windows":
            out = _run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                shell=False,
            )
            for line in out.splitlines():
                line = line.strip()
                if line and line.lower() != "name":
                    gpus.append(line)

        elif p.os_name == "macos":
            out = _run(["system_profiler", "SPDisplaysDataType"])
            for line in out.splitlines():
                if "Chipset Model" in line or "Model" in line:
                    gpus.append(line.split(":", 1)[-1].strip())

        else:
            # Try lspci
            out = _run(["lspci"])
            for line in out.splitlines():
                if re.search(r"VGA|3D|Display", line, re.I):
                    gpus.append(line.split(":", 2)[-1].strip())

        # nvidia-smi supplement
        if _which("nvidia-smi"):
            ns = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
            for name in ns.splitlines():
                name = name.strip()
                if name and name not in gpus:
                    gpus.append(name)

        p.gpu_info = gpus

    def _check_admin(self) -> bool:
        """Return True when running with admin/root privileges."""
        try:
            if os.name == "nt":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore[attr-defined]
            return os.geteuid() == 0  # type: ignore[attr-defined]
        except Exception:
            return False

    def _detect_admin(self, p: OSProfile) -> None:
        p.is_admin = self._check_admin()

    def _detect_virtualization(self, p: OSProfile) -> None:
        """Detect VM/container environment (unless already set by WSL detection)."""
        if p.is_virtual:
            return

        # Docker: presence of /.dockerenv
        if os.path.exists("/.dockerenv"):
            p.is_virtual = True
            p.virtual_platform = "docker"
            return

        # systemd-detect-virt
        virt = _run(["systemd-detect-virt"])
        if virt and virt not in ("none", ""):
            p.is_virtual = True
            p.virtual_platform = virt
            return

        # /proc/cpuinfo hypervisor flag
        cpuinfo = _read_file("/proc/cpuinfo")
        if "hypervisor" in cpuinfo.lower():
            p.is_virtual = True
            # Try to guess hypervisor from dmidecode
            dmi = _run(["dmidecode", "-s", "system-product-name"])
            p.virtual_platform = dmi.lower() if dmi else "hypervisor"
            return

        # Windows: check for known VM product names via WMI
        if p.os_name == "windows":
            out = _run(
                ["wmic", "computersystem", "get", "model"],
                shell=False,
            ).lower()
            for vm in ("virtual", "vmware", "virtualbox", "hyper-v", "kvm", "qemu", "xen"):
                if vm in out:
                    p.is_virtual = True
                    p.virtual_platform = vm
                    return

    def _detect_shell(self, p: OSProfile) -> None:
        """Populate shell and shell_path."""
        # $SHELL env var works on POSIX
        shell_env = os.environ.get("SHELL", "")
        if shell_env:
            p.shell_path = shell_env
            p.shell = os.path.basename(shell_env)
            return

        # Windows: check PSModulePath or COMSPEC
        if p.os_name == "windows" and not p.is_wsl:
            if "WindowsPowerShell" in os.environ.get("PSModulePath", "") or _which("powershell"):
                p.shell = "powershell"
                p.shell_path = _which("powershell") or "powershell"
                return
            p.shell = "cmd"
            p.shell_path = os.environ.get("COMSPEC", "cmd.exe")
            return

        # Fallback: which bash / zsh / sh
        for sh in ("zsh", "bash", "sh"):
            path = _which(sh)
            if path:
                p.shell = sh
                p.shell_path = path
                return

        p.shell = "sh"
        p.shell_path = "/bin/sh"

    def _detect_uptime(self, p: OSProfile) -> None:
        """Populate uptime_seconds."""
        if _PSUTIL:
            try:
                p.uptime_seconds = time.time() - psutil.boot_time()
                return
            except Exception:
                pass

        if p.os_name == "linux" or p.is_wsl:
            raw = _read_file("/proc/uptime")
            if raw:
                try:
                    p.uptime_seconds = float(raw.split()[0])
                    return
                except ValueError:
                    pass

        if p.os_name == "macos":
            out = _run(["sysctl", "-n", "kern.boottime"])
            match = re.search(r"sec\s*=\s*(\d+)", out)
            if match:
                boot_epoch = int(match.group(1))
                p.uptime_seconds = time.time() - boot_epoch
                return

        if p.os_name == "windows":
            out = _run(["net", "statistics", "workstation"])
            match = re.search(r"Statistics since (.+)", out)
            if match:
                from datetime import datetime
                try:
                    boot = datetime.strptime(match.group(1).strip(), "%m/%d/%Y %I:%M:%S %p")
                    p.uptime_seconds = (datetime.now() - boot).total_seconds()
                    return
                except Exception:
                    pass

    def _detect_package_managers(self, p: OSProfile) -> None:
        """Detect available package managers."""
        candidates = {
            "apt":      "apt",
            "apt-get":  "apt-get",
            "brew":     "brew",
            "pacman":   "pacman",
            "yum":      "yum",
            "dnf":      "dnf",
            "zypper":   "zypper",
            "winget":   "winget",
            "choco":    "choco",
            "scoop":    "scoop",
            "snap":     "snap",
            "flatpak":  "flatpak",
            "nix":      "nix",
            "emerge":   "emerge",     # Gentoo
            "pkg":      "pkg",        # FreeBSD / Termux
        }
        found: List[str] = []
        for name, binary in candidates.items():
            if _which(binary):
                found.append(name)
        p.package_managers = found

    def _detect_languages(self, p: OSProfile) -> None:
        """Detect installed language runtimes and their versions."""
        runtimes: Dict[str, str] = {}

        checks: Dict[str, List[str]] = {
            "python":  ["python3", "--version"],
            "python2": ["python2", "--version"],
            "node":    ["node", "--version"],
            "npm":     ["npm", "--version"],
            "java":    ["java", "-version"],
            "rust":    ["rustc", "--version"],
            "go":      ["go", "version"],
            "ruby":    ["ruby", "--version"],
            "perl":    ["perl", "--version"],
            "php":     ["php", "--version"],
            "dotnet":  ["dotnet", "--version"],
            "swift":   ["swift", "--version"],
            "kotlin":  ["kotlinc", "-version"],
            "scala":   ["scala", "-version"],
            "r":       ["Rscript", "--version"],
        }

        for name, cmd in checks.items():
            binary = cmd[0]
            if not _which(binary):
                continue
            raw = _run(cmd)
            if not raw:
                # Some tools (java) write to stderr
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    raw = result.stderr.strip() or result.stdout.strip()
                except Exception:
                    raw = ""
            # Extract version number
            ver_match = re.search(r"(\d+\.\d+[\.\d]*)", raw)
            runtimes[name] = ver_match.group(1) if ver_match else raw.split("\n")[0][:40]

        p.languages = runtimes

    def _detect_docker(self, p: OSProfile) -> None:
        """Check Docker installation and daemon status."""
        if not _which("docker"):
            p.docker_installed = False
            p.docker_running = False
            return
        p.docker_installed = True
        # Check if daemon is up
        out = _run(["docker", "info"], timeout=5)
        p.docker_running = "Server Version" in out or "server" in out.lower()

    def _detect_identity(self, p: OSProfile) -> None:
        """Populate hostname, username, home_dir."""
        import socket
        try:
            p.hostname = socket.gethostname()
        except Exception:
            p.hostname = os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", ""))

        try:
            p.username = os.getlogin()
        except Exception:
            p.username = os.environ.get("USER", os.environ.get("USERNAME", ""))

        p.home_dir = str(os.path.expanduser("~"))


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE
# ─────────────────────────────────────────────────────────────────────────────

_default_agent: Optional[OsDetectorAgent] = None


def get_profile() -> OSProfile:
    """Module-level shortcut: returns a cached OSProfile."""
    global _default_agent
    if _default_agent is None:
        _default_agent = OsDetectorAgent()
    return _default_agent.detect()


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import asdict
    import json

    agent = OsDetectorAgent()
    profile = agent.detect()
    print(json.dumps(asdict(profile), indent=2, default=str))

    print("\n--- NL queries ---")
    for q in [
        "what OS am I running?",
        "am I admin?",
        "what shell am I using?",
        "how much RAM do I have?",
        "what package managers are available?",
        "what languages are installed?",
        "is docker running?",
        "am I in WSL?",
        "what GPU do I have?",
        "how long has this system been running?",
    ]:
        print(f"Q: {q}")
        print(f"A: {agent.run_nl(q)}\n")
