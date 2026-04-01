"""
ARIA Windows Kernel Agent
==========================
Full Windows system control — works like Siri/Alexa but deeper.

Capabilities:
  - Win+R launcher (any Run dialog command)
  - Application open/close/focus by name or path
  - Chrome/Edge/Brave profile selection
  - PowerShell, CMD, Bash (WSL), Git script execution
  - Registry read (read-only by default, write with confirm)
  - Java project analyzer — detects Maven/Gradle/pom.xml/build.gradle,
    reads dependencies, checks JDK version, flags missing tools
  - Polyglot code writer — Python, Java, JS, TS, Go, Rust, C++, etc.
  - Scheduled task creation via Task Scheduler
  - System tray / notification level control
  - File search (Everything-style fallback to glob)

All DANGEROUS actions (registry write, scheduled tasks, installs) require
user confirmation before execution.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import winreg
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import ctypes
    from ctypes import wintypes
    CTYPES_OK = True
except ImportError:
    CTYPES_OK = False

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ShellResult:
    ok:         bool
    output:     str = ""
    error:      str = ""
    returncode: int = 0
    duration:   float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "returncode": self.returncode,
            "duration": round(self.duration, 3),
        }


@dataclass
class JavaProjectInfo:
    path:             str
    build_system:     str           = "unknown"  # maven | gradle | ant | none
    jdk_version:      str           = ""
    main_class:       str           = ""
    dependencies:     List[str]     = field(default_factory=list)
    missing_tools:    List[str]     = field(default_factory=list)
    suggested_commands: List[str]   = field(default_factory=list)
    warnings:         List[str]     = field(default_factory=list)


# ── Chrome profile detection ──────────────────────────────────────────────────

CHROME_PROFILE_DIRS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Edge" / "User Data",
    Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware" / "Brave-Browser" / "User Data",
]

CHROME_BROWSERS = {
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ],
    "edge": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
    "brave": [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    ],
}


def _find_browser_exe(name: str) -> Optional[str]:
    for path in CHROME_BROWSERS.get(name, []):
        if Path(path).exists():
            return path
    exe = shutil.which(name)
    return exe


def list_chrome_profiles(browser: str = "chrome") -> List[Dict[str, str]]:
    """Return list of Chrome/Edge/Brave profiles with name, email, avatar.

    Reads both:
      - Local State (master profile registry)
      - Individual profile Preferences (fallback)
    """
    profiles = []
    seen_folders: set = set()

    for profile_root in CHROME_PROFILE_DIRS:
        if not profile_root.exists():
            continue

        # ── Primary: read Local State for the authoritative profile list ──
        local_state_file = profile_root / "Local State"
        local_state_profiles: Dict[str, Dict] = {}
        if local_state_file.exists():
            try:
                ls = json.loads(local_state_file.read_text(encoding="utf-8", errors="replace"))
                local_state_profiles = ls.get("profile", {}).get("info_cache", {})
            except Exception:
                pass

        # Walk all profile folders
        for entry in profile_root.iterdir():
            if not entry.is_dir():
                continue
            # Accept Default + Profile N + any folder that has a Preferences file
            prefs_file = entry / "Preferences"
            if not prefs_file.exists():
                continue
            folder = entry.name
            if folder in seen_folders:
                continue
            seen_folders.add(folder)

            # Name from Local State first (most accurate)
            ls_info = local_state_profiles.get(folder, {})
            name    = ls_info.get("name", "") or ls_info.get("user_name", "")
            email   = ls_info.get("user_name", "") if "@" in ls_info.get("user_name", "") else ""
            avatar  = ls_info.get("last_downloaded_gaia_picture_url_with_size", "")
            shortcut_name = ls_info.get("shortcut_name", "")

            # Fallback to Preferences
            if not name:
                try:
                    prefs = json.loads(prefs_file.read_text(encoding="utf-8", errors="replace"))
                    name  = prefs.get("profile", {}).get("name", folder)
                    accs  = prefs.get("account_info", [{}])
                    if not email and accs:
                        email = accs[0].get("email", "")
                except Exception:
                    name = folder

            profiles.append({
                "folder":        folder,
                "name":          name or folder,
                "email":         email,
                "shortcut_name": shortcut_name,
                "avatar_url":    avatar,
                "path":          str(entry),
            })

    # Sort: Default first, then alphabetically by name
    profiles.sort(key=lambda p: (0 if p["folder"] == "Default" else 1, p["name"].lower()))
    return profiles


def find_profile_by_name(name_query: str, browser: str = "chrome") -> Optional[Dict[str, str]]:
    """
    Fuzzy match a profile by display name, email, or shortcut name.
    Used for NL commands like 'open Chrome as Chandan' or 'open Kamla's Chrome'.
    Returns the best matching profile dict, or None.
    """
    profiles = list_chrome_profiles(browser)
    if not profiles:
        return None

    q = name_query.lower().strip()

    # Exact match first
    for p in profiles:
        if (p["name"].lower() == q or
                p["email"].lower() == q or
                p["shortcut_name"].lower() == q):
            return p

    # Substring match
    for p in profiles:
        if (q in p["name"].lower() or
                q in p["email"].lower() or
                q in p["shortcut_name"].lower()):
            return p

    # Word overlap (handle "Chandan Singh" vs "chandan")
    q_words = set(q.split())
    best, best_score = None, 0
    for p in profiles:
        p_words = set(p["name"].lower().split()) | set(p["email"].lower().split())
        score = len(q_words & p_words)
        if score > best_score:
            best, best_score = p, score

    return best if best_score > 0 else None


def open_chrome_with_profile(
    url: str = "",
    profile_folder: str = "Default",
    browser: str = "chrome",
) -> ShellResult:
    """Open Chrome/Edge/Brave with a specific profile."""
    exe = _find_browser_exe(browser)
    if not exe:
        return ShellResult(ok=False, error=f"{browser} executable not found")

    # Find user-data-dir
    user_data_dir = None
    for d in CHROME_PROFILE_DIRS:
        candidate = d / profile_folder
        if candidate.parent.exists():
            user_data_dir = str(d)
            break

    args = [exe, f"--profile-directory={profile_folder}"]
    if user_data_dir:
        args.append(f"--user-data-dir={user_data_dir}")
    if url:
        args.append(url)

    try:
        subprocess.Popen(args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                         if sys.platform == "win32" else 0)
        return ShellResult(ok=True, output=f"Opened {browser} with profile '{profile_folder}'" +
                           (f" at {url}" if url else ""))
    except Exception as e:
        return ShellResult(ok=False, error=str(e))


# ── Win+R launcher ────────────────────────────────────────────────────────────

WIN_R_COMMON_COMMANDS = {
    "task manager":    "taskmgr",
    "taskmgr":         "taskmgr",
    "control panel":   "control",
    "device manager":  "devmgmt.msc",
    "registry":        "regedit",
    "services":        "services.msc",
    "environment":     "sysdm.cpl",
    "system restore":  "rstrui",
    "msconfig":        "msconfig",
    "event viewer":    "eventvwr",
    "disk management": "diskmgmt.msc",
    "calculator":      "calc",
    "notepad":         "notepad",
    "paint":           "mspaint",
    "wordpad":         "write",
    "cmd":             "cmd",
    "powershell":      "powershell",
    "explorer":        "explorer",
    "run":             None,  # literal
}


def win_r_launch(command: str) -> ShellResult:
    """
    Simulate Win+R → Run dialog with any command.
    Resolves common aliases first.
    """
    if sys.platform != "win32":
        return ShellResult(ok=False, error="Win+R only available on Windows")

    resolved = WIN_R_COMMON_COMMANDS.get(command.lower().strip(), command)
    if resolved is None:
        resolved = command

    t0 = time.time()
    try:
        # Use ShellExecute to mimic the Run dialog exactly
        if CTYPES_OK:
            ctypes.windll.shell32.ShellExecuteW(
                None, "open", resolved, None, None, 1
            )
            return ShellResult(ok=True,
                               output=f"Launched '{resolved}' via Win+R",
                               duration=time.time() - t0)
        else:
            subprocess.Popen(
                resolved, shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            return ShellResult(ok=True, output=f"Launched '{resolved}'",
                               duration=time.time() - t0)
    except Exception as e:
        return ShellResult(ok=False, error=str(e), duration=time.time() - t0)


# ── Shell execution ───────────────────────────────────────────────────────────

def run_powershell(
    script: str,
    *,
    timeout: int = 30,
    as_admin: bool = False,
) -> ShellResult:
    """Execute PowerShell script content."""
    t0 = time.time()
    if sys.platform != "win32":
        return ShellResult(ok=False, error="PowerShell is Windows-only")

    try:
        # Write to temp file to avoid escaping issues
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ps1", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            ps1_path = f.name

        cmd = ["powershell.exe", "-NoProfile", "-NonInteractive",
               "-ExecutionPolicy", "Bypass", "-File", ps1_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        duration = time.time() - t0
        os.unlink(ps1_path)
        return ShellResult(
            ok=result.returncode == 0,
            output=result.stdout.strip(),
            error=result.stderr.strip(),
            returncode=result.returncode,
            duration=duration,
        )
    except subprocess.TimeoutExpired:
        return ShellResult(ok=False, error=f"Timed out after {timeout}s",
                           duration=time.time() - t0)
    except Exception as e:
        return ShellResult(ok=False, error=str(e), duration=time.time() - t0)


def run_cmd(command: str, *, timeout: int = 30, cwd: Optional[str] = None) -> ShellResult:
    """Execute a Windows CMD command."""
    t0 = time.time()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            encoding="utf-8",
            errors="replace",
        )
        return ShellResult(
            ok=result.returncode == 0,
            output=result.stdout.strip(),
            error=result.stderr.strip(),
            returncode=result.returncode,
            duration=time.time() - t0,
        )
    except subprocess.TimeoutExpired:
        return ShellResult(ok=False, error=f"Timed out after {timeout}s",
                           duration=time.time() - t0)
    except Exception as e:
        return ShellResult(ok=False, error=str(e), duration=time.time() - t0)


def run_bash(script: str, *, timeout: int = 30, cwd: Optional[str] = None) -> ShellResult:
    """
    Execute bash via WSL if available, fallback to Git Bash, then sh.
    """
    t0 = time.time()
    bash_exe = None

    # Try WSL
    if sys.platform == "win32":
        wsl = shutil.which("wsl")
        if wsl:
            try:
                result = subprocess.run(
                    ["wsl", "bash", "-c", script],
                    capture_output=True, text=True,
                    timeout=timeout, cwd=cwd,
                    encoding="utf-8", errors="replace",
                )
                return ShellResult(
                    ok=result.returncode == 0,
                    output=result.stdout.strip(),
                    error=result.stderr.strip(),
                    returncode=result.returncode,
                    duration=time.time() - t0,
                )
            except Exception:
                pass

        # Try Git Bash
        git_bash_paths = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]
        for p in git_bash_paths:
            if Path(p).exists():
                bash_exe = p
                break

    else:
        bash_exe = shutil.which("bash")

    if bash_exe:
        try:
            result = subprocess.run(
                [bash_exe, "-c", script],
                capture_output=True, text=True,
                timeout=timeout, cwd=cwd,
                encoding="utf-8", errors="replace",
            )
            return ShellResult(
                ok=result.returncode == 0,
                output=result.stdout.strip(),
                error=result.stderr.strip(),
                returncode=result.returncode,
                duration=time.time() - t0,
            )
        except Exception as e:
            return ShellResult(ok=False, error=str(e), duration=time.time() - t0)

    return ShellResult(ok=False, error="bash/WSL not found. Install WSL or Git Bash.")


def run_git(command: str, *, cwd: Optional[str] = None, timeout: int = 30) -> ShellResult:
    """Execute a git command (without the 'git' prefix)."""
    t0 = time.time()
    git = shutil.which("git")
    if not git:
        return ShellResult(ok=False, error="git not found in PATH")

    full_cmd = f"git {command}"
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True,
            text=True, timeout=timeout, cwd=cwd,
            encoding="utf-8", errors="replace",
        )
        return ShellResult(
            ok=result.returncode == 0,
            output=result.stdout.strip(),
            error=result.stderr.strip(),
            returncode=result.returncode,
            duration=time.time() - t0,
        )
    except subprocess.TimeoutExpired:
        return ShellResult(ok=False, error=f"git timed out after {timeout}s",
                           duration=time.time() - t0)
    except Exception as e:
        return ShellResult(ok=False, error=str(e), duration=time.time() - t0)


# ── Java project analyzer ─────────────────────────────────────────────────────

_POM_DEP_RE  = re.compile(
    r"<dependency>\s*"
    r"<groupId>(.*?)</groupId>\s*"
    r"<artifactId>(.*?)</artifactId>",
    re.DOTALL,
)
_GRADLE_DEP_RE = re.compile(
    r'(?:implementation|compile|testImplementation|api)\s*["\']([^"\']+)["\']'
)


def analyze_java_project(project_path: str) -> JavaProjectInfo:
    """
    Scan a Java project directory and return structured info about:
    - build system (Maven/Gradle/Ant/none)
    - JDK version required
    - dependencies
    - missing tools (jdk, mvn, gradle)
    - suggested commands to build/run
    """
    root = Path(project_path).resolve()
    info = JavaProjectInfo(path=str(root))

    # Detect build system
    if (root / "pom.xml").exists():
        info.build_system = "maven"
    elif (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        info.build_system = "gradle"
    elif (root / "build.xml").exists():
        info.build_system = "ant"
    elif list(root.rglob("*.java")):
        info.build_system = "none"

    # Check JDK version requirement
    if info.build_system == "maven":
        pom = (root / "pom.xml").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"<java\.version>([\d.]+)</java\.version>", pom)
        if not m:
            m = re.search(r"<source>([\d.]+)</source>", pom)
        info.jdk_version = m.group(1) if m else ""

        # Extract dependencies
        for g, a in _POM_DEP_RE.findall(pom):
            info.dependencies.append(f"{g.strip()}:{a.strip()}")

        # Detect main class
        mc = re.search(r"<mainClass>(.*?)</mainClass>", pom)
        info.main_class = mc.group(1).strip() if mc else ""
        info.suggested_commands = ["mvn clean install", "mvn package", "mvn spring-boot:run"]

    elif info.build_system == "gradle":
        build_file = root / "build.gradle"
        if not build_file.exists():
            build_file = root / "build.gradle.kts"
        content = build_file.read_text(encoding="utf-8", errors="replace")

        # Java version
        m = re.search(r"sourceCompatibility\s*[=:]\s*['\"]?(\d+)['\"]?", content)
        if not m:
            m = re.search(r"JavaVersion\.VERSION_(\d+)", content)
        info.jdk_version = m.group(1) if m else ""

        info.dependencies = _GRADLE_DEP_RE.findall(content)
        info.suggested_commands = ["./gradlew build", "./gradlew run", "./gradlew test"]

    elif info.build_system == "ant":
        info.suggested_commands = ["ant compile", "ant jar", "ant run"]

    else:
        # Plain Java
        java_files = list(root.rglob("*.java"))
        if java_files:
            info.suggested_commands = [
                f"javac {java_files[0].name}",
                f"java {java_files[0].stem}",
            ]

    # Check for missing tools
    required = {"java": "JDK 17+ (https://adoptium.net)"}
    if info.build_system == "maven":
        required["mvn"] = "Apache Maven (https://maven.apache.org)"
    elif info.build_system == "gradle":
        required["gradle"] = "Gradle Build Tool (https://gradle.org)"

    for tool, install_hint in required.items():
        if not shutil.which(tool):
            info.missing_tools.append(tool)
            info.warnings.append(f"'{tool}' not in PATH — install {install_hint}")

    return info


# ── Polyglot code writer ──────────────────────────────────────────────────────

LANGUAGE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "python": {
        "ext": ".py",
        "runner": "python",
        "hello": "print('Hello from ARIA!')",
        "comment": "#",
    },
    "java": {
        "ext": ".java",
        "runner": "java",
        "hello": 'public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello from ARIA!");\n    }\n}',
        "comment": "//",
    },
    "javascript": {
        "ext": ".js",
        "runner": "node",
        "hello": "console.log('Hello from ARIA!');",
        "comment": "//",
    },
    "typescript": {
        "ext": ".ts",
        "runner": "ts-node",
        "hello": "console.log('Hello from ARIA!');",
        "comment": "//",
    },
    "go": {
        "ext": ".go",
        "runner": "go run",
        "hello": 'package main\nimport "fmt"\nfunc main() {\n    fmt.Println("Hello from ARIA!")\n}',
        "comment": "//",
    },
    "rust": {
        "ext": ".rs",
        "runner": "rustc",
        "hello": 'fn main() {\n    println!("Hello from ARIA!");\n}',
        "comment": "//",
    },
    "cpp": {
        "ext": ".cpp",
        "runner": "g++",
        "hello": '#include<iostream>\nint main(){\n    std::cout<<"Hello from ARIA!"<<std::endl;\n    return 0;\n}',
        "comment": "//",
    },
    "bash": {
        "ext": ".sh",
        "runner": "bash",
        "hello": "#!/bin/bash\necho 'Hello from ARIA!'",
        "comment": "#",
    },
    "powershell": {
        "ext": ".ps1",
        "runner": "powershell",
        "hello": "Write-Host 'Hello from ARIA!'",
        "comment": "#",
    },
    "sql": {
        "ext": ".sql",
        "runner": None,
        "hello": "SELECT 'Hello from ARIA!' AS message;",
        "comment": "--",
    },
}

LANG_ALIASES: Dict[str, str] = {
    "py":   "python",
    "js":   "javascript",
    "ts":   "typescript",
    "c++":  "cpp",
    "c":    "cpp",
    "rs":   "rust",
    "sh":   "bash",
    "ps1":  "powershell",
    "ps":   "powershell",
}


def detect_language(query: str) -> str:
    """Best-effort language detection from a natural language query."""
    q = query.lower()
    for lang in LANGUAGE_TEMPLATES:
        if lang in q:
            return lang
    for alias, lang in LANG_ALIASES.items():
        if alias in q.split():
            return lang
    # Check for common keywords
    if any(kw in q for kw in ["def ", "import ", "print(", "#!"]):
        return "python"
    if any(kw in q for kw in ["public class", "system.out"]):
        return "java"
    if any(kw in q for kw in ["const ", "let ", "var ", "=>"]):
        return "javascript"
    return "python"  # default


def write_code_file(
    code: str,
    language: str,
    filename: Optional[str] = None,
    directory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Write code to a file with the correct extension.
    Returns {ok, path, language, run_command}.
    """
    lang = LANG_ALIASES.get(language.lower(), language.lower())
    tmpl = LANGUAGE_TEMPLATES.get(lang, LANGUAGE_TEMPLATES["python"])
    ext  = tmpl["ext"]

    save_dir = Path(directory or PROJECT_ROOT / "workspace")
    save_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = f"aria_code_{int(time.time())}{ext}"
    elif not filename.endswith(ext):
        filename = filename + ext

    file_path = save_dir / filename
    file_path.write_text(code, encoding="utf-8")

    runner = tmpl.get("runner", "")
    run_cmd = f"{runner} {file_path.name}" if runner else f"open {file_path.name}"

    return {
        "ok":          True,
        "path":        str(file_path),
        "language":    lang,
        "run_command": run_cmd,
        "message":     f"Written to {file_path} ({lang}). Run with: {run_cmd}",
    }


# ── Main agent class ──────────────────────────────────────────────────────────

class WindowsKernelAgent:
    """
    ARIA Windows Kernel Agent — deep OS integration.

    Provides:
      - run_nl(query)         → natural language dispatch
      - win_r(command)        → Win+R launcher
      - get_chrome_profiles() → list browser profiles
      - open_with_profile(url, profile, browser)
      - powershell(script)
      - cmd(command, cwd)
      - bash(script, cwd)
      - git(command, cwd)
      - analyze_java(path)
      - write_code(code, language, filename, dir)
    """

    def __init__(self, engine=None, confirm_fn=None):
        """
        engine     : core.engine.Engine (for LLM code generation)
        confirm_fn : async callable(message) → bool (for dangerous ops)
        """
        self._engine = engine
        self._confirm = confirm_fn

    # ── Public interface ──────────────────────────────────────────────────────

    def win_r(self, command: str) -> ShellResult:
        return win_r_launch(command)

    def get_chrome_profiles(self, browser: str = "chrome") -> List[Dict[str, str]]:
        return list_chrome_profiles(browser)

    def open_with_profile(
        self, url: str = "", profile: str = "Default", browser: str = "chrome"
    ) -> ShellResult:
        return open_chrome_with_profile(url, profile, browser)

    def open_chrome_dynamic(
        self, query: str, url: str = "", browser: str = "chrome"
    ) -> Dict[str, Any]:
        """
        Open Chrome with a profile identified by natural language.
        query can be a name, email, or partial match.
        e.g. open_chrome_dynamic("Chandan") or open_chrome_dynamic("novaai")

        Returns dict with ok, matched_profile, result.
        """
        matched = find_profile_by_name(query, browser)
        if not matched:
            # No fuzzy match — list available and suggest
            profiles = list_chrome_profiles(browser)
            names = [p["name"] for p in profiles]
            return {
                "ok":      False,
                "error":   f"No profile matching '{query}' found",
                "available_profiles": names,
            }
        result = open_chrome_with_profile(url, matched["folder"], browser)
        return {
            "ok":             result.ok,
            "matched_profile": matched,
            "message":        result.output or result.error,
        }

    def powershell(self, script: str, timeout: int = 30) -> ShellResult:
        return run_powershell(script, timeout=timeout)

    def cmd(self, command: str, cwd: Optional[str] = None) -> ShellResult:
        return run_cmd(command, cwd=cwd)

    def bash(self, script: str, cwd: Optional[str] = None) -> ShellResult:
        return run_bash(script, cwd=cwd)

    def git(self, command: str, cwd: Optional[str] = None) -> ShellResult:
        return run_git(command, cwd=cwd)

    def analyze_java(self, path: str) -> JavaProjectInfo:
        return analyze_java_project(path)

    def write_code(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        return write_code_file(code, language, filename, directory)

    # ── Natural language dispatch ─────────────────────────────────────────────

    async def run_nl(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural-language command and dispatch to the right handler.
        Examples:
          "open run dialog and type calc"
          "run powershell: Get-Process"
          "git status in C:/myproject"
          "open chrome with work profile"
          "analyze java project at C:/MyApp"
          "write python code to sort a list"
        """
        q = query.lower().strip()

        # ── Win+R ──
        m = re.search(r"\brun\s+dialog\b.*?[:\-]\s*(.+)", q) or \
            re.search(r"\bwin\+r\b.*?[:\-]\s*(.+)", q) or \
            re.search(r"\bopen\s+run\b.*?[:\-]\s*(.+)", q)
        if m:
            return win_r_launch(m.group(1).strip()).to_dict()

        # Direct Win+R aliases
        for alias, cmd_val in WIN_R_COMMON_COMMANDS.items():
            if alias in q and ("open" in q or "launch" in q or "run" in q):
                return win_r_launch(alias).to_dict()

        # ── Chrome profile ──
        if re.search(r"\b(chrome|edge|brave)\b.*\bprofile\b", q):
            # "open chrome with work profile" or "which chrome profiles"
            if "list" in q or "which" in q or "show" in q:
                browser = "edge" if "edge" in q else ("brave" if "brave" in q else "chrome")
                profiles = list_chrome_profiles(browser)
                return {
                    "ok": True,
                    "profiles": profiles,
                    "message": f"Found {len(profiles)} {browser} profile(s)",
                }

            # Open with specific profile
            browser = "edge" if "edge" in q else ("brave" if "brave" in q else "chrome")
            profiles = list_chrome_profiles(browser)
            if profiles:
                # Try to match profile name from query
                chosen = profiles[0]
                for p in profiles:
                    if p["name"].lower() in q or p["email"].lower() in q:
                        chosen = p
                        break
                url_m = re.search(r"https?://\S+", query)
                url = url_m.group(0) if url_m else ""
                return open_chrome_with_profile(url, chosen["folder"], browser).to_dict()
            return open_chrome_with_profile("", "Default", browser).to_dict()

        # ── PowerShell ──
        if re.search(r"\bpowershell\b", q):
            # Extract script after colon or "run powershell script: ..."
            script_m = re.search(r"powershell[:\s]+(.+)", query, re.IGNORECASE | re.DOTALL)
            if script_m:
                return run_powershell(script_m.group(1).strip()).to_dict()
            return {"ok": False, "error": "Provide a PowerShell script to run"}

        # ── CMD ──
        if re.search(r"\b(cmd|command\s+prompt)\b.*[:\-]\s*(.+)", q):
            m2 = re.search(r"(?:cmd|command\s+prompt).*?[:\-]\s*(.+)", query, re.IGNORECASE)
            if m2:
                return run_cmd(m2.group(1).strip()).to_dict()

        # ── Bash ──
        if re.search(r"\bbash\b.*[:\-]\s*(.+)", q):
            m3 = re.search(r"bash.*?[:\-]\s*(.+)", query, re.IGNORECASE | re.DOTALL)
            if m3:
                return run_bash(m3.group(1).strip()).to_dict()

        # ── Git ──
        git_m = re.search(r"\bgit\s+(status|log|diff|add|commit|push|pull|clone|branch|checkout|stash|reset|fetch|merge|rebase|init)\b(.+)?", query, re.IGNORECASE)
        if git_m:
            git_cmd = git_m.group(1) + (git_m.group(2) or "")
            cwd_m = re.search(r"(?:in|at|from)\s+([\w/\\:.\- ]+)$", q)
            cwd = cwd_m.group(1).strip() if cwd_m else None
            return run_git(git_cmd.strip(), cwd=cwd).to_dict()

        # ── Java project analyzer ──
        if re.search(r"\b(java|maven|gradle|spring)\b.*\b(project|analyze|dependencies|build)\b", q):
            path_m = re.search(r"(?:at|in|path:?)\s+([\w/\\:.\- ]+)$", query, re.IGNORECASE)
            path = path_m.group(1).strip() if path_m else "."
            info = analyze_java_project(path)
            return {
                "ok":        True,
                "project":   info.__dict__,
                "message":   f"Java project analyzed at {info.path}. "
                             f"Build system: {info.build_system}. "
                             f"Dependencies: {len(info.dependencies)}.",
            }

        # ── Code writer ──
        if re.search(r"\b(write|create|generate|code|script|program|algorithm|algo)\b", q):
            lang = detect_language(query)
            if self._engine:
                # Use LLM to generate code
                prompt = (
                    f"Write a complete, working {lang} program that: {query}\n\n"
                    "Requirements:\n"
                    "- Clean, well-commented code\n"
                    "- Include all imports/dependencies\n"
                    "- Handle edge cases\n"
                    "- Return only the code, no markdown fencing\n"
                )
                try:
                    code = await asyncio.get_event_loop().run_in_executor(
                        None, self._engine.generate, prompt
                    )
                    # Strip markdown fences if present
                    code = re.sub(r"^```\w*\n?", "", code, flags=re.MULTILINE)
                    code = re.sub(r"\n?```$", "", code, flags=re.MULTILINE)
                    result = write_code_file(code, lang)
                    result["code"] = code
                    return result
                except Exception as e:
                    logger.error("Code generation failed: %s", e)
                    return {"ok": False, "error": f"Code generation error: {e}"}
            else:
                return {
                    "ok":      True,
                    "message": f"Code writer ready for {lang}. Connect engine for LLM generation.",
                    "language": lang,
                }

        # ── Generic app open ──
        if re.search(r"\b(open|launch|start)\b", q):
            app_m = re.search(r"(?:open|launch|start)\s+(.+?)(?:\s+with|\s+in|$)", query, re.IGNORECASE)
            if app_m:
                app = app_m.group(1).strip()
                return win_r_launch(app).to_dict()

        return {
            "ok":     False,
            "error":  "Command not recognized",
            "query":  query,
            "hint":   "Try: 'open chrome with work profile', 'run powershell: Get-Process', "
                      "'git status', 'analyze java project at C:/MyApp', "
                      "'write python code to sort a list'",
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[WindowsKernelAgent] = None


def get_agent(engine=None) -> WindowsKernelAgent:
    global _instance
    if _instance is None:
        _instance = WindowsKernelAgent(engine=engine)
    return _instance
