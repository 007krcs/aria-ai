"""
ARIA Terminal Agent — Full Cross-Platform
==========================================
Comprehensive terminal execution agent with AI-powered error fixing, streaming,
application testing, code analysis, and natural-language control.

Key classes:
  - CommandResult   — dataclass for command execution output
  - TestReport      — dataclass for application test results
  - Limitation      — dataclass for discovered code limitations
  - TerminalAgent   — main agent class

All OS detection delegated to agents.os_detector.OsDetectorAgent.
AI fix generation uses engine.generate() (core.engine.Engine).
"""

from __future__ import annotations

import ast
import io
import os
import re
import sys
import json
import time
import shlex
import shutil
import tempfile
import textwrap
import traceback
import subprocess
import threading
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT & OPTIONAL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agents.os_detector import OsDetectorAgent, OSProfile, get_profile
    _OS_DETECTOR = True
except ImportError:
    _OS_DETECTOR = False
    OSProfile = None  # type: ignore[misc,assignment]

try:
    from core.engine import Engine
    _ENGINE = True
except ImportError:
    _ENGINE = False
    Engine = None  # type: ignore[misc,assignment]


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CommandResult:
    """Result of a terminal command execution."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration: float = 0.0
    cmd: str = ""
    success: bool = True
    timed_out: bool = False

    def __post_init__(self) -> None:
        self.success = self.returncode == 0 and not self.timed_out

    def combined(self) -> str:
        """Return stdout + stderr combined."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout)
        if self.stderr.strip():
            parts.append(self.stderr)
        return "\n".join(parts)


@dataclass
class TestReport:
    """Result of testing an application."""
    app_path: str = ""
    launched: bool = False
    launch_error: str = ""
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    overall_success: bool = False
    summary: str = ""
    duration: float = 0.0


@dataclass
class Limitation:
    """A discovered limitation in application code or configuration."""
    title: str = ""
    description: str = ""
    severity: str = "medium"    # "low" | "medium" | "high" | "critical"
    location: str = ""          # file path or "runtime"
    line_number: int = 0
    suggestion: str = ""
    category: str = ""          # "performance" | "security" | "compatibility" | "feature" | "bug"


# ─────────────────────────────────────────────────────────────────────────────
# ERROR CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

ERROR_PATTERNS: Dict[str, List[str]] = {
    "SyntaxError":        ["SyntaxError", "syntax error", "unexpected token", "IndentationError"],
    "ImportError":        ["ImportError", "ModuleNotFoundError", "No module named", "cannot import name"],
    "TypeError":          ["TypeError", "type error", "unsupported operand", "not subscriptable",
                           "not callable", "argument of type"],
    "NameError":          ["NameError", "name '", "is not defined"],
    "AttributeError":     ["AttributeError", "has no attribute", "object has no attribute"],
    "FileNotFoundError":  ["FileNotFoundError", "No such file or directory", "cannot find the path",
                           "not found", "ENOENT"],
    "PermissionError":    ["PermissionError", "Permission denied", "Access is denied",
                           "EACCES", "Operation not permitted"],
    "NetworkError":       ["ConnectionRefusedError", "Connection refused", "No route to host",
                           "Network is unreachable", "TimeoutError", "timed out", "ECONNREFUSED",
                           "ETIMEDOUT", "SSL", "certificate"],
    "MemoryError":        ["MemoryError", "Out of memory", "Cannot allocate memory", "OOM"],
    "RuntimeError":       ["RuntimeError", "RecursionError", "OverflowError", "ZeroDivisionError"],
    "DependencyError":    ["pip install", "npm install", "yarn add", "gem install",
                           "package not found", "dependency"],
    "VersionError":       ["version mismatch", "requires python", "requires node",
                           "incompatible", "version"],
    "ConfigError":        ["config", "configuration", "missing key", "invalid value",
                           "environment variable"],
    "DockerError":        ["docker", "container", "image not found", "docker daemon"],
}


def parse_error(stderr: str, stdout: str = "") -> str:
    """
    Classify an error from stderr/stdout into a known error type.

    Parameters
    ----------
    stderr : str
    stdout : str

    Returns
    -------
    str
        One of the keys in ERROR_PATTERNS, or "UnknownError".
    """
    combined = (stderr + "\n" + stdout).lower()
    # Priority order: more specific first
    priority = [
        "SyntaxError", "ImportError", "ModuleNotFoundError",
        "DependencyError", "FileNotFoundError", "PermissionError",
        "NetworkError", "TypeError", "NameError", "AttributeError",
        "MemoryError", "DockerError", "VersionError", "ConfigError",
        "RuntimeError",
    ]
    for error_type in priority:
        patterns = ERROR_PATTERNS.get(error_type, [])
        for pat in patterns:
            if pat.lower() in combined:
                return error_type
    return "UnknownError"


# ─────────────────────────────────────────────────────────────────────────────
# PACKAGE MANAGER LOGIC
# ─────────────────────────────────────────────────────────────────────────────

# Map package name patterns to preferred install command templates
_PM_OVERRIDES: Dict[str, Tuple[str, str]] = {
    # python_package: (manager_binary, install_template)
}

_LINUX_PKG_MANAGERS = ["apt-get", "apt", "dnf", "yum", "pacman", "zypper", "snap", "flatpak"]
_MAC_PKG_MANAGERS = ["brew"]
_WIN_PKG_MANAGERS = ["winget", "choco", "scoop"]


def _best_package_manager(profile: Optional[Any]) -> Tuple[str, str]:
    """
    Return (manager_binary, install_subcommand) for the current system.
    E.g. ("apt-get", "install") or ("winget", "install") or ("pip", "install").
    """
    if profile is None:
        # Fallback: detect via shutil
        for pm in _LINUX_PKG_MANAGERS + _MAC_PKG_MANAGERS + _WIN_PKG_MANAGERS:
            if shutil.which(pm):
                return pm, "install"
        return "pip", "install"

    pms: List[str] = getattr(profile, "package_managers", [])
    os_name: str = getattr(profile, "os_name", "unknown")

    if os_name == "windows":
        for win_pm in _WIN_PKG_MANAGERS:
            if win_pm in pms:
                return win_pm, "install"
    elif os_name == "macos":
        if "brew" in pms:
            return "brew", "install"
    else:
        # Linux preference order
        for lpm in _LINUX_PKG_MANAGERS:
            if lpm in pms:
                suffix = "-y" if lpm in ("apt-get", "apt", "dnf", "yum") else ""
                return lpm, f"install {suffix}".strip()

    return "pip", "install"


# ─────────────────────────────────────────────────────────────────────────────
# SYNTAX CHECKER
# ─────────────────────────────────────────────────────────────────────────────

_SYNTAX_CHECKERS: Dict[str, Callable[[str], Tuple[bool, str]]] = {}


def _register_syntax_checker(ext: str):
    def decorator(fn: Callable[[str], Tuple[bool, str]]):
        _SYNTAX_CHECKERS[ext] = fn
        return fn
    return decorator


@_register_syntax_checker(".py")
def _check_python_syntax(file_path: str) -> Tuple[bool, str]:
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        ast.parse(source, filename=file_path)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"
    except Exception as exc:
        return False, str(exc)


@_register_syntax_checker(".js")
@_register_syntax_checker(".mjs")
@_register_syntax_checker(".cjs")
def _check_js_syntax(file_path: str) -> Tuple[bool, str]:
    node = shutil.which("node")
    if not node:
        return True, "node not found — skipping JS syntax check"
    result = subprocess.run(
        [node, "--check", file_path],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


@_register_syntax_checker(".ts")
def _check_ts_syntax(file_path: str) -> Tuple[bool, str]:
    tsc = shutil.which("tsc")
    if not tsc:
        return True, "tsc not found — skipping TypeScript syntax check"
    result = subprocess.run(
        [tsc, "--noEmit", "--allowJs", file_path],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


@_register_syntax_checker(".go")
def _check_go_syntax(file_path: str) -> Tuple[bool, str]:
    go = shutil.which("go")
    if not go:
        return True, "go not found — skipping Go syntax check"
    result = subprocess.run(
        [go, "vet", file_path],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


@_register_syntax_checker(".rb")
def _check_ruby_syntax(file_path: str) -> Tuple[bool, str]:
    ruby = shutil.which("ruby")
    if not ruby:
        return True, "ruby not found — skipping Ruby syntax check"
    result = subprocess.run(
        [ruby, "-c", file_path],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


@_register_syntax_checker(".sh")
@_register_syntax_checker(".bash")
def _check_shell_syntax(file_path: str) -> Tuple[bool, str]:
    bash = shutil.which("bash")
    if not bash:
        return True, "bash not found — skipping shell syntax check"
    result = subprocess.run(
        [bash, "-n", file_path],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL AGENT
# ─────────────────────────────────────────────────────────────────────────────

class TerminalAgent:
    """
    ARIA cross-platform terminal agent.

    Provides command execution, AI-powered error fixing, streaming, application
    testing, code limitation analysis, and natural-language control.

    Parameters
    ----------
    engine : Engine, optional
        ARIA engine for AI-powered fix generation.
    os_profile : OSProfile, optional
        Pre-computed OS profile. Auto-detected if omitted.
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        os_profile: Optional[Any] = None,
    ) -> None:
        if os_profile is not None:
            self._profile = os_profile
        elif _OS_DETECTOR:
            detector = OsDetectorAgent()
            self._profile = detector.detect()
        else:
            self._profile = None

        self.engine = engine
        self._os_detector: Optional[OsDetectorAgent] = None
        if _OS_DETECTOR:
            self._os_detector = OsDetectorAgent()
            if os_profile is None:
                self._os_detector._cached_profile = self._profile

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def profile(self) -> Optional[Any]:
        return self._profile

    @property
    def os_name(self) -> str:
        if self._profile:
            return getattr(self._profile, "os_name", "unknown")
        return sys.platform

    @property
    def is_windows(self) -> bool:
        return self.os_name == "windows"

    # ── core execution ───────────────────────────────────────────────────────

    def run_command(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        shell: Optional[str] = None,
        timeout: int = 30,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        """
        Execute a shell command and return a CommandResult.

        Parameters
        ----------
        cmd : str
            Command to run.
        cwd : str, optional
            Working directory.
        shell : str, optional
            Override shell binary ("bash", "powershell", etc.).
        timeout : int
            Seconds before timeout. Default 30.
        env : dict, optional
            Environment variables to merge with the current environment.

        Returns
        -------
        CommandResult
        """
        start = time.monotonic()
        resolved_env = {**os.environ}
        if env:
            resolved_env.update(env)

        # Build the actual invocation
        if shell is None:
            shell = self._pick_shell()

        cmd_list = self._build_cmd_list(cmd, shell)

        try:
            proc = subprocess.run(
                cmd_list,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=resolved_env,
            )
            duration = time.monotonic() - start
            return CommandResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                duration=round(duration, 3),
                cmd=cmd,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return CommandResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                returncode=-1,
                duration=round(duration, 3),
                cmd=cmd,
                timed_out=True,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return CommandResult(
                stdout="",
                stderr=str(exc),
                returncode=-1,
                duration=round(duration, 3),
                cmd=cmd,
            )

    def stream_command(
        self,
        cmd: str,
        callback: Callable[[str], None],
        cwd: Optional[str] = None,
        timeout: int = 60,
    ) -> CommandResult:
        """
        Run *cmd* streaming each output line to *callback* as it arrives.

        Parameters
        ----------
        cmd : str
        callback : Callable[[str], None]
            Called once per output line.
        cwd : str, optional
        timeout : int

        Returns
        -------
        CommandResult
        """
        start = time.monotonic()
        shell = self._pick_shell()
        cmd_list = self._build_cmd_list(cmd, shell)
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        try:
            proc = subprocess.Popen(
                cmd_list,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            def _drain_stderr():
                assert proc.stderr is not None
                for line in proc.stderr:
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            assert proc.stdout is not None
            deadline = time.monotonic() + timeout
            for line in proc.stdout:
                if time.monotonic() > deadline:
                    proc.kill()
                    callback("[ARIA] Stream timed out")
                    break
                stdout_lines.append(line)
                callback(line.rstrip("\n"))

            proc.wait(timeout=max(1, deadline - time.monotonic()))
            stderr_thread.join(timeout=3)

            duration = time.monotonic() - start
            return CommandResult(
                stdout="".join(stdout_lines),
                stderr="".join(stderr_lines),
                returncode=proc.returncode or 0,
                duration=round(duration, 3),
                cmd=cmd,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            callback(f"[ARIA] Error: {exc}")
            return CommandResult(
                stdout="".join(stdout_lines),
                stderr=str(exc),
                returncode=-1,
                duration=round(duration, 3),
                cmd=cmd,
            )

    # ── AI-powered fix loop ──────────────────────────────────────────────────

    def run_and_fix(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        max_retries: int = 3,
        engine: Optional[Any] = None,
    ) -> CommandResult:
        """
        Run a command; if it fails, ask the AI for a fix and retry.

        Parameters
        ----------
        cmd : str
        cwd : str, optional
        max_retries : int
            Maximum number of fix attempts (default 3).
        engine : Engine, optional
            Override the instance's engine.

        Returns
        -------
        CommandResult
            Last result (success or final failure).
        """
        eng = engine or self.engine
        current_cmd = cmd
        last_result = CommandResult(cmd=cmd)

        for attempt in range(max_retries + 1):
            result = self.run_command(current_cmd, cwd=cwd)
            last_result = result

            if result.success:
                return result

            if attempt == max_retries:
                break

            # Classify and fix
            error_type = parse_error(result.stderr, result.stdout)
            fix = self.suggest_fix(
                error_type=error_type,
                stderr=result.stderr,
                stdout=result.stdout,
                cmd=current_cmd,
                engine=eng,
            )

            if not fix:
                break

            # If fix is a replacement command, use it; else try installing deps
            if fix.startswith("CMD:"):
                current_cmd = fix[4:].strip()
            elif fix.startswith("INSTALL:"):
                pkg = fix[8:].strip()
                install_result = self.install_dependency(pkg)
                if not install_result.success:
                    break
            elif fix.startswith("PATCH:"):
                # The AI returned a code patch — apply to a temp file and re-run
                pass  # handled in implement_fix
            else:
                # Treat as a new command to run
                current_cmd = fix

        return last_result

    def suggest_fix(
        self,
        error_type: str,
        stderr: str,
        stdout: str,
        cmd: str,
        engine: Optional[Any] = None,
    ) -> str:
        """
        Ask the AI engine to suggest a fix for a failed command.

        Returns a string in one of the following forms:
          - "CMD:<new_command>"      — replacement command
          - "INSTALL:<package>"      — package to install
          - "<new_command>"          — plain replacement
          - "" (empty)               — could not generate a fix

        Parameters
        ----------
        error_type : str
            Classified error type from parse_error().
        stderr : str
        stdout : str
        cmd : str
        engine : Engine, optional

        Returns
        -------
        str
        """
        # Rule-based fixes first (no LLM needed)
        rule_fix = self._rule_based_fix(error_type, stderr, stdout, cmd)
        if rule_fix:
            return rule_fix

        eng = engine or self.engine
        if eng is None:
            return ""

        prompt = textwrap.dedent(f"""
            A command failed on {self.os_name}. Provide a single fix.

            FAILED COMMAND:
            {cmd}

            ERROR TYPE: {error_type}

            STDERR:
            {stderr[:1500]}

            STDOUT:
            {stdout[:500]}

            Reply with ONE of:
              CMD:<replacement_command>
              INSTALL:<package_name>
              <replacement_command>

            No explanation. No markdown. Just the fix on one line.
        """).strip()

        try:
            response = eng.generate(prompt).strip()
            # Clean up markdown fences if present
            response = re.sub(r"^```[a-z]*\n?", "", response)
            response = re.sub(r"\n?```$", "", response).strip()
            return response
        except Exception:
            return ""

    def _rule_based_fix(
        self, error_type: str, stderr: str, stdout: str, cmd: str
    ) -> str:
        """Rule-based fixes that don't require an AI call."""
        combined = (stderr + "\n" + stdout).lower()

        # Missing Python module
        if error_type in ("ImportError", "DependencyError"):
            match = re.search(r"no module named ['\"]?([a-zA-Z0-9_\-]+)", combined)
            if match:
                pkg = match.group(1).replace("_", "-")
                return f"INSTALL:{pkg}"

        # Node module not found
        if "cannot find module" in combined:
            match = re.search(r"cannot find module ['\"]([^'\"]+)['\"]", combined)
            if match:
                pkg = match.group(1)
                if not pkg.startswith("."):
                    return f"INSTALL:{pkg}"

        # Permission denied on Unix: try with sudo
        if error_type == "PermissionError" and self.os_name != "windows":
            if not cmd.startswith("sudo "):
                return f"CMD:sudo {cmd}"

        # Python not found — try python3
        if "python: not found" in combined or "'python' is not recognized" in combined:
            new_cmd = re.sub(r"\bpython\b", "python3", cmd)
            if new_cmd != cmd:
                return f"CMD:{new_cmd}"

        # pip not found
        if "pip: not found" in combined or "'pip' is not recognized" in combined:
            new_cmd = re.sub(r"\bpip\b", "pip3", cmd)
            if new_cmd != cmd:
                return f"CMD:{new_cmd}"

        return ""

    # ── Python runner ────────────────────────────────────────────────────────

    def run_python(self, code: str, timeout: int = 30) -> CommandResult:
        """
        Execute a Python code snippet in a subprocess sandbox.

        Parameters
        ----------
        code : str
            Python source code to run.
        timeout : int

        Returns
        -------
        CommandResult
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as fh:
            fh.write(code)
            tmp_path = fh.name

        try:
            python = shutil.which("python3") or shutil.which("python") or sys.executable
            result = self.run_command(f"{python} {tmp_path}", timeout=timeout)
            result.cmd = f"<python_snippet>"
            return result
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Shell script runner ──────────────────────────────────────────────────

    def run_in_shell(self, script: str, shell_type: str = "auto") -> CommandResult:
        """
        Run a multi-line shell script.

        Parameters
        ----------
        script : str
            Shell script content.
        shell_type : str
            "auto" | "bash" | "zsh" | "sh" | "powershell" | "cmd"

        Returns
        -------
        CommandResult
        """
        if shell_type == "auto":
            shell_type = self._pick_shell()

        # Determine suffix
        suffix_map = {
            "bash": ".sh",
            "zsh": ".sh",
            "sh": ".sh",
            "powershell": ".ps1",
            "pwsh": ".ps1",
            "cmd": ".bat",
        }
        suffix = suffix_map.get(shell_type, ".sh")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
            encoding="utf-8",
        ) as fh:
            fh.write(script)
            tmp_path = fh.name

        try:
            if shell_type in ("powershell", "pwsh"):
                cmd = f"{shell_type} -NoProfile -ExecutionPolicy Bypass -File {tmp_path}"
            elif shell_type == "cmd":
                cmd = f"cmd /c {tmp_path}"
            else:
                shell_bin = shutil.which(shell_type) or shell_type
                cmd = f"{shell_bin} {tmp_path}"
            return self.run_command(cmd, shell="passthrough")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Dependency installation ──────────────────────────────────────────────

    def install_dependency(self, package_name: str) -> CommandResult:
        """
        Auto-detect the best package manager and install *package_name*.

        Tries: system package manager → pip → npm (based on package hints).

        Parameters
        ----------
        package_name : str
            Package name (e.g., "requests", "numpy", "express").

        Returns
        -------
        CommandResult
        """
        # Decide if it's a Python, Node, or system package
        pm, install_sub = _best_package_manager(self._profile)

        # Heuristic: if package name looks like Python (no @, no /)
        if re.match(r"^[a-zA-Z0-9_\-\.]+$", package_name):
            # Try pip first for Python packages
            pip = shutil.which("pip3") or shutil.which("pip")
            if pip:
                result = self.run_command(
                    f"{pip} install {package_name}",
                    timeout=120,
                )
                if result.success:
                    return result

        # Node packages (@scope/pkg or pure-js names)
        if package_name.startswith("@") or shutil.which("npm"):
            npm = shutil.which("npm")
            if npm:
                result = self.run_command(
                    f"{npm} install -g {package_name}",
                    timeout=120,
                )
                if result.success:
                    return result

        # System package manager
        install_args = install_sub.split()
        cmd_parts = [pm] + install_args + [package_name]
        return self.run_command(" ".join(cmd_parts), timeout=180)

    # ── Syntax checking ──────────────────────────────────────────────────────

    def check_syntax(self, file_path: str) -> Tuple[bool, str]:
        """
        Syntax-check a file based on its extension.

        Supported: .py, .js, .mjs, .cjs, .ts, .go, .rb, .sh, .bash

        Parameters
        ----------
        file_path : str

        Returns
        -------
        Tuple[bool, str]
            (ok, error_message). error_message is empty string when ok.
        """
        ext = Path(file_path).suffix.lower()
        checker = _SYNTAX_CHECKERS.get(ext)
        if checker:
            return checker(file_path)
        return True, f"No syntax checker for '{ext}' — skipping"

    # ── Application testing ──────────────────────────────────────────────────

    def test_application(
        self,
        app_path_or_name: str,
        test_cmds: Optional[List[str]] = None,
    ) -> TestReport:
        """
        Launch an application, verify it starts, optionally run test commands.

        Parameters
        ----------
        app_path_or_name : str
            Path to an executable or a command name.
        test_cmds : list of str, optional
            Commands to run after verifying the app launches.

        Returns
        -------
        TestReport
        """
        report = TestReport(app_path=app_path_or_name)
        start = time.monotonic()

        # Resolve app path
        app = shutil.which(app_path_or_name) or app_path_or_name
        if not os.path.exists(app) and not shutil.which(app_path_or_name):
            report.launch_error = f"Application '{app_path_or_name}' not found"
            report.summary = report.launch_error
            report.duration = round(time.monotonic() - start, 3)
            return report

        # Try launching (quick sanity check — short timeout)
        launch_result = self.run_command(
            f"{app} --version",
            timeout=10,
        )
        if not launch_result.success:
            # Try --help
            launch_result = self.run_command(f"{app} --help", timeout=10)
        if launch_result.returncode in (0, 1, 2):  # many tools exit 1 on --help
            report.launched = True
        else:
            report.launch_error = launch_result.stderr
            report.summary = f"Failed to launch: {report.launch_error}"
            report.duration = round(time.monotonic() - start, 3)
            return report

        # Run test commands
        test_results: List[Dict[str, Any]] = []
        all_ok = True
        for tcmd in (test_cmds or []):
            res = self.run_command(tcmd, timeout=30)
            test_results.append({
                "cmd": tcmd,
                "success": res.success,
                "stdout": res.stdout[:500],
                "stderr": res.stderr[:500],
                "returncode": res.returncode,
                "duration": res.duration,
            })
            if not res.success:
                all_ok = False

        report.test_results = test_results
        report.overall_success = report.launched and all_ok
        passed = sum(1 for r in test_results if r["success"])
        total = len(test_results)
        report.summary = (
            f"{'PASS' if report.overall_success else 'FAIL'}: "
            f"{passed}/{total} test commands succeeded"
            if test_cmds
            else f"App launched successfully: {app}"
        )
        report.duration = round(time.monotonic() - start, 3)
        return report

    # ── Limitation finder ────────────────────────────────────────────────────

    def find_limitations(
        self,
        app_path: str,
        engine: Optional[Any] = None,
    ) -> List[Limitation]:
        """
        Analyse an application's code/docs to discover limitations.

        Uses chain-of-thought reasoning via the AI engine when available.
        Falls back to static heuristics if no engine is provided.

        Parameters
        ----------
        app_path : str
            Path to a file, Python module, or directory.
        engine : Engine, optional

        Returns
        -------
        List[Limitation]
        """
        eng = engine or self.engine
        limitations: List[Limitation] = []

        path = Path(app_path)
        if not path.exists():
            return [Limitation(
                title="File not found",
                description=f"Path '{app_path}' does not exist",
                severity="critical",
                category="bug",
            )]

        # Gather source files
        source_files: List[Path] = []
        if path.is_file():
            source_files = [path]
        else:
            for ext in (".py", ".js", ".ts", ".go", ".rs", ".rb", ".java"):
                source_files.extend(path.rglob(f"*{ext}"))
            source_files = source_files[:20]  # cap to 20 files

        # Static analysis
        for src in source_files:
            limitations.extend(self._static_analysis(src))

        # AI-powered CoT analysis
        if eng and source_files:
            try:
                # Provide the AI with a snippet of each file
                snippets: List[str] = []
                for src in source_files[:5]:
                    content = src.read_text(encoding="utf-8", errors="replace")[:2000]
                    snippets.append(f"--- {src.name} ---\n{content}")
                all_code = "\n\n".join(snippets)

                prompt = textwrap.dedent(f"""
                    You are a senior software engineer reviewing code.
                    List the top limitations, bugs, and improvement areas.

                    For each limitation, output a JSON object on one line:
                    {{"title":"...", "description":"...", "severity":"low|medium|high|critical", "category":"performance|security|compatibility|feature|bug", "suggestion":"..."}}

                    CODE:
                    {all_code[:4000]}

                    Output only JSON lines. No markdown.
                """).strip()

                raw = eng.generate(prompt)
                for line in raw.splitlines():
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            obj = json.loads(line)
                            limitations.append(Limitation(
                                title=obj.get("title", ""),
                                description=obj.get("description", ""),
                                severity=obj.get("severity", "medium"),
                                category=obj.get("category", ""),
                                suggestion=obj.get("suggestion", ""),
                                location=app_path,
                            ))
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass

        return limitations

    def _static_analysis(self, file_path: Path) -> List[Limitation]:
        """Heuristic static analysis for a single source file."""
        results: List[Limitation] = []
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return results

        lines = source.splitlines()
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Hardcoded credentials
            if re.search(r'(password|secret|api_key|token)\s*=\s*["\'][^"\']{4,}["\']', stripped, re.I):
                results.append(Limitation(
                    title="Hardcoded credential",
                    description=f"Line {i}: potential hardcoded credential found",
                    severity="critical",
                    location=str(file_path),
                    line_number=i,
                    suggestion="Use environment variables or a secrets manager",
                    category="security",
                ))

            # Bare except
            if stripped == "except:" or stripped.startswith("except:"):
                results.append(Limitation(
                    title="Bare except clause",
                    description=f"Line {i}: bare except hides all exceptions",
                    severity="medium",
                    location=str(file_path),
                    line_number=i,
                    suggestion="Catch specific exceptions instead of bare except",
                    category="bug",
                ))

            # TODO / FIXME / HACK
            for marker in ("TODO", "FIXME", "HACK", "XXX"):
                if marker in line:
                    results.append(Limitation(
                        title=f"{marker} comment",
                        description=f"Line {i}: {stripped[:120]}",
                        severity="low",
                        location=str(file_path),
                        line_number=i,
                        category="feature",
                    ))
                    break

            # Very long functions (Python: count defs)
        if file_path.suffix == ".py":
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        body_lines = (node.end_lineno or 0) - node.lineno
                        if body_lines > 100:
                            results.append(Limitation(
                                title="Long function",
                                description=(
                                    f"Function '{node.name}' is {body_lines} lines long "
                                    f"(line {node.lineno})"
                                ),
                                severity="low",
                                location=str(file_path),
                                line_number=node.lineno,
                                suggestion="Consider breaking it into smaller functions",
                                category="performance",
                            ))
            except Exception:
                pass

        return results

    # ── AI-powered file fix ──────────────────────────────────────────────────

    def implement_fix(
        self,
        file_path: str,
        issue_description: str,
        engine: Optional[Any] = None,
    ) -> CommandResult:
        """
        AI reads a file, generates a fix for *issue_description*, applies it,
        then re-tests the file's syntax.

        Parameters
        ----------
        file_path : str
        issue_description : str
        engine : Engine, optional

        Returns
        -------
        CommandResult
            Result of syntax check / test after applying the fix.
        """
        eng = engine or self.engine
        path = Path(file_path)
        if not path.exists():
            return CommandResult(
                stderr=f"File not found: {file_path}", returncode=1, cmd="implement_fix"
            )

        original = path.read_text(encoding="utf-8", errors="replace")

        if eng is None:
            return CommandResult(
                stderr="No engine available for AI-powered fix",
                returncode=1,
                cmd="implement_fix",
            )

        prompt = textwrap.dedent(f"""
            Fix the following issue in this file. Return ONLY the complete fixed file
            content. No markdown fences. No explanation.

            ISSUE: {issue_description}

            FILE ({path.name}):
            {original[:6000]}
        """).strip()

        try:
            fixed_content = eng.generate(prompt).strip()
            # Strip possible markdown code fences
            fixed_content = re.sub(r"^```[a-z]*\n?", "", fixed_content)
            fixed_content = re.sub(r"\n?```$", "", fixed_content).strip()
        except Exception as exc:
            return CommandResult(
                stderr=f"Engine error: {exc}", returncode=1, cmd="implement_fix"
            )

        if not fixed_content:
            return CommandResult(
                stderr="Engine returned empty fix", returncode=1, cmd="implement_fix"
            )

        # Back up original
        backup_path = str(path) + ".aria_backup"
        try:
            path.with_suffix(path.suffix + ".aria_backup").write_text(
                original, encoding="utf-8"
            )
        except Exception:
            pass

        # Apply fix
        path.write_text(fixed_content, encoding="utf-8")

        # Validate with syntax check
        ok, error_msg = self.check_syntax(file_path)
        if ok:
            return CommandResult(
                stdout=f"Fix applied successfully to {file_path}",
                returncode=0,
                cmd="implement_fix",
            )
        else:
            # Rollback
            path.write_text(original, encoding="utf-8")
            return CommandResult(
                stdout="",
                stderr=f"Fix introduced a syntax error — rolled back. {error_msg}",
                returncode=1,
                cmd="implement_fix",
            )

    # ── Natural language interface ───────────────────────────────────────────

    def run_nl(self, query: str) -> Any:
        """
        Natural-language interface to the terminal agent.

        Recognised intents:
          "run <command>"
          "fix the error in <file>"
          "test my <app>"
          "install <package>"
          "check syntax of <file>"
          "find limitations in <path>"
          "what OS am I running?" (delegated to os_detector)
          "stream <command>"

        Parameters
        ----------
        query : str

        Returns
        -------
        str or CommandResult or TestReport or List[Limitation]
        """
        q = query.strip()
        ql = q.lower()

        # Delegate OS questions
        if self._os_detector and any(
            kw in ql for kw in ("what os", "am i admin", "what shell", "how much ram",
                                "what cpu", "what gpu", "am i in wsl", "uptime",
                                "package manager", "language")
        ):
            return self._os_detector.run_nl(q)

        # "run <cmd>"
        m = re.match(r"run\s+(.+)", q, re.I)
        if m:
            result = self.run_command(m.group(1))
            return result.combined() or f"Done (exit {result.returncode})"

        # "stream <cmd>"
        m = re.match(r"stream\s+(.+)", q, re.I)
        if m:
            lines: List[str] = []
            self.stream_command(m.group(1), callback=lambda l: lines.append(l))
            return "\n".join(lines)

        # "fix the error in <file>" or "fix <file>"
        m = re.match(r"fix(?:\s+the\s+error\s+in)?\s+(.+)", q, re.I)
        if m:
            target = m.group(1).strip()
            if os.path.isfile(target):
                result = self.implement_fix(target, "Fix all errors and warnings")
                return result.stdout or result.stderr
            # Maybe it's a command
            result = self.run_and_fix(target)
            return result.combined() or f"Done (exit {result.returncode})"

        # "test my <app>" or "test <app>"
        m = re.match(r"test\s+(?:my\s+)?(.+)", q, re.I)
        if m:
            app = m.group(1).strip()
            report = self.test_application(app)
            return report.summary

        # "install <package>"
        m = re.match(r"install\s+(.+)", q, re.I)
        if m:
            pkg = m.group(1).strip()
            result = self.install_dependency(pkg)
            if result.success:
                return f"Successfully installed {pkg}"
            return f"Failed to install {pkg}: {result.stderr[:300]}"

        # "check syntax of <file>" or "syntax <file>"
        m = re.match(r"(?:check\s+syntax\s+(?:of\s+)?|syntax\s+)(.+)", q, re.I)
        if m:
            fpath = m.group(1).strip()
            ok, msg = self.check_syntax(fpath)
            if ok:
                return f"Syntax OK: {fpath}"
            return f"Syntax error in {fpath}: {msg}"

        # "find limitations in <path>" or "limitations <path>"
        m = re.match(r"(?:find\s+limitations\s+(?:in\s+)?|limitations\s+)(.+)", q, re.I)
        if m:
            target = m.group(1).strip()
            lims = self.find_limitations(target)
            if not lims:
                return f"No limitations found in {target}"
            return "\n".join(
                f"[{lim.severity.upper()}] {lim.title}: {lim.description}"
                for lim in lims
            )

        # "python <code>" — inline python
        m = re.match(r"python\s+(.+)", q, re.I | re.S)
        if m:
            result = self.run_python(m.group(1))
            return result.combined()

        # Fallback: treat entire query as a command
        result = self.run_command(q)
        return result.combined() or f"Done (exit {result.returncode})"

    # ── internal helpers ─────────────────────────────────────────────────────

    def _pick_shell(self) -> str:
        """Pick the default shell for the current OS."""
        if self._os_detector:
            return self._os_detector.get_shell()
        if os.name == "nt":
            return "powershell" if shutil.which("powershell") else "cmd"
        return "bash" if shutil.which("bash") else "sh"

    def _build_cmd_list(self, cmd: str, shell: str) -> List[str]:
        """
        Build the subprocess cmd list from a command string and shell name.

        Special value "passthrough" means run cmd directly without a wrapper shell.
        """
        if shell == "passthrough":
            try:
                return shlex.split(cmd)
            except ValueError:
                return [cmd]

        if shell in ("powershell", "pwsh"):
            return [shell, "-NoProfile", "-NonInteractive", "-Command", cmd]

        if shell == "cmd":
            return ["cmd", "/c", cmd]

        # POSIX shells: bash, zsh, sh, fish, etc.
        shell_path = shutil.which(shell) or shell
        return [shell_path, "-c", cmd]


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE
# ─────────────────────────────────────────────────────────────────────────────

_default_agent: Optional[TerminalAgent] = None


def get_agent(engine: Optional[Any] = None) -> TerminalAgent:
    """Return a module-level cached TerminalAgent."""
    global _default_agent
    if _default_agent is None:
        _default_agent = TerminalAgent(engine=engine)
    elif engine is not None and _default_agent.engine is None:
        _default_agent.engine = engine
    return _default_agent


def run(cmd: str, **kwargs) -> CommandResult:
    """Module-level shortcut to run a command."""
    return get_agent().run_command(cmd, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = TerminalAgent()

    print("=== OS Profile ===")
    if agent.profile:
        print(f"  OS:    {agent.profile.os_name}")
        print(f"  Shell: {agent.profile.shell}")
        print(f"  Admin: {agent.profile.is_admin}")

    print("\n=== run_command: echo hello ===")
    r = agent.run_command("echo hello")
    print(f"  stdout={r.stdout!r}  rc={r.returncode}  t={r.duration}s")

    print("\n=== run_python ===")
    r2 = agent.run_python("import sys; print(sys.version)")
    print(f"  {r2.stdout.strip()}")

    print("\n=== check_syntax: this file ===")
    ok, msg = agent.check_syntax(__file__)
    print(f"  ok={ok}  msg={msg!r}")

    print("\n=== parse_error ===")
    cases = [
        ("ModuleNotFoundError: No module named 'requests'", ""),
        ("PermissionError: [Errno 13] Permission denied", ""),
        ("SyntaxError: invalid syntax", ""),
        ("ConnectionRefusedError: [Errno 111] Connection refused", ""),
    ]
    for stderr, stdout in cases:
        print(f"  {parse_error(stderr, stdout):20s} ← {stderr[:50]}")

    print("\n=== NL interface ===")
    for q in ["run echo 'ARIA online'", "install requests"]:
        print(f"  Q: {q}")
        result = agent.run_nl(q)
        print(f"  A: {str(result)[:120]}")
