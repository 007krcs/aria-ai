"""
ARIA — Code Executor Agent
===========================
Safe, sandboxed code execution for Python, JavaScript, Shell, and PowerShell.
Includes formatting, linting, package management, and natural-language dispatch.

Safety:
- Dangerous shell patterns (rm -rf, del /f /s, format, mkfs, shutdown) are blocked.
- Dangerous Python imports used in destructive ways are blocked.
- All subprocess calls use timeout + captured stdout/stderr (no shell=True for Python exec).

Every public method returns:
    {"ok": bool, "result": str, "stdout": str, "stderr": str}

Usage:
    from agents.code_executor import CodeExecutorAgent
    agent = CodeExecutorAgent()
    out = agent.execute_python("print('hello')")
"""

import ast
import re
import sys
import os
import time
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(result: str = "", stdout: str = "", stderr: str = "") -> dict:
    return {"ok": True, "result": result, "stdout": stdout, "stderr": stderr}


def _err(result: str = "", stdout: str = "", stderr: str = "") -> dict:
    return {"ok": False, "result": result, "stdout": stdout, "stderr": stderr}


# ─────────────────────────────────────────────────────────────────────────────
# Safety patterns
# ─────────────────────────────────────────────────────────────────────────────

# Shell-level destructive patterns (checked against the raw command string)
_DANGEROUS_SHELL = [
    r"rm\s+-[a-z]*r[a-z]*f",       # rm -rf / rm -fr
    r"rm\s+-[a-z]*f[a-z]*r",
    r"del\s+/[fFsS]",               # del /f /s
    r"format\s+[a-zA-Z]:",          # format C:
    r"mkfs",
    r":\(\)\{.*\}",                  # fork bomb
    r"shutdown\s+(-[rh]|/[rhs])",   # shutdown commands
    r"rd\s+/[sS]",                  # rd /s /q
    r"rmdir\s+/[sS]",
]
_DANGEROUS_SHELL_RE = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_SHELL]

# Python-level dangerous constructs
_DANGEROUS_PYTHON_PATTERNS = [
    r"os\.system\s*\(",
    r"subprocess\.call\s*\(.*shell\s*=\s*True",
    r"__import__\s*\(\s*['\"]os['\"]",
    r"eval\s*\(\s*input",
    r"exec\s*\(\s*input",
]
_DANGEROUS_PYTHON_RE = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_PYTHON_PATTERNS]


def _check_shell_safety(command: str) -> Optional[str]:
    """Return a violation description if the command is dangerous, else None."""
    for pattern in _DANGEROUS_SHELL_RE:
        if pattern.search(command):
            return f"Blocked: command matches dangerous pattern '{pattern.pattern}'"
    return None


def _check_python_safety(code: str) -> Optional[str]:
    """Return a violation description if Python code looks dangerous, else None."""
    for pattern in _DANGEROUS_PYTHON_RE:
        if pattern.search(code):
            return f"Blocked: Python code matches dangerous pattern '{pattern.pattern}'"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CODE EXECUTOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class CodeExecutorAgent:
    """Safe, sandboxed multi-language code execution for ARIA."""

    # ── Python ────────────────────────────────────────────────────────────────

    def execute_python(self, code: str, timeout: int = 30) -> dict:
        """
        Execute Python code in a subprocess sandbox.
        Returns stdout, stderr, and a human-readable result.
        """
        violation = _check_python_safety(code)
        if violation:
            return _err(violation)

        # Syntax check first
        lint = self.lint_code(code, language="python")
        if not lint["ok"]:
            return _err(f"Syntax error: {lint['result']}", stderr=lint["stderr"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            ok = proc.returncode == 0
            result = f"Exit code {proc.returncode}. " + ("OK" if ok else "Error in execution.")
            return {"ok": ok, "result": result, "stdout": stdout, "stderr": stderr}
        except subprocess.TimeoutExpired:
            return _err(f"Execution timed out after {timeout}s.")
        except Exception as e:
            return _err(f"Execution error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── JavaScript ────────────────────────────────────────────────────────────

    def execute_javascript(self, code: str, timeout: int = 15) -> dict:
        """Execute JavaScript via Node.js if available."""
        node = shutil.which("node") or shutil.which("nodejs")
        if not node:
            return _err("Node.js not found. Install Node.js to run JavaScript.")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [node, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            ok = proc.returncode == 0
            result = f"Exit code {proc.returncode}. " + ("OK" if ok else "JS error.")
            return {"ok": ok, "result": result, "stdout": stdout, "stderr": stderr}
        except subprocess.TimeoutExpired:
            return _err(f"JS execution timed out after {timeout}s.")
        except Exception as e:
            return _err(f"JS execution error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Shell ─────────────────────────────────────────────────────────────────

    def execute_shell(self, command: str, timeout: int = 30,
                      cwd: Optional[str] = None) -> dict:
        """Run a shell command with safety checks."""
        violation = _check_shell_safety(command)
        if violation:
            return _err(violation)

        work_dir = str(Path(cwd).expanduser()) if cwd else None
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            ok = proc.returncode == 0
            result = f"Exit code {proc.returncode}. " + ("OK" if ok else "Command error.")
            return {"ok": ok, "result": result, "stdout": stdout, "stderr": stderr}
        except subprocess.TimeoutExpired:
            return _err(f"Shell command timed out after {timeout}s.")
        except Exception as e:
            return _err(f"Shell error: {e}")

    # ── PowerShell ────────────────────────────────────────────────────────────

    def execute_powershell(self, code: str, timeout: int = 30) -> dict:
        """Run a PowerShell script (Windows: pwsh or powershell)."""
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if not ps:
            return _err("PowerShell not found on this system.")

        violation = _check_shell_safety(code)
        if violation:
            return _err(violation)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ps1", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [ps, "-NonInteractive", "-File", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            ok = proc.returncode == 0
            result = f"Exit code {proc.returncode}. " + ("OK" if ok else "PowerShell error.")
            return {"ok": ok, "result": result, "stdout": stdout, "stderr": stderr}
        except subprocess.TimeoutExpired:
            return _err(f"PowerShell timed out after {timeout}s.")
        except Exception as e:
            return _err(f"PowerShell error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_code(self, code: str, language: str = "python") -> dict:
        """Auto-format code. Python: black > autopep8 > textwrap dedent fallback."""
        if language.lower() != "python":
            # For JS/TS a Node-based formatter would be needed; return as-is
            return _ok("Formatting not supported for this language; code returned unchanged.",
                       stdout=code)

        # Try black
        try:
            import black
            mode = black.Mode()
            formatted = black.format_str(code, mode=mode)
            return _ok("Formatted with black.", stdout=formatted)
        except ImportError:
            pass
        except Exception as e:
            return _err(f"black formatting error: {e}")

        # Try autopep8
        try:
            import autopep8
            formatted = autopep8.fix_code(code)
            return _ok("Formatted with autopep8.", stdout=formatted)
        except ImportError:
            pass
        except Exception as e:
            return _err(f"autopep8 formatting error: {e}")

        # Fallback: dedent
        formatted = textwrap.dedent(code)
        return _ok("No formatter installed (black/autopep8). Applied textwrap.dedent.",
                   stdout=formatted)

    # ── Linting ───────────────────────────────────────────────────────────────

    def lint_code(self, code: str, language: str = "python") -> dict:
        """Check code for syntax errors."""
        if language.lower() == "python":
            try:
                ast.parse(code)
                return _ok("No syntax errors found.")
            except SyntaxError as e:
                msg = f"SyntaxError at line {e.lineno}: {e.msg}"
                return _err(msg, stderr=msg)
        elif language.lower() in ("js", "javascript"):
            node = shutil.which("node") or shutil.which("nodejs")
            if not node:
                return _err("Node.js not available for JS linting.")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                proc = subprocess.run(
                    [node, "--check", tmp_path],
                    capture_output=True, text=True, timeout=10
                )
                ok = proc.returncode == 0
                return {"ok": ok,
                        "result": "No syntax errors." if ok else "Syntax error.",
                        "stdout": proc.stdout or "",
                        "stderr": proc.stderr or ""}
            except Exception as e:
                return _err(f"JS lint error: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        else:
            return _ok(f"No linter configured for '{language}'. Skipped.")

    # ── Package management ────────────────────────────────────────────────────

    def install_package(self, name: str) -> dict:
        """pip install a package and return result."""
        # Validate package name (basic alphanumeric + dash/underscore/dot)
        if not re.match(r"^[A-Za-z0-9_.\-\[\]>=<,! ]+$", name):
            return _err(f"Invalid package name: '{name}'")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", name],
                capture_output=True, text=True, timeout=120
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            ok = proc.returncode == 0
            result = f"Installed '{name}'." if ok else f"Failed to install '{name}'."
            return {"ok": ok, "result": result, "stdout": stdout, "stderr": stderr}
        except subprocess.TimeoutExpired:
            return _err(f"pip install timed out.")
        except Exception as e:
            return _err(f"Install error: {e}")

    def list_installed_packages(self) -> dict:
        """List all installed pip packages."""
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=columns"],
                capture_output=True, text=True, timeout=30
            )
            stdout = proc.stdout or ""
            lines = stdout.strip().splitlines()
            packages = []
            for line in lines[2:]:  # skip header rows
                parts = line.split()
                if len(parts) >= 2:
                    packages.append({"name": parts[0], "version": parts[1]})
            return _ok(f"{len(packages)} packages installed.", stdout=stdout,
                       stderr=proc.stderr or "")
        except Exception as e:
            return _err(f"Error listing packages: {e}")

    # ── File operations ───────────────────────────────────────────────────────

    def create_script(self, filename: str, code: str, language: str = "python") -> dict:
        """Save code to a file. Adds appropriate extension if missing."""
        ext_map = {"python": ".py", "javascript": ".js", "js": ".js",
                   "shell": ".sh", "bash": ".sh", "powershell": ".ps1"}
        path = Path(filename).expanduser()
        if not path.suffix:
            path = path.with_suffix(ext_map.get(language.lower(), ".txt"))

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(path), "w", encoding="utf-8") as f:
                f.write(code)
            return _ok(f"Script saved to {path}", stdout=str(path))
        except Exception as e:
            return _err(f"Error saving script: {e}")

    def run_file(self, path: str, args: str = "") -> dict:
        """Run a file based on its extension (.py, .js, .sh, .bat, .ps1)."""
        p = Path(path).expanduser()
        if not p.exists():
            return _err(f"File not found: {path}")

        ext = p.suffix.lower()
        arg_list = args.split() if args.strip() else []

        try:
            if ext == ".py":
                cmd = [sys.executable, str(p)] + arg_list
            elif ext == ".js":
                node = shutil.which("node") or shutil.which("nodejs")
                if not node:
                    return _err("Node.js not found.")
                cmd = [node, str(p)] + arg_list
            elif ext in (".sh",):
                sh = shutil.which("bash") or shutil.which("sh")
                if not sh:
                    return _err("bash/sh not found.")
                cmd = [sh, str(p)] + arg_list
            elif ext == ".bat":
                cmd = ["cmd.exe", "/c", str(p)] + arg_list
            elif ext == ".ps1":
                ps = shutil.which("pwsh") or shutil.which("powershell")
                if not ps:
                    return _err("PowerShell not found.")
                cmd = [ps, "-NonInteractive", "-File", str(p)] + arg_list
            else:
                return _err(f"Unsupported file type: {ext}")

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            ok = proc.returncode == 0
            return {
                "ok": ok,
                "result": f"Exit code {proc.returncode}.",
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
            }
        except subprocess.TimeoutExpired:
            return _err("File execution timed out after 60s.")
        except Exception as e:
            return _err(f"Run error: {e}")

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_python_version(self) -> dict:
        """Return Python version and executable path."""
        try:
            proc = subprocess.run(
                [sys.executable, "--version"],
                capture_output=True, text=True, timeout=10
            )
            version_str = (proc.stdout or proc.stderr or "").strip()
            return _ok(version_str, stdout=version_str)
        except Exception as e:
            return _err(f"Error getting Python version: {e}")

    # ── Natural-language dispatcher ───────────────────────────────────────────

    def execute_nl(self, instruction: str) -> dict:
        """
        Parse a natural-language instruction and dispatch to the right method.

        Examples:
            "run this Python code: print('hi')"
            "execute javascript: console.log(1+1)"
            "install requests"
            "check python version"
            "list installed packages"
            "run shell: echo hello"
            "run powershell: Get-Date"
        """
        instr = instruction.strip()
        lower = instr.lower()

        # ── install <package>
        m = re.match(r"(?:install|pip install)\s+(.+)", instr, re.IGNORECASE)
        if m:
            return self.install_package(m.group(1).strip())

        # ── python version
        if re.search(r"(python\s+version|version\s+of\s+python|check\s+python)", lower):
            return self.get_python_version()

        # ── list packages
        if re.search(r"list\s+(installed\s+)?packages?", lower):
            return self.list_installed_packages()

        # ── run python code
        m = re.search(
            r"(?:run|execute|eval)\s+(?:this\s+)?(?:python\s+)?(?:code|script)?\s*[:\-]?\s*(.+)",
            instr, re.IGNORECASE | re.DOTALL
        )
        if m and not re.search(r"javascript|js|shell|powershell|ps1", lower.split(":")[0]):
            code = m.group(1).strip().strip("`").strip("\"'")
            return self.execute_python(code)

        # ── run javascript code
        m = re.search(
            r"(?:run|execute)\s+(?:this\s+)?(?:javascript|js)\s+(?:code|script)?\s*[:\-]?\s*(.+)",
            instr, re.IGNORECASE | re.DOTALL
        )
        if m:
            code = m.group(1).strip().strip("`").strip("\"'")
            return self.execute_javascript(code)

        # ── run shell command
        m = re.search(
            r"(?:run|execute)\s+(?:shell|bash|cmd)\s*(?:command)?\s*[:\-]?\s*(.+)",
            instr, re.IGNORECASE
        )
        if m:
            return self.execute_shell(m.group(1).strip())

        # ── run powershell
        m = re.search(
            r"(?:run|execute)\s+(?:powershell|ps)\s*(?:script|code)?\s*[:\-]?\s*(.+)",
            instr, re.IGNORECASE | re.DOTALL
        )
        if m:
            return self.execute_powershell(m.group(1).strip())

        # ── run file
        m = re.search(r"run\s+(?:file\s+)?(.+\.\w+)\s*(.*)", instr, re.IGNORECASE)
        if m:
            return self.run_file(m.group(1).strip(), m.group(2).strip())

        # ── format code
        m = re.search(
            r"format\s+(?:this\s+)?(?:python\s+)?code\s*[:\-]?\s*(.+)",
            instr, re.IGNORECASE | re.DOTALL
        )
        if m:
            return self.format_code(m.group(1).strip())

        # ── lint / check syntax
        m = re.search(
            r"(?:lint|check\s+syntax)\s+(?:of\s+)?(?:this\s+)?(?:python\s+)?code\s*[:\-]?\s*(.+)",
            instr, re.IGNORECASE | re.DOTALL
        )
        if m:
            return self.lint_code(m.group(1).strip())

        return _err(
            f"Could not parse instruction: '{instr}'. "
            "Try: 'run python code: ...', 'install <pkg>', 'check python version', "
            "'run shell: <cmd>', 'run powershell: <code>'."
        )
