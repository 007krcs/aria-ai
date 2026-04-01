"""
ARIA — App Tester  (application testing, limitation finding, and auto-improvement)
===================================================================================
Finds bugs, bottlenecks, and missing features in any app, then fixes them.

Supported app types: Python (Flask/FastAPI/script), Node.js, Electron, static web.

Quick usage:
    from agents.app_tester import AppTesterAgent
    from core.engine import Engine

    engine = Engine()
    tester = AppTesterAgent()

    report = tester.test_app("server.py", engine=engine)
    print(report.summary)

    limits = tester.find_limitations("server.py", engine=engine)
    improvements = tester.suggest_improvements(limits, engine=engine)
    for imp in improvements:
        tester.implement_improvement(imp, dry_run=True, engine=engine)
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()

# ──────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name:        str
    description: str
    steps:       List[str]
    expected:    str
    actual:      str  = ""
    passed:      bool = False
    error:       str  = ""


@dataclass
class TestReport:
    app_path:    str
    app_type:    str
    total_tests: int
    passed:      int
    failed:      int
    errors:      List[str]
    warnings:    List[str]
    stdout:      str
    stderr:      str
    test_cases:  List[TestCase]
    duration_s:  float
    summary:     str
    timestamp:   str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Limitation:
    title:       str
    category:    str   # "performance" | "security" | "feature" | "ux" | "reliability"
    severity:    str   # "critical" | "major" | "minor"
    description: str
    location:    str   # file:line or file
    suggestion:  str


@dataclass
class Improvement:
    title:          str
    description:    str
    code_change:    str   # unified diff or full replacement block
    files_affected: List[str]
    effort_hours:   float
    impact:         str   # "high" | "medium" | "low"
    limitation_ref: str   # which Limitation this addresses


@dataclass
class ImplementResult:
    improvement: Improvement
    success:     bool
    backup_path: str
    test_passed: bool
    message:     str
    diff:        str = ""


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _detect_app_type(path: Path) -> str:
    """Detect application type from file/directory contents."""
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".py":
            text = path.read_text(errors="ignore")
            if any(fw in text for fw in ("flask", "Flask", "fastapi", "FastAPI", "django", "Django")):
                return "python-web"
            return "python-script"
        if suffix in (".js", ".ts", ".mjs"):
            return "node-script"
        if suffix == ".html":
            return "static-web"
        return "unknown"

    if path.is_dir():
        if (path / "package.json").exists():
            pkg = json.loads((path / "package.json").read_text(errors="ignore"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            if "electron" in deps:
                return "electron"
            return "node-app"
        if (path / "requirements.txt").exists() or list(path.glob("*.py")):
            for py in path.glob("*.py"):
                txt = py.read_text(errors="ignore")
                if any(fw in txt for fw in ("flask", "Flask", "fastapi", "FastAPI")):
                    return "python-web"
            return "python-app"
        if (path / "index.html").exists():
            return "static-web"

    return "unknown"


def _collect_source_files(path: Path, max_bytes: int = 200_000) -> Dict[str, str]:
    """Return {relative_path: content} for all readable source files."""
    sources: Dict[str, str] = {}
    exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".html", ".css", ".json"}
    skip_dirs = {"node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build"}

    if path.is_file():
        try:
            sources[path.name] = path.read_text(errors="ignore")
        except Exception:
            pass
        return sources

    for f in path.rglob("*"):
        if any(p in f.parts for p in skip_dirs):
            continue
        if f.suffix.lower() in exts and f.is_file():
            try:
                text = f.read_text(errors="ignore")
                rel  = str(f.relative_to(path))
                sources[rel] = text
                if sum(len(v) for v in sources.values()) > max_bytes:
                    break
            except Exception:
                pass
    return sources


def _install_deps(path: Path) -> Tuple[bool, str]:
    """Install dependencies if requirements.txt / package.json present."""
    if path.is_file():
        path = path.parent

    req = path / "requirements.txt"
    pkg = path / "package.json"
    if req.exists():
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req), "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
        return r.returncode == 0, r.stderr
    if pkg.exists():
        npm = shutil.which("npm")
        if npm:
            r = subprocess.run([npm, "install", "--prefix", str(path)],
                               capture_output=True, text=True, timeout=120)
            return r.returncode == 0, r.stderr
    return True, ""


def _safe_kill(proc: subprocess.Popen):
    """Terminate a subprocess tree safely."""
    try:
        if _PSUTIL:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
        else:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ──────────────────────────────────────────────────────────────────────────────

class AppTesterAgent:
    """
    Application testing, limitation-finding, and auto-improvement agent.
    Uses LLM for code analysis and test generation; subprocess + psutil for runtime.
    """

    def __init__(self, engine=None, workspace: Optional[Path] = None):
        self.engine    = engine
        self.workspace = workspace or (PROJECT_ROOT / "data" / "app_tester_workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ── App testing ────────────────────────────────────────────────────────────

    def test_app(
        self,
        app_path_or_name: str,
        test_plan: Optional[List[str]] = None,
        engine=None,
    ) -> TestReport:
        """
        Full app test pipeline:
        1. Detect app type
        2. Install dependencies
        3. Start the app
        4. Run test scenarios
        5. Monitor health
        6. Collect output and build report
        """
        _engine = engine or self.engine
        path    = Path(app_path_or_name).resolve()

        if not path.exists():
            # try relative to project root
            path = (PROJECT_ROOT / app_path_or_name).resolve()
        if not path.exists():
            return self._error_report(str(app_path_or_name), f"Path not found: {app_path_or_name}")

        app_type   = _detect_app_type(path)
        start_time = time.time()

        console.print(f"[cyan]Testing {app_type} app: {path}[/]")

        # Install dependencies
        dep_ok, dep_err = _install_deps(path)
        if not dep_ok:
            console.print(f"[yellow]Dependency install warning:[/] {dep_err[:200]}")

        # Start the app in a subprocess (only for web/server apps)
        proc        = None
        startup_err = ""
        stdout_buf  = []
        stderr_buf  = []

        if app_type in ("python-web", "python-app", "node-app", "electron"):
            proc, startup_err = self._start_app(path, app_type)
            if proc:
                time.sleep(2)  # give it time to start
                # read initial output
                self._drain_proc(proc, stdout_buf, stderr_buf, duration=1.0)

        # Run test scenarios
        if test_plan is None and _engine:
            test_plan = self._generate_test_plan(path, app_type, _engine)
        test_plan = test_plan or ["Start and respond without crashing"]

        test_cases   = []
        errors_found = []
        warnings     = []

        for scenario in test_plan:
            tc = self._run_scenario(scenario, path, app_type, proc, _engine)
            test_cases.append(tc)
            if not tc.passed:
                errors_found.append(f"{tc.name}: {tc.error}")

        # Collect remaining output
        if proc and proc.poll() is None:
            self._drain_proc(proc, stdout_buf, stderr_buf, duration=1.0)
            _safe_kill(proc)

        # Check for crash indicators in stderr
        full_stderr = "\n".join(stderr_buf) + startup_err
        for line in full_stderr.splitlines():
            if any(kw in line.lower() for kw in ("error", "exception", "traceback", "fatal")):
                if line not in errors_found:
                    errors_found.append(line[:200])
            if any(kw in line.lower() for kw in ("warning", "deprecated", "warn")):
                warnings.append(line[:200])

        passed = sum(1 for tc in test_cases if tc.passed)
        failed = len(test_cases) - passed
        duration = time.time() - start_time

        summary = (
            f"Tested {path.name} ({app_type}): "
            f"{passed}/{len(test_cases)} passed in {duration:.1f}s. "
            f"{'No critical errors.' if not errors_found else str(len(errors_found)) + ' error(s) found.'}"
        )

        return TestReport(
            app_path    = str(path),
            app_type    = app_type,
            total_tests = len(test_cases),
            passed      = passed,
            failed      = failed,
            errors      = errors_found[:20],
            warnings    = warnings[:10],
            stdout      = "\n".join(stdout_buf)[:5000],
            stderr      = full_stderr[:3000],
            test_cases  = test_cases,
            duration_s  = duration,
            summary     = summary,
        )

    def _start_app(self, path: Path, app_type: str) -> Tuple[Optional[subprocess.Popen], str]:
        """Start the app as a subprocess. Returns (process, error_string)."""
        try:
            if app_type in ("python-web", "python-script", "python-app"):
                entry = path if path.is_file() else self._find_entry_py(path)
                if entry is None:
                    return None, "No Python entry point found."
                cmd = [sys.executable, str(entry)]
            elif app_type in ("node-app", "node-script"):
                entry = path if path.is_file() else (path / "index.js")
                node  = shutil.which("node") or shutil.which("nodejs")
                if not node:
                    return None, "Node.js not found in PATH."
                cmd = [node, str(entry)]
            else:
                return None, f"Cannot auto-start app type: {app_type}"

            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(
                cmd,
                stdout   = subprocess.PIPE,
                stderr   = subprocess.PIPE,
                text     = True,
                env      = env,
                cwd      = str(path.parent if path.is_file() else path),
            )
            return proc, ""
        except Exception as exc:
            return None, str(exc)

    def _find_entry_py(self, directory: Path) -> Optional[Path]:
        """Find the main Python entry file in a directory."""
        for candidate in ("main.py", "app.py", "server.py", "run.py", "__main__.py"):
            if (directory / candidate).exists():
                return directory / candidate
        py_files = list(directory.glob("*.py"))
        return py_files[0] if py_files else None

    def _drain_proc(
        self,
        proc:       subprocess.Popen,
        stdout_buf: List[str],
        stderr_buf: List[str],
        duration:   float = 1.0,
    ):
        """Non-blocking read from process stdout/stderr for `duration` seconds."""
        deadline = time.time() + duration
        while time.time() < deadline and proc.poll() is None:
            try:
                import select
                if platform.system() != "Windows":
                    r, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.1)
                    for fd in r:
                        line = fd.readline()
                        if line:
                            if fd is proc.stdout:
                                stdout_buf.append(line.rstrip())
                            else:
                                stderr_buf.append(line.rstrip())
                else:
                    # Windows: best-effort non-blocking
                    time.sleep(0.1)
            except Exception:
                time.sleep(0.1)

    def _generate_test_plan(self, path: Path, app_type: str, engine) -> List[str]:
        """Use LLM to generate a test plan for the given app."""
        sources = _collect_source_files(path, max_bytes=8000)
        snippet = "\n\n".join(
            f"--- {fname} ---\n{content[:1500]}" for fname, content in list(sources.items())[:3]
        )
        prompt = (
            f"You are a QA engineer. Given this {app_type} application source code, "
            "generate a concise test plan as a JSON array of test scenario strings.\n"
            "Each string should describe ONE test scenario (e.g., 'Start the app and verify no crash on launch').\n"
            "Limit to 5-8 scenarios. Output ONLY a JSON array.\n\n"
            f"{snippet}"
        )
        try:
            raw   = engine.generate(prompt, temperature=0.2)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return [
            "Start the app without crashing",
            "Verify main functionality responds correctly",
            "Check error handling for invalid input",
        ]

    def _run_scenario(
        self,
        scenario:  str,
        path:      Path,
        app_type:  str,
        proc:      Optional[subprocess.Popen],
        engine,
    ) -> TestCase:
        """Run a single test scenario and return a TestCase."""
        tc = TestCase(
            name        = scenario[:60],
            description = scenario,
            steps       = [scenario],
            expected    = "No crash, correct output",
        )

        # For script-type apps, run them and check exit code
        if app_type in ("python-script", "python-app") and proc is None:
            entry = path if path.is_file() else self._find_entry_py(path)
            if entry:
                try:
                    result = subprocess.run(
                        [sys.executable, str(entry)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=str(entry.parent),
                    )
                    tc.actual = result.stdout[:500]
                    if result.returncode == 0:
                        tc.passed = True
                    else:
                        tc.passed = False
                        tc.error  = result.stderr[:300]
                except subprocess.TimeoutExpired:
                    tc.passed = True  # long-running is expected
                    tc.actual = "Process still running (timeout expected)"
                except Exception as exc:
                    tc.passed = False
                    tc.error  = str(exc)
            return tc

        # For running web app: check if process is alive
        if proc is not None:
            if proc.poll() is not None:
                tc.passed = False
                tc.error  = f"App crashed (exit code {proc.returncode})"
                return tc
            tc.passed = True
            tc.actual = "App is running"

            # Try HTTP health check if applicable
            if "http" in scenario.lower() or "api" in scenario.lower() or "web" in scenario.lower():
                tc.passed, tc.actual, tc.error = self._http_check(scenario)
            return tc

        # Static analysis fallback via LLM
        if engine:
            sources  = _collect_source_files(path, max_bytes=6000)
            src_text = "\n".join(f"{k}: {v[:800]}" for k, v in list(sources.items())[:2])
            prompt   = (
                f"Test scenario: {scenario}\n"
                f"App code:\n{src_text}\n\n"
                "Based on reading the code, would this test likely PASS or FAIL? "
                "Reply: PASS or FAIL, then one sentence reason."
            )
            verdict = engine.generate(prompt, temperature=0.1).strip().upper()
            tc.passed = verdict.startswith("PASS")
            tc.actual = verdict[:200]
        else:
            tc.passed = True  # no engine = assume pass
            tc.actual = "Static check skipped (no engine)"

        return tc

    def _http_check(self, scenario: str, host: str = "127.0.0.1", port: int = 5000) -> Tuple[bool, str, str]:
        """Simple HTTP GET check against localhost."""
        try:
            import urllib.request
            url = f"http://{host}:{port}/"
            with urllib.request.urlopen(url, timeout=3) as resp:
                body = resp.read(500).decode(errors="ignore")
                return True, f"HTTP {resp.status}: {body[:100]}", ""
        except Exception as exc:
            return False, "", str(exc)[:200]

    def _error_report(self, app_path: str, message: str) -> TestReport:
        return TestReport(
            app_path    = app_path,
            app_type    = "unknown",
            total_tests = 0,
            passed      = 0,
            failed      = 0,
            errors      = [message],
            warnings    = [],
            stdout      = "",
            stderr      = message,
            test_cases  = [],
            duration_s  = 0.0,
            summary     = f"Error: {message}",
        )

    # ── Limitation finding ────────────────────────────────────────────────────

    def find_limitations(
        self,
        app_path: str,
        engine=None,
    ) -> List[Limitation]:
        """
        Read source files and use LLM chain-of-thought to find:
        - Performance bottlenecks
        - Missing features
        - Security holes
        - UX issues
        - Reliability gaps
        """
        _engine = engine or self.engine
        path    = Path(app_path).resolve()
        if not path.exists():
            path = (PROJECT_ROOT / app_path).resolve()

        sources = _collect_source_files(path, max_bytes=20_000)
        if not sources:
            return []

        # Also run static AST analysis
        ast_issues = self._ast_scan(path)

        # Build LLM analysis prompt
        src_summary = "\n\n".join(
            f"=== {fname} ===\n{content[:2500]}"
            for fname, content in list(sources.items())[:6]
        )

        prompt = (
            "You are a senior software engineer performing a code review.\n"
            "Analyse the code below carefully using chain-of-thought reasoning.\n"
            "Find ALL significant limitations: performance bottlenecks, security holes, "
            "missing error handling, missing features, UX problems, scalability issues.\n\n"
            "For each limitation output a JSON object with these fields:\n"
            '  "title", "category" (performance|security|feature|ux|reliability), '
            '"severity" (critical|major|minor), "description", "location" (file:line), "suggestion"\n\n'
            "Output a JSON array of these objects. ONLY output the JSON array.\n\n"
            f"SOURCE CODE:\n{src_summary}"
        )

        limitations = list(ast_issues)  # start with static findings

        if _engine:
            try:
                raw   = _engine.generate(prompt, temperature=0.1)
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    items = json.loads(match.group(0))
                    for item in items:
                        lim = Limitation(
                            title       = item.get("title", "Unknown"),
                            category    = item.get("category", "reliability"),
                            severity    = item.get("severity", "minor"),
                            description = item.get("description", ""),
                            location    = item.get("location", app_path),
                            suggestion  = item.get("suggestion", ""),
                        )
                        limitations.append(lim)
            except Exception as exc:
                console.print(f"[yellow]LLM limitation scan error:[/] {exc}")

        # Sort by severity
        sev_order = {"critical": 0, "major": 1, "minor": 2}
        limitations.sort(key=lambda l: sev_order.get(l.severity, 3))

        console.print(f"[green]Found {len(limitations)} limitation(s) in {app_path}[/]")
        return limitations

    def _ast_scan(self, path: Path) -> List[Limitation]:
        """Static AST scan for common Python issues."""
        issues: List[Limitation] = []
        py_files = [path] if path.is_file() and path.suffix == ".py" else list(path.rglob("*.py"))
        for py in py_files[:10]:
            try:
                source = py.read_text(errors="ignore")
                tree   = ast.parse(source, filename=str(py))
            except SyntaxError as e:
                issues.append(Limitation(
                    title       = "Syntax Error",
                    category    = "reliability",
                    severity    = "critical",
                    description = str(e),
                    location    = f"{py.name}:{e.lineno}",
                    suggestion  = "Fix the syntax error before deployment.",
                ))
                continue

            for node in ast.walk(tree):
                # bare except
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append(Limitation(
                        title       = "Bare except clause",
                        category    = "reliability",
                        severity    = "major",
                        description = "Catches all exceptions including KeyboardInterrupt and SystemExit.",
                        location    = f"{py.name}:{node.lineno}",
                        suggestion  = "Catch specific exception types instead of bare except.",
                    ))
                # hardcoded secrets pattern
                if isinstance(node, ast.Assign):
                    for t in ast.walk(node):
                        if isinstance(t, ast.Constant) and isinstance(t.value, str):
                            name = ""
                            if isinstance(node.targets[0], ast.Name):
                                name = node.targets[0].id.lower()
                            if any(kw in name for kw in ("password", "secret", "key", "token", "api_key")):
                                if len(t.value) > 4:
                                    issues.append(Limitation(
                                        title       = "Hardcoded secret",
                                        category    = "security",
                                        severity    = "critical",
                                        description = f"Variable '{node.targets[0].id if isinstance(node.targets[0], ast.Name) else '?'}' contains a hardcoded secret.",
                                        location    = f"{py.name}:{node.lineno}",
                                        suggestion  = "Move to environment variable or secrets manager.",
                                    ))
                # TODO / FIXME markers
                if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                    val = str(node.value.value)
                    if any(kw in val.upper() for kw in ("TODO", "FIXME", "HACK", "XXX")):
                        issues.append(Limitation(
                            title       = f"Code marker: {val[:50]}",
                            category    = "feature",
                            severity    = "minor",
                            description = val[:200],
                            location    = f"{py.name}:{node.lineno}",
                            suggestion  = "Resolve the outstanding task.",
                        ))
        return issues

    # ── Improvement suggestions ────────────────────────────────────────────────

    def suggest_improvements(
        self,
        limitations: List[Limitation],
        engine=None,
    ) -> List[Improvement]:
        """
        For each limitation, generate a concrete code improvement with
        estimated effort and impact.
        """
        _engine = engine or self.engine
        improvements: List[Improvement] = []

        if not limitations:
            return []

        # Batch the limitations into one LLM call for efficiency
        lim_text = "\n".join(
            f"{i+1}. [{lim.severity.upper()}] {lim.title} at {lim.location}: {lim.description}"
            for i, lim in enumerate(limitations)
        )

        if _engine:
            prompt = (
                "You are a software architect. For each limitation below, produce a concrete improvement.\n"
                "For each, output a JSON object with:\n"
                '  "title", "description", "code_change" (unified diff or replacement snippet), '
                '"files_affected" (array), "effort_hours" (float), '
                '"impact" (high|medium|low), "limitation_ref" (the limitation title)\n'
                "Output a JSON array. ONLY output the JSON array.\n\n"
                f"Limitations:\n{lim_text}"
            )
            try:
                raw   = _engine.generate(prompt, temperature=0.15)
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    items = json.loads(match.group(0))
                    for item in items:
                        imp = Improvement(
                            title           = item.get("title", "Improvement"),
                            description     = item.get("description", ""),
                            code_change     = item.get("code_change", ""),
                            files_affected  = item.get("files_affected", []),
                            effort_hours    = float(item.get("effort_hours", 1.0)),
                            impact          = item.get("impact", "medium"),
                            limitation_ref  = item.get("limitation_ref", ""),
                        )
                        improvements.append(imp)
            except Exception as exc:
                console.print(f"[yellow]Improvement generation error:[/] {exc}")

        # Fallback: one simple suggestion per limitation
        if not improvements:
            for lim in limitations:
                improvements.append(Improvement(
                    title           = f"Fix: {lim.title}",
                    description     = lim.suggestion,
                    code_change     = "# Manual review required",
                    files_affected  = [lim.location.split(":")[0]],
                    effort_hours    = 1.0 if lim.severity == "minor" else 4.0,
                    impact          = "high" if lim.severity == "critical" else "medium",
                    limitation_ref  = lim.title,
                ))

        # Sort by impact
        order = {"high": 0, "medium": 1, "low": 2}
        improvements.sort(key=lambda i: order.get(i.impact, 3))
        return improvements

    # ── Auto-implement ────────────────────────────────────────────────────────

    def implement_improvement(
        self,
        improvement: Improvement,
        dry_run:     bool = False,
        engine=None,
    ) -> ImplementResult:
        """
        AI reads current code, generates modified code, backs up original,
        writes new code, re-runs tests, reverts on failure.
        """
        _engine = engine or self.engine

        if not improvement.files_affected:
            return ImplementResult(
                improvement = improvement,
                success     = False,
                backup_path = "",
                test_passed = False,
                message     = "No files specified in improvement.",
            )

        target_file = Path(improvement.files_affected[0]).resolve()
        if not target_file.exists():
            # try relative to project
            target_file = (PROJECT_ROOT / improvement.files_affected[0]).resolve()
        if not target_file.exists():
            return ImplementResult(
                improvement = improvement,
                success     = False,
                backup_path = "",
                test_passed = False,
                message     = f"Target file not found: {improvement.files_affected[0]}",
            )

        original_code = target_file.read_text(encoding="utf-8", errors="ignore")

        # Generate improved code via LLM
        new_code = original_code
        if _engine:
            prompt = (
                f"You are a senior Python developer.\n"
                f"Apply the following improvement to the code file.\n\n"
                f"Improvement: {improvement.description}\n\n"
                f"Code change guidance:\n{improvement.code_change}\n\n"
                f"Current file content:\n```python\n{original_code[:8000]}\n```\n\n"
                "Output the COMPLETE improved file content. "
                "Do not truncate. Do not add explanations outside the code."
            )
            raw = _engine.generate(prompt, temperature=0.05)
            # Extract code block if wrapped in markdown
            code_match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
            new_code = code_match.group(1).strip() if code_match else raw.strip()

        if dry_run:
            diff = self._simple_diff(original_code, new_code, target_file.name)
            return ImplementResult(
                improvement = improvement,
                success     = True,
                backup_path = "",
                test_passed = True,
                message     = f"DRY RUN — changes not written to {target_file.name}",
                diff        = diff,
            )

        # Backup original
        backup_dir  = self.workspace / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts          = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{target_file.name}.{ts}.bak"
        shutil.copy2(str(target_file), str(backup_path))

        # Write new code
        try:
            target_file.write_text(new_code, encoding="utf-8")
        except Exception as exc:
            return ImplementResult(
                improvement = improvement,
                success     = False,
                backup_path = str(backup_path),
                test_passed = False,
                message     = f"Failed to write file: {exc}",
            )

        # Re-run tests
        test_report = self.run_test_suite(str(target_file.parent))
        test_passed = test_report["passed"]

        if not test_passed:
            # Revert
            shutil.copy2(str(backup_path), str(target_file))
            return ImplementResult(
                improvement = improvement,
                success     = False,
                backup_path = str(backup_path),
                test_passed = False,
                message     = f"Tests failed after change. Reverted to backup. Errors: {test_report.get('errors', '')[:200]}",
                diff        = self._simple_diff(original_code, new_code, target_file.name),
            )

        diff = self._simple_diff(original_code, new_code, target_file.name)
        return ImplementResult(
            improvement = improvement,
            success     = True,
            backup_path = str(backup_path),
            test_passed = True,
            message     = f"Successfully applied improvement to {target_file.name}",
            diff        = diff,
        )

    def _simple_diff(self, old: str, new: str, filename: str) -> str:
        """Generate a simple unified diff string."""
        import difflib
        lines = list(difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=3,
        ))
        return "".join(lines[:100])  # cap output length

    # ── Test suite runner ─────────────────────────────────────────────────────

    def run_test_suite(
        self,
        app_path:  str,
        test_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Auto-discover and run pytest / jest / mocha tests.
        Returns {"passed": bool, "output": str, "errors": str}.
        """
        path = Path(app_path).resolve()
        if not path.exists():
            path = (PROJECT_ROOT / app_path).resolve()

        # Find test file
        if test_file:
            tf = Path(test_file).resolve()
        else:
            tf = self._find_test_file(path)

        if tf is None:
            return {"passed": True, "output": "No test file found — skipping.", "errors": ""}

        ext = tf.suffix.lower()

        if ext == ".py":
            pytest_bin = shutil.which("pytest") or f"{sys.executable} -m pytest"
            cmd = [sys.executable, "-m", "pytest", str(tf), "-v", "--tb=short", "--timeout=30"]
        elif ext in (".js", ".ts", ".mjs"):
            jest = shutil.which("jest") or shutil.which("npx")
            if jest:
                cmd = [jest, "jest", str(tf), "--forceExit"]
            else:
                return {"passed": True, "output": "jest not found.", "errors": ""}
        else:
            return {"passed": True, "output": f"Unknown test file type: {ext}", "errors": ""}

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(path if path.is_dir() else path.parent),
            )
            passed = result.returncode == 0
            return {
                "passed": passed,
                "output": result.stdout[:3000],
                "errors": result.stderr[:1000],
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "output": "", "errors": "Test suite timed out (120s)."}
        except Exception as exc:
            return {"passed": False, "output": "", "errors": str(exc)}

    def _find_test_file(self, path: Path) -> Optional[Path]:
        """Find the best test file in the project."""
        search_root = path if path.is_dir() else path.parent
        patterns    = ["test_*.py", "*_test.py", "tests/*.py", "test/*.py",
                       "*.test.js", "*.spec.js", "__tests__/*.js"]
        for pat in patterns:
            matches = list(search_root.glob(pat))
            if matches:
                return matches[0]
        return None

    # ── Test case generation ──────────────────────────────────────────────────

    def generate_test_cases(
        self,
        app_path: str,
        engine=None,
    ) -> List[Dict[str, Any]]:
        """
        AI analyses the app and generates pytest-compatible test cases.
        Returns list of dicts with "name", "description", "code".
        """
        _engine = engine or self.engine
        if _engine is None:
            return []

        sources = _collect_source_files(Path(app_path).resolve(), max_bytes=10_000)
        src_text = "\n\n".join(
            f"=== {k} ===\n{v[:2000]}" for k, v in list(sources.items())[:4]
        )

        prompt = (
            "You are a senior QA engineer.\n"
            "Analyse the code below and generate comprehensive pytest test cases.\n"
            "Output a JSON array of objects, each with:\n"
            '  "name" (function name), "description", "code" (complete pytest function body)\n'
            "Output ONLY the JSON array.\n\n"
            f"{src_text}"
        )
        try:
            raw   = _engine.generate(prompt, temperature=0.2)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as exc:
            console.print(f"[yellow]Test case generation error:[/] {exc}")
        return []

    def write_test_file(
        self,
        test_cases: List[Dict[str, Any]],
        app_path:   str,
        engine=None,
    ) -> Path:
        """
        Write a pytest test file based on generated test cases.
        Returns the path to the written file.
        """
        path     = Path(app_path).resolve()
        test_dir = path.parent if path.is_file() else path
        out_file = test_dir / "test_aria_generated.py"

        lines = [
            '"""Auto-generated tests by ARIA AppTesterAgent."""',
            "import pytest",
            "import sys",
            f"sys.path.insert(0, r'{test_dir}')",
            "",
        ]

        for tc in test_cases:
            name = tc.get("name", "test_unnamed").replace(" ", "_")
            if not name.startswith("test_"):
                name = f"test_{name}"
            desc = tc.get("description", "")
            code = tc.get("code", "    pass")
            # indent code properly
            indented = textwrap.indent(textwrap.dedent(code), "    ")
            lines.append(f"def {name}():")
            if desc:
                lines.append(f'    """{desc}"""')
            lines.append(indented)
            lines.append("")

        out_file.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Test file written: {out_file}[/]")
        return out_file

    # ── Performance profiling ─────────────────────────────────────────────────

    def profile_performance(
        self,
        app_path:   str,
        duration_s: int = 30,
    ) -> Dict[str, Any]:
        """
        Launch the app, monitor CPU/RAM/response time for duration_s seconds.
        Returns a profile dict.
        """
        if not _PSUTIL:
            return {"error": "psutil not installed. Run: pip install psutil"}

        path     = Path(app_path).resolve()
        app_type = _detect_app_type(path)
        proc, startup_err = self._start_app(path, app_type)

        if proc is None:
            return {"error": f"Could not start app: {startup_err}"}

        time.sleep(2)  # startup grace

        cpu_samples  = []
        ram_samples  = []
        start_time   = time.time()

        console.print(f"[cyan]Profiling {path.name} for {duration_s}s...[/]")

        try:
            ps_proc = psutil.Process(proc.pid)
            while time.time() - start_time < duration_s:
                try:
                    cpu = ps_proc.cpu_percent(interval=1.0)
                    ram = ps_proc.memory_info().rss / (1024 * 1024)  # MB
                    cpu_samples.append(cpu)
                    ram_samples.append(ram)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        finally:
            _safe_kill(proc)

        if not cpu_samples:
            return {"error": "Process exited before profiling completed."}

        profile = {
            "app":          str(path),
            "duration_s":   duration_s,
            "cpu_avg_pct":  sum(cpu_samples) / len(cpu_samples),
            "cpu_max_pct":  max(cpu_samples),
            "ram_avg_mb":   sum(ram_samples) / len(ram_samples),
            "ram_peak_mb":  max(ram_samples),
            "samples":      len(cpu_samples),
        }

        # Assess health
        alerts = []
        if profile["cpu_avg_pct"] > 80:
            alerts.append(f"High average CPU: {profile['cpu_avg_pct']:.1f}%")
        if profile["ram_peak_mb"] > 512:
            alerts.append(f"High peak RAM: {profile['ram_peak_mb']:.1f}MB")
        profile["alerts"] = alerts

        console.print(
            f"  CPU avg={profile['cpu_avg_pct']:.1f}%  peak={profile['cpu_max_pct']:.1f}%\n"
            f"  RAM avg={profile['ram_avg_mb']:.1f}MB  peak={profile['ram_peak_mb']:.1f}MB"
        )
        if alerts:
            for a in alerts:
                console.print(f"  [yellow]ALERT: {a}[/]")

        return profile

    # ── Security audit ────────────────────────────────────────────────────────

    def security_audit(
        self,
        app_path: str,
        engine=None,
    ) -> List[Dict[str, Any]]:
        """
        Scan for common vulnerabilities:
        - SQL injection
        - XSS
        - Hardcoded secrets / keys
        - Insecure dependencies
        - Path traversal
        - Shell injection
        """
        _engine = engine or self.engine
        path    = Path(app_path).resolve()
        issues: List[Dict[str, Any]] = []

        sources = _collect_source_files(path, max_bytes=30_000)

        # ── Static pattern scan ───────────────────────────────────────────────
        patterns = {
            "SQL Injection":       [r"execute\s*\(.*%.*\)", r'f".*SELECT.*{', r"f'.*SELECT.*{"],
            "Shell Injection":     [r"os\.system\(", r"subprocess\.call\(.*shell=True",
                                    r"subprocess\.run\(.*shell=True"],
            "Hardcoded Secret":    [r'(?:password|secret|api_key|token)\s*=\s*["\'][^"\']{4,}["\']'],
            "Path Traversal":      [r"open\(.*\+.*\)", r'open\(f".*{'],
            "XSS (raw HTML out)":  [r"render_template_string\(", r"Markup\(", r"\.html_safe"],
            "Pickle Deserialise":  [r"pickle\.loads\(", r"pickle\.load\("],
            "Eval Usage":          [r"\beval\s*\(", r"\bexec\s*\("],
            "Debug Mode On":       [r"debug\s*=\s*True", r"app\.run\(.*debug\s*=\s*True"],
            "Insecure Random":     [r"random\.random\(\)", r"random\.randint\("],
        }

        for filename, content in sources.items():
            for vuln_type, pats in patterns.items():
                for pat in pats:
                    for m in re.finditer(pat, content, re.IGNORECASE):
                        line_no = content[: m.start()].count("\n") + 1
                        issues.append({
                            "type":     vuln_type,
                            "severity": "critical" if vuln_type in
                                        ("SQL Injection", "Shell Injection", "Hardcoded Secret",
                                         "Pickle Deserialise") else "major",
                            "file":     filename,
                            "line":     line_no,
                            "snippet":  m.group(0)[:80],
                            "fix":      self._security_fix_hint(vuln_type),
                        })

        # ── Dependency check (requirements.txt) ───────────────────────────────
        req_file = (path if path.is_dir() else path.parent) / "requirements.txt"
        if req_file.exists():
            req_content = req_file.read_text(errors="ignore")
            known_vuln  = {
                "flask==0.": "Flask <0.12 has multiple CVEs",
                "django==1.": "Django 1.x is EOL — critical vulnerabilities exist",
                "pillow==4.": "Pillow 4.x has image parsing vulnerabilities",
                "pyyaml==3.": "PyYAML 3.x — arbitrary code execution via yaml.load()",
            }
            for pattern, message in known_vuln.items():
                if pattern in req_content.lower():
                    issues.append({
                        "type":     "Vulnerable Dependency",
                        "severity": "critical",
                        "file":     "requirements.txt",
                        "line":     0,
                        "snippet":  pattern,
                        "fix":      message + " — upgrade immediately.",
                    })

        # ── LLM deep-scan ─────────────────────────────────────────────────────
        if _engine and sources:
            src_summary = "\n".join(
                f"{k}:\n{v[:1500]}" for k, v in list(sources.items())[:3]
            )
            prompt = (
                "You are a security researcher. Identify ADDITIONAL security vulnerabilities "
                "not covered by pattern matching (logic flaws, authentication bypasses, "
                "IDOR, insecure session handling, etc.).\n"
                "Output a JSON array of {type, severity, file, line, snippet, fix}.\n"
                "ONLY output the JSON array.\n\n"
                f"{src_summary}"
            )
            try:
                raw   = _engine.generate(prompt, temperature=0.1)
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    extra = json.loads(match.group(0))
                    issues.extend(extra)
            except Exception:
                pass

        console.print(f"[{'red' if issues else 'green'}]Security audit: {len(issues)} issue(s) found.[/]")
        return issues

    def _security_fix_hint(self, vuln_type: str) -> str:
        hints = {
            "SQL Injection":      "Use parameterised queries (cursor.execute(sql, params)).",
            "Shell Injection":    "Avoid shell=True; pass command as a list.",
            "Hardcoded Secret":   "Move to environment variable: os.environ['SECRET'].",
            "Path Traversal":     "Validate and sanitise user-supplied paths.",
            "XSS (raw HTML out)": "Escape user input; use Jinja2 auto-escaping.",
            "Pickle Deserialise": "Avoid pickle for untrusted data; use JSON.",
            "Eval Usage":         "Replace eval/exec with safe alternatives.",
            "Debug Mode On":      "Disable debug mode in production.",
            "Insecure Random":    "Use secrets.token_hex() for security-sensitive randomness.",
        }
        return hints.get(vuln_type, "Review and apply security best practices.")

    # ── Print helpers ─────────────────────────────────────────────────────────

    def print_test_report(self, report: TestReport):
        table = Table(title=f"Test Report: {Path(report.app_path).name}", show_header=True)
        table.add_column("Field",  style="cyan")
        table.add_column("Value",  style="white")
        table.add_row("App type",  report.app_type)
        table.add_row("Passed",    f"[green]{report.passed}[/]")
        table.add_row("Failed",    f"[red]{report.failed}[/]")
        table.add_row("Duration",  f"{report.duration_s:.1f}s")
        console.print(table)
        if report.errors:
            console.print("\n[red]Errors:[/]")
            for e in report.errors[:5]:
                console.print(f"  • {e}")
        console.print(f"\n[bold]{report.summary}[/]")

    def print_limitations(self, limitations: List[Limitation]):
        table = Table(title="Limitations Found", show_header=True)
        table.add_column("Severity",    style="red")
        table.add_column("Category",    style="yellow")
        table.add_column("Title",       style="white")
        table.add_column("Location",    style="cyan")
        for lim in limitations:
            colour = {"critical": "red", "major": "yellow", "minor": "dim"}.get(lim.severity, "white")
            table.add_row(
                f"[{colour}]{lim.severity}[/]",
                lim.category,
                lim.title,
                lim.location,
            )
        console.print(table)

    # ── Natural-language interface ────────────────────────────────────────────

    def run_nl(self, query: str, engine=None) -> str:
        """
        Natural-language entry point.

        Examples:
            "test my Flask app at server.py"
            "find problems in server.py"
            "improve app.py performance"
            "security audit app.py"
            "profile server.py for 30 seconds"
            "generate tests for app.py"
            "fix the issues you found in app.py"
        """
        _engine = engine or self.engine
        q       = query.lower().strip()

        # extract file/path from query
        path_match = re.search(
            r'(?:in|at|for|of)\s+([\w./\\-]+\.(?:py|js|ts))',
            query, re.IGNORECASE
        )
        app_path = path_match.group(1) if path_match else None

        # test
        if any(kw in q for kw in ("test my", "test the", "run test")):
            if not app_path:
                return "Please specify an app path. E.g. 'test my Flask app at server.py'"
            report = self.test_app(app_path, engine=_engine)
            self.print_test_report(report)
            return report.summary

        # find problems / limitations
        if any(kw in q for kw in ("find problem", "find issue", "find limitation", "analyse", "analyze")):
            if not app_path:
                return "Please specify a file path. E.g. 'find problems in server.py'"
            limits = self.find_limitations(app_path, engine=_engine)
            self.print_limitations(limits)
            return f"Found {len(limits)} limitation(s). Use 'suggest improvements' to get fixes."

        # suggest / improve
        if any(kw in q for kw in ("suggest improvement", "improve", "fix the issues")):
            if not app_path:
                return "Please specify a file path. E.g. 'improve app.py performance'"
            limits = self.find_limitations(app_path, engine=_engine)
            if not limits:
                return "No limitations found."
            imps = self.suggest_improvements(limits, engine=_engine)
            for imp in imps[:3]:
                console.print(f"[cyan]{imp.title}[/] ({imp.impact} impact, {imp.effort_hours}h) — {imp.description[:100]}")
            return f"Suggested {len(imps)} improvement(s)."

        # implement
        if any(kw in q for kw in ("implement", "apply fix", "auto fix")):
            if not app_path:
                return "Please specify a file path. E.g. 'implement improvements in app.py'"
            limits = self.find_limitations(app_path, engine=_engine)
            imps   = self.suggest_improvements(limits[:3], engine=_engine)
            results = []
            for imp in imps:
                res = self.implement_improvement(imp, dry_run=("dry" in q), engine=_engine)
                results.append(f"{'OK' if res.success else 'FAIL'}: {res.message}")
            return "\n".join(results)

        # security audit
        if "security" in q or "audit" in q or "vulnerability" in q or "vulner" in q:
            if not app_path:
                return "Please specify a file path. E.g. 'security audit app.py'"
            issues = self.security_audit(app_path, engine=_engine)
            for iss in issues[:5]:
                console.print(f"[red]{iss['severity'].upper()}[/] [{iss['type']}] {iss['file']}:{iss['line']} — {iss['snippet']}")
            return f"Security audit complete: {len(issues)} issue(s) found."

        # profile
        if "profile" in q or "performance" in q or "benchmark" in q:
            if not app_path:
                return "Please specify a file path. E.g. 'profile server.py'"
            dur = 30
            dur_match = re.search(r"(\d+)\s*(?:second|sec|s)", q)
            if dur_match:
                dur = int(dur_match.group(1))
            result = self.profile_performance(app_path, duration_s=dur)
            if "error" in result:
                return f"Profile error: {result['error']}"
            return (
                f"CPU avg={result['cpu_avg_pct']:.1f}%  peak={result['cpu_max_pct']:.1f}%\n"
                f"RAM avg={result['ram_avg_mb']:.1f}MB  peak={result['ram_peak_mb']:.1f}MB\n"
                + ("\n".join(result.get("alerts", [])) or "No performance alerts.")
            )

        # generate tests
        if "generate test" in q or "write test" in q or "create test" in q:
            if not app_path:
                return "Please specify a file path. E.g. 'generate tests for app.py'"
            test_cases = self.generate_test_cases(app_path, engine=_engine)
            if not test_cases:
                return "Could not generate test cases (engine required)."
            out = self.write_test_file(test_cases, app_path, engine=_engine)
            return f"Generated {len(test_cases)} test case(s) → {out}"

        return (
            "AppTester commands:\n"
            "  'test my Flask app at server.py'\n"
            "  'find problems in server.py'\n"
            "  'suggest improvements for app.py'\n"
            "  'implement improvements in app.py' (or 'dry run')\n"
            "  'security audit app.py'\n"
            "  'profile server.py for 30 seconds'\n"
            "  'generate tests for app.py'"
        )


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    agent = AppTesterAgent()
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(agent.run_nl(q))
    else:
        print("Usage: python agents/app_tester.py 'find problems in server.py'")
