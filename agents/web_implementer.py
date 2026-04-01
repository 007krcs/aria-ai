"""
ARIA — Web Implementer Agent
===============================
Reads web documentation and implements working code like a developer.

Pipeline:
1. Fetch URL (requests + trafilatura or beautifulsoup4)
2. Extract code examples, API patterns, dependencies
3. Analyze framework / library / key concepts (LLM)
4. CoT planning: step-by-step implementation plan (LLM)
5. Generate code files (LLM)
6. Write to output directory
7. Install dependencies (pip / npm)
8. Run and validate (run entry point, check output)
9. Iterate-fix: AI reads errors, generates patches, applies them
10. Return ImplementationResult

Also supports:
- clone_functionality(url) — understand a working app, reimplement similar
- run_nl(query) — natural language interface

Dependencies (all optional with graceful fallback):
  requests, trafilatura, beautifulsoup4
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import hashlib
import tempfile
import textwrap
import subprocess
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

# ── optional imports ─────────────────────────────────────────────────────────

try:
    import requests as _requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    import trafilatura as _traf
    _TRAFILATURA = True
except ImportError:
    _TRAFILATURA = False

try:
    from bs4 import BeautifulSoup as _BS
    _BS4 = True
except ImportError:
    _BS4 = False

# ── project root ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPL_DIR = PROJECT_ROOT / "data" / "implementations"
IMPL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ImplementationResult:
    success: bool
    output_dir: str = ""
    files_created: List[str] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)
    iterations: int = 0
    final_code: Dict[str, str] = field(default_factory=dict)
    install_output: str = ""
    run_output: str = ""
    framework_detected: str = ""
    dependencies: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: _now())


@dataclass
class FetchResult:
    url: str
    text: str
    code_blocks: List[Tuple[str, str]]  # (language, code)
    title: str = ""
    success: bool = True
    error: str = ""


# ── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(url: str) -> str:
    """URL → filesystem-safe slug."""
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    parsed = urlparse(url)
    base = re.sub(r'[^a-z0-9]+', '_', parsed.netloc + parsed.path, flags=re.I).strip('_')
    return f"{base[:40]}_{h}"


def _run(
    cmd: List[str],
    cwd: Optional[str] = None,
    timeout: int = 60,
    env: Optional[Dict] = None,
) -> Tuple[int, str, str]:
    """Run subprocess; return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=cwd, env=env or os.environ.copy()
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s"
    except Exception as exc:
        return 1, "", str(exc)


def _llm_call(engine, prompt: str, system: str = "", max_tokens: int = 4096) -> str:
    """Call ARIA's LLM engine and return response text."""
    if engine is None:
        return ""
    try:
        kwargs: Dict[str, Any] = {}
        if system:
            kwargs["system"] = system
        if hasattr(engine, "chat"):
            resp = engine.chat(prompt, **kwargs)
        elif hasattr(engine, "generate"):
            resp = engine.generate(prompt)
        elif callable(engine):
            resp = engine(prompt)
        else:
            return ""
        if isinstance(resp, dict):
            return resp.get("response") or resp.get("text") or str(resp)
        return str(resp)
    except Exception as exc:
        return f"[LLM error: {exc}]"


def _extract_json(text: str) -> Any:
    """Extract first JSON object or array from text."""
    for pattern in (r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', r'\[.*?\]'):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _extract_code_blocks_from_text(text: str) -> List[Tuple[str, str]]:
    """Extract fenced code blocks from markdown/text. Returns (lang, code) tuples."""
    blocks = []
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    for m in pattern.finditer(text):
        lang = m.group(1).lower() or "text"
        code = m.group(2).strip()
        if code:
            blocks.append((lang, code))
    return blocks


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _patch_apply(file_path: Path, old: str, new: str) -> bool:
    """Apply a simple string replacement patch to a file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        if old in content:
            file_path.write_text(content.replace(old, new, 1), encoding="utf-8")
            return True
        return False
    except Exception:
        return False


# ── fetching ──────────────────────────────────────────────────────────────────

def _fetch_url_raw(url: str, timeout: int = 20) -> Tuple[bool, str, str]:
    """
    Fetch URL, return (success, html_content, error).
    Uses requests if available, else urllib.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ARIA-WebImplementer/1.0; "
            "+https://github.com/aria-ai)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if _REQUESTS:
        try:
            resp = _requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return True, resp.text, ""
        except Exception as exc:
            return False, "", str(exc)
    else:
        import urllib.request
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return True, r.read().decode("utf-8", errors="replace"), ""
        except Exception as exc:
            return False, "", str(exc)


def _html_to_text(html: str) -> str:
    """Convert HTML to clean text. Uses trafilatura > bs4 > regex."""
    if _TRAFILATURA:
        text = _traf.extract(html, include_comments=False, include_tables=True)
        if text:
            return text
    if _BS4:
        soup = _BS(html, "html.parser")
        # remove scripts/styles
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    # minimal regex fallback
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.I)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_code_from_html(html: str) -> List[Tuple[str, str]]:
    """Extract code blocks from HTML. Returns (language, code) tuples."""
    blocks: List[Tuple[str, str]] = []
    if _BS4:
        soup = _BS(html, "html.parser")
        # <pre><code class="language-python">
        for pre in soup.find_all("pre"):
            code_tag = pre.find("code")
            if code_tag:
                classes = code_tag.get("class") or []
                lang = "text"
                for cls in classes:
                    m = re.match(r'language-(\w+)', str(cls))
                    if m:
                        lang = m.group(1).lower()
                        break
                code = code_tag.get_text()
                if code.strip():
                    blocks.append((lang, code.strip()))
            else:
                code = pre.get_text().strip()
                if code:
                    blocks.append(("text", code))
    else:
        # regex fallback
        for m in re.finditer(
            r'<pre[^>]*><code[^>]*class=["\']language-(\w+)["\'][^>]*>(.*?)</code></pre>',
            html, re.DOTALL | re.I
        ):
            lang = m.group(1).lower()
            code = re.sub(r'<[^>]+>', '', m.group(2))
            code = code.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            if code.strip():
                blocks.append((lang, code.strip()))

    return blocks


# ── main agent ────────────────────────────────────────────────────────────────

class WebImplementerAgent:
    """
    Reads web documentation and implements working code.

    Usage:
        agent = WebImplementerAgent(engine=aria_engine)
        result = agent.read_and_implement(
            url="https://fastapi.tiangolo.com",
            task_description="Build a simple REST API with two endpoints",
        )
    """

    MAX_FIX_ITERATIONS = 5
    SYSTEM_PROMPT = (
        "You are an expert software developer. "
        "You write clean, production-ready Python (or appropriate language) code. "
        "You follow best practices and add comments where helpful. "
        "When generating code files, always output valid, runnable code."
    )

    def __init__(self, engine=None, memory=None):
        self.engine = engine
        self.memory = memory

    # ── 1. Full pipeline ──────────────────────────────────────────────────────

    def read_and_implement(
        self,
        url: str,
        task_description: str,
        output_dir: Optional[str] = None,
        engine=None,
    ) -> ImplementationResult:
        """
        Full pipeline: fetch → analyze → plan → generate → write → install →
        run → fix → return ImplementationResult.
        """
        eng = engine or self.engine
        errors: List[str] = []
        iterations = 0

        # 1. Fetch
        fetch_result = self.fetch_documentation(url)
        if not fetch_result.success:
            return ImplementationResult(
                success=False,
                errors_encountered=[f"Fetch failed: {fetch_result.error}"],
            )

        # 2. Analyze framework
        framework_info = self.analyze_framework(fetch_result.text, eng)

        # 3. Build implementation plan
        plan = self.build_implementation_plan(
            task=task_description,
            context=fetch_result.text[:8000],
            engine=eng,
        )

        # 4. Generate code files
        context = {
            "url": url,
            "framework": framework_info,
            "code_examples": fetch_result.code_blocks[:10],
            "documentation": fetch_result.text[:6000],
        }
        files_dict = self.generate_code_files(plan, context, eng)

        if not files_dict:
            return ImplementationResult(
                success=False,
                errors_encountered=["Code generation produced no files"],
                framework_detected=framework_info.get("framework", ""),
            )

        # 5. Determine output directory
        if output_dir is None:
            slug = _slug(url)
            output_dir = str(IMPL_DIR / slug)
        out_path = Path(output_dir)

        # 6. Write files
        created = self.create_project_structure(files_dict, output_dir)

        # 7. Install dependencies
        deps = framework_info.get("dependencies", [])
        install_out = ""
        if deps:
            install_result = self.install_dependencies(deps, cwd=output_dir)
            install_out = install_result.get("output", "")
            if not install_result.get("success", True):
                errors.append(f"Install warning: {install_out}")

        # 8. Run and validate
        entry = self._guess_entry_point(files_dict)
        run_result = self.run_and_validate(
            cwd=output_dir, entry_point=entry, engine=eng
        )
        run_out = run_result.get("output", "")
        run_success = run_result.get("success", False)

        # 9. Iterate-fix
        while not run_success and iterations < self.MAX_FIX_ITERATIONS:
            iterations += 1
            error_msg = run_result.get("error", run_out)
            errors.append(f"Run error (iteration {iterations}): {error_msg[:500]}")

            fixed = False
            for filename in created:
                fp = out_path / filename
                if fp.exists():
                    fix = self.iterate_fix(error_msg, str(fp), eng)
                    if fix.get("patched"):
                        fixed = True
                        break

            if not fixed:
                break

            run_result = self.run_and_validate(
                cwd=output_dir, entry_point=entry, engine=eng
            )
            run_out = run_result.get("output", "")
            run_success = run_result.get("success", False)

        # collect final code
        final_code: Dict[str, str] = {}
        for fname in created:
            fp = out_path / fname
            if fp.exists():
                try:
                    final_code[fname] = fp.read_text(encoding="utf-8")
                except Exception:
                    pass

        return ImplementationResult(
            success=run_success or len(errors) == 0,
            output_dir=output_dir,
            files_created=created,
            errors_encountered=errors,
            iterations=iterations,
            final_code=final_code,
            install_output=install_out,
            run_output=run_out,
            framework_detected=framework_info.get("framework", ""),
            dependencies=deps,
        )

    # ── 2. Fetch documentation ────────────────────────────────────────────────

    def fetch_documentation(self, url: str) -> FetchResult:
        """Fetch URL and return clean text + code blocks."""
        ok, html, err = _fetch_url_raw(url)
        if not ok:
            return FetchResult(url=url, text="", code_blocks=[], success=False, error=err)

        # title
        title = ""
        title_m = re.search(r'<title[^>]*>(.*?)</title>', html, re.I | re.DOTALL)
        if title_m:
            title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip()

        text = _html_to_text(html)
        code_blocks = _extract_code_from_html(html)
        # also grab from text (markdown-style)
        code_blocks += _extract_code_blocks_from_text(text)

        # deduplicate by code content
        seen: set = set()
        unique_blocks: List[Tuple[str, str]] = []
        for lang, code in code_blocks:
            key = code[:200]
            if key not in seen:
                seen.add(key)
                unique_blocks.append((lang, code))

        return FetchResult(
            url=url, text=text, code_blocks=unique_blocks,
            title=title, success=True
        )

    # ── 3. Extract code examples ──────────────────────────────────────────────

    def extract_code_examples(self, html_or_text: str) -> List[Tuple[str, str]]:
        """Extract (language, code) tuples from HTML or markdown text."""
        blocks = _extract_code_from_html(html_or_text)
        if not blocks:
            blocks = _extract_code_blocks_from_text(html_or_text)
        return blocks

    # ── 4. Analyze framework ──────────────────────────────────────────────────

    def analyze_framework(self, content: str, engine) -> Dict[str, Any]:
        """
        Detect framework, version, key concepts, and dependencies.
        Returns dict with keys: framework, version, language, key_concepts, dependencies.
        """
        # heuristic detection first
        info = self._heuristic_framework(content)

        if engine:
            prompt = (
                "Analyze this documentation and extract:\n"
                "1. Framework/library name\n"
                "2. Version (if mentioned)\n"
                "3. Programming language\n"
                "4. Key concepts (up to 5)\n"
                "5. Required pip/npm packages (list only package names)\n\n"
                f"Documentation excerpt:\n{content[:4000]}\n\n"
                "Reply with JSON: {\"framework\": \"...\", \"version\": \"...\", "
                "\"language\": \"...\", \"key_concepts\": [...], \"dependencies\": [...]}"
            )
            raw = _llm_call(engine, prompt, system=self.SYSTEM_PROMPT)
            parsed = _extract_json(raw)
            if isinstance(parsed, dict):
                # merge, preferring LLM result
                info.update({k: v for k, v in parsed.items() if v})

        return info

    def _heuristic_framework(self, content: str) -> Dict[str, Any]:
        """Rule-based framework detection."""
        lower = content.lower()
        framework = "unknown"
        language = "python"
        deps: List[str] = []

        checks = [
            ("react", "React", "javascript", ["react", "react-dom"]),
            ("vue", "Vue.js", "javascript", ["vue"]),
            ("angular", "Angular", "javascript", ["@angular/core"]),
            ("fastapi", "FastAPI", "python", ["fastapi", "uvicorn"]),
            ("flask", "Flask", "python", ["flask"]),
            ("django", "Django", "python", ["django"]),
            ("express", "Express.js", "javascript", ["express"]),
            ("next.js", "Next.js", "javascript", ["next", "react"]),
            ("langchain", "LangChain", "python", ["langchain"]),
            ("pytorch", "PyTorch", "python", ["torch"]),
            ("tensorflow", "TensorFlow", "python", ["tensorflow"]),
            ("sklearn", "scikit-learn", "python", ["scikit-learn"]),
            ("sqlalchemy", "SQLAlchemy", "python", ["sqlalchemy"]),
            ("pydantic", "Pydantic", "python", ["pydantic"]),
            ("streamlit", "Streamlit", "python", ["streamlit"]),
            ("gradio", "Gradio", "python", ["gradio"]),
        ]

        for keyword, name, lang, pkgs in checks:
            if keyword in lower:
                framework = name
                language = lang
                deps = pkgs
                break

        return {
            "framework": framework,
            "version": "",
            "language": language,
            "key_concepts": [],
            "dependencies": deps,
        }

    # ── 5. Build implementation plan ─────────────────────────────────────────

    def build_implementation_plan(
        self,
        task: str,
        context: str,
        engine,
    ) -> List[Dict[str, str]]:
        """
        CoT-based step-by-step implementation plan.
        Returns list of {step, description, file, type} dicts.
        """
        if not engine:
            return self._fallback_plan(task)

        prompt = (
            f"You are planning a software implementation task.\n\n"
            f"Task: {task}\n\n"
            f"Context from documentation:\n{context[:5000]}\n\n"
            "Think step by step. Create a detailed implementation plan.\n"
            "For each step, specify:\n"
            "- step: step number\n"
            "- description: what to do\n"
            "- file: filename to create or modify\n"
            "- type: 'create_file' | 'install_package' | 'configure' | 'test'\n\n"
            "Reply with JSON array: [{\"step\": 1, \"description\": \"...\", "
            "\"file\": \"...\", \"type\": \"...\"}]\n\n"
            "Be specific. Include all necessary files (main app, config, requirements, tests)."
        )
        raw = _llm_call(engine, prompt, system=self.SYSTEM_PROMPT)
        parsed = _extract_json(raw)
        if isinstance(parsed, list) and parsed:
            return parsed
        return self._fallback_plan(task)

    def _fallback_plan(self, task: str) -> List[Dict[str, str]]:
        return [
            {"step": 1, "description": f"Create main application: {task}",
             "file": "main.py", "type": "create_file"},
            {"step": 2, "description": "Create requirements file",
             "file": "requirements.txt", "type": "create_file"},
            {"step": 3, "description": "Create README",
             "file": "README.md", "type": "create_file"},
        ]

    # ── 6. Generate code files ────────────────────────────────────────────────

    def generate_code_files(
        self,
        plan: List[Dict[str, str]],
        context: Dict[str, Any],
        engine,
    ) -> Dict[str, str]:
        """
        Generate code for each file in the plan.
        Returns {filename: code_content} dict.
        """
        files: Dict[str, str] = {}
        framework = context.get("framework", {})
        framework_name = (
            framework.get("framework", "Python") if isinstance(framework, dict) else str(framework)
        )
        doc_excerpt = (
            context.get("documentation", "")[:4000]
            if isinstance(context.get("documentation"), str) else ""
        )
        code_examples = context.get("code_examples", [])
        examples_text = ""
        for lang, code in (code_examples or [])[:5]:
            examples_text += f"\n```{lang}\n{code[:1000]}\n```\n"

        file_steps = [s for s in plan if s.get("type") == "create_file"]

        if not engine:
            # no LLM: generate stub files
            for step in file_steps:
                fname = step.get("file", "main.py")
                files[fname] = f"# {step.get('description', '')}\n# TODO: implement\n"
            return files

        for step in file_steps:
            fname = step.get("file", "main.py")
            description = step.get("description", "")

            # already generated (same file referenced multiple times)
            if fname in files:
                continue

            all_files_context = (
                "\n".join(
                    f"File: {f}\n```\n{c[:800]}\n```"
                    for f, c in files.items()
                )
                if files else "None yet"
            )

            prompt = (
                f"Framework/Library: {framework_name}\n"
                f"Task: {description}\n"
                f"File to create: {fname}\n\n"
                f"Documentation context:\n{doc_excerpt}\n\n"
                f"Code examples from docs:\n{examples_text}\n\n"
                f"Other files already created:\n{all_files_context}\n\n"
                f"Write the complete, production-ready content for '{fname}'.\n"
                "Output ONLY the file content — no explanation, no markdown fences, "
                "just the raw code/text."
            )
            code = _llm_call(engine, prompt, system=self.SYSTEM_PROMPT)
            # strip accidental fences
            code = re.sub(r'^```\w*\n?', '', code.strip())
            code = re.sub(r'\n?```$', '', code.strip())
            files[fname] = code

        # always ensure requirements.txt exists if we have deps
        deps = (
            context.get("framework", {}).get("dependencies", [])
            if isinstance(context.get("framework"), dict) else []
        )
        if deps and "requirements.txt" not in files:
            files["requirements.txt"] = "\n".join(deps) + "\n"

        return files

    # ── 7. Create project structure ───────────────────────────────────────────

    def create_project_structure(
        self, files_dict: Dict[str, str], output_dir: str
    ) -> List[str]:
        """Write all files to output_dir. Returns list of created filenames."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        created: List[str] = []
        for fname, content in files_dict.items():
            fp = out / fname
            _write_file(fp, content)
            created.append(fname)
        return created

    # ── 8. Install dependencies ───────────────────────────────────────────────

    def install_dependencies(
        self, requirements: List[str], cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Install requirements via pip or npm depending on package names.
        Returns {"success": bool, "output": str}.
        """
        if not requirements:
            return {"success": True, "output": "No dependencies to install"}

        # check for requirements.txt in cwd
        req_file = Path(cwd or ".") / "requirements.txt" if cwd else None
        if req_file and req_file.exists():
            rc, out, err = _run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
                cwd=cwd, timeout=120,
            )
            return {"success": rc == 0, "output": out or err}

        # split into python / node packages
        py_pkgs = [p for p in requirements if not p.startswith("@") and "/" not in p.split("@")[0]]
        node_pkgs = [p for p in requirements if p not in py_pkgs]

        output_parts: List[str] = []
        overall_success = True

        if py_pkgs:
            rc, out, err = _run(
                [sys.executable, "-m", "pip", "install"] + py_pkgs + ["--quiet"],
                cwd=cwd, timeout=120,
            )
            output_parts.append(f"pip: {out or err}")
            if rc != 0:
                overall_success = False

        if node_pkgs:
            rc, out, err = _run(
                ["npm", "install"] + node_pkgs, cwd=cwd, timeout=120
            )
            output_parts.append(f"npm: {out or err}")
            if rc != 0:
                overall_success = False

        return {"success": overall_success, "output": "\n".join(output_parts)}

    # ── 9. Run and validate ───────────────────────────────────────────────────

    def run_and_validate(
        self,
        cwd: str,
        entry_point: Optional[str] = None,
        engine=None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Run the project entry point and check it works.
        Returns {"success": bool, "output": str, "error": str}.
        """
        eng = engine or self.engine
        cwd_path = Path(cwd)

        if not entry_point:
            entry_point = self._guess_entry_point_from_dir(cwd_path)

        if not entry_point:
            return {"success": False, "output": "", "error": "No entry point found"}

        ep = cwd_path / entry_point
        if not ep.exists():
            return {
                "success": False, "output": "",
                "error": f"Entry point not found: {entry_point}"
            }

        # determine run command
        ext = ep.suffix.lower()
        if ext == ".py":
            cmd = [sys.executable, str(ep)]
        elif ext in (".js", ".mjs"):
            cmd = ["node", str(ep)]
        elif ext == ".ts":
            cmd = ["ts-node", str(ep)]
        elif ext == ".sh":
            cmd = ["bash", str(ep)]
        else:
            cmd = [str(ep)]

        rc, out, err = _run(cmd, cwd=cwd, timeout=timeout)
        combined = (out + "\n" + err).strip()
        success = rc == 0

        # LLM sanity check
        if not success and eng and combined:
            verdict_prompt = (
                f"A program ran and produced this output (exit code {rc}):\n\n"
                f"{combined[:1000]}\n\n"
                "Is this a real error that needs fixing, or is it normal output? "
                "Reply with JSON: {\"is_error\": true/false, \"reason\": \"...\"}"
            )
            raw = _llm_call(eng, verdict_prompt)
            parsed = _extract_json(raw)
            if isinstance(parsed, dict) and not parsed.get("is_error", True):
                success = True  # LLM says it's fine

        return {"success": success, "output": out, "error": err, "exit_code": rc}

    def _guess_entry_point(self, files_dict: Dict[str, str]) -> Optional[str]:
        """Guess the main entry point from a files dict."""
        for candidate in ("main.py", "app.py", "server.py", "index.py",
                          "index.js", "app.js", "server.js", "index.ts"):
            if candidate in files_dict:
                return candidate
        py_files = [f for f in files_dict if f.endswith(".py")]
        if py_files:
            return py_files[0]
        js_files = [f for f in files_dict if f.endswith(".js")]
        if js_files:
            return js_files[0]
        return None

    def _guess_entry_point_from_dir(self, cwd: Path) -> Optional[str]:
        for candidate in ("main.py", "app.py", "server.py", "index.py",
                          "index.js", "app.js", "server.js"):
            if (cwd / candidate).exists():
                return candidate
        py_files = list(cwd.glob("*.py"))
        if py_files:
            return py_files[0].name
        return None

    # ── 10. Iterate-fix ───────────────────────────────────────────────────────

    def iterate_fix(
        self, error: str, file_path: str, engine=None
    ) -> Dict[str, Any]:
        """
        AI reads error + file content, generates a patch, applies it.
        Returns {"patched": bool, "message": str}.
        """
        eng = engine or self.engine
        fp = Path(file_path)
        if not fp.exists():
            return {"patched": False, "message": "File not found"}

        try:
            original = fp.read_text(encoding="utf-8")
        except Exception as exc:
            return {"patched": False, "message": str(exc)}

        if not eng:
            return {"patched": False, "message": "No LLM engine available for fixing"}

        prompt = (
            f"The following file has an error. Fix it.\n\n"
            f"FILE: {fp.name}\n"
            f"ERROR:\n{error[:1000]}\n\n"
            f"CURRENT FILE CONTENT:\n```\n{original[:4000]}\n```\n\n"
            "Output the COMPLETE fixed file content. "
            "No explanation, no markdown fences — just the corrected code."
        )
        fixed_code = _llm_call(eng, prompt, system=self.SYSTEM_PROMPT)
        fixed_code = re.sub(r'^```\w*\n?', '', fixed_code.strip())
        fixed_code = re.sub(r'\n?```$', '', fixed_code.strip())

        if fixed_code and fixed_code != original:
            try:
                fp.write_text(fixed_code, encoding="utf-8")
                return {"patched": True, "message": f"Fixed {fp.name}"}
            except Exception as exc:
                return {"patched": False, "message": str(exc)}

        return {"patched": False, "message": "LLM produced no change"}

    # ── 11. Clone functionality ───────────────────────────────────────────────

    def clone_functionality(
        self,
        url: str,
        engine=None,
        output_dir: Optional[str] = None,
    ) -> ImplementationResult:
        """
        Look at a working app at URL, understand what it does,
        and reimplement similar functionality from scratch.
        """
        eng = engine or self.engine
        fetch = self.fetch_documentation(url)
        if not fetch.success:
            return ImplementationResult(
                success=False,
                errors_encountered=[f"Fetch failed: {fetch.error}"]
            )

        # understand what the app does
        if eng:
            understand_prompt = (
                f"Look at this web page content and understand what the application does.\n\n"
                f"URL: {url}\n"
                f"Content:\n{fetch.text[:5000]}\n\n"
                "Describe in 2-3 sentences: what does this app do? "
                "What are its core features? What technology does it use?"
            )
            understanding = _llm_call(eng, understand_prompt, system=self.SYSTEM_PROMPT)
        else:
            understanding = f"Application at {url}"

        task = (
            f"Implement an application similar to what is described here: {understanding}\n\n"
            f"Base it on patterns from: {url}\n"
            f"Recreate the core functionality from scratch in Python."
        )

        return self.read_and_implement(
            url=url,
            task_description=task,
            output_dir=output_dir,
            engine=eng,
        )

    # ── 12. Natural language interface ────────────────────────────────────────

    def run_nl(self, query: str) -> Dict[str, Any]:
        """
        Natural language interface.

        Examples:
          "read https://fastapi.tiangolo.com and build a todo REST API"
          "implement auth from https://jwt.io/introduction"
          "clone https://example-app.com"
          "fetch docs from https://docs.example.com"
        """
        q = query.strip()
        eng = self.engine

        # extract URL
        url_match = re.search(r'https?://[^\s"\']+', q)
        url = url_match.group(0) if url_match else None

        # clone intent
        if re.search(r'\bclone\b', q, re.I) and url:
            result = self.clone_functionality(url, engine=eng)
            return {
                "action": "clone",
                "url": url,
                "success": result.success,
                "output_dir": result.output_dir,
                "files_created": result.files_created,
                "errors": result.errors_encountered,
                "framework": result.framework_detected,
            }

        # fetch/docs only
        if re.search(r'\b(fetch|read docs?|get docs?|documentation)\b', q, re.I) and url:
            fetch = self.fetch_documentation(url)
            return {
                "action": "fetch",
                "url": url,
                "success": fetch.success,
                "title": fetch.title,
                "text_length": len(fetch.text),
                "code_blocks": len(fetch.code_blocks),
                "error": fetch.error,
            }

        # implement
        if url:
            # extract task description — everything that's not the URL
            task = re.sub(r'https?://[^\s"\']+', '', q)
            task = re.sub(
                r'\b(read|implement|build|create|make|from|using|with|based on|and)\b',
                ' ', task, flags=re.I
            ).strip()
            if not task:
                task = "Implement a working application based on this documentation"

            result = self.read_and_implement(
                url=url, task_description=task, engine=eng
            )
            return {
                "action": "implement",
                "url": url,
                "task": task,
                "success": result.success,
                "output_dir": result.output_dir,
                "files_created": result.files_created,
                "framework": result.framework_detected,
                "iterations": result.iterations,
                "errors": result.errors_encountered[:5],
                "run_output": result.run_output[:500],
            }

        return {
            "action": "error",
            "message": (
                "Please include a URL in your query. "
                "Example: 'read https://fastapi.tiangolo.com and build a REST API'"
            ),
        }

    # ── convenience: batch implement from multiple URLs ───────────────────────

    def implement_from_multiple(
        self,
        sources: List[Dict[str, str]],
        output_dir: Optional[str] = None,
        engine=None,
    ) -> List[ImplementationResult]:
        """
        Implement from multiple documentation sources.
        sources: [{"url": "...", "task": "..."}]
        """
        results = []
        for src in sources:
            r = self.read_and_implement(
                url=src["url"],
                task_description=src.get("task", "Implement based on this documentation"),
                output_dir=output_dir,
                engine=engine or self.engine,
            )
            results.append(r)
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARIA Web Implementer")
    parser.add_argument("query", help=(
        'Natural language query. Examples:\n'
        '  "read https://fastapi.tiangolo.com and build a todo API"\n'
        '  "clone https://example.com"'
    ))
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    agent = WebImplementerAgent()  # no engine in CLI mode
    result = agent.run_nl(args.query)
    print(json.dumps(result, indent=2, default=str))
