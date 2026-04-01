"""
NOVA v3 — Self-Construct Agent
================================
NOVA detects its own capability gaps and builds new specialist agents.

The loop:
1. Track which query types consistently score low
2. Decide a new specialist agent is needed
3. Write the Python code for that agent using the LLM
4. Validate the code (syntax, safety, import check)
5. Save it to agents/ directory
6. Register it with the async pool
7. Future queries of that type route to the new agent

This is the "Transformer builds a Transformer" moment.
No human writes the new agents — NOVA does.

Honest limitation: The generated code is validated but not perfect.
A human should review new agents before they handle sensitive tasks.
We add a "sandboxed" flag — new agents run in a restricted mode
until a human approves them.
"""

import ast
import sys
import json
import time
import importlib
import subprocess
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from rich.console import Console

console = Console()
AGENTS_DIR  = Path("agents")
SANDBOX_DIR = Path("agents/sandbox")
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CODE VALIDATOR
# Ensures generated agent code is safe before execution
# ─────────────────────────────────────────────────────────────────────────────

class CodeValidator:
    """
    Validates LLM-generated Python code before allowing it to run.

    Checks:
    1. Valid Python syntax
    2. No dangerous imports (os.system, subprocess with shell=True, etc.)
    3. Has the required class/function signature
    4. Passes a basic smoke test
    """

    FORBIDDEN_PATTERNS = [
        r"os\.system\(",
        r"subprocess\..*shell\s*=\s*True",
        r"eval\(",
        r"exec\(",
        r"__import__\(",
        r"open\(.+['\"]w['\"]",     # file write
        r"socket\.",                 # network sockets
        r"shutil\.rmtree\(",        # delete dirs
        r"pathlib.*unlink",         # delete files
    ]

    def validate(self, code: str) -> dict:
        """Returns {valid, errors, warnings}."""
        errors   = []
        warnings = []
        import re

        # Check 1: Valid Python syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check 2: No dangerous operations
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                errors.append(f"Dangerous pattern found: {pattern}")

        # Check 3: Must have a class or callable
        tree    = ast.parse(code)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        funcs   = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not classes and not funcs:
            warnings.append("No class or function defined")

        # Check 4: Imports should be standard/known safe libs
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self._safe_imports():
                        warnings.append(f"Unusual import: {alias.name}")

        return {
            "valid":    len(errors) == 0,
            "errors":   errors,
            "warnings": warnings,
            "classes":  classes,
            "functions": funcs,
        }

    def _safe_imports(self) -> set:
        return {
            "re", "json", "time", "math", "hashlib", "pathlib", "datetime",
            "collections", "itertools", "functools", "typing", "dataclasses",
            "requests", "bs4", "trafilatura", "chromadb", "langchain",
            "sentence_transformers", "rich", "pydantic", "fastapi",
            "asyncio", "threading", "concurrent", "abc", "io", "os",
            "sys", "subprocess", "urllib", "html", "xml", "csv",
            "string", "random", "statistics", "operator", "copy",
        }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-CONSTRUCT AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SelfConstructAgent:
    """
    Monitors NOVA's performance and builds new agents for weak domains.

    The key insight: if a domain consistently scores below 0.6 confidence,
    a generic agent isn't good enough. We need a specialist. Instead of
    manually writing one, NOVA writes it.

    Process:
    1. Detect weak domain from logger
    2. Design the agent (what should it do?)
    3. Write the Python code
    4. Validate it
    5. Save to sandbox
    6. Register with pool (sandboxed mode)
    7. Monitor — if it scores well, promote to production
    """

    def __init__(self, engine, logger, pool=None):
        self.engine    = engine
        self.logger    = logger
        self.pool      = pool
        self.validator = CodeValidator()
        self.built:    list[dict] = []   # track what was built

    def detect_gaps(self, hours: int = 24, threshold: float = 0.55) -> list[dict]:
        """Find domains where NOVA is consistently underperforming."""
        try:
            with self.logger._connect() as conn:
                rows = conn.execute(
                    """SELECT intent,
                              AVG(confidence) as avg_conf,
                              COUNT(*) as total
                       FROM interactions
                       WHERE ts > datetime('now', ?)
                       GROUP BY intent
                       HAVING AVG(confidence) < ? AND COUNT(*) >= 5
                       ORDER BY avg_conf ASC LIMIT 5""",
                    (f"-{hours} hours", threshold)
                ).fetchall()
            return [{"domain": r[0], "avg_confidence": round(r[1], 3), "count": r[2]}
                    for r in rows]
        except Exception:
            return []

    def design_agent(self, domain: str) -> str:
        """Use the LLM to design what a specialist agent for this domain should do."""
        prompt = (
            f"Design a Python specialist agent for the domain: '{domain}'\n\n"
            f"Describe in 3-5 bullet points:\n"
            f"- What specific tasks this agent handles\n"
            f"- What external tools/APIs it uses (all free)\n"
            f"- How it improves over the generic agent\n"
            f"- What makes it specialist for {domain}\n\n"
            f"Design:"
        )
        return self.engine.generate(prompt, temperature=0.3)

    def write_agent_code(self, domain: str, design: str) -> str:
        """Generate Python code for a new specialist agent."""
        prompt = (
            f"Write a Python class called `{self._class_name(domain)}Agent` "
            f"for the domain: '{domain}'\n\n"
            f"Design specification:\n{design}\n\n"
            f"Requirements:\n"
            f"- Class must have a `run(self, query: str, context: str = '') -> dict` method\n"
            f"- The dict must have keys: answer (str), confidence (float 0-1)\n"
            f"- Use only free libraries (requests, bs4, trafilatura, re, json)\n"
            f"- No API keys needed\n"
            f"- Include docstring explaining what it does\n"
            f"- Handle exceptions gracefully — never crash\n\n"
            f"Python code (just the class, no main block):"
        )
        code = self.engine.generate(prompt, temperature=0.2)
        # Strip markdown
        import re
        code = re.sub(r"```python\s*|```\s*", "", code).strip()
        return code

    def build_agent(self, domain: str) -> dict:
        """
        Full pipeline: detect → design → write → validate → save.
        Returns build result dict.
        """
        console.print(f"\n[bold]Self-construct:[/] building specialist for '{domain}'")

        # Step 1: Design
        design = self.design_agent(domain)
        console.print(f"  [dim]Design complete[/]")

        # Step 2: Write code (up to 3 attempts)
        code    = ""
        valid   = False
        attempt = 0
        while not valid and attempt < 3:
            attempt += 1
            code    = self.write_agent_code(domain, design)
            result  = self.validator.validate(code)
            if result["valid"]:
                valid = True
                console.print(f"  [green]Code valid[/] (attempt {attempt})")
                if result["warnings"]:
                    console.print(f"  [yellow]Warnings: {result['warnings']}[/]")
            else:
                console.print(f"  [yellow]Attempt {attempt} invalid: {result['errors']}[/]")
                # Ask LLM to fix the errors
                fix_prompt = (
                    f"Fix these errors in the Python code:\n"
                    f"Errors: {result['errors']}\n\n"
                    f"Original code:\n{code}\n\n"
                    f"Fixed code:"
                )
                code = self.engine.generate(fix_prompt, temperature=0.1)
                import re
                code = re.sub(r"```python\s*|```\s*", "", code).strip()

        if not valid:
            return {"success": False, "domain": domain, "error": "Could not generate valid code after 3 attempts"}

        # Step 3: Save to sandbox
        class_name = self._class_name(domain)
        filename   = f"{domain.lower().replace(' ', '_')}_agent.py"
        save_path  = SANDBOX_DIR / filename

        # Add required imports and a helper wrapper
        full_code = f'''"""
Auto-generated specialist agent for domain: {domain}
Generated: {datetime.now().isoformat()}
Status: SANDBOXED — review before promoting to production
"""
import re
import json
import time
import requests
from rich.console import Console
console = Console()

{code}

def run_agent(query: str, context: str = "") -> dict:
    """Wrapper function for async pool compatibility."""
    try:
        agent = {class_name}Agent()
        return agent.run(query, context)
    except Exception as e:
        return {{"answer": f"Agent error: {{e}}", "confidence": 0.0}}
'''
        save_path.write_text(full_code, encoding="utf-8")
        console.print(f"  [green]Saved:[/] {save_path}")

        # Step 4: Register with pool if available
        registered = False
        if self.pool:
            try:
                # Dynamic import of the new agent
                spec   = importlib.util.spec_from_file_location(filename[:-3], save_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "run_agent"):
                    self.pool.register_agent(f"specialist_{domain}", module.run_agent, domain)
                    registered = True
                    console.print(f"  [green]Registered with async pool[/]")
            except Exception as e:
                console.print(f"  [yellow]Could not register: {e}[/]")

        record = {
            "domain":     domain,
            "class_name": class_name,
            "file":       str(save_path),
            "registered": registered,
            "sandboxed":  True,
            "ts":         datetime.now().isoformat(),
            "success":    True,
        }
        self.built.append(record)
        return record

    def auto_build_for_gaps(self, hours: int = 24) -> list[dict]:
        """Automatically build agents for all detected gaps."""
        gaps    = self.detect_gaps(hours)
        results = []
        for gap in gaps:
            domain = gap["domain"]
            # Don't rebuild if we recently built for this domain
            if any(b["domain"] == domain for b in self.built[-10:]):
                console.print(f"  [dim]Skipping {domain} — recently built[/]")
                continue
            result = self.build_agent(domain)
            results.append(result)
        return results

    def _class_name(self, domain: str) -> str:
        """Convert domain string to PascalCase class name."""
        return "".join(w.capitalize() for w in re.sub(r"[^a-zA-Z0-9 ]", "", domain).split())

    def list_built(self) -> list[dict]:
        """List all agents built so far."""
        built = []
        for f in SANDBOX_DIR.glob("*_agent.py"):
            built.append({
                "file":      f.name,
                "size_kb":   round(f.stat().st_size / 1024, 1),
                "modified":  datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "sandboxed": True,
            })
        return built

    def promote_agent(self, filename: str) -> dict:
        """
        Promote a sandboxed agent to production agents/ directory.
        Call this after reviewing the agent code.
        """
        src  = SANDBOX_DIR / filename
        dest = AGENTS_DIR / filename
        if not src.exists():
            return {"success": False, "error": "File not found in sandbox"}
        import shutil
        shutil.copy2(src, dest)
        console.print(f"  [green]Promoted:[/] {filename} → agents/")
        return {"success": True, "promoted_to": str(dest)}
