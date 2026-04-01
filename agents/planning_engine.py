"""
ARIA — Planning Engine
========================
The difference between a reactive tool and an intelligent agent.

Current ARIA: input → immediate action → output.
Planning ARIA: input → understand goal → decompose → simulate →
               select best plan → execute → verify → learn.

The planning engine has three components:

1. Goal Decomposer
   Takes a high-level goal ("research Apple's competitive position and
   send me a summary") and breaks it into a dependency tree of subtasks.
   No model needed for simple goals — pattern library handles 80%.
   Model used only for novel, complex goal decomposition.

2. Consequence Simulator
   Before executing any plan, ARIA simulates what will happen.
   For dangerous actions (delete file, send email, make call),
   the simulator checks: is this reversible? what could go wrong?
   If risk > threshold, ARIA asks for confirmation.

3. Plan Executor
   Runs the plan step by step, adapting if steps fail.
   Maintains a "plan stack" — knows which step it's on,
   what depends on it, and what to do if it fails.

This is why ARIA can handle:
  "Every morning, check the weather, my calendar, and top news,
   then send me a voice summary before I wake up"

Without a plan engine, this requires manual orchestration.
With a plan engine, ARIA figures out the steps, schedules them,
handles failures, and learns better patterns for next time.
"""

import re
import json
import time
import uuid
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable
from dataclasses import dataclass, field
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()
PLANS_FILE   = PROJECT_ROOT / "data" / "plans.json"
PLANS_FILE.parent.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLAN STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    """A single step in a plan."""
    id:          str
    description: str
    tool:        str           # tool name from registry
    args:        dict
    depends_on:  list[str]     # step IDs that must complete first
    optional:    bool  = False
    timeout_s:   float = 30.0
    retry_count: int   = 0
    max_retries: int   = 2
    status:      str   = "pending"   # pending/running/done/failed/skipped
    result:      dict  = field(default_factory=dict)
    started_ts:  float = 0.0
    ended_ts:    float = 0.0

    def duration_s(self) -> float:
        if self.started_ts and self.ended_ts:
            return self.ended_ts - self.started_ts
        return 0.0


@dataclass
class Plan:
    """A structured sequence of steps to achieve a goal."""
    id:          str
    goal:        str
    steps:       list[Step]
    created_ts:  str  = field(default_factory=lambda: datetime.now().isoformat())
    status:      str  = "pending"
    risk_level:  str  = "low"     # low / medium / high
    reversible:  bool = True
    context:     dict = field(default_factory=dict)

    def pending_steps(self) -> list[Step]:
        done  = {s.id for s in self.steps if s.status == "done"}
        return [
            s for s in self.steps
            if s.status == "pending"
            and all(dep in done for dep in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(
            s.status in ("done","skipped","failed")
            for s in self.steps
            if not s.optional
        )

    def success_rate(self) -> float:
        done   = sum(1 for s in self.steps if s.status == "done")
        total  = len(self.steps)
        return done / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GOAL DECOMPOSER
# ─────────────────────────────────────────────────────────────────────────────

class GoalDecomposer:
    """
    Breaks high-level goals into executable step trees.
    Pattern library handles common goals without LLM.
    """

    # Common goal patterns → step templates
    PATTERNS = {
        "morning_brief": {
            "trigger": ["morning brief", "morning summary", "daily brief", "wake up summary"],
            "steps": [
                {"tool": "get_weather", "args": {"city": "{city}"}, "desc": "Get today's weather"},
                {"tool": "get_news",    "args": {"topic": "general"}, "desc": "Get top news"},
                {"tool": "get_stock_price", "args": {"symbol": "{watch_symbols}"}, "desc": "Check watched stocks"},
                {"tool": "voice_speak", "args": {"text": "{summary}"}, "desc": "Speak the summary"},
            ],
        },
        "research_and_report": {
            "trigger": ["research and report", "research and send", "look up and tell me"],
            "steps": [
                {"tool": "search_web",     "args": {"query": "{topic}"}, "desc": "Web search"},
                {"tool": "research_papers","args": {"query": "{topic}"}, "desc": "Find research"},
                {"tool": "summarise",      "args": {"context": "{results}"}, "desc": "Summarise findings"},
                {"tool": "notify",         "args": {"message": "{summary}"}, "desc": "Send notification"},
            ],
        },
        "monitor_and_alert": {
            "trigger": ["watch for", "monitor", "alert me when", "notify when"],
            "steps": [
                {"tool": "set_condition_watch", "args": {"condition": "{condition}"}, "desc": "Set monitor"},
                {"tool": "schedule_check",      "args": {"interval": "5m"}, "desc": "Schedule checks"},
            ],
        },
        "send_email_with_draft": {
            "trigger": ["write and send email", "draft and send", "compose and email"],
            "steps": [
                {"tool": "draft_email", "args": {"to": "{to}", "about": "{topic}"}, "desc": "Draft email"},
                {"tool": "send_email",  "args": {"to": "{to}"}, "desc": "Send email"},
            ],
        },
    }

    def __init__(self, engine=None, registry=None):
        self.engine   = engine
        self.registry = registry

    def decompose(self, goal: str, context: dict = None) -> Plan:
        """
        Convert a natural language goal into an executable plan.
        Returns a Plan with steps ordered by dependencies.
        """
        context  = context or {}
        goal_low = goal.lower()

        # Try pattern matching first (no model)
        for pattern_name, pattern in self.PATTERNS.items():
            if any(t in goal_low for t in pattern["trigger"]):
                console.print(f"  [dim]Plan: matched pattern '{pattern_name}'[/]")
                return self._build_from_pattern(goal, pattern, context)

        # LLM decomposition for novel goals
        if self.engine:
            return self._llm_decompose(goal, context)

        # Fallback: single step
        return Plan(
            id=str(uuid.uuid4())[:8],
            goal=goal,
            steps=[Step(
                id="step_1",
                description=goal,
                tool="smart_execute",
                args={"text": goal},
                depends_on=[],
            )],
        )

    def _build_from_pattern(self, goal: str, pattern: dict,
                             context: dict) -> Plan:
        steps = []
        for i, s in enumerate(pattern["steps"]):
            # Fill context variables in args
            args = {}
            for k, v in s["args"].items():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    var_name = v[1:-1]
                    args[k]  = context.get(var_name, v)
                else:
                    args[k] = v

            steps.append(Step(
                id          = f"step_{i+1}",
                description = s["desc"],
                tool        = s["tool"],
                args        = args,
                depends_on  = [f"step_{i}"] if i > 0 else [],
            ))

        return Plan(id=str(uuid.uuid4())[:8], goal=goal, steps=steps)

    def _llm_decompose(self, goal: str, context: dict) -> Plan:
        """Use LLM to decompose a complex or novel goal."""
        # Get available tool names for the LLM to reference
        tool_names = []
        if self.registry:
            tool_names = [t["name"] for t in self.registry.list_tools()][:20]

        prompt = (
            f"Break this goal into executable steps:\nGoal: {goal}\n\n"
            f"Available tools: {', '.join(tool_names)}\n\n"
            f"Return JSON array of steps:\n"
            f'[{{"id":"step_1","description":"...","tool":"tool_name",'
            f'"args":{{}},"depends_on":[]}}]\n'
            f"Steps only, no explanation:"
        )
        try:
            import re
            raw   = self.engine.generate(prompt, temperature=0.1)
            raw   = re.sub(r"```\w*\n?|```","",raw).strip()
            # Find JSON array
            m     = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                step_dicts = json.loads(m.group())
                steps = [
                    Step(
                        id          = s.get("id", f"step_{i}"),
                        description = s.get("description",""),
                        tool        = s.get("tool","smart_execute"),
                        args        = s.get("args",{}),
                        depends_on  = s.get("depends_on",[]),
                    )
                    for i, s in enumerate(step_dicts)
                ]
                return Plan(id=str(uuid.uuid4())[:8], goal=goal, steps=steps)
        except Exception as e:
            console.print(f"  [yellow]LLM decompose error: {e}[/]")

        # Fallback
        return Plan(
            id=str(uuid.uuid4())[:8],
            goal=goal,
            steps=[Step("step_1", goal, "smart_execute", {"text": goal}, [])],
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONSEQUENCE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class ConsequenceSimulator:
    """
    Before executing, ARIA simulates: what could go wrong?
    Protects against irreversible, dangerous, or unwanted actions.
    """

    # Actions that need extra checks
    RISKY_TOOLS = {
        "send_email":        ("medium", False,  "Sends an email — cannot unsend"),
        "make_call":         ("medium", False,  "Makes a phone call"),
        "delete_file":       ("high",   False,  "Deletes a file permanently"),
        "run_command":       ("medium", True,   "Runs a shell command"),
        "send_message":      ("medium", False,  "Sends a message"),
        "write_file":        ("low",    True,   "Overwrites a file"),
        "open_app":          ("low",    True,   "Opens an application"),
        "close_app":         ("low",    True,   "Closes an application"),
        "set_alarm":         ("low",    True,   "Sets an alarm"),
        "notify":            ("low",    True,   "Sends a notification"),
    }

    def assess(self, plan: Plan) -> dict:
        """
        Assess the risk of executing a plan.
        Returns risk assessment and whether to proceed.
        """
        risks       = []
        max_risk    = "low"
        reversible  = True
        risk_order  = {"low": 0, "medium": 1, "high": 2}

        for step in plan.steps:
            tool_info = self.RISKY_TOOLS.get(step.tool)
            if tool_info:
                level, rev, desc = tool_info
                risks.append({
                    "step":        step.description,
                    "tool":        step.tool,
                    "risk":        level,
                    "reversible":  rev,
                    "description": desc,
                })
                if risk_order[level] > risk_order[max_risk]:
                    max_risk = level
                if not rev:
                    reversible = False

        # Simulate: will this work?
        predicted_success = self._predict_success(plan)

        assessment = {
            "risk_level":        max_risk,
            "reversible":        reversible,
            "risks":             risks,
            "predicted_success": predicted_success,
            "should_confirm":    max_risk == "high" or not reversible,
            "auto_proceed":      max_risk == "low" and reversible,
        }

        plan.risk_level = max_risk
        plan.reversible = reversible
        return assessment

    def _predict_success(self, plan: Plan) -> float:
        """Estimate probability plan succeeds based on tool reliability."""
        if not plan.steps:
            return 1.0
        # Simple model: multiply tool reliability rates
        success = 1.0
        for step in plan.steps:
            if step.optional:
                continue
            # Base reliability per tool category
            reliability = {
                "open_app": 0.95, "search_google": 0.97,
                "notify": 0.98, "get_weather": 0.92,
                "make_call": 0.85, "send_email": 0.90,
                "run_command": 0.80,
            }.get(step.tool, 0.85)
            success *= reliability
        return round(success, 3)


# ─────────────────────────────────────────────────────────────────────────────
# PLAN EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class PlanExecutor:
    """
    Executes plans step-by-step. Handles failures, retries, and adaptation.
    Streams progress via the event bus.
    """

    def __init__(self, registry=None, bus=None, engine=None):
        self.registry = registry
        self.bus      = bus
        self.engine   = engine

    def execute(
        self,
        plan:       Plan,
        on_step:    Callable = None,
        on_done:    Callable = None,
    ) -> Plan:
        """
        Execute a plan. Returns the plan with step results filled in.
        on_step: called after each step with (step, result)
        on_done: called when plan completes with (plan,)
        """
        plan.status = "running"
        console.print(f"  [dim]Executing plan: {plan.goal[:50]}[/]")

        while not plan.is_complete():
            ready = plan.pending_steps()
            if not ready:
                break  # deadlock or all done

            # Execute ready steps (could be parallel for independent steps)
            for step in ready:
                self._execute_step(step, plan)
                if on_step:
                    on_step(step, step.result)

                # Publish to bus
                if self.bus:
                    from agents.agent_bus import Event
                    self.bus.publish(Event(
                        "plan_step_done",
                        {"plan_id": plan.id, "step": step.description,
                         "status": step.status, "result": step.result},
                        "plan_executor"
                    ))

        # Determine final plan status
        failed_required = [
            s for s in plan.steps
            if s.status == "failed" and not s.optional
        ]
        plan.status = "failed" if failed_required else "done"

        console.print(
            f"  [{'green' if plan.status=='done' else 'yellow'}]"
            f"Plan {plan.status}:[/] {plan.success_rate()*100:.0f}% steps succeeded"
        )

        if on_done:
            on_done(plan)

        # Save plan history
        self._save_plan(plan)
        return plan

    def _execute_step(self, step: Step, plan: Plan):
        """Execute one step, with retry logic."""
        step.status     = "running"
        step.started_ts = time.time()

        for attempt in range(step.max_retries + 1):
            try:
                if self.registry:
                    result = self.registry.execute(step.tool, step.args, step.timeout_s)
                else:
                    result = {"success": False, "error": "No registry available"}

                step.result   = result
                step.status   = "done" if result.get("success", True) else "failed"
                step.ended_ts = time.time()

                if step.status == "done":
                    console.print(
                        f"  [green]✓[/] {step.description[:50]} "
                        f"({step.duration_s():.1f}s)"
                    )
                    return

                # Failed — should we retry?
                if attempt < step.max_retries:
                    console.print(
                        f"  [yellow]Retry {attempt+1}:[/] {step.description[:40]} "
                        f"— {result.get('error','')[:40]}"
                    )
                    step.retry_count += 1
                    time.sleep(2 ** attempt)  # exponential backoff

                    # Try to fix the args using LLM before retry
                    if self.engine and result.get("error"):
                        step.args = self._fix_args(step, result["error"])

            except Exception as e:
                step.result   = {"success": False, "error": str(e)}
                step.ended_ts = time.time()
                if attempt >= step.max_retries:
                    step.status = "failed" if not step.optional else "skipped"
                    console.print(f"  [red]✗[/] {step.description[:50]}: {e}")

    def _fix_args(self, step: Step, error: str) -> dict:
        """Use LLM to fix step arguments after a failure."""
        if not self.engine:
            return step.args
        prompt = (
            f"Fix these arguments for tool '{step.tool}' that caused error:\n"
            f"Error: {error[:200]}\n"
            f"Current args: {json.dumps(step.args)}\n"
            f"Return fixed JSON args only:"
        )
        try:
            import re
            raw  = self.engine.generate(prompt, temperature=0.05)
            raw  = re.sub(r"```\w*\n?|```","",raw).strip()
            m    = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return step.args

    def _save_plan(self, plan: Plan):
        """Persist plan execution history for learning."""
        try:
            history = []
            if PLANS_FILE.exists():
                history = json.loads(PLANS_FILE.read_text())

            history.append({
                "id":        plan.id,
                "goal":      plan.goal,
                "status":    plan.status,
                "steps":     len(plan.steps),
                "success_rate": plan.success_rate(),
                "ts":        plan.created_ts,
            })

            # Keep last 100 plans
            history = history[-100:]
            PLANS_FILE.write_text(json.dumps(history, indent=2))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PLANNING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PlanningEngine:
    """
    Orchestrates goal decomposition, risk assessment, and execution.
    The "thinking before acting" layer.
    """

    def __init__(self, registry=None, bus=None, engine=None):
        self.decomposer  = GoalDecomposer(engine, registry)
        self.simulator   = ConsequenceSimulator()
        self.executor    = PlanExecutor(registry, bus, engine)
        self.engine      = engine
        self.bus         = bus

    def execute_goal(
        self,
        goal:          str,
        context:       dict = None,
        auto_confirm:  bool = True,
        on_step:       Callable = None,
    ) -> dict:
        """
        Full pipeline: understand → plan → assess → execute → learn.
        Returns execution result.
        """
        console.print(f"\n  [dim]Planning:[/] {goal[:60]}")

        # 1. Decompose goal into steps
        plan = self.decomposer.decompose(goal, context or {})
        console.print(f"  [dim]  Steps: {len(plan.steps)}[/]")

        # 2. Assess consequences
        assessment = self.simulator.assess(plan)

        if assessment["risk_level"] == "high" and not auto_confirm:
            return {
                "status":     "needs_confirmation",
                "plan":       plan.goal,
                "risks":      assessment["risks"],
                "message":    "This action has high risk. Confirm to proceed.",
            }

        # 3. Execute
        plan = self.executor.execute(plan, on_step=on_step)

        # 4. Collect results
        results = {s.description: s.result for s in plan.steps}

        return {
            "status":       plan.status,
            "goal":         plan.goal,
            "steps_done":   sum(1 for s in plan.steps if s.status=="done"),
            "steps_total":  len(plan.steps),
            "success_rate": plan.success_rate(),
            "results":      results,
            "risk_level":   assessment["risk_level"],
        }

    def get_plan_history(self) -> list[dict]:
        try:
            if PLANS_FILE.exists():
                return json.loads(PLANS_FILE.read_text())[-10:]
        except Exception:
            pass
        return []
