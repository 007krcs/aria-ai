"""
ARIA — Activity Trainer & Personalization Engine
==================================================
Learns who the user is from what they actually do — without sending a byte
of data anywhere. All state lives in data/ on the local machine.

What it builds:
  • UserProfile — preferred topics, communication style, working hours,
    expertise per domain, favourite apps, response length preference
  • Interaction history — every query/response pair with quality rating
  • App usage log — which apps, how long, what action type

What it does with that data:
  • personalize_prompt()    — injects user prefs into any system prompt
  • predict_next_task()     — time + pattern based next-action prediction
  • get_personalized_greeting() — time-aware, style-aware greeting
  • generate_training_pairs()   — mine high-quality Q&A for fine-tuning
  • export_finetune_dataset()   — write Ollama-ready JSONL
  • auto_schedule_finetune()    — trigger training when enough data arrives

Privacy controls:
  • enable_tracking(False)  — freezes all writes instantly
  • export_my_data()        — dump everything to a single JSON file
  • delete_my_data()        — wipe all stored data (irreversible)

NL interface (run_nl):
  "what are my preferences?"  → pretty-print UserProfile
  "show my activity"          → last N interactions summary
  "generate training data"    → export JSONL and report count
  "predict next task"         → show prediction
  "greeting"                  → personalised greeting for right now
"""

import json
import re
import time
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR        = PROJECT_ROOT / "data"
PROFILE_FILE    = DATA_DIR / "user_profile.json"
INTERACTIONS_FILE = DATA_DIR / "interactions.jsonl"
APP_USAGE_FILE  = DATA_DIR / "app_usage.jsonl"
FINETUNE_FILE   = DATA_DIR / "training" / "activity_finetune.jsonl"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "training").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    preferred_topics:      list  = field(default_factory=list)
    communication_style:   str   = "balanced"     # concise | detailed | technical | casual
    frequent_tasks:        list  = field(default_factory=list)
    working_hours:         dict  = field(default_factory=dict)  # {"Mon": {"start": 9, "end": 18}}
    language:              str   = "en"
    expertise_level:       dict  = field(default_factory=dict)  # {"python": "expert", ...}
    favorite_apps:         list  = field(default_factory=list)
    response_length_pref:  str   = "medium"        # short | medium | long
    name:                  str   = ""
    timezone:              str   = "UTC"
    created_at:            str   = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at:            str   = field(default_factory=lambda: datetime.utcnow().isoformat())
    total_interactions:    int   = 0
    avg_rating:            float = 0.0


@dataclass
class Interaction:
    query:        str
    response:     str
    timestamp:    str  = field(default_factory=lambda: datetime.utcnow().isoformat())
    user_rating:  Optional[float] = None   # 1–5 stars
    task_type:    Optional[str]   = None
    duration_ms:  int  = 0
    session_id:   str  = ""


@dataclass
class AppUsageRecord:
    app_name:    str
    duration_s:  float
    action_type: str   # open | close | focus | interact
    timestamp:   str   = field(default_factory=lambda: datetime.utcnow().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, engine: str = "llama3.2", temperature: float = 0.3,
                     system: str = "") -> str:
    try:
        import requests
        payload: dict = {
            "model": engine, "prompt": prompt,
            "stream": False, "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as exc:
        console.print(f"[red][Trainer] Ollama error: {exc}[/red]")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class ActivityTrainer:
    """
    Learns the user's preferences and work patterns from their ARIA usage.
    Completely local, privacy-first. All data lives in data/.

    Usage:
        trainer = ActivityTrainer()
        trainer.record_interaction("Write a regex for email", "Here's the pattern...", user_rating=5)
        profile = trainer.build_user_profile()
        print(trainer.get_personalized_greeting())
    """

    DEFAULT_ENGINE = "llama3.2"

    def __init__(self, engine: Optional[str] = None, tracking: bool = True):
        self._engine          = engine or self.DEFAULT_ENGINE
        self._tracking        = tracking
        self._profile: Optional[UserProfile] = None
        self._lock            = threading.Lock()
        self._finetune_count  = 0  # tracks new pairs since last training run
        self._load_profile()

    # ── privacy controls ──────────────────────────────────────────────────────

    def enable_tracking(self, enabled: bool) -> None:
        """Toggle all data recording on or off."""
        self._tracking = enabled
        console.print(
            f"[{'green' if enabled else 'yellow'}][Trainer] "
            f"Tracking {'enabled' if enabled else 'DISABLED'}.[/{'green' if enabled else 'yellow'}]"
        )

    def export_my_data(self, output_path: Optional[str] = None) -> Path:
        """
        Export all stored data (profile + interactions + app usage) to one JSON file.
        Returns the output path.
        """
        out = Path(output_path) if output_path else (DATA_DIR / "aria_my_data_export.json")

        interactions = []
        if INTERACTIONS_FILE.exists():
            with open(INTERACTIONS_FILE, encoding="utf-8") as f:
                for line in f:
                    try:
                        interactions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        app_usage = []
        if APP_USAGE_FILE.exists():
            with open(APP_USAGE_FILE, encoding="utf-8") as f:
                for line in f:
                    try:
                        app_usage.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        export = {
            "profile":      asdict(self._profile) if self._profile else {},
            "interactions": interactions,
            "app_usage":    app_usage,
            "exported_at":  datetime.utcnow().isoformat(),
        }
        out.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green][Trainer] Data exported → {out}[/green]")
        return out

    def delete_my_data(self) -> None:
        """
        Permanently delete all stored activity data.
        Profile, interactions, app usage, and fine-tune data are all removed.
        """
        for p in [PROFILE_FILE, INTERACTIONS_FILE, APP_USAGE_FILE, FINETUNE_FILE]:
            if p.exists():
                p.unlink()
                console.print(f"[yellow][Trainer] Deleted {p.name}[/yellow]")
        self._profile         = UserProfile()
        self._finetune_count  = 0
        console.print("[yellow][Trainer] All personal data deleted.[/yellow]")

    # ── recording ─────────────────────────────────────────────────────────────

    def record_interaction(self, query: str, response: str,
                            user_rating: Optional[float] = None,
                            task_type: Optional[str] = None,
                            duration_ms: int = 0) -> None:
        """
        Store a query/response interaction.

        Parameters
        ----------
        query        : The user's input
        response     : ARIA's response
        user_rating  : 1–5 stars (None if not rated)
        task_type    : e.g. 'code', 'search', 'writing', 'math'
        duration_ms  : Time taken to generate response
        """
        if not self._tracking:
            return

        interaction = Interaction(
            query=query, response=response,
            user_rating=user_rating, task_type=task_type or self._classify_task(query),
            duration_ms=duration_ms,
            session_id=self._session_id(),
        )

        with self._lock:
            with open(INTERACTIONS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(interaction), ensure_ascii=False) + "\n")

        # increment fine-tune counter for rated interactions
        if user_rating and user_rating >= 4:
            self._finetune_count += 1

        # async profile update every 10 interactions
        if self._should_update_profile():
            threading.Thread(target=self._async_profile_update, daemon=True).start()

    def record_app_usage(self, app_name: str, duration_s: float,
                          action_type: str = "interact") -> None:
        """
        Log that the user used an application.

        Parameters
        ----------
        app_name    : e.g. 'chrome', 'vscode', 'terminal'
        duration_s  : Seconds spent in the app
        action_type : 'open' | 'close' | 'focus' | 'interact'
        """
        if not self._tracking:
            return

        record = AppUsageRecord(app_name=app_name.lower().strip(),
                                duration_s=duration_s, action_type=action_type)
        with self._lock:
            with open(APP_USAGE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    # ── profile building ──────────────────────────────────────────────────────

    def build_user_profile(self) -> UserProfile:
        """
        Analyse all stored interactions and app usage to build/update UserProfile.
        Combines statistical analysis with LLM-assisted inference.
        Returns the updated UserProfile.
        """
        interactions = self._load_interactions()
        app_usage    = self._load_app_usage()

        if not interactions:
            console.print("[yellow][Trainer] No interactions yet — returning default profile.[/yellow]")
            return self._profile or UserProfile()

        profile = self._profile or UserProfile()

        # ── topics ──
        topics = self._infer_topics(interactions)
        if topics:
            profile.preferred_topics = topics

        # ── communication style ──
        profile.communication_style = self._infer_comm_style(interactions)

        # ── frequent tasks ──
        task_types = [i.get("task_type") for i in interactions if i.get("task_type")]
        if task_types:
            counter                = Counter(task_types)
            profile.frequent_tasks = [t for t, _ in counter.most_common(5)]

        # ── working hours ──
        profile.working_hours = self._infer_working_hours(interactions)

        # ── expertise per topic ──
        for topic in profile.preferred_topics[:5]:
            topic_interactions = [
                i for i in interactions
                if topic.lower() in i.get("query", "").lower()
            ]
            if topic_interactions:
                profile.expertise_level[topic] = self.infer_expertise(
                    topic, [i.get("query", "") for i in topic_interactions]
                )

        # ── response length preference ──
        profile.response_length_pref = self._infer_length_pref(interactions)

        # ── favourite apps ──
        if app_usage:
            app_times: dict = defaultdict(float)
            for rec in app_usage:
                app_times[rec.get("app_name", "")] += rec.get("duration_s", 0)
            profile.favorite_apps = [
                app for app, _ in sorted(app_times.items(), key=lambda x: -x[1])[:5]
                if app
            ]

        # ── language ──
        profile.language = self._infer_language(interactions)

        # ── stats ──
        ratings = [i["user_rating"] for i in interactions if i.get("user_rating")]
        profile.avg_rating          = round(sum(ratings) / len(ratings), 2) if ratings else 0.0
        profile.total_interactions  = len(interactions)
        profile.updated_at          = datetime.utcnow().isoformat()

        self._profile = profile
        self._save_profile()
        console.print(f"[green][Trainer] Profile updated — {len(interactions)} interactions analysed.[/green]")
        return profile

    # ── prompt personalisation ────────────────────────────────────────────────

    def personalize_prompt(self, base_prompt: str,
                            user_profile: Optional[UserProfile] = None) -> str:
        """
        Inject user-specific context into a base system prompt.

        The returned prompt includes:
        - Communication style directive
        - Response length directive
        - Relevant expertise context
        - Language preference
        """
        p = user_profile or self._profile or UserProfile()

        style_map = {
            "concise":   "Be brief and to the point. Avoid padding.",
            "detailed":  "Give thorough, complete answers with context.",
            "technical": "Use precise technical language. Assume expertise.",
            "casual":    "Be friendly and conversational. Avoid jargon.",
            "balanced":  "Be clear and appropriately detailed.",
        }
        length_map = {
            "short":  "Keep responses under 3 sentences unless critical detail is required.",
            "medium": "Aim for focused paragraphs. Don't over-explain.",
            "long":   "Provide comprehensive answers. Include examples.",
        }

        style_directive  = style_map.get(p.communication_style, style_map["balanced"])
        length_directive = length_map.get(p.response_length_pref, length_map["medium"])
        lang_directive   = f"Respond in {p.language.upper()}." if p.language != "en" else ""

        expertise_lines = []
        for domain, level in list(p.expertise_level.items())[:3]:
            expertise_lines.append(f"  - {domain}: {level}")
        expertise_block = (
            "User expertise:\n" + "\n".join(expertise_lines) if expertise_lines else ""
        )

        personalisation = "\n".join(filter(None, [
            "--- User Preferences ---",
            style_directive,
            length_directive,
            lang_directive,
            expertise_block,
            "------------------------",
        ]))

        return f"{personalisation}\n\n{base_prompt}"

    # ── training data ─────────────────────────────────────────────────────────

    def generate_training_pairs(self, min_rating: float = 4.0) -> list[dict]:
        """
        Export high-quality (rating >= min_rating) interaction pairs as JSONL records.
        Returns list of {"prompt": ..., "response": ...} dicts.
        """
        interactions = self._load_interactions()
        pairs = []
        for interaction in interactions:
            rating = interaction.get("user_rating")
            if rating is not None and float(rating) >= min_rating:
                pairs.append({
                    "prompt":    interaction.get("query", ""),
                    "response":  interaction.get("response", ""),
                    "metadata": {
                        "rating":   rating,
                        "task_type": interaction.get("task_type"),
                        "timestamp": interaction.get("timestamp"),
                    },
                })
        console.print(f"[green][Trainer] {len(pairs)} training pairs at rating ≥ {min_rating}[/green]")
        return pairs

    def export_finetune_dataset(self, output_path: Optional[str] = None) -> Path:
        """
        Write all high-quality pairs to an Ollama-compatible JSONL file.
        Returns the path written.
        """
        pairs = self.generate_training_pairs()
        out   = Path(output_path) if output_path else FINETUNE_FILE
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            for pair in pairs:
                # strip metadata — Ollama fine-tune format is just prompt/response
                f.write(json.dumps(
                    {"prompt": pair["prompt"], "response": pair["response"]},
                    ensure_ascii=False,
                ) + "\n")

        console.print(f"[green][Trainer] Fine-tune dataset → {out} ({len(pairs)} pairs)[/green]")
        return out

    # ── inference helpers ─────────────────────────────────────────────────────

    def infer_expertise(self, topic: str, query_history: list[str]) -> str:
        """
        Judge user expertise in *topic* from question complexity.
        Returns 'beginner' | 'intermediate' | 'expert'.
        """
        if not query_history:
            return "intermediate"

        sample = query_history[-20:]  # last 20 queries on this topic
        text   = "\n".join(f"- {q}" for q in sample)

        prompt = (
            f"Based on these questions about {topic}, rate the user's expertise level.\n\n"
            f"Questions:\n{text}\n\n"
            f"Criteria:\n"
            f"  beginner: basic how-to questions, no assumed knowledge\n"
            f"  intermediate: implementation questions, knows fundamentals\n"
            f"  expert: architecture decisions, edge cases, optimisation\n\n"
            f"Reply with ONLY one word: beginner / intermediate / expert"
        )
        raw = _ollama_generate(prompt, engine=self._engine)

        for level in ("expert", "intermediate", "beginner"):
            if level in raw.lower():
                return level
        return "intermediate"

    def predict_next_task(self) -> dict:
        """
        Predict what the user will likely do next based on:
          - Time of day / day of week patterns
          - Recency of task types
          - Sequential task chains observed in history

        Returns {"task": str, "confidence": float, "reason": str}
        """
        interactions = self._load_interactions()
        if len(interactions) < 5:
            return {"task": "general query", "confidence": 0.3,
                    "reason": "Not enough history yet."}

        now          = datetime.utcnow()
        hour         = now.hour
        day_name     = now.strftime("%A")

        # tasks done in this time window historically
        window_tasks: list[str] = []
        for rec in interactions:
            try:
                ts   = datetime.fromisoformat(rec["timestamp"])
                diff = abs(ts.hour - hour)
                if diff <= 1 and ts.strftime("%A") == day_name:
                    if rec.get("task_type"):
                        window_tasks.append(rec["task_type"])
            except (KeyError, ValueError):
                pass

        # last N task types
        recent_tasks = [
            rec["task_type"] for rec in interactions[-10:]
            if rec.get("task_type")
        ]

        if window_tasks:
            counter     = Counter(window_tasks)
            best_task   = counter.most_common(1)[0][0]
            confidence  = min(0.9, counter.most_common(1)[0][1] / max(len(window_tasks), 1))
            reason      = f"You usually do '{best_task}' on {day_name}s around {hour}:00."
        elif recent_tasks:
            # next in sequential chain heuristic
            counter     = Counter(recent_tasks)
            best_task   = counter.most_common(1)[0][0]
            confidence  = 0.55
            reason      = f"Based on your last {len(recent_tasks)} interactions."
        else:
            best_task   = "general query"
            confidence  = 0.3
            reason      = "Pattern not established yet."

        return {"task": best_task, "confidence": round(confidence, 2), "reason": reason}

    def get_personalized_greeting(self) -> str:
        """
        Generate a time-aware, preference-aware greeting for the user.
        Considers: time of day, day of week, working hours, name, style.
        """
        profile = self._profile or UserProfile()
        now     = datetime.utcnow()
        hour    = now.hour
        day     = now.strftime("%A")

        # time-of-day salutation
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Hey"  # late night casual

        name_part = f", {profile.name}" if profile.name else ""

        # style adaptation
        if profile.communication_style == "casual":
            base = f"Hey{name_part}! How can I help you today? 👋"
        elif profile.communication_style == "technical":
            base = f"{time_greeting}{name_part}. Ready when you are."
        elif profile.communication_style == "concise":
            base = f"{time_greeting}{name_part}."
        else:
            base = f"{time_greeting}{name_part}! What can I help you with?"

        # upcoming task hint
        prediction = self.predict_next_task()
        if prediction["confidence"] >= 0.6:
            task_hint = f" Looks like you might want to work on {prediction['task']}."
        else:
            task_hint = ""

        # working-hours awareness
        today_hours = profile.working_hours.get(day[:3], {})
        end_hour    = today_hours.get("end", 18)
        if hour >= end_hour and end_hour > 0:
            overtime_hint = " (It's past your usual end time — I'll keep it brief.)"
        else:
            overtime_hint = ""

        return base + task_hint + overtime_hint

    # ── auto fine-tune scheduler ──────────────────────────────────────────────

    def auto_schedule_finetune(self, engine: Optional[str] = None,
                                threshold: int = 100) -> bool:
        """
        Check if enough new high-quality pairs have accumulated.
        If yes, trigger an Ollama fine-tune run in a background thread.

        Returns True if training was triggered, False otherwise.
        """
        if self._finetune_count < threshold:
            console.print(
                f"[dim][Trainer] Fine-tune check: {self._finetune_count}/{threshold} pairs.[/dim]"
            )
            return False

        console.print(
            f"[cyan][Trainer] {self._finetune_count} pairs collected — "
            f"triggering fine-tune run.[/cyan]"
        )
        eng = engine or self._engine

        def _run():
            dataset_path = self.export_finetune_dataset()
            # Try to call NOVA's LoRA trainer if available
            try:
                from agents.nova_self_train import OnlineLoraTrainer
                trainer = OnlineLoraTrainer()
                trainer.train_on_file(str(dataset_path))
            except ImportError:
                console.print(
                    "[yellow][Trainer] NOVA LoRA trainer not found. "
                    "Dataset saved for manual training.[/yellow]"
                )
            except Exception as exc:
                console.print(f"[red][Trainer] Fine-tune error: {exc}[/red]")

        threading.Thread(target=_run, daemon=True).start()
        self._finetune_count = 0
        return True

    # ── NL interface ──────────────────────────────────────────────────────────

    def run_nl(self, query: str) -> str:
        """
        Natural language interface.

        Supported queries:
          "what are my preferences?"    → profile summary
          "show my activity"            → recent interactions
          "generate training data"      → export JSONL
          "predict next task"           → next task prediction
          "greeting"                    → personalised greeting
          "delete my data"              → confirm + wipe
          "export my data"              → data export
        """
        q = query.lower().strip()

        if any(k in q for k in ("prefer", "profile", "settings", "who am i", "about me")):
            return self._format_profile()

        if any(k in q for k in ("activity", "history", "interactions", "show")):
            return self._format_activity()

        if any(k in q for k in ("training", "fine-tune", "finetune", "export train")):
            path = self.export_finetune_dataset()
            pairs = self.generate_training_pairs()
            return f"Training data exported: {len(pairs)} pairs → {path}"

        if any(k in q for k in ("next task", "predict", "what will")):
            p = self.predict_next_task()
            return (
                f"Predicted next task: {p['task']} "
                f"(confidence {p['confidence']:.0%})\n"
                f"Reason: {p['reason']}"
            )

        if any(k in q for k in ("greeting", "hello", "good morning", "good evening")):
            return self.get_personalized_greeting()

        if "export my data" in q:
            path = self.export_my_data()
            return f"Your data has been exported to {path}"

        if "delete my data" in q:
            return (
                "This will permanently delete all stored activity data. "
                "To confirm, call trainer.delete_my_data() directly."
            )

        if "rebuild profile" in q or "update profile" in q:
            profile = self.build_user_profile()
            return f"Profile rebuilt. {profile.total_interactions} interactions processed."

        # fallback: use LLM to answer from profile context
        profile_json = json.dumps(asdict(self._profile), indent=2) if self._profile else "{}"
        prompt = (
            f"You are ARIA's activity tracking system. Answer this query using the "
            f"user's profile data below.\n\nUser profile:\n{profile_json}\n\nQuery: {query}"
        )
        return _ollama_generate(prompt, engine=self._engine)

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_profile(self) -> None:
        if PROFILE_FILE.exists():
            try:
                data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
                self._profile = UserProfile(**{
                    k: v for k, v in data.items()
                    if k in UserProfile.__dataclass_fields__
                })
                return
            except Exception as exc:
                console.print(f"[yellow][Trainer] Could not load profile: {exc}[/yellow]")
        self._profile = UserProfile()

    def _save_profile(self) -> None:
        if self._profile:
            PROFILE_FILE.write_text(
                json.dumps(asdict(self._profile), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _load_interactions(self) -> list[dict]:
        if not INTERACTIONS_FILE.exists():
            return []
        interactions = []
        with open(INTERACTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        interactions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return interactions

    def _load_app_usage(self) -> list[dict]:
        if not APP_USAGE_FILE.exists():
            return []
        records = []
        with open(APP_USAGE_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _classify_task(self, query: str) -> str:
        """Quick heuristic task classifier — no LLM call needed."""
        q = query.lower()
        patterns = {
            "code":     ["code", "function", "class", "debug", "error", "implement",
                          "python", "javascript", "sql", "bash", "script"],
            "search":   ["search", "find", "look up", "what is", "who is", "where is"],
            "writing":  ["write", "draft", "email", "letter", "essay", "summarise",
                          "summarize", "rewrite"],
            "math":     ["calculate", "how many", "what is", "formula", "solve", "equation"],
            "planning": ["plan", "schedule", "organise", "organize", "todo", "task list"],
            "analysis": ["analyse", "analyze", "compare", "explain why", "pros and cons"],
            "creative": ["story", "poem", "creative", "imagine", "generate idea"],
        }
        for task, keywords in patterns.items():
            if any(kw in q for kw in keywords):
                return task
        return "general"

    def _infer_topics(self, interactions: list[dict]) -> list[str]:
        """Extract top topics from query corpus using LLM."""
        if not interactions:
            return []
        sample = interactions[-50:]
        queries = "\n".join(f"- {i.get('query', '')[:100]}" for i in sample if i.get("query"))

        prompt = (
            f"Analyse these user queries and identify the top 5 subject topics "
            f"the user is interested in. Return as a comma-separated list.\n\n"
            f"Queries:\n{queries}\n\n"
            f"Example output: python programming, machine learning, productivity, finance, cooking\n"
            f"Topics:"
        )
        raw    = _ollama_generate(prompt, engine=self._engine)
        topics = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return topics[:5]

    def _infer_comm_style(self, interactions: list[dict]) -> str:
        """Infer communication style preference from rated interactions."""
        if not interactions:
            return "balanced"

        # use highly-rated responses to infer style
        good = [i for i in interactions if (i.get("user_rating") or 0) >= 4]
        if not good:
            good = interactions[-20:]

        avg_response_len = sum(len(i.get("response", "")) for i in good) / max(len(good), 1)
        tech_terms       = ["function", "parameter", "algorithm", "struct", "async",
                             "interface", "schema", "endpoint", "binary", "latency"]

        tech_count = sum(
            1 for i in good
            for term in tech_terms if term in i.get("query", "").lower()
        )
        tech_ratio = tech_count / max(len(good), 1)

        if tech_ratio > 0.4:
            return "technical"
        if avg_response_len < 200:
            return "concise"
        if avg_response_len > 800:
            return "detailed"
        return "balanced"

    def _infer_working_hours(self, interactions: list[dict]) -> dict:
        """
        Infer typical working hours per day of week from interaction timestamps.
        Returns e.g. {"Mon": {"start": 9, "end": 18}, ...}
        """
        hours_by_day: dict = defaultdict(list)
        for rec in interactions:
            try:
                ts = datetime.fromisoformat(rec["timestamp"])
                day_abbr = ts.strftime("%a")  # Mon, Tue, ...
                hours_by_day[day_abbr].append(ts.hour)
            except (KeyError, ValueError):
                pass

        working_hours: dict = {}
        for day, hours in hours_by_day.items():
            if len(hours) >= 3:
                working_hours[day] = {
                    "start": int(sorted(hours)[len(hours) // 5]),   # 20th percentile
                    "end":   int(sorted(hours)[int(len(hours) * 0.8)]),  # 80th percentile
                }
        return working_hours

    def _infer_length_pref(self, interactions: list[dict]) -> str:
        """Infer response length preference from user's own query lengths and ratings."""
        good = [i for i in interactions if (i.get("user_rating") or 0) >= 4]
        if not good:
            good = interactions[-30:]
        if not good:
            return "medium"

        avg_resp = sum(len(i.get("response", "")) for i in good) / max(len(good), 1)
        if avg_resp < 300:
            return "short"
        if avg_resp > 1000:
            return "long"
        return "medium"

    def _infer_language(self, interactions: list[dict]) -> str:
        """
        Detect primary language from recent queries.
        Falls back to 'en' if detection unavailable.
        """
        recent_queries = [i.get("query", "") for i in interactions[-20:] if i.get("query")]
        if not recent_queries:
            return "en"
        sample = " ".join(recent_queries[:5])[:500]
        try:
            from langdetect import detect
            return detect(sample)
        except Exception:
            # heuristic: count non-ASCII chars
            non_ascii = sum(1 for c in sample if ord(c) > 127)
            if non_ascii / max(len(sample), 1) > 0.15:
                return "unknown"
            return "en"

    def _should_update_profile(self) -> bool:
        """Return True every 10 interactions to trigger async profile rebuild."""
        interactions = self._load_interactions()
        return len(interactions) % 10 == 0 and len(interactions) > 0

    def _async_profile_update(self) -> None:
        """Background profile rebuild — won't block the main thread."""
        try:
            self.build_user_profile()
        except Exception as exc:
            console.print(f"[red][Trainer] Async profile update failed: {exc}[/red]")

    def _session_id(self) -> str:
        """Generate a daily session ID."""
        return datetime.utcnow().strftime("%Y%m%d")

    def _format_profile(self) -> str:
        """Pretty-print the current user profile."""
        p = self._profile or UserProfile()
        lines = [
            "── ARIA User Profile ──────────────────────",
            f"  Communication style : {p.communication_style}",
            f"  Response length     : {p.response_length_pref}",
            f"  Language            : {p.language}",
            f"  Topics              : {', '.join(p.preferred_topics) or 'none yet'}",
            f"  Frequent tasks      : {', '.join(p.frequent_tasks) or 'none yet'}",
            f"  Favourite apps      : {', '.join(p.favorite_apps) or 'none yet'}",
            f"  Total interactions  : {p.total_interactions}",
            f"  Average rating      : {p.avg_rating}/5.0",
        ]
        if p.expertise_level:
            lines.append("  Expertise:")
            for domain, level in p.expertise_level.items():
                lines.append(f"    {domain}: {level}")
        if p.working_hours:
            lines.append("  Working hours:")
            for day, hours in p.working_hours.items():
                lines.append(f"    {day}: {hours.get('start',9)}:00 – {hours.get('end',18)}:00")
        lines.append("────────────────────────────────────────────")
        return "\n".join(lines)

    def _format_activity(self, n: int = 10) -> str:
        """Pretty-print recent interaction summary."""
        interactions = self._load_interactions()
        if not interactions:
            return "No interactions recorded yet."

        recent = interactions[-n:]
        lines  = [f"── Last {len(recent)} Interactions ─────────────"]
        for rec in reversed(recent):
            ts     = rec.get("timestamp", "")[:16].replace("T", " ")
            rating = f"★ {rec['user_rating']:.0f}" if rec.get("user_rating") else "unrated"
            task   = rec.get("task_type", "general")
            query  = rec.get("query", "")[:60]
            lines.append(f"  [{ts}] [{task}] [{rating}] {query}…")
        lines.append(f"\nTotal: {len(interactions)} interactions stored.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    trainer = ActivityTrainer()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ").strip()

    print(trainer.run_nl(query))


if __name__ == "__main__":
    main()
