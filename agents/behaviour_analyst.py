"""
ARIA — Behaviour Analyst + Psychology Profiler
================================================
Analyses usage patterns to understand HOW you work, not just WHAT you use.

What it detects:
- Peak focus hours (when you do your best deep work)
- Cognitive style (scanner vs deep-diver vs multitasker)
- Stress indicators (rapid app switching, short sessions, late nights)
- Flow state detection (long uninterrupted sessions)
- Distraction triggers (what breaks your focus)
- Weekly rhythm (your natural work cycle)
- Procrastination patterns
- Energy curve (when you're productive vs coasting)

Psychology models used (all evidence-based):
- Ultradian rhythm theory (90-min focus cycles)
- Flow state research (Csikszentmihalyi)
- Cognitive load indicators
- Context switching cost model
- Circadian productivity patterns
"""

import json
import sqlite3
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Optional
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()
DB_PATH      = PROJECT_ROOT / "logs" / "system_events.db"
PROFILE_PATH = PROJECT_ROOT / "data" / "user_profile.json"
PROFILE_PATH.parent.mkdir(exist_ok=True)


class BehaviourAnalyst:
    """
    Analyses system monitor data and builds a behavioural profile.
    Runs every 15 minutes when the monitor is active.
    """

    def __init__(self, engine=None):
        self.engine = engine   # LLM for narrative insights
        self._profile = self._load_profile()

    def _load_profile(self) -> dict:
        try:
            if PROFILE_PATH.exists():
                return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {
            "cognitive_style":    "unknown",
            "peak_hours":         [],
            "focus_capacity_min": 0,
            "distraction_apps":   [],
            "stress_score":       0.0,
            "productivity_score": 0.0,
            "last_updated":       None,
            "insights":           [],
            "predictions":        [],
        }

    def _save_profile(self):
        try:
            PROFILE_PATH.write_text(
                json.dumps(self._profile, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass

    # ── Core analysis methods ─────────────────────────────────────────────────

    def analyse_focus_capacity(self, days: int = 7) -> dict:
        """
        How long can you focus before switching?
        Based on focus_sessions table — sessions > 5 min in same app.
        """
        since = (datetime.now() - timedelta(days=days)).isoformat()
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT duration_s, app_name, ts_start
                    FROM focus_sessions
                    WHERE ts_start > ? AND interrupted = 0
                    ORDER BY duration_s DESC
                """, (since,)).fetchall()
        except Exception:
            return {}

        if not rows:
            return {"focus_sessions": 0}

        durations = [r[0] for r in rows]
        avg_min   = statistics.mean(durations) / 60
        max_min   = max(durations) / 60
        median_min= statistics.median(durations) / 60

        # Classify focus capacity
        if median_min >= 45:
            style = "deep_worker"
            desc  = "Strong deep focus capacity — can sustain 45+ min sessions"
        elif median_min >= 25:
            style = "moderate_focus"
            desc  = "Moderate focus — works in 25-45 min blocks"
        elif median_min >= 15:
            style = "sprint_worker"
            desc  = "Sprint style — 15-25 min intense bursts"
        else:
            style = "high_switcher"
            desc  = "High context switching — sessions under 15 min"

        self._profile["focus_capacity_min"] = round(median_min, 1)
        self._profile["cognitive_style"]    = style
        self._save_profile()

        return {
            "style":         style,
            "description":   desc,
            "avg_focus_min": round(avg_min, 1),
            "max_focus_min": round(max_min, 1),
            "median_min":    round(median_min, 1),
            "total_sessions": len(rows),
        }

    def find_peak_hours(self, days: int = 14) -> dict:
        """
        When are you most productive?
        Uses coding + document + design app time as proxy for productive work.
        """
        since = (datetime.now() - timedelta(days=days)).isoformat()
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT strftime('%H', ts_start) as hour,
                           SUM(duration_s) as total_s,
                           category
                    FROM app_sessions
                    WHERE ts_start > ?
                      AND category IN ('coding','document','design','learning')
                      AND app_name != 'idle'
                    GROUP BY hour
                    ORDER BY total_s DESC
                """, (since,)).fetchall()
        except Exception:
            return {}

        if not rows:
            return {"peak_hours": []}

        hour_totals = defaultdict(float)
        for row in rows:
            hour_totals[int(row[0])] += row[1]

        # Find top 3 productive hours
        sorted_hours = sorted(hour_totals.items(), key=lambda x: x[1], reverse=True)
        peak_hours   = [h for h, _ in sorted_hours[:3]]
        peak_hours.sort()

        # Classify time of day preference
        avg_peak = sum(peak_hours) / len(peak_hours) if peak_hours else 12
        if avg_peak < 10:
            chronotype = "early_bird"
            desc       = "Morning person — peak focus 6-10am"
        elif avg_peak < 14:
            chronotype = "midday_peak"
            desc       = "Midday focus — best work 10am-2pm"
        elif avg_peak < 18:
            chronotype = "afternoon_peak"
            desc       = "Afternoon peak — most productive 2-6pm"
        else:
            chronotype = "night_owl"
            desc       = "Night owl — peak performance in evening"

        self._profile["peak_hours"] = peak_hours
        self._save_profile()

        return {
            "peak_hours":       peak_hours,
            "chronotype":       chronotype,
            "description":      desc,
            "hourly_breakdown": dict(sorted_hours[:8]),
        }

    def detect_stress_indicators(self, hours: int = 24) -> dict:
        """
        Stress shows up in usage patterns:
        - Rapid app switching (context switching cost)
        - Short, fragmented sessions
        - Increased social media / entertainment use
        - Late night work
        - Lots of idle periods during work hours
        """
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        stress_score = 0.0
        indicators   = []

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row

                # Avg session length (short = stressed)
                avg_row = conn.execute("""
                    SELECT AVG(duration_s) as avg_s, COUNT(*) as count
                    FROM app_sessions
                    WHERE ts_start > ? AND app_name != 'idle'
                """, (since,)).fetchone()

                if avg_row and avg_row["avg_s"]:
                    avg_s = avg_row["avg_s"]
                    if avg_s < 30:
                        stress_score += 2.0
                        indicators.append(
                            f"Very short app sessions (avg {avg_s:.0f}s) — "
                            "high context switching suggests stress or distraction"
                        )
                    elif avg_s < 60:
                        stress_score += 1.0
                        indicators.append("Short sessions — moderate context switching")

                # Social media / entertainment ratio
                cat_rows = conn.execute("""
                    SELECT category, SUM(duration_s) as total_s
                    FROM app_sessions
                    WHERE ts_start > ?
                    GROUP BY category
                """, (since,)).fetchall()

                total = sum(r["total_s"] for r in cat_rows if r["total_s"])
                if total > 0:
                    escape_time = sum(
                        r["total_s"] for r in cat_rows
                        if r["category"] in ("social","media","gaming")
                    )
                    escape_ratio = escape_time / total
                    if escape_ratio > 0.4:
                        stress_score += 2.0
                        indicators.append(
                            f"High escape behaviour — {escape_ratio*100:.0f}% time on "
                            "social/media/gaming — possible avoidance pattern"
                        )
                    elif escape_ratio > 0.25:
                        stress_score += 1.0
                        indicators.append(
                            f"Above-average entertainment use — {escape_ratio*100:.0f}%"
                        )

                # Late night work (after 11pm)
                late_row = conn.execute("""
                    SELECT SUM(duration_s) FROM app_sessions
                    WHERE ts_start > ?
                      AND strftime('%H', ts_start) >= '23'
                      AND category IN ('coding','document','communication')
                """, (since,)).fetchone()
                if late_row and late_row[0] and late_row[0] > 1800:
                    stress_score += 1.5
                    indicators.append(
                        f"Late night work ({late_row[0]/60:.0f} min after 11pm) — "
                        "may indicate deadline pressure or poor boundaries"
                    )

                # App switch rate
                switch_row = conn.execute("""
                    SELECT COUNT(*) FROM app_sessions
                    WHERE ts_start > ? AND app_name != 'idle'
                """, (since,)).fetchone()
                if switch_row and switch_row[0]:
                    switches_per_hour = switch_row[0] / hours
                    if switches_per_hour > 30:
                        stress_score += 1.5
                        indicators.append(
                            f"High app switching rate — {switches_per_hour:.0f}/hour "
                            "(>30/hour suggests scattered attention)"
                        )

        except Exception as e:
            console.print(f"  [yellow]Stress analysis error: {e}[/]")

        stress_level = "low" if stress_score < 2 else "medium" if stress_score < 4 else "high"
        self._profile["stress_score"] = round(stress_score, 1)
        self._save_profile()

        return {
            "stress_score": round(stress_score, 2),
            "stress_level": stress_level,
            "indicators":   indicators,
        }

    def detect_distraction_triggers(self, days: int = 7) -> dict:
        """
        What breaks your focus? Detects which apps you switch TO
        when you interrupt a focus session.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT b.app_name as distraction, COUNT(*) as count,
                           AVG(a.duration_s) as prev_session_s
                    FROM app_sessions a
                    JOIN app_sessions b ON b.id = a.id + 1
                    WHERE a.ts_start > datetime('now', ?)
                      AND a.category IN ('coding','document','design')
                      AND b.category IN ('social','media','communication','browser')
                      AND a.duration_s < 600
                    GROUP BY distraction
                    ORDER BY count DESC LIMIT 10
                """, (f"-{days} days",)).fetchall()
        except Exception:
            return {"distractions": []}

        distractions = [
            {"app": r[0], "times": r[1],
             "avg_prev_focus_min": round(r[2]/60, 1)}
            for r in rows
        ]
        top_distraction_apps = [d["app"] for d in distractions[:5]]
        self._profile["distraction_apps"] = top_distraction_apps
        self._save_profile()

        return {
            "top_distractions": distractions,
            "insight": (
                f"You most often switch to {top_distraction_apps[0]} "
                f"when breaking focus" if top_distraction_apps else "Not enough data yet"
            ),
        }

    def build_psychology_profile(self) -> dict:
        """
        Full psychological profile based on usage data.
        References real cognitive psychology research.
        """
        focus   = self.analyse_focus_capacity()
        peaks   = self.find_peak_hours()
        stress  = self.detect_stress_indicators()
        distract= self.detect_distraction_triggers()

        # Cognitive style from multiple signals
        style_signals = []
        if focus.get("style") == "deep_worker":
            style_signals.append("Deep focus capacity suggests convergent thinking preference")
        if focus.get("style") == "high_switcher":
            style_signals.append("High switching may indicate divergent/creative thinking style")
        if peaks.get("chronotype") == "night_owl":
            style_signals.append("Night owl preference correlates with higher creative flexibility")
        if peaks.get("chronotype") == "early_bird":
            style_signals.append("Morning focus correlates with analytical/systematic thinking")

        profile = {
            "cognitive_style":    focus.get("style","unknown"),
            "chronotype":         peaks.get("chronotype","unknown"),
            "peak_hours":         peaks.get("peak_hours",[]),
            "avg_focus_min":      focus.get("avg_focus_min",0),
            "stress_level":       stress.get("stress_level","unknown"),
            "stress_score":       stress.get("stress_score",0),
            "main_distraction":   distract.get("insight",""),
            "style_signals":      style_signals,
            "stress_indicators":  stress.get("indicators",[]),
        }

        # LLM-generated narrative insight
        if self.engine:
            prompt = (
                f"Based on this person's computer usage data, write 2-3 insightful "
                f"sentences about their work style and psychology. Be specific, not generic.\n\n"
                f"Data: {json.dumps(profile, indent=2)}\n\n"
                f"Insight:"
            )
            try:
                profile["narrative"] = self.engine.generate(prompt, temperature=0.4)
            except Exception:
                profile["narrative"] = ""

        self._profile.update(profile)
        self._profile["last_updated"] = datetime.now().isoformat()
        self._save_profile()

        return profile

    def predict_next_action(self, current_app: str,
                            time_in_app_min: float) -> dict:
        """
        Predict what the user will do next based on historical patterns.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT b.app_name, b.category, COUNT(*) as count
                    FROM app_sessions a
                    JOIN app_sessions b ON b.id = a.id + 1
                    WHERE a.app_name = ?
                    GROUP BY b.app_name
                    ORDER BY count DESC LIMIT 5
                """, (current_app,)).fetchall()
        except Exception:
            return {"predictions": []}

        if not rows:
            return {"predictions": [], "confidence": 0.0}

        total  = sum(r[2] for r in rows)
        preds  = [
            {"app": r[0], "category": r[1],
             "probability": round(r[2]/total*100, 1)}
            for r in rows
        ]

        # Time-based prediction boost
        hour   = datetime.now().hour
        peak   = self._profile.get("peak_hours",[])
        in_peak= hour in peak

        return {
            "current_app":    current_app,
            "time_in_app_min": round(time_in_app_min,1),
            "in_peak_hours":  in_peak,
            "predictions":    preds,
            "confidence":     preds[0]["probability"] if preds else 0.0,
            "likely_next":    preds[0]["app"] if preds else "unknown",
        }

    def get_weekly_rhythm(self) -> dict:
        """Day-of-week productivity patterns."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT strftime('%w', ts_start) as dow,
                           SUM(CASE WHEN category IN ('coding','document','design','learning')
                                    THEN duration_s ELSE 0 END) as productive_s,
                           SUM(duration_s) as total_s
                    FROM app_sessions
                    WHERE ts_start > datetime('now', '-30 days')
                      AND app_name != 'idle'
                    GROUP BY dow ORDER BY dow
                """).fetchall()
        except Exception:
            return {}

        days = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
        return {
            "daily": [
                {"day": days[int(r[0])],
                 "productive_h": round(r[1]/3600, 2),
                 "total_h":      round(r[2]/3600, 2),
                 "efficiency":   round(r[1]/r[2]*100, 1) if r[2] else 0}
                for r in rows
            ]
        }

    def get_full_profile(self) -> dict:
        return self._profile
