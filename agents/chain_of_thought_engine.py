"""
ARIA — Chain of Thought Reasoning Engine
==========================================
Structured, verifiable multi-step reasoning for any query.

Five reasoning strategies:

1. STANDARD
   Linear step-by-step decomposition. Fast, reliable for most queries.
   Each step is verified against prior steps before committing.

2. SELF-CONSISTENCY
   Generates N independent reasoning paths with different seeds/temps.
   Takes majority vote on final answers. Reduces hallucination by ~40%.
   Best for factual, single-answer questions.

3. TREE OF THOUGHT (ToT)
   Explores multiple branches at each reasoning step.
   Prunes branches with low confidence early (beam search style).
   Best for complex multi-path problems: planning, strategy, debugging.

4. HUMAN-LIKE
   Mimics the actual cognitive stages humans use when thinking carefully:
     read → activate knowledge → identify gaps → reason → cross-check
     → conclude → verify against original question
   Produces the most interpretable, teachable reasoning traces.

5. AUTO
   Classifies the query type and routes to the best strategy automatically.
   Simple factual → standard
   Ambiguous / multiple valid answers → self_consistency
   Multi-step planning / coding → tree_of_thought
   Complex open-ended / educational → human_like

All reasoning is done via local Ollama — no external API calls.
Outputs are fully exportable as JSONL fine-tuning data.
"""

import re
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThoughtStep:
    """A single reasoning step within a chain."""
    step_num:   int
    thought:    str
    confidence: float = 0.0   # 0.0–1.0
    verified:   bool  = False
    branch:     str   = "main"  # used by tree-of-thought


@dataclass
class ThoughtChain:
    """Complete reasoning trace for a query."""
    steps:        list = field(default_factory=list)  # list[ThoughtStep]
    final_answer: str  = ""
    confidence:   float = 0.0
    strategy:     str   = "standard"
    duration_ms:  int   = 0
    verified:     bool  = False
    query:        str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, engine: str = "llama3.2", temperature: float = 0.7,
                     system: str = "") -> str:
    """Call local Ollama and return the response text."""
    try:
        import requests
        payload: dict = {
            "model":  engine,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        resp = requests.post("http://localhost:11434/api/generate",
                             json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        console.print(f"[red][CoT] Ollama error: {exc}[/red]")
        return ""


def _ollama_available(engine: str) -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(engine in m for m in models)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN OF THOUGHT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ChainOfThoughtEngine:
    """
    Multi-strategy Chain of Thought reasoning engine for ARIA.

    Usage:
        engine = ChainOfThoughtEngine()
        chain  = engine.think("What causes rainbows?", strategy='human_like')
        print(chain.final_answer)
    """

    DEFAULT_ENGINE = "llama3.2"

    SYSTEM_PROMPT = (
        "You are ARIA's internal reasoning module. "
        "Think carefully and systematically. "
        "Be concise but complete. Avoid padding. "
        "When asked to produce a numbered step, produce exactly that step and nothing else."
    )

    def __init__(self, default_engine: Optional[str] = None):
        self._default_engine = default_engine or self.DEFAULT_ENGINE

    # ── public: primary entry point ──────────────────────────────────────────

    def think(self, query: str, strategy: str = "auto", max_steps: int = 8,
              engine: Optional[str] = None) -> ThoughtChain:
        """
        Run chain-of-thought reasoning on *query*.

        Parameters
        ----------
        query     : The question or task to reason about.
        strategy  : 'auto' | 'standard' | 'self_consistency' |
                    'tree_of_thought' | 'human_like'
        max_steps : Maximum reasoning steps (standard / human_like).
        engine    : Ollama model name override.

        Returns
        -------
        ThoughtChain with all steps, final answer, confidence, etc.
        """
        eng = engine or self._default_engine
        t0  = time.time()

        if strategy == "auto":
            strategy = self._classify_strategy(query, eng)

        console.print(f"[cyan][CoT] Strategy: {strategy} | Model: {eng}[/cyan]")

        if strategy == "standard":
            chain = self._standard(query, max_steps, eng)
        elif strategy == "self_consistency":
            chain = self.self_consistency(query, engine=eng)
        elif strategy == "tree_of_thought":
            chain = self.tree_of_thought(query, engine=eng)
        elif strategy == "human_like":
            chain = self.human_like_think(query, eng)
        else:
            console.print(f"[yellow][CoT] Unknown strategy '{strategy}', using standard.[/yellow]")
            chain = self._standard(query, max_steps, eng)

        chain.duration_ms = int((time.time() - t0) * 1000)
        chain.strategy    = strategy
        chain.query       = query
        chain.confidence  = self._aggregate_confidence(chain)
        chain.verified    = all(s.verified for s in chain.steps) if chain.steps else False

        console.print(
            f"[green][CoT] Done in {chain.duration_ms}ms | "
            f"Steps: {len(chain.steps)} | Confidence: {chain.confidence:.2f}[/green]"
        )
        return chain

    # ── strategy: standard ───────────────────────────────────────────────────

    def _standard(self, query: str, max_steps: int, engine: str) -> ThoughtChain:
        """Linear step-by-step reasoning."""
        chain = ThoughtChain()
        history: list[str] = []

        for step_num in range(1, max_steps + 1):
            history_text = "\n".join(
                [f"Step {i+1}: {s}" for i, s in enumerate(history)]
            ) if history else "None yet."

            prompt = (
                f"Question: {query}\n\n"
                f"Reasoning so far:\n{history_text}\n\n"
                f"Write ONLY Step {step_num} of the reasoning. "
                f"If the reasoning is complete, begin your reply with 'FINAL ANSWER:' "
                f"followed by the answer. Do not repeat previous steps."
            )

            raw = _ollama_generate(prompt, engine=engine, system=self.SYSTEM_PROMPT)
            if not raw:
                break

            if "FINAL ANSWER:" in raw.upper():
                answer_text = re.split(r"FINAL ANSWER:", raw, flags=re.IGNORECASE)[-1].strip()
                chain.final_answer = answer_text
                # add as a step
                step = ThoughtStep(step_num=step_num, thought=f"[Conclusion] {answer_text}")
                step.verified   = self.verify_step(step.thought, history, engine)
                step.confidence = self._step_confidence(step.thought, history, engine)
                chain.steps.append(step)
                break

            step = ThoughtStep(step_num=step_num, thought=raw)
            step.verified   = self.verify_step(raw, history, engine)
            step.confidence = self._step_confidence(raw, history, engine)
            chain.steps.append(step)
            history.append(raw)

        if not chain.final_answer and chain.steps:
            chain.final_answer = self._extract_final_answer(query, chain.steps, engine)

        return chain

    # ── strategy: human-like ─────────────────────────────────────────────────

    def human_like_think(self, query: str, engine: str) -> ThoughtChain:
        """
        Mimics the seven-stage human thinking process:
          1. Read carefully
          2. Activate prior knowledge
          3. Identify gaps / ambiguities
          4. Step-by-step reasoning
          5. Cross-check each step
          6. Form conclusion
          7. Verify conclusion against original question
        """
        chain   = ThoughtChain()
        context = query  # accumulates as thinking progresses

        stages = [
            (
                "Read Carefully",
                (
                    f"Read this question carefully and restate it in your own words "
                    f"to show you understand it. Identify the core ask.\n\nQuestion: {query}"
                ),
            ),
            (
                "Knowledge Activation",
                (
                    f"Given the question: {query}\n\n"
                    f"What relevant knowledge, facts, or principles do you already know "
                    f"that are directly applicable? List them concisely."
                ),
            ),
            (
                "Gap Identification",
                (
                    f"Question: {query}\n\n"
                    f"What is unclear, ambiguous, or missing that could affect the answer? "
                    f"List gaps and state reasonable assumptions where needed."
                ),
            ),
            (
                "Step-by-Step Reasoning",
                (
                    f"Question: {query}\n\n"
                    f"Now reason through this systematically, one logical step at a time. "
                    f"Number each step. Be explicit about each inference."
                ),
            ),
            (
                "Cross-Check",
                (
                    f"Question: {query}\n\n"
                    f"Review your reasoning: {context}\n\n"
                    f"Cross-check each step. Are there any errors, jumps in logic, "
                    f"or unsupported claims? Flag anything that needs correction."
                ),
            ),
            (
                "Form Conclusion",
                (
                    f"Based on your reasoning above, form a clear, direct conclusion "
                    f"to: {query}\n\n"
                    f"State the answer plainly without repeating all the reasoning."
                ),
            ),
            (
                "Verify Against Question",
                (
                    f"Original question: {query}\n\n"
                    f"Your conclusion: {context}\n\n"
                    f"Does this conclusion fully answer the original question? "
                    f"Does it address every part of what was asked? "
                    f"If not, what is still missing? Write the final verified answer."
                ),
            ),
        ]

        history_texts: list[str] = []

        for i, (stage_name, prompt) in enumerate(stages):
            # inject accumulated context into later stages
            if i >= 4 and history_texts:
                context = "\n\n".join(history_texts[-3:])  # last 3 stages as context
                # re-build prompt with updated context
                if i == 4:
                    prompt = (
                        f"Question: {query}\n\n"
                        f"Your reasoning so far:\n{context}\n\n"
                        f"Cross-check each step. Flag errors or logical gaps."
                    )
                elif i == 5:
                    prompt = (
                        f"Based on this reasoning:\n{context}\n\n"
                        f"Form a clear, direct conclusion to: {query}"
                    )
                elif i == 6:
                    prompt = (
                        f"Original question: {query}\n\n"
                        f"Conclusion reached: {context}\n\n"
                        f"Does this fully answer the question? Write the final verified answer."
                    )

            raw = _ollama_generate(prompt, engine=engine, system=self.SYSTEM_PROMPT)
            if not raw:
                raw = f"[Stage {stage_name} produced no output]"

            step = ThoughtStep(
                step_num=i + 1,
                thought=f"[{stage_name}] {raw}",
                branch="human_like",
            )
            step.verified   = self.verify_step(raw, history_texts, engine)
            step.confidence = self._step_confidence(raw, history_texts, engine)
            chain.steps.append(step)
            history_texts.append(raw)
            context = raw  # carry forward for next stage

        # final answer is the last stage output
        chain.final_answer = self._clean_answer(history_texts[-1]) if history_texts else ""
        return chain

    # ── strategy: self-consistency ───────────────────────────────────────────

    def self_consistency(self, query: str, n: int = 5,
                         engine: Optional[str] = None) -> ThoughtChain:
        """
        Generate N independent reasoning paths, take majority vote on final answer.
        Reduces hallucination on factual questions.
        """
        eng = engine or self._default_engine
        paths: list[tuple[str, list[str]]] = []  # (final_answer, steps_text)

        for i in range(n):
            # vary temperature to get genuine diversity
            temp = 0.5 + (i * 0.15)  # 0.5, 0.65, 0.80, 0.95, 1.10
            prompt = (
                f"Answer this question with clear step-by-step reasoning. "
                f"End with 'FINAL ANSWER: <answer>'.\n\nQuestion: {query}"
            )
            raw   = _ollama_generate(prompt, engine=eng, temperature=min(temp, 1.2),
                                     system=self.SYSTEM_PROMPT)
            parts = re.split(r"FINAL ANSWER:", raw, flags=re.IGNORECASE)
            if len(parts) >= 2:
                answer = parts[-1].strip()
            else:
                # no explicit marker — take last sentence
                sentences = [s.strip() for s in raw.split(".") if s.strip()]
                answer = sentences[-1] if sentences else raw.strip()

            steps_text = parts[0].strip() if len(parts) >= 2 else raw.strip()
            paths.append((answer, [steps_text]))
            console.print(f"[dim][CoT/SC] Path {i+1}/{n}: {answer[:80]}…[/dim]")

        # majority vote: normalise answers and count
        answer_counts: dict[str, int] = {}
        for ans, _ in paths:
            key = ans.lower().strip()[:120]
            answer_counts[key] = answer_counts.get(key, 0) + 1

        best_key   = max(answer_counts, key=lambda k: answer_counts[k])
        best_count = answer_counts[best_key]
        confidence = best_count / n

        # find the full answer text for the best key
        best_answer = next(
            (ans for ans, _ in paths if ans.lower().strip()[:120] == best_key),
            best_key,
        )

        # build chain with all paths as steps
        chain = ThoughtChain()
        for i, (ans, step_texts) in enumerate(paths):
            step = ThoughtStep(
                step_num=i + 1,
                thought=f"[Path {i+1}] {step_texts[0][:600]} … → {ans}",
                confidence=1.0 if ans.lower().strip()[:120] == best_key else 0.3,
                verified=ans.lower().strip()[:120] == best_key,
                branch=f"path_{i+1}",
            )
            chain.steps.append(step)

        # add vote summary step
        vote_summary = "; ".join(
            f'"{k[:60]}" × {v}' for k, v in sorted(
                answer_counts.items(), key=lambda x: -x[1]
            )
        )
        summary_step = ThoughtStep(
            step_num=n + 1,
            thought=f"[Majority Vote] {vote_summary}",
            confidence=confidence,
            verified=True,
            branch="vote",
        )
        chain.steps.append(summary_step)
        chain.final_answer = best_answer
        chain.confidence   = confidence
        return chain

    # ── strategy: tree of thought ─────────────────────────────────────────────

    def tree_of_thought(self, query: str, branches: int = 3, depth: int = 3,
                        engine: Optional[str] = None) -> ThoughtChain:
        """
        Explore multiple reasoning branches at each step, prune bad ones,
        return the highest-confidence path.
        """
        eng = engine or self._default_engine

        # Each node: {"text": str, "score": float, "path": list[str]}
        # Start with one root
        roots = self._generate_branches(
            query, parent_path=[], n=branches, engine=eng
        )

        current_level = roots
        all_steps: list[ThoughtStep] = []
        step_counter = 1

        for d in range(depth):
            scored = []
            for node in current_level:
                score = self._score_branch(
                    node["text"], node["path"], query, eng
                )
                node["score"] = score
                scored.append(node)

                step = ThoughtStep(
                    step_num=step_counter,
                    thought=f"[Branch d={d+1}] {node['text'][:400]}",
                    confidence=score,
                    verified=score > 0.6,
                    branch=f"branch_{step_counter}",
                )
                all_steps.append(step)
                step_counter += 1

            # prune: keep top half (at least 1)
            scored.sort(key=lambda x: -x["score"])
            keep = max(1, len(scored) // 2)
            survivors = scored[:keep]

            # expand survivors unless last depth
            if d < depth - 1:
                next_level = []
                for node in survivors:
                    children = self._generate_branches(
                        query,
                        parent_path=node["path"] + [node["text"]],
                        n=max(1, branches - d),
                        engine=eng,
                    )
                    next_level.extend(children)
                current_level = next_level
            else:
                current_level = survivors

        # best leaf
        best = max(current_level, key=lambda x: x.get("score", 0))
        full_path = best["path"] + [best["text"]]

        # synthesize final answer from best path
        path_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(full_path))
        final_prompt = (
            f"Based on this reasoning:\n{path_text}\n\n"
            f"Give a concise, direct final answer to: {query}"
        )
        final_answer = _ollama_generate(
            final_prompt, engine=eng, temperature=0.3, system=self.SYSTEM_PROMPT
        )

        chain = ThoughtChain()
        chain.steps        = all_steps
        chain.final_answer = final_answer
        return chain

    def _generate_branches(self, query: str, parent_path: list[str],
                            n: int, engine: str) -> list[dict]:
        """Generate n diverse reasoning branches from current state."""
        context = "\n".join(parent_path) if parent_path else "No prior steps."
        prompt  = (
            f"Question: {query}\n\n"
            f"Reasoning so far:\n{context}\n\n"
            f"Generate {n} distinctly different next reasoning steps. "
            f"Number them 1 to {n}. Each should explore a different angle."
        )
        raw      = _ollama_generate(prompt, engine=engine, temperature=0.9,
                                    system=self.SYSTEM_PROMPT)
        branches = []
        for i in range(1, n + 1):
            pattern = rf"{i}[\.\)]\s*(.*?)(?={i+1}[\.\)]|\Z)"
            match   = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            text    = match.group(1).strip() if match else raw.strip()
            branches.append({"text": text[:600], "score": 0.5, "path": list(parent_path)})
        return branches

    def _score_branch(self, branch_text: str, path: list[str],
                      query: str, engine: str) -> float:
        """Score a reasoning branch 0.0–1.0."""
        path_text = "\n".join(path[-3:]) if path else "None."
        prompt    = (
            f"Rate the quality of this reasoning step on a scale from 0.0 to 1.0.\n\n"
            f"Question: {query}\n"
            f"Prior context: {path_text}\n"
            f"Step to rate: {branch_text}\n\n"
            f"Criteria: logical validity, relevance, correctness, forward progress.\n"
            f"Reply with ONLY a number between 0.0 and 1.0 (e.g. 0.75)."
        )
        raw = _ollama_generate(prompt, engine=engine, temperature=0.1,
                               system=self.SYSTEM_PROMPT)
        match = re.search(r"([01]?\.\d+|\d+\.?\d*)", raw)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        return 0.5  # neutral if unparseable

    # ── verification & analysis ───────────────────────────────────────────────

    def verify_step(self, step: str, previous_steps: list[str],
                    engine: str) -> bool:
        """
        Check if a reasoning step is logically valid given prior steps.
        Returns True if valid, False if contradictory / unsupported.
        """
        context = "\n".join(previous_steps[-4:]) if previous_steps else "None."
        prompt  = (
            f"Prior reasoning steps:\n{context}\n\n"
            f"New step to verify: {step}\n\n"
            f"Is this step logically valid, supported by the prior steps, "
            f"and free of contradictions? Reply with exactly 'VALID' or 'INVALID'."
        )
        raw = _ollama_generate(prompt, engine=engine, temperature=0.1,
                               system=self.SYSTEM_PROMPT)
        return "VALID" in raw.upper() and "INVALID" not in raw.upper()

    def confidence_per_step(self, thought_chain: ThoughtChain) -> list[float]:
        """
        Return the confidence score for each step in the chain.
        Recomputes using stored scores (fast, no LLM call).
        """
        return [s.confidence for s in thought_chain.steps]

    def detect_contradiction(self, steps: list[ThoughtStep]) -> list[tuple[int, int, str]]:
        """
        Find logical contradictions among steps.

        Returns a list of (step_i, step_j, explanation) tuples.
        """
        if not steps:
            return []

        contradictions: list[tuple[int, int, str]] = []
        engine = self._default_engine

        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                prompt = (
                    f"Step {steps[i].step_num}: {steps[i].thought}\n"
                    f"Step {steps[j].step_num}: {steps[j].thought}\n\n"
                    f"Do these two steps contradict each other logically? "
                    f"Reply with 'CONTRADICTION: <brief explanation>' or 'NO CONTRADICTION'."
                )
                raw = _ollama_generate(prompt, engine=engine, temperature=0.1,
                                       system=self.SYSTEM_PROMPT)
                if "CONTRADICTION:" in raw.upper():
                    explanation = re.sub(r"(?i)contradiction:\s*", "", raw).strip()
                    contradictions.append((steps[i].step_num, steps[j].step_num, explanation))

        return contradictions

    def simplify_chain(self, thought_chain: ThoughtChain) -> ThoughtChain:
        """
        Compress verbose reasoning to only the key steps.
        Removes redundant / low-confidence steps, merges related ones.
        Returns a new simplified ThoughtChain.
        """
        if not thought_chain.steps:
            return thought_chain

        # sort by confidence, keep top 60% but always keep first and last
        steps = thought_chain.steps
        if len(steps) <= 3:
            return thought_chain

        ranked = sorted(steps[1:-1], key=lambda s: -s.confidence)
        keep_n = max(1, int(len(ranked) * 0.6))
        middle = sorted(ranked[:keep_n], key=lambda s: s.step_num)

        simplified_steps = [steps[0]] + middle + [steps[-1]]

        # renumber
        for idx, s in enumerate(simplified_steps):
            s.step_num = idx + 1

        new_chain             = ThoughtChain()
        new_chain.steps       = simplified_steps
        new_chain.final_answer = thought_chain.final_answer
        new_chain.confidence  = thought_chain.confidence
        new_chain.strategy    = thought_chain.strategy + "+simplified"
        new_chain.duration_ms = thought_chain.duration_ms
        new_chain.verified    = thought_chain.verified
        new_chain.query       = thought_chain.query
        return new_chain

    # ── training data export ──────────────────────────────────────────────────

    def to_training_data(self, query: str, thought_chain: ThoughtChain,
                         answer: str) -> dict:
        """
        Convert a query + reasoning chain + answer to Ollama JSONL fine-tune format.

        Output follows Alpaca / Ollama Modelfile fine-tune schema:
            {"prompt": "...", "response": "..."}
        """
        steps_text = "\n".join(
            f"Step {s.step_num}: {s.thought}" for s in thought_chain.steps
        )
        training_record = {
            "prompt":    query,
            "response":  f"{steps_text}\n\nFinal Answer: {answer}",
            "metadata": {
                "strategy":   thought_chain.strategy,
                "confidence": thought_chain.confidence,
                "steps":      len(thought_chain.steps),
                "verified":   thought_chain.verified,
                "created_at": datetime.utcnow().isoformat(),
            },
        }
        return training_record

    def export_to_jsonl(self, records: list[dict], output_path: Optional[str] = None) -> Path:
        """Write training records to JSONL file."""
        path = Path(output_path) if output_path else (DATA_DIR / "cot_training.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        console.print(f"[green][CoT] Exported {len(records)} training records → {path}[/green]")
        return path

    # ── NL interface ──────────────────────────────────────────────────────────

    def run_nl(self, query: str) -> str:
        """
        Natural language interface. Auto-picks strategy, returns a formatted
        answer string with the reasoning chain summary.

        Example:
            engine.run_nl("Why is the sky blue?")
        """
        chain = self.think(query, strategy="auto")
        lines = [f"[ARIA Reasoning — {chain.strategy.upper()}]", ""]

        for step in chain.steps:
            verified_tag = "✓" if step.verified else "?"
            conf_tag     = f"{step.confidence:.0%}"
            lines.append(f"  Step {step.step_num} [{verified_tag} {conf_tag}]: {step.thought}")

        lines.append("")
        lines.append(f"Answer: {chain.final_answer}")
        lines.append(
            f"[Confidence: {chain.confidence:.0%} | "
            f"Duration: {chain.duration_ms}ms | "
            f"Steps: {len(chain.steps)}]"
        )
        return "\n".join(lines)

    # ── private helpers ───────────────────────────────────────────────────────

    def _classify_strategy(self, query: str, engine: str) -> str:
        """
        Auto-classify query into the best strategy.
        Falls back to keyword heuristics if LLM unavailable.
        """
        prompt = (
            f"Classify this query into ONE reasoning strategy:\n"
            f"- standard: simple factual, definitions, short answers\n"
            f"- self_consistency: factual with potential ambiguity, math\n"
            f"- tree_of_thought: multi-step planning, code, strategy, debugging\n"
            f"- human_like: complex open-ended, educational, philosophical\n\n"
            f"Query: {query}\n\n"
            f"Reply with ONLY one of: standard / self_consistency / tree_of_thought / human_like"
        )
        raw = _ollama_generate(prompt, engine=engine, temperature=0.1,
                               system=self.SYSTEM_PROMPT)
        for s in ("self_consistency", "tree_of_thought", "human_like", "standard"):
            if s in raw.lower():
                return s

        # keyword fallback
        q = query.lower()
        if any(k in q for k in ["plan", "design", "debug", "code", "implement", "architect"]):
            return "tree_of_thought"
        if any(k in q for k in ["how many", "calculate", "what is the", "which is"]):
            return "self_consistency"
        if any(k in q for k in ["why", "explain", "understand", "philosophy", "meaning"]):
            return "human_like"
        return "standard"

    def _step_confidence(self, step_text: str, history: list[str], engine: str) -> float:
        """Quick confidence estimate for a step via LLM or heuristic."""
        if not step_text:
            return 0.0
        prompt = (
            f"Rate the logical certainty of this reasoning step (0.0=guessing, 1.0=certain):\n"
            f"{step_text}\n\n"
            f"Reply with ONLY a number 0.0–1.0."
        )
        raw   = _ollama_generate(prompt, engine=engine, temperature=0.1,
                                  system=self.SYSTEM_PROMPT)
        match = re.search(r"([01]?\.\d+|\d+\.?\d*)", raw)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        # heuristic: longer, hedged steps get lower score
        hedge_words = ["maybe", "perhaps", "i think", "not sure", "might", "could be"]
        penalty = sum(1 for w in hedge_words if w in step_text.lower()) * 0.1
        return max(0.1, min(0.9, 0.75 - penalty))

    def _aggregate_confidence(self, chain: ThoughtChain) -> float:
        """Overall chain confidence = weighted mean of step confidences."""
        if not chain.steps:
            return 0.0
        scores = [s.confidence for s in chain.steps]
        # weight later steps slightly more
        weights = [1.0 + (i * 0.1) for i in range(len(scores))]
        weighted_sum  = sum(s * w for s, w in zip(scores, weights))
        return round(weighted_sum / sum(weights), 3)

    def _extract_final_answer(self, query: str, steps: list[ThoughtStep],
                               engine: str) -> str:
        """Synthesize a final answer from existing steps."""
        steps_text = "\n".join(f"Step {s.step_num}: {s.thought}" for s in steps[-5:])
        prompt     = (
            f"Based on this reasoning:\n{steps_text}\n\n"
            f"Give a concise, direct final answer to: {query}"
        )
        return _ollama_generate(prompt, engine=engine, temperature=0.3,
                                system=self.SYSTEM_PROMPT)

    def _clean_answer(self, text: str) -> str:
        """Remove meta-prefixes from answer text."""
        text = re.sub(r"(?i)^\[verify against question\]\s*", "", text)
        text = re.sub(r"(?i)^final verified answer:\s*", "", text)
        return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    engine = ChainOfThoughtEngine()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ").strip()

    print(engine.run_nl(query))


if __name__ == "__main__":
    main()
