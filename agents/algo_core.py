"""
ARIA Algorithm Core — Shared Algorithm Library
===============================================
Pure-Python implementations (no numpy/scipy required) used across all ARIA agents.

Classes:
    PatternEngine     — TF-IDF/BM25, semantic similarity, NER, intent, sentiment, complexity
    DecisionEngine    — weighted voting, Bayesian update, confidence intervals, BM25 ranking
    AdaptiveLearner   — Hebbian/Oja, EWMA, online update, KL-surprise, n-gram LM, perplexity
    CorrelationEngine — Pearson, Spearman, mutual information, causal score, knowledge graph
"""

from __future__ import annotations

import math
import re
import collections
import hashlib
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PatternEngine:
    """Text pattern recognition: TF-IDF/BM25, similarity, NER, intent, sentiment, complexity."""

    # ── TF-IDF with BM25 variant ──────────────────────────────────────────────

    @staticmethod
    def tfidf_score(text: str, corpus: List[str], k1: float = 1.5, b: float = 0.75) -> float:
        """
        BM25 score of `text` against `corpus`.
        Returns average BM25 score of the text document vs all corpus docs.
        """
        if not corpus or not text:
            return 0.0

        def tokenize(t: str) -> List[str]:
            return re.findall(r'\b\w+\b', t.lower())

        query_tokens = tokenize(text)
        corpus_tokens = [tokenize(doc) for doc in corpus]
        N = len(corpus_tokens)
        avg_dl = sum(len(d) for d in corpus_tokens) / max(N, 1)

        # IDF for each query term
        score = 0.0
        for term in query_tokens:
            df = sum(1 for doc in corpus_tokens if term in doc)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            # TF in text itself
            tf = query_tokens.count(term)
            dl = len(query_tokens)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            score += idf * tf_norm

        return score / max(len(query_tokens), 1)

    # ── Cosine similarity via char n-gram vectors ─────────────────────────────

    @staticmethod
    def _char_ngram_vector(text: str, n: int = 3) -> Dict[str, int]:
        """Build character n-gram frequency vector."""
        text = text.lower().strip()
        vec: Dict[str, int] = {}
        for i in range(len(text) - n + 1):
            gram = text[i:i + n]
            vec[gram] = vec.get(gram, 0) + 1
        return vec

    @classmethod
    def semantic_similarity(cls, a: str, b: str, n: int = 3) -> float:
        """
        Cosine similarity between two texts using character n-gram vectors.
        Returns 0.0–1.0 (no external dependencies).
        """
        if not a or not b:
            return 0.0
        va = cls._char_ngram_vector(a, n)
        vb = cls._char_ngram_vector(b, n)
        if not va or not vb:
            return 0.0

        # Dot product
        dot = sum(va.get(k, 0) * vb.get(k, 0) for k in va)
        mag_a = math.sqrt(sum(v * v for v in va.values()))
        mag_b = math.sqrt(sum(v * v for v in vb.values()))
        denom = mag_a * mag_b
        if denom == 0:
            return 0.0
        return min(1.0, dot / denom)

    # ── Fuzzy match (Levenshtein ratio) ───────────────────────────────────────

    @staticmethod
    def _levenshtein(s: str, t: str) -> int:
        """Compute Levenshtein edit distance."""
        m, n = len(s), len(t)
        if m < n:
            s, t, m, n = t, s, n, m
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if s[i - 1] == t[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
        return dp[n]

    @classmethod
    def fuzzy_match(cls, query: str, candidates: List[str], threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Return candidates whose Levenshtein ratio >= threshold.
        Returns list of (candidate, ratio) sorted best-first.
        """
        results = []
        q = query.lower()
        for cand in candidates:
            c = cand.lower()
            max_len = max(len(q), len(c), 1)
            dist = cls._levenshtein(q, c)
            ratio = 1.0 - dist / max_len
            if ratio >= threshold:
                results.append((cand, ratio))
        return sorted(results, key=lambda x: -x[1])

    # ── Regex-based NER ───────────────────────────────────────────────────────

    _NER_PATTERNS = {
        "number":   r'\b\d+(?:[,\.]\d+)*\b',
        "date":     r'\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)\b',
        "currency": r'\b(?:USD|EUR|GBP|INR|JPY|₹|\$|€|£)\s*\d+(?:[,\.]\d+)*(?:\s*(?:million|billion|thousand|crore|lakh))?\b',
        "email":    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        "url":      r'https?://[^\s<>"{}|\\^`\[\]]+',
        "name":     r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b',
    }

    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, List[str]]:
        """
        Regex-based NER: numbers, dates, currencies, emails, URLs, names.
        Returns dict mapping entity type to list of matches.
        """
        results: Dict[str, List[str]] = {}
        for ent_type, pattern in cls._NER_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                results[ent_type] = list(dict.fromkeys(matches))  # dedupe, preserve order
        return results

    # ── Intent detection ──────────────────────────────────────────────────────

    _INTENT_PATTERNS = {
        "question":  [r'\?$', r'\b(what|who|where|when|why|how|which|whose|whom)\b'],
        "command":   [r'^(open|run|execute|start|stop|kill|delete|remove|create|make|show|list|find|search|go|navigate|click|type|press)\b'],
        "complaint": [r'\b(not working|broken|error|fail|bug|crash|issue|problem|wrong|bad|terrible|awful|worst|hate|annoyed|frustrated)\b'],
        "request":   [r'\b(please|could you|can you|would you|help me|i need|i want|i\'d like|assist|do you know)\b'],
        "statement": [],  # fallback
    }

    @classmethod
    def detect_intent(cls, text: str) -> str:
        """
        Classify text intent: question / command / complaint / request / statement.
        Returns the best-matching intent label.
        """
        t = text.lower().strip()
        scores: Dict[str, int] = {k: 0 for k in cls._INTENT_PATTERNS}
        for intent, patterns in cls._INTENT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, t):
                    scores[intent] += 1
        # Exclude statement from competition unless nothing else matches
        top = max((k for k in scores if k != "statement"), key=lambda k: scores[k], default="statement")
        if scores.get(top, 0) == 0:
            return "statement"
        return top

    # ── Sentiment score ───────────────────────────────────────────────────────

    _POS_WORDS = {
        "good","great","excellent","amazing","wonderful","fantastic","love","like","best",
        "happy","glad","pleased","positive","helpful","useful","perfect","brilliant","awesome",
        "outstanding","superb","terrific","smart","clear","fast","reliable","safe","secure",
        "easy","simple","clean","accurate","valid","correct","right","true","yes","thanks",
        "thank","appreciate","nice","beautiful","elegant","powerful","strong","effective",
    }
    _NEG_WORDS = {
        "bad","terrible","awful","hate","dislike","worst","wrong","broken","error","fail",
        "failure","crash","bug","issue","problem","slow","hard","difficult","complex","ugly",
        "weak","dangerous","unsafe","insecure","invalid","incorrect","false","no","not",
        "never","nothing","useless","annoying","frustrating","boring","confusing","weird",
        "suspicious","malicious","fraud","scam","fake","misleading","spam",
    }
    _INTENSIFIERS = {"very","extremely","absolutely","completely","totally","highly","really"}
    _NEGATORS = {"not","no","never","neither","nor","hardly","barely","scarcely","without"}

    @classmethod
    def sentiment_score(cls, text: str) -> float:
        """
        Lexicon-based sentiment score from -1.0 (very negative) to +1.0 (very positive).
        Handles intensifiers and negators.
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        score = 0.0
        n = len(tokens)
        i = 0
        while i < n:
            tok = tokens[i]
            # Check window for negators and intensifiers
            window_start = max(0, i - 3)
            window = tokens[window_start:i]
            negated = any(w in cls._NEGATORS for w in window)
            intensified = any(w in cls._INTENSIFIERS for w in window)
            multiplier = 1.5 if intensified else 1.0
            if negated:
                multiplier *= -1.0

            if tok in cls._POS_WORDS:
                score += 1.0 * multiplier
            elif tok in cls._NEG_WORDS:
                score -= 1.0 * multiplier
            i += 1

        # Normalize to [-1, 1]
        if n == 0:
            return 0.0
        raw = score / math.sqrt(max(n, 1))
        return max(-1.0, min(1.0, raw))

    # ── Complexity / readability score ────────────────────────────────────────

    @staticmethod
    def complexity_score(text: str) -> float:
        """
        Flesch-Kincaid grade level variant. Returns approximate grade level (0–20+).
        Higher = more complex.
        """
        if not text or not text.strip():
            return 0.0

        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0.0

        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)

        # Count syllables: vowel groups per word
        def count_syllables(word: str) -> int:
            word = word.lower()
            vowels = re.findall(r'[aeiou]+', word)
            syl = len(vowels)
            if word.endswith('e') and syl > 1:
                syl -= 1
            return max(1, syl)

        total_syllables = sum(count_syllables(w) for w in words)

        asl = num_words / num_sentences       # avg sentence length
        asw = total_syllables / num_words     # avg syllables per word

        # Flesch-Kincaid Grade Level
        grade = 0.39 * asl + 11.8 * asw - 15.59
        return max(0.0, round(grade, 2))


# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DecisionEngine:
    """Weighted voting, Bayesian inference, confidence intervals, BM25 ranking, anomaly detection."""

    # ── Weighted majority vote ────────────────────────────────────────────────

    @staticmethod
    def weighted_vote(options: List[Tuple[str, float]]) -> Tuple[str, float]:
        """
        Weighted majority vote over (option_label, weight) pairs.
        Returns (winning_label, confidence) where confidence = winner_weight / total_weight.
        """
        if not options:
            return ("", 0.0)
        totals: Dict[str, float] = {}
        for label, weight in options:
            totals[label] = totals.get(label, 0.0) + max(0.0, weight)
        total_weight = sum(totals.values())
        if total_weight == 0:
            return (options[0][0], 0.0)
        winner = max(totals, key=lambda k: totals[k])
        confidence = totals[winner] / total_weight
        return (winner, confidence)

    # ── Bayesian update ───────────────────────────────────────────────────────

    @staticmethod
    def bayesian_update(prior: float, likelihood: float, evidence: float) -> float:
        """
        Compute posterior probability P(H|E) via Bayes' theorem.
        prior:      P(H)
        likelihood: P(E|H)
        evidence:   P(E) — marginal probability of evidence
        Returns posterior in [0, 1].
        """
        if evidence <= 0:
            return prior
        posterior = (likelihood * prior) / evidence
        return max(0.0, min(1.0, posterior))

    # ── Bootstrap confidence interval ─────────────────────────────────────────

    @staticmethod
    def confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for the mean of `values`.
        Returns (lower, upper) bounds.
        Fast approximation using sorted bootstrap resample without random module dependency.
        """
        if not values:
            return (0.0, 0.0)
        if len(values) == 1:
            return (values[0], values[0])

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
        std = math.sqrt(variance)
        sem = std / math.sqrt(n)

        # z-value for confidence level (approximation)
        z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
        z = z_map.get(confidence, 1.960)
        margin = z * sem
        return (mean - margin, mean + margin)

    # ── BM25 ranking ──────────────────────────────────────────────────────────

    @staticmethod
    def rank_by_relevance(items: List[str], query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Rank items by BM25 relevance to query.
        Returns top_k (item, score) pairs sorted best-first.
        """
        if not items or not query:
            return []

        def tokenize(t: str) -> List[str]:
            return re.findall(r'\b\w+\b', t.lower())

        q_tokens = tokenize(query)
        item_tokens = [tokenize(item) for item in items]
        N = len(items)
        avg_dl = sum(len(toks) for toks in item_tokens) / max(N, 1)
        k1, b = 1.5, 0.75

        scores = []
        for idx, (item, toks) in enumerate(zip(items, item_tokens)):
            dl = len(toks)
            score = 0.0
            for term in q_tokens:
                df = sum(1 for t in item_tokens if term in t)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                tf = toks.count(term)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
                score += idf * tf_norm
            scores.append((item, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    # ── Anomaly score (modified Z-score) ──────────────────────────────────────

    @staticmethod
    def anomaly_score(value: float, reference_values: List[float]) -> float:
        """
        Modified Z-score using median and MAD (robust to outliers).
        Returns 0.0 for normal, higher values indicate stronger anomaly.
        Score >= 3.5 typically considered anomalous.
        """
        if not reference_values:
            return 0.0
        n = len(reference_values)
        sorted_vals = sorted(reference_values)
        median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        deviations = [abs(v - median) for v in reference_values]
        deviations.sort()
        mad = deviations[n // 2] if n % 2 == 1 else (deviations[n // 2 - 1] + deviations[n // 2]) / 2
        if mad == 0:
            return 0.0 if value == median else float('inf')
        return abs(0.6745 * (value - median) / mad)

    # ── Ensemble prediction ───────────────────────────────────────────────────

    @staticmethod
    def ensemble_predict(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple model outputs.
        Each prediction dict should have: {"content": str, "confidence": float, "source": str}.
        Returns merged dict with best content, avg confidence, and sources list.
        """
        if not predictions:
            return {"content": "", "confidence": 0.0, "sources": []}

        # Filter out empty predictions
        valid = [p for p in predictions if p.get("content")]
        if not valid:
            return {"content": "", "confidence": 0.0, "sources": []}

        # Weight by confidence
        total_conf = sum(p.get("confidence", 0.5) for p in valid)
        if total_conf == 0:
            total_conf = len(valid)

        # Pick highest-confidence content as primary
        best = max(valid, key=lambda p: p.get("confidence", 0.5))
        avg_conf = total_conf / len(valid)
        sources = [p.get("source", "unknown") for p in valid]

        # Merge: use best as base, append unique insights from others
        merged_content = best.get("content", "")
        seen_hashes = {hashlib.md5(merged_content[:100].encode()).hexdigest()}

        for p in valid:
            if p is best:
                continue
            content = p.get("content", "")
            h = hashlib.md5(content[:100].encode()).hexdigest()
            if h not in seen_hashes and content:
                seen_hashes.add(h)
                # Append a brief excerpt if it differs substantially
                sim = PatternEngine.semantic_similarity(merged_content[:200], content[:200])
                if sim < 0.7:
                    merged_content += f"\n\n*Additional perspective ({p.get('source','?')}):* {content[:300]}"

        return {
            "content":    merged_content,
            "confidence": min(1.0, avg_conf),
            "sources":    sources,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE LEARNER
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLearner:
    """Online learning: Hebbian/Oja, EWMA, streaming update, KL-surprise, n-gram LM, perplexity."""

    # ── Oja's Hebbian update ──────────────────────────────────────────────────

    @staticmethod
    def hebbian_update(
        weight: float,
        pre_activation: float,
        post_activation: float,
        lr: float = 0.01,
    ) -> float:
        """
        Oja's rule variant (normalized Hebbian learning).
        Prevents weight explosion by subtracting a decay term.
        w_new = w + lr * post * (pre - post * w)
        Returns updated weight clamped to [0, 1].
        """
        delta = lr * post_activation * (pre_activation - post_activation * weight)
        return max(0.0, min(1.0, weight + delta))

    # ── Exponential weighted moving average ───────────────────────────────────

    @staticmethod
    def exponential_smoothing(history: List[float], alpha: float = 0.3) -> float:
        """
        EWMA of a history list. Returns current smoothed estimate.
        alpha: smoothing factor (0 = long memory, 1 = no memory).
        """
        if not history:
            return 0.0
        s = history[0]
        for x in history[1:]:
            s = alpha * x + (1 - alpha) * s
        return s

    # ── Online streaming update ───────────────────────────────────────────────

    @staticmethod
    def online_update(
        model_state: Dict[str, float],
        new_observation: Dict[str, float],
        decay: float = 0.95,
    ) -> Dict[str, float]:
        """
        Streaming model state update with exponential decay of old values.
        model_state and new_observation are dicts of feature -> value.
        Returns updated model_state.
        """
        updated = {k: v * decay for k, v in model_state.items()}
        for k, v in new_observation.items():
            updated[k] = updated.get(k, 0.0) + v * (1.0 - decay)
        return updated

    # ── KL divergence proxy (surprise) ────────────────────────────────────────

    @staticmethod
    def compute_surprise(expected: str, actual: str) -> float:
        """
        KL divergence proxy between expected and actual text distributions.
        Uses word unigram distributions. Returns surprise >= 0.0.
        Higher = more surprising (actual diverged significantly from expected).
        """
        if not expected or not actual:
            return 1.0

        def word_dist(text: str) -> Dict[str, float]:
            tokens = re.findall(r'\b\w+\b', text.lower())
            if not tokens:
                return {}
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            total = len(tokens)
            return {k: v / total for k, v in counts.items()}

        p = word_dist(expected)
        q = word_dist(actual)
        if not p or not q:
            return 1.0

        # KL(P || Q) = sum p(x) * log(p(x)/q(x))
        # Use smoothed Q to avoid log(0)
        eps = 1e-10
        all_words = set(p) | set(q)
        kl = 0.0
        for w in all_words:
            p_w = p.get(w, eps)
            q_w = q.get(w, eps)
            if p_w > 0:
                kl += p_w * math.log(p_w / q_w)

        # Normalize to [0, 1] range (sigmoid-like compression)
        return 1.0 - math.exp(-abs(kl))

    # ── Character n-gram language model ───────────────────────────────────────

    @staticmethod
    def build_ngram_model(corpus: List[str], n: int = 3) -> Dict[str, Dict[str, float]]:
        """
        Build a character n-gram language model from corpus.
        Returns dict: {context -> {next_char -> probability}}.
        Context is the (n-1)-char prefix.
        """
        counts: Dict[str, Dict[str, int]] = {}
        for doc in corpus:
            text = doc.lower()
            for i in range(len(text) - n + 1):
                context = text[i:i + n - 1]
                next_c  = text[i + n - 1]
                if context not in counts:
                    counts[context] = {}
                counts[context][next_c] = counts[context].get(next_c, 0) + 1

        # Convert to probabilities
        model: Dict[str, Dict[str, float]] = {}
        for ctx, next_counts in counts.items():
            total = sum(next_counts.values())
            model[ctx] = {c: cnt / total for c, cnt in next_counts.items()}
        return model

    # ── Perplexity ────────────────────────────────────────────────────────────

    @staticmethod
    def perplexity(text: str, ngram_model: Dict[str, Dict[str, float]], n: int = 3) -> float:
        """
        Compute perplexity of text under ngram_model.
        Low perplexity = text is well-predicted (repetitive/known).
        High perplexity = text is novel/surprising.
        Returns perplexity value >= 1.0. Returns 1000.0 if model is empty.
        """
        if not ngram_model or not text:
            return 1000.0

        text = text.lower()
        log_prob = 0.0
        count = 0
        eps = 1e-10  # smoothing

        for i in range(len(text) - n + 1):
            context = text[i:i + n - 1]
            next_c  = text[i + n - 1]
            ctx_dist = ngram_model.get(context, {})
            prob = ctx_dist.get(next_c, eps)
            log_prob += math.log(prob)
            count += 1

        if count == 0:
            return 1000.0

        avg_log_prob = log_prob / count
        perp = math.exp(-avg_log_prob)
        return min(perp, 100000.0)  # cap to prevent overflow display


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationEngine:
    """Statistical correlations, mutual information, causal scoring, knowledge graph."""

    # ── Pearson correlation ───────────────────────────────────────────────────

    @staticmethod
    def pearson(x: List[float], y: List[float]) -> float:
        """
        Pearson product-moment correlation coefficient.
        Returns value in [-1, 1]. Returns 0.0 on invalid input.
        """
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        x, y = x[:n], y[:n]
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx  = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        dy  = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if dx == 0 or dy == 0:
            return 0.0
        return max(-1.0, min(1.0, num / (dx * dy)))

    # ── Spearman rank correlation ──────────────────────────────────────────────

    @staticmethod
    def rank_correlation(x: List[float], y: List[float]) -> float:
        """
        Spearman's rho rank correlation coefficient.
        Returns value in [-1, 1]. More robust to outliers than Pearson.
        """
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        x, y = x[:n], y[:n]

        def rank(vals: List[float]) -> List[float]:
            sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
            ranks = [0.0] * len(vals)
            for r, (i, _) in enumerate(sorted_vals):
                ranks[i] = r + 1.0
            return ranks

        rx = rank(x)
        ry = rank(y)
        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))
        return max(-1.0, min(1.0, rho))

    # ── Mutual information (discretized) ──────────────────────────────────────

    @staticmethod
    def mutual_information(x: List[float], y: List[float], bins: int = 10) -> float:
        """
        Mutual information between two sequences via histogram discretization.
        Returns MI >= 0.0 (higher = more dependent).
        """
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        x, y = x[:n], y[:n]

        def discretize(vals: List[float], b: int) -> List[int]:
            lo, hi = min(vals), max(vals)
            if lo == hi:
                return [0] * len(vals)
            rng = hi - lo
            return [min(b - 1, int((v - lo) / rng * b)) for v in vals]

        xd = discretize(x, bins)
        yd = discretize(y, bins)

        # Joint and marginal counts
        joint: Dict[Tuple[int, int], int] = {}
        cx: Dict[int, int] = {}
        cy: Dict[int, int] = {}
        for xi, yi in zip(xd, yd):
            joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
            cx[xi] = cx.get(xi, 0) + 1
            cy[yi] = cy.get(yi, 0) + 1

        mi = 0.0
        for (xi, yi), cnt in joint.items():
            p_xy = cnt / n
            p_x  = cx[xi] / n
            p_y  = cy[yi] / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log(p_xy / (p_x * p_y))
        return max(0.0, mi)

    # ── Granger-inspired text causality ───────────────────────────────────────

    @staticmethod
    def causal_score(cause: str, effect: str, context: str = "") -> float:
        """
        Granger-inspired text causality score between two text snippets.
        Returns score 0.0–1.0. Checks: semantic overlap, causal keywords, entity overlap.
        """
        causal_kw = {
            "because", "therefore", "thus", "hence", "consequently", "leads to",
            "results in", "causes", "due to", "as a result", "owing to",
            "triggered by", "driven by", "produces", "generates", "implies",
        }
        combined = (cause + " " + effect + " " + context).lower()
        kw_score = sum(1 for kw in causal_kw if kw in combined) / len(causal_kw)

        sim = PatternEngine.semantic_similarity(cause, effect)
        entities_cause  = PatternEngine.extract_entities(cause)
        entities_effect = PatternEngine.extract_entities(effect)
        all_cause  = set(v for vals in entities_cause.values() for v in vals)
        all_effect = set(v for vals in entities_effect.values() for v in vals)
        entity_overlap = len(all_cause & all_effect) / max(len(all_cause | all_effect), 1)

        return min(1.0, kw_score * 0.4 + sim * 0.4 + entity_overlap * 0.2)

    # ── Knowledge graph ───────────────────────────────────────────────────────

    @staticmethod
    def build_knowledge_graph(facts: List[Tuple[str, str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Build weighted adjacency dict from (node_a, node_b, weight) fact tuples.
        Graph is bidirectional by default.
        Returns {node: {neighbor: weight}}.
        """
        graph: Dict[str, Dict[str, float]] = {}
        for node_a, node_b, weight in facts:
            na = str(node_a).strip()
            nb = str(node_b).strip()
            if not na or not nb:
                continue
            if na not in graph:
                graph[na] = {}
            if nb not in graph:
                graph[nb] = {}
            # Keep max weight for duplicate edges
            graph[na][nb] = max(graph[na].get(nb, 0.0), weight)
            graph[nb][na] = max(graph[nb].get(na, 0.0), weight)
        return graph

    # ── BFS path finding with ranking ─────────────────────────────────────────

    @staticmethod
    def find_paths(
        graph: Dict[str, Dict[str, float]],
        start: str,
        end: str,
        max_depth: int = 4,
    ) -> List[Tuple[List[str], float]]:
        """
        BFS to find all paths from start to end up to max_depth hops.
        Returns list of (path, path_score) sorted by score descending.
        path_score = product of edge weights along path.
        """
        if start not in graph or end not in graph:
            return []

        results: List[Tuple[List[str], float]] = []
        # BFS queue: (current_node, path_so_far, score_so_far)
        queue = collections.deque([(start, [start], 1.0)])
        visited_per_path: set = set()

        while queue:
            node, path, score = queue.popleft()
            if len(path) > max_depth + 1:
                continue
            if node == end and len(path) > 1:
                results.append((list(path), score))
                continue
            if len(path) == max_depth + 1:
                continue
            for neighbor, weight in graph.get(node, {}).items():
                if neighbor not in path:  # avoid cycles
                    queue.append((neighbor, path + [neighbor], score * weight))

        results.sort(key=lambda x: -x[1])
        return results[:10]  # return top 10 paths


# ─────────────────────────────────────────────────────────────────────────────
# Convenience re-exports for single-import usage
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "PatternEngine",
    "DecisionEngine",
    "AdaptiveLearner",
    "CorrelationEngine",
]
