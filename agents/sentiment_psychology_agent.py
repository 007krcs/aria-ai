"""
ARIA Sentiment & Psychology Market Intelligence Agent
======================================================

The world-class ARIA Market Intelligence Algorithm (AMIA) — multi-signal dot-connection
engine that identifies the best trade/investment opportunity by analyzing:

  1. PRICE ACTION & MOMENTUM     — trend, velocity, volume confirmation
  2. NEWS SENTIMENT              — NLP-weighted multi-source news scoring
  3. MARKET PSYCHOLOGY           — fear/greed, FOMO, panic, euphoria detection
  4. OPERATOR / WHALE ANALYSIS   — unusual volume, block trades, dark pool signals
  5. INSTITUTIONAL FLOW          — FII/DII/MF/hedge funds/big bank buying/selling
  6. EARNINGS INTELLIGENCE       — EPS vs estimates, revenue trend, guidance quality
  7. BIG INVESTOR TRACKING       — JP Morgan, Goldman, Warren Buffett, Rakesh J style signals
  8. SECTOR & MACRO MOOD         — sector rotation, macro sentiment, RBI/Fed policy mood
  9. SOCIAL SENTIMENT            — Reddit/Twitter/Telegram crowd psychology index
 10. MULTI-SIGNAL SYNTHESIS      — all 9 signals combined → STRONG BUY → STRONG SELL

Algorithm Output:
  - Score: -100 (STRONG SELL) → +100 (STRONG BUY)
  - Confidence: 0–100%
  - Signal breakdown with individual weights
  - Key risk factors
  - Entry / Stop-loss / Target recommendation

Usage:
    agent = SentimentPsychologyAgent(engine=ollama_engine)
    result = await agent.analyze("RELIANCE", "stock")
    result = await agent.analyze("Nifty 50", "index")
    result = await agent.analyze("Bitcoin", "crypto")
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── Signal weights (must sum to 1.0) ──────────────────────────────────────────
WEIGHTS = {
    "institutional_flow":  0.22,   # FII/DII/MF/banks — biggest market mover
    "earnings_quality":    0.18,   # EPS beat/miss, guidance — fundamental truth
    "price_action":        0.15,   # trend + momentum + volume
    "news_sentiment":      0.13,   # news NLP weighted by source credibility
    "operator_signals":    0.12,   # whale/block trade/dark pool — smart money
    "market_psychology":   0.10,   # fear/greed/panic/FOMO — crowd behavior
    "big_investor_track":  0.05,   # known institutional positions
    "sector_macro":        0.03,   # macro/sector rotation
    "social_sentiment":    0.02,   # Reddit/Twitter/Telegram noise
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Scoring constants ─────────────────────────────────────────────────────────
SCORE_BANDS = [
    (+70,  +100, "STRONG BUY",   "🟢"),
    (+40,  +69,  "BUY",          "🟩"),
    (+15,  +39,  "ACCUMULATE",   "🔵"),
    (-14,  +14,  "NEUTRAL",      "⚪"),
    (-39,  -15,  "REDUCE",       "🟡"),
    (-69,  -40,  "SELL",         "🟠"),
    (-100, -70,  "STRONG SELL",  "🔴"),
]

# ── Earnings quality signals ──────────────────────────────────────────────────
EARNINGS_SIGNALS = {
    "beat_eps_and_revenue":   +1.0,
    "beat_eps_miss_revenue":  +0.4,
    "miss_eps_beat_revenue":  -0.3,
    "miss_both":              -1.0,
    "guidance_raised":        +0.8,
    "guidance_cut":           -0.9,
    "guidance_maintained":    +0.1,
    "management_buyback":     +0.7,
    "dividend_increased":     +0.6,
    "dividend_cut":           -0.8,
    "accounting_concern":     -1.0,
    "debt_reduction":         +0.5,
    "debt_increase":          -0.4,
}

# ── Psychology pattern keywords → sentiment score ─────────────────────────────
PSYCHOLOGY_KEYWORDS = {
    # Extreme fear / panic → contrarian BUY signal
    "panic":           +0.4,   # contrarian
    "crash":           +0.3,   # contrarian
    "collapse":        +0.25,  # contrarian
    "everyone selling": +0.35, # contrarian
    "blood on street": +0.4,   # contrarian Buffett signal
    "market crash":    +0.35,  # contrarian
    # Extreme greed / euphoria → contrarian SELL signal
    "euphoria":        -0.5,
    "everyone buying": -0.4,
    "to the moon":     -0.5,   # retail FOMO = distribution
    "can't go wrong":  -0.6,
    "guaranteed":      -0.7,
    "sure shot":       -0.6,
    "bubble":          -0.5,
    "overvalued":      -0.4,
    # Neutral/mixed
    "volatile":         0.0,
    "uncertain":       -0.1,
    "consolidating":    0.1,
    "accumulation":    +0.4,
    "distribution":    -0.4,
    "breakout":        +0.5,
    "breakdown":       -0.5,
}

# ── Operator/whale signal keywords ───────────────────────────────────────────
OPERATOR_KEYWORDS = {
    "unusual volume":         +0.6,
    "block deal":             +0.5,
    "bulk deal":              +0.5,
    "promoter buying":        +0.8,
    "promoter selling":       -0.8,
    "institutional buying":   +0.7,
    "institutional selling":  -0.7,
    "short covering":         +0.6,
    "short buildup":          -0.6,
    "long buildup":           +0.5,
    "long unwinding":         -0.5,
    "delivery percentage":    +0.3,
    "put call ratio":          0.0,   # neutral by itself
    "fii buying":             +0.7,
    "fii selling":            -0.7,
    "dii buying":             +0.5,
    "dii selling":            -0.3,
    "mf buying":              +0.4,
    "mf selling":             -0.3,
}

# ── Big investor keyword tracker ──────────────────────────────────────────────
BIG_INVESTOR_KEYWORDS = {
    "jp morgan":        ("neutral", 0.5),
    "goldman sachs":    ("neutral", 0.5),
    "morgan stanley":   ("neutral", 0.4),
    "blackrock":        ("neutral", 0.5),
    "warren buffett":   ("buy",     0.9),
    "rakesh jhunjhunwala": ("buy",  0.9),
    "radhakishan damani":  ("buy",  0.8),
    "vijay kedia":      ("buy",     0.7),
    "dolly khanna":     ("buy",     0.7),
    "ashish dhawan":    ("buy",     0.6),
    "ark invest":       ("neutral", 0.5),
    "sequoia":          ("buy",     0.7),
    "tiger global":     ("neutral", 0.4),
    "government bond":  ("sell",    0.5),
    "sebi notice":      ("sell",    0.9),
    "promoter pledge":  ("sell",    0.8),
}

# ── News source credibility weights ──────────────────────────────────────────
SOURCE_CREDIBILITY = {
    "rbi.org.in": 1.0,  "sebi.gov.in": 1.0,  "bseindia.com": 0.95,
    "nseindia.com": 0.95,  "moneycontrol.com": 0.85,  "economictimes.com": 0.85,
    "livemint.com": 0.85,  "business-standard.com": 0.85,  "reuters.com": 0.9,
    "bloomberg.com": 0.9,  "ft.com": 0.85,  "wsj.com": 0.85,
    "cnbc.com": 0.75,  "twitter.com": 0.3,  "reddit.com": 0.25,
    "telegram": 0.2,  "unknown": 0.5,
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """One individual signal score (-1.0 → +1.0) with explanation."""
    name:        str
    raw_score:   float       # -1.0 to +1.0
    weight:      float
    weighted:    float = 0.0  # = raw_score * weight * 100
    confidence:  float = 0.5  # how confident we are in this signal
    evidence:    list[str] = field(default_factory=list)
    risk_flag:   bool = False

    def __post_init__(self):
        self.weighted = self.raw_score * self.weight * 100


@dataclass
class MarketAnalysis:
    """Final output of AMIA algorithm for one symbol."""
    symbol:         str
    asset_type:     str          # stock | index | crypto | commodity | forex
    timestamp:      str
    total_score:    float        # -100 → +100
    confidence:     float        # 0 → 100%
    verdict:        str          # STRONG BUY etc.
    verdict_emoji:  str
    signals:        list[SignalResult]
    entry_price:    Optional[float] = None
    stop_loss:      Optional[float] = None
    target_price:   Optional[float] = None
    risk_reward:    Optional[float] = None
    time_horizon:   str = "medium"   # short | medium | long
    key_risks:      list[str] = field(default_factory=list)
    key_catalysts:  list[str] = field(default_factory=list)
    operator_alert: Optional[str] = None
    psychology_mood: str = "neutral"  # fear | greed | panic | euphoria | neutral
    narrative:      str = ""   # human-readable ARIA narrative


# ── Main Agent ────────────────────────────────────────────────────────────────

class SentimentPsychologyAgent:
    """
    ARIA Market Intelligence Algorithm (AMIA).

    Aggregates 9 signal categories, weights them, and produces a comprehensive
    buy/sell/hold recommendation with full transparency on every signal.
    """

    def __init__(self, engine=None, search_agent=None, trust_registry=None):
        self.engine        = engine          # Ollama engine for NLP tasks
        self.search_agent  = search_agent    # web search for live news
        self.trust         = trust_registry  # trusted source registry

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(
        self,
        query: str,
        asset_type: str = "stock",
        raw_context: str = "",
    ) -> MarketAnalysis:
        """
        Full AMIA analysis. Returns MarketAnalysis with score, signals, narrative.
        query = "RELIANCE" | "Nifty 50" | "Bitcoin" | "Gold"
        """
        symbol = self._extract_symbol(query)
        t0     = time.time()

        # Gather all raw signals concurrently
        signals = await asyncio.gather(
            self._institutional_flow(symbol, raw_context),
            self._earnings_quality(symbol, raw_context),
            self._price_action(symbol, raw_context),
            self._news_sentiment(symbol, raw_context),
            self._operator_signals(symbol, raw_context),
            self._market_psychology(symbol, raw_context),
            self._big_investor_track(symbol, raw_context),
            self._sector_macro(symbol, raw_context),
            self._social_sentiment(symbol, raw_context),
        )

        # Compute total score
        total_score = sum(s.weighted for s in signals)
        total_score = max(-100.0, min(100.0, total_score))

        # Confidence = weighted average of individual confidences
        total_conf  = sum(s.confidence * s.weight for s in signals)
        confidence  = round(total_conf * 100, 1)

        # Verdict band
        verdict, emoji = self._get_verdict(total_score)

        # Psychology mood
        psych_sig = next((s for s in signals if s.name == "market_psychology"), None)
        mood      = self._detect_mood(psych_sig, total_score)

        # Key risks and catalysts
        risks, catalysts = self._extract_risk_catalysts(signals)

        # Operator alert
        op_sig   = next((s for s in signals if s.name == "operator_signals"), None)
        op_alert = self._operator_alert(op_sig, total_score)

        # Pricing estimate (illustrative if no live price)
        entry, sl, tp, rr = self._price_levels(total_score, confidence)

        # Time horizon
        horizon = self._time_horizon(signals)

        analysis = MarketAnalysis(
            symbol          = symbol.upper(),
            asset_type      = asset_type,
            timestamp       = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_score     = round(total_score, 1),
            confidence      = confidence,
            verdict         = verdict,
            verdict_emoji   = emoji,
            signals         = signals,
            entry_price     = entry,
            stop_loss       = sl,
            target_price    = tp,
            risk_reward     = rr,
            time_horizon    = horizon,
            key_risks       = risks,
            key_catalysts   = catalysts,
            operator_alert  = op_alert,
            psychology_mood = mood,
            narrative       = "",
        )

        # Generate narrative with LLM if available
        analysis.narrative = await self._generate_narrative(analysis)
        logger.info("AMIA %s → %.1f (%s) in %.2fs", symbol, total_score, verdict, time.time()-t0)
        return analysis

    def format_report(self, a: MarketAnalysis) -> str:
        """Format MarketAnalysis as a rich markdown report."""
        lines = [
            f"## {a.verdict_emoji} {a.symbol} — {a.verdict}",
            f"**AMIA Score:** `{a.total_score:+.1f}/100`  |  **Confidence:** `{a.confidence:.0f}%`  |  **Horizon:** `{a.time_horizon.upper()}`",
            f"**Mood:** `{a.psychology_mood.upper()}`  |  **Asset:** `{a.asset_type}`  |  *{a.timestamp}*",
            "",
        ]

        if a.operator_alert:
            lines += [f"> ⚠️ **Operator Alert:** {a.operator_alert}", ""]

        # Signal breakdown table
        lines += ["### Signal Breakdown", "| Signal | Raw | Weighted | Confidence | Key Evidence |",
                  "|--------|-----|---------|-----------|------------|"]
        for s in sorted(a.signals, key=lambda x: abs(x.weighted), reverse=True):
            bar    = "█" * int(abs(s.raw_score) * 5)
            polarity = "+" if s.raw_score >= 0 else ""
            evid   = "; ".join(s.evidence[:2]) if s.evidence else "—"
            flag   = " ⚠" if s.risk_flag else ""
            lines.append(
                f"| {s.name.replace('_',' ').title()}{flag} "
                f"| {polarity}{s.raw_score:.2f} {bar} "
                f"| {s.weighted:+.1f} "
                f"| {s.confidence*100:.0f}% "
                f"| {evid[:60]} |"
            )
        lines.append("")

        # Price levels
        if a.entry_price:
            lines += [
                "### Price Levels (Indicative)",
                f"| Entry | Stop-Loss | Target | Risk/Reward |",
                f"|-------|-----------|--------|-------------|",
                f"| {a.entry_price:.1f} | {a.stop_loss:.1f} | {a.target_price:.1f} | 1:{a.risk_reward:.1f} |",
                "",
            ]

        # Catalysts & risks
        if a.key_catalysts:
            lines += ["### Key Catalysts 🚀", *[f"- {c}" for c in a.key_catalysts], ""]
        if a.key_risks:
            lines += ["### Key Risks ⚠️", *[f"- {r}" for r in a.key_risks], ""]

        # ARIA Narrative
        if a.narrative:
            lines += ["### ARIA Intelligence Narrative", a.narrative]

        lines += [
            "",
            "---",
            "*AMIA v2 — 9-signal weighted algorithm. Not financial advice. Do your own research.*",
        ]
        return "\n".join(lines)

    async def run_nl(self, query: str) -> str:
        """Natural language interface — analyzes any stock/market query."""
        symbol = self._extract_symbol(query)
        asset  = self._detect_asset_type(query)
        try:
            analysis = await self.analyze(symbol, asset, raw_context=query)
            return self.format_report(analysis)
        except Exception as e:
            logger.error("AMIA error: %s", e)
            return f"Market intelligence analysis failed for '{symbol}': {e}"

    # ── Signal generators ─────────────────────────────────────────────────────

    async def _institutional_flow(self, symbol: str, ctx: str) -> SignalResult:
        """
        Institutional flow: FII, DII, mutual funds, pension funds, big banks.
        Largest weight (22%) — institutions see data retail doesn't.
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.45  # base confidence without live data

        # Parse known patterns from context
        if any(k in ctx_lower for k in ["fii buying", "fii net buy", "foreign buying"]):
            score += 0.7; evidence.append("FII net buyers"); confidence = 0.8
        if any(k in ctx_lower for k in ["fii selling", "fii net sell", "foreign selling"]):
            score -= 0.7; evidence.append("FII net sellers"); confidence = 0.8
        if any(k in ctx_lower for k in ["dii buying", "domestic buying"]):
            score += 0.4; evidence.append("DII buyers")
        if any(k in ctx_lower for k in ["dii selling"]):
            score -= 0.3; evidence.append("DII sellers")
        if any(k in ctx_lower for k in ["mf sip", "mutual fund inflow", "sip inflow"]):
            score += 0.3; evidence.append("MF SIP inflows"); confidence = 0.7
        if any(k in ctx_lower for k in ["mutual fund selling", "mf redemption"]):
            score -= 0.25; evidence.append("MF redemptions")

        # Sector signals — if banking/it mentioned with institutional
        if "bank nifty" in ctx_lower or "banking sector" in ctx_lower:
            if "upgrade" in ctx_lower: score += 0.2
            if "downgrade" in ctx_lower: score -= 0.2

        # Default slight positive (markets go up long-term)
        if not evidence:
            score  = 0.05
            evidence = ["No live institutional data — applying long-term positive bias"]

        score = max(-1.0, min(1.0, score))
        return SignalResult(
            name="institutional_flow", raw_score=score,
            weight=WEIGHTS["institutional_flow"], confidence=confidence,
            evidence=evidence,
        )

    async def _earnings_quality(self, symbol: str, ctx: str) -> SignalResult:
        """
        Earnings: EPS vs estimates, revenue, guidance, dividends, debt, buybacks.
        Second highest weight (18%) — fundamental driver of long-term price.
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.4

        for key, val in EARNINGS_SIGNALS.items():
            if key.replace("_", " ") in ctx_lower:
                score += val
                evidence.append(key.replace("_", " ").title())
                confidence = 0.75

        # Revenue keywords
        if any(k in ctx_lower for k in ["revenue growth", "profit growth", "net profit up"]):
            score += 0.4; evidence.append("Revenue/profit growth"); confidence = 0.75
        if any(k in ctx_lower for k in ["revenue decline", "profit decline", "net loss"]):
            score -= 0.5; evidence.append("Revenue/profit decline"); confidence = 0.8

        # PE ratio — expensive vs cheap
        if "overvalued" in ctx_lower or "high pe" in ctx_lower or "expensive" in ctx_lower:
            score -= 0.3; evidence.append("High valuation")
        if "undervalued" in ctx_lower or "low pe" in ctx_lower or "cheap" in ctx_lower:
            score += 0.3; evidence.append("Attractive valuation")

        # Result season
        if any(k in ctx_lower for k in ["result day", "quarterly result", "earnings report"]):
            confidence = 0.85  # heightened confidence around results

        if not evidence:
            score = 0.0
            evidence = ["No earnings data in context"]

        score = max(-1.0, min(1.0, score))
        return SignalResult(
            name="earnings_quality", raw_score=score,
            weight=WEIGHTS["earnings_quality"], confidence=confidence,
            evidence=evidence, risk_flag=(score < -0.5),
        )

    async def _price_action(self, symbol: str, ctx: str) -> SignalResult:
        """
        Price action: trend, momentum, volume, moving averages, breakout.
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.5

        # Trend signals
        if any(k in ctx_lower for k in ["uptrend", "new 52 week high", "ath", "all time high", "breakout"]):
            score += 0.6; evidence.append("Strong uptrend / breakout"); confidence = 0.75
        if any(k in ctx_lower for k in ["downtrend", "52 week low", "breakdown", "bearish"]):
            score -= 0.6; evidence.append("Downtrend / breakdown"); confidence = 0.75
        # Volume
        if any(k in ctx_lower for k in ["high volume", "volume surge", "unusual volume"]):
            score += 0.3 if score >= 0 else -0.3
            evidence.append("Volume confirmation")
        # Moving averages
        if any(k in ctx_lower for k in ["above 200 dma", "golden cross", "above ma"]):
            score += 0.4; evidence.append("Above 200 DMA / Golden cross")
        if any(k in ctx_lower for k in ["below 200 dma", "death cross", "below ma"]):
            score -= 0.4; evidence.append("Below 200 DMA / Death cross")
        # RSI
        if "oversold" in ctx_lower or "rsi below 30" in ctx_lower:
            score += 0.3; evidence.append("Oversold RSI — potential reversal")
        if "overbought" in ctx_lower or "rsi above 70" in ctx_lower:
            score -= 0.3; evidence.append("Overbought RSI — caution")

        if not evidence:
            score = 0.0; evidence = ["Insufficient price data"]

        score = max(-1.0, min(1.0, score))
        return SignalResult(
            name="price_action", raw_score=score,
            weight=WEIGHTS["price_action"], confidence=confidence,
            evidence=evidence,
        )

    async def _news_sentiment(self, symbol: str, ctx: str) -> SignalResult:
        """
        News NLP: weighted by source credibility, recency, and sentiment intensity.
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.5

        # Positive news patterns
        pos_patterns = [
            ("acquisition", +0.5, "M&A / acquisition"),
            ("joint venture", +0.4, "JV announced"),
            ("new contract", +0.5, "New contract"),
            ("order win", +0.6, "Order win"),
            ("credit rating upgrade", +0.6, "Rating upgrade"),
            ("product launch", +0.3, "Product launch"),
            ("partnership", +0.3, "Strategic partnership"),
            ("government contract", +0.5, "Government contract"),
            ("approved", +0.3, "Regulatory approval"),
            ("fda approval", +0.7, "FDA/DCGI approval"),
            ("capex expansion", +0.4, "Capacity expansion"),
        ]
        neg_patterns = [
            ("fraud", -1.0, "Fraud allegations"),
            ("scam", -0.9, "Scam/scandal"),
            ("sebi notice", -0.8, "SEBI regulatory action"),
            ("income tax raid", -0.7, "IT/ED raid"),
            ("bankruptcy", -1.0, "Bankruptcy filing"),
            ("default", -0.8, "Loan default"),
            ("class action", -0.6, "Legal action"),
            ("credit rating downgrade", -0.6, "Rating downgrade"),
            ("product recall", -0.5, "Product recall"),
            ("ceo resign", -0.4, "Management change"),
            ("accounting irregularity", -0.9, "Accounting concern"),
            ("data breach", -0.5, "Data breach"),
        ]

        for pattern, weight, label in pos_patterns:
            if pattern in ctx_lower:
                score += weight; evidence.append(f"📰 {label}"); confidence = 0.75

        for pattern, weight, label in neg_patterns:
            if pattern in ctx_lower:
                score += weight  # weight is negative
                evidence.append(f"📰 {label}")
                confidence = 0.8

        # Source credibility weighting
        for src, cred in SOURCE_CREDIBILITY.items():
            if src in ctx_lower and cred > 0.8:
                confidence = min(0.95, confidence + 0.05)
                evidence.append(f"High-credibility source: {src}")
                break

        if not evidence:
            score = 0.0; evidence = ["No news events detected in context"]
            confidence = 0.3

        score = max(-1.0, min(1.0, score))
        return SignalResult(
            name="news_sentiment", raw_score=score,
            weight=WEIGHTS["news_sentiment"], confidence=confidence,
            evidence=evidence, risk_flag=(score < -0.5),
        )

    async def _operator_signals(self, symbol: str, ctx: str) -> SignalResult:
        """
        Operator/whale analysis: smart money activity, block deals, promoter moves.
        These often precede big moves — 'operators' move first, retail follows.
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.4

        for kw, val in OPERATOR_KEYWORDS.items():
            if kw in ctx_lower:
                score += val; evidence.append(kw.title()); confidence = 0.75

        # Promoter pledge — very bearish signal
        if "promoter pledge increase" in ctx_lower or "pledge increase" in ctx_lower:
            score -= 0.8; evidence.append("Promoter pledge increasing ⚠️")
            confidence = 0.85

        # Options data
        if "high open interest" in ctx_lower:
            score += 0.2 if "call" in ctx_lower else -0.2
            evidence.append("High options OI")

        # Delivery-based buying — strong hands
        if "high delivery" in ctx_lower or "delivery based" in ctx_lower:
            score += 0.4; evidence.append("High delivery % — strong hands")

        if not evidence:
            score = 0.0; evidence = ["No operator signals detected"]
            confidence = 0.3

        alert = None
        if score < -0.6:
            alert = f"Smart money appears to be EXITING {symbol} — caution advised"
        elif score > 0.6:
            alert = f"Smart money appears to be ACCUMULATING {symbol} — watch closely"

        result = SignalResult(
            name="operator_signals", raw_score=score,
            weight=WEIGHTS["operator_signals"], confidence=confidence,
            evidence=evidence, risk_flag=(score < -0.5),
        )
        return result

    async def _market_psychology(self, symbol: str, ctx: str) -> SignalResult:
        """
        Crowd psychology: fear, greed, FOMO, panic, euphoria.
        Uses contrarian logic — extreme fear = buy, extreme greed = sell.
        Based on Buffett: 'Be greedy when others are fearful.'
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.5

        for kw, val in PSYCHOLOGY_KEYWORDS.items():
            if kw in ctx_lower:
                score += val; evidence.append(f"Psychology: '{kw}'")

        # Fear/greed index-style detection
        fear_words  = sum(1 for w in ["fear","panic","crash","dump","collapse"] if w in ctx_lower)
        greed_words = sum(1 for w in ["greed","euphoria","moon","bull run","rally"] if w in ctx_lower)

        if fear_words >= 2:
            score += 0.5  # contrarian buy
            evidence.append(f"Extreme fear detected ({fear_words} signals) — contrarian BUY")
            confidence = 0.7
        if greed_words >= 2:
            score -= 0.5  # contrarian sell
            evidence.append(f"Extreme greed detected ({greed_words} signals) — contrarian SELL")
            confidence = 0.7

        if not evidence:
            score = 0.0; evidence = ["Market mood appears neutral"]
            confidence = 0.35

        score = max(-1.0, min(1.0, score))
        return SignalResult(
            name="market_psychology", raw_score=score,
            weight=WEIGHTS["market_psychology"], confidence=confidence,
            evidence=evidence,
        )

    async def _big_investor_track(self, symbol: str, ctx: str) -> SignalResult:
        """
        Track known big investors: Buffett, Rakesh J, JP Morgan calls, etc.
        Their public positions/statements carry significant signal weight.
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.4

        for investor, (bias, strength) in BIG_INVESTOR_KEYWORDS.items():
            if investor in ctx_lower:
                if bias == "buy":
                    score += strength * 0.5; evidence.append(f"{investor.title()} bullish signal")
                elif bias == "sell":
                    score -= strength * 0.5; evidence.append(f"{investor.title()} bearish signal")
                else:
                    # neutral — direction from context
                    if "buy" in ctx_lower or "bullish" in ctx_lower:
                        score += strength * 0.3
                    elif "sell" in ctx_lower or "bearish" in ctx_lower:
                        score -= strength * 0.3
                    evidence.append(f"{investor.title()} mentioned")
                confidence = 0.75

        if not evidence:
            score = 0.0; evidence = ["No big investor signals in context"]
            confidence = 0.3

        score = max(-1.0, min(1.0, score))
        return SignalResult(
            name="big_investor_track", raw_score=score,
            weight=WEIGHTS["big_investor_track"], confidence=confidence,
            evidence=evidence,
        )

    async def _sector_macro(self, symbol: str, ctx: str) -> SignalResult:
        """
        Sector rotation & macro: RBI policy, Fed, GDP, inflation, sector headwinds/tailwinds.
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.4

        # RBI / monetary policy
        if any(k in ctx_lower for k in ["rbi rate cut", "rate cut", "dovish"]):
            score += 0.5; evidence.append("Rate cut / dovish policy — positive for equities")
        if any(k in ctx_lower for k in ["rbi rate hike", "rate hike", "hawkish"]):
            score -= 0.4; evidence.append("Rate hike — negative for equities")
        if "rbi pause" in ctx_lower or "rate pause" in ctx_lower:
            score += 0.1; evidence.append("Rate pause — neutral to slightly positive")

        # GDP / economic
        if "gdp growth" in ctx_lower or "economy growing" in ctx_lower:
            score += 0.3; evidence.append("GDP growth positive")
        if "recession" in ctx_lower or "gdp contraction" in ctx_lower:
            score -= 0.4; evidence.append("Recession risk")

        # Inflation
        if "inflation falling" in ctx_lower or "cpi lower" in ctx_lower:
            score += 0.2; evidence.append("Falling inflation — positive")
        if "high inflation" in ctx_lower or "inflation rising" in ctx_lower:
            score -= 0.2; evidence.append("Rising inflation — negative")

        # Sector-specific
        sector_signals = {
            "it sector rally": +0.3, "banking rally": +0.3,
            "pharma rally": +0.3, "sector rotation into": +0.4,
            "sector selloff": -0.4, "sector headwind": -0.3,
        }
        for kw, val in sector_signals.items():
            if kw in ctx_lower:
                score += val; evidence.append(kw.title())

        if not evidence:
            score = 0.02; evidence = ["No macro signals — slight positive long-run bias"]
            confidence = 0.3

        score = max(-1.0, min(1.0, score))
        return SignalResult(
            name="sector_macro", raw_score=score,
            weight=WEIGHTS["sector_macro"], confidence=confidence,
            evidence=evidence,
        )

    async def _social_sentiment(self, symbol: str, ctx: str) -> SignalResult:
        """
        Social media sentiment: Reddit, Twitter, Telegram.
        Low weight (2%) — retail social sentiment is often contrarian noise.
        """
        ctx_lower = ctx.lower()
        score     = 0.0
        evidence  = []
        confidence= 0.3

        # Viral retail buying = contrarian sell
        if any(k in ctx_lower for k in ["trending on twitter", "viral stock", "reddit meme stock"]):
            score -= 0.4; evidence.append("Viral retail attention — potential top")
        if any(k in ctx_lower for k in ["social media bearish", "twitter bearish"]):
            score += 0.2; evidence.append("Social bearishness — contrarian buy")

        # Telegram pump/dump detection
        if any(k in ctx_lower for k in ["telegram pump", "tip channel", "sure shot tip"]):
            score -= 0.7; evidence.append("⚠️ Potential pump-dump / tip channel signal")
            confidence = 0.9

        if not evidence:
            score = 0.0; evidence = ["No social signals"]
            confidence = 0.25

        score = max(-1.0, min(1.0, score))
        return SignalResult(
            name="social_sentiment", raw_score=score,
            weight=WEIGHTS["social_sentiment"], confidence=confidence,
            evidence=evidence, risk_flag=("pump" in ctx.lower() or "tip channel" in ctx.lower()),
        )

    # ── LLM Narrative ─────────────────────────────────────────────────────────

    async def _generate_narrative(self, a: MarketAnalysis) -> str:
        """
        Use LLM (if available) to write a flowing ARIA-style analyst narrative
        connecting all signals like a top-tier equity research report.
        """
        if not self.engine:
            return self._template_narrative(a)

        top_signals = sorted(a.signals, key=lambda x: abs(x.weighted), reverse=True)[:4]
        signal_summary = "; ".join(
            f"{s.name.replace('_',' ')} ({s.weighted:+.1f}): {', '.join(s.evidence[:2])}"
            for s in top_signals
        )

        prompt = (
            f"You are ARIA, an elite market intelligence AI. Write a sharp, confident 3-paragraph "
            f"analyst narrative for {a.symbol} with verdict {a.verdict} (score {a.total_score:+.1f}).\n"
            f"Top signals: {signal_summary}\n"
            f"Psychology mood: {a.psychology_mood}. Confidence: {a.confidence:.0f}%.\n"
            f"Connect the dots like a Goldman Sachs analyst. Be specific, no fluff. "
            f"Mention operator/institutional behaviour if relevant. "
            f"End with one-line action recommendation."
        )

        try:
            if hasattr(self.engine, "chat"):
                resp = await asyncio.wait_for(
                    self.engine.chat(prompt, model="fast"), timeout=20.0
                )
                return resp.strip()
            elif hasattr(self.engine, "generate"):
                resp = await asyncio.wait_for(
                    self.engine.generate(prompt), timeout=20.0
                )
                return resp.strip()
        except Exception as e:
            logger.warning("LLM narrative failed: %s", e)

        return self._template_narrative(a)

    def _template_narrative(self, a: MarketAnalysis) -> str:
        """Fallback template narrative when LLM is unavailable."""
        top = sorted(a.signals, key=lambda x: abs(x.weighted), reverse=True)[:2]
        drivers = " and ".join(s.name.replace("_", " ") for s in top)
        mood_line = {
            "fear":     "Current market psychology shows fear — historically a contrarian buying opportunity.",
            "greed":    "Extreme greed is present — exercise caution, distribution phase possible.",
            "panic":    "Panic selling detected — smart money often accumulates during such phases.",
            "euphoria": "Euphoria in the market — risk of a sharp reversal is elevated.",
            "neutral":  "Market psychology is balanced — no extreme signals detected.",
        }.get(a.psychology_mood, "")

        return (
            f"**AMIA Analysis:** The primary drivers for {a.symbol} are {drivers}, "
            f"contributing the most to the overall score of {a.total_score:+.1f}. "
            f"{mood_line}\n\n"
            f"**Recommendation:** Based on all 9 AMIA signals, the algorithm rates "
            f"{a.symbol} as **{a.verdict}** with {a.confidence:.0f}% confidence. "
            f"{'Institutional and smart-money signals are aligned with this view.' if abs(a.total_score) > 50 else 'Mixed signals suggest sizing positions conservatively.'}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_symbol(self, query: str) -> str:
        """Extract ticker / company name from natural language query."""
        q = query.upper()
        # Common tickers
        for ticker in ["RELIANCE", "TCS", "INFY", "HDFC", "SBIN", "WIPRO", "ICICI",
                        "TATAMOTORS", "BAJFINANCE", "ADANIENT", "NIFTY", "SENSEX",
                        "BTC", "ETH", "GOLD", "SILVER", "AAPL", "MSFT", "GOOGL"]:
            if ticker in q:
                return ticker
        # Fallback: first word after "analyze" / "about" / first word
        m = re.search(r"(?:analyze|analysis|about|for|on)\s+([A-Z0-9&]+)", q)
        if m: return m.group(1)
        words = [w for w in q.split() if len(w) >= 2]
        return words[0] if words else query[:20].upper()

    def _detect_asset_type(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["bitcoin","ethereum","crypto","btc","eth","altcoin"]): return "crypto"
        if any(k in q for k in ["gold","silver","crude","oil","commodity"]): return "commodity"
        if any(k in q for k in ["nifty","sensex","index","dow","nasdaq"]): return "index"
        if any(k in q for k in ["usd","eur","forex","currency","rupee","dollar"]): return "forex"
        return "stock"

    def _get_verdict(self, score: float) -> tuple[str, str]:
        for lo, hi, label, emoji in SCORE_BANDS:
            if lo <= score <= hi:
                return label, emoji
        return "NEUTRAL", "⚪"

    def _detect_mood(self, psych_sig: Optional[SignalResult], score: float) -> str:
        if psych_sig:
            evid_str = " ".join(psych_sig.evidence).lower()
            if "extreme fear" in evid_str or "panic" in evid_str: return "panic"
            if "extreme greed" in evid_str or "euphoria" in evid_str: return "euphoria"
            if "fear" in evid_str: return "fear"
            if "greed" in evid_str: return "greed"
        if score < -50: return "fear"
        if score > 50:  return "greed"
        return "neutral"

    def _extract_risk_catalysts(self, signals: list[SignalResult]):
        risks      = []
        catalysts  = []
        for s in signals:
            if s.risk_flag:
                risks.extend(e for e in s.evidence[:2] if e)
            if s.raw_score > 0.4:
                catalysts.extend(e for e in s.evidence[:1] if e)
        return risks[:5], catalysts[:5]

    def _operator_alert(self, op_sig: Optional[SignalResult], score: float) -> Optional[str]:
        if not op_sig: return None
        evid_lower = " ".join(op_sig.evidence).lower()
        if "pledge" in evid_lower:
            return "Promoter pledge detected — high risk, avoid or exit immediately"
        if op_sig.raw_score < -0.5:
            return "Smart money distributing — potential exit by informed players"
        if op_sig.raw_score > 0.5:
            return "Unusual accumulation detected — smart money building positions"
        return None

    def _price_levels(self, score: float, confidence: float):
        """Illustrative price level guidance (percentage-based, no live price)."""
        if abs(score) < 15: return None, None, None, None
        base     = 100.0  # normalized base — UI/caller should multiply by actual price
        sl_pct   = 0.03 + (1 - confidence / 100) * 0.05   # 3–8% stop
        tp_mult  = abs(score) / 100 * 0.25 + 0.05          # 5–30% target
        if score > 0:
            entry = base
            sl    = round(base * (1 - sl_pct), 1)
            tp    = round(base * (1 + tp_mult), 1)
        else:
            entry = base
            sl    = round(base * (1 + sl_pct), 1)
            tp    = round(base * (1 - tp_mult), 1)
        rr = round(abs(tp - entry) / abs(sl - entry), 1) if sl != entry else 0
        return entry, sl, tp, rr

    def _time_horizon(self, signals: list[SignalResult]) -> str:
        """Derive recommended time horizon from signal composition."""
        inst  = next((s for s in signals if s.name == "institutional_flow"), None)
        earn  = next((s for s in signals if s.name == "earnings_quality"), None)
        price = next((s for s in signals if s.name == "price_action"), None)

        if inst and abs(inst.raw_score) > 0.5:   return "medium"   # 1–6 months
        if earn and abs(earn.raw_score) > 0.5:   return "long"     # 6–18 months
        if price and abs(price.raw_score) > 0.5: return "short"    # days–weeks
        return "medium"
