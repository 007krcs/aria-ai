"""
ARIA Investment Timing Agent
============================
Mathematically rigorous investment timing engine.
All technical indicators computed from scratch — pure Python math, no TA library.
Uses Yahoo Finance unofficial API for live + historical OHLCV data.

Signal range: -100 (EXIT_NOW) to +100 (INVEST_NOW)
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import requests

logger = logging.getLogger("aria.investment_timing")

# ---------------------------------------------------------------------------
# Indian stock name fragments for auto-.NS suffix detection
# ---------------------------------------------------------------------------
INDIAN_STOCK_HINTS = {
    "RELIANCE", "TCS", "INFY", "INFOSYS", "HDFC", "ICICI", "WIPRO",
    "HCLTECH", "BAJFINANCE", "KOTAKBANK", "SBIN", "LT", "AXISBANK",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "ULTRACEMCO", "ONGC", "POWERGRID",
    "NTPC", "TATASTEEL", "TATAMOTORS", "TATACONSUM", "HINDUNILVR",
    "ADANIPORTS", "ADANIENT", "TITAN", "NESTLEIND", "BAJAJFINSV", "TECHM",
    "DRREDDY", "DIVISLAB", "CIPLA", "BRITANNIA", "EICHERMOT", "HEROMOTOCO",
    "JSWSTEEL", "COALINDIA", "BPCL", "GRASIM", "UPL", "INDUSINDBK",
    "SHREECEM", "M&M", "HINDALCO", "IOC", "VEDL", "PIDILITIND",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OHLCVData:
    symbol: str
    timestamps: list
    opens: list
    highs: list
    lows: list
    closes: list
    volumes: list


@dataclass
class IndicatorResult:
    name: str
    value: float
    signal: str        # BUY / SELL / NEUTRAL
    score: float
    description: str


@dataclass
class TimingSignal:
    symbol: str
    asset_type: str
    timestamp: str
    signal: str
    confidence: float
    timing_score: float
    current_price: float
    entry_zone: tuple
    stop_loss: float
    targets: list
    risk_reward: float
    time_horizon: str
    indicators: list
    why_moving: list
    push_trigger: bool
    push_message: str
    report_md: str


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class InvestmentTimingAgent:
    """
    Fetches OHLCV data from Yahoo Finance and computes 15 technical indicators
    from scratch to produce an investment timing score and actionable signal.
    """

    _last_scores: dict = {}   # symbol -> last timing_score for push detection

    def __init__(self, engine=None, notification_manager=None):
        self.engine = engine
        self.notification_manager = notification_manager
        self._watching = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, symbol: str, asset_type: str = "auto") -> TimingSignal:
        """Full analysis pipeline for one symbol."""
        try:
            symbol = self._normalize_symbol(symbol, asset_type)
            data = self._fetch_ohlcv(symbol)
            return self._compute_signal(data, asset_type)
        except Exception as exc:
            logger.exception("analyze(%s) failed: %s", symbol, exc)
            raise

    async def watch(self, symbols: list, interval_minutes: int = 5):
        """Background watcher — polls symbols every interval_minutes and fires push if needed."""
        self._watching = True
        logger.info("Starting watch loop for %s every %d min", symbols, interval_minutes)
        while self._watching:
            for sym in symbols:
                try:
                    signal = await self.analyze(sym)
                    if signal.push_trigger and self.notification_manager:
                        await self.notification_manager.send(signal.push_message)
                except Exception as exc:
                    logger.warning("watch: %s error: %s", sym, exc)
            await asyncio.sleep(interval_minutes * 60)

    def format_report(self, signal: TimingSignal) -> str:
        return signal.report_md

    async def run_nl(self, query: str) -> str:
        """Natural language interface — parses symbol from query and returns markdown report."""
        # Simple extraction: look for known ticker-like tokens
        tokens = query.upper().split()
        symbol = None
        for tok in tokens:
            tok_clean = tok.strip("?.,!:;")
            if 2 <= len(tok_clean) <= 10 and tok_clean.isalpha():
                symbol = tok_clean
                break
        if not symbol:
            return "Could not identify a stock symbol in your query. Please specify a ticker, e.g. 'Analyze RELIANCE' or 'What about AAPL?'"
        try:
            sig = await self.analyze(symbol)
            return sig.report_md
        except Exception as exc:
            return f"Analysis failed for {symbol}: {exc}"

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _normalize_symbol(self, symbol: str, hint: str = "auto") -> str:
        """Append .NS for Indian stocks if no exchange suffix present."""
        sym = symbol.upper().strip()
        if "." in sym:
            return sym
        if sym in INDIAN_STOCK_HINTS:
            return sym + ".NS"
        if hint and hint.lower() in ("nse", "bse", "india", "indian"):
            return sym + (".BO" if hint.lower() == "bse" else ".NS")
        return sym

    def _fetch_ohlcv(self, symbol: str) -> OHLCVData:
        """Fetch 6-month daily OHLCV from Yahoo Finance."""
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?range=6mo&interval=1d&includePrePost=false"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        body = resp.json()

        result = body["chart"]["result"][0]
        meta = result["meta"]
        timestamps = result.get("timestamp", [])
        indicators = result["indicators"]
        quote = indicators["quote"][0]

        opens   = [x if x is not None else 0.0 for x in quote.get("open",   [])]
        highs   = [x if x is not None else 0.0 for x in quote.get("high",   [])]
        lows    = [x if x is not None else 0.0 for x in quote.get("low",    [])]
        closes  = [x if x is not None else 0.0 for x in quote.get("close",  [])]
        volumes = [x if x is not None else 0   for x in quote.get("volume", [])]

        # Remove any bars where all values are 0 (gaps / bad data)
        valid = [
            i for i in range(len(closes))
            if closes[i] > 0
        ]
        return OHLCVData(
            symbol=symbol,
            timestamps=[timestamps[i] for i in valid],
            opens  =[opens[i]   for i in valid],
            highs  =[highs[i]   for i in valid],
            lows   =[lows[i]    for i in valid],
            closes =[closes[i]  for i in valid],
            volumes=[volumes[i] for i in valid],
        )

    # ------------------------------------------------------------------
    # Indicator maths — all from scratch
    # ------------------------------------------------------------------

    def _compute_ema(self, closes: list, period: int) -> list:
        """Exponential Moving Average. k = 2/(period+1)."""
        if len(closes) < period:
            return [closes[-1]] * len(closes)
        k = 2.0 / (period + 1)
        emas = [sum(closes[:period]) / period]   # seed with SMA
        for price in closes[period:]:
            emas.append(price * k + emas[-1] * (1 - k))
        # Pad front so length matches closes
        pad = len(closes) - len(emas)
        return [emas[0]] * pad + emas

    def _compute_rsi(self, closes: list, period: int = 14) -> float:
        """RSI = 100 - 100/(1 + RS); RS = avg_gain / avg_loss over period."""
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains  = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i])  / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _prev_rsi(self, closes: list, period: int = 14) -> float:
        """RSI one bar ago (for crossover detection)."""
        if len(closes) < period + 3:
            return 50.0
        return self._compute_rsi(closes[:-1], period)

    def _compute_macd(self, closes: list) -> tuple:
        """MACD(12,26,9). Returns (macd, signal, histogram)."""
        ema12 = self._compute_ema(closes, 12)
        ema26 = self._compute_ema(closes, 26)
        macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
        signal_line = self._compute_ema(macd_line, 9)
        histogram = [macd_line[i] - signal_line[i] for i in range(len(closes))]
        return macd_line[-1], signal_line[-1], histogram[-1], macd_line, signal_line, histogram

    def _compute_bbands(self, closes: list, period: int = 20, mult: float = 2.0) -> tuple:
        """Bollinger Bands. Returns (upper, middle, lower, pct_b)."""
        if len(closes) < period:
            mid = closes[-1]
            return mid, mid, mid, 0.5
        window = closes[-period:]
        mid = sum(window) / period
        variance = sum((x - mid) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper = mid + mult * std
        lower = mid - mult * std
        price = closes[-1]
        denom = upper - lower
        pct_b = (price - lower) / denom if denom != 0 else 0.5
        return upper, mid, lower, pct_b

    def _compute_stoch(self, highs: list, lows: list, closes: list,
                       k_period: int = 14, d_period: int = 3) -> tuple:
        """%K = (close - low14) / (high14 - low14) * 100; %D = SMA3(%K)."""
        if len(closes) < k_period:
            return 50.0, 50.0
        k_values = []
        for i in range(k_period - 1, len(closes)):
            window_h = highs[i - k_period + 1: i + 1]
            window_l = lows[i  - k_period + 1: i + 1]
            high14 = max(window_h)
            low14  = min(window_l)
            denom = high14 - low14
            k = (closes[i] - low14) / denom * 100 if denom != 0 else 50.0
            k_values.append(k)
        d_values = []
        for i in range(d_period - 1, len(k_values)):
            d_values.append(sum(k_values[i - d_period + 1: i + 1]) / d_period)
        k_now = k_values[-1]
        d_now = d_values[-1] if d_values else k_now
        return k_now, d_now

    def _compute_atr(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        """ATR = EMA14(TR); TR = max(H-L, |H-prev_C|, |L-prev_C|)."""
        if len(closes) < 2:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hpc = abs(highs[i] - closes[i - 1])
            lpc = abs(lows[i]  - closes[i - 1])
            trs.append(max(hl, hpc, lpc))
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0.0
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = tr * (2.0 / (period + 1)) + atr * (1 - 2.0 / (period + 1))
        return atr

    def _compute_obv(self, closes: list, volumes: list) -> list:
        """OBV: cumulative volume adjusted for direction."""
        obv = [0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        return obv

    def _compute_williams_r(self, highs: list, lows: list, closes: list,
                             period: int = 14) -> float:
        """%R = (high14 - close) / (high14 - low14) * -100."""
        if len(closes) < period:
            return -50.0
        h14 = max(highs[-period:])
        l14 = min(lows[-period:])
        denom = h14 - l14
        if denom == 0:
            return -50.0
        return (h14 - closes[-1]) / denom * -100.0

    def _compute_cci(self, highs: list, lows: list, closes: list, period: int = 20) -> float:
        """CCI = (TP - SMA20_TP) / (0.015 * mean_deviation)."""
        if len(closes) < period:
            return 0.0
        tp_series = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(len(closes))]
        window = tp_series[-period:]
        sma = sum(window) / period
        mean_dev = sum(abs(x - sma) for x in window) / period
        if mean_dev == 0:
            return 0.0
        return (tp_series[-1] - sma) / (0.015 * mean_dev)

    def _compute_adx(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        """ADX — Directional Movement Index for trend strength (0-100)."""
        if len(closes) < period + 1:
            return 20.0
        plus_dm_list, minus_dm_list, tr_list = [], [], []
        for i in range(1, len(closes)):
            up   = highs[i]  - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm  = up   if (up > down and up > 0)   else 0.0
            minus_dm = down if (down > up and down > 0) else 0.0
            hl  = highs[i]  - lows[i]
            hpc = abs(highs[i]  - closes[i - 1])
            lpc = abs(lows[i]   - closes[i - 1])
            tr  = max(hl, hpc, lpc)
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
            tr_list.append(tr)

        def _smooth(lst):
            s = sum(lst[:period])
            result = [s]
            for v in lst[period:]:
                s = s - s / period + v
                result.append(s)
            return result

        sm_tr    = _smooth(tr_list)
        sm_plus  = _smooth(plus_dm_list)
        sm_minus = _smooth(minus_dm_list)

        dx_list = []
        for i in range(len(sm_tr)):
            pdi = 100 * sm_plus[i]  / sm_tr[i] if sm_tr[i] != 0 else 0
            mdi = 100 * sm_minus[i] / sm_tr[i] if sm_tr[i] != 0 else 0
            denom = pdi + mdi
            dx = 100 * abs(pdi - mdi) / denom if denom != 0 else 0
            dx_list.append(dx)

        if len(dx_list) < period:
            return sum(dx_list) / len(dx_list) if dx_list else 20.0
        adx = sum(dx_list[:period]) / period
        for dx in dx_list[period:]:
            adx = (adx * (period - 1) + dx) / period
        return adx

    def _compute_vwap(self, highs: list, lows: list, closes: list, volumes: list) -> float:
        """VWAP = cumsum(TP * Vol) / cumsum(Vol)."""
        cum_tpv = 0.0
        cum_vol = 0.0
        for i in range(len(closes)):
            tp = (highs[i] + lows[i] + closes[i]) / 3.0
            cum_tpv += tp * volumes[i]
            cum_vol += volumes[i]
        if cum_vol == 0:
            return closes[-1]
        return cum_tpv / cum_vol

    def _compute_pivots(self, high: float, low: float, close: float) -> dict:
        """Classic pivot points from previous session H/L/C."""
        p  = (high + low + close) / 3.0
        r1 = 2 * p - low
        r2 = p + (high - low)
        r3 = r1 + (high - low)
        s1 = 2 * p - high
        s2 = p - (high - low)
        s3 = s1 - (high - low)
        return {"P": p, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}

    def _compute_fibonacci(self, closes: list) -> dict:
        """Fibonacci retracement levels from recent swing high/low."""
        n = min(50, len(closes))
        window = closes[-n:]
        swing_high = max(window)
        swing_low  = min(window)
        diff = swing_high - swing_low
        levels = {}
        for ratio in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]:
            levels[ratio] = swing_high - ratio * diff
        return {"swing_high": swing_high, "swing_low": swing_low, "levels": levels}

    def _detect_candlestick(self, opens: list, highs: list,
                             lows: list, closes: list) -> str:
        """
        Detect last-bar and recent candlestick patterns.
        Returns: BULLISH / BEARISH / NEUTRAL
        """
        if len(closes) < 3:
            return "NEUTRAL"

        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        po, ph, pl, pc = opens[-2], highs[-2], lows[-2], closes[-2]
        body     = abs(c - o)
        full_rng = h - l if (h - l) != 0 else 1e-9
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        # Doji
        if body / full_rng < 0.1:
            return "NEUTRAL"

        # Hammer (bullish)
        if lower_shadow >= 2 * body and upper_shadow < body and c > o:
            return "BULLISH"

        # Inverted Hammer / Shooting Star
        if upper_shadow >= 2 * body and lower_shadow < body and c < o:
            return "BEARISH"

        # Bullish Engulfing
        if pc > po and c > o and c > po and o < pc:
            return "BULLISH"

        # Bearish Engulfing
        if pc < po and c < o and c < po and o > pc:
            return "BEARISH"

        # Morning Star (3-bar)
        if len(closes) >= 3:
            o2, c2 = opens[-3], closes[-3]
            if c2 < o2 and abs(pc - po) < abs(c2 - o2) * 0.5 and c > o and c > (o2 + c2) / 2:
                return "BULLISH"

        # Evening Star (3-bar)
        if len(closes) >= 3:
            o2, c2 = opens[-3], closes[-3]
            if c2 > o2 and abs(pc - po) < abs(c2 - o2) * 0.5 and c < o and c < (o2 + c2) / 2:
                return "BEARISH"

        # Plain bullish/bearish
        if c > o and body / full_rng > 0.6:
            return "BULLISH"
        if c < o and body / full_rng > 0.6:
            return "BEARISH"

        return "NEUTRAL"

    def _detect_why_moving(self, data: OHLCVData) -> list:
        """Produce plain-English reasons explaining recent price movement."""
        reasons = []
        closes  = data.closes
        volumes = data.volumes
        highs   = data.highs
        lows    = data.lows

        if len(closes) < 5:
            return reasons

        # Volume surge
        avg_vol = sum(volumes[-21:-1]) / 20 if len(volumes) > 21 else sum(volumes) / len(volumes)
        if volumes[-1] > 2 * avg_vol:
            direction = "buying" if closes[-1] > closes[-2] else "selling"
            reasons.append(f"Unusual institutional {direction} — volume {volumes[-1]/avg_vol:.1f}x above average")

        # Gap detection
        gap_pct = (closes[-1] - closes[-2]) / closes[-2] * 100
        if abs(gap_pct) > 1.0:
            tag = "up" if gap_pct > 0 else "down"
            reasons.append(f"Gap {tag} detected ({gap_pct:+.2f}% from yesterday's close)")

        # ATR spike
        atr_now  = self._compute_atr(highs, lows, closes, 14)
        atr_prev = self._compute_atr(highs[:-5], lows[:-5], closes[:-5], 14)
        if atr_prev > 0 and atr_now > atr_prev * 1.5:
            reasons.append("Volatility expansion — ATR spiked significantly")

        # RSI divergence (bullish): price falling but RSI rising
        if len(closes) >= 10:
            rsi_now  = self._compute_rsi(closes, 14)
            rsi_prev = self._compute_rsi(closes[:-5], 14)
            price_change = closes[-1] - closes[-6]
            rsi_change   = rsi_now - rsi_prev
            if price_change < 0 and rsi_change > 5:
                reasons.append("Bullish RSI divergence forming — price declining but momentum recovering")
            if price_change > 0 and rsi_change < -5:
                reasons.append("Bearish RSI divergence forming — price rising but momentum weakening")

        # Multiple support confluence
        pivots = self._compute_pivots(highs[-2], lows[-2], closes[-2])
        price  = closes[-1]
        fib    = self._compute_fibonacci(closes)
        support_touches = 0
        tolerance = atr_now * 0.5
        for key in ["S1", "S2"]:
            if abs(price - pivots[key]) < tolerance:
                support_touches += 1
        for ratio, level in fib["levels"].items():
            if ratio in [0.382, 0.5, 0.618] and abs(price - level) < tolerance:
                support_touches += 1
        if support_touches >= 2:
            reasons.append("Multiple support confluence — price at key technical support zone")

        return reasons

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_signal(self, data: OHLCVData, asset_type: str) -> TimingSignal:
        closes  = data.closes
        highs   = data.highs
        lows    = data.lows
        opens   = data.opens
        volumes = data.volumes
        price   = closes[-1]

        # ---- Compute all indicators ----
        rsi       = self._compute_rsi(closes)
        prev_rsi  = self._prev_rsi(closes)
        macd_val, sig_val, hist_val, macd_line, signal_line, hist_line = self._compute_macd(closes)
        bb_upper, bb_mid, bb_lower, pct_b = self._compute_bbands(closes)
        ema9  = self._compute_ema(closes, 9)[-1]
        ema21 = self._compute_ema(closes, 21)[-1]
        ema50 = self._compute_ema(closes, 50)[-1]
        ema200= self._compute_ema(closes, 200)[-1]
        stoch_k, stoch_d = self._compute_stoch(highs, lows, closes)
        prev_k, prev_d   = self._compute_stoch(highs[:-1], lows[:-1], closes[:-1])
        atr   = self._compute_atr(highs, lows, closes)
        obv   = self._compute_obv(closes, volumes)
        will_r = self._compute_williams_r(highs, lows, closes)
        cci    = self._compute_cci(highs, lows, closes)
        adx    = self._compute_adx(highs, lows, closes)
        vwap   = self._compute_vwap(highs, lows, closes, volumes)
        pivots = self._compute_pivots(highs[-2], lows[-2], closes[-2])
        fib    = self._compute_fibonacci(closes)
        candle = self._detect_candlestick(opens, highs, lows, closes)

        # Volume surge
        avg_vol = sum(volumes[-21:-1]) / 20 if len(volumes) > 21 else max(sum(volumes) / len(volumes), 1)
        vol_ratio = volumes[-1] / avg_vol

        # OBV direction (compare last 5 bars)
        obv_rising = obv[-1] > obv[-6] if len(obv) >= 6 else False

        # MACD crossovers
        macd_bullish_cross = (macd_val > sig_val) and (macd_line[-2] < signal_line[-2]) if len(macd_line) >= 2 else False
        macd_bearish_cross = (macd_val < sig_val) and (macd_line[-2] > signal_line[-2]) if len(macd_line) >= 2 else False
        hist_increasing    = hist_line[-1] > hist_line[-2] if len(hist_line) >= 2 else False

        # RSI 50 crossover
        rsi_cross_50_up = rsi > 50 and prev_rsi < 50

        # Stochastic crossover
        stoch_bullish_cross = (stoch_k > stoch_d) and (prev_k < prev_d)

        # ---- Score computation ----
        raw_score = 0.0
        indicators: list[IndicatorResult] = []

        def _add(name: str, value: float, sig: str, pts: float, desc: str):
            nonlocal raw_score
            raw_score += pts
            indicators.append(IndicatorResult(name=name, value=value, signal=sig,
                                               score=pts, description=desc))

        # RSI
        if rsi < 30:
            _add("RSI(14)", rsi, "BUY", +15.0, f"Oversold at {rsi:.1f}")
        elif rsi > 70:
            _add("RSI(14)", rsi, "SELL", -15.0, f"Overbought at {rsi:.1f}")
        else:
            _add("RSI(14)", rsi, "NEUTRAL", 0.0, f"Neutral at {rsi:.1f}")
        if rsi_cross_50_up:
            _add("RSI 50 Cross", rsi, "BUY", +10.0, "RSI crossed above 50 — momentum turning bullish")

        # MACD
        if macd_bullish_cross:
            _add("MACD Cross", macd_val, "BUY", +20.0, "MACD bullish crossover")
        elif macd_bearish_cross:
            _add("MACD Cross", macd_val, "SELL", -20.0, "MACD bearish crossover")
        elif macd_val > sig_val:
            _add("MACD", macd_val, "BUY", +10.0, f"MACD {macd_val:.4f} above signal {sig_val:.4f}")
        else:
            _add("MACD", macd_val, "SELL", -10.0, f"MACD {macd_val:.4f} below signal {sig_val:.4f}")
        if hist_increasing:
            _add("MACD Histogram", hist_val, "BUY", +8.0, "Histogram increasing — momentum strengthening")
        else:
            _add("MACD Histogram", hist_val, "SELL", -4.0, "Histogram decreasing")

        # EMA 200 trend filter
        if price > ema200:
            _add("EMA200", ema200, "BUY", +8.0, f"Price above EMA200 ({ema200:.2f}) — long-term uptrend")
        else:
            _add("EMA200", ema200, "SELL", -8.0, f"Price below EMA200 ({ema200:.2f}) — long-term downtrend")

        # EMA 9/21
        if ema9 > ema21:
            _add("EMA9>21", ema9, "BUY", +10.0, f"EMA9 ({ema9:.2f}) > EMA21 ({ema21:.2f}) — short-term bullish")
        else:
            _add("EMA9>21", ema9, "SELL", -10.0, f"EMA9 ({ema9:.2f}) < EMA21 ({ema21:.2f}) — short-term bearish")

        # EMA 21/50
        if ema21 > ema50:
            _add("EMA21>50", ema21, "BUY", +8.0, f"EMA21 ({ema21:.2f}) > EMA50 ({ema50:.2f}) — medium-term bullish")
        else:
            _add("EMA21>50", ema21, "SELL", -8.0, f"EMA21 ({ema21:.2f}) < EMA50 ({ema50:.2f}) — medium-term bearish")

        # Bollinger Bands %B
        if pct_b < 0.05:
            _add("Bollinger %B", pct_b, "BUY", +15.0, f"Near lower band (%B={pct_b:.2f}) — strong oversold")
        elif pct_b > 0.95:
            _add("Bollinger %B", pct_b, "SELL", -15.0, f"Near upper band (%B={pct_b:.2f}) — strong overbought")
        else:
            _add("Bollinger %B", pct_b, "NEUTRAL", 0.0, f"%B={pct_b:.2f} — within bands")

        # Stochastic
        if stoch_k < 20:
            _add("Stochastic", stoch_k, "BUY", +12.0, f"Stoch %K={stoch_k:.1f} oversold")
        elif stoch_k > 80:
            _add("Stochastic", stoch_k, "SELL", -12.0, f"Stoch %K={stoch_k:.1f} overbought")
        else:
            _add("Stochastic", stoch_k, "NEUTRAL", 0.0, f"Stoch %K={stoch_k:.1f}")
        if stoch_bullish_cross:
            _add("Stoch Cross", stoch_k, "BUY", +10.0, f"%K crossed above %D ({stoch_d:.1f}) — bullish signal")

        # Williams %R
        if will_r < -80:
            _add("Williams %R", will_r, "BUY", +10.0, f"%R={will_r:.1f} — deeply oversold")
        elif will_r > -20:
            _add("Williams %R", will_r, "SELL", -10.0, f"%R={will_r:.1f} — overbought territory")
        else:
            _add("Williams %R", will_r, "NEUTRAL", 0.0, f"%R={will_r:.1f}")

        # CCI
        if cci < -100:
            _add("CCI(20)", cci, "BUY", +10.0, f"CCI={cci:.1f} — oversold")
        elif cci > 100:
            _add("CCI(20)", cci, "SELL", -10.0, f"CCI={cci:.1f} — overbought")
        else:
            _add("CCI(20)", cci, "NEUTRAL", 0.0, f"CCI={cci:.1f}")

        # OBV
        if obv_rising:
            _add("OBV", obv[-1], "BUY", +8.0, "OBV rising — accumulation in progress")
        else:
            _add("OBV", obv[-1], "SELL", -8.0, "OBV falling — distribution detected")

        # Volume surge
        if vol_ratio > 1.5:
            if closes[-1] > closes[-2]:
                _add("Volume Surge", vol_ratio, "BUY", +10.0, f"Volume surge {vol_ratio:.1f}x on up day — institutional buying")
            else:
                _add("Volume Surge", vol_ratio, "SELL", -10.0, f"Volume surge {vol_ratio:.1f}x on down day — institutional selling")
        else:
            _add("Volume", vol_ratio, "NEUTRAL", 0.0, f"Volume {vol_ratio:.1f}x avg")

        # VWAP
        if price < vwap:
            _add("VWAP", vwap, "BUY", +5.0, f"Price ({price:.2f}) below VWAP ({vwap:.2f}) — potential mean reversion up")
        else:
            _add("VWAP", vwap, "SELL", -5.0, f"Price ({price:.2f}) above VWAP ({vwap:.2f}) — slight sell bias")

        # Candlestick
        if candle == "BULLISH":
            _add("Candlestick", 1.0, "BUY", +15.0, "Bullish candlestick pattern detected")
        elif candle == "BEARISH":
            _add("Candlestick", -1.0, "SELL", -15.0, "Bearish candlestick pattern detected")
        else:
            _add("Candlestick", 0.0, "NEUTRAL", 0.0, "No clear candlestick pattern")

        # Pivot support/resistance
        tolerance = atr * 0.5
        if abs(price - pivots["S1"]) < tolerance:
            _add("Pivot S1", pivots["S1"], "BUY", +10.0, f"Price near S1 support ({pivots['S1']:.2f})")
        elif abs(price - pivots["R1"]) < tolerance:
            _add("Pivot R1", pivots["R1"], "SELL", -10.0, f"Price near R1 resistance ({pivots['R1']:.2f})")
        else:
            _add("Pivot", pivots["P"], "NEUTRAL", 0.0, f"Pivot P={pivots['P']:.2f}")

        # Fibonacci 0.618 retracement
        fib_618 = fib["levels"].get(0.618, 0.0)
        if abs(price - fib_618) < tolerance:
            _add("Fibonacci 0.618", fib_618, "BUY", +12.0, f"Price near golden ratio retracement ({fib_618:.2f})")
        else:
            _add("Fibonacci", fib_618, "NEUTRAL", 0.0, f"Fib 0.618 level: {fib_618:.2f}")

        # ADX amplification
        if adx > 25:
            raw_score *= 1.2
            indicators.append(IndicatorResult(
                name="ADX(14)", value=adx, signal="NEUTRAL",
                score=0.0, description=f"ADX={adx:.1f} — strong trend; signals amplified 1.2x"
            ))
        else:
            indicators.append(IndicatorResult(
                name="ADX(14)", value=adx, signal="NEUTRAL",
                score=0.0, description=f"ADX={adx:.1f} — weak/ranging market"
            ))

        # Clamp score
        timing_score = max(-100.0, min(100.0, raw_score))

        # ---- Signal thresholds ----
        signal_str, emoji = self._score_to_signal(timing_score)

        # ---- Entry / Exit levels ----
        entry_low  = price - atr * 0.3
        entry_high = price + atr * 0.3
        stop_loss  = price - atr * 1.5 if timing_score > 0 else price + atr * 1.5
        t1 = price + atr * 2
        t2 = price + atr * 4
        t3 = price + atr * 7
        rr_denom = abs(price - stop_loss)
        risk_reward = (t1 - price) / rr_denom if rr_denom != 0 else 0.0

        # ---- Time horizon ----
        if adx > 30:
            time_horizon = "3-10 days (strong trending)"
        elif adx > 20:
            time_horizon = "1-3 weeks (moderate trend)"
        else:
            time_horizon = "1-5 days (range-bound)"

        # ---- Confidence ----
        buy_sigs  = sum(1 for ind in indicators if ind.signal == "BUY")
        sell_sigs = sum(1 for ind in indicators if ind.signal == "SELL")
        total_sigs = max(buy_sigs + sell_sigs, 1)
        confidence = max(buy_sigs, sell_sigs) / total_sigs

        # ---- Push trigger ----
        last_score  = self._last_scores.get(data.symbol, 0.0)
        score_delta = abs(timing_score - last_score)
        prev_invest = abs(last_score) < 60
        curr_invest = abs(timing_score) >= 60
        rsi_cross30 = (rsi < 30 and prev_rsi >= 30) or (rsi > 70 and prev_rsi <= 70)

        push_trigger = (
            score_delta >= 20
            or (prev_invest and curr_invest)
            or rsi_cross30
            or macd_bullish_cross
            or macd_bearish_cross
        )
        self._last_scores[data.symbol] = timing_score

        push_message = ""
        if push_trigger:
            push_message = (
                f"{emoji} {signal_str} | {data.symbol} @ {price:.2f} | "
                f"Score: {timing_score:+.1f} | "
                f"Entry: {entry_low:.2f}-{entry_high:.2f} | "
                f"SL: {stop_loss:.2f} | T1: {t1:.2f}"
            )

        # ---- Why moving ----
        why_moving = self._detect_why_moving(data)

        # ---- Markdown report ----
        report_md = self.format_report_data(
            data.symbol, asset_type, signal_str, emoji, timing_score, confidence,
            price, entry_low, entry_high, stop_loss, t1, t2, t3, risk_reward,
            time_horizon, rsi, macd_val, sig_val, bb_upper, bb_lower, pct_b,
            ema9, ema21, ema50, ema200, stoch_k, stoch_d, atr, adx, vwap,
            indicators, why_moving, pivots, fib
        )

        return TimingSignal(
            symbol=data.symbol,
            asset_type=asset_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            signal=signal_str,
            confidence=round(confidence, 3),
            timing_score=round(timing_score, 2),
            current_price=price,
            entry_zone=(round(entry_low, 4), round(entry_high, 4)),
            stop_loss=round(stop_loss, 4),
            targets=[round(t1, 4), round(t2, 4), round(t3, 4)],
            risk_reward=round(risk_reward, 2),
            time_horizon=time_horizon,
            indicators=indicators,
            why_moving=why_moving,
            push_trigger=push_trigger,
            push_message=push_message,
            report_md=report_md,
        )

    def _score_to_signal(self, score: float) -> tuple:
        """Map numeric score to label and emoji."""
        if score >= 60:
            return "INVEST_NOW", "🟢"
        elif score >= 40:
            return "ACCUMULATE", "🔵"
        elif score >= 15:
            return "WATCH", "🔵"
        elif score >= -14:
            return "HOLD", "⚪"
        elif score >= -39:
            return "REDUCE", "🟡"
        elif score >= -60:
            return "SELL", "🟠"
        else:
            return "EXIT_NOW", "🔴"

    def format_report_data(
        self, symbol, asset_type, signal_str, emoji, timing_score, confidence,
        price, entry_low, entry_high, stop_loss, t1, t2, t3, risk_reward,
        time_horizon, rsi, macd_val, sig_val, bb_upper, bb_lower, pct_b,
        ema9, ema21, ema50, ema200, stoch_k, stoch_d, atr, adx, vwap,
        indicators, why_moving, pivots, fib
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        buy_count  = sum(1 for i in indicators if i.signal == "BUY")
        sell_count = sum(1 for i in indicators if i.signal == "SELL")

        lines = [
            f"# {emoji} {symbol} — {signal_str}",
            f"*Analysis generated: {now}*",
            f"",
            f"## Summary",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Signal** | {emoji} **{signal_str}** |",
            f"| **Timing Score** | `{timing_score:+.1f}` / 100 |",
            f"| **Confidence** | {confidence*100:.0f}% |",
            f"| **Current Price** | `{price:.4f}` |",
            f"| **Asset Type** | {asset_type} |",
            f"| **Time Horizon** | {time_horizon} |",
            f"| **Bull Signals** | {buy_count} |",
            f"| **Bear Signals** | {sell_count} |",
            f"",
            f"## Entry / Exit Levels",
            f"| Level | Price |",
            f"|-------|-------|",
            f"| **Entry Zone** | `{entry_low:.4f}` – `{entry_high:.4f}` |",
            f"| **Stop Loss** | `{stop_loss:.4f}` |",
            f"| **Target 1** | `{t1:.4f}` (+ATR×2) |",
            f"| **Target 2** | `{t2:.4f}` (+ATR×4) |",
            f"| **Target 3** | `{t3:.4f}` (+ATR×7) |",
            f"| **Risk/Reward** | `{risk_reward:.2f}x` |",
            f"",
            f"## Key Indicators",
            f"| Indicator | Value | Signal |",
            f"|-----------|-------|--------|",
            f"| RSI(14) | `{rsi:.1f}` | {'🟢 Oversold' if rsi < 30 else '🔴 Overbought' if rsi > 70 else '⚪ Neutral'} |",
            f"| MACD | `{macd_val:.4f}` (Sig: `{sig_val:.4f}`) | {'🟢' if macd_val > sig_val else '🔴'} |",
            f"| BB %B | `{pct_b:.2f}` | {'🟢 Near Lower' if pct_b < 0.2 else '🔴 Near Upper' if pct_b > 0.8 else '⚪'} |",
            f"| EMA 9/21/50/200 | `{ema9:.2f}` / `{ema21:.2f}` / `{ema50:.2f}` / `{ema200:.2f}` | {'🟢' if ema9 > ema21 else '🔴'} |",
            f"| Stoch %K/%D | `{stoch_k:.1f}` / `{stoch_d:.1f}` | {'🟢 OS' if stoch_k < 20 else '🔴 OB' if stoch_k > 80 else '⚪'} |",
            f"| ATR(14) | `{atr:.4f}` | ⚪ Volatility |",
            f"| ADX(14) | `{adx:.1f}` | {'🟢 Strong Trend' if adx > 25 else '⚪ Weak Trend'} |",
            f"| VWAP | `{vwap:.4f}` | {'🟢 Below VWAP' if price < vwap else '🔴 Above VWAP'} |",
            f"| BB Upper/Lower | `{bb_upper:.4f}` / `{bb_lower:.4f}` | ⚪ |",
            f"",
            f"## Pivot Points",
            f"| Level | Price |",
            f"|-------|-------|",
        ]
        for k, v in pivots.items():
            lines.append(f"| {k} | `{v:.4f}` |")

        lines += [
            f"",
            f"## Fibonacci Levels",
            f"| Ratio | Price |",
            f"|-------|-------|",
        ]
        for ratio, level in fib["levels"].items():
            lines.append(f"| {ratio*100:.1f}% | `{level:.4f}` |")

        if why_moving:
            lines += [f"", f"## Why Is It Moving?"]
            for reason in why_moving:
                lines.append(f"- {reason}")

        lines += [f"", f"## All Indicator Signals"]
        lines.append(f"| Indicator | Value | Signal | Score | Notes |")
        lines.append(f"|-----------|-------|--------|-------|-------|")
        for ind in indicators:
            sig_icon = "🟢" if ind.signal == "BUY" else "🔴" if ind.signal == "SELL" else "⚪"
            lines.append(
                f"| {ind.name} | `{ind.value:.4f}` | {sig_icon} {ind.signal} | "
                f"`{ind.score:+.1f}` | {ind.description} |"
            )

        lines += [
            f"",
            f"---",
            f"*ARIA Investment Timing Engine — For informational purposes only. Not financial advice.*",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main():
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    agent = InvestmentTimingAgent()
    print(f"Analyzing {symbol}...")
    signal = await agent.analyze(symbol)
    print(signal.report_md)


if __name__ == "__main__":
    asyncio.run(_main())
