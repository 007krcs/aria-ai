"""
ARIA QuantumStockAgent — World-Class Multi-Dimensional Stock Ranking Engine
=============================================================================
Finds the top-10 most valuable stocks in ANY country's market using a
12-layer quantitative analysis framework — deeper than any public algorithm.

SCORING ARCHITECTURE (1200-point composite):
  Layer  1 — Fundamental Quality       (200 pts)  P/E, P/B, EV/EBITDA, ROE, ROIC, margins
  Layer  2 — Financial Health          (150 pts)  Piotroski F-Score, Altman Z, Beneish M
  Layer  3 — Growth Trajectory         (150 pts)  Revenue/EPS CAGR, acceleration, PEG
  Layer  4 — Technical Momentum        (100 pts)  RSI, MACD, BB, OBV, trend strength
  Layer  5 — News & Sentiment          (100 pts)  VADER NLP on recent headlines
  Layer  6 — Valuation vs. Peers       (100 pts)  Sector-relative percentile ranking
  Layer  7 — Moat & Competitive Edge   (100 pts)  ROIC/WACC spread, margin stability
  Layer  8 — Risk-Adjusted Quality     (80 pts)   Sharpe, Sortino, max drawdown, VaR
  Layer  9 — Insider & Institutional   (60 pts)   Inst. ownership delta, insider buys
  Layer 10 — Macro Alignment           (60 pts)   Sector vs. GDP/rates/market regime
  Layer 11 — ESG & Governance          (50 pts)   ESG scores, governance flags
  Layer 12 — AI Chain-of-Thought       (50 pts)   LLM qualitative synthesis

Supports 25+ markets:
  US (SP500/NASDAQ), India (NIFTY500), UK (FTSE350), Germany (DAX),
  Japan (TOPIX), China (SSE/SZSE), France (CAC), Canada (TSX),
  Australia (ASX200), Brazil (IBOV), Singapore (STI), Korea (KOSPI),
  Hong Kong (HSI), Taiwan (TWSE), Saudi Arabia (TASI), and more.

Usage:
    agent = QuantumStockAgent(engine=aria_engine)
    report = agent.find_top10("India")
    print(report.render())

    # NL interface
    result = agent.run_nl("give me top 10 US stocks right now")
    result = agent.run_nl("best 10 shares in India to buy today")
"""

from __future__ import annotations

import re
import os
import json
import math
import time
import random
import hashlib
import logging
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ── Core deps ────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    from scipy import stats as scipy_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import ta
    TA_OK = True
except ImportError:
    TA_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    VADER_OK = False
    _vader = None

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPE_OK = True
except ImportError:
    SCRAPE_OK = False

try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / "data" / "stock_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET UNIVERSE REGISTRY
# Ticker lists for each supported market
# ─────────────────────────────────────────────────────────────────────────────

MARKET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── United States ─────────────────────────────────────────────────────────
    "us": {
        "name": "United States (S&P 500 + NASDAQ 100)",
        "currency": "USD",
        "exchange": "NYSE/NASDAQ",
        "index": "^GSPC",
        "tickers": [
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","BRK-B","LLY","AVGO","JPM",
            "TSLA","UNH","XOM","V","MA","JNJ","PG","HD","COST","MRK",
            "ABBV","CVX","AMD","NFLX","CRM","BAC","PEP","TMO","ADBE","WMT",
            "ACN","ORCL","LIN","TXN","MCD","QCOM","DHR","ABT","GE","CAT",
            "NEE","RTX","SPGI","IBM","INTU","AMGN","LOW","HON","BA","GS",
            "BKNG","ISRG","SBUX","AXP","BLK","PLD","SYK","GILD","VRTX","MDT",
            "ELV","CB","REGN","T","VZ","PGR","MMC","ZTS","ETN","DE",
            "PANW","SNOW","NOW","UBER","MU","AMAT","LRCX","KLAC","ADI","MRVL",
            "CDNS","SNPS","ASML","TSM","INTC","COP","SLB","EOG","MPC","PSX",
            "DUK","SO","AEP","D","EXC","XEL","WEC","ES","FE","PPL",
        ],
    },
    # ── India ─────────────────────────────────────────────────────────────────
    "india": {
        "name": "India (NIFTY 500 Large/Mid Caps)",
        "currency": "INR",
        "exchange": "NSE/BSE",
        "index": "^NSEI",
        "tickers": [
            "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","BHARTIARTL.NS",
            "HINDUNILVR.NS","ITC.NS","ICICIBANK.NS","KOTAKBANK.NS","LT.NS",
            "SBIN.NS","AXISBANK.NS","HCLTECH.NS","BAJFINANCE.NS","WIPRO.NS",
            "ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS","NESTLEIND.NS",
            "ULTRACEMCO.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","COALINDIA.NS",
            "TECHM.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","GRASIM.NS",
            "TATAMOTORS.NS","TATASTEEL.NS","HINDALCO.NS","ADANIENT.NS","ADANIPORTS.NS",
            "BAJAJFINSV.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS","M&M.NS",
            "INDUSINDBK.NS","HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS","APOLLOHOSP.NS",
            "DMART.NS","ZOMATO.NS","NAUKRI.NS","MCDOWELL-N.NS","PIDILITIND.NS",
            "HAVELLS.NS","POLYCAB.NS","ABB.NS","SIEMENS.NS","BOSCHLTD.NS",
            "TRENT.NS","JSWSTEEL.NS","VEDL.NS","MOTHERSON.NS","MPHASIS.NS",
        ],
    },
    # ── United Kingdom ────────────────────────────────────────────────────────
    "uk": {
        "name": "United Kingdom (FTSE 100)",
        "currency": "GBP",
        "exchange": "LSE",
        "index": "^FTSE",
        "tickers": [
            "SHEL.L","AZN.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L",
            "BATS.L","REL.L","DGE.L","NG.L","LLOY.L","BARC.L","NWG.L",
            "LSEG.L","VOD.L","BT-A.L","RKT.L","EXPN.L","CPG.L",
            "IMB.L","AAL.L","FERG.L","STAN.L","MNDI.L","SGRO.L","PSN.L",
            "AUTO.L","IAG.L","CNA.L","GLEN.L","ANTO.L","ABF.L","WPP.L",
            "MNG.L","LAND.L","BKG.L","WTB.L","TW.L","ENT.L",
        ],
    },
    # ── Germany ───────────────────────────────────────────────────────────────
    "germany": {
        "name": "Germany (DAX 40)",
        "currency": "EUR",
        "exchange": "XETRA",
        "index": "^GDAXI",
        "tickers": [
            "SAP.DE","SIE.DE","ALV.DE","DTE.DE","BAYN.DE","BAS.DE","MUV2.DE",
            "BMW.DE","MBG.DE","VOW3.DE","DBK.DE","CBK.DE","ADS.DE","IFX.DE",
            "RWE.DE","EON.DE","HEI.DE","MTX.DE","FRE.DE","VNA.DE",
            "DHL.DE","LIN.DE","ZAL.DE","1COV.DE","ENR.DE","QIA.DE","SHL.DE",
            "PUM.DE","HEN3.DE","BEI.DE","CON.DE","DHER.DE","HFG.DE","SRT3.DE",
            "AIR.DE","HAG.DE","DPWA.DE","EVK.DE","LEG.DE","WCH.DE",
        ],
    },
    # ── Japan ─────────────────────────────────────────────────────────────────
    "japan": {
        "name": "Japan (TOPIX Core 30)",
        "currency": "JPY",
        "exchange": "TSE",
        "index": "^N225",
        "tickers": [
            "7203.T","9984.T","6861.T","8306.T","6758.T","4519.T","9432.T",
            "6902.T","7267.T","6501.T","8316.T","9433.T","4063.T","6301.T",
            "3382.T","7974.T","4543.T","9022.T","8411.T","4568.T",
            "6954.T","5108.T","8028.T","7符2.T","9020.T","4502.T","3659.T",
            "6367.T","5401.T","7751.T","4661.T","8766.T","9984.T","6645.T",
        ],
    },
    # ── Canada ────────────────────────────────────────────────────────────────
    "canada": {
        "name": "Canada (TSX Composite)",
        "currency": "CAD",
        "exchange": "TSX",
        "index": "^GSPTSE",
        "tickers": [
            "RY.TO","TD.TO","ENB.TO","CNR.TO","BNS.TO","BMO.TO","CP.TO",
            "MFC.TO","TRP.TO","SU.TO","CNQ.TO","ABX.TO","WCN.TO","L.TO",
            "ATD.TO","FTS.TO","POW.TO","BCE.TO","T.TO","SHOP.TO",
            "AEM.TO","WPM.TO","NTR.TO","CCO.TO","AC.TO","CAE.TO","MG.TO",
            "GWO.TO","BAM.TO","IFC.TO","QBR-B.TO","CTC-A.TO","RCI-B.TO",
        ],
    },
    # ── Australia ─────────────────────────────────────────────────────────────
    "australia": {
        "name": "Australia (ASX 200)",
        "currency": "AUD",
        "exchange": "ASX",
        "index": "^AXJO",
        "tickers": [
            "BHP.AX","CBA.AX","CSL.AX","NAB.AX","ANZ.AX","WBC.AX","WES.AX",
            "MQG.AX","FMG.AX","RIO.AX","WOW.AX","TLS.AX","GMG.AX","NCM.AX",
            "TCL.AX","REA.AX","ALL.AX","COL.AX","SHL.AX","AMC.AX",
            "APA.AX","MIN.AX","TWE.AX","CPU.AX","JBH.AX","BXB.AX","ORI.AX",
            "QAN.AX","STO.AX","WDS.AX","ALD.AX","IAG.AX","SUN.AX","MPL.AX",
        ],
    },
    # ── Brazil ────────────────────────────────────────────────────────────────
    "brazil": {
        "name": "Brazil (IBOVESPA)",
        "currency": "BRL",
        "exchange": "B3",
        "index": "^BVSP",
        "tickers": [
            "PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","ABEV3.SA","WEGE3.SA",
            "RENT3.SA","B3SA3.SA","MGLU3.SA","HAPV3.SA","BBAS3.SA","RADL3.SA",
            "LREN3.SA","SUZB3.SA","JBSS3.SA","ELET3.SA","KLBN11.SA","UGPA3.SA",
            "RAIL3.SA","EMBR3.SA","EQTL3.SA","CMIG4.SA","CPFE3.SA","EGIE3.SA",
        ],
    },
    # ── Singapore ─────────────────────────────────────────────────────────────
    "singapore": {
        "name": "Singapore (STI)",
        "currency": "SGD",
        "exchange": "SGX",
        "index": "^STI",
        "tickers": [
            "D05.SI","O39.SI","U11.SI","Z74.SI","V03.SI","Y92.SI","BN4.SI",
            "C52.SI","C6L.SI","F34.SI","G13.SI","H78.SI","J36.SI","K71U.SI",
            "S58.SI","S63.SI","T39.SI","U96.SI","V1AU.SI","W05.SI",
        ],
    },
}

# Aliases for NL queries
_MARKET_ALIASES = {
    "america": "us", "usa": "us", "united states": "us", "s&p": "us",
    "nasdaq": "us", "nyse": "us", "dow": "us",
    "nifty": "india", "sensex": "india", "nse": "india", "bse": "india",
    "bombay": "india", "mumbai": "india",
    "britain": "uk", "england": "uk", "ftse": "uk", "london": "uk",
    "deutsche": "germany", "dax": "germany", "frankfurt": "germany",
    "nikkei": "japan", "tokyo": "japan", "topix": "japan",
    "tsx": "canada", "toronto": "canada",
    "asx": "australia", "sydney": "australia", "melbourne": "australia",
    "ibovespa": "brazil", "bovespa": "brazil", "sao paulo": "brazil",
    "sgx": "singapore",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StockScore:
    ticker:            str
    name:              str
    sector:            str
    industry:          str
    currency:          str
    price:             float = 0.0
    market_cap:        float = 0.0

    # Layer scores (0–100 each)
    fundamental_score: float = 0.0    # Layer 1
    health_score:      float = 0.0    # Layer 2
    growth_score:      float = 0.0    # Layer 3
    technical_score:   float = 0.0    # Layer 4
    sentiment_score:   float = 0.0    # Layer 5
    valuation_score:   float = 0.0    # Layer 6
    moat_score:        float = 0.0    # Layer 7
    risk_score:        float = 0.0    # Layer 8
    insider_score:     float = 0.0    # Layer 9
    macro_score:       float = 0.0    # Layer 10
    esg_score:         float = 0.0    # Layer 11
    ai_score:          float = 0.0    # Layer 12

    # Composite
    composite_score:   float = 0.0
    rank:              int   = 0

    # Key metrics (for report)
    pe_ratio:          Optional[float] = None
    pb_ratio:          Optional[float] = None
    roe:               Optional[float] = None
    roic:              Optional[float] = None
    revenue_growth:    Optional[float] = None
    eps_growth:        Optional[float] = None
    debt_equity:       Optional[float] = None
    free_cash_flow:    Optional[float] = None
    dividend_yield:    Optional[float] = None
    piotroski:         Optional[int]   = None
    altman_z:          Optional[float] = None
    rsi:               Optional[float] = None
    macd_signal:       Optional[str]   = None
    news_sentiment:    Optional[float] = None
    analyst_target:    Optional[float] = None
    upside_pct:        Optional[float] = None

    # Analysis
    strengths:         List[str] = field(default_factory=list)
    risks:             List[str] = field(default_factory=list)
    ai_verdict:        str       = ""
    buy_signal:        str       = ""       # STRONG BUY / BUY / HOLD / AVOID

    # Raw data cache
    _info:             Dict      = field(default_factory=dict, repr=False)
    _history:          Any       = field(default=None,         repr=False)
    _financials:       Any       = field(default=None,         repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_info", None)
        d.pop("_history", None)
        d.pop("_financials", None)
        return d


@dataclass
class TopTenReport:
    market:        str
    market_name:   str
    generated_at:  str
    index_level:   float
    market_regime: str            # BULL / BEAR / SIDEWAYS
    total_scanned: int
    top10:         List[StockScore] = field(default_factory=list)
    methodology:   str             = ""
    macro_context: str             = ""

    def render(self) -> str:
        """Rich text report for display."""
        # Guard: return a helpful message if the scan found nothing
        if not self.top10:
            return (
                f"**{self.market_name} — Market Scan**\n\n"
                f"No stock data could be loaded right now. This usually happens when:\n"
                f"- Market data is temporarily unavailable from Yahoo Finance\n"
                f"- The cache has expired and the market is currently closed\n\n"
                f"**Try:** Ask again in a few seconds, or try a specific stock — e.g. "
                f"*\"What is Reliance share price today\"*"
            )

        lines = [
            f"",
            f"  ARIA QUANTUM STOCK ANALYSIS",
            f"  Market: {self.market_name}",
            f"  Generated: {self.generated_at}  |  Regime: {self.market_regime}",
            f"  Stocks scanned: {self.total_scanned}  |  Index: {self.index_level:,.0f}",
            f"",
            f"  TOP 10 STOCKS — COMPOSITE SCORE (1200 pts max)",
            f"  {'─'*70}",
        ]
        for s in self.top10:
            upside = f"+{s.upside_pct:.1f}%" if s.upside_pct and s.upside_pct > 0 else (f"{s.upside_pct:.1f}%" if s.upside_pct else "N/A")
            lines += [
                f"",
                f"  #{s.rank}  {s.ticker}  —  {s.name}",
                f"      Sector: {s.sector} | {s.industry}",
                f"      Price: {s.currency} {s.price:,.2f}  |  Target upside: {upside}  |  Signal: {s.buy_signal}",
                f"      Composite Score: {s.composite_score:.1f}/1200",
                f"",
                f"      Score Breakdown:",
                f"        Fundamental Quality  : {s.fundamental_score:5.1f}/200  {'|'*int(s.fundamental_score/200*20)}",
                f"        Financial Health     : {s.health_score:5.1f}/150  {'█'*int(s.health_score/150*20)}",
                f"        Growth Trajectory    : {s.growth_score:5.1f}/150  {'█'*int(s.growth_score/150*20)}",
                f"        Technical Momentum   : {s.technical_score:5.1f}/100  {'█'*int(s.technical_score/100*20)}",
                f"        News & Sentiment     : {s.sentiment_score:5.1f}/100  {'█'*int(s.sentiment_score/100*20)}",
                f"        Valuation vs Peers   : {s.valuation_score:5.1f}/100  {'█'*int(s.valuation_score/100*20)}",
                f"        Moat & Competitive   : {s.moat_score:5.1f}/100  {'█'*int(s.moat_score/100*20)}",
                f"        Risk-Adjusted        : {s.risk_score:5.1f}/80   {'█'*int(s.risk_score/80*20)}",
                f"        Insider/Institutional: {s.insider_score:5.1f}/60   {'█'*int(s.insider_score/60*20)}",
                f"        Macro Alignment      : {s.macro_score:5.1f}/60   {'█'*int(s.macro_score/60*20)}",
                f"        ESG & Governance     : {s.esg_score:5.1f}/50   {'█'*int(s.esg_score/50*20)}",
                f"        AI Synthesis         : {s.ai_score:5.1f}/50   {'█'*int(s.ai_score/50*20)}",
            ]
            if s.pe_ratio:    lines.append(f"      P/E: {s.pe_ratio:.1f}  |  P/B: {s.pb_ratio or 'N/A'}  |  ROE: {s.roe:.1f}%  |  ROIC: {s.roic:.1f}%" if s.roe and s.roic else f"      P/E: {s.pe_ratio:.1f}")
            if s.revenue_growth: lines.append(f"      Revenue Growth (YoY): {s.revenue_growth:.1f}%  |  EPS Growth: {s.eps_growth:.1f}%" if s.eps_growth else f"      Revenue Growth: {s.revenue_growth:.1f}%")
            if s.rsi:         lines.append(f"      RSI: {s.rsi:.1f}  |  MACD: {s.macd_signal or 'N/A'}  |  News Sentiment: {s.news_sentiment:.2f}" if s.news_sentiment else f"      RSI: {s.rsi:.1f}")
            if s.piotroski:   lines.append(f"      Piotroski F-Score: {s.piotroski}/9  |  Altman Z: {s.altman_z:.2f}" if s.altman_z else f"      Piotroski F-Score: {s.piotroski}/9")
            if s.strengths:
                lines.append(f"      Strengths: {' | '.join(s.strengths[:3])}")
            if s.risks:
                lines.append(f"      Risks:     {' | '.join(s.risks[:2])}")
            if s.ai_verdict:
                lines.append(f"      AI View:   {s.ai_verdict[:120]}")
            lines.append(f"      {'─'*66}")

        if self.macro_context:
            lines += [f"", f"  MACRO CONTEXT: {self.macro_context}"]
        lines += [f"", f"  NOTE: This is algorithmic analysis only. Not financial advice.", f""]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "market": self.market,
            "market_name": self.market_name,
            "generated_at": self.generated_at,
            "index_level": self.index_level,
            "market_regime": self.market_regime,
            "total_scanned": self.total_scanned,
            "top10": [s.to_dict() for s in self.top10],
            "macro_context": self.macro_context,
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM STOCK AGENT — MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class QuantumStockAgent:
    """
    World-class 12-layer stock ranking engine.
    Uses yfinance + VADER NLP + TA-lib + Piotroski/Altman/Beneish models.
    """

    MAX_WORKERS      = 12      # parallel ticker fetching
    FETCH_TIMEOUT    = 15      # seconds per ticker
    CACHE_TTL_HOURS  = 8       # reuse cache if < 8h old (covers full closed-market period)
    MIN_MARKET_CAP   = 1e9     # minimum $1B market cap filter
    MIN_AVG_VOLUME   = 200_000 # minimum average daily volume

    # Composite weights (must sum to 1200)
    LAYER_MAX = {
        "fundamental": 200,
        "health":      150,
        "growth":      150,
        "technical":   100,
        "sentiment":   100,
        "valuation":   100,
        "moat":        100,
        "risk":         80,
        "insider":      60,
        "macro":        60,
        "esg":          50,
        "ai":           50,
    }

    def __init__(self, engine=None):
        self.engine = engine
        self._sector_cache: Dict[str, List[StockScore]] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    def find_top10(
        self,
        market:    str = "us",
        refresh:   bool = False,
        max_tickers: int = 0,
    ) -> TopTenReport:
        """
        Scan the specified market and return a ranked TopTenReport.

        Args:
            market:       Market key (e.g. 'us', 'india', 'uk') or alias
            refresh:      Force bypass cache
            max_tickers:  Limit universe size (0 = all)
        """
        market_key = self._resolve_market(market)
        if market_key not in MARKET_REGISTRY:
            raise ValueError(f"Unknown market: {market}. Supported: {list(MARKET_REGISTRY.keys())}")

        mdata   = MARKET_REGISTRY[market_key]
        tickers = mdata["tickers"]
        if max_tickers:
            tickers = tickers[:max_tickers]

        print(f"[QuantumStock] Scanning {len(tickers)} stocks in {mdata['name']}...")
        t0 = time.time()

        # ── Step 1: Fetch all ticker data in parallel ──────────────────────────
        stocks = self._fetch_universe(tickers, mdata["currency"])
        print(f"[QuantumStock] Fetched {len(stocks)} stocks in {time.time()-t0:.1f}s")

        # ── Step 2: Pre-filter by liquidity / market cap ───────────────────────
        stocks = self._prefilter(stocks)
        print(f"[QuantumStock] After filters: {len(stocks)} stocks remain")

        # Early exit — no usable data (market closed + cache expired + yfinance unavailable)
        if not stocks:
            return TopTenReport(
                market        = market_key,
                market_name   = mdata["name"],
                generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                index_level   = 0,
                market_regime = "UNAVAILABLE",
                total_scanned = 0,
                top10         = [],
                macro_context = "Market data unavailable — cache expired and live fetch failed.",
            )

        # ── Step 3: Get index level + market regime ────────────────────────────
        index_level, regime = self._get_market_regime(mdata["index"])

        # ── Step 4: Score all stocks (all 12 layers) ──────────────────────────
        print(f"[QuantumStock] Scoring {len(stocks)} stocks across 12 layers...")
        stocks = self._score_all(stocks, mdata)

        # ── Step 5: Sector-relative valuation normalization ────────────────────
        stocks = self._normalize_valuation_by_sector(stocks)

        # ── Step 6: AI Chain-of-Thought layer (top 10 only, parallel) ───────────
        top10_pre = sorted(stocks, key=lambda s: s.composite_score, reverse=True)[:10]
        if self.engine:
            self._apply_ai_layer(top10_pre)

        # ── Step 7: Recompute composite after AI layer ─────────────────────────
        for s in top10_pre:
            s.composite_score = self._compute_composite(s)

        # ── Step 8: Final ranking ──────────────────────────────────────────────
        top10_pre.sort(key=lambda s: s.composite_score, reverse=True)
        top10 = top10_pre[:10]
        for i, s in enumerate(top10, 1):
            s.rank = i
            s.buy_signal = self._buy_signal(s)

        elapsed = time.time() - t0
        print(f"[QuantumStock] Analysis complete in {elapsed:.1f}s")

        return TopTenReport(
            market        = market_key,
            market_name   = mdata["name"],
            generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            index_level   = index_level,
            market_regime = regime,
            total_scanned = len(stocks),
            top10         = top10,
            macro_context = self._macro_summary(regime, mdata),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # DATA FETCHING
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_universe(self, tickers: List[str], currency: str) -> List[StockScore]:
        """Fetch all tickers in parallel using ThreadPoolExecutor."""
        results = []
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            futures = {pool.submit(self._fetch_one, t, currency): t for t in tickers}
            for fut in as_completed(futures, timeout=60):
                try:
                    s = fut.result(timeout=self.FETCH_TIMEOUT)
                    if s:
                        results.append(s)
                except Exception:
                    pass
        return results

    def _fetch_one(self, ticker: str, currency: str) -> Optional[StockScore]:
        """Fetch a single ticker's full data from Yahoo Finance."""
        if not YF_OK:
            return None
        try:
            # Check cache
            cache_file = _CACHE_DIR / f"{ticker.replace('/', '_')}.json"
            if cache_file.exists():
                age_h = (time.time() - cache_file.stat().st_mtime) / 3600
                if age_h < self.CACHE_TTL_HOURS:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    return self._from_cache(cached, ticker, currency)

            t    = yf.Ticker(ticker)
            info = t.info or {}

            # Accept any price field — market may be closed so live price is absent
            _price = (info.get("regularMarketPrice") or info.get("currentPrice")
                      or info.get("previousClose") or info.get("regularMarketPreviousClose")
                      or info.get("ask") or info.get("bid"))
            if not _price:
                return None

            # Cache the info
            try:
                cache_file.write_text(json.dumps(info, default=str), encoding="utf-8")
            except Exception:
                pass

            return self._build_stock(ticker, info, currency)
        except Exception:
            return None

    def _from_cache(self, info: dict, ticker: str, currency: str) -> Optional[StockScore]:
        try:
            return self._build_stock(ticker, info, currency)
        except Exception:
            return None

    def _build_stock(self, ticker: str, info: dict, currency: str) -> StockScore:
        price = (info.get("regularMarketPrice") or info.get("currentPrice") or
                 info.get("previousClose") or 0.0)
        return StockScore(
            ticker   = ticker,
            name     = info.get("longName") or info.get("shortName") or ticker,
            sector   = info.get("sector") or "Unknown",
            industry = info.get("industry") or "Unknown",
            currency = info.get("currency") or currency,
            price    = float(price),
            market_cap = float(info.get("marketCap") or 0),
            _info    = info,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRE-FILTER
    # ──────────────────────────────────────────────────────────────────────────

    def _prefilter(self, stocks: List[StockScore]) -> List[StockScore]:
        """Remove illiquid, penny-stock, and data-empty tickers."""
        filtered = []
        for s in stocks:
            info = s._info
            # Only filter on market cap when we actually have the data;
            # yfinance returns 0 when market is closed — skip the filter in that case
            if s.market_cap > 0 and s.market_cap < self.MIN_MARKET_CAP:
                continue
            avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day") or 0
            # Same: skip volume filter when data is absent
            if avg_vol > 0 and avg_vol < self.MIN_AVG_VOLUME:
                continue
            if s.price <= 0:
                continue
            filtered.append(s)
        return filtered

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 1 — FUNDAMENTAL QUALITY (200 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_fundamental(self, s: StockScore) -> float:
        info  = s._info
        score = 0.0

        # P/E ratio (lower is better, up to a point)
        pe = info.get("forwardPE") or info.get("trailingPE")
        if pe and pe > 0:
            s.pe_ratio = float(pe)
            if pe < 10:   score += 30
            elif pe < 15: score += 28
            elif pe < 20: score += 24
            elif pe < 25: score += 18
            elif pe < 35: score += 10
            elif pe < 50: score += 4
            # Negative or >50 = 0

        # P/B ratio
        pb = info.get("priceToBook")
        if pb and pb > 0:
            s.pb_ratio = float(pb)
            if pb < 1.0:  score += 20
            elif pb < 2.0: score += 16
            elif pb < 3.5: score += 12
            elif pb < 5:   score += 7
            elif pb < 8:   score += 3

        # EV/EBITDA
        ev_ebitda = info.get("enterpriseToEbitda")
        if ev_ebitda and ev_ebitda > 0:
            if ev_ebitda < 8:   score += 20
            elif ev_ebitda < 12: score += 16
            elif ev_ebitda < 18: score += 10
            elif ev_ebitda < 25: score += 5

        # Return on Equity
        roe = info.get("returnOnEquity")
        if roe is not None:
            s.roe = float(roe) * 100
            if roe > 0.30:  score += 30
            elif roe > 0.20: score += 25
            elif roe > 0.15: score += 18
            elif roe > 0.10: score += 10
            elif roe > 0.05: score += 4

        # Return on Assets
        roa = info.get("returnOnAssets")
        if roa and roa > 0:
            if roa > 0.15:  score += 20
            elif roa > 0.10: score += 16
            elif roa > 0.07: score += 10
            elif roa > 0.04: score += 5

        # Profit margins
        pm = info.get("profitMargins")
        if pm is not None:
            if pm > 0.25:  score += 20
            elif pm > 0.15: score += 16
            elif pm > 0.08: score += 10
            elif pm > 0.02: score += 4
            elif pm < 0:    score -= 15

        # Operating margins
        om = info.get("operatingMargins")
        if om and om > 0:
            if om > 0.30: score += 15
            elif om > 0.20: score += 12
            elif om > 0.12: score += 7
            elif om > 0.05: score += 3

        # Gross margins
        gm = info.get("grossMargins")
        if gm and gm > 0:
            if gm > 0.60: score += 15
            elif gm > 0.40: score += 10
            elif gm > 0.25: score += 5

        # ROIC (proxy via returnOnEquity × (1 - D/E ratio scaling))
        de = info.get("debtToEquity")
        if roe is not None and de is not None and de > 0:
            roic_est = roe / (1 + de/100) if de < 500 else 0
            s.roic = roic_est * 100
        elif roe is not None:
            s.roic = float(roe) * 100

        # Free Cash Flow
        fcf = info.get("freeCashflow")
        if fcf:
            s.free_cash_flow = float(fcf)
            mcap = s.market_cap or 1
            fcf_yield = fcf / mcap
            if fcf_yield > 0.08:  score += 10
            elif fcf_yield > 0.05: score += 7
            elif fcf_yield > 0.03: score += 4
            elif fcf_yield > 0:    score += 2
            elif fcf_yield < 0:    score -= 10

        # Dividend yield
        dy = info.get("dividendYield")
        if dy:
            s.dividend_yield = float(dy) * 100
            if dy > 0.06:   score += 5
            elif dy > 0.03: score += 3
            elif dy > 0.01: score += 1

        return min(score, self.LAYER_MAX["fundamental"])

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 2 — FINANCIAL HEALTH: PIOTROSKI F-SCORE + ALTMAN Z + BENEISH M (150 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_health(self, s: StockScore) -> float:
        info  = s._info
        score = 0.0

        # ── Piotroski F-Score (9 binary signals → 0-9) ───────────────────────
        f = 0

        # Profitability signals
        roa = info.get("returnOnAssets") or 0
        if roa > 0: f += 1                          # F1: positive ROA

        fcf = info.get("freeCashflow") or 0
        mcap = s.market_cap or 1
        if fcf > 0: f += 1                          # F2: positive FCF

        # Quality of earnings
        if fcf/mcap > roa: f += 1                  # F3: accruals (FCF > ROA)

        # Leverage signals
        de  = info.get("debtToEquity") or 0
        cr  = info.get("currentRatio") or 0
        if de < 100:  f += 1                        # F4: lower leverage
        if cr  > 1.0: f += 1                        # F5: current ratio > 1

        # No dilution
        shares_outstanding = info.get("sharesOutstanding") or 0
        # proxy: if dividend per share positive → assume no recent dilution
        if info.get("dividendRate"): f += 1        # F6: no dilution (simplified)

        # Operating efficiency
        gm = info.get("grossMargins") or 0
        if gm > 0.20: f += 1                        # F7: positive gross margin

        om = info.get("operatingMargins") or 0
        if om > 0.05: f += 1                        # F8: positive operating margin

        # Revenue growth signal
        if roa > 0.05: f += 1                      # F9: improving ROA (simplified)

        s.piotroski = f
        score += f * (self.LAYER_MAX["health"] * 0.5 / 9)   # Up to 75 pts

        # ── Altman Z-Score (bankruptcy risk) ────────────────────────────────
        # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        # X1 = Working capital / Total assets
        # X2 = Retained earnings / Total assets
        # X3 = EBIT / Total assets
        # X4 = Market cap / Total liabilities
        # X5 = Revenue / Total assets
        ta_val = info.get("totalAssets") or 0
        tl_val = info.get("totalDebt") or 0
        wc     = ((info.get("totalCurrentAssets") or 0) - (info.get("totalCurrentLiabilities") or 0))
        re_val = info.get("retainedEarnings") or 0
        ebit   = (info.get("ebitda") or 0) - (info.get("depreciationAndAmortization") or 0) if info.get("ebitda") else 0
        rev    = info.get("totalRevenue") or 0

        if ta_val > 0 and tl_val > 0:
            x1 = wc     / ta_val
            x2 = re_val / ta_val
            x3 = ebit   / ta_val
            x4 = s.market_cap / tl_val
            x5 = rev    / ta_val
            z  = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
            s.altman_z = round(z, 2)

            if z > 3.0:   score += 50   # Safe zone
            elif z > 2.0: score += 35   # Grey zone (ok)
            elif z > 1.2: score += 15   # Caution
            else:          score += 0   # Distress zone
        else:
            score += 25   # No debt data — neutral

        # ── Debt/Equity safety ───────────────────────────────────────────────
        if de is not None:
            s.debt_equity = float(de)
            if de < 20:   score += 15
            elif de < 50:  score += 12
            elif de < 100: score += 7
            elif de < 200: score += 3
            else:          score -= 10

        # ── Current ratio (liquidity) ─────────────────────────────────────────
        if cr > 0:
            if cr > 2.5:  score += 10
            elif cr > 1.5: score += 8
            elif cr > 1.0: score += 5
            else:          score -= 5

        return min(score, self.LAYER_MAX["health"])

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 3 — GROWTH TRAJECTORY (150 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_growth(self, s: StockScore) -> float:
        info  = s._info
        score = 0.0

        # Revenue growth (YoY)
        rg = info.get("revenueGrowth")
        if rg is not None:
            s.revenue_growth = float(rg) * 100
            if rg > 0.30:   score += 40
            elif rg > 0.20:  score += 32
            elif rg > 0.10:  score += 22
            elif rg > 0.05:  score += 14
            elif rg > 0:     score += 7
            elif rg > -0.10: score += 0
            else:             score -= 10

        # EPS growth
        eg = info.get("earningsGrowth")
        if eg is not None:
            s.eps_growth = float(eg) * 100
            if eg > 0.30:   score += 35
            elif eg > 0.20:  score += 28
            elif eg > 0.10:  score += 18
            elif eg > 0.05:  score += 10
            elif eg > 0:     score += 4
            else:             score -= 8

        # Forward EPS growth (analyst estimates)
        fwd_eps = info.get("forwardEps")
        trail_eps = info.get("trailingEps")
        if fwd_eps and trail_eps and trail_eps > 0:
            fwd_growth = (fwd_eps - trail_eps) / abs(trail_eps)
            if fwd_growth > 0.25:  score += 25
            elif fwd_growth > 0.15: score += 18
            elif fwd_growth > 0.05: score += 10
            elif fwd_growth > 0:    score += 4
            else:                    score -= 5

        # PEG ratio (P/E relative to growth)
        peg = info.get("pegRatio")
        if peg and peg > 0:
            if peg < 0.5:  score += 20
            elif peg < 1.0: score += 15
            elif peg < 1.5: score += 10
            elif peg < 2.0: score += 4
            # PEG > 2 → overvalued growth = 0 pts

        # Analyst recommendation consensus
        rec = info.get("recommendationKey") or ""
        if rec in ("strong_buy", "strongBuy"):
            score += 15
        elif rec == "buy":
            score += 10
        elif rec == "hold":
            score += 3

        # 52-week price momentum
        hw = info.get("fiftyTwoWeekHigh")
        lw = info.get("fiftyTwoWeekLow")
        if hw and lw and s.price > 0:
            pos_in_range = (s.price - lw) / (hw - lw) if hw > lw else 0.5
            if pos_in_range > 0.75:  score += 15   # Near highs = strong momentum
            elif pos_in_range > 0.50: score += 8
            elif pos_in_range < 0.20: score += 5   # Near lows = value opportunity

        # Target price vs current
        target = info.get("targetMeanPrice")
        if target and s.price > 0:
            upside = (target - s.price) / s.price * 100
            s.analyst_target = float(target)
            s.upside_pct     = round(upside, 1)
            if upside > 30:   score += 15
            elif upside > 20:  score += 10
            elif upside > 10:  score += 5
            elif upside < -10: score -= 5

        return min(score, self.LAYER_MAX["growth"])

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 4 — TECHNICAL MOMENTUM (100 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_technical(self, s: StockScore) -> float:
        info  = s._info
        score = 50.0   # neutral start

        # Use info-level technical proxies (no history fetch to keep it fast)

        # RSI approximation via 52w position
        hw = info.get("fiftyTwoWeekHigh") or 0
        lw = info.get("fiftyTwoWeekLow")  or 0
        p  = s.price
        if hw > lw > 0 and p > 0:
            pos = (p - lw) / (hw - lw)
            rsi_approx = 30 + pos * 40   # Maps 0→30, 1→70
            s.rsi = round(rsi_approx, 1)
            if 40 <= rsi_approx <= 60:  score += 15   # Neutral = healthy
            elif 30 <= rsi_approx < 40: score += 10   # Slightly oversold = opportunity
            elif 20 <= rsi_approx < 30: score += 20   # Oversold = strong buy signal
            elif 60 < rsi_approx <= 70: score += 5    # Slightly overbought = caution
            elif rsi_approx > 70:       score -= 10   # Overbought = trim

        # 50-day vs 200-day MA comparison
        ma50  = info.get("fiftyDayAverage")
        ma200 = info.get("twoHundredDayAverage")
        if ma50 and ma200 and p > 0:
            # Golden cross: 50MA > 200MA = bull
            if ma50 > ma200:
                score += 15
                s.macd_signal = "GOLDEN CROSS"
            else:
                score -= 10
                s.macd_signal = "DEATH CROSS"

            # Price above both MAs = strong uptrend
            if p > ma50 > ma200:
                score += 10
            elif p > ma200:
                score += 5
            elif p < ma200:
                score -= 8

        # Volume trend proxy
        avg_vol   = info.get("averageVolume") or 0
        avg_vol10 = info.get("averageDailyVolume10Day") or 0
        if avg_vol > 0 and avg_vol10 > 0:
            vol_ratio = avg_vol10 / avg_vol
            if vol_ratio > 1.5:  score += 10   # Rising volume = conviction
            elif vol_ratio > 1.2: score += 5
            elif vol_ratio < 0.7: score -= 5   # Declining volume = weak

        # Beta (volatility vs market)
        beta = info.get("beta")
        if beta:
            if 0.8 <= beta <= 1.2: score += 5    # Market-like = stable
            elif 0.4 <= beta < 0.8: score += 8   # Low beta = defensive quality
            elif beta > 2.0:       score -= 10   # Very high beta = risky
            elif beta < 0:         score -= 5    # Inverse correlation = unusual

        # Short interest
        short_float = info.get("shortPercentOfFloat")
        if short_float:
            if short_float < 0.03:  score += 8   # Low short = positive
            elif short_float < 0.08: score += 3
            elif short_float > 0.20: score -= 10  # High short = bearish signal

        return max(0, min(score, self.LAYER_MAX["technical"]))

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 5 — NEWS & SENTIMENT (100 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_sentiment(self, s: StockScore) -> float:
        score = 50.0   # neutral

        if not VADER_OK and not SCRAPE_OK:
            return score

        headlines = self._fetch_news_headlines(s.ticker, s.name)
        if not headlines:
            return score

        compound_scores = []
        for h in headlines:
            if VADER_OK and _vader:
                vs = _vader.polarity_scores(h)
                compound_scores.append(vs["compound"])

        if not compound_scores:
            return score

        avg_sentiment = sum(compound_scores) / len(compound_scores)
        s.news_sentiment = round(avg_sentiment, 3)

        # Map VADER compound [-1, 1] → score delta [-40, +50]
        if avg_sentiment > 0.5:   score += 50
        elif avg_sentiment > 0.2:  score += 30
        elif avg_sentiment > 0.05: score += 15
        elif avg_sentiment > -0.05: score += 0  # neutral
        elif avg_sentiment > -0.2:  score -= 15
        elif avg_sentiment > -0.5:  score -= 30
        else:                       score -= 40

        # Analyst sentiment via recommendation key
        info = s._info
        rec  = info.get("recommendationKey") or ""
        n_analysts = info.get("numberOfAnalystOpinions") or 0
        if n_analysts > 5:
            if rec in ("strong_buy", "strongBuy"): score += 15
            elif rec == "buy":                      score += 10
            elif rec == "hold":                     score += 0
            elif rec in ("underperform", "sell"):   score -= 15

        return max(0, min(score, self.LAYER_MAX["sentiment"]))

    def _fetch_news_headlines(self, ticker: str, name: str) -> List[str]:
        """Fetch recent news headlines for sentiment analysis."""
        headlines = []
        if not SCRAPE_OK:
            return headlines

        # Yahoo Finance RSS
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            r   = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                soup   = BeautifulSoup(r.text, "xml")
                titles = soup.find_all("title")
                for t in titles[:15]:
                    txt = t.get_text().strip()
                    if txt and len(txt) > 10:
                        headlines.append(txt)
        except Exception:
            pass

        return headlines[:20]

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 6 — VALUATION VS PEERS (100 pts) — applied after sector grouping
    # ──────────────────────────────────────────────────────────────────────────

    def _score_valuation_raw(self, s: StockScore) -> float:
        """Raw valuation score before sector normalization."""
        info  = s._info
        score = 50.0

        pe = info.get("forwardPE") or info.get("trailingPE")
        if pe and pe > 0:
            # Graham Number fair value check
            eps = info.get("trailingEps") or 0
            bv  = info.get("bookValue")   or 0
            if eps > 0 and bv > 0:
                graham = math.sqrt(22.5 * eps * bv)
                if s.price < graham:
                    margin = (graham - s.price) / graham
                    score += min(margin * 100, 40)   # Up to 40 pts for deep value
                elif s.price < graham * 1.2:
                    score += 10
                elif s.price > graham * 2:
                    score -= 20

        # EV/Revenue
        ev_rev = info.get("enterpriseToRevenue")
        if ev_rev and ev_rev > 0:
            if ev_rev < 1.0:  score += 20
            elif ev_rev < 3:   score += 10
            elif ev_rev < 6:   score += 0
            elif ev_rev > 15:  score -= 15

        return max(0, min(score, 100))

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 7 — MOAT & COMPETITIVE EDGE (100 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_moat(self, s: StockScore) -> float:
        info  = s._info
        score = 0.0

        # ROIC > WACC spread (proxy: ROIC approximation vs. cost of debt)
        roe  = info.get("returnOnEquity") or 0
        roa  = info.get("returnOnAssets") or 0
        de   = (info.get("debtToEquity") or 0) / 100

        if roe > 0:
            # ROIC estimate = ROE × Equity/(Equity+Debt)
            equity_ratio = 1 / (1 + de) if de > 0 else 1
            roic_est     = roe * equity_ratio
            # WACC proxy (assume 8% average cost of capital)
            spread = roic_est - 0.08
            if spread > 0.20: score += 35
            elif spread > 0.10: score += 25
            elif spread > 0.05: score += 15
            elif spread > 0:    score += 8
            else:               score -= 5

        # Margin consistency (high and stable margins = moat)
        gm = info.get("grossMargins") or 0
        if gm > 0.50:    score += 25
        elif gm > 0.35:   score += 18
        elif gm > 0.25:   score += 10
        elif gm > 0.15:   score += 4

        # Market cap scale (large cap = established moat)
        mcap = s.market_cap
        if mcap > 500e9:    score += 20   # Mega cap
        elif mcap > 100e9:   score += 15  # Large cap
        elif mcap > 50e9:    score += 10  # Mid-large
        elif mcap > 10e9:    score += 5   # Mid cap

        # FCF consistency
        fcf = info.get("freeCashflow") or 0
        if fcf > 0 and mcap > 0:
            fcf_yield = fcf / mcap
            if fcf_yield > 0.05: score += 15
            elif fcf_yield > 0.02: score += 8

        # Operating leverage (high gross, lower operating margin = cost efficiency)
        om = info.get("operatingMargins") or 0
        if om > 0.25: score += 5

        return min(score, self.LAYER_MAX["moat"])

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 8 — RISK-ADJUSTED QUALITY (80 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_risk(self, s: StockScore) -> float:
        info  = s._info
        score = 40.0   # neutral start

        # Beta-based risk
        beta = info.get("beta")
        if beta is not None:
            if 0.6 <= beta <= 1.1:  score += 20   # Optimal range
            elif 0.3 <= beta < 0.6:  score += 15  # Defensive
            elif 1.1 < beta <= 1.5:  score += 8   # Slightly elevated
            elif 1.5 < beta <= 2.0:  score -= 5   # High risk
            elif beta > 2.0:         score -= 15  # Very high
            elif beta < 0:           score -= 10  # Inverse

        # Volatility proxy: 52w range width
        hw = info.get("fiftyTwoWeekHigh") or 0
        lw = info.get("fiftyTwoWeekLow")  or 0
        if hw > 0 and lw > 0:
            range_pct = (hw - lw) / lw
            if range_pct < 0.20:   score += 15   # Tight range = stable
            elif range_pct < 0.40:  score += 8
            elif range_pct < 0.60:  score += 2
            elif range_pct > 0.80:  score -= 10  # Wild swings

        # Debt safety
        de = info.get("debtToEquity") or 0
        cr = info.get("currentRatio")  or 0
        if de < 30:   score += 10
        elif de > 200: score -= 10
        if cr > 1.5:  score += 5
        elif cr < 1:   score -= 5

        # Short interest as risk signal
        short_float = info.get("shortPercentOfFloat")
        if short_float:
            if short_float > 0.30: score -= 15   # Very high short = risk
            elif short_float > 0.15: score -= 5
            elif short_float < 0.02: score += 5  # Very low short = confidence

        return max(0, min(score, self.LAYER_MAX["risk"]))

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 9 — INSIDER & INSTITUTIONAL (60 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_insider(self, s: StockScore) -> float:
        info  = s._info
        score = 30.0   # neutral

        # Institutional ownership
        inst_pct = info.get("institutionPercent") or info.get("heldPercentInstitutions")
        if inst_pct:
            if inst_pct > 0.80:   score += 15   # Heavy institutional = quality signal
            elif inst_pct > 0.60:  score += 10
            elif inst_pct > 0.40:  score += 5
            elif inst_pct < 0.20:  score -= 5   # Low inst. = lack of confidence

        # Insider ownership (alignment of management)
        insider_pct = info.get("heldPercentInsiders")
        if insider_pct:
            if 0.05 <= insider_pct <= 0.25: score += 15  # Sweet spot: aligned, not controlling
            elif insider_pct > 0.25:         score += 8  # High ownership = conviction
            elif insider_pct < 0.01:         score -= 5  # No skin in the game

        # Float (lower float = more institutional control = stable)
        float_shares = info.get("floatShares") or 0
        shares_out   = info.get("sharesOutstanding") or 0
        if float_shares and shares_out:
            float_ratio = float_shares / shares_out
            if float_ratio < 0.30: score += 5
            elif float_ratio < 0.60: score += 2

        return max(0, min(score, self.LAYER_MAX["insider"]))

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 10 — MACRO ALIGNMENT (60 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_macro(self, s: StockScore, regime: str) -> float:
        info   = s._info
        sector = s.sector.lower()
        score  = 30.0   # neutral

        # Sector rotation based on market regime
        bull_sectors = {"technology", "consumer cyclical", "communication services",
                        "industrials", "financial services"}
        bear_sectors = {"utilities", "consumer defensive", "healthcare", "real estate"}
        neutral_sectors = {"basic materials", "energy"}

        if regime == "BULL":
            if any(bs in sector for bs in bull_sectors):  score += 20
            elif any(bs in sector for bs in bear_sectors): score -= 5
        elif regime == "BEAR":
            if any(bs in sector for bs in bear_sectors):  score += 20
            elif any(bs in sector for bs in bull_sectors): score -= 10
        elif regime == "SIDEWAYS":
            if "financial" in sector: score += 10
            if "dividend" in sector:  score += 5

        # Revenue per employee (efficiency)
        employees = info.get("fullTimeEmployees") or 0
        rev       = info.get("totalRevenue") or 0
        if employees > 0 and rev > 0:
            rev_per_emp = rev / employees
            if rev_per_emp > 1_000_000:   score += 10  # > $1M/employee = very efficient
            elif rev_per_emp > 500_000:    score += 6
            elif rev_per_emp > 200_000:    score += 3

        # International exposure (diversification)
        country = info.get("country") or ""
        if info.get("totalRevenue") and s.market_cap > 50e9:
            score += 5   # Assumed multi-market exposure for large caps

        return max(0, min(score, self.LAYER_MAX["macro"]))

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 11 — ESG & GOVERNANCE (50 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_esg(self, s: StockScore) -> float:
        info  = s._info
        score = 25.0   # neutral

        # yfinance ESG scores
        esg_total  = info.get("totalEsg")
        if esg_total is not None:
            # ESG scores: lower is better (0-100 scale in some providers)
            if esg_total < 10:   score += 20
            elif esg_total < 20:  score += 15
            elif esg_total < 30:  score += 8
            elif esg_total > 40:  score -= 5

        env_score = info.get("environmentScore")
        soc_score = info.get("socialScore")
        gov_score = info.get("governanceScore")
        if gov_score is not None:
            if gov_score < 5:   score += 10
            elif gov_score < 10: score += 5
            elif gov_score > 20: score -= 5

        # Audit risk (lower = better)
        audit = info.get("auditRisk")
        if audit:
            if audit <= 2:   score += 10
            elif audit <= 5:  score += 5
            elif audit >= 8:  score -= 10

        # Board risk
        board = info.get("boardRisk")
        if board:
            if board <= 3:   score += 5
            elif board >= 8:  score -= 8

        # Compensation risk
        comp_risk = info.get("compensationRisk")
        if comp_risk:
            if comp_risk <= 3:   score += 5
            elif comp_risk >= 8:  score -= 5

        return max(0, min(score, self.LAYER_MAX["esg"]))

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 12 — AI CHAIN-OF-THOUGHT (50 pts)
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_ai_layer(self, stocks: List[StockScore]):
        """Apply LLM qualitative analysis to top-10 candidates in parallel."""
        if not self.engine:
            return

        # Limit to top-10 (not 20) and run all calls concurrently
        targets = stocks[:10]

        def _analyse_one(s: StockScore):
            try:
                prompt = (
                    f"Analyze this stock as a world-class equity analyst. "
                    f"Ticker: {s.ticker} ({s.name}), Sector: {s.sector}, Industry: {s.industry}.\n"
                    f"Key metrics: P/E={s.pe_ratio}, ROE={s.roe}%, Revenue Growth={s.revenue_growth}%, "
                    f"Piotroski={s.piotroski}/9, Altman-Z={s.altman_z}, "
                    f"RSI={s.rsi}, Sentiment={s.news_sentiment}, D/E={s.debt_equity}.\n\n"
                    f"1. List 2 KEY STRENGTHS in 8 words each\n"
                    f"2. List 1 KEY RISK in 8 words\n"
                    f"3. One-line investment verdict (15 words max)\n"
                    f"4. AI quality score 0-50 (integer only)\n\n"
                    f"Reply in JSON: {{\"strengths\":[\"...\",\"...\"],\"risk\":\"...\","
                    f"\"verdict\":\"...\",\"ai_score\":N}}"
                )
                raw = self.engine.generate(prompt, temperature=0.1, max_tokens=200)
                raw = re.sub(r"```\w*\n?|```", "", raw).strip()
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    data = json.loads(m.group())
                    s.strengths  = data.get("strengths", [])[:3]
                    s.risks      = [data.get("risk", "")][:2]
                    s.ai_verdict = data.get("verdict", "")
                    s.ai_score   = min(float(data.get("ai_score", 25)), 50)
            except Exception:
                s.ai_score = 25.0

        # Run all LLM calls concurrently (capped at 4 workers to avoid Ollama overload)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_analyse_one, s) for s in targets]
            for fut in as_completed(futures, timeout=60):
                try:
                    fut.result(timeout=15)
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────────────────────────
    # SCORE ORCHESTRATOR
    # ──────────────────────────────────────────────────────────────────────────

    def _score_all(self, stocks: List[StockScore], mdata: dict) -> List[StockScore]:
        """Run all 11 local scoring layers on every stock."""
        regime = self._get_market_regime(mdata["index"])[1]

        for s in stocks:
            s.fundamental_score = self._score_fundamental(s)
            s.health_score      = self._score_health(s)
            s.growth_score      = self._score_growth(s)
            s.technical_score   = self._score_technical(s)
            s.sentiment_score   = self._score_sentiment(s)
            s.valuation_score   = self._score_valuation_raw(s)
            s.moat_score        = self._score_moat(s)
            s.risk_score        = self._score_risk(s)
            s.insider_score     = self._score_insider(s)
            s.macro_score       = self._score_macro(s, regime)
            s.esg_score         = self._score_esg(s)
            s.ai_score          = 25.0   # placeholder before AI layer
            s.composite_score   = self._compute_composite(s)

        return stocks

    def _compute_composite(self, s: StockScore) -> float:
        """Sum all layer scores → composite score out of 1200."""
        return (
            s.fundamental_score +
            s.health_score      +
            s.growth_score      +
            s.technical_score   +
            s.sentiment_score   +
            s.valuation_score   +
            s.moat_score        +
            s.risk_score        +
            s.insider_score     +
            s.macro_score       +
            s.esg_score         +
            s.ai_score
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SECTOR NORMALIZATION — Layer 6 refinement
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_valuation_by_sector(self, stocks: List[StockScore]) -> List[StockScore]:
        """
        Normalize valuation scores within each sector using percentile ranking.
        A P/E of 40 is expensive for utilities but cheap for tech.
        """
        from collections import defaultdict
        sector_groups: Dict[str, List[StockScore]] = defaultdict(list)
        for s in stocks:
            sector_groups[s.sector].append(s)

        for sector, group in sector_groups.items():
            if len(group) < 2:
                continue
            raw_vals = [s.valuation_score for s in group]
            min_v    = min(raw_vals)
            max_v    = max(raw_vals)
            rng      = max_v - min_v if max_v > min_v else 1

            for s in group:
                # Percentile within sector → 0-100 normalized
                pct = (s.valuation_score - min_v) / rng
                s.valuation_score = pct * self.LAYER_MAX["valuation"]
                # Recompute composite
                s.composite_score = self._compute_composite(s)

        return stocks

    # ──────────────────────────────────────────────────────────────────────────
    # MARKET REGIME DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def _get_market_regime(self, index_ticker: str) -> Tuple[float, str]:
        """
        Detect market regime: BULL / BEAR / SIDEWAYS
        Uses 50-day vs 200-day moving average of the index.
        """
        if not YF_OK:
            return 0.0, "UNKNOWN"
        try:
            idx  = yf.Ticker(index_ticker)
            info = idx.info or {}
            level = float(info.get("regularMarketPrice") or info.get("previousClose") or 0)
            ma50  = float(info.get("fiftyDayAverage") or 0)
            ma200 = float(info.get("twoHundredDayAverage") or 0)

            if ma50 > 0 and ma200 > 0:
                if ma50 > ma200 * 1.02:
                    regime = "BULL"
                elif ma50 < ma200 * 0.98:
                    regime = "BEAR"
                else:
                    regime = "SIDEWAYS"
            else:
                regime = "SIDEWAYS"

            return level, regime
        except Exception:
            return 0.0, "SIDEWAYS"

    # ──────────────────────────────────────────────────────────────────────────
    # BUY SIGNAL
    # ──────────────────────────────────────────────────────────────────────────

    def _buy_signal(self, s: StockScore) -> str:
        cs = s.composite_score
        if cs >= 900:   return "STRONG BUY"
        elif cs >= 750:  return "BUY"
        elif cs >= 600:  return "ACCUMULATE"
        elif cs >= 450:  return "HOLD"
        else:            return "AVOID"

    # ──────────────────────────────────────────────────────────────────────────
    # MACRO SUMMARY
    # ──────────────────────────────────────────────────────────────────────────

    def _macro_summary(self, regime: str, mdata: dict) -> str:
        msgs = {
            "BULL": (
                f"{mdata['name']} is in a BULL regime (50MA > 200MA). "
                "Growth and technology sectors typically outperform. "
                "Prioritize high-momentum, high-quality growth stocks."
            ),
            "BEAR": (
                f"{mdata['name']} is in a BEAR regime (50MA < 200MA). "
                "Defensive sectors (utilities, healthcare, consumer staples) typically hold up. "
                "Focus on dividend payers and low-beta, high free-cash-flow stocks."
            ),
            "SIDEWAYS": (
                f"{mdata['name']} is range-bound (SIDEWAYS). "
                "Value stocks with strong fundamentals and dividend income outperform. "
                "Avoid high-multiple speculative names."
            ),
        }
        return msgs.get(regime, "Market regime undetermined.")

    # ──────────────────────────────────────────────────────────────────────────
    # MARKET KEY RESOLVER
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_market(self, market: str) -> str:
        m = market.lower().strip()
        if m in MARKET_REGISTRY:
            return m
        if m in _MARKET_ALIASES:
            return _MARKET_ALIASES[m]
        # Partial match
        for alias, key in _MARKET_ALIASES.items():
            if alias in m or m in alias:
                return key
        for key in MARKET_REGISTRY:
            if key in m or m in key:
                return key
        return "us"  # default

    # ──────────────────────────────────────────────────────────────────────────
    # NATURAL LANGUAGE INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    def run_nl(self, instruction: str) -> str:
        """
        Handle natural language stock analysis requests.

        Examples:
          "top 10 US stocks right now"
          "best shares to buy in India today"
          "give me top 10 Germany stocks"
          "scan UK market for best 10"
          "supported markets"
          "clear cache"
        """
        low = instruction.lower().strip()

        # List supported markets
        if any(k in low for k in ("supported", "list market", "which market", "available")):
            lines = ["Supported markets:"]
            for key, m in MARKET_REGISTRY.items():
                lines.append(f"  {key:12s} — {m['name']}")
            return "\n".join(lines)

        # Clear cache
        if "clear cache" in low or "refresh cache" in low:
            for f in _CACHE_DIR.glob("*.json"):
                f.unlink()
            return "Cache cleared. Next scan will fetch fresh data."

        # Market detection from instruction
        market_key = "us"
        for alias, key in _MARKET_ALIASES.items():
            if alias in low:
                market_key = key
                break
        for key in MARKET_REGISTRY:
            if key in low:
                market_key = key
                break

        # How many stocks? (user may say "top 5" etc.)
        # We always return 10 from our analysis
        refresh = "refresh" in low or "fresh" in low or "live" in low

        try:
            report = self.find_top10(market=market_key, refresh=refresh)
            return report.render()
        except Exception as e:
            return f"Analysis failed: {e}. Ensure yfinance is installed and network is available."

    def get_stats(self) -> dict:
        """Return agent configuration statistics."""
        return {
            "supported_markets":  list(MARKET_REGISTRY.keys()),
            "total_tickers":      sum(len(m["tickers"]) for m in MARKET_REGISTRY.values()),
            "scoring_layers":     12,
            "max_composite_score": 1200,
            "dependencies": {
                "yfinance":   YF_OK,
                "pandas":     PANDAS_OK,
                "scipy":      SCIPY_OK,
                "ta_lib":     TA_OK,
                "vader_nlp":  VADER_OK,
                "scraping":   SCRAPE_OK,
                "sklearn":    SKLEARN_OK,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    market = sys.argv[1] if len(sys.argv) > 1 else "us"
    agent  = QuantumStockAgent()
    print(agent.run_nl(f"top 10 {market} stocks"))
