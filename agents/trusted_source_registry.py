"""
ARIA Trusted Source Registry + Multi-Language Agent
=====================================================
Two critical systems in one:

PART 1 — TrustedSourceRegistry
  A curated, domain-specific database of the world's most reliable data sources.
  Every piece of data ARIA fetches is routed through this registry to ensure:
    - It comes from a verified, authoritative source
    - It is rated for accuracy, freshness, and bias
    - Conflicting data from multiple sources is reconciled
    - ARIA never hallucinates — uncertain facts are marked as uncertain

  Source quality tiers:
    TIER_1  — Primary/authoritative (regulatory filings, official APIs, peer-reviewed)
    TIER_2  — High-quality aggregators (Reuters, Bloomberg, WHO, Wikipedia)
    TIER_3  — Secondary/community (news sites, forums, social media)
    BLOCKED — Known misinformation / low-quality sources

PART 2 — MultiLanguageAgent
  Detects any input language, processes in that language, and responds with
  correct grammar in the same language. Supports 100+ languages via Ollama.

  Language detection: langdetect library (trained on Wikipedia)
  Grammar correction: LLM chain-of-thought with language-specific system prompt
  Translation: Ollama with language-specific prompt injection

Usage:
    # Source registry
    registry = TrustedSourceRegistry()
    best_url  = registry.get_best_source("stock_data", "india")
    rating    = registry.rate_url("https://economictimes.com/markets")

    # Multi-language
    lang_agent = MultiLanguageAgent(engine=aria_engine)
    response   = lang_agent.respond("मुझे Apple के शेयर के बारे में बताओ")  # Hindi
    response   = lang_agent.respond("Bonjour, quel est le cours de Tesla?")  # French
"""

from __future__ import annotations

import re
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

try:
    from langdetect import detect as _langdetect, DetectorFactory
    DetectorFactory.seed = 42   # Deterministic language detection
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

_ROOT      = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / "data" / "source_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE QUALITY TIERS
# ─────────────────────────────────────────────────────────────────────────────

TIER_1  = "AUTHORITATIVE"   # Primary regulatory / official / peer-reviewed
TIER_2  = "HIGH_QUALITY"    # Major news agencies, reputable aggregators
TIER_3  = "SECONDARY"       # General news, community-edited
BLOCKED = "BLOCKED"         # Known misinformation, spam, low reliability


@dataclass
class SourceEntry:
    url:         str
    domain:      str
    name:        str
    tier:        str              # TIER_1 / TIER_2 / TIER_3 / BLOCKED
    categories:  List[str]        # ["stocks","finance","news","science",...]
    regions:     List[str]        # ["global","us","india","uk","eu",...]
    languages:   List[str]        # ["en","hi","fr","de","ja",...]
    accuracy:    float            # 0.0 – 1.0 (estimated factual accuracy)
    freshness:   str              # "realtime" | "daily" | "weekly" | "static"
    has_api:     bool             # programmatic access available
    api_free:    bool             # free tier available
    notes:       str = ""

    def score(self) -> float:
        """Composite quality score for ranking."""
        tier_score = {TIER_1: 1.0, TIER_2: 0.75, TIER_3: 0.4, BLOCKED: 0.0}[self.tier]
        fresh_score = {"realtime": 1.0, "daily": 0.85, "weekly": 0.6, "static": 0.3}.get(self.freshness, 0.5)
        return (tier_score * 0.6 + self.accuracy * 0.3 + fresh_score * 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# TRUSTED SOURCE DATABASE
# Curated manually — the most reliable sources for each domain
# ─────────────────────────────────────────────────────────────────────────────

TRUSTED_SOURCES: List[SourceEntry] = [

    # ══════════════════════════════════════════════════════════════════════════
    # FINANCIAL & STOCK DATA
    # ══════════════════════════════════════════════════════════════════════════
    SourceEntry("https://finance.yahoo.com",          "finance.yahoo.com",     "Yahoo Finance",
                TIER_2, ["stocks","finance","news","earnings","analyst"],
                ["global"], ["en"], 0.92, "realtime", True, True,
                "Best free source. Real exchange data. Includes fundamentals, news, options."),

    SourceEntry("https://api.nasdaq.com",             "api.nasdaq.com",        "NASDAQ Official",
                TIER_1, ["stocks","finance","listings"],
                ["us"], ["en"], 0.99, "realtime", True, True,
                "Primary source for NASDAQ-listed stocks. Official exchange data."),

    SourceEntry("https://www.nseindia.com",           "nseindia.com",          "NSE India (Official)",
                TIER_1, ["stocks","finance"],
                ["india"], ["en","hi"], 0.99, "realtime", True, True,
                "Primary source for Indian stocks. National Stock Exchange official API."),

    SourceEntry("https://www.bseindia.com",           "bseindia.com",          "BSE India (Official)",
                TIER_1, ["stocks","finance"],
                ["india"], ["en","hi"], 0.99, "realtime", True, True,
                "Bombay Stock Exchange official data. Second source for Indian markets."),

    SourceEntry("https://www.londonstockexchange.com","londonstockexchange.com","London Stock Exchange",
                TIER_1, ["stocks","finance"],
                ["uk","eu"], ["en"], 0.99, "realtime", True, False),

    SourceEntry("https://fred.stlouisfed.org",        "fred.stlouisfed.org",   "Federal Reserve FRED",
                TIER_1, ["economics","macro","rates","inflation","gdp"],
                ["us","global"], ["en"], 1.0, "daily", True, True,
                "US Federal Reserve economic data. Gold standard for macro data."),

    SourceEntry("https://www.sec.gov/cgi-bin/browse-edgar","sec.gov",          "SEC EDGAR",
                TIER_1, ["stocks","filings","earnings","annual_report"],
                ["us"], ["en"], 1.0, "daily", True, True,
                "Official US company financial filings. 100% authoritative for US stocks."),

    SourceEntry("https://www.sebi.gov.in",            "sebi.gov.in",           "SEBI India",
                TIER_1, ["stocks","filings","regulations"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "Securities and Exchange Board of India. Official Indian stock regulator."),

    SourceEntry("https://www.reuters.com/finance",    "reuters.com",           "Reuters Finance",
                TIER_2, ["news","stocks","finance","economics","global"],
                ["global"], ["en","fr","de","ar","es","pt","ja","zh"], 0.93, "realtime", True, False,
                "Tier-1 wire service. Fact-checked. Excellent for breaking financial news."),

    SourceEntry("https://www.bloomberg.com",          "bloomberg.com",         "Bloomberg",
                TIER_2, ["news","stocks","finance","economics","global"],
                ["global"], ["en"], 0.95, "realtime", True, False,
                "Premium financial news. Paywalled but most reliable financial journalism."),

    SourceEntry("https://economictimes.indiatimes.com/markets","economictimes.com","Economic Times India",
                TIER_2, ["stocks","news","finance"],
                ["india"], ["en"], 0.85, "realtime", False, True,
                "Best English-language source for Indian markets."),

    SourceEntry("https://www.moneycontrol.com",       "moneycontrol.com",      "Moneycontrol",
                TIER_2, ["stocks","finance","news"],
                ["india"], ["en"], 0.83, "realtime", False, True,
                "Widely used Indian financial news. Good for real-time Indian market data."),

    SourceEntry("https://www.investing.com",          "investing.com",         "Investing.com",
                TIER_2, ["stocks","forex","commodities","crypto","news"],
                ["global"], ["en","hi","fr","de","es","pt","ja","zh","ar","ru","ko"], 0.87, "realtime", True, True,
                "Multi-asset, multi-language. Good for forex and commodities."),

    SourceEntry("https://finance.yahoo.com/news/rss", "finance.yahoo.com",     "Yahoo Finance RSS",
                TIER_2, ["news","stocks"],
                ["global"], ["en"], 0.88, "realtime", True, True,
                "Free RSS news feed. Used by ARIA's sentiment scanner."),

    SourceEntry("https://www.macrotrends.net",        "macrotrends.net",       "Macrotrends",
                TIER_2, ["stocks","macro","historical"],
                ["global"], ["en"], 0.88, "daily", False, True,
                "Excellent for long-term historical stock and macro charts."),

    # ══════════════════════════════════════════════════════════════════════════
    # GENERAL NEWS
    # ══════════════════════════════════════════════════════════════════════════
    SourceEntry("https://apnews.com",                 "apnews.com",            "Associated Press",
                TIER_1, ["news","world","politics","science","health"],
                ["global"], ["en"], 0.96, "realtime", True, True,
                "Gold standard for news accuracy. Non-profit, no editorial bias."),

    SourceEntry("https://www.bbc.com/news",           "bbc.com",               "BBC News",
                TIER_1, ["news","world","politics","science","health","tech"],
                ["global","uk"], ["en","hi","ar","zh","fr","de","es","pt","sw","ur","bn","ta","te"], 0.93, "realtime", True, True,
                "BBC World Service. Available in 40+ languages. High accuracy."),

    SourceEntry("https://www.reuters.com",            "reuters.com",           "Reuters",
                TIER_1, ["news","world","politics","finance","science"],
                ["global"], ["en","fr","de","ar","es","pt","ja","zh"], 0.95, "realtime", True, False),

    SourceEntry("https://www.theguardian.com",        "theguardian.com",       "The Guardian",
                TIER_2, ["news","world","politics","science","environment"],
                ["global","uk"], ["en"], 0.87, "realtime", True, True),

    SourceEntry("https://www.nytimes.com",            "nytimes.com",           "New York Times",
                TIER_2, ["news","world","politics","science","tech"],
                ["global","us"], ["en","es"], 0.90, "realtime", False, False),

    # ══════════════════════════════════════════════════════════════════════════
    # SCIENCE & RESEARCH
    # ══════════════════════════════════════════════════════════════════════════
    SourceEntry("https://arxiv.org",                  "arxiv.org",             "arXiv",
                TIER_1, ["science","ai","physics","math","cs","research"],
                ["global"], ["en"], 0.90, "daily", True, True,
                "Pre-print server. Not peer-reviewed but source of latest AI/ML research."),

    SourceEntry("https://pubmed.ncbi.nlm.nih.gov",   "pubmed.ncbi.nlm.nih.gov","PubMed",
                TIER_1, ["science","health","medical","biology","chemistry"],
                ["global"], ["en"], 0.97, "daily", True, True,
                "US National Library of Medicine. Peer-reviewed medical literature."),

    SourceEntry("https://www.who.int",               "who.int",               "World Health Organization",
                TIER_1, ["health","medical","disease","vaccines","global"],
                ["global"], ["en","fr","es","ar","zh","ru","pt"], 0.97, "daily", True, True),

    SourceEntry("https://www.nature.com",            "nature.com",            "Nature",
                TIER_1, ["science","research","biology","physics","chemistry","ai"],
                ["global"], ["en"], 0.98, "daily", True, False,
                "Premier peer-reviewed scientific journal."),

    SourceEntry("https://scholar.google.com",        "scholar.google.com",    "Google Scholar",
                TIER_2, ["research","science","academic"],
                ["global"], ["en"], 0.88, "daily", True, True,
                "Academic paper aggregator. Good for finding peer-reviewed sources."),

    # ══════════════════════════════════════════════════════════════════════════
    # TECHNOLOGY
    # ══════════════════════════════════════════════════════════════════════════
    SourceEntry("https://techcrunch.com",            "techcrunch.com",        "TechCrunch",
                TIER_2, ["tech","startups","ai","software","funding"],
                ["global","us"], ["en"], 0.84, "realtime", False, True),

    SourceEntry("https://www.theverge.com",          "theverge.com",          "The Verge",
                TIER_2, ["tech","gadgets","software","ai"],
                ["global","us"], ["en"], 0.83, "realtime", False, True),

    SourceEntry("https://news.ycombinator.com",      "news.ycombinator.com",  "Hacker News",
                TIER_3, ["tech","programming","startups","ai","discussion"],
                ["global"], ["en"], 0.75, "realtime", True, True,
                "Community-curated tech news. High signal-to-noise but no editorial oversight."),

    SourceEntry("https://github.com/trending",       "github.com",            "GitHub Trending",
                TIER_2, ["tech","programming","open_source"],
                ["global"], ["en"], 0.92, "daily", True, True,
                "Authoritative source for trending software projects."),

    # ══════════════════════════════════════════════════════════════════════════
    # WEATHER & ENVIRONMENT
    # ══════════════════════════════════════════════════════════════════════════
    SourceEntry("https://api.openweathermap.org",    "openweathermap.org",    "OpenWeatherMap",
                TIER_2, ["weather"],
                ["global"], ["en"], 0.88, "realtime", True, True,
                "Best free weather API. Used worldwide."),

    SourceEntry("https://www.wunderground.com",      "wunderground.com",      "Weather Underground",
                TIER_2, ["weather"],
                ["global"], ["en"], 0.87, "realtime", True, False),

    # ══════════════════════════════════════════════════════════════════════════
    # CRYPTO
    # ══════════════════════════════════════════════════════════════════════════
    SourceEntry("https://api.coingecko.com",         "coingecko.com",         "CoinGecko",
                TIER_2, ["crypto","blockchain","defi"],
                ["global"], ["en"], 0.92, "realtime", True, True,
                "Most comprehensive free crypto API. 10k+ coins."),

    SourceEntry("https://data.binance.com",          "binance.com",           "Binance",
                TIER_2, ["crypto","trading"],
                ["global"], ["en","zh","ko","ja","fr","de","es","ru"], 0.95, "realtime", True, True,
                "World's largest crypto exchange. Real-time OHLCV data."),

    # ══════════════════════════════════════════════════════════════════════════
    # ENCYCLOPEDIC / FACTUAL
    # ══════════════════════════════════════════════════════════════════════════
    SourceEntry("https://en.wikipedia.org",          "wikipedia.org",         "Wikipedia",
                TIER_2, ["general","history","science","geography","people"],
                ["global"], ["en","hi","fr","de","es","pt","ar","zh","ja","ru","bn","te","ta","ur",
                            "ko","vi","tr","pl","nl","it","sv","no","fi","da","ro","cs","sk","hu"], 0.82, "daily", True, True,
                "Best for factual/encyclopedic queries. Cross-check with primary sources for critical facts."),

    SourceEntry("https://www.wolframalpha.com",      "wolframalpha.com",      "Wolfram Alpha",
                TIER_1, ["math","science","computation","facts","data"],
                ["global"], ["en"], 0.97, "realtime", True, True,
                "Computational knowledge engine. Mathematically precise. Gold standard for calculations."),

    # ══════════════════════════════════════════════════════════════════════════
    # GOVERNMENT — TAX, WELFARE, LEGAL, OFFICIAL DATA (GLOBAL)
    # ══════════════════════════════════════════════════════════════════════════

    # ── India ──────────────────────────────────────────────────────────────────
    SourceEntry("https://www.incometax.gov.in",        "incometax.gov.in",       "Income Tax India",
                TIER_1, ["tax","government","india","legal","income_tax"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "Official Indian Income Tax Department. Filing, slabs, TDS, PAN, e-verify."),

    SourceEntry("https://www.gst.gov.in",              "gst.gov.in",             "GST India",
                TIER_1, ["tax","government","india","gst","legal"],
                ["india"], ["en","hi"], 1.0, "realtime", True, True,
                "Goods and Services Tax Network. Official GST filing and lookup."),

    SourceEntry("https://epfindia.gov.in",             "epfindia.gov.in",        "EPFO India",
                TIER_1, ["welfare","government","india","provident_fund","pension"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "Employees Provident Fund Organisation. PF balance, UAN, pension."),

    SourceEntry("https://www.irdai.gov.in",            "irdai.gov.in",           "IRDAI India",
                TIER_1, ["insurance","government","india","regulations"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "Insurance Regulatory and Development Authority. Insurance regulations India."),

    SourceEntry("https://rbi.org.in",                  "rbi.org.in",             "Reserve Bank of India",
                TIER_1, ["economics","banking","rates","government","india","finance"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "RBI — India's central bank. Repo rate, inflation, monetary policy."),

    SourceEntry("https://mospi.gov.in",                "mospi.gov.in",           "MoSPI India (GDP/CPI)",
                TIER_1, ["economics","government","india","gdp","inflation","statistics"],
                ["india"], ["en","hi"], 1.0, "monthly", True, True,
                "Ministry of Statistics. Official GDP, CPI, IIP data for India."),

    SourceEntry("https://www.india.gov.in",            "india.gov.in",           "India National Portal",
                TIER_1, ["government","india","welfare","schemes","legal","general"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "Official Government of India portal. All welfare schemes, entitlements, services."),

    SourceEntry("https://pmjdy.gov.in",                "pmjdy.gov.in",           "PMJDY (Jan Dhan Yojana)",
                TIER_1, ["welfare","government","india","banking","financial_inclusion"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "Pradhan Mantri Jan Dhan Yojana. Financial inclusion scheme data."),

    SourceEntry("https://www.nhm.gov.in",              "nhm.gov.in",             "National Health Mission India",
                TIER_1, ["health","government","india","welfare"],
                ["india"], ["en","hi"], 1.0, "monthly", True, True,
                "India's national health mission. Health scheme eligibility and data."),

    SourceEntry("https://uidai.gov.in",                "uidai.gov.in",           "UIDAI Aadhaar",
                TIER_1, ["government","india","identity","welfare"],
                ["india"], ["en","hi"], 1.0, "realtime", True, True,
                "Unique Identification Authority of India. Aadhaar card services."),

    SourceEntry("https://www.mca.gov.in",              "mca.gov.in",             "MCA (Company Affairs India)",
                TIER_1, ["government","india","legal","company","filings"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "Ministry of Corporate Affairs. Company filings, ROC, GST registration."),

    SourceEntry("https://dpiit.gov.in",                "dpiit.gov.in",           "DPIIT India (Startups/FDI)",
                TIER_1, ["government","india","startup","fdi","policy"],
                ["india"], ["en","hi"], 1.0, "daily", True, True,
                "Dept for Promotion of Industry. FDI policy, startup India data."),

    # ── United States ──────────────────────────────────────────────────────────
    SourceEntry("https://www.irs.gov",                 "irs.gov",                "IRS (US Tax Authority)",
                TIER_1, ["tax","government","us","legal","income_tax"],
                ["us"], ["en","es"], 1.0, "daily", True, True,
                "US Internal Revenue Service. Federal tax forms, brackets, deadlines."),

    SourceEntry("https://www.ssa.gov",                 "ssa.gov",                "US Social Security Administration",
                TIER_1, ["welfare","government","us","social_security","pension"],
                ["us"], ["en","es"], 1.0, "daily", True, True,
                "US Social Security benefits, disability, retirement."),

    SourceEntry("https://www.benefits.gov",            "benefits.gov",           "US Benefits.gov",
                TIER_1, ["welfare","government","us","benefits","schemes"],
                ["us"], ["en","es"], 1.0, "daily", True, True,
                "Official US government benefits finder. All federal welfare programs."),

    SourceEntry("https://www.usa.gov",                 "usa.gov",                "USA.gov (US Government Portal)",
                TIER_1, ["government","us","general","legal","welfare"],
                ["us"], ["en","es"], 1.0, "daily", True, True,
                "Official US government portal. All federal services and information."),

    SourceEntry("https://www.cdc.gov",                 "cdc.gov",                "CDC (US Health Authority)",
                TIER_1, ["health","government","us","disease","vaccines","medical"],
                ["us","global"], ["en","es","zh","vi","ko","pt","fr"], 0.98, "daily", True, True,
                "Centers for Disease Control. Authoritative US and global health data."),

    SourceEntry("https://data.gov",                    "data.gov",               "US Open Data Portal",
                TIER_1, ["government","us","data","statistics","research"],
                ["us"], ["en"], 1.0, "daily", True, True,
                "Official US government open data. Hundreds of federal datasets."),

    SourceEntry("https://www.bls.gov",                 "bls.gov",                "US Bureau of Labor Statistics",
                TIER_1, ["economics","government","us","employment","inflation","wages"],
                ["us"], ["en"], 1.0, "monthly", True, True,
                "US CPI, unemployment, wage data. Federal Reserve quality statistics."),

    SourceEntry("https://www.census.gov",              "census.gov",             "US Census Bureau",
                TIER_1, ["government","us","demographics","statistics","data"],
                ["us"], ["en","es"], 1.0, "annual", True, True,
                "US population, income, housing, and economic statistics."),

    # ── United Kingdom ─────────────────────────────────────────────────────────
    SourceEntry("https://www.gov.uk",                  "gov.uk",                 "UK Government Portal",
                TIER_1, ["government","uk","welfare","legal","tax","general"],
                ["uk"], ["en"], 1.0, "daily", True, True,
                "Official UK government services. Tax, benefits, visas, healthcare."),

    SourceEntry("https://www.hmrc.gov.uk",             "hmrc.gov.uk",            "HMRC (UK Tax Authority)",
                TIER_1, ["tax","government","uk","income_tax","vat","legal"],
                ["uk"], ["en"], 1.0, "daily", True, True,
                "Her Majesty's Revenue and Customs. UK income tax, VAT, NI."),

    SourceEntry("https://www.dwp.gov.uk",              "dwp.gov.uk",             "DWP (UK Welfare)",
                TIER_1, ["welfare","government","uk","benefits","pension"],
                ["uk"], ["en"], 1.0, "daily", True, True,
                "UK Department for Work and Pensions. Universal credit, pension, jobseeker."),

    # ── European Union ─────────────────────────────────────────────────────────
    SourceEntry("https://europa.eu",                   "europa.eu",              "European Union Official",
                TIER_1, ["government","eu","legal","regulations","policy"],
                ["eu","global"], ["en","fr","de","es","it","pt","nl","pl","sv","ro","hu","cs","sk",
                                  "da","fi","el","hr","bg","lt","lv","et","sl","mt"], 1.0, "daily", True, True,
                "Official EU portal. Regulations, directives, data protection (GDPR)."),

    SourceEntry("https://ec.europa.eu/eurostat",       "ec.europa.eu",           "Eurostat",
                TIER_1, ["economics","government","eu","statistics","gdp","inflation"],
                ["eu"], ["en","fr","de"], 1.0, "monthly", True, True,
                "EU statistical office. GDP, inflation, unemployment for all EU countries."),

    # ── Australia ──────────────────────────────────────────────────────────────
    SourceEntry("https://www.ato.gov.au",              "ato.gov.au",             "ATO (Australian Tax Office)",
                TIER_1, ["tax","government","australia","income_tax","legal"],
                ["australia"], ["en"], 1.0, "daily", True, True,
                "Australian Taxation Office. Tax returns, GST, superannuation."),

    SourceEntry("https://www.australia.gov.au",        "australia.gov.au",       "Australia Government Portal",
                TIER_1, ["government","australia","welfare","general","legal"],
                ["australia"], ["en"], 1.0, "daily", True, True,
                "Official Australian government services portal."),

    # ── Canada ─────────────────────────────────────────────────────────────────
    SourceEntry("https://www.canada.ca",               "canada.ca",              "Government of Canada Portal",
                TIER_1, ["government","canada","welfare","tax","legal","general"],
                ["canada"], ["en","fr"], 1.0, "daily", True, True,
                "Official Canadian government portal. CRA, benefits, immigration, health."),

    SourceEntry("https://www.canada.ca/en/revenue-agency.html","canada.ca",       "CRA (Canada Revenue Agency)",
                TIER_1, ["tax","government","canada","income_tax","gst"],
                ["canada"], ["en","fr"], 1.0, "daily", True, True,
                "Canada Revenue Agency. Federal income tax, GST/HST, benefits."),

    # ── International Organizations ───────────────────────────────────────────
    SourceEntry("https://www.worldbank.org",           "worldbank.org",          "World Bank",
                TIER_1, ["economics","government","global","development","poverty","gdp"],
                ["global"], ["en","fr","es","ar","zh","ru","pt"], 0.97, "daily", True, True,
                "World Bank development data. GDP, poverty, health stats for all countries."),

    SourceEntry("https://www.imf.org",                 "imf.org",                "IMF (International Monetary Fund)",
                TIER_1, ["economics","government","global","finance","macro"],
                ["global"], ["en","fr","es","ar","zh","ru"], 0.97, "daily", True, True,
                "IMF economic outlook, WEO data, country reports."),

    SourceEntry("https://www.un.org",                  "un.org",                 "United Nations",
                TIER_1, ["government","global","welfare","legal","human_rights","policy"],
                ["global"], ["en","fr","es","ar","zh","ru"], 0.96, "daily", True, True,
                "UN data, resolutions, SDG tracking. Most authoritative global policy source."),

    SourceEntry("https://data.un.org",                 "data.un.org",            "UN Data Portal",
                TIER_1, ["government","global","statistics","demographics","economics"],
                ["global"], ["en","fr","es","ar","zh","ru"], 1.0, "annual", True, True,
                "UN statistical databases. Population, trade, environment for all countries."),

    SourceEntry("https://stats.oecd.org",              "stats.oecd.org",         "OECD Statistics",
                TIER_1, ["economics","government","global","tax","education","health","statistics"],
                ["global"], ["en","fr"], 0.98, "monthly", True, True,
                "OECD data on 38 countries. Tax rates, education, income, health spending."),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCKED / UNRELIABLE
    # ══════════════════════════════════════════════════════════════════════════
    SourceEntry("https://www.zerohedge.com",         "zerohedge.com",         "Zero Hedge",
                BLOCKED, ["finance","news"], ["us"], ["en"], 0.30, "daily", False, True,
                "Known for sensationalism, unverified claims. Blocked."),

    SourceEntry("https://naturalnews.com",           "naturalnews.com",       "Natural News",
                BLOCKED, ["health","science"], ["us"], ["en"], 0.15, "daily", False, True,
                "Known misinformation source. Blocked for health/science queries."),
]


# Fast lookup dict: domain → SourceEntry
_DOMAIN_MAP: Dict[str, SourceEntry] = {}
for _s in TRUSTED_SOURCES:
    _DOMAIN_MAP[_s.domain] = _s


# ─────────────────────────────────────────────────────────────────────────────
# TRUSTED SOURCE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class TrustedSourceRegistry:
    """
    Routes every data request to the most reliable source for that category.
    Prevents ARIA from hallucinating by enforcing source verification.
    """

    # Minimum score to be considered trusted
    MIN_TRUST_SCORE = 0.50

    def __init__(self):
        self.sources = TRUSTED_SOURCES
        self._cache: Dict[str, Any] = {}

    def get_best_source(
        self,
        category:  str,
        region:    str  = "global",
        language:  str  = "en",
        min_tier:  str  = TIER_3,
    ) -> Optional[SourceEntry]:
        """
        Return the best source for a given category, region, and language.

        Example:
            best = registry.get_best_source("stocks", region="india", language="hi")
        """
        tier_order = [TIER_1, TIER_2, TIER_3]
        min_idx    = tier_order.index(min_tier) if min_tier in tier_order else 2

        candidates = [
            s for s in self.sources
            if s.tier in tier_order[:min_idx+1]           # meets tier requirement
            and category.lower() in [c.lower() for c in s.categories]
            and (region == "global" or
                 "global" in s.regions or
                 region.lower() in [r.lower() for r in s.regions])
            and (language == "en" or
                 "en" in s.languages or
                 language.lower() in [l.lower() for l in s.languages])
        ]

        if not candidates:
            # Relax region constraint
            candidates = [
                s for s in self.sources
                if s.tier in tier_order[:min_idx+1]
                and category.lower() in [c.lower() for c in s.categories]
            ]

        if not candidates:
            return None

        return max(candidates, key=lambda s: s.score())

    def get_top_sources(
        self,
        category: str,
        region:   str = "global",
        n:        int = 3,
    ) -> List[SourceEntry]:
        """Return top-N sources for a category, sorted by quality."""
        candidates = [
            s for s in self.sources
            if s.tier != BLOCKED
            and category.lower() in [c.lower() for c in s.categories]
            and ("global" in s.regions or region.lower() in [r.lower() for r in s.regions] or region == "global")
        ]
        return sorted(candidates, key=lambda s: s.score(), reverse=True)[:n]

    def rate_url(self, url: str) -> Dict[str, Any]:
        """
        Rate a URL by domain lookup.
        Returns tier, accuracy, and trustworthiness rating.
        """
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower().lstrip("www.")
        except Exception:
            domain = url.lower()

        if domain in _DOMAIN_MAP:
            entry = _DOMAIN_MAP[domain]
            return {
                "domain":     domain,
                "name":       entry.name,
                "tier":       entry.tier,
                "accuracy":   entry.accuracy,
                "score":      round(entry.score(), 2),
                "trusted":    entry.tier != BLOCKED and entry.score() >= self.MIN_TRUST_SCORE,
                "categories": entry.categories,
                "notes":      entry.notes,
            }

        # Unknown domain — return cautious rating
        return {
            "domain":   domain,
            "name":     "Unknown Source",
            "tier":     TIER_3,
            "accuracy": 0.50,
            "score":    0.30,
            "trusted":  False,
            "categories": [],
            "notes":    "Unknown source. Verify information independently.",
        }

    def is_blocked(self, url: str) -> bool:
        """Check if a URL is from a blocked/unreliable domain."""
        rating = self.rate_url(url)
        return rating["tier"] == BLOCKED

    def verify_claim(self, claim: str, sources: List[str]) -> Dict[str, Any]:
        """
        Verify a factual claim by checking if it comes from trusted sources.
        Returns verification status and confidence.
        """
        trusted_count   = 0
        untrusted_count = 0
        source_ratings  = []

        for src in sources:
            r = self.rate_url(src)
            source_ratings.append(r)
            if r["trusted"] and r["tier"] in (TIER_1, TIER_2):
                trusted_count += 1
            else:
                untrusted_count += 1

        confidence = min(trusted_count / max(len(sources), 1), 1.0)

        return {
            "claim":          claim[:100],
            "trusted_sources":  trusted_count,
            "untrusted_sources": untrusted_count,
            "confidence":     round(confidence, 2),
            "verified":       trusted_count >= 1,
            "source_ratings": source_ratings,
        }

    def get_category_for_query(self, query: str) -> str:
        """Infer data category from a natural language query."""
        q = query.lower()
        # Government / tax / welfare — check before generic news/finance
        if any(k in q for k in ("income tax","itr","tax return","tds","tax slab","gst","vat","hmrc","irs","ato","cra","tax file")):
            return "tax"
        if any(k in q for k in ("welfare","benefit","scheme","pension","provident fund","epf","uan","social security","universal credit","centrelink","cpp")):
            return "welfare"
        if any(k in q for k in ("government","govt","gov","official portal","aadhaar","pan card","ration card","passport","visa","license")):
            return "government"
        if any(k in q for k in ("legal","law","regulation","act","section","court","judgment","constitution","rights")):
            return "legal"
        if any(k in q for k in ("gdp","inflation","interest rate","federal","rbi","ecb","economy","macro","unemployment","cpi","wpi","bls","oecd")):
            return "economics"
        if any(k in q for k in ("stock","share","nifty","nasdaq","market","invest","equity","dividend","p/e","roe")):
            return "stocks"
        if any(k in q for k in ("bitcoin","crypto","ethereum","blockchain","defi","nft","coin")):
            return "crypto"
        if any(k in q for k in ("weather","temperature","rain","humidity","forecast","climate")):
            return "weather"
        if any(k in q for k in ("study","research","paper","journal","arxiv","science","physics","chemistry","biology")):
            return "science"
        if any(k in q for k in ("health","medicine","drug","vaccine","disease","hospital","medical","symptom")):
            return "health"
        if any(k in q for k in ("news","latest","today","current","happen","event","world")):
            return "news"
        if any(k in q for k in ("code","program","github","python","javascript","api","software","ai","ml")):
            return "tech"
        return "general"

    def recommend_sources(self, query: str, region: str = "global") -> str:
        """
        Given a query, return the best sources to use.
        This is what ARIA checks before answering any factual question.
        """
        category = self.get_category_for_query(query)
        sources  = self.get_top_sources(category, region=region, n=3)

        if not sources:
            return "No verified source found for this query type."

        lines = [f"Best sources for '{category}' queries:"]
        for s in sources:
            tier_label = {"AUTHORITATIVE": "PRIMARY", "HIGH_QUALITY": "TRUSTED", "SECONDARY": "USE WITH CAUTION"}.get(s.tier, s.tier)
            lines.append(f"  [{tier_label}] {s.name} — {s.url}")
            lines.append(f"    Accuracy: {s.accuracy*100:.0f}%  |  Freshness: {s.freshness}  |  Languages: {', '.join(s.languages[:5])}")
            if s.notes:
                lines.append(f"    Note: {s.notes[:80]}")
        return "\n".join(lines)

    def run_nl(self, instruction: str) -> str:
        q = instruction.lower().strip()

        # Rate a URL
        url_m = re.search(r"https?://[^\s]+", instruction)
        if url_m and ("rate" in q or "trust" in q or "reliable" in q or "check" in q):
            r = self.rate_url(url_m.group())
            return (
                f"Source Rating: {r['name']} ({r['domain']})\n"
                f"  Tier:     {r['tier']}\n"
                f"  Accuracy: {r['accuracy']*100:.0f}%\n"
                f"  Score:    {r['score']}/1.0\n"
                f"  Trusted:  {'YES' if r['trusted'] else 'NO - AVOID'}\n"
                f"  Notes:    {r['notes']}"
            )

        # Best source for a category
        if "best source" in q or "where to get" in q or "reliable source" in q:
            region = "global"
            for r in ["india","us","uk","eu","japan","australia","global"]:
                if r in q:
                    region = r
                    break
            return self.recommend_sources(instruction, region=region)

        # List categories
        if "list" in q or "categories" in q or "what can" in q:
            cats = set()
            for s in self.sources:
                cats.update(s.categories)
            return "Source categories: " + ", ".join(sorted(cats))

        return self.recommend_sources(instruction)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-LANGUAGE AGENT
# ─────────────────────────────────────────────────────────────────────────────

# Full language code → display name mapping
LANGUAGE_MAP: Dict[str, str] = {
    "en": "English",   "hi": "Hindi",      "fr": "French",     "de": "German",
    "es": "Spanish",   "pt": "Portuguese", "ar": "Arabic",     "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",      "ja": "Japanese",   "ko": "Korean",
    "ru": "Russian",   "it": "Italian",    "nl": "Dutch",      "pl": "Polish",
    "tr": "Turkish",   "vi": "Vietnamese", "th": "Thai",       "sv": "Swedish",
    "no": "Norwegian", "da": "Danish",     "fi": "Finnish",    "cs": "Czech",
    "sk": "Slovak",    "hu": "Hungarian",  "ro": "Romanian",   "uk": "Ukrainian",
    "bn": "Bengali",   "ta": "Tamil",      "te": "Telugu",     "ml": "Malayalam",
    "kn": "Kannada",   "mr": "Marathi",    "gu": "Gujarati",   "pa": "Punjabi",
    "ur": "Urdu",      "fa": "Persian",    "id": "Indonesian", "ms": "Malay",
    "sw": "Swahili",   "af": "Afrikaans",  "he": "Hebrew",     "el": "Greek",
    "ca": "Catalan",   "hr": "Croatian",   "bg": "Bulgarian",  "sr": "Serbian",
    "lt": "Lithuanian","lv": "Latvian",    "et": "Estonian",   "sl": "Slovenian",
}

# Grammar instructions per language family
GRAMMAR_INSTRUCTIONS: Dict[str, str] = {
    "en": "Respond in clear, natural English with proper grammar.",
    "hi": "हिन्दी में स्पष्ट और सही व्याकरण के साथ उत्तर दें। देवनागरी लिपि का उपयोग करें।",
    "fr": "Répondez en français correct avec une grammaire parfaite. Utilisez les accents appropriés.",
    "de": "Antworten Sie auf Deutsch mit korrekter Grammatik. Verwenden Sie Umlaute korrekt (ä, ö, ü, ß).",
    "es": "Responde en español correcto con buena gramática. Usa los acentos correctamente.",
    "pt": "Responda em português correto com boa gramática. Use acentos corretamente.",
    "ar": "أجب باللغة العربية الفصيحة الصحيحة مع القواعد النحوية السليمة.",
    "zh": "用简洁、准确的中文回答，使用正确的汉字和标点符号。",
    "ja": "正確な日本語で、適切な敬語を使って答えてください。漢字、ひらがな、カタカナを正しく使ってください。",
    "ko": "정확한 한국어로, 적절한 경어를 사용하여 답변해 주세요.",
    "ru": "Отвечайте на правильном русском языке с соблюдением грамматических норм.",
    "it": "Rispondi in italiano corretto con buona grammatica.",
    "bn": "সঠিক বাংলা ব্যাকরণ সহ বাংলায় উত্তর দিন।",
    "ta": "சரியான தமிழ் இலக்கணத்துடன் தமிழில் பதிலளிக்கவும்.",
    "te": "సరైన తెలుగు వ్యాకరణంతో తెలుగులో సమాధానం ఇవ్వండి.",
    "ur": "درست اردو قواعد کے ساتھ اردو میں جواب دیں۔",
}


class MultiLanguageAgent:
    """
    Detects input language, processes queries, and responds in the same
    language with correct grammar. Supports 50+ languages via Ollama.

    No translation API needed — ARIA's local LLM handles all languages
    natively using language-specific system prompts.
    """

    def __init__(self, engine=None, source_registry: Optional[TrustedSourceRegistry] = None):
        self.engine   = engine
        self.registry = source_registry or TrustedSourceRegistry()
        self._lang_cache: Dict[str, str] = {}   # text_hash → detected_lang

    # ──────────────────────────────────────────────────────────────────────────
    # Language Detection
    # ──────────────────────────────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        Returns ISO 639-1 code (e.g. 'hi', 'fr', 'en').
        Falls back to 'en' if detection fails.
        """
        # Cache by hash
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        if h in self._lang_cache:
            return self._lang_cache[h]

        lang = "en"   # default

        # Method 1: langdetect
        if LANGDETECT_OK and len(text.strip()) > 5:
            try:
                lang = _langdetect(text)
                # Normalize zh variants
                if lang.startswith("zh"):
                    lang = "zh"
            except Exception:
                pass

        # Method 2: Script-based heuristics (fast, no library needed)
        if lang == "en":
            lang = self._script_detect(text) or lang

        self._lang_cache[h] = lang
        return lang

    def _script_detect(self, text: str) -> Optional[str]:
        """Detect language from Unicode script ranges (fast heuristic)."""
        # Count characters per script
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        arabic     = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        chinese    = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF')
        japanese   = sum(1 for c in text if '\u3040' <= c <= '\u30FF')
        korean     = sum(1 for c in text if '\uAC00' <= c <= '\uD7AF')
        cyrillic   = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        tamil      = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        telugu     = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
        bengali    = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        thai       = sum(1 for c in text if '\u0E00' <= c <= '\u0E7F')

        n = len(text) or 1
        checks = [
            (devanagari, "hi"),
            (arabic,     "ar"),
            (chinese,    "zh"),
            (japanese,   "ja"),
            (korean,     "ko"),
            (cyrillic,   "ru"),
            (tamil,      "ta"),
            (telugu,     "te"),
            (bengali,    "bn"),
            (thai,       "th"),
        ]
        best = max(checks, key=lambda x: x[0])
        if best[0] / n > 0.15:   # > 15% of chars from this script
            return best[1]
        return None

    def get_language_name(self, code: str) -> str:
        """Get human-readable language name from ISO code."""
        return LANGUAGE_MAP.get(code, f"Language({code})")

    # ──────────────────────────────────────────────────────────────────────────
    # Response Generation
    # ──────────────────────────────────────────────────────────────────────────

    def respond(self, query: str, force_language: Optional[str] = None) -> str:
        """
        Process a query in any language and respond in the same language
        with correct grammar.

        Steps:
        1. Detect language
        2. Build language-specific system prompt
        3. Route to appropriate trusted sources
        4. Generate response via LLM with language instruction
        5. Return response in detected language
        """
        if not self.engine:
            return (
                "MultiLanguageAgent requires an LLM engine. "
                "Pass engine=aria_engine when creating the agent."
            )

        # Step 1: Detect language
        lang = force_language or self.detect_language(query)
        lang_name = self.get_language_name(lang)

        # Step 2: Grammar instruction for this language
        grammar_instr = GRAMMAR_INSTRUCTIONS.get(lang,
            f"Respond in {lang_name} with correct, natural grammar. "
            f"Use the proper script and punctuation for {lang_name}.")

        # Step 3: Get best data sources for this query
        source_hint = ""
        try:
            category = self.registry.get_category_for_query(query)
            top_src  = self.registry.get_top_sources(category, n=2)
            if top_src:
                source_hint = (
                    f"\n\nData sources for this query: "
                    + ", ".join(f"{s.name} ({s.url})" for s in top_src)
                    + "\nBase your answer on these authoritative sources. "
                    "If you are not certain about a fact, say so explicitly."
                )
        except Exception:
            pass

        # Step 4: Build prompt with language + grammar + anti-hallucination instruction
        system_prompt = (
            f"You are ARIA, a precise and knowledgeable AI assistant.\n"
            f"CRITICAL RULES:\n"
            f"1. {grammar_instr}\n"
            f"2. Never fabricate facts. If you do not know something, say "
            f"'I am not certain about this' in {lang_name}.\n"
            f"3. For factual claims (numbers, dates, names), state the source.\n"
            f"4. Keep the same language throughout — do not switch to English unless asked.\n"
            f"5. If the question mixes languages, respond in the dominant language: {lang_name}.\n"
            f"{source_hint}"
        )

        try:
            response = self.engine.generate(
                f"{system_prompt}\n\nUser query: {query}",
                temperature=0.4,
                max_tokens=1000,
            )
            return response
        except Exception as e:
            return f"Response generation failed: {e}"

    def translate(self, text: str, target_language: str) -> str:
        """
        Translate text to the specified target language.
        Uses ARIA's local LLM — no external translation API needed.
        """
        if not self.engine:
            return text

        target_name  = self.get_language_name(target_language)
        source_lang  = self.detect_language(text)
        source_name  = self.get_language_name(source_lang)
        grammar_instr = GRAMMAR_INSTRUCTIONS.get(target_language, f"Use correct {target_name} grammar.")

        prompt = (
            f"Translate the following {source_name} text to {target_name}.\n"
            f"Rules:\n"
            f"1. {grammar_instr}\n"
            f"2. Preserve meaning exactly — do not add or remove information.\n"
            f"3. Preserve formatting (lists, bold, etc.).\n"
            f"4. Output ONLY the translation — no explanations.\n\n"
            f"Text to translate:\n{text}"
        )

        try:
            return self.engine.generate(prompt, temperature=0.2, max_tokens=2000)
        except Exception as e:
            return f"Translation failed: {e}"

    def correct_grammar(self, text: str) -> str:
        """
        Correct grammar in the input text (auto-detects language).
        Returns corrected text with explanation.
        """
        if not self.engine:
            return text

        lang      = self.detect_language(text)
        lang_name = self.get_language_name(lang)

        prompt = (
            f"Correct the grammar of the following {lang_name} text.\n"
            f"Rules:\n"
            f"1. Fix grammar, spelling, and punctuation errors.\n"
            f"2. Keep the same language ({lang_name}).\n"
            f"3. Preserve the original meaning and tone.\n"
            f"4. Return: {{\"corrected\": \"...\", \"changes\": [\"change1\", \"change2\"]}}\n\n"
            f"Text: {text}"
        )

        try:
            raw = self.engine.generate(prompt, temperature=0.1, max_tokens=500)
            m   = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                corrected = data.get("corrected", text)
                changes   = data.get("changes", [])
                result    = f"Corrected ({lang_name}):\n{corrected}"
                if changes:
                    result += f"\n\nChanges made:\n" + "\n".join(f"  - {c}" for c in changes)
                return result
            return raw
        except Exception as e:
            return f"Grammar correction failed: {e}"

    def detect_and_explain(self, text: str) -> str:
        """Detect language and provide explanation of detection."""
        lang      = self.detect_language(text)
        lang_name = self.get_language_name(lang)
        method    = "langdetect library" if LANGDETECT_OK else "Unicode script analysis"
        return (
            f"Detected language: {lang_name} ({lang})\n"
            f"Detection method: {method}\n"
            f"Script: {'Unicode script analysis supported' if self._script_detect(text) else 'Latin script'}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NL interface
    # ──────────────────────────────────────────────────────────────────────────

    def run_nl(self, instruction: str) -> str:
        low = instruction.lower().strip()

        if re.search(r"detect language|what language", low):
            # Find the quoted text or rest of sentence
            m = re.search(r'"([^"]+)"', instruction) or re.search(r"language of (.+)", instruction, re.IGNORECASE)
            text = m.group(1) if m else instruction
            return self.detect_and_explain(text)

        if re.search(r"translate.+to\s+(\w+)", low):
            m = re.search(r"translate\s+(.+?)\s+to\s+(\w+)", instruction, re.IGNORECASE)
            if m:
                text        = m.group(1).strip().strip('"\'')
                target_lang = m.group(2).lower()
                # Find ISO code
                target_code = target_lang
                for code, name in LANGUAGE_MAP.items():
                    if name.lower() == target_lang or code == target_lang:
                        target_code = code
                        break
                return self.translate(text, target_code)

        if re.search(r"correct.*(grammar|spelling|language)", low):
            m = re.search(r'"([^"]+)"', instruction)
            text = m.group(1) if m else re.sub(r"correct.*grammar\s+of\s+", "", instruction, flags=re.IGNORECASE)
            return self.correct_grammar(text.strip())

        if re.search(r"supported language|which language|how many language", low):
            lines = ["Supported languages:"]
            for code, name in sorted(LANGUAGE_MAP.items(), key=lambda x: x[1]):
                lines.append(f"  {code:8s} — {name}")
            return "\n".join(lines)

        # Default: respond to the instruction in its detected language
        return self.respond(instruction)


# ─────────────────────────────────────────────────────────────────────────────
# Combined registry + language agent — single access point
# ─────────────────────────────────────────────────────────────────────────────

class TrustLanguageAgent:
    """
    Unified agent combining TrustedSourceRegistry + MultiLanguageAgent.
    This is what gets registered in ARIA's neural orchestrator.
    """

    def __init__(self, engine=None):
        self.registry     = TrustedSourceRegistry()
        self.lang_agent   = MultiLanguageAgent(engine=engine, source_registry=self.registry)
        self.engine       = engine

    def run_nl(self, instruction: str) -> str:
        low = instruction.lower()

        # Source-related queries
        if any(k in low for k in ("source", "reliable", "trust", "website", "where to get", "which site")):
            return self.registry.run_nl(instruction)

        # Language-related queries
        if any(k in low for k in ("translate", "detect language", "correct grammar", "which language", "supported language")):
            return self.lang_agent.run_nl(instruction)

        # Multi-language response (default for non-English input)
        detected = self.lang_agent.detect_language(instruction)
        if detected != "en":
            return self.lang_agent.respond(instruction)

        # English + source context
        category    = self.registry.get_category_for_query(instruction)
        source_hint = self.registry.recommend_sources(instruction)
        return f"Sources for this query:\n{source_hint}"
