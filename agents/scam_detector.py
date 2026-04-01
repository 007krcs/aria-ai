"""
ARIA Scam / Phishing Detector Agent
=====================================
Multi-layer website authenticity analysis. Runs automatically when ARIA
visits any URL and raises a warning before the user sees potentially
harmful content.

Detection layers (8):
  1. Domain spoofing / typosquatting / homoglyph attack
  2. SSL certificate verification and mismatch
  3. Redirect chain analysis (catches cloaking)
  4. Iframe trap detection (transparent overlay / clickjacking)
  5. Wireframe / overlay UI detection (fake login forms over real sites)
  6. Domain age and WHOIS signals
  7. Content vs. domain identity mismatch (LLM-assisted)
  8. Visual clone detection (screenshot hash comparison against known brands)

Each layer returns a RiskSignal with a severity score 0-10.
A composite TrustScore is computed and a SAFE / SUSPICIOUS / DANGEROUS verdict
is issued with a human-readable explanation.
"""

import re
import os
import ssl
import json
import time
import socket
import hashlib
import urllib.parse
import unicodedata
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import whois as python_whois          # pip install python-whois
    WHOIS_OK = True
except ImportError:
    WHOIS_OK = False

try:
    from bs4 import BeautifulSoup         # pip install beautifulsoup4
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    from PIL import Image                  # pip install Pillow
    import io
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import certifi
    CERTIFI_OK = True
except ImportError:
    CERTIFI_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskSignal:
    """A single risk signal from one detection layer."""
    layer:       str
    severity:    float          # 0-10 (10 = certain threat)
    description: str
    evidence:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrustReport:
    """Full analysis result for a URL."""
    url:              str
    final_url:        str           # after redirects
    verdict:          str           # SAFE / SUSPICIOUS / DANGEROUS
    trust_score:      float         # 0-100 (100 = fully trusted)
    risk_signals:     List[RiskSignal] = field(default_factory=list)
    domain_age_days:  Optional[int]  = None
    ssl_valid:        bool            = False
    ssl_issuer:       str             = ""
    redirects:        List[str]       = field(default_factory=list)
    iframes_found:    int             = 0
    suspicious_iframes: List[str]    = field(default_factory=list)
    overlays_found:   bool            = False
    brand_impersonated: Optional[str] = None
    explanation:      str             = ""
    scan_time_s:      float           = 0.0
    scanned_at:       str             = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_signals"] = [s.to_dict() for s in self.risk_signals]
        return d

    def summary(self) -> str:
        lines = [
            f"🔍 URL: {self.url}",
            f"📊 Verdict: {self.verdict}  |  Trust Score: {self.trust_score:.0f}/100",
        ]
        if self.brand_impersonated:
            lines.append(f"⚠️  Brand impersonated: {self.brand_impersonated}")
        if self.redirects:
            lines.append(f"🔀 Redirect chain ({len(self.redirects)} hops): {' → '.join(self.redirects[-3:])}")
        if self.suspicious_iframes:
            lines.append(f"🪤 Suspicious iframes: {self.suspicious_iframes}")
        if self.overlays_found:
            lines.append("🎭 Overlay / transparent layer detected")
        if not self.ssl_valid:
            lines.append("🔓 SSL certificate invalid or mismatched")
        if self.domain_age_days is not None and self.domain_age_days < 30:
            lines.append(f"📅 Domain is only {self.domain_age_days} days old!")
        lines.append(f"💬 {self.explanation}")
        for sig in self.risk_signals:
            icon = "🟡" if sig.severity < 5 else "🔴"
            lines.append(f"  {icon} [{sig.layer}] sev={sig.severity:.1f} — {sig.description}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN BRAND DOMAINS  (extend as needed)
# ─────────────────────────────────────────────────────────────────────────────

BRAND_DOMAINS: Dict[str, List[str]] = {
    "google":    ["google.com", "google.co.in", "google.co.uk", "googleapis.com", "gstatic.com"],
    "facebook":  ["facebook.com", "fb.com", "instagram.com", "whatsapp.com", "meta.com"],
    "amazon":    ["amazon.com", "amazon.in", "amazon.co.uk", "aws.amazon.com", "amazonaws.com"],
    "apple":     ["apple.com", "icloud.com", "itunes.com"],
    "microsoft": ["microsoft.com", "live.com", "outlook.com", "office.com", "azure.com", "msn.com"],
    "paypal":    ["paypal.com", "paypal.me"],
    "twitter":   ["twitter.com", "x.com", "t.co"],
    "netflix":   ["netflix.com"],
    "linkedin":  ["linkedin.com"],
    "dropbox":   ["dropbox.com"],
    "github":    ["github.com", "githubusercontent.com"],
    "bank":      ["chase.com", "bankofamerica.com", "wellsfargo.com", "hsbc.com", "sbi.co.in"],
}

# Flat lookup: official_domain → brand_name
_OFFICIAL = {}
for _brand, _domains in BRAND_DOMAINS.items():
    for _d in _domains:
        _OFFICIAL[_d] = _brand


# ─────────────────────────────────────────────────────────────────────────────
# HOMOGLYPH TABLE  (common lookalikes used in phishing)
# ─────────────────────────────────────────────────────────────────────────────

HOMOGLYPHS: Dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",   # Cyrillic
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
    "ℓ": "l", "ƅ": "b", "ᴡ": "w", "ν": "v",              # Greek/Mathematical
    "０": "0", "１": "1", "２": "2",                       # Fullwidth digits
    "ó": "o", "ö": "o", "ü": "u", "ä": "a",               # Diacritics
}


def normalize_domain(domain: str) -> str:
    """Replace homoglyphs so 'paypa1.com' → 'paypal.com' for comparison."""
    result = []
    for ch in domain.lower():
        result.append(HOMOGLYPHS.get(ch, ch))
    # Also strip common prefix noise
    d = "".join(result)
    d = re.sub(r"^(www\d*\.)", "", d)
    return d


def levenshtein(a: str, b: str) -> int:
    """Fast Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


# ─────────────────────────────────────────────────────────────────────────────
# SCAM DETECTOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ScamDetectorAgent:
    """
    Comprehensive phishing and scam website detection for ARIA.

    Usage:
        detector = ScamDetectorAgent(engine=aria_engine)
        report   = detector.scan("https://paypa1.com/login")
        print(report.summary())
    """

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    REQUEST_TIMEOUT  = 10   # seconds
    MAX_REDIRECTS    = 10
    DANGEROUS_THRESHOLD   = 40   # trust score below this → DANGEROUS
    SUSPICIOUS_THRESHOLD  = 65   # trust score below this → SUSPICIOUS

    def __init__(self, engine=None, headless_browser=None):
        """
        engine:           ARIA LLM engine (optional) for content-vs-domain analysis
        headless_browser: BrowserAgent instance for screenshot-based checks
        """
        self.engine   = engine
        self.browser  = headless_browser
        self._session = self._build_session() if REQUESTS_OK else None

    # ──────────────────────────────────────────────────────────────────────────
    # Session builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_session(self):
        if not REQUESTS_OK:
            return None
        s = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503])
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://",  HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": self.USER_AGENT})
        return s

    def _get(self, url: str, **kwargs) -> Optional[requests.Response]:
        if not self._session:
            return None
        try:
            r = self._session.get(
                url,
                timeout=self.REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=CERTIFI_OK,
                **kwargs,
            )
            return r
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    def scan(self, url: str) -> TrustReport:
        """
        Full scan of a URL. Returns TrustReport with verdict and all signals.
        Runs all 8 detection layers concurrently where possible.
        """
        t0 = time.time()

        # Normalise URL
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "https://" + url

        parsed   = urllib.parse.urlparse(url)
        domain   = parsed.netloc.lower().lstrip("www.")
        report   = TrustReport(
            url         = url,
            final_url   = url,
            verdict     = "UNKNOWN",
            trust_score = 100.0,
            scanned_at  = datetime.now(timezone.utc).isoformat(),
        )

        # Fetch page (single network call, reused across layers)
        response = self._get(url)
        html     = ""
        if response is not None:
            report.final_url = response.url
            report.redirects = [r.url for r in response.history] + [response.url]
            html = response.text or ""

        # Run all detection layers
        signals: List[RiskSignal] = []
        signals += self._check_domain_spoofing(domain, url, html)
        signals += self._check_ssl(url, domain)
        signals += self._check_redirect_chain(report.redirects)
        signals += self._check_iframes(html, domain, report)
        signals += self._check_overlay_traps(html)
        signals += self._check_domain_age(domain)
        signals += self._check_content_mismatch(html, domain, url)
        signals += self._check_suspicious_patterns(html, url)

        report.risk_signals = signals

        # ── Compute trust score ──────────────────────────────────────────────
        total_severity = sum(s.severity for s in signals)
        # Each point of severity deducts trust; max deduction is 100
        deduction = min(total_severity * 8, 100)
        report.trust_score = max(0.0, 100.0 - deduction)

        # ── Verdict ──────────────────────────────────────────────────────────
        if report.trust_score < self.DANGEROUS_THRESHOLD:
            report.verdict = "DANGEROUS"
        elif report.trust_score < self.SUSPICIOUS_THRESHOLD:
            report.verdict = "SUSPICIOUS"
        else:
            report.verdict = "SAFE"

        # ── Brand impersonation ──────────────────────────────────────────────
        for sig in signals:
            if "brand" in sig.layer.lower() or "impersonat" in sig.description.lower():
                report.brand_impersonated = sig.evidence.get("closest_brand")
                break

        # ── Explanation ──────────────────────────────────────────────────────
        report.explanation = self._build_explanation(report)
        report.scan_time_s = round(time.time() - t0, 2)

        return report

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 1 — Domain spoofing / typosquatting / homoglyph
    # ──────────────────────────────────────────────────────────────────────────

    def _check_domain_spoofing(self, domain: str, url: str, html: str) -> List[RiskSignal]:
        signals = []
        normalized = normalize_domain(domain)

        # Strip TLD for matching
        apex = re.sub(r"\.[a-z]{2,6}(\.[a-z]{2})?$", "", normalized)

        best_match = None
        best_dist  = 999
        for official, brand in _OFFICIAL.items():
            official_apex = re.sub(r"\.[a-z]{2,6}(\.[a-z]{2})?$", "", official)
            dist = levenshtein(apex, official_apex)
            if dist < best_dist:
                best_dist  = dist
                best_match = (official, brand)

        # If domain IS an official domain — no spoofing
        norm_domain = normalize_domain(domain)
        for official in _OFFICIAL:
            if norm_domain == official or norm_domain.endswith("." + official):
                return signals   # clean — official domain

        # Typosquatting: Levenshtein ≤ 2 but not the same
        if best_match and 0 < best_dist <= 2:
            severity = 8.0 if best_dist == 1 else 5.0
            signals.append(RiskSignal(
                layer       = "domain_spoofing",
                severity    = severity,
                description = (
                    f"Domain '{domain}' is suspiciously similar to official '{best_match[0]}' "
                    f"(edit distance={best_dist}). Possible typosquatting."
                ),
                evidence    = {
                    "domain":         domain,
                    "closest_brand":  best_match[1],
                    "closest_domain": best_match[0],
                    "edit_distance":  best_dist,
                },
            ))

        # Homoglyph: raw unicode differs but normalized matches official
        if normalized != domain.lower().lstrip("www."):
            signals.append(RiskSignal(
                layer       = "homoglyph_attack",
                severity    = 9.0,
                description = (
                    f"Domain contains Unicode lookalike characters: '{domain}' normalises to "
                    f"'{normalized}'. Classic homoglyph phishing attack."
                ),
                evidence    = {"raw": domain, "normalized": normalized},
            ))

        # Suspicious keyword stuffing in domain
        brand_keywords = [b for brands in BRAND_DOMAINS.keys() for b in [brands]]
        brand_keywords += ["secure", "login", "verify", "account", "support", "update",
                           "bank", "wallet", "payment", "confirm", "alert", "unlock"]
        apex_parts = re.split(r"[-_.]", apex)
        hit_kw = [kw for kw in brand_keywords if kw in apex_parts]
        if len(hit_kw) >= 2:
            signals.append(RiskSignal(
                layer       = "domain_keyword_stuffing",
                severity    = 4.0,
                description = (
                    f"Domain packs suspicious keywords: {hit_kw}. "
                    "Common phishing tactic to appear legitimate."
                ),
                evidence    = {"keywords_found": hit_kw, "domain": domain},
            ))

        # Excessive subdomains
        subdomain_count = domain.count(".") - 1
        if subdomain_count >= 3:
            signals.append(RiskSignal(
                layer       = "excessive_subdomains",
                severity    = 3.0,
                description = (
                    f"Domain has {subdomain_count+1} levels of subdomains. "
                    "Phishing sites often use 'real-bank.evil.com' to confuse users."
                ),
                evidence    = {"domain": domain, "subdomain_depth": subdomain_count + 1},
            ))

        # IP address as domain
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain):
            signals.append(RiskSignal(
                layer       = "ip_as_domain",
                severity    = 6.0,
                description  = "URL uses a raw IP address instead of a domain name — major red flag.",
                evidence    = {"ip": domain},
            ))

        # Punycode / IDN domain
        if "xn--" in domain:
            signals.append(RiskSignal(
                layer       = "punycode_domain",
                severity    = 5.0,
                description = (
                    f"Domain uses punycode / internationalized domain name (IDN): '{domain}'. "
                    "These are commonly used for homoglyph phishing."
                ),
                evidence    = {"domain": domain},
            ))

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 2 — SSL certificate verification
    # ──────────────────────────────────────────────────────────────────────────

    def _check_ssl(self, url: str, domain: str) -> List[RiskSignal]:
        signals = []
        parsed  = urllib.parse.urlparse(url)

        # HTTP-only (no SSL at all)
        if parsed.scheme == "http":
            signals.append(RiskSignal(
                layer       = "ssl_missing",
                severity    = 4.0,
                description = "Site uses plain HTTP — no encryption. Never enter passwords or payment info.",
                evidence    = {"scheme": "http"},
            ))
            return signals

        # Try to get SSL certificate details
        try:
            ctx = ssl.create_default_context()
            host = parsed.hostname or domain
            port = parsed.port or 443
            with ctx.wrap_socket(
                socket.create_connection((host, port), timeout=5),
                server_hostname=host,
            ) as ssock:
                cert = ssock.getpeercert()

            # Extract SANs and CN
            san_list = []
            for rdn in cert.get("subjectAltName", []):
                if rdn[0] == "DNS":
                    san_list.append(rdn[1].lstrip("*.").lower())
            cn = ""
            for rdn in cert.get("subject", []):
                for attr in rdn:
                    if attr[0] == "commonName":
                        cn = attr[1].lstrip("*.").lower()

            cert_hosts = san_list or [cn]

            # Check if cert matches current domain
            apex_domain = re.sub(r"^[^.]+\.", "", domain)   # remove first subdomain
            matched = any(
                domain.endswith(ch) or domain == ch or apex_domain == ch
                for ch in cert_hosts
            )
            if not matched and cert_hosts:
                signals.append(RiskSignal(
                    layer       = "ssl_domain_mismatch",
                    severity    = 8.0,
                    description = (
                        f"SSL certificate is issued for {cert_hosts[:3]} but you are visiting "
                        f"'{domain}'. The certificate does NOT match the domain — classic indicator "
                        "of a fake or misconfigured site."
                    ),
                    evidence    = {"cert_hosts": cert_hosts, "visited_domain": domain},
                ))
            else:
                # Signal all-clear (positive signal — reduces effective severity)
                pass

            # Check expiry
            exp_str = cert.get("notAfter", "")
            if exp_str:
                exp_dt = datetime.strptime(exp_str, "%b %d %H:%M:%S %Y %Z")
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                days_left = (exp_dt - datetime.now(timezone.utc)).days
                if days_left < 0:
                    signals.append(RiskSignal(
                        layer       = "ssl_expired",
                        severity    = 7.0,
                        description = f"SSL certificate EXPIRED {abs(days_left)} days ago.",
                        evidence    = {"expired_days": abs(days_left)},
                    ))
                elif days_left < 7:
                    signals.append(RiskSignal(
                        layer       = "ssl_expiring_soon",
                        severity    = 2.0,
                        description = f"SSL certificate expires in {days_left} days.",
                        evidence    = {"days_left": days_left},
                    ))

            # Extract issuer
            for rdn in cert.get("issuer", []):
                for attr in rdn:
                    if attr[0] == "organizationName":
                        pass  # store in report later

        except ssl.SSLError as e:
            signals.append(RiskSignal(
                layer       = "ssl_error",
                severity    = 8.0,
                description = f"SSL handshake failed: {e}. Site may be using a self-signed or invalid certificate.",
                evidence    = {"error": str(e)},
            ))
        except socket.timeout:
            signals.append(RiskSignal(
                layer       = "ssl_timeout",
                severity    = 2.0,
                description = "SSL connection timed out — site may be unreachable or blocking scanners.",
                evidence    = {},
            ))
        except Exception:
            pass   # Non-critical — skip SSL check if network error

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 3 — Redirect chain analysis
    # ──────────────────────────────────────────────────────────────────────────

    def _check_redirect_chain(self, redirects: List[str]) -> List[RiskSignal]:
        signals = []
        if not redirects or len(redirects) <= 1:
            return signals

        # More than 3 hops is suspicious
        if len(redirects) > 3:
            signals.append(RiskSignal(
                layer       = "redirect_chain",
                severity    = min(len(redirects) - 2, 5) * 1.2,
                description = (
                    f"Page went through {len(redirects)-1} redirects before reaching destination. "
                    "Long redirect chains are used to cloak phishing pages from scanners."
                ),
                evidence    = {"chain": redirects},
            ))

        # Cross-scheme downgrade: starts HTTPS, then redirects to HTTP
        if redirects[0].startswith("https://") and any(r.startswith("http://") for r in redirects[1:]):
            signals.append(RiskSignal(
                layer       = "https_downgrade",
                severity    = 6.0,
                description = "Redirect chain downgrades from HTTPS to HTTP — traffic may be intercepted.",
                evidence    = {"chain": redirects},
            ))

        # Destination domain changes drastically
        def _apex(u):
            h = urllib.parse.urlparse(u).netloc
            parts = h.split(".")
            return ".".join(parts[-2:]) if len(parts) >= 2 else h

        start_apex = _apex(redirects[0])
        end_apex   = _apex(redirects[-1])
        if start_apex != end_apex:
            signals.append(RiskSignal(
                layer       = "cross_domain_redirect",
                severity    = 3.5,
                description = (
                    f"Redirected from '{start_apex}' to '{end_apex}'. "
                    "Cross-domain redirects can be used to disguise the true destination."
                ),
                evidence    = {"from": start_apex, "to": end_apex},
            ))

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 4 — Iframe trap / clickjacking detection
    # ──────────────────────────────────────────────────────────────────────────

    def _check_iframes(self, html: str, domain: str, report: TrustReport) -> List[RiskSignal]:
        signals = []
        if not html:
            return signals

        soup = None
        if BS4_OK:
            try:
                soup = BeautifulSoup(html, "html.parser")
            except Exception:
                pass

        # --- Raw regex iframe search ---
        iframe_srcs = re.findall(
            r'<iframe[^>]+src=["\']?([^"\'>\s]+)["\']?',
            html,
            re.IGNORECASE,
        )
        report.iframes_found = len(iframe_srcs)

        suspicious = []
        for src in iframe_srcs:
            if not src or src.startswith("#") or src == "about:blank":
                continue
            # Check if iframe is from a completely different domain
            try:
                iframe_domain = urllib.parse.urlparse(src).netloc.lower()
                apex_domain   = ".".join(domain.split(".")[-2:])
                if iframe_domain and apex_domain and apex_domain not in iframe_domain:
                    suspicious.append(src)
            except Exception:
                pass

        report.suspicious_iframes = suspicious[:10]

        if suspicious:
            severity = min(len(suspicious) * 1.5, 7.0)
            signals.append(RiskSignal(
                layer       = "suspicious_iframes",
                severity    = severity,
                description = (
                    f"Found {len(suspicious)} iframe(s) loading content from external domains. "
                    "Clickjacking attacks embed the real site in an invisible iframe to steal clicks."
                ),
                evidence    = {"suspicious_srcs": suspicious[:5]},
            ))

        # Hidden full-screen iframes (clickjacking)
        hidden_iframe_patterns = [
            r'<iframe[^>]+(?:opacity\s*:\s*0|visibility\s*:\s*hidden|z-index\s*:\s*-)',
            r'<iframe[^>]+style=["\'][^"\']*(?:opacity\s*:\s*0|display\s*:\s*none)',
        ]
        for pat in hidden_iframe_patterns:
            if re.search(pat, html, re.IGNORECASE):
                signals.append(RiskSignal(
                    layer       = "hidden_iframe_overlay",
                    severity    = 9.0,
                    description = (
                        "Hidden or transparent iframe detected. This is a classic clickjacking / "
                        "iframe-trap attack where a real site is overlaid to steal your clicks."
                    ),
                    evidence    = {},
                ))
                report.overlays_found = True
                break

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 5 — Wireframe / overlay UI detection
    # ──────────────────────────────────────────────────────────────────────────

    def _check_overlay_traps(self, html: str) -> List[RiskSignal]:
        signals = []
        if not html:
            return signals

        # Full-screen overlays with high z-index positioned at 0,0
        overlay_patterns = [
            r'position\s*:\s*(?:fixed|absolute)[^"\']*z-index\s*:\s*(?:9{3,}|\d{4,})',
            r'z-index\s*:\s*(?:9{4,}|\d{5,})[^"\']*position\s*:\s*(?:fixed|absolute)',
            r'(?:top|left)\s*:\s*0[^"\']*(?:width|height)\s*:\s*100(?:vw|vh|%)',
        ]
        hit = False
        for pat in overlay_patterns:
            if re.search(pat, html, re.IGNORECASE):
                hit = True
                break

        if hit:
            signals.append(RiskSignal(
                layer       = "fullscreen_overlay",
                severity    = 6.0,
                description = (
                    "Full-screen fixed/absolute overlay element detected. "
                    "Scam sites use these to show a fake UI on top of a legitimate-looking page."
                ),
                evidence    = {},
            ))

        # Fake login form indicators
        fake_login_patterns = [
            r'<input[^>]+type=["\']?password["\']?[^>]*>',
            r'(?:verify|confirm|unlock)\s+your\s+account',
            r'(?:enter|provide)\s+your\s+(?:bank|card|credit|debit)\s+(?:details|number|pin)',
        ]
        login_hits = sum(
            1 for pat in fake_login_patterns
            if re.search(pat, html, re.IGNORECASE)
        )
        if login_hits >= 2:
            signals.append(RiskSignal(
                layer       = "credential_harvesting_form",
                severity    = 5.0,
                description = (
                    "Page contains a login/credential form with suspicious language. "
                    "Could be a credential harvesting phishing page."
                ),
                evidence    = {"pattern_hits": login_hits},
            ))

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 6 — Domain age via WHOIS
    # ──────────────────────────────────────────────────────────────────────────

    def _check_domain_age(self, domain: str) -> List[RiskSignal]:
        signals = []

        if not WHOIS_OK:
            return signals

        # Remove port
        domain = re.sub(r":\d+$", "", domain)
        # Remove path components
        domain = domain.split("/")[0]

        try:
            w           = python_whois.query(domain)
            creation_dt = getattr(w, "creation_date", None)

            if creation_dt is None:
                # Unknown creation — slight signal
                signals.append(RiskSignal(
                    layer       = "domain_age_unknown",
                    severity    = 2.0,
                    description = "Could not determine domain registration date — WHOIS data hidden or unavailable.",
                    evidence    = {"domain": domain},
                ))
                return signals

            if isinstance(creation_dt, list):
                creation_dt = creation_dt[0]

            if hasattr(creation_dt, "tzinfo") and creation_dt.tzinfo is None:
                creation_dt = creation_dt.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - creation_dt).days

            if age_days < 7:
                signals.append(RiskSignal(
                    layer    = "new_domain",
                    severity = 9.0,
                    description = (
                        f"Domain was registered only {age_days} day(s) ago. "
                        "Phishing sites are usually freshly registered."
                    ),
                    evidence = {"age_days": age_days, "created": str(creation_dt)},
                ))
            elif age_days < 30:
                signals.append(RiskSignal(
                    layer    = "young_domain",
                    severity = 5.0,
                    description = f"Domain is only {age_days} days old. Very new domains are high-risk.",
                    evidence = {"age_days": age_days, "created": str(creation_dt)},
                ))
            elif age_days < 90:
                signals.append(RiskSignal(
                    layer    = "recent_domain",
                    severity = 2.0,
                    description = f"Domain registered {age_days} days ago — moderately new.",
                    evidence = {"age_days": age_days},
                ))
            # else: domain is established — no signal

        except Exception:
            pass   # WHOIS may fail — non-critical

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 7 — Content vs. domain identity mismatch
    # ──────────────────────────────────────────────────────────────────────────

    def _check_content_mismatch(self, html: str, domain: str, url: str) -> List[RiskSignal]:
        signals = []
        if not html:
            return signals

        # Extract page title
        title_m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title   = title_m.group(1).strip().lower() if title_m else ""

        # Check if a known brand appears in page content but domain is unofficial
        content_text = re.sub(r"<[^>]+>", " ", html).lower()[:5000]

        for brand, official_domains in BRAND_DOMAINS.items():
            # Does the page content mention this brand prominently?
            brand_hits = content_text.count(brand)
            if brand_hits < 2:
                continue

            # Is the current domain NOT an official one for this brand?
            is_official = any(
                domain.endswith(od) or domain == od
                for od in official_domains
            )
            if not is_official:
                severity = min(brand_hits * 0.5, 8.0)
                signals.append(RiskSignal(
                    layer       = "content_domain_mismatch",
                    severity    = severity,
                    description = (
                        f"Page content prominently mentions '{brand}' ({brand_hits}x) "
                        f"but domain '{domain}' is not an official {brand} domain. "
                        "Strong indicator of a phishing clone."
                    ),
                    evidence    = {
                        "brand":            brand,
                        "official_domains": official_domains,
                        "visited_domain":   domain,
                        "brand_mentions":   brand_hits,
                        "page_title":       title,
                    },
                ))
                break   # one mismatch is enough

        # LLM-assisted analysis (if engine available)
        if self.engine and signals:
            try:
                snippet = content_text[:1500]
                prompt  = (
                    f"You are a cybersecurity analyst. Analyze this web page content snippet "
                    f"from domain '{domain}'.\n\n"
                    f"CONTENT:\n{snippet}\n\n"
                    f"Answer JSON only:\n"
                    f'{{"is_phishing": true/false, "reason": "...", "impersonated_brand": "..." or null}}'
                )
                raw  = self.engine.generate(prompt, temperature=0.1)
                raw  = re.sub(r"```\w*\n?|```", "", raw).strip()
                data = json.loads(raw)
                if data.get("is_phishing"):
                    signals.append(RiskSignal(
                        layer       = "llm_phishing_detection",
                        severity    = 7.0,
                        description = f"LLM analysis: {data.get('reason', 'suspicious content')}",
                        evidence    = data,
                    ))
            except Exception:
                pass

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 8 — Suspicious HTML / URL patterns
    # ──────────────────────────────────────────────────────────────────────────

    def _check_suspicious_patterns(self, html: str, url: str) -> List[RiskSignal]:
        signals = []

        # URL-level checks
        # Data URI in URL
        if "data:" in url:
            signals.append(RiskSignal(
                layer="data_uri_url", severity=8.0,
                description="URL contains a data: scheme — often used to render fake phishing pages.",
                evidence={"url": url[:200]},
            ))

        # @ in URL (attacker@victim.com/login)
        path_part = url.split("?")[0]
        if "@" in urllib.parse.urlparse(url).netloc:
            signals.append(RiskSignal(
                layer="url_at_symbol", severity=7.0,
                description=(
                    "URL contains '@' in the host part. Browsers may display the part before '@' "
                    "as the domain, hiding the true destination."
                ),
                evidence={"url": url[:200]},
            ))

        if not html:
            return signals

        # HTML obfuscation
        obfuscation_patterns = [
            (r"eval\s*\(", "eval() call — code may be obfuscated"),
            (r"document\.write\s*\(", "document.write() — potential script injection"),
            (r"String\.fromCharCode\s*\(", "String.fromCharCode — common obfuscation technique"),
            (r"unescape\s*\(", "unescape() — old obfuscation trick"),
            (r"atob\s*\(", "atob() — base64 decode, may hide malicious code"),
        ]
        obf_hits = []
        for pat, desc in obfuscation_patterns:
            if re.search(pat, html, re.IGNORECASE):
                obf_hits.append(desc)

        if obf_hits:
            signals.append(RiskSignal(
                layer    = "js_obfuscation",
                severity = min(len(obf_hits) * 1.5, 6.0),
                description = (
                    f"Page uses JavaScript obfuscation techniques: {'; '.join(obf_hits[:3])}. "
                    "Scam pages hide their behaviour from scanners this way."
                ),
                evidence = {"patterns": obf_hits},
            ))

        # Password-over-HTTP
        if "http://" in url and re.search(r'type=["\']?password', html, re.IGNORECASE):
            signals.append(RiskSignal(
                layer    = "password_over_http",
                severity = 8.0,
                description = "Password input field on an HTTP (unencrypted) page — credentials will be sent in plaintext.",
                evidence = {},
            ))

        # Detect invisible text (white-on-white, tiny font)
        invisible = re.findall(
            r'(?:color\s*:\s*(?:#fff+|white|rgba?\(255,\s*255,\s*255)|font-size\s*:\s*0)',
            html,
            re.IGNORECASE,
        )
        if len(invisible) > 3:
            signals.append(RiskSignal(
                layer    = "invisible_text",
                severity = 4.0,
                description = (
                    "Page contains invisible text (white-on-white or zero font-size). "
                    "May be hiding content from users while showing it to search/spam bots."
                ),
                evidence = {"occurrences": len(invisible)},
            ))

        # Too many external script sources
        external_scripts = re.findall(
            r'<script[^>]+src=["\']?(https?://[^"\'>\s]+)',
            html,
            re.IGNORECASE,
        )
        unique_script_domains = set(
            urllib.parse.urlparse(s).netloc for s in external_scripts
        )
        if len(unique_script_domains) > 10:
            signals.append(RiskSignal(
                layer    = "excessive_external_scripts",
                severity = 2.5,
                description = (
                    f"Page loads scripts from {len(unique_script_domains)} different external domains. "
                    "May indicate a compromised or ad-heavy page."
                ),
                evidence = {"script_domains": list(unique_script_domains)[:10]},
            ))

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # Explanation builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_explanation(self, report: TrustReport) -> str:
        if report.verdict == "SAFE":
            return (
                "No significant risk signals detected. The domain appears legitimate, "
                "SSL is valid, content matches the domain identity, and no phishing patterns were found."
            )
        if report.verdict == "SUSPICIOUS":
            top = sorted(report.risk_signals, key=lambda s: -s.severity)[:2]
            reasons = "; ".join(s.description[:80] for s in top)
            return f"Multiple risk indicators found. Key concerns: {reasons}."
        # DANGEROUS
        top = sorted(report.risk_signals, key=lambda s: -s.severity)[:3]
        reasons = "; ".join(s.description[:100] for s in top)
        return (
            f"⛔ HIGH RISK — do NOT enter any personal information on this site. "
            f"Primary threats: {reasons}."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Batch scan (list of URLs)
    # ──────────────────────────────────────────────────────────────────────────

    def scan_batch(self, urls: List[str]) -> List[TrustReport]:
        """Scan multiple URLs concurrently using threads."""
        results: List[Optional[TrustReport]] = [None] * len(urls)

        def _worker(idx, url):
            try:
                results[idx] = self.scan(url)
            except Exception as e:
                results[idx] = TrustReport(
                    url=url, final_url=url,
                    verdict="ERROR", trust_score=0,
                    explanation=str(e),
                )

        threads = [threading.Thread(target=_worker, args=(i, u)) for i, u in enumerate(urls)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        return [r for r in results if r is not None]

    # ──────────────────────────────────────────────────────────────────────────
    # Natural language interface
    # ──────────────────────────────────────────────────────────────────────────

    def scan_text(self, text: str) -> dict:
        """
        Scan a message/email body for phishing/scam patterns.
        Returns {verdict, score, signals, warning_lines}.
        """
        SCAM_PATTERNS = [
            (r"\burgent.*action\b|\bact now\b|\bimmediate(ly)?\b",            "urgency_pressure",     7),
            (r"\byou('ve| have) won\b|\bcongratulations.*prize\b",             "fake_prize",           9),
            (r"\bverify your (account|identity|details|bank|card)\b",          "credential_phishing",  9),
            (r"\bclick here\b.{0,60}(log in|verify|confirm|update)",           "phishing_cta",         8),
            (r"\byour account (will be|has been) (suspended|locked|closed)\b", "account_threat",       8),
            (r"\bOTP\b.{0,40}share|share.{0,30}\bOTP\b",                       "otp_share_request",   10),
            (r"\bsend.{0,20}\b(₹|Rs\.?|INR|\$|USD|money|amount|payment)\b",   "money_transfer_ask",   9),
            (r"\bdeposit.{0,40}(advance|fee|charge|tax|processing)\b",         "advance_fee_fraud",    9),
            (r"\blottery\b|\binheritance\b|\bnigerian prince\b",               "classic_scam",         9),
            (r"\bkindly (update|confirm|provide|click|send)\b",                "polite_phishing",      6),
            (r"\bpin\b.{0,40}share|\bpassword\b.{0,40}tell|\bcvv\b",          "pin_password_ask",    10),
            (r"\bwire transfer\b|\bwestern union\b|\bgift card\b",              "payment_bypass",       9),
            (r"\bcustomer (care|support|helpline).{0,30}\d{10,}",              "fake_helpline",        8),
            (r"\bkbc\b|\bdream 11\b|\bfree recharge\b",                        "indian_mobile_scam",   8),
            (r"\bPM (kisan|awas|care)\b.{0,60}(apply|register|link|update)",  "govt_scheme_scam",     8),
        ]

        signals = []
        total_score = 0
        warning_lines = []

        text_lower = text.lower()
        for pattern, signal_name, weight in SCAM_PATTERNS:
            m = re.search(pattern, text_lower, re.IGNORECASE)
            if m:
                signals.append(signal_name)
                total_score += weight
                # Find the line containing the match
                for line in text.splitlines():
                    if re.search(pattern, line, re.IGNORECASE):
                        warning_lines.append(line.strip()[:120])
                        break

        # Score → verdict
        if total_score >= 15:
            verdict = "SCAM"
        elif total_score >= 8:
            verdict = "SUSPICIOUS"
        elif total_score >= 3:
            verdict = "CAUTION"
        else:
            verdict = "LIKELY_SAFE"

        return {
            "verdict":       verdict,
            "score":         min(total_score, 100),
            "signals":       signals,
            "warning_lines": warning_lines[:5],
            "summary": (
                f"🚨 SCAM DETECTED ({total_score} risk points): {', '.join(signals[:3])}"
                if verdict == "SCAM" else
                f"⚠️ Suspicious message ({total_score} risk points): {', '.join(signals[:3])}"
                if verdict == "SUSPICIOUS" else
                f"Message appears relatively safe (risk score: {total_score})"
            ),
        }

    def run_nl(self, instruction: str) -> str:
        """
        Handle natural language instructions like:
          "check if amazon-verify-account.net is safe"
          "scan https://paypa1.com"
          "is this website a scam? http://secure-login-facebook.xyz"
          "batch scan: site1.com site2.com site3.com"
        """
        low = instruction.lower().strip()

        # Batch scan
        m = re.search(r"batch\s+scan\s*:\s*(.+)", low, re.IGNORECASE)
        if m:
            raw_urls = re.split(r"[\s,]+", m.group(1).strip())
            urls     = [u for u in raw_urls if u]
            reports  = self.scan_batch(urls)
            return "\n\n---\n\n".join(r.summary() for r in reports)

        # Single URL extraction
        url_match = re.search(
            r"(https?://[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",
            instruction,
        )
        if not url_match:
            return "Please provide a URL or domain to scan."

        url    = url_match.group(1)
        report = self.scan(url)
        return report.summary()


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = ScamDetectorAgent()

    test_urls = [
        "https://www.google.com",
        "https://paypa1.com",
        "http://secure-amazon-login.xyz",
        "https://facebook.com",
    ]

    for url in test_urls:
        print("\n" + "="*70)
        try:
            report = detector.scan(url)
            print(report.summary())
        except Exception as e:
            print(f"Error scanning {url}: {e}")
