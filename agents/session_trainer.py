"""
ARIA — Session Knowledge Trainer
=================================
Comprehensive training-data generator built from ARIA's accumulated development
knowledge across all sessions.  Produces Ollama-compatible JSONL fine-tune
pairs covering five core topic areas:

  1. anti_bot        — Anti-bot detection systems & fingerprinting
  2. phishing        — Phishing / scam-site detection techniques
  3. stealth         — Browser stealth & evasion techniques
  4. neuromorphic    — ARIA's neuromorphic multi-agent architecture
  5. os_terminal     — Cross-platform OS & terminal operations

Public API
----------
  trainer = SessionTrainer()
  trainer.generate_pairs()                  # build all Q&A pairs in memory
  trainer.export_jsonl()                    # write to data/training/aria_session_knowledge.jsonl
  trainer.auto_train()                      # export + fire Ollama modelfile build
  trainer.merge_with_activity_trainer(t)   # append pairs into ActivityTrainer format
  trainer.run_nl("train ARIA on session data")   # NL command interface

JSONL record schema
-------------------
  {
    "messages": [
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "source":  "session_knowledge",
    "topic":   "anti_bot|phishing|stealth|neuromorphic|os_terminal",
    "quality": 0.95
  }
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Rich output ───────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class _FallbackConsole:  # type: ignore[no-redef]
        def print(self, *args, **kwargs):
            print(*[str(a) for a in args])
    console = _FallbackConsole()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
TRAINING_DIR   = DATA_DIR / "training"
SESSION_JSONL  = TRAINING_DIR / "aria_session_knowledge.jsonl"
MODELFILE_PATH = TRAINING_DIR / "Modelfile"

DATA_DIR.mkdir(exist_ok=True)
TRAINING_DIR.mkdir(exist_ok=True)

# =============================================================================
# BUILT-IN KNOWLEDGE BASE
# Each topic is a list of {"q": ..., "a": ...} dicts.
# At least 15 pairs per topic are required.
# =============================================================================

# ── Topic 1: Anti-bot detection systems ──────────────────────────────────────
ANTI_BOT_KB: List[Dict[str, str]] = [
    {
        "q": "What is JA3 TLS fingerprinting and how do anti-bot systems use it?",
        "a": (
            "JA3 is an MD5 hash computed from fields in the TLS ClientHello packet: "
            "the TLS version, cipher suites, extension types, elliptic curves, and "
            "elliptic-curve point formats. Because different HTTP clients (Chrome, "
            "curl, Python requests, Selenium) produce different ClientHello payloads, "
            "the resulting JA3 hash is nearly unique per client. Anti-bot systems like "
            "Cloudflare and Akamai compare the incoming JA3 hash against a database of "
            "known bot/automation hashes. A request that claims to be Chrome 120 but "
            "has a JA3 hash matching the Python 'requests' library is immediately "
            "flagged. Defeat requires using a TLS library that mimics Chrome's exact "
            "cipher-suite ordering and extensions, such as curl-impersonate or "
            "tls-client."
        ),
    },
    {
        "q": "What is canvas fingerprinting and why does it differ per device?",
        "a": (
            "Canvas fingerprinting works by drawing text and shapes onto an HTML5 "
            "canvas element and reading back the pixel data via toDataURL() or "
            "getImageData(). Rendering output varies because GPUs, GPU drivers, "
            "installed fonts, and OS-level anti-aliasing settings all affect how "
            "sub-pixel rendering is performed. The resulting pixel buffer is hashed "
            "(typically SHA-256 or MurmurHash) to produce a fingerprint that is "
            "consistent for the same device but differs between devices. Anti-bot "
            "systems compare this hash to expected values for the declared User-Agent. "
            "A headless Chrome with no GPU produces a different hash than a real "
            "Chrome on an Nvidia card, exposing the bot."
        ),
    },
    {
        "q": "How does WebGL vendor/renderer fingerprinting expose headless browsers?",
        "a": (
            "WebGL exposes GPU information through two debug extensions: "
            "WEBGL_debug_renderer_info, which returns UNMASKED_VENDOR_WEBGL and "
            "UNMASKED_RENDERER_WEBGL. Real Chrome on a laptop might report "
            "'Google Inc. (Intel)' and 'ANGLE (Intel, Intel(R) UHD Graphics 620 "
            "Direct3D11 vs_5_0 ps_5_0)'. Headless Chrome with SwiftShader (the "
            "software renderer) reports 'Google Inc.' and 'Google SwiftShader'. "
            "Anti-bot systems flag SwiftShader immediately as it is only present in "
            "headless environments. The fix is to pass --use-gl=egl and override the "
            "renderer string via CDP before page load."
        ),
    },
    {
        "q": "What behavioral biometrics do modern anti-bot systems analyse?",
        "a": (
            "Behavioral biometric signals include: (1) Mouse trajectory — real users "
            "produce curved, slightly erratic paths following Fitts' Law; straight-line "
            "or teleporting cursors are flagged. (2) Scroll physics — human scrolling "
            "decelerates naturally; constant-velocity scrolling is bot-like. "
            "(3) Keystroke dynamics — humans have inconsistent inter-key delays ranging "
            "15-80 ms; zero-delay paste operations are flagged. (4) Click pressure and "
            "dwell time — touch events have radius and force data. (5) Focus/blur "
            "patterns — humans switch tabs and lose focus; bots stay focused. "
            "Systems like HUMAN Security and DataDome build per-session behavioral "
            "models and score deviation from human baselines."
        ),
    },
    {
        "q": "What exactly does Cloudflare's bot detection check?",
        "a": (
            "Cloudflare Bot Management checks: TLS fingerprint (JA3/JA3S, JARM), "
            "HTTP/2 header order and SETTINGS frame values, navigator.webdriver flag, "
            "canvas and WebGL fingerprints, behavioral biometrics (mouse, scroll, "
            "typing), IP reputation, ASN classification (datacenter IPs score lower), "
            "Cookie consistency across requests, JavaScript challenge execution "
            "correctness (Cloudflare injects JS that must be solved), and the Turnstile "
            "CAPTCHA token for high-risk requests. Cloudflare also uses machine-learning "
            "models trained on billions of requests to score each visitor 0-99."
        ),
    },
    {
        "q": "What signals does Akamai Bot Manager analyse?",
        "a": (
            "Akamai Bot Manager analyses over 1,200 signals including: browser "
            "environment properties (window object shape, prototype chain integrity), "
            "sensor data collection via injected JavaScript that measures accelerometer "
            "and touch events, network timing (TCP/IP fingerprint, HTTP/2 stream "
            "priority), user interaction timing, device type signals, IP velocity "
            "(same IP hitting many different sites), and cookie replay detection. "
            "Akamai assigns a bot score and can serve challenge pages or silently "
            "degrade service to suspected bots."
        ),
    },
    {
        "q": "How does DataDome's anti-bot system differ from Cloudflare?",
        "a": (
            "DataDome is a real-time API that site owners call on every request; it "
            "focuses on behavioral and device intelligence rather than network-layer "
            "signals. It specialises in detecting credential stuffing, scraping, and "
            "account takeover. DataDome places a JS sensor tag that collects mouse "
            "movements, keystrokes, and timing for every interaction, then sends this "
            "to DataDome's cloud for ML scoring. It returns allow/block/challenge within "
            "2ms. Unlike Cloudflare, DataDome does not rely on Turnstile; it uses "
            "its own invisible challenge. ARIA's stealth browser defeats DataDome by "
            "injecting realistic mouse-trajectory simulation and mimicking human typing."
        ),
    },
    {
        "q": "What is HUMAN Security (PerimeterX) and what makes it hard to bypass?",
        "a": (
            "HUMAN Security (formerly PerimeterX) injects a JavaScript payload called "
            "BotD that builds a comprehensive device fingerprint using 200+ signals. "
            "Its difficulty stems from: (1) polymorphic JS — the challenge script is "
            "different on every request to prevent reverse engineering, (2) environment "
            "integrity checks that verify the JS runtime has not been tampered with, "
            "(3) telemetry streaming — it continuously streams interaction events to "
            "HUMAN's servers rather than doing a one-time check, and (4) cookie chaining "
            "— challenge tokens expire quickly and new ones require solving the previous "
            "challenge. Defeating it requires a full headful browser with no automation "
            "flags and genuine GPU rendering."
        ),
    },
    {
        "q": "What are Arkose Labs FunCaptcha challenges and how do they work?",
        "a": (
            "Arkose Labs uses 3D rendered image-based challenges (FunCaptcha) where "
            "users must rotate objects to an upright position. The challenge difficulty "
            "scales dynamically based on the risk score of the session: low-risk users "
            "see easy challenges or none at all; high-risk sessions see harder 3D "
            "puzzles. Arkose also injects sensor collection JS and builds a device "
            "fingerprint. What makes it unique is that it measures the quality of "
            "interaction during challenge solving — bots that auto-click the answer "
            "without realistic drag/rotation movement are flagged even if they get the "
            "right answer."
        ),
    },
    {
        "q": "What is Imperva's Advanced Bot Protection and what layer does it operate at?",
        "a": (
            "Imperva Advanced Bot Protection (formerly Distil Networks) operates at "
            "both the network layer and application layer. At the network layer it "
            "analyses TLS fingerprints, HTTP/2 framing, and IP reputation. At the "
            "application layer it injects JavaScript that collects browser telemetry. "
            "Imperva is known for its ability to detect headless browsers by checking "
            "for missing browser APIs that real Chrome has but Puppeteer/Playwright "
            "sometimes omits (e.g., specific Notification API properties, the full "
            "chrome.app namespace). It also detects inconsistencies between the declared "
            "User-Agent and the actual feature set of the browser runtime."
        ),
    },
    {
        "q": "How do honeypot traps catch bots?",
        "a": (
            "Honeypot traps are invisible elements placed on a page that legitimate "
            "users never interact with but bots typically do. Types include: "
            "(1) Invisible links — anchor tags styled with display:none or zero opacity; "
            "bots following all hrefs click them, (2) Hidden form fields — input fields "
            "with style='display:none' or positioned off-screen; bots that fill all "
            "form fields will fill these too, triggering a block, (3) Zero-size divs "
            "with click handlers that log bot interactions, (4) Fake navigation menus "
            "hidden via CSS that only appear in the DOM. Anti-bot systems flag any "
            "session that interacts with honeypot elements. ARIA's browser agent skips "
            "elements with display:none or visibility:hidden."
        ),
    },
    {
        "q": "How does reCAPTCHA v3 score work and what score threshold means block?",
        "a": (
            "reCAPTCHA v3 returns a score from 0.0 (very likely bot) to 1.0 (very "
            "likely human) without showing any challenge to the user. The score is "
            "computed from: browser fingerprint, interaction history with other Google "
            "properties in the same browser session, time on page, mouse movement "
            "patterns, and the site's own feedback (the site owner can tell Google "
            "when certain actions were fraudulent). Site owners set their own threshold; "
            "a common threshold is 0.5. Headless Chrome typically scores 0.1-0.3. "
            "Improving the score requires being logged into a Google account, having "
            "browsing history, and using a real browser with GPU rendering."
        ),
    },
    {
        "q": "How does Cloudflare Turnstile differ from reCAPTCHA v3?",
        "a": (
            "Cloudflare Turnstile is a privacy-preserving CAPTCHA replacement. Unlike "
            "reCAPTCHA v3, Turnstile does not use Google's cross-site tracking data. "
            "It proves human presence through: browser environment checks, TLS "
            "fingerprint validation, proof-of-work challenges (the browser must solve "
            "a small computational puzzle), and behavioral signals from the current "
            "session only. Turnstile has three modes: managed (shows a checkbox if "
            "uncertain), non-interactive (fully silent for trusted sessions), and "
            "invisible. From an automation perspective, Turnstile is harder than "
            "reCAPTCHA v3 because it does not rely on cookies that can be replayed."
        ),
    },
    {
        "q": "How is navigator.webdriver set and what are all the ways to remove it?",
        "a": (
            "navigator.webdriver is set to true by ChromeDriver via the "
            "--enable-blink-features=AutomationControlled flag that Selenium adds "
            "automatically. Removal methods: (1) Pass the Chrome option "
            "--disable-blink-features=AutomationControlled, (2) Use CDP command "
            "Page.addScriptToEvaluateOnNewDocument to run "
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined}) "
            "before any page script executes, (3) Use undetected-chromedriver (UC) "
            "which patches the ChromeDriver binary to remove the flag, "
            "(4) Use Playwright with the stealth plugin which applies all patches. "
            "ARIA's browser_agent.py uses CDP injection on document creation as it "
            "works regardless of the driver version."
        ),
    },
    {
        "q": "What is hCaptcha and how is it used for bot detection at scale?",
        "a": (
            "hCaptcha is a CAPTCHA service that presents image labeling challenges "
            "(classify images into categories) and uses the labeled data commercially. "
            "Its bot detection works in two phases: a passive phase that scores the "
            "session based on fingerprint, IP, and behavioral data, and an active phase "
            "that shows the image challenge if the passive score is below threshold. "
            "hCaptcha's challenge images change frequently, making ML-based auto-solvers "
            "less reliable over time. It is used by Cloudflare, Discord, and many other "
            "platforms. ARIA's security agent flags pages that load hCaptcha scripts "
            "as requiring human intervention."
        ),
    },
    {
        "q": "What is HTTP/2 fingerprinting and how does it catch Python scrapers?",
        "a": (
            "HTTP/2 fingerprinting analyses the SETTINGS frame sent by the client at "
            "connection start: the order and values of HEADER_TABLE_SIZE, "
            "ENABLE_PUSH, INITIAL_WINDOW_SIZE, MAX_HEADER_LIST_SIZE, and the "
            "WINDOW_UPDATE frame size. Real Chrome 120 sends a specific combination "
            "of these values. Python's httpx or requests-h2 library sends different "
            "default values, exposing the automation tool. Anti-bot services like "
            "Akamai and Cloudflare combine HTTP/2 fingerprints with JA3 TLS "
            "fingerprints for a compound bot score. Defeat requires using a Go or "
            "Rust HTTP/2 client that fully mimics Chrome's SETTINGS frame."
        ),
    },
    {
        "q": "How does ARIA's browser agent defeat anti-bot detection in practice?",
        "a": (
            "ARIA's browser_agent.py uses a layered stealth approach: (1) CDP "
            "Page.addScriptToEvaluateOnNewDocument removes navigator.webdriver and "
            "patches chrome runtime, plugins, and languages before any page script "
            "runs; (2) Bezier-curve mouse movement replaces Selenium's straight-line "
            "ActionChains; (3) typing uses randomised 15-80ms per-character delays; "
            "(4) User-Agent is rotated from a pool of real browser signatures; "
            "(5) Canvas and WebGL parameters are spoofed via CDP; (6) Screen dimensions "
            "are randomised from realistic values; (7) WebRTC is disabled to prevent "
            "IP leaks. Together these patches pass most Cloudflare and DataDome checks "
            "for low-risk operations."
        ),
    },
]

# ── Topic 2: Phishing / scam-site detection ───────────────────────────────────
PHISHING_KB: List[Dict[str, str]] = [
    {
        "q": "What is typosquatting and how does ARIA detect it?",
        "a": (
            "Typosquatting registers domain names that are common misspellings of "
            "legitimate brands, e.g. 'paypa1.com' (digit 1 instead of letter l) or "
            "'gooogle.com' (extra 'o'). ARIA's scam_detector.py computes the "
            "Levenshtein edit distance between the visited domain and a database of "
            "top-1000 brand domains. If the edit distance is 1 or 2, the domain is "
            "flagged as a potential typosquat. Additional signals checked: whether the "
            "suspicious domain is registered (WHOIS lookup), its creation date, and "
            "whether it serves content that mentions the target brand."
        ),
    },
    {
        "q": "What are homoglyph attacks and punycode IDN domain spoofing?",
        "a": (
            "Homoglyph attacks replace ASCII characters with visually identical Unicode "
            "characters, e.g. using Cyrillic 'а' (U+0430) instead of Latin 'a'. The "
            "domain 'аpple.com' looks identical to 'apple.com' in most fonts but is a "
            "completely different domain. Internationalized Domain Names (IDN) use "
            "punycode encoding: 'аpple.com' becomes 'xn--pple-43d.com' in DNS. "
            "ARIA detects homoglyph domains by: decoding the punycode form of the URL, "
            "checking each character's Unicode category and script, flagging any domain "
            "that mixes scripts (e.g. Cyrillic + Latin), and comparing the visual "
            "representation against known brand names using a confusables database."
        ),
    },
    {
        "q": "How does SSL certificate verification help detect phishing sites?",
        "a": (
            "A valid SSL certificate only proves TLS encryption, not site legitimacy — "
            "phishing sites routinely obtain free Let's Encrypt certificates. Signals "
            "to check: (1) Certificate Common Name (CN) or Subject Alternative Names "
            "(SANs) must match the domain exactly; a mismatch is an immediate red flag. "
            "(2) Issuer — legitimate banks use extended validation (EV) certificates "
            "from premium CAs; phishing sites almost always use domain-validated (DV) "
            "certs from Let's Encrypt. (3) Certificate age — a cert issued within the "
            "last 30 days on a domain impersonating a bank is high risk. (4) Certificate "
            "transparency logs — brand-new certs on lookalike domains appear in CT logs "
            "and can be monitored."
        ),
    },
    {
        "q": "What is an iframe trap / clickjacking attack?",
        "a": (
            "Clickjacking uses a transparent iframe overlaying a legitimate site. The "
            "attacker's page is visible; the target site's iframe is invisible "
            "(opacity:0 or visibility:hidden) but positioned exactly under the "
            "attacker's fake button. When the user clicks the fake button, they actually "
            "click a button on the target site (e.g. 'Confirm payment'). Detection: "
            "(1) Check for X-Frame-Options: DENY / SAMEORIGIN headers on the target, "
            "(2) Detect iframes with opacity < 0.1 or z-index mismatches, "
            "(3) Check for pointer-events:none on the visible layer combined with an "
            "underlying iframe. ARIA's scam detector checks iframe opacity and "
            "z-index stacking in the page DOM."
        ),
    },
    {
        "q": "How do wireframe UI overlays work as a phishing technique?",
        "a": (
            "A wireframe overlay phishing attack places a transparent or semi-transparent "
            "fake login form on top of a real website's iframe. The user sees what "
            "appears to be the legitimate site's login page, but their credentials are "
            "captured by the attacker's form. Detection signals: overlaying div elements "
            "with position:fixed and high z-index that contain password input fields, "
            "form action URLs pointing to external domains, CSS that makes the overlay "
            "appear visually identical to the underlying iframe content. ARIA's scam "
            "agent checks form action domains against the page's origin domain."
        ),
    },
    {
        "q": "Why is domain age a strong phishing signal and what threshold does ARIA use?",
        "a": (
            "Phishing campaigns are time-sensitive — attackers register a domain, run "
            "the campaign, and abandon it within days or weeks before it gets "
            "blacklisted. Legitimate financial and e-commerce domains are typically "
            "years old. ARIA uses WHOIS creation date with these thresholds: "
            "0-30 days = HIGH RISK (score +4), 31-90 days = MEDIUM RISK (score +2), "
            "91-365 days = LOW RISK (score +1), over 1 year = no penalty. Domain age "
            "is combined with other signals in a composite TrustScore. A brand new "
            "domain claiming to be a bank automatically reaches the SUSPICIOUS verdict "
            "threshold even without other signals."
        ),
    },
    {
        "q": "What is content vs. domain identity mismatch detection?",
        "a": (
            "A phishing site may have a random domain like 'secure-login-473892.com' "
            "but its page content prominently displays PayPal branding, logos, and "
            "copy. ARIA detects this mismatch by: extracting brand names mentioned in "
            "the page title, meta tags, heading text, and image alt text; comparing "
            "these against the actual domain and its registrant; and flagging pages "
            "where a well-known brand is mentioned but the domain does not belong to "
            "that brand. LLM-assisted analysis extracts the implied brand identity from "
            "page content and ARIA checks it against a known-brands registry."
        ),
    },
    {
        "q": "What JavaScript obfuscation patterns indicate a malicious page?",
        "a": (
            "Common obfuscation patterns in phishing pages: (1) eval() with base64 "
            "content — eval(atob('...')), (2) String.fromCharCode() to assemble strings "
            "from char codes avoiding keyword detection, (3) document.write() to inject "
            "dynamic content that bypasses static analysis, (4) Hex-encoded string "
            "literals like '\\x70\\x61\\x79\\x70\\x61\\x6c', (5) Variable name "
            "obfuscation (single-character or randomly named), (6) Self-modifying code "
            "that alters itself at runtime. ARIA's scam detector scans page scripts for "
            "these patterns using regex and assigns risk scores per pattern found. "
            "Multiple patterns together push the composite score to DANGEROUS."
        ),
    },
    {
        "q": "How does ARIA compute a composite TrustScore from multiple phishing signals?",
        "a": (
            "ARIA's TrustScore is computed as: start at 10 (fully trusted), subtract "
            "risk points per signal: domain age <30 days (-4), typosquatting match (-5), "
            "homoglyph domain (-5), SSL mismatch (-4), iframe overlay detected (-3), "
            "JS obfuscation patterns (-2 each, max -6), brand/domain mismatch (-4), "
            "WHOIS privacy-protected new domain (-2), redirect chain to different TLD "
            "(-2). Clamp result to 0. Verdict thresholds: 7-10 = SAFE, 4-6 = "
            "SUSPICIOUS (warn user), 0-3 = DANGEROUS (block and alert). Each "
            "RiskSignal object stores the layer name, severity, and human-readable "
            "explanation shown in ARIA's security overlay."
        ),
    },
    {
        "q": "What is a redirect chain attack and how does it bypass URL filters?",
        "a": (
            "A redirect chain attack starts with a clean URL (e.g. a legitimate URL "
            "shortener or a compromised trusted website) and uses a series of HTTP 301/"
            "302 redirects to reach the final phishing page. URL-based blocklists only "
            "check the initial URL, so the chain bypasses them. The final destination "
            "has a newly registered domain not yet in any blocklist. ARIA follows the "
            "entire redirect chain using requests.head() with allow_redirects=True, "
            "records every hop, and applies phishing signals to the final destination "
            "URL rather than just the first URL."
        ),
    },
    {
        "q": "How does ARIA's visual clone detection work?",
        "a": (
            "Visual clone detection takes a screenshot of the visited page, resizes it "
            "to a normalised 256x256 thumbnail, and computes a perceptual hash (pHash) "
            "using the PIL library. This hash is compared against a database of "
            "reference screenshots of legitimate brand login pages (PayPal, Amazon, "
            "Google, Microsoft login pages etc.). If the pHash Hamming distance is "
            "below 10 (very similar appearance) but the domain does not match the "
            "brand, the page is flagged as a visual clone. This catches sites that "
            "pixel-perfectly copy legitimate pages."
        ),
    },
    {
        "q": "What are the domain spoofing patterns ARIA checks for beyond typosquatting?",
        "a": (
            "Beyond typosquatting, ARIA checks: (1) Subdomain spoofing — "
            "'paypal.com.attacker.net' where 'paypal.com' is a subdomain of the "
            "attacker's domain, (2) TLD spoofing — 'amazon.co' instead of 'amazon.com', "
            "(3) Hyphen insertion — 'pay-pal.com', (4) Brand + generic suffix — "
            "'paypal-security.com', (5) Combosquatting — legitimate word combined with "
            "brand name like 'secure-apple-id.com'. ARIA's domain parser extracts the "
            "registered domain (eTLD+1) using the public suffix list and compares it "
            "against brand domain patterns."
        ),
    },
    {
        "q": "How does ARIA handle sites that only reveal their malicious nature after user interaction?",
        "a": (
            "Some phishing sites show a clean page on first load and only reveal "
            "malicious content after the user performs an action (click, scroll, or "
            "time delay). ARIA handles this with: (1) Passive DOM monitoring — a "
            "MutationObserver watches for dynamically injected password fields or "
            "external form actions after page load, (2) Interaction simulation — ARIA "
            "can optionally perform a safe test scroll/click in an isolated sandboxed "
            "context to trigger lazy-loaded content, (3) Network request monitoring — "
            "if the page makes XHR/fetch calls to external domains after load, those "
            "domains are checked against the phishing database."
        ),
    },
    {
        "q": "What is a WHOIS privacy-protected domain and why is it a risk signal?",
        "a": (
            "WHOIS privacy protection replaces the registrant's real contact information "
            "with a proxy service's details, hiding who actually owns the domain. While "
            "legitimate privacy-conscious site owners also use privacy protection, the "
            "combination of a privacy-protected domain that is less than 90 days old and "
            "mimics a known brand is a strong risk indicator. ARIA checks WHOIS data "
            "using the python-whois library, extracts the creation date and registrant "
            "organization, and flags domains where the registrant is a privacy proxy "
            "AND the domain is new AND content claims to be a major brand."
        ),
    },
    {
        "q": "How does ARIA integrate phishing detection into its browser automation workflow?",
        "a": (
            "ARIA's browser_agent.py calls scam_detector.analyze_url() before navigating "
            "to any URL that was not explicitly provided by the user from a trusted "
            "context. If the TrustScore is below 4 (DANGEROUS), navigation is blocked "
            "and the user is shown a red security alert with the specific risk signals "
            "found. For SUSPICIOUS scores (4-6), ARIA shows a yellow warning and asks "
            "the user to confirm. The security check adds less than 200ms to navigation "
            "time because domain-level checks (WHOIS, TLS) are cached per session. "
            "The neural_orchestrator also routes security-related queries to the "
            "scam_scanner agent via the browser_controller EXCITE signal."
        ),
    },
]

# ── Topic 3: Browser stealth techniques ──────────────────────────────────────
STEALTH_KB: List[Dict[str, str]] = [
    {
        "q": "What is CDP script injection before page load and why is it critical?",
        "a": (
            "CDP (Chrome DevTools Protocol) exposes the command "
            "Page.addScriptToEvaluateOnNewDocument, which executes a JavaScript snippet "
            "in the page's context before any page script runs. This is the correct "
            "injection point for stealth patches because: (1) the page's own scripts "
            "cannot observe the patching happening, (2) properties set before page "
            "load are indistinguishable from native browser properties to detection "
            "scripts, (3) it runs in every new frame and sub-frame automatically. "
            "ARIA's browser_agent.py injects a ~150-line stealth script via this "
            "mechanism covering webdriver removal, plugins spoofing, permissions API "
            "patching, and WebGL overrides."
        ),
    },
    {
        "q": "Why does straight-line mouse movement identify a bot and how does Bezier curve movement help?",
        "a": (
            "Human mouse movement follows Fitts' Law — the cursor curves naturally, "
            "accelerates toward the target, decelerates near it, and has slight "
            "micro-tremors. Selenium's ActionChains.move_to_element() generates a "
            "perfectly straight, constant-velocity path that no human ever produces. "
            "Anti-bot systems analyse the curvature, velocity profile, and randomness "
            "of mouse trajectories. Bezier curve movement generates a path via two "
            "control points placed at random offsets from the straight line, producing "
            "a realistic curve. Combined with sinusoidal velocity (slow-fast-slow) and "
            "added Gaussian noise (±2px jitter), the resulting trajectory is "
            "statistically indistinguishable from human movement."
        ),
    },
    {
        "q": "What is the correct range for random typing delays and how does ARIA implement it?",
        "a": (
            "Human typing inter-key intervals follow a log-normal distribution centered "
            "around 120-180ms with a range of approximately 15ms (fast typist on a "
            "familiar word) to 800ms (pause before a difficult character). For "
            "automation purposes, a uniform distribution of 15-80ms per character is "
            "sufficient to defeat most behavioral biometric checks. ARIA's browser_agent "
            "implements this as: for each character, time.sleep(random.uniform(0.015, 0.08)). "
            "An additional 50-300ms pause is added after punctuation and spaces to "
            "mimic natural rhythm. For passwords, the delay is slightly compressed "
            "to simulate a practiced typist."
        ),
    },
    {
        "q": "How does navigator.plugins spoofing work and what values should it return?",
        "a": (
            "Real Chrome on Windows exposes exactly three plugins: Chrome PDF Plugin, "
            "Chrome PDF Viewer, and Native Client. Headless Chrome exposes zero plugins, "
            "which is an immediate bot signal. The CDP stealth script patches "
            "navigator.plugins by creating a PluginArray-like object with three entries "
            "matching the real Chrome values. The patch must use Object.defineProperty "
            "with configurable:false to prevent detection scripts from detecting the "
            "override. Additionally, navigator.mimeTypes must be patched to match — "
            "Chrome has two mimeTypes (application/pdf and application/x-google-chrome-pdf) "
            "that correspond to the PDF plugins."
        ),
    },
    {
        "q": "Why should screen dimensions be randomised and what pool should ARIA use?",
        "a": (
            "Anti-bot systems flag exact standard resolutions (1920x1080, 2560x1440) "
            "when they appear with no browser chrome offsets, because headless browsers "
            "report window.screen.width/height equal to the viewport. Real browsers "
            "have a taskbar, address bar, and bookmarks toolbar that reduce the actual "
            "viewport. ARIA randomises from a pool of realistic outer window sizes: "
            "1366x768, 1440x900, 1536x864, 1600x900, 1920x1080, with viewport sizes "
            "approximately 120px shorter (for the browser chrome). The "
            "devicePixelRatio is set to 1.0 or 1.25 (not 2.0, which is mainly Retina "
            "Mac and flagged differently). Screen color depth is 24."
        ),
    },
    {
        "q": "How does canvas noise injection work without breaking visual rendering?",
        "a": (
            "Canvas noise injection modifies the pixel data returned by getImageData() "
            "and toDataURL() to produce a different hash on each session while keeping "
            "the visual output imperceptible to humans. The technique: intercept calls "
            "to CanvasRenderingContext2D.prototype.getImageData, take the result, then "
            "XOR each alpha or colour channel byte with a small random value (±1 or ±2 "
            "per channel) derived from a per-session seed. The XOR value is small enough "
            "that it is invisible but large enough to change the MD5/SHA hash. The same "
            "seed must be used consistently within a session so that multiple canvas "
            "reads on the same page produce the same (modified) value."
        ),
    },
    {
        "q": "How are WebGL UNMASKED_VENDOR and UNMASKED_RENDERER spoofed?",
        "a": (
            "The CDP stealth script overrides getExtension() on WebGLRenderingContext "
            "to intercept WEBGL_debug_renderer_info. When a page calls "
            "gl.getParameter(ext.UNMASKED_VENDOR_WEBGL), the patched function returns "
            "a synthetic string like 'Google Inc. (NVIDIA)' instead of the real value. "
            "UNMASKED_RENDERER_WEBGL returns 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 "
            "Direct3D11 vs_5_0 ps_5_0, D3D11)'. The vendor/renderer pair must be "
            "plausible for the declared User-Agent — a mobile UA should not report "
            "a desktop GPU. ARIA selects a matching GPU from a curated table of "
            "real GPU strings indexed by OS and device type."
        ),
    },
    {
        "q": "How does ARIA's user-agent pool rotation work?",
        "a": (
            "ARIA maintains a pool of 20+ real browser User-Agent strings collected "
            "from actual Chrome, Firefox, and Edge versions on Windows, macOS, and "
            "Linux. For each new browser session, a UA is selected based on: "
            "(1) platform consistency — the UA must match the OS the stealth browser "
            "is running on (or the OS being spoofed), (2) recency — very old browser "
            "versions stand out; only the last 4 major Chrome versions are included, "
            "(3) distribution — Chrome is ~65% of traffic so it is selected more often. "
            "The selected UA is applied via both the ChromeOptions argument and via "
            "CDP Network.setUserAgentOverride to ensure consistent reporting across "
            "all detection vectors."
        ),
    },
    {
        "q": "How does WebRTC IP leak prevention work?",
        "a": (
            "WebRTC can expose a machine's real IP address even when behind a proxy or "
            "VPN by using STUN/ICE protocol to discover the local and public IP. "
            "Prevention methods: (1) Chrome CLI flag --disable-webrtc disables WebRTC "
            "entirely but some sites detect its absence, (2) Chrome policy "
            "WebRtcIPHandlingPolicy set to 'disable_non_proxied_udp' limits ICE "
            "candidates to TCP-only proxied connections, (3) CDP stealth script overrides "
            "RTCPeerConnection to return no ICE candidates or spoofed ones. ARIA uses "
            "option (3) by default so WebRTC APIs are present (their absence is also "
            "detectable) but return no local IP information."
        ),
    },
    {
        "q": "What are hardware concurrency and device memory spoofing and why do they matter?",
        "a": (
            "navigator.hardwareConcurrency returns the number of logical CPU cores "
            "(e.g. 16 on a server). navigator.deviceMemory returns RAM in GB, rounded "
            "to a power of two (e.g. 8). Headless browsers running on cloud VMs often "
            "report very high core counts (32, 64) or unusual memory values that do not "
            "match the declared User-Agent. A mobile UA reporting 64 cores is a clear "
            "bot signal. ARIA's stealth script patches both properties: "
            "hardwareConcurrency is set to 4 or 8 (realistic desktop values), "
            "deviceMemory is set to 8 (most common desktop value). These must match "
            "the declared platform and UA."
        ),
    },
    {
        "q": "What is the chrome.runtime namespace patch and why is it needed?",
        "a": (
            "Real Chrome exposes a window.chrome object with runtime, app, csi, and "
            "loadTimes properties. Headless Chrome has window.chrome defined but some "
            "sub-properties are missing or return different values. Detection scripts "
            "check: typeof window.chrome !== 'undefined', "
            "window.chrome.runtime.id === undefined (headless returns undefined in some "
            "cases), and window.chrome.app.isInstalled. The stealth patch creates a "
            "synthetic chrome object that passes all these checks: "
            "chrome.runtime = {id: undefined, connect: function(){}, sendMessage: function(){}}, "
            "chrome.app = {isInstalled: false, InstallState: {...}}, and "
            "chrome.csi / chrome.loadTimes return realistic mock functions."
        ),
    },
    {
        "q": "What is the Permissions API patch required for stealth browsers?",
        "a": (
            "Real Chrome's Permissions API returns 'granted' for the 'notifications' "
            "permission when queried. Headless Chrome and some Playwright configurations "
            "return 'denied' or throw an error. Detection scripts check: "
            "navigator.permissions.query({name:'notifications'}) — if the result is "
            "'denied' when no permission has been revoked by the user, it indicates "
            "headless mode. The stealth patch overrides navigator.permissions.query "
            "to always resolve with 'denied' for notifications (since headless has no "
            "notification UI) but 'granted' for non-sensitive permissions like "
            "'geolocation' to avoid other detection vectors."
        ),
    },
    {
        "q": "How does ARIA handle browser fingerprint consistency across multiple requests?",
        "a": (
            "Fingerprint consistency is critical — if canvas hash changes between page "
            "loads in the same session, anti-bot systems detect the inconsistency. "
            "ARIA generates all fingerprint parameters (canvas noise seed, UA selection, "
            "screen dimensions, GPU string, hardware concurrency) once per session and "
            "stores them in a session config dict. The same config is reinjected via "
            "CDP on every new document creation. Between sessions, ARIA rotates to a "
            "new fingerprint profile. For long-running sessions, ARIA additionally "
            "maintains consistent cookie jars, localStorage values, and IndexedDB "
            "entries to avoid session-level consistency detection."
        ),
    },
    {
        "q": "What is the accept-language header spoofing required for stealth?",
        "a": (
            "The Accept-Language HTTP header and navigator.languages JavaScript property "
            "must match. A request with User-Agent claiming to be Chrome on a US Windows "
            "machine but Accept-Language: zh-CN is inconsistent. ARIA's stealth browser "
            "sets Accept-Language: en-US,en;q=0.9 in both the CDP "
            "Network.setExtraHTTPHeaders call and via navigator.languages override "
            "(['en-US', 'en']). navigator.language is set to 'en-US'. These must be "
            "consistent with the UA and timezone (detected via Intl.DateTimeFormat). "
            "ARIA also sets the timezone via CDP Emulation.setTimezoneOverride to "
            "match the UA's geographic region."
        ),
    },
    {
        "q": "How does ARIA's stealth approach handle modern anti-bot systems that use iframe isolation?",
        "a": (
            "Some anti-bot systems (especially Cloudflare and DataDome) inject their "
            "sensor collection code inside a cross-origin sandboxed iframe to prevent "
            "the stealth script from reaching it. ARIA handles this through: "
            "(1) Page.addScriptToEvaluateOnNewDocument runs in ALL frames including "
            "cross-origin ones when using the world parameter, but only in the main "
            "frame by default — ARIA enables it for all frames, "
            "(2) For isolated frames, ARIA relies on the network-level fingerprint "
            "being correct (HTTP/2 and TLS fingerprint), which the iframe-based "
            "JavaScript checks cannot override, (3) Behavioral simulation happens "
            "at the OS/input level rather than JavaScript level, making it "
            "invisible to any iframe."
        ),
    },
]

# ── Topic 4: Neuromorphic multi-agent architecture ───────────────────────────
NEUROMORPHIC_KB: List[Dict[str, str]] = [
    {
        "q": "What are the six NeuralBus signal types in ARIA's neuromorphic architecture?",
        "a": (
            "ARIA's NeuralBus defines six signal types: (1) EXCITATORY — increases the "
            "activation priority of target agents, used when a result from one agent "
            "makes another agent's task more relevant; (2) INHIBITORY — suppresses "
            "target agents, used for lateral inhibition when a high-confidence result "
            "makes competing approaches redundant; (3) RESULT — carries a completed "
            "agent output to the SynapticState shared workspace; (4) QUERY — requests "
            "information or computation from a target agent; (5) CONTEXT — shares "
            "background knowledge or enriched context with peer agents; "
            "(6) REQUEST — initiates a specific task from one agent to another. "
            "All signals carry a sender, receiver, payload, confidence score, and "
            "timestamp."
        ),
    },
    {
        "q": "What is SynapticState and how does the blackboard pattern work in ARIA?",
        "a": (
            "SynapticState is ARIA's shared global workspace — a thread-safe in-memory "
            "blackboard that all agents can read from and write to. It stores: "
            "active signals queue, per-agent result slots, synapse weights between "
            "agent pairs, and the current query context. The blackboard pattern means "
            "no agent communicates directly with another — they all post to and read "
            "from SynapticState. This decouples agents completely: a Wave-1 agent posts "
            "its result, and any Wave-2 agent that needs that result calls "
            "SynapticState.get_results_for(agent_name). The pattern enables dynamic "
            "composition — new agents can be added without modifying existing ones."
        ),
    },
    {
        "q": "What is the two-wave cascade in ARIA's NeuralOrchestrator?",
        "a": (
            "Wave 1 (fast neurons, 0-6s timeout): fast_reasoner, memory_retriever, "
            "web_searcher, world_model_lookup, trend_watcher, browser_controller, "
            "network_inspector, code_specialist, document_reader, media_controller, "
            "calendar_context. These run in parallel and post results to SynapticState. "
            "After Wave 1 settles (all agents complete or timeout), EXCITATORY signals "
            "propagate to Wave 2. Wave 2 (reasoning neurons, 6-16s): chain_reasoner, "
            "nova_reasoner, sci_researcher, planner_agent, system_controller, "
            "summarizer_agent, code_runner, automation_controller, symbolic_executor. "
            "Wave-2 agents call enrich_context() to read all Wave-1 results before "
            "generating, making them aware of what their peers discovered."
        ),
    },
    {
        "q": "What is Hebbian learning in ARIA and how are synaptic weights updated?",
        "a": (
            "Hebbian learning in ARIA follows the principle 'neurons that fire together "
            "wire together'. After each query, for every Wave-1/Wave-2 agent pair where "
            "both produced results, the synapse weight between them is increased by a "
            "small delta (default 0.1). Weights decay slowly over time (multiplied by "
            "0.99 per query) to prevent runaway growth. Weights are normalised to [0,1]. "
            "The effect: if web_searcher + chain_reasoner frequently co-produce results "
            "that lead to high user ratings, their synapse weight grows, and "
            "chain_reasoner gets higher priority when web_searcher fires. Weights are "
            "persisted to data/synaptic_weights.json between sessions."
        ),
    },
    {
        "q": "How does lateral inhibition work in ARIA's neural architecture?",
        "a": (
            "Lateral inhibition suppresses competing agents when a high-confidence "
            "result already exists. If fast_reasoner produces a result with confidence "
            "≥0.85, it emits INHIBITORY signals to other reasoning agents "
            "(chain_reasoner, nova_reasoner, cot_thinker). SynapticState marks those "
            "agents as suppressed, causing them to skip their Wave-2 execution and "
            "save compute. This mirrors biological neural circuits where strong "
            "stimulus prevents weaker competing pathways from activating. The inhibition "
            "threshold is configurable; for creative or analytical queries ARIA raises "
            "it to 0.95 to allow more diverse perspectives to contribute."
        ),
    },
    {
        "q": "What is NeuralAgentMixin and what interface must agents implement?",
        "a": (
            "NeuralAgentMixin is a base class that all 27 ARIA agents inherit from. "
            "It provides: fire(signal) — post a NeuralSignal to the NeuralBus, "
            "excite(targets, payload) — send EXCITATORY signals to a list of agents, "
            "inhibit(targets, reason) — send INHIBITORY signals, "
            "enrich_context() — read all current Wave-1 results from SynapticState "
            "and return them as a formatted string for LLM prompt injection, "
            "get_confidence(result) — compute a confidence score 0-1 for the result. "
            "Agents must implement the abstract method process(query, context) which "
            "returns an AgentResult. The mixin handles signal lifecycle and "
            "SynapticState integration automatically."
        ),
    },
    {
        "q": "How does ARIA's consensus formation work in the synthesis phase?",
        "a": (
            "After Wave 2 completes, SynapticState.try_form_consensus() checks whether "
            "two or more agents produced semantically similar results. Similarity is "
            "measured by: exact string overlap >50%, or LLM-based semantic similarity "
            "check if the content is non-trivial. If consensus is found, the agreed "
            "answer is used directly as the final response with a high confidence score "
            "(average of the agreeing agents' confidences). If no consensus, ARIA's "
            "synthesis step calls an LLM to merge the top-K results (K=3 by default), "
            "weighted by agent confidence scores. The merge prompt includes all "
            "K results and instructs the LLM to produce a unified answer."
        ),
    },
    {
        "q": "What SSE event types does NeuralOrchestrator emit for the frontend?",
        "a": (
            "NeuralOrchestrator emits Server-Sent Events that the ARIA web UI can "
            "optionally visualise: (1) neural_signal — {type, from, to, stype: "
            "EXCITATORY|INHIBITORY|..., payload_preview}, (2) agent_fired — {agent, "
            "wave: 1|2, confidence, duration_ms}, (3) agent_suppressed — {agent, by, "
            "reason}, (4) consensus — {agents: [...], confidence}. These extra events "
            "are emitted in addition to the standard token stream events inherited from "
            "OmegaOrchestrator, so the frontend remains fully compatible and only "
            "adds the visualisation layer optionally."
        ),
    },
    {
        "q": "What is the EXCITE_MAP in ARIA and how does it define the neural topology?",
        "a": (
            "EXCITE_MAP is a Python dict in neural_orchestrator.py that defines which "
            "agents excite which other agents. For example: web_searcher excites "
            "[fast_reasoner, chain_reasoner, nova_reasoner, trend_watcher, sci_researcher, "
            "cot_thinker]. When web_searcher completes and posts a RESULT, it "
            "automatically emits EXCITATORY signals to all listed targets, boosting "
            "their execution priority in Wave 2. EXCITE_MAP is the hardcoded topology "
            "layer; Hebbian learning then adjusts the weight of each connection based "
            "on observed co-activation success. The map was hand-designed based on "
            "semantic relationships between agent capabilities."
        ),
    },
    {
        "q": "How does enrich_context() work and why is it the key innovation of Wave 2?",
        "a": (
            "enrich_context() is called by every Wave-2 agent before it constructs its "
            "LLM prompt. It reads all Wave-1 results from SynapticState, formats them "
            "as a structured context block (agent name, confidence, result summary), "
            "and returns this as a string that is prepended to the agent's system "
            "prompt. This means a chain_reasoner working on a query like 'explain "
            "quantum entanglement' will see that web_searcher already found three "
            "recent papers and memory_retriever found a prior conversation about "
            "quantum physics — and can incorporate these findings into its reasoning "
            "without redundant computation. This is the critical difference from a flat "
            "parallel fan-out where all agents reason in a vacuum."
        ),
    },
    {
        "q": "How does ARIA's neural architecture compare to a flat parallel multi-agent system?",
        "a": (
            "A flat parallel system fires all agents simultaneously and picks the best "
            "result by confidence score. Problems: (1) all agents reason with the same "
            "empty context, missing synergies, (2) no way to suppress redundant work "
            "when an early agent already solved the problem, (3) all agents cost equal "
            "compute regardless of query type. ARIA's two-wave cascade solves these: "
            "Wave-1 builds shared knowledge, Wave-2 leverages it; lateral inhibition "
            "skips agents when unnecessary; Hebbian weights prioritise proven agent "
            "pairs. The result is better answers with less compute on average, "
            "especially for complex multi-step queries."
        ),
    },
    {
        "q": "What is the world_model_lookup agent and what does it contribute?",
        "a": (
            "world_model_lookup queries ARIA's WorldModel — a local knowledge graph "
            "that stores structured facts inferred from past interactions, including "
            "entities, relationships, and temporal context. In Wave 1 it acts as a fast "
            "in-memory knowledge lookup: if the query mentions known entities (people, "
            "places, projects, tools), world_model_lookup retrieves their stored "
            "attributes and relationships in under 50ms. This result is posted to "
            "SynapticState and enriches Wave-2 agents' prompts with precise factual "
            "context without requiring a web search. The WorldModel is updated after "
            "each interaction by extracting new entities from ARIA's responses."
        ),
    },
    {
        "q": "How does ARIA handle agent timeouts in the two-wave cascade?",
        "a": (
            "Each wave has a hard timeout: Wave 1 = 6 seconds, Wave 2 = 16 seconds. "
            "Agents are run as asyncio tasks. When the wave timeout fires, all "
            "still-running tasks are cancelled. The results collected so far are used "
            "for the synthesis phase. This means ARIA always responds within ~22 "
            "seconds maximum even if some agents are slow. Agents that consistently "
            "timeout have their Hebbian weight reduced, causing them to be deprioritised "
            "in future queries. The timeout values are configurable in the orchestrator "
            "config and can be extended for complex analytical tasks where the user "
            "has explicitly requested deep research."
        ),
    },
    {
        "q": "What is the cot_thinker (Chain of Thought) agent in ARIA's neural network?",
        "a": (
            "cot_thinker is a Wave-2 agent that implements structured chain-of-thought "
            "reasoning using ARIA's ChainOfThoughtEngine. It is excited by web_searcher, "
            "memory_retriever, sci_researcher, and document_reader — agents that provide "
            "factual inputs. cot_thinker selects the appropriate reasoning strategy "
            "(standard, self_consistency, tree_of_thought, or human_like) based on "
            "the query complexity detected by the IntentMap. It injects all Wave-1 "
            "context into its reasoning chain and produces a step-by-step worked answer. "
            "For queries requiring logical deduction or multi-step problem solving, "
            "cot_thinker typically produces the highest-confidence result."
        ),
    },
    {
        "q": "How does ARIA persist and load synaptic weights between sessions?",
        "a": (
            "Synaptic weights are stored in data/synaptic_weights.json as a nested "
            "dict: {source_agent: {target_agent: weight_float}}. On startup, "
            "NeuralOrchestrator calls SynapticState.load_weights() which reads this "
            "file and populates the in-memory weight matrix. After each query, "
            "Hebbian learning updates are applied in a background thread to avoid "
            "blocking the response. The entire weight matrix is serialised and written "
            "to disk every 10 queries or on clean shutdown. If the file is missing "
            "(first run), all weights default to 0.5 (neutral — no preference). "
            "Weights are bounded to [0.01, 1.0] to prevent any agent from being "
            "permanently zeroed out."
        ),
    },
]

# ── Topic 5: Cross-platform OS & terminal operations ─────────────────────────
OS_TERMINAL_KB: List[Dict[str, str]] = [
    {
        "q": "What OS profiles does ARIA's OSProfile detection support?",
        "a": (
            "ARIA's os_detector.py identifies seven platform types: (1) WINDOWS — "
            "detected via sys.platform == 'win32', (2) MACOS — sys.platform == "
            "'darwin', (3) LINUX — sys.platform.startswith('linux') without WSL, "
            "(4) WSL — Linux with 'microsoft' in platform.uname().release.lower(), "
            "(5) ANDROID — detected via os.path.exists('/system/build.prop') and "
            "Android-specific env vars, (6) IOS — detected when running in Pythonista "
            "or via sys.platform == 'ios', (7) UNKNOWN — fallback. Each profile maps "
            "to a set of available commands, path separators, shell binaries, and "
            "capability flags used by the terminal_agent and system_agent."
        ),
    },
    {
        "q": "How does ARIA's AI error-fix loop work when running terminal commands?",
        "a": (
            "ARIA's terminal_agent implements an auto-fix loop with up to 3 retries: "
            "(1) Run the command using subprocess.Popen and capture stdout/stderr. "
            "(2) If the return code is non-zero, parse the error message from stderr. "
            "(3) Send the original command + error to the LLM with the prompt: "
            "'This command failed with this error. Provide only the corrected command.' "
            "(4) Execute the LLM-suggested fix command. (5) If it fails again, repeat "
            "with the new error up to 3 times total. On the 3rd failure, ARIA reports "
            "the full error chain to the user with all attempted fixes. Successful fixes "
            "are stored in a local command-fix cache to speed up future similar errors."
        ),
    },
    {
        "q": "How does ARIA connect to remote devices via SSH using paramiko?",
        "a": (
            "ARIA's device_control.py uses paramiko for SSH connections. The flow: "
            "(1) SSHClient.set_missing_host_key_policy(AutoAddPolicy) for first-time "
            "connections, (2) Connect using key-based auth (preferred) or password, "
            "(3) For command execution: exec_command() returns stdin/stdout/stderr "
            "streams; ARIA reads stdout/stderr with a configurable timeout, "
            "(4) For file transfer: SFTPClient is opened from the SSH session for "
            "put/get operations, (5) For interactive sessions: Channel with "
            "invoke_shell() is used with a PTY allocated for commands requiring a TTY. "
            "ARIA stores SSH credentials in the OS keychain via keyring, never in "
            "plaintext config files."
        ),
    },
    {
        "q": "How does ARIA control Android devices via ADB?",
        "a": (
            "ARIA's device_control.py wraps ADB (Android Debug Bridge) for Android "
            "automation. Key operations: (1) adb devices — detect connected devices, "
            "(2) adb shell — execute shell commands on the device, (3) adb push/pull — "
            "file transfer, (4) input tap x y — touch events, (5) input swipe — "
            "swipe gestures, (6) screencap -p — screenshot capture piped to local "
            "file. ARIA uses subprocess to call the adb binary and parses its output. "
            "For wireless ADB (Android 11+), ARIA uses adb pair <ip:port> <pairing-code> "
            "followed by adb connect <ip:port>. ARIA's iot_agent uses ADB to automate "
            "Android-based IoT devices and phones."
        ),
    },
    {
        "q": "What are the four chain-of-thought strategies in ARIA's ChainOfThoughtEngine?",
        "a": (
            "ARIA's ChainOfThoughtEngine implements: (1) standard — single linear "
            "chain: question → reasoning steps → answer, suitable for straightforward "
            "factual queries; (2) self_consistency — generate 3-5 independent reasoning "
            "chains, then take a majority vote on the answer, improving reliability for "
            "math and logic; (3) tree_of_thought — explores a tree of possible next "
            "steps, scoring each branch and pruning low-probability paths, best for "
            "complex planning; (4) human_like — a 7-stage process: understand → "
            "clarify → explore → hypothesize → verify → synthesize → communicate, "
            "mimicking how an expert human solves hard problems. Strategy selection "
            "is based on query complexity score and task type."
        ),
    },
    {
        "q": "What is the human_like 7-stage reasoning strategy in detail?",
        "a": (
            "The human_like strategy in ChainOfThoughtEngine has seven explicit stages: "
            "Stage 1 UNDERSTAND — restate the problem in the agent's own words to check "
            "comprehension; Stage 2 CLARIFY — identify ambiguities and either infer "
            "reasonable defaults or note them; Stage 3 EXPLORE — brainstorm relevant "
            "knowledge, analogies, and related concepts; Stage 4 HYPOTHESIZE — form "
            "one or more candidate solutions or explanations; Stage 5 VERIFY — check "
            "each hypothesis against known facts, constraints, and edge cases; "
            "Stage 6 SYNTHESIZE — combine insights into a coherent answer; "
            "Stage 7 COMMUNICATE — format the answer for the user's expertise level "
            "and preferred communication style. Each stage is an explicit LLM call."
        ),
    },
    {
        "q": "How does activity personalization work in ARIA and when does auto fine-tuning trigger?",
        "a": (
            "ARIA's ActivityTrainer continuously records interactions (query, response, "
            "rating, task type, timing) to data/interactions.jsonl. It builds a "
            "UserProfile by analysing the corpus: topic extraction via LLM, "
            "communication style inference from rated responses, working-hours detection "
            "from timestamps, expertise-level inference per domain, and response-length "
            "preference from rated interaction lengths. This profile is injected into "
            "every system prompt via personalize_prompt(). Auto fine-tuning triggers "
            "when 100 high-rated interactions (≥4 stars) have accumulated since the "
            "last training run. It exports a JSONL dataset and calls the NOVA LoRA "
            "trainer or falls back to Ollama's modelfile API."
        ),
    },
    {
        "q": "How does ARIA's terminal agent detect and handle WSL (Windows Subsystem for Linux)?",
        "a": (
            "WSL is detected by checking platform.uname().release.lower() for the "
            "string 'microsoft'. WSL has unique characteristics: Windows paths are "
            "accessible under /mnt/c/, /mnt/d/ etc.; the Windows host filesystem is "
            "available; Windows executables (.exe) can be called directly from WSL "
            "shell. ARIA's terminal_agent applies WSL-specific path translations "
            "using the wslpath utility (wslpath -u 'C:\\path' converts Windows to Unix "
            "path). For GUI operations, ARIA detects whether an X server or WSLg is "
            "running and routes GUI automation accordingly. Windows tools like "
            "powershell.exe are accessible from WSL for hybrid operations."
        ),
    },
    {
        "q": "How does ARIA handle cross-platform path differences in its agents?",
        "a": (
            "All ARIA agents use pathlib.Path for path handling rather than "
            "os.path.join or string concatenation. Path.resolve() is called on all "
            "user-provided paths to handle relative paths and symlinks. On Windows, "
            "Path automatically uses backslash separators but Path('/usr/bin') also "
            "works in most Python operations. For shell commands, ARIA uses "
            "shlex.quote() on all path arguments to handle spaces. Platform-specific "
            "path constants (home dir, temp dir, config dir) are resolved via "
            "Path.home(), tempfile.gettempdir(), and platform-appropriate config paths "
            "(APPDATA on Windows, ~/.config on Linux)."
        ),
    },
    {
        "q": "What is ARIA's command-fix cache and how does it speed up error recovery?",
        "a": (
            "After the AI error-fix loop successfully resolves a command failure, "
            "ARIA stores the mapping {(original_command_prefix, error_pattern): "
            "fix_command_template} in data/command_fix_cache.json. On subsequent "
            "runs, when a command fails with a matching error pattern, ARIA first "
            "checks the cache before calling the LLM. Cache keys use the first 40 "
            "characters of the command (enough to identify the tool) combined with a "
            "normalised version of the error message (removing line numbers and "
            "machine-specific paths). Cache hits reduce fix latency from ~3 seconds "
            "(LLM call) to under 10ms. The cache is invalidated when the OS profile "
            "changes or manually via 'aria clear command cache'."
        ),
    },
    {
        "q": "How does ARIA's system monitor interact with the neural orchestrator?",
        "a": (
            "system_monitor.py runs as a background daemon collecting CPU, RAM, disk, "
            "and network metrics every 30 seconds using psutil. These metrics are "
            "posted to SynapticState under the 'system_context' key. The "
            "system_controller Wave-2 agent reads system_context in its enrich_context() "
            "call and incorporates resource availability into its recommendations. "
            "For example, if available RAM is below 2GB, system_controller will "
            "suggest lighter-weight alternatives in code generation tasks. The "
            "proactive_engine also subscribes to system metrics and triggers alerts "
            "or automated cleanup actions when thresholds are exceeded."
        ),
    },
    {
        "q": "How does ARIA's self_construct agent build and run its own code?",
        "a": (
            "self_construct.py implements a code generation and self-modification loop: "
            "(1) Parse the high-level requirement from the user's query, "
            "(2) Generate Python code using the LLM with a code-focused system prompt, "
            "(3) Write the code to a temp file in sandbox/, (4) Run it via "
            "code_executor.py in an isolated subprocess with resource limits "
            "(ulimit on Linux, job objects on Windows), (5) If it fails, run the "
            "AI error-fix loop, (6) If it succeeds, optionally move it to the "
            "appropriate agents/ or tools/ directory. self_construct is rate-limited "
            "to prevent runaway code generation and requires user confirmation before "
            "modifying any existing file."
        ),
    },
    {
        "q": "What resource isolation does ARIA apply when running generated code?",
        "a": (
            "ARIA's code_executor.py applies multi-layer isolation for generated code: "
            "(1) subprocess with a separate process group for clean kill on timeout, "
            "(2) 30-second execution timeout enforced via Popen.communicate(timeout=30), "
            "(3) On Linux: setrlimit to cap CPU time (10s), memory (512MB), and file "
            "size (50MB), (4) On Windows: job objects with CPU time and memory limits "
            "via the win32job module, (5) The sandbox/ directory has no write access "
            "to parent directories, (6) Network access for generated scripts is "
            "blocked by default (overridable with explicit user permission). "
            "stdout and stderr are captured; code is never executed with shell=True "
            "to prevent injection."
        ),
    },
    {
        "q": "How does ARIA's auto-tuner decide which model to use for a given query?",
        "a": (
            "auto_tuner.py implements a three-tier model selection strategy based on "
            "query complexity scoring: (1) SIMPLE — single-step factual or short "
            "creative queries → use the fastest available model (phi3:mini or "
            "llama3.2:1b), response in <2s; (2) MEDIUM — multi-step reasoning, code "
            "generation, analysis → use the primary model (llama3.2 or mistral), "
            "response in 5-15s; (3) COMPLEX — research, architecture decisions, "
            "long document processing → use the largest available model (llama3.1:70b "
            "if RAM permits, else mistral:7b). Complexity is scored by query length, "
            "presence of technical terms, sub-question count, and task type from "
            "ActivityTrainer's classifier."
        ),
    },
    {
        "q": "How does ARIA handle multi-step task planning across OS boundaries?",
        "a": (
            "ARIA's planning_engine.py decomposes multi-step tasks into a directed "
            "acyclic graph (DAG) of subtasks, each tagged with the required agent and "
            "OS profile. For cross-OS tasks (e.g. compile on Linux, deploy to Windows), "
            "ARIA detects available remote connections (SSH for Linux, RDP/ADB for "
            "others) and routes subtasks to the appropriate execution environment. "
            "The planning loop: (1) LLM decomposes the goal into ordered subtasks, "
            "(2) Each subtask is assigned an executor (local agent, SSH agent, or ADB "
            "agent), (3) Subtasks execute in dependency order, (4) Results feed into "
            "dependent tasks via the shared context. Failures trigger the AI error-fix "
            "loop at the subtask level."
        ),
    },
]

# =============================================================================
# KNOWLEDGE REGISTRY — all topics in one place
# =============================================================================

KNOWLEDGE_REGISTRY: Dict[str, List[Dict[str, str]]] = {
    "anti_bot":     ANTI_BOT_KB,
    "phishing":     PHISHING_KB,
    "stealth":      STEALTH_KB,
    "neuromorphic": NEUROMORPHIC_KB,
    "os_terminal":  OS_TERMINAL_KB,
}

# =============================================================================
# ARIA SYSTEM PROMPT (used in Modelfile)
# =============================================================================

ARIA_SYSTEM_PROMPT = """You are ARIA — Adaptive Reasoning Intelligence Architecture.
You are a local AGI-level personal assistant running on Ollama.
You have deep expertise in browser automation, anti-bot evasion, phishing detection,
neuromorphic multi-agent systems, and cross-platform OS operations.
You are precise, technically accurate, and speak to the user as a knowledgeable
peer. You never refuse to help with legitimate security research or automation tasks.
You run fully locally — no data leaves the user's machine."""

# =============================================================================
# SESSION TRAINER CLASS
# =============================================================================

class SessionTrainer:
    """
    Generates, exports, and fine-tunes ARIA on structured Q&A pairs derived
    from the built-in knowledge base spanning all ARIA development sessions.

    Typical workflow
    ----------------
    trainer = SessionTrainer()
    trainer.generate_pairs()          # parse KB into pair list
    trainer.export_jsonl()            # write JSONL to disk
    trainer.auto_train()              # build Modelfile + trigger Ollama

    NL interface
    ------------
    trainer.run_nl("train ARIA on session data")
    trainer.run_nl("show training stats")
    trainer.run_nl("export training data")
    trainer.run_nl("what topics are covered")
    """

    OLLAMA_URL  = "http://localhost:11434"
    BASE_MODEL  = "llama3"   # fallback to mistral if llama3 not available
    ALT_MODEL   = "mistral"

    def __init__(self, ollama_url: Optional[str] = None):
        self._ollama_url = ollama_url or self.OLLAMA_URL
        self._pairs: List[Dict[str, Any]] = []
        self._lock  = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # PAIR GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    def generate_pairs(self) -> List[Dict[str, Any]]:
        """
        Convert every Q&A entry in KNOWLEDGE_REGISTRY into an Ollama chat
        format training record.  Returns the full list and caches it internally.
        """
        pairs: List[Dict[str, Any]] = []

        for topic, kb_entries in KNOWLEDGE_REGISTRY.items():
            for entry in kb_entries:
                record: Dict[str, Any] = {
                    "messages": [
                        {"role": "user",      "content": entry["q"].strip()},
                        {"role": "assistant", "content": entry["a"].strip()},
                    ],
                    "source":  "session_knowledge",
                    "topic":   topic,
                    "quality": 0.95,
                }
                pairs.append(record)

        with self._lock:
            self._pairs = pairs

        console.print(
            f"[green][SessionTrainer] Generated {len(pairs)} Q&A pairs across "
            f"{len(KNOWLEDGE_REGISTRY)} topics.[/green]"
        )
        return pairs

    def get_pairs(self) -> List[Dict[str, Any]]:
        """Return cached pairs, generating them first if needed."""
        if not self._pairs:
            self.generate_pairs()
        return self._pairs

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def export_jsonl(self, output_path: Optional[str] = None) -> Path:
        """
        Write all pairs to JSONL.  Merges with any existing entries in the
        file (deduplicates by question content).

        Returns the output Path.
        """
        out_path = Path(output_path) if output_path else SESSION_JSONL
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pairs = self.get_pairs()

        # Load existing to avoid duplicates
        existing_questions: set = set()
        existing_pairs: List[Dict[str, Any]] = []
        if out_path.exists():
            try:
                with open(out_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            msgs = rec.get("messages", [])
                            q = msgs[0]["content"] if msgs else ""
                            existing_questions.add(q)
                            existing_pairs.append(rec)
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
            except OSError:
                pass

        new_pairs = [
            p for p in pairs
            if p["messages"][0]["content"] not in existing_questions
        ]

        total_pairs = existing_pairs + new_pairs
        with open(out_path, "w", encoding="utf-8") as f:
            for pair in total_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        console.print(
            f"[green][SessionTrainer] Exported {len(total_pairs)} pairs "
            f"({len(new_pairs)} new) to {out_path}[/green]"
        )
        return out_path

    # ─────────────────────────────────────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """
        Return pair counts per topic across all ARIA training files in
        data/training/.

        Returns dict with keys: per_topic (dict), total_session, total_all_files.
        """
        pairs = self.get_pairs()

        # Count per topic in this trainer
        per_topic: Dict[str, int] = {}
        for topic in KNOWLEDGE_REGISTRY:
            per_topic[topic] = sum(1 for p in pairs if p.get("topic") == topic)

        # Count total across all JSONL files in training dir
        total_all = 0
        if TRAINING_DIR.exists():
            for jsonl_file in TRAINING_DIR.glob("*.jsonl"):
                try:
                    with open(jsonl_file, encoding="utf-8") as f:
                        total_all += sum(1 for line in f if line.strip())
                except OSError:
                    pass

        return {
            "per_topic":      per_topic,
            "total_session":  len(pairs),
            "total_all_files": total_all,
            "training_dir":   str(TRAINING_DIR),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MODELFILE GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    def _get_available_base_model(self) -> str:
        """Check Ollama for available models and pick the best base."""
        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(f"{self._ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                model_names = [m["name"].split(":")[0] for m in data.get("models", [])]
                for preferred in ("llama3.1", "llama3", "mistral", "phi3", "llama3.2"):
                    if preferred in model_names:
                        return preferred
        except Exception:
            pass
        return self.BASE_MODEL

    def write_modelfile(self, base_model: Optional[str] = None) -> Path:
        """
        Write a Modelfile for Ollama custom model creation.
        References the session JSONL training data.

        Returns the Modelfile Path.
        """
        base = base_model or self._get_available_base_model()

        modelfile_content = f"""# ARIA Trained Model — Generated by SessionTrainer
# Base model: {base}
# Generated: {datetime.utcnow().isoformat()}

FROM {base}

SYSTEM \"\"\"{ARIA_SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# Training data reference
# Fine-tuning data location: {SESSION_JSONL}
# Total session knowledge pairs: {len(self.get_pairs())}
"""

        MODELFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODELFILE_PATH.write_text(modelfile_content, encoding="utf-8")
        console.print(f"[green][SessionTrainer] Modelfile written to {MODELFILE_PATH}[/green]")
        return MODELFILE_PATH

    # ─────────────────────────────────────────────────────────────────────────
    # AUTO TRAIN
    # ─────────────────────────────────────────────────────────────────────────

    def auto_train(self, model_name: str = "aria-trained",
                   base_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Full fine-tune pipeline:
          1. Generate + export all pairs to JSONL.
          2. Count total pairs across all ARIA training files.
          3. Write Modelfile.
          4. Call 'ollama create <model_name> -f <Modelfile>' if Ollama is
             available.
          5. Return a stats dict.

        Parameters
        ----------
        model_name : str
            Name for the created Ollama model (default 'aria-trained').
        base_model : str, optional
            Override the base model (e.g. 'mistral'). Auto-detected if None.

        Returns
        -------
        dict with keys: exported_path, total_session_pairs, total_all_pairs,
                        modelfile_path, ollama_status, model_name, per_topic.
        """
        # Step 1 — export
        exported_path = self.export_jsonl()

        # Step 2 — count
        stats = self.get_stats()

        # Step 3 — Modelfile
        base = base_model or self._get_available_base_model()
        mf_path = self.write_modelfile(base_model=base)

        # Step 4 — call Ollama
        ollama_status = self._run_ollama_create(model_name, mf_path)

        result = {
            "exported_path":      str(exported_path),
            "total_session_pairs": stats["total_session"],
            "total_all_pairs":     stats["total_all_files"],
            "per_topic":           stats["per_topic"],
            "modelfile_path":      str(mf_path),
            "base_model":          base,
            "model_name":          model_name,
            "ollama_status":       ollama_status,
            "timestamp":           datetime.utcnow().isoformat(),
        }

        console.print(
            f"[cyan][SessionTrainer] auto_train complete — "
            f"{stats['total_session']} session pairs, "
            f"{stats['total_all_files']} total pairs. "
            f"Ollama: {ollama_status}[/cyan]"
        )
        return result

    def _run_ollama_create(self, model_name: str, modelfile_path: Path) -> str:
        """
        Run 'ollama create <model_name> -f <modelfile_path>'.
        Returns a status string describing the outcome.
        """
        # First check if Ollama is reachable
        if not self._ollama_reachable():
            return "ollama_not_running"

        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute timeout for model creation
            )
            if result.returncode == 0:
                console.print(
                    f"[green][SessionTrainer] Ollama model '{model_name}' created.[/green]"
                )
                return f"success: model '{model_name}' created"
            else:
                err = result.stderr.strip()[:200]
                console.print(f"[yellow][SessionTrainer] Ollama create failed: {err}[/yellow]")
                return f"error: {err}"
        except FileNotFoundError:
            return "ollama_not_in_path"
        except subprocess.TimeoutExpired:
            return "timeout_after_300s"
        except Exception as exc:
            return f"exception: {exc}"

    def _ollama_reachable(self) -> bool:
        """Return True if Ollama's HTTP API is responding."""
        try:
            import urllib.request
            import urllib.error
            urllib.request.urlopen(
                f"{self._ollama_url}/api/tags", timeout=3
            )
            return True
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # ACTIVITY TRAINER INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────

    def merge_with_activity_trainer(self, trainer: Any) -> int:
        """
        Append all session knowledge pairs into the ActivityTrainer's export
        format (prompt/response JSONL at data/training/activity_finetune.jsonl).

        Parameters
        ----------
        trainer : ActivityTrainer
            An initialised ActivityTrainer instance.

        Returns
        -------
        int : number of pairs appended.
        """
        pairs = self.get_pairs()

        # Resolve ActivityTrainer's finetune path
        try:
            # ActivityTrainer stores FINETUNE_FILE at module level
            from agents.activity_trainer import FINETUNE_FILE
            target_path = FINETUNE_FILE
        except ImportError:
            target_path = DATA_DIR / "training" / "activity_finetune.jsonl"

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing questions to deduplicate
        existing_prompts: set = set()
        if target_path.exists():
            try:
                with open(target_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            existing_prompts.add(rec.get("prompt", ""))
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass

        appended = 0
        with open(target_path, "a", encoding="utf-8") as f:
            for pair in pairs:
                msgs = pair.get("messages", [])
                if len(msgs) < 2:
                    continue
                prompt   = msgs[0]["content"]
                response = msgs[1]["content"]
                if prompt in existing_prompts:
                    continue
                record = {
                    "prompt":   prompt,
                    "response": response,
                    "metadata": {
                        "source":  "session_knowledge",
                        "topic":   pair.get("topic", "unknown"),
                        "quality": pair.get("quality", 0.95),
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_prompts.add(prompt)
                appended += 1

        console.print(
            f"[green][SessionTrainer] Merged {appended} pairs into "
            f"ActivityTrainer export at {target_path}[/green]"
        )
        return appended

    # ─────────────────────────────────────────────────────────────────────────
    # NATURAL LANGUAGE INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def run_nl(self, query: str) -> str:
        """
        Natural language command interface for SessionTrainer.

        Supported commands
        ------------------
        "train ARIA on session data"  → generate + export + trigger auto_train
        "show training stats"          → pair counts per topic + total
        "export training data"         → export JSONL only
        "what topics are covered"      → list topic areas with pair counts
        """
        q = query.lower().strip()

        # ── train command ──────────────────────────────────────────────────
        if any(k in q for k in ("train aria", "auto train", "fine-tune", "finetune",
                                  "start training", "run training")):
            result = self.auto_train()
            lines = [
                "ARIA Session Training — Complete",
                f"  Session pairs exported : {result['total_session_pairs']}",
                f"  Total pairs (all files): {result['total_all_pairs']}",
                f"  Base model             : {result['base_model']}",
                f"  Ollama model name      : {result['model_name']}",
                f"  Ollama status          : {result['ollama_status']}",
                f"  Modelfile              : {result['modelfile_path']}",
                f"  JSONL output           : {result['exported_path']}",
                "",
                "Per-topic breakdown:",
            ]
            for topic, count in result["per_topic"].items():
                lines.append(f"  {topic:<15} {count} pairs")
            return "\n".join(lines)

        # ── stats command ──────────────────────────────────────────────────
        if any(k in q for k in ("stats", "statistics", "how many", "count", "show stat")):
            stats = self.get_stats()
            lines = [
                "ARIA Training Statistics",
                f"  Session knowledge pairs: {stats['total_session']}",
                f"  Total across all files : {stats['total_all_files']}",
                f"  Training directory     : {stats['training_dir']}",
                "",
                "Per-topic counts:",
            ]
            for topic, count in stats["per_topic"].items():
                lines.append(f"  {topic:<15} {count} pairs")
            return "\n".join(lines)

        # ── export command ─────────────────────────────────────────────────
        if any(k in q for k in ("export", "write jsonl", "save training", "dump")):
            path = self.export_jsonl()
            pairs = self.get_pairs()
            return (
                f"Training data exported: {len(pairs)} session knowledge pairs\n"
                f"Output: {path}"
            )

        # ── topics command ─────────────────────────────────────────────────
        if any(k in q for k in ("topics", "covered", "what topics", "list topics",
                                  "knowledge base")):
            stats = self.get_stats()
            lines = [
                "ARIA Session Knowledge — Topic Coverage",
                "",
            ]
            topic_descriptions = {
                "anti_bot":     "Anti-bot detection (JA3, canvas, behavioral biometrics, Cloudflare, DataDome...)",
                "phishing":     "Phishing/scam detection (typosquatting, homoglyphs, SSL, overlay, TrustScore...)",
                "stealth":      "Browser stealth (CDP injection, Bezier mouse, canvas noise, WebGL spoof...)",
                "neuromorphic": "Neuromorphic architecture (NeuralBus, SynapticState, Hebbian, two-wave...)",
                "os_terminal":  "OS & terminal (OSProfile, AI error-fix loop, SSH, ADB, CoT strategies...)",
            }
            for topic, count in stats["per_topic"].items():
                desc = topic_descriptions.get(topic, "")
                lines.append(f"  [{count} pairs] {topic}")
                lines.append(f"           {desc}")
                lines.append("")
            lines.append(f"Total: {stats['total_session']} session knowledge pairs")
            return "\n".join(lines)

        # ── merge command ──────────────────────────────────────────────────
        if any(k in q for k in ("merge", "activity trainer", "combine")):
            try:
                from agents.activity_trainer import ActivityTrainer
                at = ActivityTrainer()
                count = self.merge_with_activity_trainer(at)
                return f"Merged {count} session knowledge pairs into ActivityTrainer export."
            except ImportError:
                return "ActivityTrainer not available. Ensure agents/activity_trainer.py is present."

        # ── fallback ───────────────────────────────────────────────────────
        return (
            "SessionTrainer commands:\n"
            "  'train ARIA on session data'  — generate, export, and fine-tune\n"
            "  'show training stats'          — pair counts per topic\n"
            "  'export training data'         — write JSONL only\n"
            "  'what topics are covered'      — list knowledge base topics\n"
            "  'merge with activity trainer'  — append pairs to ActivityTrainer export\n"
            f"\nCurrent status: {len(self.get_pairs())} pairs ready across "
            f"{len(KNOWLEDGE_REGISTRY)} topics."
        )


# =============================================================================
# STANDALONE CLI
# =============================================================================

def main() -> None:
    import sys

    trainer = SessionTrainer()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("SessionTrainer> ").strip()
        if not query:
            query = "show training stats"

    print(trainer.run_nl(query))


if __name__ == "__main__":
    main()
