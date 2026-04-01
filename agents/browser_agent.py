"""
ARIA Browser Agent
Full Selenium-based browser automation agent with natural language instruction parsing.
Auto-installs correct ChromeDriver via webdriver-manager.

Stealth mode patches 23 known Selenium/Chrome fingerprint leaks:
  - Removes navigator.webdriver flag
  - Patches navigator.plugins, navigator.languages, chrome runtime
  - Injects realistic screen/window dimensions
  - Adds Bezier-curve human-like mouse movement
  - Randomised typing speed (15-80 ms/char jitter)
  - Rotates user-agent from a pool of real browser signatures
"""

import os
import re
import time
import math
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (
        WebDriverException,
        NoSuchElementException,
        TimeoutException,
        StaleElementReferenceException,
    )
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WDM_AVAILABLE = True
except ImportError:
    WDM_AVAILABLE = False


# ── Stealth JS injected on every page load ───────────────────────────────────
# Patches the 23 most-fingerprinted browser properties
_STEALTH_JS = """
// 1. Remove webdriver flag
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

// 2. Fake plugins (real Chrome has 3 default plugins)
Object.defineProperty(navigator, 'plugins', {
  get: () => {
    const p = [
      {name:'Chrome PDF Plugin', filename:'internal-pdf-viewer', description:'Portable Document Format'},
      {name:'Chrome PDF Viewer', filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai', description:''},
      {name:'Native Client', filename:'internal-nacl-plugin', description:''},
    ];
    p.__proto__ = PluginArray.prototype;
    return p;
  }
});

// 3. Language spoofing
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});

// 4. Chrome runtime (headless Chrome lacks this)
window.chrome = {
  runtime: {
    connect: function() {},
    sendMessage: function() {},
    onMessage: {addListener: function() {}}
  },
  loadTimes: function() {},
  csi: function() {}
};

// 5. Permissions API — avoid "Notification" prompt fingerprint
const origQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) =>
  parameters.name === 'notifications'
    ? Promise.resolve({state: Notification.permission})
    : origQuery(parameters);

// 6. Canvas noise — add imperceptible pixel noise to defeat canvas fingerprinting
(function() {
  const origToDataURL = HTMLCanvasElement.prototype.toDataURL;
  HTMLCanvasElement.prototype.toDataURL = function(type) {
    const ctx = this.getContext('2d');
    if (ctx) {
      const imageData = ctx.getImageData(0, 0, this.width, this.height);
      for (let i = 0; i < imageData.data.length; i += 4) {
        imageData.data[i]   = imageData.data[i]   ^ (Math.random() * 3 | 0);
        imageData.data[i+1] = imageData.data[i+1] ^ (Math.random() * 3 | 0);
      }
      ctx.putImageData(imageData, 0, 0);
    }
    return origToDataURL.apply(this, arguments);
  };
})();

// 7. WebGL vendor/renderer spoofing
(function() {
  const getParameter = WebGLRenderingContext.prototype.getParameter;
  WebGLRenderingContext.prototype.getParameter = function(param) {
    if (param === 37445) return 'Intel Inc.';        // UNMASKED_VENDOR
    if (param === 37446) return 'Intel Iris OpenGL'; // UNMASKED_RENDERER
    return getParameter.call(this, param);
  };
})();

// 8. Screen dimensions (match a common real monitor)
Object.defineProperty(screen, 'width',       {get: () => 1920});
Object.defineProperty(screen, 'height',      {get: () => 1080});
Object.defineProperty(screen, 'availWidth',  {get: () => 1920});
Object.defineProperty(screen, 'availHeight', {get: () => 1040});
Object.defineProperty(screen, 'colorDepth',  {get: () => 24});

// 9. Hardware concurrency (fake 8 cores)
Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});

// 10. Device memory (fake 8 GB)
Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
"""

# Pool of real Chrome user-agent strings to rotate
_UA_POOL: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def _ok(result: str = "", data: Any = None) -> Dict:
    return {"ok": True, "result": result, "data": data}


def _err(result: str = "", data: Any = None) -> Dict:
    return {"ok": False, "result": result, "data": data}


class BrowserAgent:
    """
    Full Selenium-based browser automation agent for ARIA.
    Lazily initializes Chrome on first use and auto-restarts on crash.

    Stealth mode (enabled by default) patches 10 fingerprinting vectors,
    uses Bezier-curve mouse movement, and rotates user-agent strings.
    """

    SEARCH_URLS = {
        "google":     "https://www.google.com/search?q=",
        "bing":       "https://www.bing.com/search?q=",
        "duckduckgo": "https://duckduckgo.com/?q=",
    }

    def __init__(self, headless: bool = False, timeout: int = 15, stealth: bool = True):
        self.headless = headless
        self.timeout  = timeout
        self.stealth  = stealth
        self._driver: Optional[Any] = None
        self._ua      = random.choice(_UA_POOL)
        self._ensure_deps()

    # ------------------------------------------------------------------
    # Dependency bootstrap
    # ------------------------------------------------------------------

    def _ensure_deps(self):
        """Auto-install selenium and webdriver-manager if missing."""
        global SELENIUM_AVAILABLE, WDM_AVAILABLE
        missing = []
        if not SELENIUM_AVAILABLE:
            missing.append("selenium")
        if not WDM_AVAILABLE:
            missing.append("webdriver-manager")
        if missing:
            import subprocess, sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing
            )
            # Re-import after install
            try:
                import importlib
                importlib.invalidate_caches()
                from selenium import webdriver  # noqa: F401
                SELENIUM_AVAILABLE = True
            except ImportError:
                pass
            try:
                from webdriver_manager.chrome import ChromeDriverManager  # noqa: F401
                WDM_AVAILABLE = True
            except ImportError:
                pass

    # ------------------------------------------------------------------
    # Driver lifecycle
    # ------------------------------------------------------------------

    def _build_options(self) -> "ChromeOptions":
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_argument(f"--user-agent={self._ua}")

        if self.stealth:
            # Disable WebRTC (prevents IP leak)
            opts.add_argument("--disable-webrtc")
            # Randomise window size (common bot tells: exact 1920x1080)
            w = random.randint(1280, 1920)
            h = random.randint(768,  1080)
            opts.add_argument(f"--window-size={w},{h}")
            # Disable AutoFill save prompts (reduce detectability)
            prefs = {
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
                "profile.default_content_setting_values.notifications": 2,
            }
            opts.add_experimental_option("prefs", prefs)
            # Suppress "Chrome is being controlled" infobar
            opts.add_argument("--disable-infobars")

        return opts

    def _init_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        opts    = self._build_options()
        service = Service(ChromeDriverManager().install())
        driver  = webdriver.Chrome(service=service, options=opts)
        driver.set_page_load_timeout(self.timeout)
        driver.implicitly_wait(5)

        if self.stealth:
            # Use full 23-vector stealth engine (replaces the 10-patch version)
            try:
                from agents.browser_stealth import StealthEngine
                self._stealth_engine = StealthEngine(driver)
                self._stealth_engine.apply_all()
            except ImportError:
                # Fallback to basic 10-patch if stealth module missing
                driver.execute_cdp_cmd(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": _STEALTH_JS},
                )
                self._stealth_engine = None
        else:
            self._stealth_engine = None

        return driver

    # ------------------------------------------------------------------
    # Human-like mouse movement (Bezier curve)
    # ------------------------------------------------------------------

    def _bezier_points(
        self,
        start: Tuple[int, int],
        end:   Tuple[int, int],
        steps: int = 20,
    ) -> List[Tuple[int, int]]:
        """Generate a cubic Bezier curve between two points with random control points."""
        x0, y0 = start
        x1, y1 = end
        # Two random control points for natural curve
        cx1 = x0 + random.randint(-80, 80)
        cy1 = y0 + random.randint(-80, 80)
        cx2 = x1 + random.randint(-80, 80)
        cy2 = y1 + random.randint(-80, 80)
        points = []
        for i in range(steps + 1):
            t  = i / steps
            mt = 1 - t
            x  = mt**3*x0 + 3*mt**2*t*cx1 + 3*mt*t**2*cx2 + t**3*x1
            y  = mt**3*y0 + 3*mt**2*t*cy1 + 3*mt*t**2*cy2 + t**3*y1
            points.append((int(x), int(y)))
        return points

    def _human_move_to(self, element) -> None:
        """Move mouse to element via Bezier curve, simulating human hand trajectory."""
        if not self.stealth or not SELENIUM_AVAILABLE:
            return
        try:
            from selenium.webdriver.common.action_chains import ActionChains
            # Get current mouse position (approximate via JS)
            mx = self.driver.execute_script("return window.innerWidth  / 2;") or 400
            my = self.driver.execute_script("return window.innerHeight / 2;") or 300
            rect = self.driver.execute_script(
                "const r = arguments[0].getBoundingClientRect();"
                "return {x: r.left + r.width/2, y: r.top + r.height/2};",
                element,
            )
            tx, ty = int(rect["x"]), int(rect["y"])
            points = self._bezier_points((int(mx), int(my)), (tx, ty), steps=random.randint(15, 30))
            actions = ActionChains(self.driver)
            for px, py in points:
                actions.move_by_offset(px - int(mx), py - int(my))
                mx, my = px, py
                actions.pause(random.uniform(0.005, 0.025))
            actions.perform()
        except Exception:
            pass   # Non-critical — fall back to normal click

    def _human_type(self, element, text: str) -> None:
        """Type text with randomized per-character delay (15–80 ms)."""
        for ch in text:
            element.send_keys(ch)
            time.sleep(random.uniform(0.015, 0.08))

    @property
    def driver(self):
        """Lazily initialize and return the Chrome driver."""
        if self._driver is None:
            self._driver = self._init_driver()
        return self._driver

    def _restart_driver(self):
        """Attempt to quit the broken driver and start fresh."""
        try:
            if self._driver:
                self._driver.quit()
        except Exception:
            pass
        self._driver = None
        self._driver = self._init_driver()

    def _safe(self, fn, *args, **kwargs) -> Dict:
        """Execute a callable; auto-restart driver on WebDriverException crash."""
        try:
            return fn(*args, **kwargs)
        except WebDriverException as exc:
            msg = str(exc).splitlines()[0]
            # Try restart once
            try:
                self._restart_driver()
                return fn(*args, **kwargs)
            except Exception as exc2:
                return _err(f"Driver error (restarted): {exc2}", data=msg)
        except Exception as exc:
            return _err(str(exc), data=traceback.format_exc())

    # ------------------------------------------------------------------
    # Core navigation
    # ------------------------------------------------------------------

    def open_url(self, url: str) -> Dict:
        """
        Navigate to a URL with full stealth pipeline:
        - Applies 23-vector stealth patches on page load
        - Detects WAF blocks (Cloudflare, DataDome, Akamai, Imperva)
        - Attempts auto-bypass before returning content
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        def _do():
            self.driver.get(url)

            # Post-load stealth hooks
            se = getattr(self, "_stealth_engine", None)
            if se:
                se.on_page_load()
                # Check for WAF block and try bypass
                waf = se.detect_block()
                if waf:
                    se.bypass_page(url)

            return _ok(
                f"Opened: {url}",
                data={
                    "title": self.driver.title,
                    "url":   self.driver.current_url,
                    "waf":   (se.detect_block() if se else None),
                },
            )

        return self._safe(_do)

    def read_url(self, url: str) -> Dict:
        """
        Read a URL's full text content using the stealth extraction pipeline.
        Falls back through: requests → Selenium → Selenium+bypass → Google Cache.
        Better than open_url when you need the text and don't need to interact.
        """
        se = getattr(self, "_stealth_engine", None)
        if se:
            content = se.read_website(url)
            return _ok("Content extracted", data={"text": content, "url": url})

        # Fallback: open normally and get text
        nav = self.open_url(url)
        if nav["ok"]:
            return self.get_page_text()
        return nav

    def back(self) -> Dict:
        def _do():
            self.driver.back()
            return _ok("Navigated back", data={"url": self.driver.current_url})
        return self._safe(_do)

    def forward(self) -> Dict:
        def _do():
            self.driver.forward()
            return _ok("Navigated forward", data={"url": self.driver.current_url})
        return self._safe(_do)

    def refresh(self) -> Dict:
        def _do():
            self.driver.refresh()
            return _ok("Page refreshed", data={"url": self.driver.current_url})
        return self._safe(_do)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_web(self, query: str, engine: str = "google") -> Dict:
        """Search on Google, Bing, or DuckDuckGo."""
        engine = engine.lower()
        base = self.SEARCH_URLS.get(engine)
        if not base:
            return _err(f"Unknown engine: {engine}. Choose google/bing/duckduckgo")
        from urllib.parse import quote_plus
        url = base + quote_plus(query)
        result = self.open_url(url)
        if result["ok"]:
            result["result"] = f"Searched '{query}' on {engine}"
        return result

    # ------------------------------------------------------------------
    # Page info
    # ------------------------------------------------------------------

    def get_page_text(self) -> Dict:
        """Extract all visible text from the current page."""
        def _do():
            body = self.driver.find_element(By.TAG_NAME, "body")
            text = body.text
            return _ok("Extracted page text", data={"text": text, "length": len(text)})
        return self._safe(_do)

    def get_page_title(self) -> Dict:
        def _do():
            title = self.driver.title
            return _ok(title, data={"title": title, "url": self.driver.current_url})
        return self._safe(_do)

    def get_page_html(self) -> Dict:
        """Return the full HTML source of the current page."""
        def _do():
            html = self.driver.page_source
            return _ok("Got page HTML", data={"html": html, "length": len(html)})
        return self._safe(_do)

    def get_links(self) -> Dict:
        """Extract all anchor links from the current page."""
        def _do():
            anchors = self.driver.find_elements(By.TAG_NAME, "a")
            links = []
            for a in anchors:
                href = a.get_attribute("href")
                text = a.text.strip()
                if href:
                    links.append({"text": text, "href": href})
            return _ok(f"Found {len(links)} links", data={"links": links})
        return self._safe(_do)

    # ------------------------------------------------------------------
    # Scam / phishing detection (integrated)
    # ------------------------------------------------------------------

    def scan_for_scams(self, url: str, engine=None) -> Dict:
        """
        Navigate to a URL and run ARIA's full 8-layer scam detection before
        showing the user any content.

        Returns {"ok": bool, "result": verdict_str, "data": report_dict}
        """
        def _do():
            from agents.scam_detector import ScamDetectorAgent
            detector = ScamDetectorAgent(engine=engine, headless_browser=self)
            report   = detector.scan(url)
            return _ok(report.summary(), data=report.to_dict())
        return self._safe(_do)

    def safe_open(self, url: str, engine=None) -> Dict:
        """
        Open a URL only after passing a scam check.
        If DANGEROUS, abort and return the threat report.
        If SUSPICIOUS, open but prepend a warning.
        If SAFE, open normally.
        """
        scan = self.scan_for_scams(url, engine=engine)
        if not scan["ok"]:
            return scan

        report = scan.get("data", {})
        verdict = report.get("verdict", "UNKNOWN")

        if verdict == "DANGEROUS":
            return _err(
                f"⛔ BLOCKED — Site flagged as DANGEROUS (trust score: {report.get('trust_score', 0):.0f}/100).\n"
                + scan["result"],
                data=report,
            )

        # Open the page
        nav = self.open_url(url)
        if verdict == "SUSPICIOUS":
            nav["result"] = (
                f"⚠️ WARNING — This site is SUSPICIOUS (trust score: {report.get('trust_score', 0):.0f}/100). "
                "Proceed with caution.\n\n" + nav.get("result", "")
            )
        return nav

    # ------------------------------------------------------------------
    # Element interaction
    # ------------------------------------------------------------------

    def _find_element(self, selector_or_text: str):
        """
        Try to find an element by:
        1. CSS selector
        2. XPath
        3. Visible link/button text
        4. Partial text match via XPath
        """
        wait = WebDriverWait(self.driver, self.timeout)

        # 1. Try as CSS selector
        try:
            el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector_or_text)))
            return el
        except Exception:
            pass

        # 2. Try as XPath
        if selector_or_text.startswith(("//", "(")):
            try:
                el = wait.until(EC.presence_of_element_located((By.XPATH, selector_or_text)))
                return el
            except Exception:
                pass

        # 3. Exact link text
        try:
            el = self.driver.find_element(By.LINK_TEXT, selector_or_text)
            return el
        except Exception:
            pass

        # 4. Partial link text
        try:
            el = self.driver.find_element(By.PARTIAL_LINK_TEXT, selector_or_text)
            return el
        except Exception:
            pass

        # 5. XPath by text content of any element
        xpath = f"//*[normalize-space(text())='{selector_or_text}']"
        try:
            el = self.driver.find_element(By.XPATH, xpath)
            return el
        except Exception:
            pass

        # 6. XPath partial text
        xpath_partial = f"//*[contains(normalize-space(text()),'{selector_or_text}')]"
        try:
            el = self.driver.find_element(By.XPATH, xpath_partial)
            return el
        except Exception:
            pass

        raise NoSuchElementException(f"Element not found: {selector_or_text}")

    def click_element(self, selector_or_text: str) -> Dict:
        """Click an element by CSS selector, XPath, or visible text (human-like movement)."""
        def _do():
            el = self._find_element(selector_or_text)
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(random.uniform(0.2, 0.5))
            self._human_move_to(el)
            time.sleep(random.uniform(0.05, 0.2))
            try:
                el.click()
            except Exception:
                self.driver.execute_script("arguments[0].click();", el)
            return _ok(f"Clicked: {selector_or_text}")
        return self._safe(_do)

    def fill_form(self, field_selector: str, value: str) -> Dict:
        """Fill an input field with human-like typing speed."""
        def _do():
            el = self._find_element(field_selector)
            self._human_move_to(el)
            el.click()
            time.sleep(random.uniform(0.1, 0.3))
            el.clear()
            if self.stealth:
                self._human_type(el, value)
            else:
                el.send_keys(value)
            return _ok(f"Filled '{field_selector}' with '{value}'")
        return self._safe(_do)

    def submit_form(self, selector: Optional[str] = None) -> Dict:
        """Submit a form. If selector given, submit that element's form or press Enter on it."""
        def _do():
            if selector:
                el = self._find_element(selector)
                try:
                    el.submit()
                except Exception:
                    el.send_keys(Keys.RETURN)
                return _ok(f"Submitted form via: {selector}")
            else:
                # Try to find a submit button
                for s in ["[type='submit']", "button[type='submit']", "input[type='submit']"]:
                    try:
                        btn = self.driver.find_element(By.CSS_SELECTOR, s)
                        btn.click()
                        return _ok("Submitted form via submit button")
                    except Exception:
                        pass
                # Fallback: press Enter on focused element
                active = self.driver.switch_to.active_element
                active.send_keys(Keys.RETURN)
                return _ok("Submitted form via Enter key")
        return self._safe(_do)

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    def scroll(self, direction: str = "down", amount: int = 3) -> Dict:
        """Scroll the page up or down by `amount` viewport heights."""
        def _do():
            direction_lower = direction.lower()
            sign = "-" if direction_lower == "up" else ""
            pixels = amount * 400
            self.driver.execute_script(f"window.scrollBy(0, {sign}{pixels});")
            return _ok(f"Scrolled {direction} by {pixels}px")
        return self._safe(_do)

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def take_screenshot(self, path: str = "") -> Dict:
        """Take a screenshot. Saves to path (auto-generated if empty)."""
        def _do():
            if not path:
                ts = int(time.time())
                save_path = os.path.join(os.getcwd(), f"aria_screenshot_{ts}.png")
            else:
                save_path = path
            self.driver.save_screenshot(save_path)
            return _ok(f"Screenshot saved: {save_path}", data={"path": save_path})
        return self._safe(_do)

    # ------------------------------------------------------------------
    # JavaScript
    # ------------------------------------------------------------------

    def execute_js(self, code: str) -> Dict:
        """Execute arbitrary JavaScript in the current page context."""
        def _do():
            result = self.driver.execute_script(code)
            return _ok("JS executed", data={"return_value": result})
        return self._safe(_do)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> Dict:
        """Close the browser."""
        try:
            if self._driver:
                self._driver.quit()
                self._driver = None
            return _ok("Browser closed")
        except Exception as exc:
            self._driver = None
            return _err(f"Error closing browser: {exc}")

    # ------------------------------------------------------------------
    # Natural language instruction parser
    # ------------------------------------------------------------------

    def run_nl(self, instruction: str) -> Dict:
        """
        Parse and execute a natural language browser instruction.

        Supported patterns (case-insensitive):
          - "go to <url>" / "open <url>" / "navigate to <url>" / "visit <url>"
          - "search for <query>" / "search <query> on <engine>"
          - "click <element>" / "click on <element>"
          - "fill <field> with <value>" / "type <value> in <field>"
          - "submit" / "submit form" / "press enter"
          - "scroll down [N times]" / "scroll up [N times]"
          - "screenshot" / "take screenshot"
          - "get text" / "get page text"
          - "get title" / "get page title"
          - "get links"
          - "back" / "go back"
          - "forward" / "go forward"
          - "refresh" / "reload"
          - "run js <code>" / "execute <code>"
          - "close" / "close browser"
        """
        raw = instruction.strip()
        low = raw.lower()

        # --- go to / open / navigate / visit ---
        m = re.match(
            r"(?:go to|open|navigate to|visit|load)\s+(.+)",
            low,
        )
        if m:
            url = raw[m.start(1):].strip()
            return self.open_url(url)

        # --- search ---
        m = re.match(
            r"search(?:\s+for)?\s+(.+?)(?:\s+on\s+(google|bing|duckduckgo))?$",
            low,
        )
        if m:
            query = raw[m.start(1): m.start(1) + len(m.group(1))].strip()
            engine = (m.group(2) or "google").lower()
            return self.search_web(query, engine)

        # --- click ---
        m = re.match(r"click(?:\s+on)?\s+(.+)", raw, re.IGNORECASE)
        if m:
            target = m.group(1).strip()
            return self.click_element(target)

        # --- fill / type ---
        m = re.match(
            r"(?:fill|type|enter|input)\s+(.+?)\s+(?:with|in|into|as)\s+(.+)",
            raw,
            re.IGNORECASE,
        )
        if m:
            field = m.group(1).strip()
            value = m.group(2).strip()
            return self.fill_form(field, value)

        # --- fill in field with value (alternative phrasing) ---
        m = re.match(
            r"(?:fill|set)\s+(?:the\s+)?(.+?)\s+(?:field|input|box)\s+(?:to|with)\s+(.+)",
            raw,
            re.IGNORECASE,
        )
        if m:
            field = m.group(1).strip()
            value = m.group(2).strip()
            return self.fill_form(field, value)

        # --- submit ---
        if re.match(r"submit(?:\s+form)?|press\s+enter", low):
            return self.submit_form()

        # --- scroll ---
        m = re.match(r"scroll\s+(down|up)(?:\s+(\d+)(?:\s+times?)?)?", low)
        if m:
            direction = m.group(1)
            amount = int(m.group(2)) if m.group(2) else 3
            return self.scroll(direction, amount)

        # --- screenshot ---
        if re.match(r"(?:take\s+)?screenshot", low):
            return self.take_screenshot()

        # --- get text ---
        if re.match(r"get\s+(?:page\s+)?text|extract\s+text", low):
            return self.get_page_text()

        # --- get title ---
        if re.match(r"get\s+(?:page\s+)?title", low):
            return self.get_page_title()

        # --- get links ---
        if re.match(r"get\s+(?:all\s+)?links", low):
            return self.get_links()

        # --- back ---
        if re.match(r"(?:go\s+)?back", low):
            return self.back()

        # --- forward ---
        if re.match(r"(?:go\s+)?forward", low):
            return self.forward()

        # --- refresh / reload ---
        if re.match(r"refresh|reload", low):
            return self.refresh()

        # --- execute JS ---
        m = re.match(r"(?:run|execute)\s+(?:js|javascript)?\s+(.+)", raw, re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            return self.execute_js(code)

        # --- close ---
        if re.match(r"close(?:\s+browser)?", low):
            return self.close()

        return _err(f"Instruction not understood: '{raw}'")


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    agent = BrowserAgent(headless=False)

    print(agent.open_url("https://www.google.com"))
    print(agent.get_page_title())
    print(agent.search_web("ARIA AI assistant Python"))
    print(agent.get_page_text())
    print(agent.take_screenshot())
    print(agent.run_nl("go to https://duckduckgo.com"))
    print(agent.run_nl("search for Ollama local LLM"))
    print(agent.run_nl("scroll down 2"))
    print(agent.close())
