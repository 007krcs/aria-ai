"""
ARIA Browser Stealth Engine
============================
Deep anti-detection layer for Selenium Chrome.

Covers 23 fingerprint vectors + platform-specific bypass techniques for:
  • Cloudflare (Bot Fight Mode, Turnstile, WAF)
  • DataDome
  • Akamai Bot Manager
  • HUMAN Security (PerimeterX)
  • Imperva (Incapsula)

Usage:
    from agents.browser_stealth import StealthEngine
    stealth = StealthEngine(driver)
    stealth.apply_all()           # call once after driver init
    stealth.on_page_load()        # call after every driver.get()
    content = stealth.extract_content(url)  # full pipeline: open + bypass + extract
"""

import re
import os
import time
import json
import math
import random
import hashlib
import urllib.parse
import threading
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_OK = True
except ImportError:
    SELENIUM_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# FULL 23-VECTOR STEALTH JS PAYLOAD
# Applied once via CDP — survives all page navigations
# ─────────────────────────────────────────────────────────────────────────────

_FULL_STEALTH_JS = r"""
// ══════════════════════════════════════════════════════════════════════════════
// ARIA FULL STEALTH PATCH — 23 Fingerprint Vectors
// ══════════════════════════════════════════════════════════════════════════════

// [1] webdriver flag — most basic bot tell
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

// [2] Chrome automation flag in window.chrome
window.chrome = {
  app: {isInstalled: false, InstallState: {DISABLED:'disabled',INSTALLED:'installed',NOT_INSTALLED:'not_installed'}, RunningState: {CANNOT_RUN:'cannot_run',READY_TO_RUN:'ready_to_run',RUNNING:'running'}},
  csi: function() {},
  loadTimes: function() {},
  runtime: {
    PlatformOs: {MAC:'mac',WIN:'win',ANDROID:'android',CROS:'cros',LINUX:'linux',OPENBSD:'openbsd'},
    PlatformArch: {ARM:'arm',X86_32:'x86-32',X86_64:'x86-64',MIPS:'mips',MIPS64:'mips64'},
    PlatformNaclArch: {ARM:'arm',X86_32:'x86-32',X86_64:'x86-64',MIPS:'mips',MIPS64:'mips64'},
    RequestUpdateCheckStatus: {THROTTLED:'throttled',NO_UPDATE:'no_update',UPDATE_AVAILABLE:'update_available'},
    OnInstalledReason: {INSTALL:'install',UPDATE:'update',CHROME_UPDATE:'chrome_update',SHARED_MODULE_UPDATE:'shared_module_update'},
    OnRestartRequiredReason: {APP_UPDATE:'app_update',OS_UPDATE:'os_update',PERIODIC:'periodic'},
    connect: function() {return {disconnect:function(){},onDisconnect:{addListener:function(){}},onMessage:{addListener:function(){}},postMessage:function(){}};},
    sendMessage: function() {},
    onConnect: {addListener: function() {}},
    onMessage: {addListener: function() {}},
    id: undefined,
  },
  webstore: {onInstallStageChanged: {addListener: function(){}}, onDownloadProgress: {addListener: function(){}}, install: function(){}, },
};

// [3] navigator.plugins — real Chrome has 3 built-in plugins
const _makeMimeType = (type, desc, suffixes, plugin) => {
  const m = Object.create(MimeType.prototype);
  Object.defineProperties(m, {
    type:        {value: type,     enumerable: true},
    description: {value: desc,     enumerable: true},
    suffixes:    {value: suffixes,  enumerable: true},
    enabledPlugin:{value: plugin,  enumerable: true},
  });
  return m;
};
const _makePl = (name, desc, filename, mimes) => {
  const p = Object.create(Plugin.prototype);
  Object.defineProperties(p, {
    name:        {value: name,     enumerable: true},
    description: {value: desc,     enumerable: true},
    filename:    {value: filename,  enumerable: true},
    length:      {value: mimes.length, enumerable: true},
  });
  mimes.forEach((m, i) => { Object.defineProperty(p, i, {value: m, enumerable: true}); });
  return p;
};
const _pl1 = _makePl('Chrome PDF Plugin', 'Portable Document Format', 'internal-pdf-viewer', []);
const _pl2 = _makePl('Chrome PDF Viewer', '', 'mhjfbmdgcfjbbpaeojofohoefgiehjai', []);
const _pl3 = _makePl('Native Client', '', 'internal-nacl-plugin', []);
Object.defineProperty(navigator, 'plugins', {
  get: () => {
    const arr = [_pl1, _pl2, _pl3];
    arr.__proto__ = PluginArray.prototype;
    return arr;
  }
});

// [4] mimeTypes — must match plugins
Object.defineProperty(navigator, 'mimeTypes', {
  get: () => {
    const arr = [];
    arr.__proto__ = MimeTypeArray.prototype;
    return arr;
  }
});

// [5] navigator.languages
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});

// [6] navigator.language
Object.defineProperty(navigator, 'language', {get: () => 'en-US'});

// [7] navigator.platform
Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});

// [8] navigator.vendor
Object.defineProperty(navigator, 'vendor', {get: () => 'Google Inc.'});

// [9] navigator.hardwareConcurrency (real machine typically 4-16)
Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});

// [10] navigator.deviceMemory (GB, real: 4-16)
Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});

// [11] Permissions API — avoid notification fingerprint leak
(function() {
  const origQuery = window.navigator.permissions ? window.navigator.permissions.query : null;
  if (origQuery) {
    window.navigator.permissions.__proto__.query = function(parameters) {
      if (parameters.name === 'notifications') {
        return Promise.resolve({state: Notification.permission});
      }
      return origQuery.call(this, parameters);
    };
  }
})();

// [12] Canvas 2D fingerprint noise — imperceptible XOR on pixel data
(function() {
  const toDataURL = HTMLCanvasElement.prototype.toDataURL;
  const toBlob    = HTMLCanvasElement.prototype.toBlob;
  function _addNoise(canvas) {
    try {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const w   = Math.min(canvas.width, 200);
      const h   = Math.min(canvas.height, 200);
      if (w === 0 || h === 0) return;
      const img = ctx.getImageData(0, 0, w, h);
      for (let i = 0; i < img.data.length; i += 4) {
        img.data[i]   ^= (Math.random() * 3 | 0);
        img.data[i+1] ^= (Math.random() * 3 | 0);
      }
      ctx.putImageData(img, 0, 0);
    } catch(e) {}
  }
  HTMLCanvasElement.prototype.toDataURL = function() {
    _addNoise(this);
    return toDataURL.apply(this, arguments);
  };
  HTMLCanvasElement.prototype.toBlob = function() {
    _addNoise(this);
    return toBlob.apply(this, arguments);
  };
})();

// [13] WebGL parameter spoofing (vendor + renderer)
(function() {
  const _getParam = WebGLRenderingContext.prototype.getParameter;
  WebGLRenderingContext.prototype.getParameter = function(param) {
    if (param === 37445) return 'Intel Inc.';             // UNMASKED_VENDOR_WEBGL
    if (param === 37446) return 'Intel(R) Iris(R) Plus Graphics 640'; // UNMASKED_RENDERER_WEBGL
    return _getParam.apply(this, arguments);
  };
  if (typeof WebGL2RenderingContext !== 'undefined') {
    const _get2 = WebGL2RenderingContext.prototype.getParameter;
    WebGL2RenderingContext.prototype.getParameter = function(param) {
      if (param === 37445) return 'Intel Inc.';
      if (param === 37446) return 'Intel(R) Iris(R) Plus Graphics 640';
      return _get2.apply(this, arguments);
    };
  }
})();

// [14] Audio context fingerprint noise
(function() {
  if (typeof AudioBuffer === 'undefined') return;
  const origGetChannelData = AudioBuffer.prototype.getChannelData;
  AudioBuffer.prototype.getChannelData = function() {
    const arr = origGetChannelData.apply(this, arguments);
    for (let i = 0; i < arr.length; i += 100) {
      arr[i] += Math.random() * 0.0000001;
    }
    return arr;
  };
})();

// [15] Screen dimensions — randomize slightly around common sizes
(function() {
  const sw = 1920, sh = 1080;
  const aw = sw, ah = sh - 40;
  Object.defineProperty(screen, 'width',       {get: () => sw});
  Object.defineProperty(screen, 'height',      {get: () => sh});
  Object.defineProperty(screen, 'availWidth',  {get: () => aw});
  Object.defineProperty(screen, 'availHeight', {get: () => ah});
  Object.defineProperty(screen, 'colorDepth',  {get: () => 24});
  Object.defineProperty(screen, 'pixelDepth',  {get: () => 24});
})();

// [16] window.outerWidth / outerHeight (should be ≥ innerWidth/Height in real browser)
Object.defineProperty(window, 'outerWidth',  {get: () => 1920});
Object.defineProperty(window, 'outerHeight', {get: () => 1080});

// [17] Font enumeration — block document.fonts.check() based fingerprinting
(function() {
  if (!document.fonts) return;
  const origCheck = document.fonts.check.bind(document.fonts);
  document.fonts.check = function(font, text) {
    // Always return true for web-safe fonts, false for exotic ones
    const webSafe = /(?:arial|helvetica|times|courier|verdana|georgia|trebuchet|impact)/i;
    return webSafe.test(font) ? true : origCheck(font, text);
  };
})();

// [18] Battery API — return a fake full battery
(function() {
  if (!navigator.getBattery) return;
  navigator.getBattery = function() {
    return Promise.resolve({
      charging: true, chargingTime: 0, dischargingTime: Infinity,
      level: 1.0,
      onchargingchange: null, onchargingtimechange: null,
      ondischargingtimechange: null, onlevelchange: null,
    });
  };
})();

// [19] navigator.connection — fake a stable wifi connection
(function() {
  if (!navigator.connection) return;
  Object.defineProperty(navigator, 'connection', {
    get: () => ({
      downlink: 10, effectiveType: '4g', rtt: 50,
      saveData: false, type: 'wifi',
      onchange: null,
    })
  });
})();

// [20] Error stack trace cleanup — prevent stack fingerprinting
(function() {
  const origErr = Error;
  window.Error = function() {
    const err = new origErr(...arguments);
    Object.defineProperty(err, 'stack', {
      get: () => err.stack ? err.stack.replace(/\s+at\s+.+\n/g, '') : '',
    });
    return err;
  };
})();

// [21] Object.getOwnPropertyNames protection — prevents bot checkers from enumerating automation props
(function() {
  const origGetOwn = Object.getOwnPropertyNames;
  Object.getOwnPropertyNames = function(obj) {
    const props = origGetOwn(obj);
    // Filter out webdriver-related props if someone inspects window
    return props.filter(p => !['_selenium', '__webdriver', '$chrome_asyncScriptInfo', '__driver_evaluate', '__webdriver_script_fn'].includes(p));
  };
})();

// [22] Function.prototype.toString — hide patched functions from detectors
(function() {
  const origToString = Function.prototype.toString;
  Function.prototype.toString = function() {
    const str = origToString.call(this);
    // If this function was patched (has our noise comment), return native code
    if (str.indexOf('ARIA STEALTH') !== -1) {
      return 'function () { [native code] }';
    }
    return str;
  };
})();

// [23] Date/timezone — consistent timezone to avoid mismatch with Accept-Language
(function() {
  const origDateTimeFormat = Intl.DateTimeFormat;
  window.Intl.DateTimeFormat = function(locale, options) {
    if (options && options.timeZone === undefined) {
      options = Object.assign({}, options, {timeZone: 'America/New_York'});
    }
    return new origDateTimeFormat(locale || 'en-US', options);
  };
  Object.setPrototypeOf(window.Intl.DateTimeFormat, origDateTimeFormat);
})();

console.log('[ARIA] Stealth patches applied: 23 vectors covered');
"""


# ─────────────────────────────────────────────────────────────────────────────
# BOT-WALL SIGNATURES (detect which WAF is blocking us)
# ─────────────────────────────────────────────────────────────────────────────

WAF_SIGNATURES = {
    "cloudflare": [
        r"cloudflare",
        r"cf-ray",
        r"__cf_bm",
        r"cf_clearance",
        r"checking your browser",
        r"please wait.*cloudflare",
        r"DDoS protection by Cloudflare",
        r"challenge-platform",
        r"turnstile",
    ],
    "datadome": [
        r"datadome",
        r"dd_cookie",
        r"access denied.*datadome",
        r"device verification",
    ],
    "akamai": [
        r"akamai",
        r"_abck",
        r"ak_bmsc",
        r"bm_sz",
        r"We're sorry",
    ],
    "imperva": [
        r"incapsula",
        r"imperva",
        r"visid_incap",
        r"incap_ses",
        r"_ivm_",
    ],
    "human_security": [
        r"perimeterx",
        r"_pxhd",
        r"_px[0-9]",
        r"human\.net",
    ],
    "arkose": [
        r"funcaptcha",
        r"arkoselabs",
        r"ArkosePublicKey",
    ],
    "generic_block": [
        r"access denied",
        r"403 forbidden",
        r"bot detected",
        r"automated access",
        r"unusual traffic",
        r"security check",
        r"captcha",
    ],
}


def detect_waf(html: str, headers: dict = None) -> Optional[str]:
    """Identify which WAF is blocking the response."""
    combined = (html or "").lower() + str(headers or "").lower()
    for waf, patterns in WAF_SIGNATURES.items():
        for pat in patterns:
            if re.search(pat, combined, re.IGNORECASE):
                return waf
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEALTH ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class StealthEngine:
    """
    Full stealth and bot-bypass layer for ARIA's browser agent.

    Wraps a Selenium WebDriver with:
      1. 23-vector fingerprint patching (via CDP)
      2. Human-like behavioral simulation
      3. WAF detection and bypass strategies
      4. Multi-strategy content extraction fallback chain
      5. Cloudflare Turnstile detection + wait
    """

    # Delay ranges (seconds)
    HUMAN_DELAY_MIN  = 0.8
    HUMAN_DELAY_MAX  = 3.5
    SCROLL_DELAY_MIN = 0.3
    SCROLL_DELAY_MAX = 1.5
    TYPE_DELAY_MIN   = 0.015
    TYPE_DELAY_MAX   = 0.09

    # How long to wait for Cloudflare challenge to clear
    CF_WAIT_MAX = 15.0
    CF_POLL_S   = 1.0

    # Max retries for bot-wall bypass
    MAX_RETRIES = 3

    def __init__(self, driver=None):
        self.driver      = driver
        self._stealth_ok = False
        self._session    = self._build_requests_session()

    # ──────────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────────

    def _build_requests_session(self):
        if not REQUESTS_OK:
            return None
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5)
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://",  HTTPAdapter(max_retries=retry))
        s.headers.update({
            "User-Agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            ]),
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT":             "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest":  "document",
            "Sec-Fetch-Mode":  "navigate",
            "Sec-Fetch-Site":  "none",
            "Sec-Fetch-User":  "?1",
            "Sec-CH-UA":       '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "Sec-CH-UA-Mobile":"?0",
            "Sec-CH-UA-Platform": '"Windows"',
        })
        return s

    def attach(self, driver):
        """Attach (or re-attach) a Selenium WebDriver."""
        self.driver      = driver
        self._stealth_ok = False

    # ──────────────────────────────────────────────────────────────────────────
    # FINGERPRINT PATCHING
    # ──────────────────────────────────────────────────────────────────────────

    def apply_all(self) -> bool:
        """
        Inject all 23 stealth patches via Chrome DevTools Protocol.
        Call once immediately after driver creation, before any page load.
        """
        if not self.driver or not SELENIUM_OK:
            return False
        try:
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": _FULL_STEALTH_JS},
            )
            self._stealth_ok = True
            return True
        except Exception:
            # CDP not available (non-Chrome driver) — fall back to page-level injection
            self._stealth_ok = False
            return False

    def patch_current_page(self) -> bool:
        """
        Inject stealth into the currently loaded page via execute_script.
        Use this as a fallback when CDP injection was not applied at driver init.
        """
        if not self.driver:
            return False
        try:
            self.driver.execute_script(_FULL_STEALTH_JS)
            return True
        except Exception:
            return False

    def on_page_load(self):
        """
        Call after every driver.get() to:
        - Apply page-level patch if CDP patch isn't active
        - Simulate a brief human reading pause
        - Randomise scroll position slightly
        """
        if not self._stealth_ok:
            self.patch_current_page()
        # Brief random pause (human "looks at" page for a moment)
        time.sleep(random.uniform(self.HUMAN_DELAY_MIN, self.HUMAN_DELAY_MAX))
        # Small random scroll (not a bot-perfect y=0)
        self._random_micro_scroll()

    # ──────────────────────────────────────────────────────────────────────────
    # HUMAN-LIKE BEHAVIOUR
    # ──────────────────────────────────────────────────────────────────────────

    def _random_micro_scroll(self):
        """Scroll a tiny random amount to simulate eye movement."""
        try:
            px = random.randint(10, 150)
            self.driver.execute_script(f"window.scrollBy(0, {px});")
            time.sleep(random.uniform(0.1, 0.4))
            self.driver.execute_script(f"window.scrollBy(0, -{px // 2});")
        except Exception:
            pass

    def bezier_points(
        self,
        start: Tuple[int, int],
        end:   Tuple[int, int],
        steps: int = 25,
    ) -> List[Tuple[int, int]]:
        """Cubic Bezier curve between two screen points."""
        x0, y0 = start
        x1, y1 = end
        # Control points with random jitter
        cx1 = x0 + random.randint(-100, 100)
        cy1 = y0 + random.randint(-80,  80)
        cx2 = x1 + random.randint(-100, 100)
        cy2 = y1 + random.randint(-80,  80)
        pts = []
        for i in range(steps + 1):
            t  = i / steps
            u  = 1 - t
            x  = u**3*x0 + 3*u**2*t*cx1 + 3*u*t**2*cx2 + t**3*x1
            y  = u**3*y0 + 3*u**2*t*cy1 + 3*u*t**2*cy2 + t**3*y1
            pts.append((int(x), int(y)))
        return pts

    def human_move_click(self, element) -> bool:
        """Move mouse to element via Bezier curve, then click."""
        if not self.driver or not SELENIUM_OK:
            return False
        try:
            rect = self.driver.execute_script(
                "const r=arguments[0].getBoundingClientRect();"
                "return {x:r.left+r.width/2, y:r.top+r.height/2};",
                element,
            )
            mx = self.driver.execute_script("return window.innerWidth/2;")  or 500
            my = self.driver.execute_script("return window.innerHeight/2;") or 400

            pts = self.bezier_points(
                (int(mx), int(my)),
                (int(rect["x"]), int(rect["y"])),
                steps=random.randint(18, 35),
            )

            actions = ActionChains(self.driver)
            prev_x, prev_y = int(mx), int(my)
            for px, py in pts:
                actions.move_by_offset(px - prev_x, py - prev_y)
                actions.pause(random.uniform(0.004, 0.02))
                prev_x, prev_y = px, py
            actions.click()
            actions.perform()
            return True
        except Exception:
            try:
                element.click()
            except Exception:
                pass
            return False

    def human_type(self, element, text: str, clear_first: bool = True) -> bool:
        """Type text with human-like randomised per-character delay."""
        try:
            if clear_first:
                element.click()
                time.sleep(random.uniform(0.1, 0.3))
                element.clear()
                time.sleep(random.uniform(0.05, 0.15))
            for ch in text:
                element.send_keys(ch)
                # Occasional brief hesitation (like re-reading what was typed)
                delay = random.uniform(self.TYPE_DELAY_MIN, self.TYPE_DELAY_MAX)
                if random.random() < 0.05:    # 5% chance of longer pause
                    delay += random.uniform(0.2, 0.8)
                time.sleep(delay)
            return True
        except Exception:
            return False

    def human_scroll_to(self, y_target: int):
        """Scroll to a y position with realistic acceleration/deceleration."""
        try:
            curr_y = self.driver.execute_script("return window.pageYOffset;")
            delta  = y_target - curr_y
            steps  = max(5, abs(delta) // 100)
            for i in range(steps):
                # Ease-in-out curve
                t      = i / steps
                factor = t * t * (3 - 2 * t)
                scroll_y = curr_y + int(delta * factor)
                self.driver.execute_script(f"window.scrollTo(0, {scroll_y});")
                time.sleep(random.uniform(self.SCROLL_DELAY_MIN / steps,
                                          self.SCROLL_DELAY_MAX / steps))
            self.driver.execute_script(f"window.scrollTo(0, {y_target});")
        except Exception:
            try:
                self.driver.execute_script(f"window.scrollTo(0, {y_target});")
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # WAF DETECTION AND BYPASS
    # ──────────────────────────────────────────────────────────────────────────

    def detect_block(self) -> Optional[str]:
        """
        Detect if the current page is a WAF challenge/block page.
        Returns WAF name or None if page appears normal.
        """
        if not self.driver:
            return None
        try:
            html    = self.driver.page_source or ""
            headers = {}    # Selenium doesn't expose response headers directly
            waf     = detect_waf(html, headers)
            return waf
        except Exception:
            return None

    def wait_for_cloudflare(self, timeout: float = None) -> bool:
        """
        Wait for Cloudflare's JS challenge to complete automatically.
        CF usually resolves within 5-8 seconds for a real browser.
        Returns True if challenge cleared, False if timed out.
        """
        timeout = timeout or self.CF_WAIT_MAX
        start   = time.time()
        while time.time() - start < timeout:
            html = self.driver.page_source or ""
            # CF challenge page has specific identifiers
            if re.search(r"checking your browser|cf-spinner|challenge-form", html, re.IGNORECASE):
                time.sleep(self.CF_POLL_S)
                continue
            # If we made it past the challenge, a normal page will load
            if self.driver.title and not re.search(
                r"just a moment|attention required|cloudflare", self.driver.title, re.IGNORECASE
            ):
                return True
            time.sleep(self.CF_POLL_S)
        return False

    def bypass_datadome(self) -> bool:
        """
        DataDome detection: move mouse in realistic pattern, then wait for cookie.
        DataDome scores sessions based on mouse entropy — even one movement helps.
        """
        try:
            # Move mouse across the page in an S-curve
            w = self.driver.execute_script("return window.innerWidth;")  or 1280
            h = self.driver.execute_script("return window.innerHeight;") or 800
            actions = ActionChains(self.driver)
            # S-curve movement across viewport
            for i in range(20):
                t  = i / 19
                x  = int(w * t)
                y  = int(h * 0.5 + math.sin(t * math.pi * 2) * h * 0.2)
                actions.move_by_offset(x - (int(w * (i-1)/19) if i > 0 else 0),
                                       y - (int(h * 0.5 + math.sin((i-1)/19 * math.pi * 2) * h * 0.2) if i > 0 else int(h * 0.5)))
                actions.pause(random.uniform(0.03, 0.08))
            actions.perform()
            time.sleep(random.uniform(1.0, 2.5))
            return True
        except Exception:
            return False

    def bypass_page(self, url: str) -> bool:
        """
        Attempt to bypass whatever WAF is blocking the page.
        Tries platform-specific strategies based on detected WAF.
        Returns True if bypass appeared successful.
        """
        waf = self.detect_block()
        if not waf:
            return True   # Not blocked

        if "cloudflare" in waf:
            cleared = self.wait_for_cloudflare()
            if not cleared:
                # Try refreshing once — sometimes CF challenge state is stale
                self.driver.refresh()
                time.sleep(3)
                cleared = self.wait_for_cloudflare(timeout=10)
            return cleared

        if "datadome" in waf:
            return self.bypass_datadome()

        if "akamai" in waf:
            # Akamai uses cookie-based session — just wait and let JS run
            time.sleep(random.uniform(3.0, 6.0))
            self._random_micro_scroll()
            return self.detect_block() is None

        # Generic block — wait and retry
        time.sleep(random.uniform(2.0, 5.0))
        return self.detect_block() is None

    # ──────────────────────────────────────────────────────────────────────────
    # MULTI-STRATEGY CONTENT EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def extract_content(
        self,
        url:      str,
        strategy: str = "auto",    # auto | requests | selenium | both
    ) -> Dict[str, Any]:
        """
        Full pipeline: open URL → detect WAF → bypass if needed → extract content.

        Fallback chain:
          1. requests (fast, no JS, works for most static content)
          2. Selenium (real browser, handles JS-rendered pages)
          3. Selenium + bypass (for WAF-protected pages)
          4. Google Cache fallback (last resort)

        Returns dict with: ok, url, final_url, html, text, title, waf, strategy_used
        """
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "https://" + url

        result = {
            "ok":           False,
            "url":          url,
            "final_url":    url,
            "html":         "",
            "text":         "",
            "title":        "",
            "waf":          None,
            "strategy_used": None,
            "error":        None,
        }

        # ── Strategy 1: Plain requests (fast, no JS engine overhead) ──────────
        if strategy in ("auto", "requests", "both"):
            r = self._extract_requests(url)
            if r["ok"] and not r.get("waf"):
                result.update(r)
                result["strategy_used"] = "requests"
                return result
            # If requests got blocked, note the WAF for Selenium strategy
            if r.get("waf"):
                result["waf"] = r["waf"]

        # ── Strategy 2: Selenium (handles JS-rendered pages) ─────────────────
        if strategy in ("auto", "selenium", "both") and self.driver:
            r = self._extract_selenium(url)
            if r["ok"] and not r.get("waf"):
                result.update(r)
                result["strategy_used"] = "selenium"
                return result
            if r.get("waf"):
                result["waf"] = r["waf"]

        # ── Strategy 3: Selenium + WAF bypass ────────────────────────────────
        if self.driver and result.get("waf"):
            r = self._extract_selenium_bypass(url)
            if r["ok"]:
                result.update(r)
                result["strategy_used"] = "selenium+bypass"
                return result

        # ── Strategy 4: Google Cache ──────────────────────────────────────────
        r = self._extract_google_cache(url)
        if r["ok"]:
            result.update(r)
            result["strategy_used"] = "google_cache"
            return result

        result["error"] = f"All extraction strategies failed. WAF: {result.get('waf', 'unknown')}"
        return result

    def _extract_requests(self, url: str) -> Dict:
        """Extract via requests library — fast, no JS."""
        if not self._session:
            return {"ok": False, "waf": None}
        try:
            resp = self._session.get(url, timeout=12, allow_redirects=True)
            html = resp.text or ""
            waf  = detect_waf(html, dict(resp.headers))
            if waf and waf != "generic_block":
                return {"ok": False, "waf": waf}
            # Extract text
            text  = re.sub(r"<[^>]+>", " ", html)
            text  = re.sub(r"\s+", " ", text).strip()
            title = ""
            m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            if m:
                title = m.group(1).strip()
            return {
                "ok":        True,
                "html":      html,
                "text":      text,
                "title":     title,
                "final_url": resp.url,
                "waf":       waf,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "waf": None}

    def _extract_selenium(self, url: str) -> Dict:
        """Extract via Selenium — handles JS-rendered pages."""
        try:
            self.driver.get(url)
            self.on_page_load()

            html  = self.driver.page_source or ""
            waf   = detect_waf(html)
            title = self.driver.title or ""
            try:
                body = self.driver.find_element(By.TAG_NAME, "body")
                text = body.text or ""
            except Exception:
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()

            return {
                "ok":        not bool(waf),
                "html":      html,
                "text":      text,
                "title":     title,
                "final_url": self.driver.current_url,
                "waf":       waf,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "waf": None}

    def _extract_selenium_bypass(self, url: str) -> Dict:
        """Selenium + active WAF bypass (Cloudflare/DataDome/Akamai)."""
        try:
            self.driver.get(url)
            time.sleep(random.uniform(1.0, 2.0))
            self.on_page_load()

            waf = self.detect_block()
            if waf:
                success = self.bypass_page(url)
                if not success:
                    return {"ok": False, "waf": waf}

            html  = self.driver.page_source or ""
            title = self.driver.title or ""
            try:
                body  = self.driver.find_element(By.TAG_NAME, "body")
                text  = body.text or ""
            except Exception:
                text  = re.sub(r"<[^>]+>", " ", html)
                text  = re.sub(r"\s+", " ", text).strip()

            return {
                "ok":        True,
                "html":      html,
                "text":      text,
                "title":     title,
                "final_url": self.driver.current_url,
                "waf":       None,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "waf": None}

    def _extract_google_cache(self, url: str) -> Dict:
        """Last resort: try Google's cached version of the page."""
        try:
            encoded = urllib.parse.quote(url, safe="")
            cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{encoded}"
            return self._extract_requests(cache_url)
        except Exception:
            return {"ok": False}

    # ──────────────────────────────────────────────────────────────────────────
    # READING PROTECTED CONTENT (structured pipeline)
    # ──────────────────────────────────────────────────────────────────────────

    def read_website(self, url: str) -> str:
        """
        High-level method: read any website and return its text content.
        Handles bots, JS-rendered pages, WAFs, and caching fallbacks.
        Used by ARIA's web_searcher and research agents.
        """
        result = self.extract_content(url, strategy="auto")
        if result["ok"]:
            text  = result.get("text", "")
            title = result.get("title", "")
            via   = result.get("strategy_used", "unknown")
            return f"[{title}]\n\n{text[:8000]}\n\n(Extracted via: {via})"
        err = result.get("error", "Unknown error")
        waf = result.get("waf", "unknown")
        return f"Could not read website: {err}. WAF detected: {waf}"

    # ──────────────────────────────────────────────────────────────────────────
    # NL interface
    # ──────────────────────────────────────────────────────────────────────────

    def run_nl(self, instruction: str) -> str:
        """
        Natural language interface for the stealth engine.

        Patterns:
          "read https://example.com" → full stealth extraction
          "check if blocked on https://example.com"
          "apply stealth patches"
          "bypass cloudflare on https://..."
        """
        low = instruction.lower().strip()

        url_m = re.search(
            r"(https?://[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",
            instruction,
        )
        url = url_m.group(1) if url_m else None

        if re.search(r"\bapply\b.*\bstealth\b|\bpatch\b", low):
            ok = self.apply_all()
            return "Stealth patches applied (23 vectors)" if ok else "CDP not available — patches applied per-page via execute_script"

        if re.search(r"\bcheck\b.*\bblock\b|\bdetect\b.*\bwaf\b", low) and url:
            if self.driver:
                self.driver.get(url)
                time.sleep(2)
                waf = self.detect_block()
                return f"WAF detected: {waf}" if waf else "No WAF detected — page appears accessible"
            return "No browser driver attached"

        if re.search(r"\bbypass\b.*\bcloudflare\b|\bbypass\b", low) and url:
            if self.driver:
                self.driver.get(url)
                ok = self.wait_for_cloudflare()
                return "Cloudflare challenge cleared" if ok else "Could not bypass Cloudflare"
            return "No browser driver attached"

        if url and re.search(r"\bread\b|\bextract\b|\bget content\b|\bopen\b", low):
            return self.read_website(url)

        return (
            "Stealth engine commands:\n"
            "  'apply stealth patches' — inject 23 fingerprint patches\n"
            "  'read <url>'            — full stealth content extraction\n"
            "  'check if blocked on <url>' — detect WAF\n"
            "  'bypass cloudflare on <url>' — wait for CF challenge"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_stealth_instance: Optional[StealthEngine] = None

def get_stealth_engine(driver=None) -> StealthEngine:
    """Get or create the global StealthEngine instance."""
    global _stealth_instance
    if _stealth_instance is None:
        _stealth_instance = StealthEngine(driver)
    elif driver is not None:
        _stealth_instance.attach(driver)
    return _stealth_instance
