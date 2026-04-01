"""
ARIA Live Browser Agent
========================
Persistent, profile-aware Chrome session that ARIA controls via natural language.

Features:
  - Profile selection dialogue   ("Which profile should I use?")
  - Profile switching on command  ("switch to novaai profile")
  - Live search                  ("search for Python tutorials")
  - Navigate to URL              ("go to github.com")
  - Page crawl + summarise       ("what's on this page?", "read this page")
  - Continuous listening mode    (browser stays open, ARIA keeps watching)
  - Stop command                 ("stop browsing", "close browser")
  - 365/24/7 converse mode       (ARIA answers chat even while browsing)

Integrates with:
  - windows_kernel_agent.list_chrome_profiles()  for profile discovery
  - BrowserAgent                                 for Selenium control
  - core.engine                                  for LLM summarisation
"""

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger("aria.live_browser")

# ── Search engine URLs ────────────────────────────────────────────────────────
SEARCH_ENGINES = {
    "google":    "https://www.google.com/search?q=",
    "bing":      "https://www.bing.com/search?q=",
    "duckduckgo":"https://duckduckgo.com/?q=",
    "youtube":   "https://www.youtube.com/results?search_query=",
    "amazon":    "https://www.amazon.in/s?k=",
    "flipkart":  "https://www.flipkart.com/search?q=",
    "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search=",
}

# ── NL command patterns ───────────────────────────────────────────────────────
_SEARCH_PATTERNS = [
    r"(?:search for|search|look up|find|google|bing)\s+(.+)",
    r"(?:what is|who is|tell me about)\s+(.+)",
    r"(?:show me|find me)\s+(.+)",
]
_NAVIGATE_PATTERNS = [
    r"(?:go to|open|navigate to|visit|take me to)\s+(https?://\S+)",
    r"(?:go to|open|navigate to|visit|take me to)\s+([\w\-]+\.[\w\-./]+)",
]
_READ_PATTERNS = [
    r"(?:read|summarise|summarize|what.?s on|tell me about) (?:this|the) (?:page|site|website|article)",
    r"(?:read this|what does it say|summarise it|extract info)",
    r"crawl (?:this|the) (?:page|site)",
]
_STOP_PATTERNS = [
    r"stop (?:browsing|searching|the browser|chrome)",
    r"close (?:browser|chrome|the tab|the window)",
    r"(?:done|finished|that.?s all|exit browser)",
]
_SWITCH_PATTERNS = [
    r"switch (?:to|profile)\s+(.+?)(?:\s+profile)?$",
    r"(?:use|open)\s+(.+?)(?:'s)?\s+(?:chrome|profile|browser)",
    r"change profile to\s+(.+)",
]
_SCROLL_PATTERNS = [
    r"scroll (?:down|up)(?: (\d+))?",
    r"(?:page )?(?:down|up)",
]
_BACK_PATTERNS    = [r"go back", r"back", r"previous page"]
_FORWARD_PATTERNS = [r"go forward", r"forward", r"next page"]
_SCREENSHOT_PATTERNS = [r"(?:take a )?screenshot", r"capture (?:this|the) (?:page|screen)"]
_LINKS_PATTERNS   = [r"(?:list|show|get) (?:links|urls)", r"what links are here"]


def _match_any(text: str, patterns: List[str]) -> Optional[re.Match]:
    text = text.strip()
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m
    return None


def _url_from_query(query: str) -> str:
    """If query looks like a URL, return it with scheme. Otherwise None."""
    query = query.strip()
    if re.match(r'^https?://', query):
        return query
    # bare domain like "github.com", "youtube.com/watch?v=..."
    if re.match(r'^[\w\-]+\.[a-z]{2,}(/\S*)?$', query, re.IGNORECASE):
        return "https://" + query
    return ""


# ── Session state ─────────────────────────────────────────────────────────────

class BrowserSession:
    """Holds state for one active browser session (one Chrome profile)."""

    def __init__(self, profile: Dict[str, str], headless: bool = False):
        self.profile         = profile          # {folder, name, email, path}
        self.headless        = headless
        self._agent          = None             # BrowserAgent instance
        self._lock           = threading.Lock()
        self._alive          = False
        self.current_url     = ""
        self.current_title   = ""
        self.last_search     = ""

    def ensure_open(self) -> bool:
        """Start BrowserAgent with this profile if not already running."""
        with self._lock:
            if self._alive and self._agent:
                return True
            try:
                from agents.browser_agent import BrowserAgent
                from selenium.webdriver.chrome.options import Options
                from agents.windows_kernel_agent import CHROME_PROFILE_DIRS, _find_browser_exe
                import os

                # Build a BrowserAgent subclass that uses profile options
                agent = BrowserAgent(headless=self.headless, stealth=True)

                # Patch _build_options to inject profile flags
                profile_folder = self.profile.get("folder", "Default")
                profile_root   = None
                for d in CHROME_PROFILE_DIRS:
                    if (d / profile_folder).exists():
                        profile_root = str(d)
                        break

                _orig_build = agent._build_options.__func__

                def _patched_build_options(self_inner):
                    opts = _orig_build(self_inner)
                    if profile_root:
                        opts.add_argument(f"--user-data-dir={profile_root}")
                    opts.add_argument(f"--profile-directory={profile_folder}")
                    return opts

                import types
                agent._build_options = types.MethodType(_patched_build_options, agent)

                # Force driver init
                agent._init_driver()
                self._agent = agent
                self._alive = True
                logger.info(f"Browser session opened: profile={profile_folder}")
                return True
            except Exception as e:
                logger.error(f"Failed to open browser session: {e}")
                self._alive = False
                return False

    def close(self) -> None:
        with self._lock:
            if self._agent:
                try:
                    self._agent.close()
                except Exception:
                    pass
                self._agent = None
                self._alive = False

    @property
    def alive(self) -> bool:
        return self._alive and self._agent is not None

    def agent(self):
        return self._agent


# ── Main LiveBrowserAgent ─────────────────────────────────────────────────────

class LiveBrowserAgent:
    """
    ARIA's persistent browser companion.

    Usage in server chat fast-path:
        lba = LiveBrowserAgent(engine=engine)
        for chunk in lba.handle_command("search for Python tutorials"):
            yield chunk
    """

    def __init__(self, engine=None, headless: bool = False):
        self._engine          = engine
        self._headless        = headless
        self._session: Optional[BrowserSession] = None
        self._active_profile: Optional[Dict]    = None
        self._browsing        = False            # True when browser is open
        self._lock            = threading.Lock()

    # ── Public: main entry ────────────────────────────────────────────────────

    def handle_command(self, text: str) -> Generator[str, None, None]:
        """
        Parse a natural language command and execute it.
        Yields SSE-formatted text chunks.
        """
        text = text.strip()

        # ── STOP ──────────────────────────────────────────────────────────────
        if _match_any(text, _STOP_PATTERNS):
            yield from self._stop_browsing()
            return

        # ── PROFILE SWITCH ────────────────────────────────────────────────────
        m = _match_any(text, _SWITCH_PATTERNS)
        if m:
            yield from self._switch_profile(m.group(1).strip())
            return

        # ── No active session — need to pick a profile first ─────────────────
        if not self._browsing:
            yield from self._start_session_with_query(text)
            return

        # ── Active session commands ───────────────────────────────────────────
        yield from self._dispatch_active(text)

    def handle_profile_reply(self, reply: str) -> Generator[str, None, None]:
        """
        Called when user replies to ARIA's 'Which profile?' question.
        reply is the user's input (name or number).
        """
        yield from self._open_with_selected_profile(reply)

    def is_browsing(self) -> bool:
        return self._browsing

    def status(self) -> dict:
        return {
            "browsing":       self._browsing,
            "profile":        self._active_profile,
            "current_url":    self._session.current_url if self._session else "",
            "current_title":  self._session.current_title if self._session else "",
            "last_search":    self._session.last_search if self._session else "",
        }

    def close(self) -> None:
        if self._session:
            self._session.close()
        self._browsing = False
        self._session  = None

    # ── Profile selection flow ────────────────────────────────────────────────

    def _start_session_with_query(self, pending_query: str) -> Generator[str, None, None]:
        """Ask user to pick a profile, stash the pending query."""
        self._pending_query = pending_query
        try:
            from agents.windows_kernel_agent import list_chrome_profiles
            profiles = list_chrome_profiles("chrome")
        except Exception:
            profiles = []

        if not profiles:
            yield self._sse("No Chrome profiles found. Please open Chrome manually first.")
            return

        self._available_profiles = profiles
        lines = []
        for i, p in enumerate(profiles, 1):
            line = f"  **{i}.** {p['name']}"
            if p["email"]:
                line += f" ({p['email']})"
            lines.append(line)

        msg = (
            "**Which Chrome profile should I use?**\n\n"
            + "\n".join(lines)
            + "\n\nSay the name or number (e.g. *'Chandan'* or *'2'*)."
        )
        yield self._sse(msg, meta={"awaiting": "profile_selection"})

    def _open_with_selected_profile(self, reply: str) -> Generator[str, None, None]:
        """Match reply to a profile and open the browser."""
        profiles = getattr(self, "_available_profiles", [])
        if not profiles:
            try:
                from agents.windows_kernel_agent import list_chrome_profiles
                profiles = list_chrome_profiles("chrome")
                self._available_profiles = profiles
            except Exception:
                pass

        profile = None
        # Number selection
        if reply.strip().isdigit():
            idx = int(reply.strip()) - 1
            if 0 <= idx < len(profiles):
                profile = profiles[idx]
        else:
            # NL match
            try:
                from agents.windows_kernel_agent import find_profile_by_name
                profile = find_profile_by_name(reply, "chrome")
            except Exception:
                for p in profiles:
                    if reply.lower() in p["name"].lower() or reply.lower() in p["email"].lower():
                        profile = p
                        break

        if not profile:
            names = ", ".join(p["name"] for p in profiles)
            yield self._sse(f"Couldn't match '{reply}' to a profile. Available: {names}")
            return

        yield self._sse(f"Opening Chrome with profile **{profile['name']}**...")

        session = BrowserSession(profile, headless=self._headless)
        ok = session.ensure_open()
        if not ok:
            yield self._sse("Failed to open Chrome. Make sure Chrome is installed.")
            return

        with self._lock:
            if self._session:
                self._session.close()
            self._session        = session
            self._active_profile = profile
            self._browsing       = True

        yield self._sse(f"Chrome opened with **{profile['name']}**'s profile.")

        # Execute any pending query
        pending = getattr(self, "_pending_query", "")
        if pending:
            self._pending_query = ""
            yield from self._dispatch_active(pending)

    def _switch_profile(self, query: str) -> Generator[str, None, None]:
        """Switch to a different Chrome profile."""
        try:
            from agents.windows_kernel_agent import find_profile_by_name, list_chrome_profiles
            profile = find_profile_by_name(query, "chrome")
        except Exception:
            profile = None

        if not profile:
            yield self._sse(f"No profile matching '{query}' found.")
            return

        if self._active_profile and profile["folder"] == self._active_profile["folder"]:
            yield self._sse(f"Already using **{profile['name']}**'s profile.")
            return

        yield self._sse(f"Switching to **{profile['name']}**'s Chrome profile...")

        # Close current session
        if self._session:
            self._session.close()

        session = BrowserSession(profile, headless=self._headless)
        ok = session.ensure_open()
        if not ok:
            yield self._sse("Failed to switch profile.")
            return

        with self._lock:
            self._session        = session
            self._active_profile = profile
            self._browsing       = True

        yield self._sse(f"Switched to **{profile['name']}**'s profile. Chrome is ready.")

    # ── Active session dispatch ───────────────────────────────────────────────

    def _dispatch_active(self, text: str) -> Generator[str, None, None]:
        agent = self._session.agent() if self._session else None
        if not agent:
            yield self._sse("Browser session lost. Say **open Chrome** to start a new session.")
            self._browsing = False
            return

        # NAVIGATE to URL
        m = _match_any(text, _NAVIGATE_PATTERNS)
        if m:
            url = _url_from_query(m.group(1)) or m.group(1)
            yield from self._navigate(agent, url)
            return

        # SEARCH
        m = _match_any(text, _SEARCH_PATTERNS)
        if m:
            yield from self._search(agent, m.group(1).strip())
            return

        # READ / CRAWL page
        if _match_any(text, _READ_PATTERNS):
            yield from self._read_page(agent)
            return

        # SCROLL
        m = _match_any(text, _SCROLL_PATTERNS)
        if m:
            direction = "down" if "down" in text.lower() else "up"
            amount = int(m.group(1)) if m.lastindex and m.group(1) else 3
            result = agent.scroll(direction, amount)
            yield self._sse("Scrolled " + direction + ".")
            return

        # BACK / FORWARD
        if _match_any(text, _BACK_PATTERNS):
            agent.back()
            yield self._sse("Went back.")
            return
        if _match_any(text, _FORWARD_PATTERNS):
            agent.forward()
            yield self._sse("Went forward.")
            return

        # SCREENSHOT
        if _match_any(text, _SCREENSHOT_PATTERNS):
            result = agent.take_screenshot()
            if result.get("ok"):
                yield self._sse("Screenshot saved to: " + result.get("path", "unknown"))
            else:
                yield self._sse("Screenshot failed: " + result.get("result", ""))
            return

        # LINKS
        if _match_any(text, _LINKS_PATTERNS):
            result = agent.get_links()
            if result.get("ok"):
                links = result.get("data", [])[:10]
                lines = [f"  • [{l.get('text','?')}]({l.get('href','')})" for l in links]
                yield self._sse("Links on this page:\n\n" + "\n".join(lines))
            return

        # CURRENT URL / TITLE
        if any(kw in text.lower() for kw in ("what page", "current page", "where am i", "current url")):
            title  = agent.get_page_title().get("data", "")
            url    = self._session.current_url
            yield self._sse(f"Currently on: **{title}**\n{url}")
            return

        # Fallback: treat as a search query
        yield from self._search(agent, text)

    # ── Search ────────────────────────────────────────────────────────────────

    def _search(self, agent, query: str, engine: str = "google") -> Generator[str, None, None]:
        # Detect explicit engine requests
        for eng in SEARCH_ENGINES:
            if eng in query.lower():
                query = re.sub(rf'\b{eng}\b', '', query, flags=re.IGNORECASE).strip()
                engine = eng
                break

        yield self._sse(f"Searching **{engine}** for: *{query}*...")
        result = agent.search_web(query, engine)

        if not result.get("ok"):
            yield self._sse("Search failed: " + result.get("result", "unknown error"))
            return

        try:
            url = agent.driver().current_url
            title = agent.driver().title
            if self._session:
                self._session.current_url   = url
                self._session.current_title = title
                self._session.last_search   = query
        except Exception:
            pass

        # Extract top results from page
        page_result = agent.get_page_text()
        if page_result.get("ok"):
            raw_text = page_result.get("data", "")
            summary  = self._summarise(query, raw_text, mode="search_results")
            yield self._sse(summary)
        else:
            yield self._sse(f"Search opened. Say **read this page** for results, or **go to [URL]** to navigate.")

        yield self._sse(
            "\n\n---\n*Browser is open. Say:*\n"
            "• **go to [url]** — navigate\n"
            "• **read this page** — extract content\n"
            "• **search for [query]** — new search\n"
            "• **stop browsing** — close browser",
            meta={"browsing": True}
        )

    # ── Navigate ──────────────────────────────────────────────────────────────

    def _navigate(self, agent, url: str) -> Generator[str, None, None]:
        if not url.startswith("http"):
            url = "https://" + url

        yield self._sse(f"Navigating to **{url}**...")
        result = agent.open_url(url)

        if not result.get("ok"):
            yield self._sse("Navigation failed: " + result.get("result", ""))
            return

        try:
            title = agent.driver().title
            curr  = agent.driver().current_url
            if self._session:
                self._session.current_url   = curr
                self._session.current_title = title
        except Exception:
            title = url

        yield self._sse(f"Opened: **{title}**\n{url}")
        yield self._sse(
            "\nSay **read this page** to get content, or give me your next command.",
            meta={"browsing": True}
        )

    # ── Read / crawl page ─────────────────────────────────────────────────────

    def _read_page(self, agent) -> Generator[str, None, None]:
        yield self._sse("Reading this page...")

        title_r = agent.get_page_title()
        text_r  = agent.get_page_text()

        title = title_r.get("data", "this page") if title_r.get("ok") else "this page"
        try:
            url = agent.driver().current_url
        except Exception:
            url = ""

        if not text_r.get("ok"):
            yield self._sse("Couldn't read the page. It may require login or use heavy JavaScript.")
            return

        raw_text = text_r.get("data", "")
        if not raw_text.strip():
            yield self._sse("Page appears empty or uses dynamic loading. Try scrolling and reading again.")
            return

        summary = self._summarise(title, raw_text, mode="page_summary")
        yield self._sse(f"**{title}**\n{url}\n\n{summary}", meta={"browsing": True})

    # ── Stop ─────────────────────────────────────────────────────────────────

    def _stop_browsing(self) -> Generator[str, None, None]:
        profile_name = self._active_profile.get("name", "browser") if self._active_profile else "browser"
        self.close()
        yield self._sse(
            f"Closed **{profile_name}**'s Chrome. Browser session ended.\n\n"
            "I'm still here in conversation mode — just ask me anything!"
        )

    # ── LLM summarisation ────────────────────────────────────────────────────

    def _summarise(self, topic: str, raw_text: str, mode: str = "page_summary") -> str:
        """Use ARIA's engine to summarise crawled content."""
        # Truncate to ~6000 chars to fit LLM context
        truncated = raw_text[:6000]
        if not truncated.strip():
            return "(No readable content found on this page.)"

        if self._engine:
            try:
                if mode == "search_results":
                    prompt = (
                        f"Based on these Google search results for '{topic}', "
                        f"give a concise, helpful answer in 3-5 bullet points:\n\n{truncated}"
                    )
                else:
                    prompt = (
                        f"Summarise this web page about '{topic}' in a clear, concise way. "
                        f"Extract the key information:\n\n{truncated}"
                    )
                result = self._engine.generate(prompt, max_tokens=400, temperature=0.3)
                return result
            except Exception as e:
                logger.warning(f"LLM summarise failed: {e}")

        # Fallback: return first 800 chars
        return truncated[:800] + ("\n\n*(truncated — say 'read more' for more)*" if len(raw_text) > 800 else "")

    # ── SSE helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _sse(text: str, meta: Optional[dict] = None) -> str:
        payload = {"type": "text", "text": text}
        if meta:
            payload.update(meta)
        return "data: " + json.dumps(payload) + "\n\n"


# ── Module-level singleton ────────────────────────────────────────────────────

_agent: Optional[LiveBrowserAgent] = None
_lock  = threading.Lock()


def get_agent(engine=None) -> LiveBrowserAgent:
    global _agent
    with _lock:
        if _agent is None:
            _agent = LiveBrowserAgent(engine=engine)
        elif engine and not _agent._engine:
            _agent._engine = engine
        return _agent
