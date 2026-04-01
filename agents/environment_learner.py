"""
ARIA Environment Learner
=========================
On first install, ARIA scans the system and builds a personal knowledge base.
Continuously learns from every new file, app, document, and conversation.

Capabilities:
  1. Installed apps discovery (Registry + Program Files + PATH)
  2. Document scanner (Desktop, Documents, Downloads — PDF/DOCX/TXT/MD)
  3. Code project detector (git repos, package.json, requirements.txt, pom.xml)
  4. Browser bookmarks reader (Chrome/Edge/Brave Bookmarks JSON)
  5. Recent files tracker (Windows recent items)
  6. Incremental learning (only scans new/changed files)
  7. Personal graph builder (who the user is, what they work on, their tools)
  8. Privacy-first: everything stays local, nothing sent externally
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional / graceful imports
# ---------------------------------------------------------------------------
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    psutil = None  # type: ignore
    _PSUTIL_OK = False

try:
    import winreg
    _WINREG_OK = True
except ImportError:
    winreg = None  # type: ignore
    _WINREG_OK = False

try:
    from PyPDF2 import PdfReader as _PdfReader
    _PDF_OK = True
except ImportError:
    try:
        from pypdf import PdfReader as _PdfReader  # type: ignore
        _PDF_OK = True
    except ImportError:
        _PdfReader = None  # type: ignore
        _PDF_OK = False

try:
    from docx import Document as _DocxDocument
    _DOCX_OK = True
except ImportError:
    _DocxDocument = None  # type: ignore
    _DOCX_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG = logging.getLogger("aria.environment_learner")

ARIA_DATA_DIR = Path.home() / ".aria"
DB_PATH = ARIA_DATA_DIR / "environment.db"
PROFILE_PATH = ARIA_DATA_DIR / "user_profile.json"

SCAN_ROOTS = [
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "OneDrive",
]

DOC_EXTENSIONS = {".txt", ".md", ".rst", ".pdf", ".docx", ".log", ".csv", ".json", ".yaml", ".yml"}

CODE_MARKERS = [
    "package.json", "requirements.txt", "setup.py", "pyproject.toml",
    "Cargo.toml", "pom.xml", "build.gradle", "go.mod", "Makefile",
    "CMakeLists.txt", ".git",
]

APP_CATEGORIES = {
    "browser": {"chrome", "firefox", "edge", "brave", "opera", "vivaldi"},
    "editor": {"code", "vscode", "notepad++", "sublime_text", "vim", "neovim", "emacs", "cursor"},
    "terminal": {"windows terminal", "wt", "powershell", "cmd", "bash", "alacritty", "hyper"},
    "database": {"dbeaver", "tableplus", "pgadmin", "mysql workbench", "datagrip"},
    "design": {"figma", "photoshop", "illustrator", "gimp", "inkscape", "affinity"},
    "communication": {"slack", "discord", "teams", "zoom", "telegram", "signal"},
    "productivity": {"notion", "obsidian", "logseq", "onenote", "evernote"},
    "dev_tool": {"git", "docker", "postman", "insomnia", "wireshark"},
}

BROWSER_BOOKMARK_PATHS: dict[str, list[Path]] = {
    "chrome": [
        Path.home() / "AppData/Local/Google/Chrome/User Data/Default/Bookmarks",
    ],
    "edge": [
        Path.home() / "AppData/Local/Microsoft/Edge/User Data/Default/Bookmarks",
    ],
    "brave": [
        Path.home() / "AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/Bookmarks",
    ],
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AppInfo:
    name: str
    path: str
    version: str = ""
    category: str = "other"
    last_used: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentInfo:
    path: str
    type: str
    summary: str
    keywords: list[str]
    last_modified: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UserProfile:
    name: str = ""
    expertise_areas: list[str] = field(default_factory=list)
    preferred_tools: list[str] = field(default_factory=list)
    working_hours: str = ""           # e.g. "09:00-18:00"
    language: str = "English"
    os: str = ""
    code_languages: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS apps (
            name TEXT PRIMARY KEY,
            path TEXT,
            version TEXT,
            category TEXT,
            last_used TEXT,
            scanned_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            path TEXT PRIMARY KEY,
            type TEXT,
            summary TEXT,
            keywords TEXT,
            last_modified TEXT,
            scanned_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS code_projects (
            path TEXT PRIMARY KEY,
            markers TEXT,
            languages TEXT,
            scanned_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            url TEXT PRIMARY KEY,
            title TEXT,
            folder TEXT,
            browser TEXT,
            scanned_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recent_files (
            path TEXT PRIMARY KEY,
            extension TEXT,
            accessed_at TEXT,
            scanned_at TEXT
        )
    """)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_txt(path: Path, max_chars: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _extract_text_pdf(path: Path, max_chars: int = 2000) -> str:
    if not _PDF_OK:
        return ""
    try:
        reader = _PdfReader(str(path))
        text = ""
        for page in reader.pages[:5]:
            text += page.extract_text() or ""
            if len(text) >= max_chars:
                break
        return text[:max_chars]
    except Exception:
        return ""


def _extract_text_docx(path: Path, max_chars: int = 2000) -> str:
    if not _DOCX_OK:
        return ""
    try:
        doc = _DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text[:max_chars]
    except Exception:
        return ""


def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_text_pdf(path)
    if ext == ".docx":
        return _extract_text_docx(path)
    if ext in {".txt", ".md", ".rst", ".log", ".csv", ".json", ".yaml", ".yml"}:
        return _extract_text_txt(path)
    return ""


def _summarise(text: str, max_words: int = 30) -> str:
    """Very lightweight extractive summary — first non-empty sentence."""
    for sentence in re.split(r"[.!?\n]", text):
        stripped = sentence.strip()
        if len(stripped.split()) >= 5:
            words = stripped.split()
            return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")
    return text[:120].strip()


def _extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """Simple frequency-based keyword extraction (no NLTK needed)."""
    STOPWORDS = {
        "the", "a", "an", "and", "or", "is", "in", "to", "of", "for",
        "this", "that", "with", "on", "at", "by", "from", "as", "it",
        "be", "are", "was", "were", "has", "have", "had", "not", "but",
        "if", "so", "do", "did", "will", "can", "may", "should", "would",
    }
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:top_n]]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EnvironmentLearner:
    """
    Scans the local environment and builds ARIA's personal knowledge base.
    All data is stored locally in SQLite and JSON — nothing leaves the device.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        profile_path: Path = PROFILE_PATH,
        scan_roots: Optional[list[Path]] = None,
        max_docs: int = 200,
    ):
        self.db_path = db_path
        self.profile_path = profile_path
        self.scan_roots = scan_roots or SCAN_ROOTS
        self.max_docs = max_docs
        self._conn: Optional[sqlite3.Connection] = None
        self._profile: Optional[UserProfile] = None
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _init_db(self.db_path)
        return self._conn

    def _doc_already_scanned(self, path: Path) -> bool:
        """Return True if the file has not been modified since last scan."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT last_modified FROM documents WHERE path=?", (str(path),)
        ).fetchone()
        if not row:
            return False
        try:
            stored_mtime = row[0]
            actual_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            return stored_mtime == actual_mtime
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 1. App scanning
    # ------------------------------------------------------------------

    def scan_apps(self) -> list[AppInfo]:
        """Discover installed applications via Registry, PATH, and Program Files."""
        apps: dict[str, AppInfo] = {}

        # Windows Registry uninstall keys
        if _WINREG_OK and platform.system() == "Windows":
            reg_keys = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]
            for hive, key_path in reg_keys:
                try:
                    with winreg.OpenKey(hive, key_path) as key:
                        for i in range(winreg.QueryInfoKey(key)[0]):
                            try:
                                sub_name = winreg.EnumKey(key, i)
                                with winreg.OpenKey(key, sub_name) as sub:
                                    def _reg(name: str) -> str:
                                        try:
                                            return str(winreg.QueryValueEx(sub, name)[0])
                                        except Exception:
                                            return ""
                                    name = _reg("DisplayName").strip()
                                    path = _reg("InstallLocation") or _reg("DisplayIcon")
                                    version = _reg("DisplayVersion")
                                    if name and name not in apps:
                                        cat = _categorise_app(name)
                                        apps[name] = AppInfo(
                                            name=name, path=path,
                                            version=version, category=cat,
                                        )
                            except Exception:
                                continue
                except Exception:
                    continue

        # PATH executables
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for dir_str in path_dirs:
            dir_path = Path(dir_str)
            if not dir_path.is_dir():
                continue
            for exe in dir_path.glob("*.exe" if platform.system() == "Windows" else "*"):
                if exe.is_file() and exe.name not in apps:
                    cat = _categorise_app(exe.stem)
                    apps[exe.stem] = AppInfo(
                        name=exe.stem, path=str(exe), category=cat
                    )

        # Program Files directories
        prog_dirs = [
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
            Path.home() / "AppData/Local/Programs",
        ]
        for prog_dir in prog_dirs:
            if not prog_dir.is_dir():
                continue
            for item in prog_dir.iterdir():
                if item.is_dir() and item.name not in apps:
                    apps[item.name] = AppInfo(
                        name=item.name,
                        path=str(item),
                        category=_categorise_app(item.name),
                    )

        result = list(apps.values())

        # Persist
        conn = self._get_conn()
        now = _now()
        for app in result:
            conn.execute(
                """INSERT OR REPLACE INTO apps
                   (name, path, version, category, last_used, scanned_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (app.name, app.path, app.version, app.category, app.last_used, now),
            )
        conn.commit()
        LOG.info("Scanned %d apps", len(result))
        return result

    # ------------------------------------------------------------------
    # 2. Document scanning
    # ------------------------------------------------------------------

    def scan_documents(self, max_docs: Optional[int] = None) -> list[DocumentInfo]:
        """Scan user document directories for readable files."""
        limit = max_docs or self.max_docs
        docs: list[DocumentInfo] = []

        for root in self.scan_roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if len(docs) >= limit:
                    break
                if not path.is_file():
                    continue
                if path.suffix.lower() not in DOC_EXTENSIONS:
                    continue
                if path.stat().st_size > 10 * 1024 * 1024:   # skip > 10 MB
                    continue
                if self._doc_already_scanned(path):
                    continue

                mtime = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat()

                text = _extract_text(path)
                summary = _summarise(text) if text else "(no text extracted)"
                keywords = _extract_keywords(text) if text else []

                doc = DocumentInfo(
                    path=str(path),
                    type=path.suffix.lower().lstrip("."),
                    summary=summary,
                    keywords=keywords,
                    last_modified=mtime,
                )
                docs.append(doc)

                conn = self._get_conn()
                conn.execute(
                    """INSERT OR REPLACE INTO documents
                       (path, type, summary, keywords, last_modified, scanned_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (str(path), doc.type, doc.summary,
                     json.dumps(keywords), mtime, _now()),
                )
                conn.commit()

        LOG.info("Scanned %d documents", len(docs))
        return docs

    # ------------------------------------------------------------------
    # 3. Code project detector
    # ------------------------------------------------------------------

    def scan_code_projects(self) -> list[dict]:
        """Find git repos and project roots in common locations."""
        projects: list[dict] = []
        search_roots = [
            Path.home() / "projects",
            Path.home() / "repos",
            Path.home() / "code",
            Path.home() / "workspace",
            Path.home() / "dev",
            Path.home() / "Desktop",
            Path.home() / "Documents",
        ]

        for root in search_roots:
            if not root.is_dir():
                continue
            # Walk up to 3 levels deep
            for path in _walk_limited(root, max_depth=3):
                if not path.is_dir():
                    continue
                found_markers = [m for m in CODE_MARKERS if (path / m).exists()]
                if found_markers:
                    languages = _detect_languages(path)
                    project = {
                        "path": str(path),
                        "name": path.name,
                        "markers": found_markers,
                        "languages": languages,
                    }
                    projects.append(project)
                    conn = self._get_conn()
                    conn.execute(
                        """INSERT OR REPLACE INTO code_projects
                           (path, markers, languages, scanned_at)
                           VALUES (?, ?, ?, ?)""",
                        (str(path), json.dumps(found_markers),
                         json.dumps(languages), _now()),
                    )
                    conn.commit()

        LOG.info("Found %d code projects", len(projects))
        return projects

    # ------------------------------------------------------------------
    # 4. Browser bookmarks
    # ------------------------------------------------------------------

    def scan_bookmarks(self) -> list[dict]:
        """Read Chrome/Edge/Brave bookmarks JSON files."""
        all_bookmarks: list[dict] = []

        for browser, paths in BROWSER_BOOKMARK_PATHS.items():
            for bm_path in paths:
                if not bm_path.is_file():
                    continue
                try:
                    data = json.loads(bm_path.read_text(encoding="utf-8"))
                    roots = data.get("roots", {})
                    for root_name, root_node in roots.items():
                        _extract_bookmarks(root_node, browser, root_name, all_bookmarks)
                except Exception as exc:
                    LOG.debug("Bookmark read failed (%s): %s", browser, exc)

        # Persist
        conn = self._get_conn()
        now = _now()
        for bm in all_bookmarks:
            conn.execute(
                """INSERT OR REPLACE INTO bookmarks
                   (url, title, folder, browser, scanned_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (bm.get("url", ""), bm.get("title", ""),
                 bm.get("folder", ""), bm.get("browser", ""), now),
            )
        conn.commit()
        LOG.info("Scanned %d bookmarks", len(all_bookmarks))
        return all_bookmarks

    # ------------------------------------------------------------------
    # 5. Recent files
    # ------------------------------------------------------------------

    def scan_recent_files(self) -> list[dict]:
        """Read Windows recent items from %APPDATA%/Microsoft/Windows/Recent."""
        recent: list[dict] = []
        recent_dir = Path.home() / "AppData/Roaming/Microsoft/Windows/Recent"
        if not recent_dir.is_dir():
            return recent

        conn = self._get_conn()
        now = _now()

        for lnk in recent_dir.glob("*.lnk"):
            try:
                accessed = datetime.fromtimestamp(
                    lnk.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                ext = lnk.suffix.lower()
                item = {"path": str(lnk), "extension": ext, "accessed_at": accessed}
                recent.append(item)
                conn.execute(
                    """INSERT OR REPLACE INTO recent_files
                       (path, extension, accessed_at, scanned_at)
                       VALUES (?, ?, ?, ?)""",
                    (str(lnk), ext, accessed, now),
                )
            except Exception:
                continue
        conn.commit()
        LOG.info("Scanned %d recent files", len(recent))
        return recent

    # ------------------------------------------------------------------
    # 6. Build user profile
    # ------------------------------------------------------------------

    def build_user_profile(self) -> UserProfile:
        """Infer who the user is from scanned environment data."""
        # Open a fresh connection for this call — safe across threads
        conn = _init_db(self.db_path)

        # App categories
        rows = conn.execute("SELECT category, COUNT(*) FROM apps GROUP BY category").fetchall()
        category_counts = {r[0]: r[1] for r in rows}

        # Preferred tools: top apps by category
        top_app_rows = conn.execute(
            "SELECT name FROM apps ORDER BY category, name LIMIT 20"
        ).fetchall()
        preferred_tools = [r[0] for r in top_app_rows]

        # Code languages
        lang_rows = conn.execute("SELECT languages FROM code_projects").fetchall()
        lang_set: set[str] = set()
        for row in lang_rows:
            try:
                langs = json.loads(row[0])
                lang_set.update(langs)
            except Exception:
                pass

        # Expertise from document keywords
        kw_rows = conn.execute("SELECT keywords FROM documents").fetchall()
        keyword_freq: dict[str, int] = {}
        for row in kw_rows:
            try:
                kws = json.loads(row[0])
                for kw in kws:
                    keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
            except Exception:
                pass
        top_keywords = sorted(keyword_freq.items(), key=lambda x: -x[1])
        expertise = [kw for kw, _ in top_keywords[:10]]

        # OS info
        os_info = f"{platform.system()} {platform.release()}"

        # Name: try $USERNAME env var
        name = os.environ.get("USERNAME") or os.environ.get("USER") or ""

        profile = UserProfile(
            name=name,
            expertise_areas=expertise,
            preferred_tools=preferred_tools[:15],
            os=os_info,
            code_languages=sorted(lang_set),
            language="English",   # could be extended with locale detection
        )

        self._profile = profile
        self.save_to_disk()
        return profile

    # ------------------------------------------------------------------
    # 7. Context for query
    # ------------------------------------------------------------------

    def get_context_for_query(self, query: str) -> str:
        """Return environment context relevant to the query (for RAG injection)."""
        conn = self._get_conn()
        query_lower = query.lower()
        query_words = set(re.findall(r"[a-zA-Z]{3,}", query_lower))
        context_parts: list[str] = []

        # Matching apps
        app_rows = conn.execute("SELECT name, category FROM apps").fetchall()
        matching_apps = [
            f"{r[0]} ({r[1]})" for r in app_rows
            if any(w in r[0].lower() for w in query_words)
        ][:5]
        if matching_apps:
            context_parts.append("Installed apps: " + ", ".join(matching_apps))

        # Matching documents
        doc_rows = conn.execute("SELECT path, summary, keywords FROM documents").fetchall()
        for path, summary, kw_json in doc_rows:
            try:
                kws = set(json.loads(kw_json))
            except Exception:
                kws = set()
            if kws & query_words:
                context_parts.append(f"Document: {Path(path).name} — {summary}")
            if len(context_parts) >= 5:
                break

        # Code projects
        proj_rows = conn.execute("SELECT path, languages FROM code_projects").fetchall()
        for path, lang_json in proj_rows:
            try:
                langs = json.loads(lang_json)
            except Exception:
                langs = []
            if any(w in Path(path).name.lower() for w in query_words) or \
               any(w in l.lower() for l in langs for w in query_words):
                context_parts.append(f"Code project: {Path(path).name} ({', '.join(langs)})")
            if len(context_parts) >= 8:
                break

        # Bookmarks
        bm_rows = conn.execute("SELECT title, url FROM bookmarks").fetchall()
        for title, url in bm_rows:
            if any(w in title.lower() for w in query_words):
                context_parts.append(f"Bookmark: {title} ({url[:60]})")
            if len(context_parts) >= 10:
                break

        if not context_parts:
            return "(no specific environment context found)"
        return "\n".join(context_parts)

    # ------------------------------------------------------------------
    # 8. Full initial scan
    # ------------------------------------------------------------------

    def scan_environment(self) -> dict:
        """Run all scanners and return a summary dict."""
        LOG.info("Starting full environment scan…")
        apps = self.scan_apps()
        docs = self.scan_documents()
        projects = self.scan_code_projects()
        bookmarks = self.scan_bookmarks()
        recent = self.scan_recent_files()
        profile = self.build_user_profile()

        summary = {
            "apps_found": len(apps),
            "documents_found": len(docs),
            "code_projects_found": len(projects),
            "bookmarks_found": len(bookmarks),
            "recent_files_found": len(recent),
            "user_profile": profile.to_dict(),
            "scanned_at": _now(),
        }
        LOG.info("Environment scan complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # 9. Incremental update
    # ------------------------------------------------------------------

    def incremental_update(self) -> dict:
        """Re-scan only new or changed files/apps."""
        LOG.info("Running incremental environment update…")
        new_docs = self.scan_documents()
        new_apps = self.scan_apps()
        return {
            "new_documents": len(new_docs),
            "apps_refreshed": len(new_apps),
            "updated_at": _now(),
        }

    # ------------------------------------------------------------------
    # 10. Background learning
    # ------------------------------------------------------------------

    def start_background_learning(self, interval_hours: float = 6.0) -> None:
        """Spawn a background daemon thread that periodically rescans."""
        if self._bg_thread and self._bg_thread.is_alive():
            LOG.info("Background learning already running")
            return

        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.incremental_update()
                except Exception as exc:
                    LOG.error("Background learning error: %s", exc)
                self._stop_event.wait(timeout=interval_hours * 3600)

        self._bg_thread = threading.Thread(target=_loop, daemon=True, name="aria-env-learner")
        self._bg_thread.start()
        LOG.info("Background learning started (interval=%.1fh)", interval_hours)

    def stop_background_learning(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_disk(self) -> None:
        """Persist user profile to JSON."""
        if self._profile is None:
            return
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(
            json.dumps(self._profile.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOG.debug("Profile saved to %s", self.profile_path)

    def load_from_disk(self) -> Optional[UserProfile]:
        """Load previously saved user profile from JSON."""
        if self.profile_path.is_file():
            try:
                data = json.loads(self.profile_path.read_text(encoding="utf-8"))
                self._profile = UserProfile.from_dict(data)
                return self._profile
            except Exception as exc:
                LOG.warning("Failed to load profile: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Natural-language entry point
    # ------------------------------------------------------------------

    async def run_nl(self, query: str) -> dict:
        """
        Async natural-language entry point for the agent bus.

        Recognised intents:
          - "scan" / "full scan" — triggers scan_environment()
          - "update" / "refresh"  — triggers incremental_update()
          - "profile"             — returns user profile
          - "context <query>"     — returns environment context for the query
          - anything else         — returns context for the query text
        """
        q = query.strip().lower()

        if any(kw in q for kw in ("full scan", "scan everything", "scan environment")):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.scan_environment)
            return result

        if any(kw in q for kw in ("update", "refresh", "incremental")):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.incremental_update)
            return result

        if "profile" in q:
            profile = self._profile or self.load_from_disk() or self.build_user_profile()
            return profile.to_dict()

        # Default: context lookup
        context = self.get_context_for_query(query)
        return {"context": context, "query": query}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.stop_background_learning()
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "EnvironmentLearner":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _categorise_app(name: str) -> str:
    name_lower = name.lower()
    for category, keywords in APP_CATEGORIES.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "other"


def _walk_limited(root: Path, max_depth: int) -> list[Path]:
    """Yield directories up to max_depth deep."""
    results: list[Path] = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        results.append(current)
        if depth < max_depth:
            try:
                for child in current.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        stack.append((child, depth + 1))
            except PermissionError:
                pass
    return results


def _detect_languages(project_path: Path) -> list[str]:
    """Infer programming languages from file extensions present in a project."""
    EXT_LANG = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".java": "Java", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
        ".php": "PHP", ".c": "C", ".cpp": "C++", ".cs": "C#",
        ".swift": "Swift", ".kt": "Kotlin", ".scala": "Scala",
        ".r": "R", ".m": "MATLAB", ".sh": "Shell", ".ps1": "PowerShell",
        ".html": "HTML", ".css": "CSS", ".sql": "SQL",
    }
    found: set[str] = set()
    try:
        for f in project_path.rglob("*"):
            if f.is_file():
                lang = EXT_LANG.get(f.suffix.lower())
                if lang:
                    found.add(lang)
            if len(found) >= 8:
                break
    except PermissionError:
        pass
    return sorted(found)


def _extract_bookmarks(
    node: Any,
    browser: str,
    folder: str,
    out: list[dict],
) -> None:
    """Recursively extract bookmarks from Chrome/Edge bookmark JSON node."""
    if not isinstance(node, dict):
        return
    node_type = node.get("type")
    if node_type == "url":
        out.append({
            "title": node.get("name", ""),
            "url": node.get("url", ""),
            "folder": folder,
            "browser": browser,
        })
    elif node_type == "folder" or "children" in node:
        sub_folder = node.get("name", folder)
        for child in node.get("children", []):
            _extract_bookmarks(child, browser, sub_folder, out)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="ARIA Environment Learner")
    parser.add_argument(
        "command",
        nargs="?",
        default="scan",
        choices=["scan", "update", "profile", "context"],
        help="Command to run",
    )
    parser.add_argument("query", nargs="?", default="", help="Query for 'context' command")
    args = parser.parse_args()

    with EnvironmentLearner() as learner:
        if args.command == "scan":
            print(json.dumps(learner.scan_environment(), indent=2))
        elif args.command == "update":
            print(json.dumps(learner.incremental_update(), indent=2))
        elif args.command == "profile":
            profile = learner.load_from_disk() or learner.build_user_profile()
            print(json.dumps(profile.to_dict(), indent=2))
        elif args.command == "context":
            q = args.query or "programming python development"
            print(learner.get_context_for_query(q))
