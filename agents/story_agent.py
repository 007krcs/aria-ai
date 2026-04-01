"""
ARIA Story Agent — Book narrator and story teller
=================================================
When user uploads a PDF/EPUB/TXT book:
  - Extracts text content
  - Identifies title, author, chapters, key plot points
  - Creates a rich narrative retelling
  - Streams the story chapter by chapter

When user asks for a story by topic/genre:
  - Searches for relevant content via DuckDuckGo
  - Synthesizes a story from search results using LLM
  - Narrates it in an engaging style

Features:
  - Supports any language (ARIA auto-detects and responds in same language)
  - Can pause/resume storytelling
  - Can answer questions about the story mid-telling
  - Adjusts tone: dramatic, educational, children-friendly, thriller, etc.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import textwrap
import urllib.parse
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import requests

logger = logging.getLogger("aria.story")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StorySegment:
    segment_id: int
    title: str
    content: str
    word_count: int
    estimated_minutes: float   # at ~150 wpm reading aloud


@dataclass
class Book:
    title: str
    author: str
    language: str
    total_words: int
    segments: list
    summary: str
    themes: list


# ---------------------------------------------------------------------------
# Tone prompts
# ---------------------------------------------------------------------------

TONE_PROMPTS: dict = {
    "dramatic": (
        "You are a master theatrical storyteller. Narrate with rich emotion, vivid imagery, "
        "powerful voice, building tension and release. Use dramatic pauses and vivid language."
    ),
    "educational": (
        "You are a wise, engaging educator. Narrate clearly and informatively, "
        "explaining context, historical background, and key lessons in accessible language."
    ),
    "children": (
        "You are a warm, playful children's storyteller. Use simple words, "
        "short sentences, gentle humour, and a cosy, imaginative tone suitable for ages 5-10."
    ),
    "thriller": (
        "You are a gripping thriller narrator. Build suspense with short, punchy sentences, "
        "shadowy atmosphere, and a sense of urgency. Keep the listener on edge."
    ),
    "romance": (
        "You are an evocative romance storyteller. Narrate with warmth, emotional depth, "
        "sensory details, and a tender, heartfelt voice."
    ),
    "engaging": (
        "You are a captivating, versatile storyteller. Narrate with energy, clarity, "
        "emotional resonance, and a natural conversational flow that keeps the listener hooked."
    ),
    "comedy": (
        "You are a witty, light-hearted comedian storyteller. Use clever wordplay, "
        "comic timing, absurd comparisons, and a playful voice that keeps the audience laughing. "
        "Add jokes, puns, and funny observations wherever they fit naturally."
    ),
    "humor": (
        "You are a hilarious storyteller with brilliant comic timing. Retell this with exaggerated "
        "descriptions, unexpected twists, self-aware jokes, and laugh-out-loud moments. "
        "Imagine you're a stand-up comedian narrating a story — keep it funny from start to finish."
    ),
    "sarcastic": (
        "You are a dry, sarcastic narrator — think David Attenborough meets a bored millennial. "
        "Use deadpan delivery, ironic observations, and witty asides that make the audience smirk."
    ),
    "epic": (
        "You are an epic fantasy narrator with a booming voice. Describe everything with grand scale — "
        "heroes, battles, and stakes that shake the universe. Make even mundane events sound legendary."
    ),
    "casual": (
        "You are a chill friend retelling this story over coffee. Speak conversationally, use everyday "
        "language, a few filler words, and make it feel like a relaxed chat — not a formal recitation."
    ),
}


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class StoryAgent:
    """
    ARIA's story-telling and book-narration agent.
    Supports PDF, EPUB, TXT files and topic-based story generation.
    """

    def __init__(self, engine=None, search_agent=None):
        self.engine = engine                    # ARIA LLM engine (optional)
        self.search_agent = search_agent        # optional search helper
        self._paused = False
        self._story_state: dict = {}            # stores in-progress narration state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def narrate_file(
        self, filepath: str, tone: str = "engaging"
    ) -> AsyncGenerator[str, None]:
        """
        Extract text from PDF/TXT/EPUB, identify structure, narrate in chunks.
        Yields SSE-compatible text chunks.
        """
        try:
            raw_text = self._extract_text(filepath)
        except Exception as exc:
            yield f"[ARIA] Could not read file: {exc}\n"
            return

        if not raw_text.strip():
            yield "[ARIA] The file appears to be empty or unreadable.\n"
            return

        title, author = self._detect_title_author(raw_text, filepath)
        chapters = self._detect_chapters(raw_text)

        if not chapters:
            # No chapter structure — treat whole text as one story
            chapters = [{"title": "Story", "text": raw_text}]

        tone_key = self._normalise_tone(tone)
        yield f"# {title}\n*by {author}*\n\n"
        yield f"*Narrating in **{tone_key}** style — {len(chapters)} section(s) detected.*\n\n"
        yield "---\n\n"

        # Store state for pause/resume
        self._story_state = {
            "filepath": filepath,
            "tone": tone_key,
            "chapters": chapters,
            "current_chapter": 0,
            "title": title,
            "author": author,
        }

        for idx, chapter in enumerate(chapters):
            self._story_state["current_chapter"] = idx
            # Respect pause
            while self._paused:
                await asyncio.sleep(0.5)

            yield f"## {chapter['title']}\n\n"
            segments = self._chunk_for_narration(chapter["text"])

            context_summary = f"This is from '{title}' by {author}. Chapter: {chapter['title']}."

            for seg in segments:
                while self._paused:
                    await asyncio.sleep(0.5)
                try:
                    narrated = await self._narrate_chunk(seg.content, tone_key, context_summary)
                    yield narrated + "\n\n"
                    context_summary = seg.content[-200:]  # rolling context
                except Exception as exc:
                    logger.warning("Narration chunk failed: %s", exc)
                    yield seg.content + "\n\n"

        yield "\n---\n*[ARIA] End of narration.*\n"

    async def tell_story(
        self,
        topic: str,
        genre: str = "auto",
        language: str = "en",
    ) -> AsyncGenerator[str, None]:
        """
        Search for story content via DuckDuckGo, synthesise and narrate.
        Yields SSE-compatible text chunks.
        """
        tone_key = genre if genre in TONE_PROMPTS else "engaging"
        if genre == "auto":
            tone_key = self._detect_tone(topic)

        yield f"# Story: {topic.title()}\n\n"
        yield f"*Searching for story material...*\n\n"

        search_results = self._search_story_content(topic, language)

        if not search_results:
            # Fallback: ask LLM to generate directly
            yield "*No search results found — generating story from knowledge...*\n\n"
            source_text = f"Topic: {topic}"
        else:
            yield f"*Found {len(search_results)} sources. Crafting your story...*\n\n"
            source_text = "\n\n".join(
                f"Source {i+1}: {r['abstract']}" for i, r in enumerate(search_results[:5]) if r.get("abstract")
            )
            if not source_text.strip():
                source_text = f"Topic: {topic}"

        yield "---\n\n"

        # Break into coherent story segments
        segments = self._plan_story_segments(topic, genre, source_text)

        # Store state
        self._story_state = {
            "topic": topic,
            "tone": tone_key,
            "segments": segments,
            "current_segment": 0,
            "source_text": source_text,
        }

        for idx, segment in enumerate(segments):
            self._story_state["current_segment"] = idx
            while self._paused:
                await asyncio.sleep(0.5)

            yield f"## {segment['title']}\n\n"
            try:
                narrated = await self._narrate_chunk(
                    segment["content"], tone_key,
                    f"Story about {topic}. Section: {segment['title']}."
                )
                yield narrated + "\n\n"
            except Exception as exc:
                logger.warning("Story narration segment failed: %s", exc)
                yield segment["content"] + "\n\n"

        yield "\n---\n*[ARIA] The end. Hope you enjoyed the story!*\n"

    async def summarize_book(self, filepath: str) -> str:
        """Quick book summary without full narration."""
        try:
            raw_text = self._extract_text(filepath)
        except Exception as exc:
            return f"Could not read file: {exc}"

        if not raw_text.strip():
            return "The file appears to be empty or unreadable."

        title, author = self._detect_title_author(raw_text, filepath)
        chapters = self._detect_chapters(raw_text)
        chapter_titles = [c["title"] for c in chapters] if chapters else ["(No chapter structure detected)"]

        word_count = len(raw_text.split())
        reading_minutes = word_count // 200

        # Sample text from beginning, middle, end for summary
        excerpt_len = 1500
        beginning = raw_text[:excerpt_len]
        mid_start = max(0, len(raw_text) // 2 - excerpt_len // 2)
        middle = raw_text[mid_start: mid_start + excerpt_len]
        end = raw_text[-excerpt_len:]

        excerpt = f"BEGINNING:\n{beginning}\n\nMIDDLE:\n{middle}\n\nEND:\n{end}"

        prompt = (
            f"Based on the following excerpts from '{title}' by {author}, provide:\n"
            f"1. A concise 3-5 sentence plot summary\n"
            f"2. Main themes (3-5 bullet points)\n"
            f"3. Key characters mentioned\n"
            f"4. The overall tone/genre\n\n"
            f"Excerpts:\n{excerpt}"
        )

        try:
            summary_text = await self._llm_call(prompt)
        except Exception as exc:
            logger.warning("LLM summary failed: %s", exc)
            summary_text = (
                f"Could not generate AI summary. The book '{title}' by {author} "
                f"contains approximately {word_count:,} words ({reading_minutes} minutes reading time) "
                f"with {len(chapters)} detected chapters."
            )

        lines = [
            f"# Book Summary: {title}",
            f"**Author:** {author}",
            f"**Words:** {word_count:,} (~{reading_minutes} min reading)",
            f"**Chapters detected:** {len(chapters)}",
            f"",
            f"### Chapter List",
        ]
        for t in chapter_titles[:20]:
            lines.append(f"- {t}")
        if len(chapter_titles) > 20:
            lines.append(f"- *(and {len(chapter_titles) - 20} more...)*")

        lines += ["", "### AI Summary", summary_text]
        return "\n".join(lines)

    async def run_nl(
        self, query: str, uploaded_file: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Natural language interface — detects intent and routes to the right method.
        Yields SSE-compatible text chunks.
        """
        q_lower = query.lower().strip()

        # Detect language from query
        language = self._detect_language(query)

        # ── URL detected — fetch & narrate ────────────────────────────────────
        url_match = re.search(r"https?://[^\s]+", query)
        if url_match:
            url  = url_match.group(0).rstrip(".,!?)")
            tone = self._detect_tone(query)
            async for chunk in self.narrate_url(url, tone=tone):
                yield chunk
            return

        # Continue/resume
        if any(kw in q_lower for kw in ["continue", "resume", "go on", "keep going", "aage", "jaari"]):
            if self._story_state:
                self._paused = False
                async for chunk in self._resume_story():
                    yield chunk
                return
            else:
                yield "[ARIA] No story is currently in progress. Ask me to tell a story or narrate a book!\n"
                return

        # Pause
        if any(kw in q_lower for kw in ["pause", "stop", "wait", "ruk"]):
            self._paused = True
            yield "[ARIA] Story paused. Say 'continue' to resume.\n"
            return

        # Summarize book
        if any(kw in q_lower for kw in ["summarize", "summary", "summarise", "brief", "synopsis"]):
            if uploaded_file:
                summary = await self.summarize_book(uploaded_file)
                yield summary
                return
            else:
                yield "[ARIA] Please upload a book file for me to summarize.\n"
                return

        # Narrate uploaded file
        if uploaded_file and any(kw in q_lower for kw in [
            "narrate", "read", "tell", "story", "book", "read aloud", "read this"
        ]):
            tone = self._detect_tone(query)
            async for chunk in self.narrate_file(uploaded_file, tone=tone):
                yield chunk
            return

        # File uploaded but no explicit instruction — default to narrate
        if uploaded_file:
            tone = self._detect_tone(query)
            async for chunk in self.narrate_file(uploaded_file, tone=tone):
                yield chunk
            return

        # "Tell me a story about X"
        story_match = re.search(
            r"(?:tell|narrate|write|give|share|create|make|compose|generate)\s+"
            r"(?:me\s+)?(?:a\s+)?(?:story|tale|narrative|fable|anecdote)\s+(?:about\s+|on\s+|of\s+)?(.+)",
            q_lower
        )
        if story_match:
            topic = story_match.group(1).strip().rstrip("?.!")
            genre = self._detect_genre_from_query(query)
            async for chunk in self.tell_story(topic, genre=genre, language=language):
                yield chunk
            return

        # "Story in Hindi" / "story in Tamil" etc.
        lang_story = re.search(
            r"(?:story|tale|narrative)\s+in\s+(\w+)",
            q_lower
        )
        if lang_story:
            lang_name = lang_story.group(1)
            lang_code = self._lang_name_to_code(lang_name)
            # Extract topic from the rest
            topic = re.sub(r"(?:story|tale|narrative)\s+in\s+\w+", "", q_lower).strip()
            topic = topic or "an interesting adventure"
            async for chunk in self.tell_story(topic, language=lang_code):
                yield chunk
            return

        # Question about current story
        question_words = ["who", "what", "why", "how", "when", "where", "explain", "describe"]
        if self._story_state and any(q_lower.startswith(w) for w in question_words):
            answer = await self._answer_story_question(query)
            yield answer + "\n"
            return

        # Fallback: treat entire query as story topic
        yield "[ARIA] I'll tell you a story based on that!\n\n"
        async for chunk in self.tell_story(query.strip(), language=language):
            yield chunk

    # ------------------------------------------------------------------
    # URL fetching
    # ------------------------------------------------------------------

    def _fetch_url(self, url: str) -> str:
        """
        Fetch a webpage and extract readable text.
        Strips HTML tags, scripts, navigation boilerplate.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        # Try BeautifulSoup first
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Remove noise elements
            for tag in soup(["script", "style", "nav", "header", "footer",
                              "aside", "advertisement", "noscript", "meta"]):
                tag.decompose()
            # Prefer article/main body
            main = (
                soup.find("article") or
                soup.find("main") or
                soup.find(id=re.compile(r"(content|article|story|post|body)", re.I)) or
                soup.find(class_=re.compile(r"(content|article|story|post|body)", re.I)) or
                soup.body
            )
            if main:
                text = main.get_text(separator="\n")
            else:
                text = soup.get_text(separator="\n")
            # Clean up whitespace
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return "\n".join(lines)
        except ImportError:
            pass

        # Fallback: strip HTML with regex
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 20]
        return "\n".join(lines)

    async def narrate_url(
        self, url: str, tone: str = "engaging"
    ) -> AsyncGenerator[str, None]:
        """
        Fetch a URL, extract text, detect structure, and narrate in the given tone.
        Works for news articles, blog posts, Wikipedia pages, short stories, etc.
        """
        yield f"*Fetching content from URL…*\n\n"
        try:
            raw_text = self._fetch_url(url)
        except Exception as exc:
            yield f"[ARIA] Could not fetch URL: {exc}\n"
            return

        if not raw_text or len(raw_text) < 100:
            yield "[ARIA] The page appears to be empty or could not be read.\n"
            return

        # Try to detect a title from the text
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        title = lines[0][:120] if lines else url

        word_count = len(raw_text.split())
        tone_key   = self._normalise_tone(tone)

        yield f"# {title}\n\n"
        yield f"*Narrating in **{tone_key}** tone · ~{word_count:,} words fetched*\n\n"
        yield "---\n\n"

        # Store state
        self._story_state = {
            "url":    url,
            "tone":   tone_key,
            "title":  title,
        }

        # If long enough, detect chapters/sections; else treat as one block
        chapters = self._detect_chapters(raw_text)
        if not chapters:
            chapters = [{"title": "Content", "text": raw_text}]

        context = f"Article/story from {url}. Title: {title}."

        for chapter in chapters:
            yield f"## {chapter['title']}\n\n"
            segments = self._chunk_for_narration(chapter["text"], chunk_size=400)
            for seg in segments:
                while self._paused:
                    await asyncio.sleep(0.5)
                try:
                    narrated = await self._narrate_chunk(seg.content, tone_key, context)
                    yield narrated + "\n\n"
                    context = seg.content[-150:]  # rolling context
                except Exception as exc:
                    logger.warning("Narration chunk failed: %s", exc)
                    yield seg.content + "\n\n"

        yield "\n---\n*[ARIA] End of narration.*\n"

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, filepath: str) -> str:
        """
        Extract plain text from PDF, EPUB, or TXT.
        Falls back gracefully through multiple methods.
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".txt":
            return self._extract_txt(filepath)
        elif ext == ".pdf":
            return self._extract_pdf(filepath)
        elif ext == ".epub":
            return self._extract_epub(filepath)
        else:
            # Try as text first
            try:
                return self._extract_txt(filepath)
            except Exception:
                pass
            # Try as PDF
            try:
                return self._extract_pdf(filepath)
            except Exception:
                pass
            # Binary fallback
            try:
                with open(filepath, "rb") as f:
                    raw = f.read()
                return raw.decode("utf-8", errors="replace")
            except Exception as exc:
                raise RuntimeError(f"Could not extract text from {filepath}: {exc}")

    def _extract_txt(self, filepath: str) -> str:
        """Read plain text file with encoding fallback."""
        for enc in ("utf-8", "utf-16", "latin-1", "cp1252"):
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        with open(filepath, "rb") as f:
            return f.read().decode("utf-8", errors="replace")

    def _extract_pdf(self, filepath: str) -> str:
        """Extract text from PDF using pdfminer.six, then PyPDF2, then raw."""
        # Method 1: pdfminer.six
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(filepath)
            if text and text.strip():
                return text
        except ImportError:
            logger.debug("pdfminer.six not installed, trying PyPDF2")
        except Exception as exc:
            logger.warning("pdfminer extraction failed: %s", exc)

        # Method 2: PyPDF2
        try:
            import PyPDF2
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for page in reader.pages:
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pages.append("")
            text = "\n".join(pages)
            if text.strip():
                return text
        except ImportError:
            logger.debug("PyPDF2 not installed")
        except Exception as exc:
            logger.warning("PyPDF2 extraction failed: %s", exc)

        # Method 3: Raw byte extraction — last resort
        try:
            with open(filepath, "rb") as f:
                raw = f.read()
            # Extract readable ASCII/UTF-8 sequences from binary
            text = re.sub(rb"[^\x20-\x7e\n\r\t]", b" ", raw).decode("ascii", errors="ignore")
            # Collapse whitespace
            text = re.sub(r"\s{3,}", "\n", text)
            return text
        except Exception as exc:
            raise RuntimeError(f"All PDF extraction methods failed for {filepath}: {exc}")

    def _extract_epub(self, filepath: str) -> str:
        """Extract text from EPUB using ebooklib or zipfile fallback."""
        # Method 1: ebooklib
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser

            class _HTMLStripper(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._parts = []
                def handle_data(self, data):
                    self._parts.append(data)
                def get_text(self):
                    return " ".join(self._parts)

            book = epub.read_epub(filepath)
            parts = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode("utf-8", errors="replace")
                    stripper = _HTMLStripper()
                    stripper.feed(content)
                    parts.append(stripper.get_text())
            return "\n\n".join(parts)
        except ImportError:
            logger.debug("ebooklib not installed, trying zipfile method")
        except Exception as exc:
            logger.warning("ebooklib extraction failed: %s", exc)

        # Method 2: EPUB is a ZIP — extract HTML files directly
        try:
            import zipfile
            from html.parser import HTMLParser

            class _StripHTML(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._buf = []
                def handle_data(self, d):
                    self._buf.append(d)
                def get_text(self):
                    return " ".join(self._buf)

            parts = []
            with zipfile.ZipFile(filepath, "r") as z:
                for name in z.namelist():
                    if name.endswith((".html", ".htm", ".xhtml")):
                        try:
                            html = z.read(name).decode("utf-8", errors="replace")
                            stripper = _StripHTML()
                            stripper.feed(html)
                            parts.append(stripper.get_text())
                        except Exception:
                            pass
            return "\n\n".join(parts)
        except Exception as exc:
            raise RuntimeError(f"EPUB extraction failed for {filepath}: {exc}")

    # ------------------------------------------------------------------
    # Structure detection
    # ------------------------------------------------------------------

    def _detect_title_author(self, text: str, filepath: str) -> tuple:
        """Heuristically extract title and author from beginning of text."""
        lines = [ln.strip() for ln in text[:2000].splitlines() if ln.strip()]
        title = "Unknown Title"
        author = "Unknown Author"

        if lines:
            # First non-empty line is likely the title
            title = lines[0][:100]

        # Author patterns
        author_patterns = [
            r"(?:by|author|written by|translated by)[:\s]+([A-Z][a-zA-Z\s\.\-]+)",
            r"^([A-Z][a-z]+ [A-Z][a-z]+)$",  # "First Last" on its own line
        ]
        for ln in lines[:20]:
            for pat in author_patterns:
                m = re.search(pat, ln, re.IGNORECASE)
                if m:
                    author = m.group(1).strip()
                    break

        # Fallback: use filename as title
        if title == "Unknown Title":
            title = os.path.splitext(os.path.basename(filepath))[0].replace("_", " ").title()

        return title, author

    def _detect_chapters(self, text: str) -> list:
        """
        Detect chapter boundaries using common heading patterns.
        Returns list of {'title': str, 'text': str}.
        """
        patterns = [
            r"(?m)^(CHAPTER\s+[IVXLCDM\d]+[^\n]*)",
            r"(?m)^(Chapter\s+\d+[^\n]*)",
            r"(?m)^(PART\s+[IVXLCDM\d]+[^\n]*)",
            r"(?m)^(Part\s+\d+[^\n]*)",
            r"(?m)^(\d+\.\s+[A-Z][^\n]{3,60})$",
            r"(?m)^(#{1,3}\s+[^\n]+)",   # Markdown headings
        ]

        matches = []
        for pat in patterns:
            found = [(m.start(), m.group(1).strip()) for m in re.finditer(pat, text)]
            if found:
                matches = found
                break

        if not matches or len(matches) < 2:
            return []

        chapters = []
        for i, (start, title) in enumerate(matches):
            end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            chapter_text = text[start:end].strip()
            # Remove the heading line from body text
            body = chapter_text[len(title):].strip()
            chapters.append({"title": title, "text": body})

        return chapters

    def _chunk_for_narration(self, text: str, chunk_size: int = 500) -> list:
        """
        Split text into narration-sized segments on natural paragraph breaks.
        Returns list of StorySegment.
        """
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        segments = []
        current_parts = []
        current_words = 0
        seg_id = 0

        for para in paragraphs:
            words = len(para.split())
            if current_words + words > chunk_size and current_parts:
                content = "\n\n".join(current_parts)
                wc = len(content.split())
                segments.append(StorySegment(
                    segment_id=seg_id,
                    title=f"Segment {seg_id + 1}",
                    content=content,
                    word_count=wc,
                    estimated_minutes=round(wc / 150, 1),
                ))
                seg_id += 1
                current_parts = [para]
                current_words = words
            else:
                current_parts.append(para)
                current_words += words

        if current_parts:
            content = "\n\n".join(current_parts)
            wc = len(content.split())
            segments.append(StorySegment(
                segment_id=seg_id,
                title=f"Segment {seg_id + 1}",
                content=content,
                word_count=wc,
                estimated_minutes=round(wc / 150, 1),
            ))

        return segments if segments else [
            StorySegment(0, "Full Text", text, len(text.split()), round(len(text.split()) / 150, 1))
        ]

    # ------------------------------------------------------------------
    # Tone / genre / language detection
    # ------------------------------------------------------------------

    def _detect_tone(self, query: str) -> str:
        """Auto-detect tone from query keywords."""
        q = query.lower()
        if any(w in q for w in ["child", "kids", "children", "bedtime", "fairy"]):
            return "children"
        if any(w in q for w in ["thriller", "horror", "scary", "mystery", "suspense", "dark"]):
            return "thriller"
        if any(w in q for w in ["romance", "love", "romantic", "heart"]):
            return "romance"
        if any(w in q for w in ["educational", "learn", "history", "science", "facts"]):
            return "educational"
        if any(w in q for w in ["epic", "legendary", "grand", "fantasy"]):
            return "epic"
        if any(w in q for w in ["dramatic", "powerful", "intense"]):
            return "dramatic"
        if any(w in q for w in ["sarcastic", "dry", "deadpan", "ironic"]):
            return "sarcastic"
        if any(w in q for w in ["casual", "chill", "relaxed", "friendly", "simple"]):
            return "casual"
        if any(w in q for w in ["humor", "humour", "funny", "laugh", "joke", "hilarious", "comedy"]):
            return "humor"
        return "engaging"

    def _detect_genre_from_query(self, query: str) -> str:
        """Extract genre hint from query phrasing."""
        q = query.lower()
        genres = {
            "thriller": ["thriller", "horror", "scary", "suspense", "mystery"],
            "romance": ["romance", "romantic", "love story"],
            "children": ["children", "kids", "bedtime", "fairy tale"],
            "educational": ["educational", "history", "science", "learn about"],
            "dramatic": ["dramatic", "epic"],
            "comedy": ["funny", "comedy", "humorous", "hilarious"],
        }
        for genre, keywords in genres.items():
            if any(kw in q for kw in keywords):
                return genre
        return "auto"

    def _normalise_tone(self, tone: str) -> str:
        tone = tone.lower().strip()
        # Accept aliases
        _aliases = {"funny": "humor", "laugh": "humor", "comic": "comedy",
                    "sad": "dramatic", "intense": "dramatic", "grand": "epic",
                    "chill": "casual", "normal": "engaging", "plain": "engaging"}
        tone = _aliases.get(tone, tone)
        return tone if tone in TONE_PROMPTS else "engaging"

    def _detect_language(self, query: str) -> str:
        """Lightweight language detection based on script/common words."""
        # Hindi/Devanagari
        if re.search(r"[\u0900-\u097F]", query):
            return "hi"
        # Arabic
        if re.search(r"[\u0600-\u06FF]", query):
            return "ar"
        # Chinese
        if re.search(r"[\u4E00-\u9FFF]", query):
            return "zh"
        # Simple keyword hints
        q = query.lower()
        if any(w in q for w in ["hindi", "हिंदी", "mujhe", "ek", "kahani"]):
            return "hi"
        if any(w in q for w in ["tamil", "telugu", "kannada", "malayalam"]):
            return "ta"
        return "en"

    def _lang_name_to_code(self, name: str) -> str:
        mapping = {
            "hindi": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te",
            "kannada": "kn", "malayalam": "ml", "marathi": "mr", "gujarati": "gu",
            "punjabi": "pa", "urdu": "ur", "arabic": "ar", "french": "fr",
            "spanish": "es", "german": "de", "portuguese": "pt", "japanese": "ja",
            "chinese": "zh", "korean": "ko", "russian": "ru", "italian": "it",
        }
        return mapping.get(name.lower(), "en")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_story_content(self, topic: str, language: str = "en") -> list:
        """
        Search DuckDuckGo for story content.
        Falls back to Project Gutenberg if topic looks like a classic.
        """
        results = []

        # Try DuckDuckGo Instant Answer API
        try:
            q = urllib.parse.quote_plus(f"{topic} story")
            url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Main abstract
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", topic),
                    "abstract": data["Abstract"],
                    "url": data.get("AbstractURL", ""),
                })

            # Related topics
            for rt in data.get("RelatedTopics", [])[:5]:
                if isinstance(rt, dict) and rt.get("Text"):
                    results.append({
                        "title": rt.get("FirstURL", ""),
                        "abstract": rt["Text"],
                        "url": rt.get("FirstURL", ""),
                    })
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)

        # If no results or classic book — try Project Gutenberg
        if not results or any(kw in topic.lower() for kw in [
            "shakespeare", "dickens", "austen", "tolstoy", "dostoevsky",
            "twain", "poe", "wilde", "homer", "dante"
        ]):
            try:
                gutenberg_results = self._search_gutenberg(topic)
                results.extend(gutenberg_results)
            except Exception as exc:
                logger.warning("Gutenberg search failed: %s", exc)

        return results

    def _search_gutenberg(self, topic: str) -> list:
        """Search Project Gutenberg for classic books."""
        q = urllib.parse.quote_plus(topic)
        url = f"https://gutendex.com/books/?search={q}"
        headers = {"User-Agent": "ARIA/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for book in data.get("results", [])[:3]:
            title  = book.get("title", "")
            authors = ", ".join(a.get("name", "") for a in book.get("authors", []))
            results.append({
                "title": title,
                "abstract": f"'{title}' by {authors} — available on Project Gutenberg.",
                "url": f"https://gutenberg.org/ebooks/{book.get('id', '')}",
            })
        return results

    # ------------------------------------------------------------------
    # Story planning
    # ------------------------------------------------------------------

    def _plan_story_segments(self, topic: str, genre: str, source_text: str) -> list:
        """
        Given a topic and source material, create a structured story plan.
        Returns list of {'title': str, 'content': str}.
        """
        # Heuristic story structure: Opening, Rising Action, Climax, Resolution
        structures = {
            "thriller": ["The Setup", "A Dark Discovery", "Mounting Tension", "The Confrontation", "Aftermath"],
            "romance":  ["First Meeting", "Growing Feelings", "The Obstacle", "The Turning Point", "Together At Last"],
            "children": ["Once Upon a Time", "The Adventure Begins", "The Challenge", "The Magic Moment", "Happily Ever After"],
            "educational": ["Introduction", "The Context", "Key Events", "What We Learned", "Conclusion"],
            "dramatic": ["The World Before", "The Inciting Moment", "Rising Conflict", "The Crisis", "Resolution"],
            "engaging": ["The Beginning", "Complications", "The Turning Point", "The Climax", "The End"],
        }
        tone_key = genre if genre in structures else "engaging"
        section_titles = structures[tone_key]

        # Distribute source_text across sections (or just seed prompts)
        words = source_text.split()
        chunk_size = max(len(words) // len(section_titles), 1)
        segments = []
        for i, title in enumerate(section_titles):
            start = i * chunk_size
            end   = start + chunk_size
            content = " ".join(words[start:end]) if start < len(words) else f"Continue the story of {topic}."
            segments.append({"title": title, "content": content})
        return segments

    # ------------------------------------------------------------------
    # LLM narration
    # ------------------------------------------------------------------

    async def _narrate_chunk(self, chunk: str, tone: str, context: str) -> str:
        """
        Use ARIA's LLM engine to narrate one chunk in the specified tone.
        Falls back to lightly formatted original text if no engine available.
        """
        tone_instruction = TONE_PROMPTS.get(tone, TONE_PROMPTS["engaging"])
        prompt = (
            f"{tone_instruction}\n\n"
            f"Context: {context}\n\n"
            f"Narrate the following passage in your storytelling style. "
            f"Keep it engaging, immersive, and true to the source material. "
            f"Do not add a title or heading — just the narration:\n\n"
            f"{chunk}"
        )
        return await self._llm_call(prompt)

    async def _llm_call(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call ARIA's LLM engine if available, otherwise return a formatted fallback.
        """
        if self.engine is not None:
            try:
                # Support both async and sync engines
                if asyncio.iscoroutinefunction(getattr(self.engine, "generate", None)):
                    return await self.engine.generate(prompt, max_tokens=max_tokens)
                elif hasattr(self.engine, "generate"):
                    # Pass max_tokens if the engine signature supports it
                    try:
                        return self.engine.generate(prompt, max_tokens=max_tokens)
                    except TypeError:
                        return self.engine.generate(prompt)
                elif asyncio.iscoroutinefunction(self.engine):
                    return await self.engine(prompt)
                else:
                    return self.engine(prompt)
            except Exception as exc:
                logger.warning("LLM call failed: %s — using fallback", exc)

        # Graceful fallback: wrap text nicely without LLM
        wrapped = textwrap.fill(prompt.split("narration:\n\n", 1)[-1], width=80)
        return wrapped

    async def _answer_story_question(self, question: str) -> str:
        """Answer a question about the currently loaded story."""
        state = self._story_state
        if not state:
            return "No story is currently loaded."

        context_info = ""
        if "title" in state:
            context_info = f"Book: '{state['title']}' by {state.get('author', 'unknown author')}."
        elif "topic" in state:
            context_info = f"Story about: {state['topic']}."

        prompt = (
            f"You are ARIA, a knowledgeable storytelling assistant. "
            f"{context_info}\n\n"
            f"The user is asking: {question}\n\n"
            f"Answer helpfully based on the story context."
        )
        try:
            return await self._llm_call(prompt)
        except Exception as exc:
            return f"I couldn't retrieve an answer right now: {exc}"

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    async def _resume_story(self) -> AsyncGenerator[str, None]:
        """Resume narration from stored state."""
        state = self._story_state
        if not state:
            yield "[ARIA] No story in progress.\n"
            return

        self._paused = False
        yield "[ARIA] Resuming story...\n\n"

        # File-based resume
        if "filepath" in state:
            chapters = state.get("chapters", [])
            current = state.get("current_chapter", 0)
            tone    = state.get("tone", "engaging")
            title   = state.get("title", "Story")
            author  = state.get("author", "")
            context = f"Continuing '{title}' by {author}."

            for idx in range(current, len(chapters)):
                state["current_chapter"] = idx
                chapter = chapters[idx]
                while self._paused:
                    await asyncio.sleep(0.5)
                yield f"## {chapter['title']}\n\n"
                for seg in self._chunk_for_narration(chapter["text"]):
                    while self._paused:
                        await asyncio.sleep(0.5)
                    try:
                        narrated = await self._narrate_chunk(seg.content, tone, context)
                        yield narrated + "\n\n"
                    except Exception:
                        yield seg.content + "\n\n"
            yield "\n---\n*[ARIA] End of narration.*\n"

        # Topic-based resume
        elif "topic" in state:
            segments = state.get("segments", [])
            current  = state.get("current_segment", 0)
            tone     = state.get("tone", "engaging")
            topic    = state.get("topic", "story")
            for idx in range(current, len(segments)):
                state["current_segment"] = idx
                seg = segments[idx]
                while self._paused:
                    await asyncio.sleep(0.5)
                yield f"## {seg['title']}\n\n"
                try:
                    narrated = await self._narrate_chunk(
                        seg["content"], tone, f"Story about {topic}."
                    )
                    yield narrated + "\n\n"
                except Exception:
                    yield seg["content"] + "\n\n"
            yield "\n---\n*[ARIA] The end.*\n"
        else:
            yield "[ARIA] Cannot resume — state is incomplete.\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main():
    import sys

    agent = StoryAgent()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            print(f"Narrating file: {arg}\n")
            async for chunk in agent.narrate_file(arg):
                print(chunk, end="", flush=True)
        else:
            query = " ".join(sys.argv[1:])
            print(f"Generating story: {query}\n")
            async for chunk in agent.run_nl(query):
                print(chunk, end="", flush=True)
    else:
        print("Usage: python story_agent.py 'tell me a thriller story about a heist'")
        print("       python story_agent.py /path/to/book.pdf")


if __name__ == "__main__":
    asyncio.run(_main())
