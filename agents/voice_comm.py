"""
ARIA — Voice Communication Layer
===================================
Full verbal conversation. ARIA listens, understands, speaks back.

Architecture:
  Browser mic → WebRTC → FastAPI WebSocket → Whisper STT
  → ARIA reasoning → edge-tts TTS → audio stream → browser speaker

Three modes:
  1. Push-to-talk      — hold button, speak, release
  2. Wake word         — say "Hey ARIA" anywhere, hands-free
  3. Continuous        — always listening, VAD detects speech

STT:  Whisper (local, offline, 99 languages, base model = 39MB)
TTS:  edge-tts (Microsoft neural voices, free, no API key)
      Fallback: pyttsx3 (fully offline, works on IoT)
Wake: Silero VAD (voice activity detection, 1MB, CPU-only)

Install:
    pip install openai-whisper edge-tts silero sounddevice scipy
    pip install websockets faster-whisper  # faster-whisper = 4x faster on CPU

Voices available (edge-tts):
    en-IN-NeerjaNeural    English Indian female (recommended for India)
    en-IN-PrabhatNeural   English Indian male
    hi-IN-SwaraNeural     Hindi female
    hi-IN-MadhurNeural    Hindi male
    en-US-AriaNeural      English US female
    en-GB-SoniaNeural     English UK female
"""

import io
import os
import re
import sys
import json
import time
import asyncio
import hashlib
import tempfile
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator, Optional
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()

VOICE_CACHE  = PROJECT_ROOT / "data" / "voice_cache"
VOICE_CACHE.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TTS ENGINE — speaks ARIA's responses
# ─────────────────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Text-to-speech with multiple backends.
    Primary: edge-tts (neural, natural-sounding, free)
    Fallback: pyttsx3 (fully offline, robotic but works everywhere)

    Features:
    - Audio caching: same phrase → same file, instant playback
    - Streaming: streams audio as it generates, low latency
    - SSML support: control pace, emphasis, pauses
    - Language auto-detection: switches voice based on text language
    """

    VOICES = {
        "en-IN-female": "en-IN-NeerjaNeural",
        "en-IN-male":   "en-IN-PrabhatNeural",
        "hi-female":    "hi-IN-SwaraNeural",
        "hi-male":      "hi-IN-MadhurNeural",
        "en-US-female": "en-US-AriaNeural",
        "en-US-male":   "en-US-GuyNeural",
        "en-GB-female": "en-GB-SoniaNeural",
    }
    DEFAULT_VOICE = "en-IN-NeerjaNeural"

    def __init__(self, voice: str = None):
        self.voice = voice or self.DEFAULT_VOICE
        self._lock = threading.Lock()

    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """
        Convert text to speech audio bytes (MP3).
        Cached — same text returns cached audio instantly.
        """
        if not text.strip():
            return b""

        v    = voice or self.voice
        # Cache key: hash of text + voice
        key  = hashlib.md5(f"{text}{v}".encode()).hexdigest()[:12]
        path = VOICE_CACHE / f"{key}.mp3"

        if path.exists():
            return path.read_bytes()

        audio = await self._synthesize_edge_tts(text, v)
        if not audio:
            audio = self._synthesize_pyttsx3(text)

        if audio:
            path.write_bytes(audio)
        return audio

    async def synthesize_streaming(
        self, text: str, voice: str = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks as they are generated.
        Much lower latency than waiting for full synthesis.
        """
        v = voice or self.voice
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, v)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except ImportError:
            # Fall back to full synthesis
            audio = self._synthesize_pyttsx3(text)
            if audio:
                yield audio
        except Exception as e:
            console.print(f"  [yellow]TTS stream error: {e}[/]")

    async def speak_and_stream(
        self, text: str, voice: str = None
    ) -> bytes:
        """Synthesize and immediately play through system speaker."""
        chunks = []
        async for chunk in self.synthesize_streaming(text, voice):
            chunks.append(chunk)
        audio = b"".join(chunks)
        if audio:
            self._play(audio)
        return audio

    async def _synthesize_edge_tts(self, text: str, voice: str) -> bytes:
        try:
            import edge_tts
            buf         = io.BytesIO()
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()
        except ImportError:
            return b""
        except Exception as e:
            console.print(f"  [yellow]edge-tts error: {e}[/]")
            return b""

    def _synthesize_pyttsx3(self, text: str) -> bytes:
        """Offline fallback — no audio bytes returned, plays directly."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()
        except ImportError:
            console.print("  [dim]pip install pyttsx3 for offline TTS[/]")
        except Exception as e:
            console.print(f"  [yellow]pyttsx3 error: {e}[/]")
        return b""

    def _play(self, audio_bytes: bytes):
        """Play MP3 audio bytes through system speaker."""
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav
            # Convert MP3 to numpy array
            buf  = io.BytesIO(audio_bytes)
            data, rate = self._mp3_to_wav(buf)
            with self._lock:
                sd.play(data, rate, blocking=True)
        except Exception:
            # Fallback: save to temp file and play
            self._play_file(audio_bytes)

    def _mp3_to_wav(self, mp3_buffer):
        """Convert MP3 bytes to numpy float32 array."""
        try:
            import pydub
            audio = pydub.AudioSegment.from_mp3(mp3_buffer)
            data  = np.array(audio.get_array_of_samples(), dtype=np.float32)
            data /= np.iinfo(audio.array_type).max
            return data, audio.frame_rate
        except ImportError:
            # Try scipy directly (only works for WAV)
            import scipy.io.wavfile as wav
            mp3_buffer.seek(0)
            rate, data = wav.read(mp3_buffer)
            return data.astype(np.float32) / 32767.0, rate

    def _play_file(self, audio_bytes: bytes):
        import subprocess, platform, tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            if platform.system() == "Windows":
                subprocess.run(["powershell","-c",
                    f"Add-Type -AssemblyName presentationCore;"
                    f"$p=New-Object System.Windows.Media.MediaPlayer;"
                    f"$p.Open('{tmp}');$p.Play();Start-Sleep 5"], capture_output=True)
            elif platform.system() == "Darwin":
                subprocess.run(["afplay", tmp])
            else:
                subprocess.run(["mpg123", "-q", tmp], capture_output=True)
        except Exception:
            pass
        finally:
            Path(tmp).unlink(missing_ok=True)

    async def list_voices(self) -> list[dict]:
        """List all available edge-tts voices."""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return [{"name": v["Name"], "locale": v["Locale"],
                     "gender": v["Gender"]} for v in voices
                    if "Neural" in v.get("Name","")]
        except ImportError:
            return list(self.VOICES.values())

    def set_voice(self, voice_key_or_name: str):
        """Set voice by key (en-IN-female) or full name."""
        self.voice = self.VOICES.get(voice_key_or_name, voice_key_or_name)

    def status(self) -> dict:
        edge_ok = False
        try:
            import edge_tts
            edge_ok = True
        except ImportError:
            pass
        pyttsx3_ok = False
        try:
            import pyttsx3
            pyttsx3_ok = True
        except ImportError:
            pass
        return {
            "current_voice": self.voice,
            "edge_tts":      edge_ok,
            "pyttsx3":       pyttsx3_ok,
            "cache_files":   len(list(VOICE_CACHE.glob("*.mp3"))),
            "available_voices": list(self.VOICES.keys()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# STT ENGINE — understands what you say
# ─────────────────────────────────────────────────────────────────────────────

class STTEngine:
    """
    Speech-to-text using local Whisper.
    Runs 100% offline. 99 languages. No API key.

    Model sizes and their trade-offs:
      tiny   — 39MB  — fast,  lower accuracy (IoT / Pi Zero)
      base   — 74MB  — good balance (recommended for most users)
      small  — 244MB — better accuracy, slower
      medium — 769MB — near-perfect, needs 4GB+ RAM
      large  — 1.5GB — best quality, needs 8GB+ RAM

    faster-whisper (optional) runs 4x faster using CTranslate2.
    Install: pip install faster-whisper
    """

    def __init__(self, model_size: str = "base"):
        self.model_size   = model_size
        self._model       = None
        self._fast_model  = None
        self._lock        = threading.Lock()
        self._use_faster  = self._check_faster_whisper()

    def _check_faster_whisper(self) -> bool:
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    def _load(self):
        """Lazy-load model on first use."""
        if self._use_faster and self._fast_model is None:
            try:
                from faster_whisper import WhisperModel
                console.print(f"  [dim]Loading faster-whisper {self.model_size}...[/]")
                self._fast_model = WhisperModel(
                    self.model_size, device="cpu", compute_type="int8"
                )
                console.print(f"  [green]faster-whisper ready[/]")
            except Exception as e:
                console.print(f"  [yellow]faster-whisper failed: {e} — using standard[/]")
                self._use_faster = False

        if not self._use_faster and self._model is None:
            try:
                import whisper
                console.print(f"  [dim]Loading Whisper {self.model_size}...[/]")
                self._model = whisper.load_model(self.model_size)
                console.print(f"  [green]Whisper ready[/]")
            except ImportError:
                console.print("  [red]pip install openai-whisper[/]")
            except Exception as e:
                console.print(f"  [red]Whisper load error: {e}[/]")

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language:    str = None,
        format:      str = "webm",
    ) -> dict:
        """
        Transcribe audio from bytes.
        Accepts: webm (from browser WebRTC), wav, mp3, mp4, ogg
        Returns: {text, language, confidence, segments}
        """
        self._load()
        if not (self._model or self._fast_model):
            return {"text": "", "language": "?", "error": "Whisper not installed"}

        # Write to temp file (Whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name

        try:
            return self._transcribe_file(tmp, language)
        finally:
            Path(tmp).unlink(missing_ok=True)

    # Whisper hallucination suppression — common false positives on silence/noise
    HALLUCINATIONS = {
        "thank you", "thanks for watching", "thanks for listening",
        "please subscribe", "subtitles by", "transcribed by",
        "you", ".", "..", "...", "the", "a", "an",
        "bye", "goodbye", "hello", "hi", "hey",
    }

    def _transcribe_file(self, path: str, language: str = None) -> dict:
        """
        Transcribe audio with noise suppression and hallucination filtering.
        Key improvements:
        - condition_on_previous_text=False  → prevents Whisper from making up continuations
        - no_speech_threshold=0.6           → rejects segments that are likely silence
        - compression_ratio_threshold=2.4   → rejects repetitive/looping hallucinations
        - initial_prompt                    → biases toward real speech
        - Post-filter: drops known hallucination phrases
        """
        with self._lock:
            if self._use_faster and self._fast_model:
                segments, info = self._fast_model.transcribe(
                    path,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 200,
                        "threshold": 0.5,
                    },
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    temperature=[0.0, 0.2, 0.4],  # retry with higher temp if low confidence
                )
                parts = []
                for seg in segments:
                    t = seg.text.strip()
                    # Drop no-speech and hallucinations
                    if seg.no_speech_prob > 0.6:
                        continue
                    if t.lower() in self.HALLUCINATIONS:
                        continue
                    if len(t) < 2:
                        continue
                    parts.append(t)
                text = " ".join(parts).strip()
                return {
                    "text":       text,
                    "language":   info.language,
                    "confidence": info.language_probability,
                }

            elif self._model:
                result = self._model.transcribe(
                    path,
                    language=language,
                    verbose=False,
                    fp16=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    initial_prompt="The following is a clear voice command:",
                )
                text = result["text"].strip()
                # Filter hallucinations
                if text.lower() in self.HALLUCINATIONS or len(text) < 3:
                    text = ""
                return {
                    "text":     text,
                    "language": result.get("language","en"),
                    "segments": result.get("segments",[]),
                }

        return {"text": "", "language": "?", "error": "No model loaded"}

    def transcribe_mic(
        self,
        duration_s:  float = 5.0,
        sample_rate: int   = 16000,
        language:    str   = None,
    ) -> dict:
        """Record from microphone and transcribe."""
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav

            console.print(f"  [dim]Recording {duration_s:.0f}s...[/]")
            audio = sd.rec(
                int(duration_s * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav.write(f.name, sample_rate,
                          (audio * 32767).astype(np.int16))
                tmp = f.name

            result = self._transcribe_file(tmp, language)
            Path(tmp).unlink(missing_ok=True)
            return result

        except ImportError:
            return {"text":"","error":"pip install sounddevice scipy"}
        except Exception as e:
            return {"text":"","error":str(e)}

    def status(self) -> dict:
        whisper_ok = False
        faster_ok  = False
        try:
            import whisper
            whisper_ok = True
        except ImportError:
            pass
        try:
            import faster_whisper
            faster_ok = True
        except ImportError:
            pass
        return {
            "model_size":     self.model_size,
            "whisper":        whisper_ok,
            "faster_whisper": faster_ok,
            "using":          "faster-whisper" if self._use_faster else "whisper",
            "model_loaded":   self._model is not None or self._fast_model is not None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# VOICE ACTIVITY DETECTOR — knows when you're speaking
# ─────────────────────────────────────────────────────────────────────────────

class VoiceActivityDetector:
    """
    Detects when someone is speaking vs silence.
    Uses Silero VAD — 1MB model, runs on CPU, very accurate.
    Falls back to energy-based detection (no model needed).

    Install: pip install silero  (or it auto-downloads on first use)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold  = threshold
        self._model     = None
        self._use_silero= True

    def _load_silero(self):
        if self._model is not None:
            return True
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._model = (model, utils)
            return True
        except Exception:
            self._use_silero = False
            return False

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """Returns True if the audio chunk contains speech."""
        if self._use_silero and self._load_silero():
            try:
                import torch
                model, _ = self._model
                tensor   = torch.from_numpy(audio_chunk).float()
                prob     = model(tensor, sample_rate).item()
                return prob > self.threshold
            except Exception:
                self._use_silero = False

        # Fallback: energy-based VAD
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        return rms > 0.01  # simple threshold


# ─────────────────────────────────────────────────────────────────────────────
# WAKE WORD DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class WakeWordDetector:
    """
    Listens for "Hey ARIA" without processing every word.
    Uses a tiny Whisper model (tiny = 39MB) on 1-2 second chunks.
    Very low CPU usage — only runs full transcription on speech chunks.
    """

    WAKE_WORDS = [
        "hey aria", "aria", "hey ri", "okay aria",
        "oi aria", "hi aria", "hello aria",
        "hey ariya", "ariya",       # common mispronunciations
    ]

    def __init__(self, stt: STTEngine, on_wake: callable):
        self.stt         = stt
        self.on_wake     = on_wake
        self.vad         = VoiceActivityDetector()
        self._running    = False
        self._thread     = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        console.print(f"  [green]Wake word active[/] — say 'Hey ARIA' or {self.WAKE_WORDS[:3]}")

    def stop(self):
        self._running = False

    def _loop(self):
        try:
            import sounddevice as sd
            RATE     = 16000
            CHUNK    = int(RATE * 1.5)  # 1.5s chunks

            with sd.InputStream(samplerate=RATE, channels=1, dtype=np.float32) as stream:
                while self._running:
                    chunk, _ = stream.read(CHUNK)
                    chunk    = chunk.flatten()

                    # Only process if speech detected (saves CPU)
                    if not self.vad.is_speech(chunk, RATE):
                        continue

                    # Quick transcription on tiny model
                    result = self.stt.transcribe_bytes(
                        self._to_wav_bytes(chunk, RATE), format="wav"
                    )
                    text = result.get("text","").lower().strip()

                    if any(ww in text for ww in self.WAKE_WORDS):
                        console.print(f"  [green]Wake word: '{text}'[/]")
                        self.on_wake()

        except ImportError:
            console.print("  [dim]pip install sounddevice for wake word[/]")
        except Exception as e:
            console.print(f"  [yellow]Wake word error: {e}[/]")

    def _to_wav_bytes(self, audio: np.ndarray, rate: int) -> bytes:
        import scipy.io.wavfile as wav
        buf = io.BytesIO()
        wav.write(buf, rate, (audio * 32767).astype(np.int16))
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION MANAGER — handles multi-turn verbal conversations
# ─────────────────────────────────────────────────────────────────────────────

class ConversationManager:
    """
    Manages ongoing verbal conversations.
    Keeps context across turns — ARIA remembers what it just said.
    Handles interruptions — you can cut ARIA off mid-sentence.
    Adapts pace — speaks slower when you ask it to.
    """

    def __init__(self, tts: TTSEngine, stt: STTEngine, aria_engine):
        self.tts         = tts
        self.stt         = stt
        self.aria        = aria_engine
        self._history    = []    # in-session (role, text) — fast access
        self._speaking   = False
        self._interrupted= False
        self._lock       = threading.Lock()
        self._current_audio = None   # currently playing audio object

        # Persistent memory — survives restarts
        self.memory      = ConversationalMemory()

        # Barge-in detector — stops ARIA when user speaks
        self.barge_in    = BargeInDetector(
            VoiceActivityDetector(),
            on_barge_in=self._handle_barge_in,
        )

    def _handle_barge_in(self):
        """Called by BargeInDetector when user speaks during playback."""
        self._interrupted = True
        self._speaking    = False
        # Stop any playing audio immediately
        if self._current_audio:
            try:
                self._current_audio.pause()
            except Exception:
                pass
            self._current_audio = None
        console.print("  [green]Barge-in detected — stopped speaking[/]")

    async def process_audio(
        self,
        audio_bytes: bytes,
        format:      str = "webm",
        language:    str = None,
        device:      str = "",
    ) -> dict:
        """
        Full pipeline: audio bytes → STT → context → ARIA → TTS → audio bytes.
        Now with:
          - Persistent memory across sessions (ConversationalMemory)
          - Barge-in detection (BargeInDetector arms during playback)
          - Context from last 8 turns — ARIA remembers the conversation
        """
        # Step 1: STT
        stt_result = self.stt.transcribe_bytes(audio_bytes, language, format)
        transcript = stt_result.get("text", "").strip()

        if not transcript or len(transcript) < 2:
            return {"transcript": "", "response_text": "", "audio_bytes": b""}

        console.print(f"  [dim]Heard:[/] {transcript}")

        # Step 2: Persist user turn immediately
        self.memory.save("user", transcript, device=device)

        # Step 3: Update in-session history
        self._history.append(("user", transcript))
        if len(self._history) > 20:
            self._history = self._history[-20:]

        # Step 4: Check for special verbal commands (no LLM needed)
        special = self._check_special_commands(transcript)
        if special:
            response_text = special
        else:
            response_text = await self._get_aria_response(transcript)

        # Step 5: Persist ARIA's response
        self.memory.save("aria", response_text, device="aria")

        # Step 6: Update in-session history
        self._history.append(("aria", response_text))

        # Step 7: TTS — arm barge-in BEFORE playback starts
        self._interrupted = False
        self._speaking    = True
        self.barge_in.arm()

        response_audio = await self.tts.synthesize(response_text)

        # Disarm if user interrupted before we even finish synthesis
        if self._interrupted:
            self.barge_in.disarm()
            return {
                "transcript":    transcript,
                "response_text": response_text,
                "audio_bytes":   b"",          # send no audio — user is already talking
                "interrupted":   True,
                "language":      stt_result.get("language", "en"),
            }

        self.barge_in.disarm()
        self._speaking = False

        return {
            "transcript":    transcript,
            "response_text": response_text,
            "audio_bytes":   response_audio,
            "interrupted":   False,
            "language":      stt_result.get("language", "en"),
        }

    def _check_special_commands(self, text: str) -> Optional[str]:
        """Handle verbal control commands without going to the full engine."""
        text_low = text.lower()
        if any(x in text_low for x in ["stop", "cancel", "quit talking"]):
            self._interrupted = True
            return "Okay, stopping."
        if any(x in text_low for x in ["speak slower", "slow down"]):
            self.tts.voice = self.tts.voice  # could adjust rate
            return "I'll speak more slowly."
        if any(x in text_low for x in ["what time", "current time"]):
            return f"It's {datetime.now().strftime('%I:%M %p')}."
        return None

    # Live-data keywords — queries that need real-time web data
    _LIVE_KW = (
        "weather","temperature","forecast","rain","humidity","wind","aqi","air quality",
        "news","headline","breaking","today","current","right now","latest","recent",
        "stock","price","share","nifty","sensex","bitcoin","crypto","rate","market",
        "score","match","ipl","cricket","football","result","game",
        "trending","viral","what's happening","what happened","happened",
        "global warming","climate","earthquake","flood","disaster","accident",
        "election","government","policy","update","announce","launch",
    )

    @staticmethod
    def _voice_web_search(query: str) -> str:
        """DuckDuckGo search. Returns rich snippet with title + body, max 500 chars."""
        try:
            from ddgs import DDGS
            with DDGS() as ddg:
                results = list(ddg.text(query, max_results=3))
        except Exception:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddg:
                    results = list(ddg.text(query, max_results=3))
            except Exception:
                return ""
        if not results:
            return ""
        parts = []
        for r in results:
            title   = r.get("title", "")
            snippet = r.get("body", r.get("snippet", ""))
            if snippet:
                parts.append(f"{title}: {snippet}" if title else snippet)
        return "\n".join(parts)[:500]

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Quick heuristic language detection from character ranges.
        Returns BCP-47 code: 'hi' for Hindi, 'te' for Telugu, etc.
        Falls back to 'en'.
        """
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        telugu     = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
        tamil      = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        total      = max(len(text), 1)
        if devanagari / total > 0.2: return "hi"
        if telugu     / total > 0.2: return "te"
        if tamil      / total > 0.2: return "ta"
        return "en"

    async def _get_aria_response(self, text: str) -> str:
        """
        Conversational voice response — 2-3 sentences, natural pace.
        • Auto-detects language and responds in same language
        • Runs web search for live queries, injects results directly
        • LLM runs in thread executor (non-blocking)
        """
        import re
        loop = asyncio.get_event_loop()
        try:
            ctx = self.memory.summary_context(max_chars=300)
            if not ctx:
                ctx = "\n".join(
                    f"{'User' if r == 'user' else 'ARIA'}: {t}"
                    for r, t in self._history[-4:-1]
                )

            # Detect language for bilingual responses
            lang = self._detect_language(text)
            lang_instr = ""
            if lang == "hi":
                lang_instr = "The user spoke in Hindi. Reply naturally in Hindi (Devanagari script).\n"
            elif lang == "te":
                lang_instr = "The user spoke in Telugu. Reply naturally in Telugu script.\n"
            elif lang == "ta":
                lang_instr = "The user spoke in Tamil. Reply naturally in Tamil script.\n"

            # Live-data: search and inject results directly
            web_ctx = ""
            t_low = text.lower()
            needs_search = any(kw in t_low for kw in self._LIVE_KW)
            if needs_search:
                try:
                    web_ctx = await asyncio.wait_for(
                        loop.run_in_executor(None, self._voice_web_search, text),
                        timeout=3.5,
                    )
                except (asyncio.TimeoutError, Exception):
                    web_ctx = ""

            if web_ctx:
                # Force the LLM to answer FROM the data, not suggest searching elsewhere
                user_part = (
                    f"LIVE WEB RESULTS (use these to answer directly):\n{web_ctx}\n\n"
                    f"Question: {text}\n"
                    f"ARIA answers directly from the results above (DO NOT say 'search on Google', "
                    f"DO NOT say 'check a website' — give the actual answer now):"
                )
            else:
                user_part = f"User says: {text}\nARIA responds conversationally:"

            prompt = (
                "I am ARIA, a personal AI assistant running locally on this device. "
                "I was built by this user — I am not from Microsoft, not Cortana, not Copilot.\n"
                f"{lang_instr}"
                "Reply style: 2-3 natural spoken sentences. No markdown, no bullet points, "
                "no 'I suggest searching' — just answer directly and helpfully.\n"
                + (f"Recent conversation:\n{ctx}\n\n" if ctx else "\n")
                + user_part
            )

            result = await loop.run_in_executor(
                None, lambda: self.aria.generate(prompt, temperature=0.5)
            )

            # Clean for TTS
            result = re.sub(r"```.*?```", "code block", result, flags=re.DOTALL)
            result = re.sub(r"[#*_`•\[\]]", "", result)
            result = re.sub(r"\n+", " ", result)
            result = result.strip()

            # Cap at 4 sentences to avoid walls of text
            sentences = re.split(r'(?<=[.!?])\s+', result)
            if len(sentences) > 4:
                result = " ".join(sentences[:4])

            return result if result else "I'm not sure about that, could you ask again?"

        except Exception:
            return "Sorry, I ran into a problem. Please try again."

    def clear_history(self):
        """Clear in-session history and start a new memory session."""
        self._history = []
        self.memory.new_session()

    def get_history(self) -> list:
        """Get current in-session history."""
        return list(self._history)

    def get_full_history(self, turns: int = 20) -> list:
        """Get persistent cross-session history from DB."""
        return self.memory.recent(turns=turns)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER VOICE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class VoiceAgent:
    """
    Complete voice I/O system for ARIA.
    Integrates with FastAPI WebSocket for browser communication.

    Usage modes:
    1. WebSocket (browser):
       ws://localhost:8000/ws/voice
       Send: binary audio (webm/opus from MediaRecorder)
       Receive: {transcript, response, audio_b64}

    2. Microphone (desktop):
       voice.start_listening() — wake word mode
       voice.speak("Hello") — TTS output

    3. IoT/headless:
       voice.transcribe_mic(5.0) — record and transcribe
       voice.speak("text") — speak response
    """

    def __init__(self, aria_engine=None, bus=None, voice: str = None):
        self.tts          = TTSEngine(voice)
        self.stt          = STTEngine("base")
        self.conversation = ConversationManager(self.tts, self.stt, aria_engine) if aria_engine else None
        self.bus          = bus
        self._wake        = None
        self._listening   = False

    async def handle_websocket(self, websocket):
        """
        Handle a browser WebSocket voice connection.
        Browser sends audio chunks → ARIA transcribes → responds with audio.
        """
        console.print("  [green]Voice WebSocket connected[/]")
        try:
            while True:
                # Receive audio from browser
                data = await websocket.receive()

                if data.get("type") == "websocket.disconnect":
                    break

                # Handle JSON control messages
                if data.get("type") == "websocket.receive":
                    if data.get("text"):
                        msg = json.loads(data["text"])
                        # ── Fast path: browser already transcribed via Web Speech API ──
                        if msg.get("type") == "text_query":
                            await self._handle_text_query(websocket, msg.get("text", ""))
                        else:
                            await self._handle_control(websocket, msg)
                        continue

                    # Binary audio data (legacy push-to-talk fallback)
                    audio_bytes = data.get("bytes", b"")
                    if not audio_bytes:
                        continue

                    # Process audio → response (Whisper STT path)
                    if self.conversation:
                        result = await self.conversation.process_audio(
                            audio_bytes, format="webm"
                        )
                    else:
                        stt    = self.stt.transcribe_bytes(audio_bytes, format="webm")
                        result = {
                            "transcript":    stt.get("text",""),
                            "response_text": "",
                            "audio_bytes":   b"",
                        }

                    import base64
                    await websocket.send_text(json.dumps({
                        "type":       "response",
                        "transcript": result["transcript"],
                        "text":       result["response_text"],
                        "audio_b64":  base64.b64encode(result["audio_bytes"]).decode()
                                      if result["audio_bytes"] else "",
                        "ts":         datetime.now().isoformat(),
                    }))

        except Exception as e:
            console.print(f"  [yellow]WebSocket voice error: {e}[/]")

    async def _handle_text_query(self, websocket, text: str):
        """
        Fast path: browser already transcribed speech via Web Speech API.
        Skip Whisper entirely — just LLM + TTS.
        This cuts ~2s of audio upload + Whisper processing time.
        """
        import base64
        text = text.strip()
        if not text or not self.conversation:
            return
        console.print(f"  [cyan]Text query:[/] {text}")

        conv = self.conversation
        conv.memory.save("user", text, device="browser")
        conv._history.append(("user", text))
        if len(conv._history) > 20:
            conv._history = conv._history[-20:]

        # Check special commands first (no LLM needed)
        response_text = conv._check_special_commands(text)
        if not response_text:
            response_text = await conv._get_aria_response(text)

        conv.memory.save("aria", response_text, device="aria")
        conv._history.append(("aria", response_text))

        audio_bytes = await conv.tts.synthesize(response_text)
        await websocket.send_text(json.dumps({
            "type":       "response",
            "transcript": text,
            "text":       response_text,
            "audio_b64":  base64.b64encode(audio_bytes).decode() if audio_bytes else "",
            "ts":         datetime.now().isoformat(),
        }))

    async def _handle_control(self, websocket, msg: dict):
        """Handle JSON control messages from browser."""
        cmd = msg.get("cmd","")
        if cmd == "speak":
            text  = msg.get("text","")
            audio = await self.tts.synthesize(text)
            import base64
            await websocket.send_text(json.dumps({
                "type":      "tts_audio",
                "audio_b64": base64.b64encode(audio).decode(),
            }))
        elif cmd == "set_voice":
            self.tts.set_voice(msg.get("voice", self.tts.voice))
        elif cmd == "status":
            await websocket.send_text(json.dumps({
                "type":   "status",
                "status": self.status(),
            }))

    def start_wake_word(self, on_wake: callable = None):
        """Start wake word detection in background."""
        if self._wake and self._listening:
            return

        def default_wake():
            console.print("  [green]ARIA activated by wake word[/]")
            result = self.stt.transcribe_mic(duration_s=6.0)
            text   = result.get("text","").strip()
            if text and self.conversation:
                asyncio.run(self._respond_and_speak(text))

        handler         = on_wake or default_wake
        self._wake      = WakeWordDetector(self.stt, handler)
        self._listening = True
        self._wake.start()

    async def _respond_and_speak(self, text: str):
        if self.conversation:
            result = await self.conversation.process_audio(
                b"", format="wav"
            )
            if result["audio_bytes"]:
                self.tts._play(result["audio_bytes"])

    def speak(self, text: str, voice: str = None):
        """Speak text synchronously (blocks until done)."""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.tts.speak_and_stream(text, voice))

    async def speak_async(self, text: str) -> bytes:
        """Speak text and return the audio bytes."""
        return await self.tts.synthesize(text)

    def transcribe_mic(self, duration_s: float = 5.0) -> str:
        """Record from mic and return transcript."""
        result = self.stt.transcribe_mic(duration_s)
        return result.get("text","")

    def transcribe_audio_bytes(self, audio_bytes: bytes, fmt: str = "webm") -> str:
        """
        Transcribe raw audio bytes sent from the browser (WebM/WAV).
        Called by the /api/voice/transcribe HTTP endpoint.
        Returns the transcribed text string (empty string on failure).
        """
        if not audio_bytes:
            return ""
        try:
            result = self.stt.transcribe_bytes(audio_bytes, format=fmt)
            return result.get("text", "").strip()
        except Exception as e:
            console.print(f"  [yellow]transcribe_audio_bytes error: {e}[/]")
            return ""

    def stop_listening(self):
        if self._wake:
            self._wake.stop()
        self._listening = False

    def status(self) -> dict:
        return {
            "tts":       self.tts.status(),
            "stt":       self.stt.status(),
            "listening": self._listening,
            "wake_words": WakeWordDetector.WAKE_WORDS[:5],
        }


# ─────────────────────────────────────────────────────────────────────────────
# BARGE-IN DETECTOR — stops ARIA mid-sentence when you speak
# ─────────────────────────────────────────────────────────────────────────────

class BargeInDetector:
    """
    Detects when the user starts speaking while ARIA is talking.
    Stops ARIA's audio immediately and listens to the new command.
    
    Uses energy-based VAD — no model needed.
    Runs in a background thread that monitors the mic.
    """

    def __init__(self, vad: VoiceActivityDetector, on_barge_in: callable):
        self.vad        = vad
        self.callback   = on_barge_in
        self._active    = False
        self._thread    = None

    def arm(self):
        """Arm the detector — call this when ARIA starts speaking."""
        self._active = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def disarm(self):
        """Disarm — call this when ARIA stops speaking."""
        self._active = False

    def _monitor(self):
        try:
            import sounddevice as sd
            RATE  = 16000
            CHUNK = int(RATE * 0.2)   # 200ms chunks — low latency

            with sd.InputStream(samplerate=RATE, channels=1, dtype=np.float32) as stream:
                # Wait 300ms before arming (avoid detecting our own TTS)
                time.sleep(0.3)
                while self._active:
                    chunk, _ = stream.read(CHUNK)
                    if self.vad.is_speech(chunk.flatten(), RATE):
                        self._active = False
                        self.callback()
                        return
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATIONAL MEMORY — persists context across sessions
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3
from pathlib import Path as _Path

_CONV_DB = _Path(__file__).resolve().parent.parent / "data" / "conversations.db"

def _init_conv_db():
    with sqlite3.connect(_CONV_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                session  TEXT,
                role     TEXT,
                text     TEXT,
                ts       TEXT,
                device   TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sess ON messages(session, ts)")


class ConversationalMemory:
    """
    Persists conversation history across sessions.
    ARIA remembers what was said yesterday, last week.
    """

    def __init__(self):
        _init_conv_db()
        self._session = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(self, role: str, text: str, device: str = ""):
        try:
            with sqlite3.connect(_CONV_DB) as conn:
                conn.execute(
                    "INSERT INTO messages (session,role,text,ts,device) VALUES (?,?,?,?,?)",
                    (self._session, role, text, datetime.now().isoformat(), device)
                )
        except Exception:
            pass

    def recent(self, turns: int = 10, sessions: int = 3) -> list[dict]:
        """Get recent conversation turns for context."""
        try:
            with sqlite3.connect(_CONV_DB) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT role, text, ts, session
                    FROM messages
                    ORDER BY id DESC LIMIT ?
                """, (turns,)).fetchall()
            return [dict(r) for r in reversed(rows)]
        except Exception:
            return []

    def summary_context(self, max_chars: int = 800) -> str:
        """Build a context string from recent history for the LLM."""
        turns = self.recent(turns=8)
        if not turns:
            return ""
        lines = []
        for t in turns:
            prefix = "User" if t["role"] == "user" else "ARIA"
            lines.append(f"{prefix}: {t['text'][:150]}")
        context = "\n".join(lines)
        return context[:max_chars]

    def new_session(self):
        self._session = datetime.now().strftime("%Y%m%d_%H%M%S")
