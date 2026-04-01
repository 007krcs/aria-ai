"""
ARIA — Voice Agent  (v3 — production-grade)
=============================================
Real-time voice I/O with:
  • Silero VAD  — detects actual speech; ignores silence & background noise
  • noisereduce — spectral subtraction before feeding Whisper
  • faster-whisper — 4× faster transcription, same accuracy
  • Adaptive wake-word detection — no fixed-length polling
  • Continuous conversation mode — stays listening after wake word
  • Audio energy gate — secondary guard against low-level noise bleed

Dependency install (one-time):
  pip install silero-vad noisereduce faster-whisper sounddevice scipy edge-tts pyttsx3

Fallback chain:
  VAD           → energy threshold fallback
  faster-whisper → openai-whisper → none (text-only mode)
  edge-tts      → pyttsx3 → silent
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
import wave
import json
import queue
import struct
import asyncio
import tempfile
import threading
import platform
import subprocess
from pathlib import Path
from collections import deque
from typing import Optional, Callable, List
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()

# ── Voice preferences ─────────────────────────────────────────────────────────
DEFAULT_VOICE  = "en-IN-NeerjaNeural"
HINDI_VOICE    = "hi-IN-SwaraNeural"
SAMPLE_RATE    = 16000   # Hz — Whisper native rate
CHANNELS       = 1
FRAME_MS       = 30      # ms per VAD frame (10, 20, or 30)
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples per frame

# ── Wake words ────────────────────────────────────────────────────────────────
WAKE_WORDS = [
    "hey aria", "aria", "hey ri", "okay aria", "ok aria",
    "aye aria", "hi aria", "hey ara",
]

# ─────────────────────────────────────────────────────────────────────────────
# NOISE REDUCTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _denoise_audio(audio_np, sample_rate: int = SAMPLE_RATE):
    """
    Apply spectral noise reduction to a float32 numpy array.
    Falls back to no-op if noisereduce is not installed.
    """
    try:
        import noisereduce as nr
        import numpy as np
        # Estimate noise from first 0.5 s of audio (assumed to be mostly background)
        noise_sample = audio_np[: int(sample_rate * 0.5)]
        if len(noise_sample) > 100:
            return nr.reduce_noise(
                y=audio_np,
                sr=sample_rate,
                y_noise=noise_sample,
                prop_decrease=0.85,   # aggression — 0.0 = off, 1.0 = max
                stationary=False,     # handles non-stationary noise (fans, AC, traffic)
            )
        return audio_np
    except ImportError:
        return audio_np
    except Exception:
        return audio_np


def _energy_level(audio_np) -> float:
    """RMS energy of audio array (0.0 – 1.0)."""
    try:
        import numpy as np
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        return min(1.0, rms)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SILERO VAD  — Voice Activity Detection
# ─────────────────────────────────────────────────────────────────────────────

class SileroVAD:
    """
    Lightweight VAD using Silero's torch model.
    Detects speech frames at 16 kHz without sending data anywhere.
    Falls back to energy-threshold when torch is unavailable.
    """

    _model  = None
    _utils  = None
    _loaded = False
    _lock   = threading.Lock()

    @classmethod
    def load(cls):
        with cls._lock:
            if cls._loaded:
                return
            try:
                import torch
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    verbose=False,
                    trust_repo=True,
                )
                cls._model  = model
                cls._utils  = utils
                cls._loaded = True
                console.print("  [green]Silero VAD loaded[/]")
            except Exception as e:
                console.print(f"  [yellow]Silero VAD unavailable ({e}), using energy gate[/]")
                cls._loaded = True  # mark loaded so we don't retry

    @classmethod
    def is_speech(cls, audio_chunk_np, threshold: float = 0.5) -> bool:
        """Returns True if the audio chunk contains speech."""
        if not cls._loaded:
            cls.load()

        if cls._model is None:
            # Fallback: energy threshold
            return _energy_level(audio_chunk_np) > 0.01

        try:
            import torch
            import numpy as np
            tensor = torch.from_numpy(audio_chunk_np.copy()).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                prob = cls._model(tensor, SAMPLE_RATE).item()
            return prob > threshold
        except Exception:
            return _energy_level(audio_chunk_np) > 0.01


# ─────────────────────────────────────────────────────────────────────────────
# STT ENGINE  — faster-whisper with noisereduce preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class STTEngine:
    """
    Speech-to-Text using faster-whisper (4× faster than openai-whisper).
    Falls back to openai-whisper, then fails gracefully.

    Auto-selects model size based on available RAM:
      < 2 GB  → tiny   (75 MB)
      2–4 GB  → base   (150 MB)
      4–8 GB  → small  (500 MB)
      > 8 GB  → medium (1.5 GB)
    """

    def __init__(self, model_size: str = "auto"):
        self.model_size = model_size if model_size != "auto" else self._auto_model()
        self._model     = None
        self._lock      = threading.Lock()
        self._engine    = None   # "faster-whisper" | "whisper" | None

    @staticmethod
    def _auto_model() -> str:
        """Pick Whisper model size based on available system RAM."""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            if ram_gb < 2:   return "tiny"
            if ram_gb < 4:   return "base"
            if ram_gb < 8:   return "small"
            return "medium"
        except Exception:
            return "base"

    def _load(self):
        if self._model is not None:
            return
        console.print(f"  [dim]Loading STT model ({self.model_size})...[/]")

        # Try faster-whisper first (GGML, much faster)
        try:
            from faster_whisper import WhisperModel
            device   = "cpu"
            compute  = "int8"  # quantized — fast on CPU, low RAM
            try:
                import torch
                if torch.cuda.is_available():
                    device  = "cuda"
                    compute = "float16"
            except Exception:
                pass
            self._model  = WhisperModel(self.model_size, device=device, compute_type=compute)
            self._engine = "faster-whisper"
            console.print(f"  [green]STT ready[/] — faster-whisper/{self.model_size} on {device}")
            return
        except ImportError:
            pass

        # Fallback: openai-whisper
        try:
            import whisper
            self._model  = whisper.load_model(self.model_size)
            self._engine = "whisper"
            console.print(f"  [green]STT ready[/] — openai-whisper/{self.model_size}")
            return
        except ImportError:
            console.print("  [red]No STT engine found.[/] Run: pip install faster-whisper")

    def transcribe_numpy(self, audio_np, language: str = None) -> dict:
        """
        Transcribe a float32 numpy array.
        Applies noise reduction before transcription.
        Returns {"text": str, "language": str, "confidence": float}
        """
        self._load()
        if self._model is None:
            return {"text": "", "language": "unknown", "confidence": 0.0}

        import numpy as np

        # Clamp values
        audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)

        # Noise reduction
        audio_np = _denoise_audio(audio_np)

        # Energy gate — skip if too quiet
        if _energy_level(audio_np) < 0.003:
            return {"text": "", "language": language or "en", "confidence": 0.0}

        # Write to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                pcm = (audio_np * 32767).astype(np.int16)
                wf.writeframes(pcm.tobytes())

        try:
            with self._lock:
                if self._engine == "faster-whisper":
                    return self._transcribe_faster(tmp_path, language)
                else:
                    return self._transcribe_openai(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _transcribe_faster(self, path: str, language: str = None) -> dict:
        segments, info = self._model.transcribe(
            path,
            language=language,
            beam_size=3,
            vad_filter=True,           # built-in VAD in faster-whisper too
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
            word_timestamps=False,
        )
        text = " ".join(s.text for s in segments).strip()
        return {
            "text":       text,
            "language":   info.language,
            "confidence": getattr(info, "language_probability", 1.0),
        }

    def _transcribe_openai(self, path: str) -> dict:
        result = self._model.transcribe(path, verbose=False)
        return {
            "text":       result["text"].strip(),
            "language":   result.get("language", "en"),
            "confidence": 1.0,
        }

    def transcribe_bytes(self, audio_bytes: bytes, fmt: str = "wav") -> dict:
        """Transcribe raw audio bytes (e.g. from WebRTC)."""
        import numpy as np
        import scipy.io.wavfile as wav_io

        try:
            buf              = io.BytesIO(audio_bytes)
            rate, data       = wav_io.read(buf)
            if data.dtype != np.float32:
                data = data.astype(np.float32) / (np.iinfo(data.dtype).max or 1)
            if data.ndim == 2:
                data = data.mean(axis=1)  # stereo → mono
            # Resample if needed
            if rate != SAMPLE_RATE:
                data = _resample(data, rate, SAMPLE_RATE)
            return self.transcribe_numpy(data)
        except Exception as e:
            return {"text": "", "language": "unknown", "error": str(e)}


def _resample(audio_np, from_rate: int, to_rate: int):
    """Simple linear resample."""
    try:
        import numpy as np
        target_len = int(len(audio_np) * to_rate / from_rate)
        indices    = np.linspace(0, len(audio_np) - 1, target_len)
        return np.interp(indices, np.arange(len(audio_np)), audio_np).astype(np.float32)
    except Exception:
        return audio_np


# ─────────────────────────────────────────────────────────────────────────────
# TTS ENGINE  — edge-tts (Microsoft neural, free) → pyttsx3 offline
# ─────────────────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Text-to-Speech.
    Primary:  edge-tts  — neural quality, many languages, free, needs internet once
    Fallback: pyttsx3   — fully offline, robotic but functional
    """

    def __init__(self, voice: str = DEFAULT_VOICE):
        self.voice    = voice
        self._lock    = threading.Lock()
        self._queue:  queue.Queue = queue.Queue()
        self._speaking: bool = False    # True while audio is playing (barge-in hook reads this)
        self._stop_flag: bool = False   # Set to True to abort current playback
        self._worker_thread = threading.Thread(
            target=self._playback_worker, daemon=True, name="tts-worker"
        )
        self._worker_thread.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def speak(self, text: str, voice: str = None, priority: bool = False):
        """
        Queue text for speaking. Non-blocking.
        If priority=True, clear the queue first and abort current playback (barge-in).
        If text is empty and priority=True, acts as a stop-speech signal only.
        """
        if priority:
            # Signal worker to abort current playback
            self._stop_flag = True
            # Drain pending queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        if not text or not text.strip():
            return
        self._queue.put((text.strip(), voice or self.voice))

    def speak_sync(self, text: str, voice: str = None):
        """Speak and block until finished."""
        audio = asyncio.run(self._synthesise(text, voice or self.voice))
        self._play_bytes(audio)

    async def speak_async(self, text: str, voice: str = None) -> bytes:
        """Return audio bytes asynchronously."""
        return await self._synthesise(text, voice or self.voice)

    def set_voice(self, voice: str):
        self.voice = voice

    async def list_voices(self) -> list:
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return [{"name": v["Name"], "locale": v["Locale"], "gender": v["Gender"]}
                    for v in voices]
        except Exception:
            return []

    # ── Internals ───────────────────────────────────────────────────────────────

    def _playback_worker(self):
        """Background thread that drains the TTS queue."""
        while True:
            try:
                text, voice = self._queue.get(timeout=0.5)
                if self._stop_flag:
                    self._stop_flag = False
                    continue
                audio = asyncio.run(self._synthesise(text, voice))
                if self._stop_flag:
                    self._stop_flag = False
                    continue
                with self._lock:
                    self._speaking = True
                    try:
                        self._play_bytes(audio)
                    finally:
                        self._speaking = False
            except queue.Empty:
                continue
            except Exception as e:
                self._speaking = False
                console.print(f"  [yellow]TTS worker error: {e}[/]")

    async def _synthesise(self, text: str, voice: str) -> bytes:
        """Generate MP3/WAV bytes from text using edge-tts."""
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            data = buf.getvalue()
            if data:
                return data
        except ImportError:
            pass
        except Exception as e:
            console.print(f"  [yellow]edge-tts error: {e}[/]")
        # Fallback — pyttsx3 (no audio bytes, plays directly)
        self._pyttsx3_speak(text)
        return b""

    def _play_bytes(self, audio_bytes: bytes):
        """Play MP3/WAV bytes through the system speaker."""
        if not audio_bytes:
            return
        try:
            import sounddevice as sd
            import numpy as np
            import scipy.io.wavfile as wav_io
            # edge-tts returns mp3 — decode via scipy (needs libsndfile/ffmpeg)
            # Try wav first, then mp3 via temp file
            try:
                rate, data = wav_io.read(io.BytesIO(audio_bytes))
                if data.dtype != np.float32:
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                sd.play(data, rate)
                sd.wait()
                return
            except Exception:
                pass
            # MP3 → play via OS
            self._play_via_os(audio_bytes)
        except ImportError:
            self._play_via_os(audio_bytes)
        except Exception as e:
            console.print(f"  [yellow]Audio play error: {e}[/]")

    def _play_via_os(self, audio_bytes: bytes, suffix: str = ".mp3"):
        """Last resort — write to temp file and play with OS player."""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                tmp = f.name
            plat = platform.system()
            if plat == "Windows":
                subprocess.run(
                    ["powershell", "-c",
                     f"Add-Type -AssemblyName presentationCore;"
                     f"$mp=New-Object system.windows.media.mediaplayer;"
                     f"$mp.open('{tmp}');$mp.Play();Start-Sleep -s 5"],
                    capture_output=True, timeout=10,
                )
            elif plat == "Darwin":
                subprocess.run(["afplay", tmp], capture_output=True, timeout=30)
            else:
                subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp],
                               capture_output=True, timeout=30)
        except Exception as e:
            console.print(f"  [yellow]OS playback error: {e}[/]")
        finally:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass

    def _pyttsx3_speak(self, text: str):
        """Offline TTS via pyttsx3."""
        try:
            import pyttsx3
            eng = pyttsx3.init()
            # Tune rate and volume for clarity
            eng.setProperty("rate", 165)      # slower = clearer
            eng.setProperty("volume", 1.0)
            eng.say(text)
            eng.runAndWait()
            eng.stop()
        except ImportError:
            console.print("  [dim]No TTS engine available. pip install edge-tts or pyttsx3[/]")
        except Exception as e:
            console.print(f"  [yellow]pyttsx3 error: {e}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# CONTINUOUS LISTENER  — VAD-based, no fixed-duration polling
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousListener:
    """
    Streams microphone audio and uses Silero VAD to detect real speech.

    State machine:
      IDLE → wake-word spotted → ACTIVE (capture full command) → IDLE

    Noise handling:
      • Silero VAD probability threshold (ignores background noise)
      • Energy gate (double protection)
      • noisereduce before feeding Whisper
      • Min speech duration filter (ignores clicks, coughs < 300 ms)

    Adaptive silence timeout:
      After command capture starts, waits for 1.5 s of silence before
      cutting the utterance — allows natural pauses in speech.
    """

    # State constants
    IDLE    = "idle"
    ACTIVE  = "active"

    def __init__(
        self,
        stt:               STTEngine,
        on_command:        Callable,
        on_wake:           Callable       = None,
        vad_threshold:     float          = 0.45,
        silence_timeout_s: float          = 1.5,
        max_command_s:     float          = 12.0,
        energy_threshold:  float          = 0.005,
    ):
        self.stt              = stt
        self.on_command       = on_command
        self.on_wake          = on_wake
        self.vad_threshold    = vad_threshold
        self.silence_timeout  = silence_timeout_s
        self.max_command_s    = max_command_s
        self.energy_threshold = energy_threshold

        self._state           = self.IDLE
        self._running         = False
        self._thread: Optional[threading.Thread] = None

        # Circular pre-roll buffer: keeps last 0.5 s before wake word
        self._preroll_frames: deque = deque(maxlen=int(500 / FRAME_MS))

        # Background noise floor estimator (rolling)
        self._noise_frames:   deque = deque(maxlen=100)

    # ── Control ────────────────────────────────────────────────────────────────

    def start(self):
        SileroVAD.load()
        self._running = True
        self._thread  = threading.Thread(
            target=self._stream_loop, daemon=True, name="aria-listener"
        )
        self._thread.start()
        console.print(
            "  [green]Voice listener active[/] — say [bold]'Hey ARIA'[/] to activate"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    # ── Main streaming loop ────────────────────────────────────────────────────

    def _stream_loop(self):
        """
        Reads 30-ms frames from the microphone in a tight loop.
        No fixed recording durations — entirely event-driven via VAD.
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            console.print("  [red]sounddevice not installed.[/] Run: pip install sounddevice")
            return

        console.print("  [dim]Microphone stream opened[/]")
        audio_buf: list = []          # accumulates frames during ACTIVE state
        silence_frames  = 0
        max_frames_cmd  = int(self.max_command_s * 1000 / FRAME_MS)
        silence_frames_thresh = int(self.silence_timeout * 1000 / FRAME_MS)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            latency="low",
        ) as stream:
            while self._running:
                try:
                    frame, _ = stream.read(FRAME_SAMPLES)
                    frame    = frame[:, 0] if frame.ndim == 2 else frame  # mono

                    # Energy gate — fast pre-filter (skip pure silence)
                    energy = _energy_level(frame)
                    if energy < self.energy_threshold * 0.3:
                        if self._state == self.ACTIVE:
                            silence_frames += 1
                            audio_buf.append(frame)
                        else:
                            self._preroll_frames.append(frame)
                        continue

                    # VAD check
                    is_speech = SileroVAD.is_speech(frame, self.vad_threshold)

                    if self._state == self.IDLE:
                        self._preroll_frames.append(frame)
                        if is_speech:
                            # Collect a 1-second snippet for wake-word check
                            snippet = self._collect_snippet(stream, duration_s=1.2)
                            if snippet is not None:
                                self._check_wake_word(snippet, np, stream, audio_buf,
                                                      silence_frames_thresh, max_frames_cmd)

                    elif self._state == self.ACTIVE:
                        audio_buf.append(frame)
                        if is_speech:
                            silence_frames = 0
                        else:
                            silence_frames += 1

                        # End of utterance
                        if (silence_frames >= silence_frames_thresh or
                                len(audio_buf) >= max_frames_cmd):
                            self._process_command(audio_buf, np)
                            audio_buf      = []
                            silence_frames = 0
                            self._state    = self.IDLE

                except Exception as e:
                    if self._running:
                        console.print(f"  [yellow]Listener frame error: {e}[/]")
                    time.sleep(0.05)

    def _collect_snippet(self, stream, duration_s: float = 1.2):
        """Read an extra N seconds of audio for wake-word detection."""
        try:
            import numpy as np
            frames = []
            n = int(duration_s * 1000 / FRAME_MS)
            for _ in range(n):
                f, _ = stream.read(FRAME_SAMPLES)
                frames.append(f[:, 0] if f.ndim == 2 else f)
            # Include pre-roll
            pre = list(self._preroll_frames)
            return np.concatenate(pre + frames).astype(np.float32)
        except Exception:
            return None

    def _check_wake_word(self, audio_np, np, stream, audio_buf,
                         silence_frames_thresh, max_frames_cmd):
        """Transcribe snippet and check for wake word."""
        result = self.stt.transcribe_numpy(audio_np)
        text   = result.get("text", "").lower().strip()
        if not text:
            return

        if any(ww in text for ww in WAKE_WORDS):
            console.print(f"  [green]Wake word detected![/] ({text!r})")
            if self.on_wake:
                try:
                    self.on_wake()
                except Exception:
                    pass
            self._state = self.ACTIVE
            self._preroll_frames.clear()

    def _process_command(self, audio_buf: list, np):
        """Denoise + transcribe + dispatch collected command audio."""
        if not audio_buf:
            return
        try:
            audio_np = np.concatenate(audio_buf).astype(np.float32)
            # Skip if too short (< 0.3 s) or too quiet
            if len(audio_np) < SAMPLE_RATE * 0.3:
                return
            if _energy_level(audio_np) < self.energy_threshold:
                return

            result   = self.stt.transcribe_numpy(audio_np)
            text     = result.get("text", "").strip()
            conf     = result.get("confidence", 1.0)

            if text and len(text) > 2 and conf > 0.3:
                console.print(f"  [cyan]Command:[/] {text!r}  [dim](conf={conf:.2f})[/]")
                try:
                    self.on_command(text)
                except Exception as e:
                    console.print(f"  [red]Command handler error: {e}[/]")
        except Exception as e:
            console.print(f"  [yellow]Process command error: {e}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER VOICE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class VoiceAgent:
    """
    Orchestrates STT + TTS + ContinuousListener.
    Drop-in replacement for the previous VoiceAgent — same public API,
    dramatically improved noise handling.

    Usage:
        va = VoiceAgent(on_command=my_handler)
        va.start_listening()      # background thread
        va.speak("Hello there")   # non-blocking
        va.speak("Stop!", priority=True)  # interrupts current speech
        va.stop_listening()
    """

    def __init__(
        self,
        bus=None,
        on_command:    Callable         = None,
        on_wake:       Callable         = None,
        voice:         str              = DEFAULT_VOICE,
        stt_model:     str              = "auto",
        vad_threshold: float            = 0.45,
    ):
        self.tts       = TTSEngine(voice)
        self.stt       = STTEngine(stt_model)
        self.bus       = bus
        self.on_command = on_command
        self.on_wake    = on_wake
        self._voice     = voice
        self._vad_thr   = vad_threshold
        self._listener: Optional[ContinuousListener] = None

    # ── TTS ────────────────────────────────────────────────────────────────────

    def speak(self, text: str, voice: str = None, priority: bool = False):
        """Non-blocking speech. Use priority=True to interrupt current speech."""
        self.tts.speak(text, voice or self._voice, priority=priority)

    def speak_sync(self, text: str, voice: str = None):
        """Blocking speech — waits until finished."""
        self.tts.speak_sync(text, voice or self._voice)

    # ── STT ────────────────────────────────────────────────────────────────────

    def listen_once(self, duration: float = 6.0, denoise: bool = True) -> str:
        """
        One-shot listen: records for up to `duration` seconds using VAD
        to cut early when silence detected. Returns transcribed text.
        """
        try:
            import sounddevice as sd
            import numpy as np

            console.print(f"  [dim]Listening (up to {duration:.0f}s)...[/]")
            SileroVAD.load()

            frames          = []
            silence_count   = 0
            max_frames      = int(duration * 1000 / FRAME_MS)
            silence_thresh  = int(1.2 * 1000 / FRAME_MS)  # 1.2 s silence = done

            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype="float32", blocksize=FRAME_SAMPLES, latency="low"
            ) as stream:
                for _ in range(max_frames):
                    frame, _ = stream.read(FRAME_SAMPLES)
                    frame    = frame[:, 0] if frame.ndim == 2 else frame
                    frames.append(frame)

                    if not SileroVAD.is_speech(frame):
                        silence_count += 1
                        if silence_count >= silence_thresh and len(frames) > 10:
                            break  # early stop on silence
                    else:
                        silence_count = 0

            if not frames:
                return ""

            audio_np = np.concatenate(frames).astype(np.float32)
            result   = self.stt.transcribe_numpy(audio_np)
            return result.get("text", "")

        except ImportError:
            return ""
        except Exception as e:
            console.print(f"  [yellow]listen_once error: {e}[/]")
            return ""

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe audio from browser WebRTC stream."""
        result = self.stt.transcribe_bytes(audio_bytes)
        return result.get("text", "")

    # ── Continuous listening ───────────────────────────────────────────────────

    def start_listening(self, barge_in: bool = True):
        """
        Start continuous wake-word + command detection in background.
        barge_in=True: if ARIA is speaking and user starts talking, ARIA stops
        and listens immediately (no wake word needed while TTS is active).
        """
        if self._listener and self._listener.is_running:
            return  # already running

        _va = self  # capture for closures

        def handle_command(text: str):
            if _va.on_command:
                _va.on_command(text)
            elif _va.bus:
                try:
                    from agents.agent_bus import Event
                    _va.bus.publish(Event(
                        "voice_command",
                        {"text": text, "source": "microphone"},
                        "voice_agent",
                    ))
                except Exception:
                    pass

        def handle_wake():
            if _va.on_wake:
                _va.on_wake()
            # Play a subtle confirmation sound
            _va.speak("Yes?", priority=True)

        self._listener = ContinuousListener(
            stt=self.stt,
            on_command=handle_command,
            on_wake=handle_wake,
            vad_threshold=self._vad_thr,
        )

        # ── Barge-in: patch stream loop to interrupt TTS on user speech ──────
        if barge_in:
            self._patch_barge_in(self._listener)

        self._listener.start()

    def _patch_barge_in(self, listener: "ContinuousListener") -> None:
        """
        Patch the ContinuousListener's IDLE→ACTIVE transition so that
        if ARIA is currently speaking and VAD detects speech, TTS is
        immediately interrupted and we go ACTIVE without wake word.
        """
        _va  = self
        _orig_check_wake = listener._check_wake_word

        def _barge_in_check(audio_np, np, stream, audio_buf,
                            silence_frames_thresh, max_frames_cmd):
            # If TTS is currently speaking, skip wake-word check and go straight ACTIVE
            if _va.tts._speaking:
                console.print("  [cyan]Barge-in detected[/] — interrupting ARIA speech")
                _va.tts.speak("", priority=True)   # stop current speech immediately
                listener._state = listener.ACTIVE
                listener._preroll_frames.clear()
                return
            # Normal wake-word check
            _orig_check_wake(audio_np, np, stream, audio_buf,
                             silence_frames_thresh, max_frames_cmd)

        import types
        listener._check_wake_word = types.MethodType(
            lambda self_l, audio_np, np, stream, audio_buf, sft, mfc:
                _barge_in_check(audio_np, np, stream, audio_buf, sft, mfc),
            listener
        )

    def stop_listening(self):
        if self._listener:
            self._listener.stop()
            self._listener = None

    # ── Voice management ───────────────────────────────────────────────────────

    def set_voice(self, voice_name: str):
        """Change TTS voice at runtime."""
        self._voice = voice_name
        self.tts.set_voice(voice_name)

    def set_vad_threshold(self, threshold: float):
        """
        Adjust VAD sensitivity.
        Lower = more sensitive (picks up quiet speech, but may catch noise).
        Higher = less sensitive (ignores noise, may miss quiet speech).
        Range: 0.3 – 0.8. Default: 0.45.
        """
        self._vad_thr = max(0.1, min(0.95, threshold))
        if self._listener:
            self._listener.vad_threshold = self._vad_thr

    async def available_voices(self) -> list:
        return await self.tts.list_voices()

    def status(self) -> dict:
        return {
            "tts_voice":    self._voice,
            "stt_model":    self.stt.model_size,
            "stt_engine":   self.stt._engine or "not loaded",
            "listening":    self._listener is not None and self._listener.is_running,
            "vad_threshold": self._vad_thr,
            "wake_words":   WAKE_WORDS,
            "noise_reduction": "noisereduce (spectral subtraction)",
            "vad_engine":   "silero-vad" if SileroVAD._model else "energy-threshold",
            "tts_engines":  ["edge-tts (primary)", "pyttsx3 (offline fallback)"],
        }

    async def handle_websocket(self, websocket) -> None:
        """
        Browser Voice WebSocket handler.
        Accepts:
            {type: "text_query",  text: "..."}          — text from browser STT
            {type: "audio_query", audio_b64: "..."}     — raw audio from browser mic
        Sends:
            {type: "response", transcript: "...", text: "...", audio_b64: "..."}
        """
        import json, base64, asyncio
        from fastapi import WebSocketDisconnect

        # Lazy import aria ref from server module to avoid circular import
        try:
            import server as _srv
            _aria = _srv.aria
        except Exception:
            _aria = {}

        async def _get_answer(text: str) -> str:
            """Run ARIA's neural pipeline and return answer text."""
            # Try NeuralOrchestrator stream first
            neural = _aria.get("neural")
            if neural and hasattr(neural, "stream"):
                answer = ""
                try:
                    async for chunk in neural.stream(text):
                        if chunk.startswith("data:"):
                            import json as _j
                            d = _j.loads(chunk[5:].strip())
                            if d.get("type") == "done" and d.get("text"):
                                answer = d["text"]
                                break
                            elif d.get("type") in ("token", "text", "delta"):
                                answer += d.get("text", "")
                except Exception:
                    pass
                if answer.strip():
                    return answer.strip()
            # Fallback: direct engine generate
            engine = _aria.get("engine")
            if engine:
                try:
                    return engine.generate(
                        text,
                        system="You are ARIA, a helpful voice assistant. Be concise — your response will be spoken aloud.",
                    )
                except Exception:
                    pass
            return "Sorry, I couldn't process that request."

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)

                transcript = ""
                if msg.get("type") == "text_query":
                    transcript = msg.get("text", "").strip()
                elif msg.get("type") == "audio_query":
                    b64 = msg.get("audio_b64", "")
                    if b64:
                        audio_bytes = base64.b64decode(b64)
                        transcript  = self.transcribe_audio_bytes(audio_bytes)

                if not transcript:
                    continue

                # Get ARIA's answer
                answer = await _get_answer(transcript)

                # Synthesize TTS
                audio_bytes = b""
                try:
                    audio_bytes = await self.tts.synthesize(answer, self._voice)
                except Exception:
                    pass

                payload = {
                    "type":       "response",
                    "transcript": transcript,
                    "text":       answer,
                    "audio_b64":  base64.b64encode(audio_bytes).decode() if audio_bytes else "",
                }
                await websocket.send_text(json.dumps(payload))

        except WebSocketDisconnect:
            pass
        except Exception as e:
            console.print(f"  [yellow]Voice WS error: {e}[/]")
