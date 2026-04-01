import { useState, useRef, useEffect, useCallback } from "react";
import { API_BASE } from "../lib/api";

const getAuthHeader = () => ({
  "Authorization": "Bearer " + (window.__ariaToken || localStorage.getItem("aria_token") || ""),
});

export default function InputBar({ onSend, disabled }) {
  const [text, setText]         = useState("");
  const [listening, setListening] = useState(false);
  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  }, [text]);

  const handleSend = useCallback(() => {
    const val = text.trim();
    if (!val || disabled) return;
    onSend(val);
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [text, disabled, onSend]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Fallback: record via MediaRecorder → send to backend Whisper
  const mediaRecRef = useRef(null);

  const _startWhisperFallback = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecRef.current = mr;
      mr.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        setListening(false);
        const blob = new Blob(chunks, { type: "audio/webm" });
        const form = new FormData();
        form.append("file", blob, "voice.webm");
        try {
          const r = await fetch(API_BASE + "/api/transcribe-quick", {
            method: "POST",
            headers: getAuthHeader(),
            body: form,
          });
          const d = await r.json().catch(() => ({}));
          const transcript = d.text || d.transcript || "";
          if (transcript.trim()) { onSend(transcript.trim()); setText(""); }
        } catch { /* network error — ignore */ }
      };
      mr.start();
      setListening(true);
      // Auto-stop after 8 seconds max
      setTimeout(() => { if (mr.state === "recording") mr.stop(); }, 8000);
    } catch {
      setListening(false);
    }
  }, [onSend]);

  const handleVoice = useCallback(() => {
    // Stop if already recording (either SR or MediaRecorder)
    if (listening) {
      recognitionRef.current?.stop();
      if (mediaRecRef.current?.state === "recording") mediaRecRef.current.stop();
      setListening(false);
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      // No Web Speech API — go straight to Whisper
      _startWhisperFallback();
      return;
    }

    const rec = new SpeechRecognition();
    rec.lang = "";          // auto-detect language
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.continuous = false;

    rec.onstart  = () => setListening(true);
    rec.onend    = () => setListening(false);
    rec.onerror  = (e) => {
      setListening(false);
      if (e.error === "network" || e.error === "service-not-allowed") {
        // Cloud SR blocked — fall back to local Whisper
        _startWhisperFallback();
      }
      // "no-speech", "aborted" are benign — ignore silently
    };

    rec.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      setText(transcript);
      setListening(false);
      setTimeout(() => {
        if (transcript.trim()) { onSend(transcript.trim()); setText(""); }
      }, 80);
    };

    recognitionRef.current = rec;
    try { rec.start(); } catch { _startWhisperFallback(); }
  }, [listening, _startWhisperFallback, onSend]);

  const handleAttach = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const r = await fetch(API_BASE + "/api/upload", {
        method: "POST",
        headers: getAuthHeader(),
        body: formData,
      });
      if (r.status === 401) {
        // Token expired — clear and reload to show login
        localStorage.removeItem("aria_token");
        window.__ariaToken = "";
        window.location.reload();
        return;
      }
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        onSend?.(`[Upload failed: ${d.detail || d.error || r.status}]`);
      } else {
        onSend?.(`[File uploaded: **${file.name}** — ${d.chunks ? d.chunks + " chunks indexed" : "ready"}]`);
      }
    } catch (err) {
      onSend?.(`[Upload error: ${err.message}]`);
    }
    e.target.value = "";
  };

  const iconBtn = {
    width: 34, height: 34, borderRadius: "var(--r)",
    border: "1px solid var(--border)", background: "var(--bg2)",
    color: "var(--text2)", fontSize: 15,
    cursor: "pointer", display: "flex", alignItems: "center",
    justifyContent: "center", flexShrink: 0,
    transition: "all .12s",
  };

  return (
    <div className="input-bar">
      {/* Voice button */}
      <button
        onClick={handleVoice}
        disabled={disabled}
        className={listening ? "listening" : ""}
        style={{
          ...iconBtn,
          borderColor: listening ? "var(--cyan)" : undefined,
          color: listening ? "var(--cyan)" : undefined,
          background: listening ? "var(--glow2)" : undefined,
        }}
        title={listening ? "Stop listening" : "Voice input"}
      >
        🎙
      </button>

      {/* Waveform (visible when listening) */}
      {listening && (
        <div className="waveform" style={{ position: "absolute", bottom: "100%", left: 16, marginBottom: 6 }}>
          {[0, 1, 2, 3, 4, 5].map(i => (
            <div
              key={i}
              className="wave-bar"
              style={{ animationDelay: `${i * 0.1}s` }}
            />
          ))}
        </div>
      )}

      {/* Textarea */}
      <textarea
        ref={textareaRef}
        className="input-textarea"
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={1}
        placeholder={listening ? "Listening…" : "Ask ARIA anything… (Enter to send)"}
      />

      {/* Attach button */}
      <label style={{ ...iconBtn, position: "relative" }} title="Attach file">
        📎
        <input
          type="file"
          style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer", width: "100%", height: "100%" }}
          onChange={handleAttach}
          disabled={disabled}
        />
      </label>

      {/* Send button */}
      <button
        onClick={handleSend}
        disabled={disabled || !text.trim()}
        style={{
          ...iconBtn,
          borderColor: (!disabled && text.trim()) ? "var(--cyan)" : "var(--border)",
          color: (!disabled && text.trim()) ? "var(--cyan)" : "var(--text3)",
          background: (!disabled && text.trim()) ? "var(--glow2)" : undefined,
          boxShadow: (!disabled && text.trim()) ? "0 0 8px var(--glow2)" : undefined,
        }}
        title="Send"
      >
        ▶
      </button>
    </div>
  );
}
