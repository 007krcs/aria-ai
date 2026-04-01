// ARIA Voice Component — React
// Works in: Tauri desktop app · browser · PWA on Android/iOS
//
// Features:
//   Push-to-talk: hold mic button, speak, release
//   Wake word: say "Hey ARIA" anytime (if enabled)
//   Audio visualizer: real-time waveform while speaking
//   Voice selection: pick from available TTS voices
//   Language: auto-detects, or set manually
//
// Install deps:
//   npm install react react-dom
//   (Web Audio API and MediaRecorder are built into every browser)

import { useState, useEffect, useRef, useCallback } from "react";

const API  = "http://localhost:8000";
const WS   = "ws://localhost:8000/ws/voice";

// ─────────────────────────────────────────────────────────────────────────────
// AUDIO VISUALIZER — shows waveform while recording
// ─────────────────────────────────────────────────────────────────────────────

function AudioVisualizer({ stream, isRecording }) {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);

  useEffect(() => {
    if (!stream || !isRecording) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      // Clear canvas
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const audioCtx  = new AudioContext();
    const source    = audioCtx.createMediaStreamSource(stream);
    const analyser  = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    const bufLen = analyser.frequencyBinCount;
    const data   = new Uint8Array(bufLen);

    const draw = () => {
      animRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(data);

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const W   = canvas.width;
      const H   = canvas.height;

      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "#7c6af7";
      ctx.lineWidth   = 2;
      ctx.beginPath();

      const sliceW = W / bufLen;
      let x = 0;
      for (let i = 0; i < bufLen; i++) {
        const v = data[i] / 128.0;
        const y = (v * H) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceW;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();
    };

    draw();
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      audioCtx.close();
    };
  }, [stream, isRecording]);

  return (
    <canvas
      ref={canvasRef}
      width={280}
      height={60}
      style={{
        borderRadius: 8,
        background: "rgba(124,106,247,0.06)",
        display: "block",
      }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CONVERSATION BUBBLE — shows transcript + ARIA response
// ─────────────────────────────────────────────────────────────────────────────

function ConversationBubble({ role, text, isLoading }) {
  const isUser = role === "user";
  return (
    <div
      style={{
        display:       "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom:  8,
      }}
    >
      <div
        style={{
          maxWidth:     "80%",
          padding:      "10px 14px",
          borderRadius: isUser ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
          background:   isUser
            ? "rgba(124,106,247,0.15)"
            : "rgba(255,255,255,0.05)",
          border:       `1px solid ${isUser ? "rgba(124,106,247,0.3)" : "rgba(255,255,255,0.08)"}`,
          fontSize:     13,
          lineHeight:   1.6,
          color:        "#e8e8f0",
        }}
      >
        {isLoading ? (
          <span style={{ color: "#9898b0", fontStyle: "italic" }}>
            ARIA is thinking…
          </span>
        ) : (
          text
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN VOICE COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export default function VoiceInterface() {
  const [mode, setMode]           = useState("idle");
  // idle | recording | processing | speaking | error
  const [conversation, setConv]   = useState([]);
  const [stream, setStream]       = useState(null);
  const [voices, setVoices]       = useState([]);
  const [selectedVoice, setVoice] = useState("en-IN-NeerjaNeural");
  const [status, setStatus]       = useState("");
  const [wsConnected, setWsConn]  = useState(false);
  const [holdToTalk, setHold]     = useState(false);

  const recorderRef = useRef(null);
  const chunksRef   = useRef([]);
  const wsRef       = useRef(null);
  const audioRef    = useRef(null);

  // ── WebSocket connection ───────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setWsConn(true);
      setStatus("Connected");
    };

    ws.onclose = () => {
      setWsConn(false);
      setStatus("Disconnected — reconnecting…");
      setTimeout(connectWS, 3000);
    };

    ws.onerror = () => {
      setStatus("WebSocket error — is server running?");
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);

        if (msg.type === "response") {
          // Show transcript
          if (msg.transcript) {
            setConv(prev => [...prev, { role: "user", text: msg.transcript }]);
          }

          // Show ARIA response text
          if (msg.text) {
            setConv(prev => [...prev, { role: "aria", text: msg.text }]);
          }

          // Play ARIA's voice response
          if (msg.audio_b64) {
            setMode("speaking");
            setStatus("ARIA is speaking…");
            await playBase64Audio(msg.audio_b64);
            setMode("idle");
            setStatus("Ready");
          } else {
            setMode("idle");
            setStatus("Ready");
          }
        }

        if (msg.type === "status") {
          setStatus(JSON.stringify(msg.status));
        }
      }
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWS();
    loadVoices();
    return () => wsRef.current?.close();
  }, [connectWS]);

  // ── Voice list ─────────────────────────────────────────────────────────────
  const loadVoices = async () => {
    try {
      const r = await fetch(`${API}/api/voice/voices`);
      const d = await r.json();
      const indianVoices = (d.voices || []).filter(v =>
        v.locale?.startsWith("en-IN") || v.locale?.startsWith("hi-IN")
      );
      setVoices(indianVoices.slice(0, 8));
    } catch (e) {
      console.error("Voice list error:", e);
    }
  };

  // ── Audio playback ─────────────────────────────────────────────────────────
  const playBase64Audio = (b64) => new Promise((resolve) => {
    const bytes   = atob(b64);
    const buffer  = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) buffer[i] = bytes.charCodeAt(i);
    const blob    = new Blob([buffer], { type: "audio/mpeg" });
    const url     = URL.createObjectURL(blob);
    const audio   = new Audio(url);
    audioRef.current = audio;
    audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
    audio.onerror = () => { URL.revokeObjectURL(url); resolve(); };
    audio.play().catch(resolve);
  });

  // ── Recording ─────────────────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate:   16000,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });
      setStream(mediaStream);
      setMode("recording");
      setStatus("Listening…");

      chunksRef.current = [];
      const recorder    = new MediaRecorder(mediaStream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        const blob       = new Blob(chunksRef.current, { type: "audio/webm" });
        const arrayBuf   = await blob.arrayBuffer();

        if (wsRef.current?.readyState === WebSocket.OPEN) {
          setMode("processing");
          setStatus("Processing…");
          wsRef.current.send(arrayBuf);
        }

        // Stop mic
        mediaStream.getTracks().forEach(t => t.stop());
        setStream(null);
      };

      recorder.start(100);  // collect data every 100ms
      recorderRef.current = recorder;

    } catch (err) {
      setMode("error");
      setStatus(`Mic error: ${err.message}`);
    }
  };

  const stopRecording = () => {
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
  };

  const stopSpeaking = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setMode("idle");
    setStatus("Ready");
  };

  // ── TTS-only (no mic) — type and hear ARIA speak ───────────────────────────
  const speakText = async (text) => {
    if (!text.trim()) return;
    setMode("speaking");
    setStatus("Speaking…");
    try {
      const r = await fetch(`${API}/api/voice/speak-text`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text, voice: selectedVoice }),
      });
      const d = await r.json();
      if (d.audio_b64) {
        await playBase64Audio(d.audio_b64);
      }
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    }
    setMode("idle");
    setStatus("Ready");
  };

  // ── Colours ────────────────────────────────────────────────────────────────
  const modeColor = {
    idle:       "#7c6af7",
    recording:  "#f06060",
    processing: "#f0a040",
    speaking:   "#3dd68c",
    error:      "#f06060",
  };

  const modeLabel = {
    idle:       "Hold to speak",
    recording:  "Release to send",
    processing: "Processing…",
    speaking:   "Speaking…",
    error:      "Error",
  };

  return (
    <div
      style={{
        display:       "flex",
        flexDirection: "column",
        height:        "100%",
        background:    "#0f0f11",
        color:         "#e8e8f0",
        fontFamily:    "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          display:        "flex",
          alignItems:     "center",
          justifyContent: "space-between",
          padding:        "12px 18px",
          borderBottom:   "1px solid #2e2e38",
          background:     "#18181c",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width:        8,
              height:       8,
              borderRadius: "50%",
              background:   wsConnected ? "#3dd68c" : "#f06060",
            }}
          />
          <span style={{ fontSize: 13, color: "#9898b0" }}>
            {wsConnected ? "Voice connected" : "Connecting…"}
          </span>
        </div>

        {/* Voice selector */}
        <select
          value={selectedVoice}
          onChange={e => setVoice(e.target.value)}
          style={{
            background:   "#222228",
            border:       "1px solid #2e2e38",
            borderRadius: 6,
            color:        "#e8e8f0",
            padding:      "4px 8px",
            fontSize:     11,
          }}
        >
          <option value="en-IN-NeerjaNeural">Neerja (English IN ♀)</option>
          <option value="en-IN-PrabhatNeural">Prabhat (English IN ♂)</option>
          <option value="hi-IN-SwaraNeural">Swara (Hindi ♀)</option>
          <option value="hi-IN-MadhurNeural">Madhur (Hindi ♂)</option>
          <option value="en-US-AriaNeural">Aria (US ♀)</option>
          {voices.filter(v => !["en-IN-NeerjaNeural","en-IN-PrabhatNeural",
            "hi-IN-SwaraNeural","hi-IN-MadhurNeural","en-US-AriaNeural"]
            .includes(v.name)).slice(0,3).map(v => (
            <option key={v.name} value={v.name}>{v.name}</option>
          ))}
        </select>
      </div>

      {/* Conversation */}
      <div
        style={{
          flex:      1,
          overflowY: "auto",
          padding:   "16px 18px",
        }}
      >
        {conversation.length === 0 && (
          <div
            style={{
              textAlign:  "center",
              color:      "#5a5a72",
              fontSize:   13,
              marginTop:  40,
            }}
          >
            <div style={{ fontSize: 32, marginBottom: 12 }}>🎙️</div>
            Hold the button below to speak to ARIA
            <br />
            <span style={{ fontSize: 11, color: "#3a3a52" }}>
              or say "Hey ARIA" if wake word is enabled
            </span>
          </div>
        )}
        {conversation.map((msg, i) => (
          <ConversationBubble key={i} role={msg.role} text={msg.text} />
        ))}
      </div>

      {/* Visualizer */}
      {mode === "recording" && (
        <div style={{ padding: "0 18px 8px", display: "flex", justifyContent: "center" }}>
          <AudioVisualizer stream={stream} isRecording={mode === "recording"} />
        </div>
      )}

      {/* Status */}
      <div
        style={{
          textAlign:  "center",
          fontSize:   11,
          color:      "#5a5a72",
          padding:    "4px 0",
        }}
      >
        {status || "Ready"}
      </div>

      {/* Controls */}
      <div
        style={{
          padding:     "16px 18px 24px",
          borderTop:   "1px solid #2e2e38",
          background:  "#18181c",
          display:     "flex",
          alignItems:  "center",
          justifyContent: "center",
          gap:         20,
        }}
      >
        {/* Stop button (visible when speaking) */}
        {mode === "speaking" && (
          <button
            onClick={stopSpeaking}
            style={{
              width:        44,
              height:       44,
              borderRadius: "50%",
              border:       "1px solid #2e2e38",
              background:   "#222228",
              color:        "#9898b0",
              cursor:       "pointer",
              fontSize:     18,
              display:      "flex",
              alignItems:   "center",
              justifyContent: "center",
            }}
          >
            ⏹
          </button>
        )}

        {/* Main mic button — hold to talk */}
        <button
          onMouseDown={() => { setHold(true); startRecording(); }}
          onMouseUp={() => { setHold(false); stopRecording(); }}
          onTouchStart={(e) => { e.preventDefault(); setHold(true); startRecording(); }}
          onTouchEnd={() => { setHold(false); stopRecording(); }}
          disabled={mode === "processing" || mode === "speaking"}
          style={{
            width:        80,
            height:       80,
            borderRadius: "50%",
            border:       `3px solid ${modeColor[mode]}`,
            background:   holdToTalk || mode === "recording"
              ? `${modeColor[mode]}22`
              : "transparent",
            color:        modeColor[mode],
            cursor:       "pointer",
            fontSize:     32,
            display:      "flex",
            alignItems:   "center",
            justifyContent: "center",
            transition:   "all 0.15s",
            transform:    holdToTalk ? "scale(0.92)" : "scale(1)",
            boxShadow:    holdToTalk || mode === "recording"
              ? `0 0 0 8px ${modeColor[mode]}18`
              : "none",
          }}
        >
          {mode === "processing" ? "⏳" :
           mode === "speaking"   ? "🔊" :
           mode === "recording"  ? "🔴" : "🎙️"}
        </button>

        {/* Clear history */}
        <button
          onClick={() => setConv([])}
          style={{
            width:        44,
            height:       44,
            borderRadius: "50%",
            border:       "1px solid #2e2e38",
            background:   "#222228",
            color:        "#5a5a72",
            cursor:       "pointer",
            fontSize:     16,
            display:      "flex",
            alignItems:   "center",
            justifyContent: "center",
          }}
          title="Clear conversation"
        >
          🗑
        </button>
      </div>

      {/* Bottom label */}
      <div
        style={{
          textAlign:  "center",
          fontSize:   11,
          color:      "#3a3a52",
          paddingBottom: 8,
        }}
      >
        {modeLabel[mode]}
      </div>
    </div>
  );
}
