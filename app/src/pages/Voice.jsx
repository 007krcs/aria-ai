import { useState, useRef, useEffect, useCallback } from "react";
import { useStore, api } from "../store";

// ── Web Speech API availability check ────────────────────────────────────────
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const HAS_SR = !!SR;
const HAS_MEDIA_DEVICES = !!(navigator.mediaDevices?.getUserMedia);

export default function Voice() {
  const { selectedVoice, setVoice, addNotification } = useStore();
  const [mode, setMode]         = useState("idle");   // idle|listening|thinking|speaking
  const [conv, setConv]         = useState([]);
  const [interim, setInterim]   = useState("");
  const [wsReady, setWsReady]   = useState(false);
  const [live, setLive]         = useState(false);
  const [wakeActive, setWakeActive] = useState(false); // always-on "Hey ARIA" listener
  const [volume, setVolume]     = useState(1.0);       // 0.0 – 1.0
  const [inputMode, setInputMode] = useState("auto"); // auto|sr|whisper
  const wsRef       = useRef(null);
  const audioRef    = useRef(null);
  const recogRef    = useRef(null);
  const mediaRecRef = useRef(null);                   // MediaRecorder for Whisper fallback
  const canvasRef   = useRef(null);
  const animRef     = useRef(null);
  const modeRef     = useRef("idle");
  const liveRef     = useRef(false);
  const convEndRef  = useRef(null);
  const wakeRecRef  = useRef(null);  // always-on wake word recogniser
  const analyserRef = useRef(null);
  const audioCtxRef = useRef(null);
  const volumeRef   = useRef(1.0);

  useEffect(() => { modeRef.current = mode; }, [mode]);
  useEffect(() => { liveRef.current = live; }, [live]);
  useEffect(() => { volumeRef.current = volume; }, [volume]);
  useEffect(() => { convEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [conv, interim]);

  // Load history
  useEffect(() => {
    api.get("/api/voice/history?turns=30")
      .then(r => r.json())
      .then(d => { if (d.history?.length) setConv(d.history.map(h => ({ role: h.role, text: h.text }))); })
      .catch(() => {});
  }, []);

  // ── WebSocket ────────────────────────────────────────────────────────────
  useEffect(() => {
    let reconnTimer;
    const connect = () => {
      const ws = new WebSocket("ws://localhost:8000/ws/voice");
      ws.onopen  = () => setWsReady(true);
      ws.onclose = () => { setWsReady(false); reconnTimer = setTimeout(connect, 3000); };
      ws.onerror = () => setWsReady(false);
      ws.onmessage = async (e) => {
        if (typeof e.data !== "string") return;
        const msg = JSON.parse(e.data);
        if (msg.type !== "response") return;

        setInterim("");
        if (msg.transcript) setConv(p => [...p, { role: "user", text: msg.transcript }]);
        if (msg.text)       setConv(p => [...p, { role: "aria", text: msg.text }]);

        if (msg.audio_b64) {
          setMode("speaking");
          stopViz();
          await playB64(msg.audio_b64);
        }
        setMode("idle");
        // Auto-restart in live mode after ARIA finishes speaking
        if (liveRef.current) setTimeout(() => { if (liveRef.current) startListening(); }, 300);
      };
      wsRef.current = ws;
    };
    connect();
    return () => { clearTimeout(reconnTimer); wsRef.current?.close(); };
  }, []);

  // ── Mic visualizer ───────────────────────────────────────────────────────
  const startViz = useCallback(async () => {
    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx      = new AudioContext();
      const src      = ctx.createMediaStreamSource(stream);
      const an       = ctx.createAnalyser();
      an.fftSize     = 128;
      src.connect(an);
      audioCtxRef.current = ctx;
      analyserRef.current = an;
      const data   = new Uint8Array(an.frequencyBinCount);
      const canvas = canvasRef.current;
      const draw = () => {
        if (!canvas || !analyserRef.current) return;
        animRef.current = requestAnimationFrame(draw);
        an.getByteFrequencyData(data);
        const c = canvas.getContext("2d");
        c.clearRect(0, 0, canvas.width, canvas.height);
        const barW = canvas.width / data.length * 2.5;
        data.forEach((v, i) => {
          const h = (v / 255) * canvas.height * 0.85;
          c.fillStyle = `rgba(124,106,247,${0.35 + v/255*0.65})`;
          c.fillRect(i * (barW + 1), canvas.height - h, barW, h);
        });
      };
      draw();
    } catch { /* no mic permission for viz */ }
  }, []);

  const stopViz = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    analyserRef.current = null;
    if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; }
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  // ── Whisper fallback: MediaRecorder → POST /api/transcribe → WebSocket ──
  const startWhisperRecording = useCallback(() => {
    if (!HAS_MEDIA_DEVICES) { addNotification("Microphone not accessible", "error"); return; }
    if (modeRef.current !== "idle") return;
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }

    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      const chunks = [];
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus" : "audio/webm";
      const mr = new MediaRecorder(stream, { mimeType });
      mediaRecRef.current = mr;

      mr.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        stopViz();
        if (chunks.length === 0 || modeRef.current === "idle") {
          setMode("idle");
          return;
        }
        setMode("thinking");
        const blob = new Blob(chunks, { type: mimeType });
        // Try backend Whisper first
        try {
          const form = new FormData();
          form.append("file", blob, "voice.webm");
          const tok = window.__ariaToken || localStorage.getItem("aria_token") || "";
          const r = await fetch("/api/transcribe-quick", {
            method: "POST",
            headers: tok ? { Authorization: "Bearer " + tok } : {},
            body: form,
          });
          const d = await r.json().catch(() => ({}));
          const transcript = d.text || d.transcript || "";
          if (transcript.trim()) {
            setInterim("");
            if (wsRef.current?.readyState === 1) {
              wsRef.current.send(JSON.stringify({ type: "text_query", text: transcript.trim() }));
            } else {
              setMode("idle");
              addNotification("Voice server disconnected", "error");
            }
          } else {
            setMode("idle");
            if (liveRef.current) setTimeout(() => { if (liveRef.current) startListeningAuto(); }, 500);
          }
        } catch {
          setMode("idle");
          addNotification("Transcription failed — check server connection", "error");
        }
      };

      setMode("listening");
      startViz();
      mr.start(200); // 200ms timeslice so ondataavailable fires regularly
      // Auto-stop after 10s
      setTimeout(() => { if (mr.state === "recording") mr.stop(); }, 10000);
    }).catch(() => {
      setMode("idle");
      addNotification("Microphone permission denied", "error");
    });
  }, [startViz, stopViz, addNotification]);

  // ── Web Speech API listening ─────────────────────────────────────────────
  const startListening = useCallback(() => {
    if (!HAS_SR) { startWhisperRecording(); return; }
    if (modeRef.current !== "idle") return;
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }

    const rec = new SR();
    rec.continuous      = false;
    rec.interimResults  = true;
    rec.lang            = "";
    rec.maxAlternatives = 1;

    rec.onstart = () => { setMode("listening"); startViz(); };

    rec.onresult = (event) => {
      const results    = Array.from(event.results);
      const transcript = results.map(r => r[0].transcript).join("");
      const isFinal    = results[results.length - 1].isFinal;
      setInterim(transcript);

      if (isFinal && transcript.trim()) {
        rec.stop();
        setInterim("");
        setMode("thinking");
        stopViz();
        if (wsRef.current?.readyState === 1) {
          wsRef.current.send(JSON.stringify({ type: "text_query", text: transcript.trim() }));
        } else {
          setMode("idle");
          addNotification("Voice server disconnected", "error");
        }
      }
    };

    rec.onerror = (e) => {
      stopViz();
      if (e.error === "no-speech") {
        setMode("idle");
        if (liveRef.current) setTimeout(() => { if (liveRef.current) startListeningAuto(); }, 500);
      } else if (e.error === "network" || e.error === "service-not-allowed") {
        // Cloud SR blocked — switch to local Whisper silently
        setMode("idle");
        setInputMode("whisper");
        setTimeout(() => startWhisperRecording(), 100);
      } else if (e.error !== "aborted") {
        addNotification(`Speech error: ${e.error}`, "error");
        setMode("idle");
      }
    };

    rec.onend = () => {
      stopViz();
      if (modeRef.current === "listening") setMode("idle");
    };

    recogRef.current = rec;
    try { rec.start(); } catch { startWhisperRecording(); }
  }, [startViz, stopViz, addNotification, startWhisperRecording]);

  // Auto-route based on last-known working mode
  const startListeningAuto = useCallback(() => {
    if (inputMode === "whisper") { startWhisperRecording(); }
    else { startListening(); }
  }, [inputMode, startListening, startWhisperRecording]);

  const stopListening = useCallback(() => {
    try { recogRef.current?.stop(); } catch {}
    try { if (mediaRecRef.current?.state === "recording") mediaRecRef.current.stop(); } catch {}
    stopViz();
    setMode("idle");
  }, [stopViz]);

  const toggleLive = useCallback(() => {
    setLive(prev => {
      const next = !prev;
      if (next) {
        setTimeout(startListeningAuto, 200);
      } else {
        stopListening();
        if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
        setMode("idle");
      }
      return next;
    });
  }, [startListeningAuto, stopListening]);

  const playB64 = useCallback((b64) => new Promise(resolve => {
    try {
      const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const blob  = new Blob([bytes], { type: "audio/mpeg" });
      const url   = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.volume = volumeRef.current;
      audioRef.current = audio;
      const cleanup = () => { URL.revokeObjectURL(url); audioRef.current = null; resolve(); };
      audio.onended = cleanup;
      audio.onerror = cleanup;
      // Some browsers need a user-gesture — use play() and handle NotAllowedError
      audio.play().catch(() => {
        // If autoplay blocked, try AudioContext decode path
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        ctx.decodeAudioData(bytes.buffer.slice(0), (buf) => {
          const src = ctx.createBufferSource();
          const gain = ctx.createGain();
          gain.gain.value = volumeRef.current;
          src.buffer = buf;
          src.connect(gain);
          gain.connect(ctx.destination);
          src.onended = () => { ctx.close(); cleanup(); };
          src.start();
          audioRef.current = { pause: () => { src.stop(); ctx.close(); }, _ctx: ctx };
        }, () => { ctx.close(); cleanup(); });
      });
    } catch { resolve(); }
  }), []);

  // ── Wake word: always-on "Hey ARIA" / "ARIA" detection ──────────────────
  const startWakeListener = useCallback(() => {
    if (!HAS_SR) return;
    const stopWake = () => {
      try { wakeRecRef.current?.stop(); } catch {}
      wakeRecRef.current = null;
    };
    const launch = () => {
      if (!wakeActive) return;  // user turned it off
      if (modeRef.current !== "idle") { setTimeout(launch, 1000); return; }
      const r = new SR();
      r.continuous     = true;
      r.interimResults = true;
      r.lang           = "";
      r.onresult = (event) => {
        const last  = event.results[event.results.length - 1];
        const heard = last[0].transcript.toLowerCase().trim();
        const TRIGGERS = ["hey aria", "ok aria", "aria", "hi aria", "hello aria"];
        if (TRIGGERS.some(t => heard.includes(t))) {
          stopWake();
          if (modeRef.current === "idle") startListeningAuto();
        }
      };
      r.onend  = () => { if (wakeActive) setTimeout(launch, 300); };
      r.onerror = () => { if (wakeActive) setTimeout(launch, 1000); };
      try { r.start(); wakeRecRef.current = r; } catch { setTimeout(launch, 1000); }
    };
    launch();
  }, [wakeActive, startListeningAuto]);

  const stopWakeListener = useCallback(() => {
    try { wakeRecRef.current?.stop(); } catch {}
    wakeRecRef.current = null;
  }, []);

  useEffect(() => {
    if (wakeActive) startWakeListener();
    else stopWakeListener();
    return stopWakeListener;
  }, [wakeActive, startWakeListener, stopWakeListener]);

  // ── Pulsing ring animation for listening state ───────────────────────────
  const ringStyle = (active) => ({
    position: "absolute",
    width: 100, height: 100,
    borderRadius: "50%",
    background: "rgba(124,106,247,.15)",
    animation: active ? "pulse 1.2s ease-out infinite" : "none",
    pointerEvents: "none",
  });

  const COLOR = { idle:"var(--accent)", listening:"#7c6af7", thinking:"var(--amber)", speaking:"var(--green)" };
  const ICON  = { idle:"🎙", listening:"🎙", thinking:"◌", speaking:"🔊" };

  const statusText = () => {
    if (!wsReady)                    return "Connecting to voice server…";
    if (inputMode === "whisper" && mode === "idle") return "Using local Whisper (offline mode)";
    if (!HAS_SR && mode === "idle")  return "Using local Whisper — Chrome preferred for cloud SR";
    if (mode === "idle" && live)     return "Listening for you…";
    if (mode === "idle" && wakeActive) return 'Say "Hey ARIA" to activate…';
    if (mode === "idle")             return "Tap mic to speak  ·  ▶ Live  ·  👂 Wake word";
    if (mode === "listening")        return interim || (inputMode === "whisper" ? "Recording… tap to stop" : "Listening…");
    if (mode === "thinking")         return "ARIA is thinking…";
    if (mode === "speaking")         return "ARIA is speaking…  (tap mic to interrupt)";
    return "";
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:"var(--bg)",
      fontFamily:"inherit" }}>

      {/* Header */}
      <div style={{ padding:"10px 16px", borderBottom:"1px solid var(--border)", background:"var(--bg2)",
        display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ width:7, height:7, borderRadius:"50%",
            background: wsReady ? "var(--green)" : "var(--red)", flexShrink:0 }} />
          <span style={{ fontSize:14, fontWeight:500 }}>Voice</span>
          <span style={{ fontSize:11, color:"var(--text3)" }}>{wsReady ? "connected" : "connecting…"}</span>
        </div>
        {live && (
          <div style={{ fontSize:11, fontWeight:700, color:"#7c6af7",
            background:"rgba(124,106,247,.15)", borderRadius:20, padding:"3px 11px",
            border:"1px solid rgba(124,106,247,.35)", letterSpacing:1, animation:"pulse 2s infinite" }}>
            ● LIVE
          </div>
        )}
        <select
          value={selectedVoice}
          onChange={e => { setVoice(e.target.value); api.post(`/api/voice/set-voice?voice=${encodeURIComponent(e.target.value)}`, {}); }}
          style={{ background:"var(--bg3)", border:"1px solid var(--border)",
            borderRadius:6, color:"var(--text)", padding:"4px 8px", fontSize:11 }}>
          {[
            ["en-IN-NeerjaNeural",  "Neerja — English Indian ♀"],
            ["en-IN-PrabhatNeural", "Prabhat — English Indian ♂"],
            ["hi-IN-SwaraNeural",   "Swara — Hindi ♀"],
            ["hi-IN-MadhurNeural",  "Madhur — Hindi ♂"],
            ["en-US-AriaNeural",    "Aria — US English ♀"],
            ["en-GB-SoniaNeural",   "Sonia — UK English ♀"],
          ].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      </div>

      {/* Conversation */}
      <div style={{ flex:1, overflowY:"auto", padding:"16px 18px" }}>
        {conv.length === 0 && !interim && (
          <div style={{ textAlign:"center", color:"var(--text3)", marginTop:56 }}>
            <div style={{ fontSize:52, marginBottom:12 }}>🎙</div>
            <div style={{ fontSize:15, color:"var(--text2)", marginBottom:6, fontWeight:500 }}>
              Talk to ARIA
            </div>
            <div style={{ fontSize:12, marginBottom:3 }}>Tap the mic to speak once</div>
            <div style={{ fontSize:12 }}>Tap <strong>▶</strong> for hands-free Live conversation</div>
          </div>
        )}
        {conv.map((m, i) => (
          <div key={i} style={{ display:"flex",
            justifyContent: m.role === "user" ? "flex-end" : "flex-start", marginBottom:12 }}>
            {m.role === "aria" && (
              <div style={{ width:30, height:30, borderRadius:"50%", background:"rgba(124,106,247,.2)",
                display:"flex", alignItems:"center", justifyContent:"center",
                fontSize:13, fontWeight:700, marginRight:8, flexShrink:0, alignSelf:"flex-end",
                color:"#7c6af7" }}>A</div>
            )}
            <div style={{
              maxWidth:"70%", padding:"10px 14px", fontSize:14, lineHeight:1.65,
              borderRadius: m.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
              background: m.role === "user" ? "rgba(124,106,247,.18)" : "var(--bg3)",
              border: `1px solid ${m.role==="user" ? "rgba(124,106,247,.3)" : "var(--border)"}`,
              color: "var(--text)",
            }}>
              {m.text}
            </div>
          </div>
        ))}

        {/* Live interim transcript — shows as user speaks */}
        {interim && (
          <div style={{ display:"flex", justifyContent:"flex-end", marginBottom:12 }}>
            <div style={{ maxWidth:"70%", padding:"10px 14px", fontSize:14, lineHeight:1.65,
              borderRadius:"18px 18px 4px 18px",
              background:"rgba(124,106,247,.08)",
              border:"1px dashed rgba(124,106,247,.3)",
              color:"rgba(124,106,247,.85)", fontStyle:"italic" }}>
              {interim}
            </div>
          </div>
        )}

        {/* ARIA thinking indicator */}
        {mode === "thinking" && (
          <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:12 }}>
            <div style={{ width:30, height:30, borderRadius:"50%", background:"rgba(124,106,247,.2)",
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:13, fontWeight:700, color:"#7c6af7" }}>A</div>
            <div style={{ display:"flex", gap:4, padding:"12px 16px",
              background:"var(--bg3)", borderRadius:"18px 18px 18px 4px",
              border:"1px solid var(--border)" }}>
              {[0,1,2].map(i => (
                <div key={i} style={{ width:6, height:6, borderRadius:"50%",
                  background:"rgba(124,106,247,.6)",
                  animation:`bounce 1.2s ${i*0.2}s ease-in-out infinite` }} />
              ))}
            </div>
          </div>
        )}
        <div ref={convEndRef} />
      </div>

      {/* Waveform */}
      <div style={{ height:48, display:"flex", justifyContent:"center", alignItems:"center", padding:"0 18px" }}>
        <canvas ref={canvasRef} width={340} height={38}
          style={{ borderRadius:8, background:"rgba(124,106,247,.04)", display:"block" }} />
      </div>

      {/* Status — shows live transcript */}
      <div style={{ textAlign:"center", fontSize:12, padding:"2px 24px 6px",
        color: mode === "listening" ? "#7c6af7" : mode === "thinking" ? "var(--amber)" : "var(--text3)",
        fontWeight: mode === "listening" ? 500 : 400,
        minHeight:18, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
        {statusText()}
      </div>

      {/* Volume slider */}
      <div style={{ padding:"4px 28px 0", display:"flex", alignItems:"center", gap:10 }}>
        <span style={{ fontSize:13, color:"var(--text3)" }}>🔈</span>
        <input
          type="range" min={0} max={1} step={0.05} value={volume}
          onChange={e => {
            const v = parseFloat(e.target.value);
            setVolume(v);
            if (audioRef.current && typeof audioRef.current.volume === "number")
              audioRef.current.volume = v;
          }}
          style={{ flex:1, accentColor:"#7c6af7", cursor:"pointer" }}
          title={`Volume: ${Math.round(volume * 100)}%`}
        />
        <span style={{ fontSize:13, color:"var(--text3)" }}>🔊</span>
        <span style={{ fontSize:11, color:"var(--text3)", minWidth:30 }}>
          {Math.round(volume * 100)}%
        </span>
        {/* Input mode indicator */}
        <span
          title={inputMode === "whisper" ? "Using local Whisper (cloud SR unavailable)" : "Using browser speech recognition"}
          style={{ fontSize:11, padding:"2px 7px", borderRadius:10, cursor:"default",
            background: inputMode === "whisper" ? "rgba(255,180,0,.15)" : "rgba(124,106,247,.1)",
            color: inputMode === "whisper" ? "var(--amber)" : "#7c6af7",
            border: `1px solid ${inputMode === "whisper" ? "rgba(255,180,0,.3)" : "rgba(124,106,247,.25)"}` }}>
          {inputMode === "whisper" ? "Whisper" : "Cloud SR"}
        </span>
      </div>

      {/* Controls */}
      <div style={{ padding:"6px 18px 24px", display:"flex",
        alignItems:"center", justifyContent:"center", gap:20,
        background:"var(--bg2)", borderTop:"1px solid var(--border)" }}>

        {/* Clear */}
        <button onClick={() => { setConv([]); setInterim(""); api.post("/api/voice/history/clear", {}).catch(()=>{}); }}
          title="Clear conversation"
          style={{ width:44, height:44, borderRadius:"50%", border:"1px solid var(--border)",
            background:"var(--bg3)", color:"var(--text3)", cursor:"pointer", fontSize:16 }}>
          🗑
        </button>

        {/* Main mic button */}
        <div style={{ position:"relative", display:"flex", alignItems:"center", justifyContent:"center" }}>
          {(mode === "listening" || (live && mode === "idle")) && <div style={ringStyle(true)} />}
          <button
            onClick={() => {
              if (mode === "speaking") {
                if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
                setMode("idle");
                if (live) setTimeout(startListeningAuto, 200);
              } else if (mode === "listening") {
                stopListening();
              } else if (mode === "idle") {
                startListeningAuto();
              }
            }}
            disabled={mode === "thinking"}
            title={mode === "speaking" ? "Tap to interrupt" : mode === "listening" ? "Tap to stop" : "Tap to speak"}
            style={{
              width:82, height:82, borderRadius:"50%", position:"relative", zIndex:1,
              border:`3px solid ${COLOR[mode]}`,
              background: mode !== "idle" ? `${COLOR[mode]}15` : live ? "rgba(124,106,247,.06)" : "transparent",
              color: COLOR[mode], fontSize:28, cursor: mode === "thinking" ? "default" : "pointer",
              display:"flex", alignItems:"center", justifyContent:"center",
              transition:"all .2s",
              boxShadow: mode === "listening" ? `0 0 0 8px rgba(124,106,247,.12)` : "none",
            }}>
            {mode === "thinking"
              ? <span style={{ fontSize:22, display:"inline-block", animation:"spin 1s linear infinite" }}>◌</span>
              : ICON[mode]}
          </button>
        </div>

        {/* Live toggle */}
        <button onClick={toggleLive}
          title={live ? "Stop Live mode" : "Start Live conversation"}
          style={{
            width:44, height:44, borderRadius:"50%",
            border:`1px solid ${live ? "rgba(124,106,247,.6)" : "var(--border)"}`,
            background: live ? "rgba(124,106,247,.15)" : "var(--bg3)",
            color: live ? "#7c6af7" : "var(--text3)",
            cursor:"pointer", fontSize:live ? 14 : 16, fontWeight:700,
          }}>
          {live ? "■" : "▶"}
        </button>

        {/* Wake word toggle */}
        <button
          onClick={() => setWakeActive(p => !p)}
          title={wakeActive ? 'Wake word ON — say "Hey ARIA" anytime' : 'Enable "Hey ARIA" wake word'}
          style={{
            width:44, height:44, borderRadius:"50%",
            border:`1px solid ${wakeActive ? "rgba(0,212,255,.5)" : "var(--border)"}`,
            background: wakeActive ? "rgba(0,212,255,.12)" : "var(--bg3)",
            color: wakeActive ? "var(--cyan)" : "var(--text3)",
            cursor:"pointer", fontSize:16,
            animation: wakeActive ? "pulse 2s infinite" : "none",
          }}>
          👂
        </button>
      </div>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:.6;transform:scale(1)} 50%{opacity:1;transform:scale(1.08)} }
        @keyframes spin  { to { transform: rotate(360deg) } }
        @keyframes bounce{ 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }
      `}</style>
    </div>
  );
}
