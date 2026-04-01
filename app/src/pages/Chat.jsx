import { useState, useRef, useEffect, useCallback } from "react";
import { useARIA } from "../App";
import MessageBubble from "../components/MessageBubble";
import InputBar from "../components/InputBar";
import { API_BASE } from "../lib/api";

const getAuthHeader = () => ({
  "Authorization": "Bearer " + (window.__ariaToken || localStorage.getItem("aria_token") || ""),
});

const QUICK_CHIPS = [
  "Top 10 Stocks today",
  "Take a screenshot",
  "What's the weather?",
  "Latest AI news",
  "Open Chrome",
  "What can you do?",
];

function fmtTime(ts) {
  return new Date(ts).toLocaleTimeString("en-US", {
    hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

function ThinkingBubble({ phase }) {
  const labels = {
    thinking:   { label: "Thinking",          color: "var(--cyan)"    },
    searching:  { label: "Searching memory",  color: "var(--blue)"    },
    websearch:  { label: "Searching the web", color: "var(--safe)"    },
    generating: { label: "Generating",        color: "var(--safe)"    },
  };
  const cfg = labels[phase] || labels.thinking;
  return (
    <div style={{
      alignSelf: "flex-start",
      background: "var(--bg2)", border: "1px solid var(--border)",
      borderRadius: 10, padding: "10px 14px",
      display: "flex", alignItems: "center", gap: 10,
    }}>
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        {[0, 1, 2].map(i => (
          <span key={i} style={{
            display: "inline-block", width: 7, height: 7, borderRadius: "50%",
            background: cfg.color,
            animation: `bncDot 1.2s ease-in-out ${i * 0.2}s infinite`,
          }} />
        ))}
      </div>
      <span style={{ fontSize: 12, color: "var(--text3)" }}>{cfg.label}…</span>
    </div>
  );
}

export default function Chat() {
  const aria = useARIA();
  const sessionId = useRef("aria-" + Date.now());
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [thinkPhase, setThinkPhase] = useState("thinking");
  const bottomRef = useRef(null);
  const streamingIdRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  const addMessage = useCallback((msg) => {
    setMessages(prev => [...prev, { id: Date.now() + Math.random(), time: fmtTime(Date.now()), ...msg }]);
  }, []);

  const updateStreamingMessage = useCallback((id, patch) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, ...patch } : m));
  }, []);

  const handleSend = useCallback(async (text) => {
    if (!text.trim() || streaming) return;

    // Add user message
    addMessage({ role: "user", text });
    setStreaming(true);
    setThinkPhase("thinking");

    // Create a placeholder streaming message
    const streamId = Date.now() + Math.random();
    streamingIdRef.current = streamId;
    setMessages(prev => [...prev, {
      id: streamId,
      role: "aria",
      text: "",
      risk: "safe",
      planId: null,
      suggest: null,
      time: fmtTime(Date.now()),
      _streaming: true,
    }]);

    let accText = "";
    let currentRisk = "safe";
    let currentPlanId = null;
    let currentSuggest = null;

    try {
      const response = await fetch(API_BASE + "/api/auto/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeader(),
        },
        body: JSON.stringify({ message: text, session_id: sessionId.current }),
      });

      if (!response.ok) {
        const err = await response.text();
        updateStreamingMessage(streamId, {
          text: `Error: ${err || response.statusText}`,
          _streaming: false,
        });
        setStreaming(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          let data = line;
          if (line.startsWith("data: ")) data = line.slice(6);
          if (data === "[DONE]") continue;

          let evt;
          try { evt = JSON.parse(data); } catch { continue; }

          if (!evt) continue;

          if (evt.type === "intent") {
            const action = evt.action || evt.intent;
            if (action) setThinkPhase("thinking");
            if (evt.risk) currentRisk = evt.risk;

          } else if (evt.type === "risk") {
            currentRisk = evt.level || evt.risk || "safe";
            updateStreamingMessage(streamId, { risk: currentRisk });

          } else if (evt.type === "plan_id") {
            currentPlanId = evt.plan_id || evt.id;
            updateStreamingMessage(streamId, { planId: currentPlanId });

          } else if (evt.type === "sources") {
            aria?.onSources?.(evt.sources || evt.data || []);

          } else if (evt.type === "neural") {
            aria?.onSignals?.(evt.signals || evt.data || {});

          } else if (evt.type === "agent") {
            const agentName = evt.name || evt.agent || "agent";
            aria?.onAgentFire?.(agentName);

          } else if (evt.type === "thinking" || evt.type === "searching" || evt.type === "status") {
            // Status/progress updates — never accumulate into the answer bubble
            setThinkPhase(evt.type);

          } else if (evt.type === "replace") {
            // Orchestrator signalling a better answer is coming — clear current text
            accText = evt.text || "";
            updateStreamingMessage(streamId, { text: accText, risk: currentRisk });

          } else if (evt.type === "token" || evt.type === "text" || evt.type === "delta"
                     || evt.type === "chunk" || evt.delta || evt.chunk
                     || (evt.text && evt.type !== "thinking" && evt.type !== "status"
                         && evt.type !== "replace" && evt.type !== "done")) {
            const chunk = evt.text || evt.delta || evt.chunk || "";
            if (chunk) {
              accText += chunk;
              setThinkPhase("generating");
              updateStreamingMessage(streamId, { text: accText, risk: currentRisk });
            }

          } else if (evt.type === "done") {
            if (evt.text) { accText = evt.text; }
            if (evt.suggest) currentSuggest = evt.suggest;
            if (evt.risk)   currentRisk = evt.risk;
            if (evt.plan_id) currentPlanId = evt.plan_id;
            // finalize
            updateStreamingMessage(streamId, {
              text: accText || evt.message || "Done.",
              risk: currentRisk,
              planId: currentPlanId,
              suggest: currentSuggest,
              _streaming: false,
            });
            setStreaming(false);
            aria?.onTaskUpdate?.({ id: sessionId.current + "-task", status: "done", title: text.slice(0, 40) });
          }
        }
      }

      // Ensure finalized even if no "done" event
      if (streaming) {
        updateStreamingMessage(streamId, {
          text: accText || "…",
          risk: currentRisk,
          planId: currentPlanId,
          suggest: currentSuggest,
          _streaming: false,
        });
      }
    } catch (err) {
      updateStreamingMessage(streamId, {
        text: `Connection error: ${err.message}`,
        _streaming: false,
      });
    }

    setStreaming(false);
  }, [streaming, addMessage, updateStreamingMessage, aria]);

  const handleConfirm = useCallback(async (planId) => {
    if (!planId) return;
    try {
      await fetch(API_BASE + "/api/auto/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeader() },
        body: JSON.stringify({ plan_id: planId, session_id: sessionId.current }),
      });
    } catch { /* ignore */ }
    // Remove planId from message to hide the confirm box
    setMessages(prev => prev.map(m => m.planId === planId ? { ...m, planId: null } : m));
  }, []);

  const isEmpty = messages.length === 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "var(--bg0)" }}>
      {/* Messages area */}
      <div className="chat-messages">
        {isEmpty ? (
          <div className="empty-state">
            {/* ARIA logo */}
            <div style={{
              width: 72, height: 72, borderRadius: "50%",
              border: "2px solid var(--cyan)",
              boxShadow: "0 0 30px var(--glow)",
              display: "flex", alignItems: "center", justifyContent: "center",
              animation: "ringPulse 3s ease-in-out infinite",
            }}>
              <span style={{ fontSize: 24, fontWeight: 800, color: "var(--cyan)", letterSpacing: 1 }}>A</span>
            </div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--text)", marginBottom: 6 }}>
                ARIA is ready
              </div>
              <div style={{ fontSize: 13, color: "var(--text3)" }}>
                Your autonomous AI assistant. Ask anything.
              </div>
            </div>
            {/* Quick chips */}
            <div style={{
              display: "flex", flexWrap: "wrap",
              gap: 8, justifyContent: "center",
              maxWidth: 500,
            }}>
              {QUICK_CHIPS.map(chip => (
                <button
                  key={chip}
                  className="quick-chip"
                  onClick={() => handleSend(chip)}
                >
                  {chip}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map(msg => (
              msg._streaming && !msg.text ? (
                <ThinkingBubble key={msg.id} phase={thinkPhase} />
              ) : (
                <MessageBubble
                  key={msg.id}
                  role={msg.role}
                  text={msg.text}
                  risk={msg.risk}
                  planId={msg.planId}
                  suggest={msg.suggest}
                  time={msg.time}
                  onConfirm={handleConfirm}
                  onSuggest={(s) => handleSend(s)}
                />
              )
            ))}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      {/* Input */}
      <InputBar onSend={handleSend} disabled={streaming} />
    </div>
  );
}
