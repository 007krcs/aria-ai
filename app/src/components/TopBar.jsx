import { useState, useEffect } from "react";

export default function TopBar({ mode, onModeChange, onReset, serverOk, llmOk }) {
  const [clock, setClock] = useState("");

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      const h = String(now.getHours()).padStart(2, "0");
      const m = String(now.getMinutes()).padStart(2, "0");
      const s = String(now.getSeconds()).padStart(2, "0");
      setClock(`${h}:${m}:${s}`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="top-bar">
      {/* Ring logo */}
      <div className="aria-ring">
        <span style={{ fontSize: 11, fontWeight: 800, color: "var(--cyan)", letterSpacing: -0.5 }}>A</span>
      </div>

      {/* Title */}
      <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.1, marginRight: 4 }}>
        <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: 2, color: "var(--cyan)" }}>ARIA</span>
        <span style={{ fontSize: 9, color: "var(--text3)", letterSpacing: 1 }}>v3.0</span>
      </div>

      {/* Status chips */}
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <div className="status-chip">
          <div className="status-dot" style={{ background: llmOk ? "var(--safe)" : "var(--danger)" }} />
          <span style={{ color: llmOk ? "var(--safe)" : "var(--danger)" }}>LLM</span>
        </div>
        <div className="status-chip">
          <div className="status-dot" style={{ background: "var(--safe)" }} />
          <span style={{ color: "var(--safe)" }}>Neural</span>
        </div>
        <div className="status-chip">
          <div
            className="status-dot"
            style={{ background: mode === "auto" ? "var(--cyan)" : "var(--caution)" }}
          />
          <span style={{ color: mode === "auto" ? "var(--cyan)" : "var(--caution)" }}>
            {mode === "auto" ? "AUTO" : "SAFE"}
          </span>
        </div>
      </div>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Clock */}
      <div style={{
        fontFamily: "'SF Mono','Fira Code',Consolas,monospace",
        fontSize: 13, letterSpacing: 1,
        color: "var(--text2)",
        minWidth: 70, textAlign: "right",
      }}>
        {clock}
      </div>

      {/* Mode buttons */}
      <div style={{ display: "flex", gap: 4 }}>
        <button
          className={`mode-btn${mode === "auto" ? " active" : ""}`}
          onClick={() => onModeChange("auto")}
        >
          AUTO
        </button>
        <button
          className={`mode-btn${mode === "safe" ? " active" : ""}`}
          onClick={() => onModeChange("safe")}
        >
          SAFE
        </button>
        <button
          className="mode-btn"
          onClick={onReset}
          style={{ borderColor: "var(--danger)", color: "var(--danger)" }}
        >
          RESET
        </button>
      </div>
    </div>
  );
}
