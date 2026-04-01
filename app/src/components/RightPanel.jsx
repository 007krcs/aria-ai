import { useState } from "react";

const STATUS_COLORS = {
  running: "var(--cyan)",
  done:    "var(--safe)",
  error:   "var(--danger)",
  queued:  "var(--caution)",
};

function StatusPill({ status }) {
  const color = STATUS_COLORS[status] || "var(--text3)";
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: "2px 7px",
      borderRadius: 10, border: `1px solid ${color}`,
      color, background: `${color}18`,
      letterSpacing: 0.5, textTransform: "uppercase",
      flexShrink: 0,
    }}>
      {status}
    </span>
  );
}

export default function RightPanel({ tasks = [], signals = {}, sources = [] }) {
  const [tab, setTab] = useState("tasks");

  const signalEntries = Object.entries(signals);

  return (
    <div className="right-panel">
      {/* Tab row */}
      <div className="tab-row">
        {["tasks", "neural", "sources"].map(t => (
          <button
            key={t}
            className={`tab-btn${tab === t ? " active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "tasks" ? "Tasks" : t === "neural" ? "Neural" : "Sources"}
          </button>
        ))}
      </div>

      {/* Tasks tab */}
      {tab === "tasks" && (
        <div style={{ flex: 1, overflowY: "auto" }}>
          {tasks.length === 0 ? (
            <div style={{ padding: 16, fontSize: 11, color: "var(--text3)", textAlign: "center", marginTop: 20, fontStyle: "italic" }}>
              No active tasks
            </div>
          ) : (
            tasks.map((task, i) => (
              <div key={task.id || i} style={{
                padding: "10px 12px",
                borderBottom: "1px solid var(--border)",
              }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 4 }}>
                  <span style={{
                    fontSize: 12, color: "var(--text)", flex: 1,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {task.title || task.name || "Untitled"}
                  </span>
                  <StatusPill status={task.status || "queued"} />
                </div>
                {task.progress != null && (
                  <div className="progress-outer">
                    <div className="progress-inner" style={{ width: `${Math.min(task.progress, 100)}%` }} />
                  </div>
                )}
                {task.detail && (
                  <div style={{ fontSize: 10, color: "var(--text3)", marginTop: 4 }}>
                    {task.detail}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Neural tab */}
      {tab === "neural" && (
        <div style={{ flex: 1, overflowY: "auto" }}>
          {signalEntries.length === 0 ? (
            <div style={{ padding: 16, fontSize: 11, color: "var(--text3)", textAlign: "center", marginTop: 20, fontStyle: "italic" }}>
              Awaiting neural signals…
            </div>
          ) : (
            signalEntries.map(([agent, conf], i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "8px 12px", borderBottom: "1px solid var(--border)",
              }}>
                <span style={{ fontSize: 11, color: "var(--cyan)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {agent}
                </span>
                <span style={{ fontSize: 11, color: "var(--safe)", fontFamily: "monospace", flexShrink: 0 }}>
                  {typeof conf === "number" ? `${Math.round(conf * 100)}%` : conf}
                </span>
              </div>
            ))
          )}
        </div>
      )}

      {/* Sources tab */}
      {tab === "sources" && (
        <div style={{ flex: 1, overflowY: "auto" }}>
          {sources.length === 0 ? (
            <div style={{ padding: 16, fontSize: 11, color: "var(--text3)", textAlign: "center", marginTop: 20, fontStyle: "italic" }}>
              No sources yet
            </div>
          ) : (
            sources.map((src, i) => (
              <div key={i} style={{
                padding: "8px 12px", borderBottom: "1px solid var(--border)",
                display: "flex", alignItems: "center", gap: 8,
              }}>
                <a
                  href={src.url || src}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    fontSize: 11, color: "var(--cyan)", flex: 1,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}
                >
                  {src.url || src}
                </a>
                <span style={{
                  fontSize: 9, padding: "1px 6px", borderRadius: 8,
                  border: "1px solid var(--safe)", color: "var(--safe)",
                  background: "rgba(0,255,136,.08)", flexShrink: 0,
                }}>
                  ✓ verified
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
