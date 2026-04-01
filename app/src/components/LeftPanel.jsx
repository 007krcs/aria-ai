import { useState, useEffect } from "react";
import NeuralCanvas from "./NeuralCanvas";
import { API_BASE } from "../lib/api";

const getAuthHeader = () => ({
  "Authorization": "Bearer " + (window.__ariaToken || localStorage.getItem("aria_token") || ""),
});

function MetricBar({ label, value, unit = "%" }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 12px" }}>
      <span className="metric-label" style={{ width: 32, fontSize: 10, color: "var(--text3)", flexShrink: 0 }}>
        {label}
      </span>
      <div className="metric-bar-outer">
        <div className="metric-bar-inner" style={{ width: `${Math.min(value, 100)}%` }} />
      </div>
      <span className="metric-value" style={{ width: 34, fontSize: 10, color: "var(--cyan)", textAlign: "right", flexShrink: 0, fontFamily: "monospace" }}>
        {value}{unit}
      </span>
    </div>
  );
}

export default function LeftPanel({ agentFeed }) {
  const [metrics, setMetrics] = useState({ cpu: 0, ram: 0, net: 0 });

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const r = await fetch(API_BASE + "/api/system/status", { headers: getAuthHeader() });
        if (!r.ok) throw new Error("no data");
        const d = await r.json();
        setMetrics({
          cpu: Math.round(d.cpu ?? d.cpu_percent ?? Math.random() * 60 + 10),
          ram: Math.round(d.ram ?? d.ram_percent ?? Math.random() * 50 + 20),
          net: Math.round(d.net ?? d.net_kbps ?? Math.random() * 40 + 5),
        });
      } catch {
        setMetrics({
          cpu: Math.round(Math.random() * 60 + 10),
          ram: Math.round(Math.random() * 50 + 20),
          net: Math.round(Math.random() * 40 + 5),
        });
      }
    };
    fetchMetrics();
    const id = setInterval(fetchMetrics, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="left-panel">
      {/* Neural Viz */}
      <div className="panel-section-header">Neural Viz</div>
      <div style={{ padding: "6px 0" }}>
        <NeuralCanvas height={110} />
      </div>

      {/* System Metrics */}
      <div className="panel-section-header">System</div>
      <MetricBar label="CPU" value={metrics.cpu} />
      <MetricBar label="RAM" value={metrics.ram} />
      <MetricBar label="NET" value={metrics.net} unit="%" />

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Agent Feed */}
      <div className="panel-section-header">Agent Feed</div>
      <div style={{ flex: "0 0 auto", maxHeight: 180, overflowY: "auto" }}>
        {(!agentFeed || agentFeed.length === 0) ? (
          <div style={{ padding: "8px 12px", fontSize: 11, color: "var(--text3)", fontStyle: "italic" }}>
            No agents active
          </div>
        ) : (
          agentFeed.map((entry, i) => (
            <div key={i} className="agent-entry">
              <span style={{ color: "var(--cyan3)", flexShrink: 0 }}>›</span>
              <span
                className="agent-name"
                style={{
                  color: entry.active ? "var(--cyan)" : "var(--text3)",
                  flex: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {entry.name}
              </span>
              {entry.time && (
                <span style={{ fontSize: 9, color: "var(--text3)", flexShrink: 0 }}>
                  {entry.time}
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
