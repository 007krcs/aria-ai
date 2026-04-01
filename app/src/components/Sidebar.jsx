import { NavLink } from "react-router-dom";
import { useStore } from "../store";

const NAV = [
  { to: "/",         icon: "💬", label: "Chat" },
  { to: "/voice",    icon: "🎙️", label: "Voice" },
  { to: "/upload",   icon: "📤", label: "Upload" },
  { to: "/crawl",    icon: "🕷️", label: "Crawl" },
  { to: "/search",   icon: "🌐", label: "Search" },
  { to: "/trends",   icon: "📈", label: "Trends" },
  { to: "/actions",  icon: "⚡", label: "Actions" },
  { to: "/analytics",icon: "📊", label: "Analytics" },
  { to: "/settings", icon: "⚙️", label: "Settings" },
];

export default function Sidebar() {
  const { serverOk, serverInfo } = useStore();

  return (
    <aside style={{
      width: 200, flexShrink: 0,
      background: "var(--bg2)",
      borderRight: "1px solid var(--border)",
      display: "flex", flexDirection: "column",
      height: "100%",
    }}>
      {/* Logo */}
      <div style={{ padding: "18px 16px 12px" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 0 }}>
          <span style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)" }}>AR</span>
          <span style={{ fontSize: 18, fontWeight: 700, color: "var(--green)" }}>IA</span>
        </div>
        <div style={{ fontSize: 10, color: "var(--text3)", marginTop: 2 }}>
          Personal AI Assistant
        </div>
      </div>

      <div style={{ height: 1, background: "var(--border)" }} />

      {/* Nav */}
      <nav style={{ padding: "8px 6px", flex: 1, overflowY: "auto" }}>
        {NAV.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            style={({ isActive }) => ({
              display: "flex", alignItems: "center", gap: 10,
              padding: "8px 10px", borderRadius: 8,
              fontSize: 13, fontWeight: isActive ? 500 : 400,
              color: isActive ? "var(--accent)" : "var(--text2)",
              background: isActive ? "rgba(124,106,247,.12)" : "transparent",
              textDecoration: "none",
              marginBottom: 2,
              transition: "all .12s",
            })}
          >
            <span style={{ fontSize: 15, width: 20, textAlign: "center" }}>{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Status footer */}
      <div style={{ height: 1, background: "var(--border)" }} />
      <div style={{
        padding: "8px 14px 10px",
        display: "flex", alignItems: "center", gap: 6,
      }}>
        <div style={{
          width: 7, height: 7, borderRadius: "50%",
          background: serverOk ? "var(--green)" : "var(--red)",
          flexShrink: 0,
        }} />
        <span style={{ fontSize: 11, color: "var(--text3)", flex:1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {serverOk ? `${serverInfo.model || "ready"}` : "Server offline"}
        </span>
        <button
          onClick={() => {
            const { settings, updateSettings } = useStore.getState();
            const next = settings.theme === "dark" ? "light" : "dark";
            updateSettings({ theme: next });
            localStorage.setItem("aria_theme", next);
          }}
          title="Toggle theme"
          style={{
            background:"none", border:"none", cursor:"pointer",
            fontSize:13, color:"var(--text3)", padding:"2px 4px",
            flexShrink:0,
          }}
        >
          {useStore.getState().settings?.theme === "light" ? "🌙" : "☀️"}
        </button>
      </div>
    </aside>
  );
}
