import { Routes, Route } from "react-router-dom";
import { API_BASE } from "./lib/api";
import { useState, useEffect, createContext, useContext, useCallback } from "react";
import { NavLink } from "react-router-dom";
import TopBar from "./components/TopBar";
import LeftPanel from "./components/LeftPanel";
import RightPanel from "./components/RightPanel";
import ToastContainer from "./components/Toast";
import Chat from "./pages/Chat";
import Voice from "./pages/Voice";
import Login, { TokenStore } from "./pages/Login";
import Onboarding from "./pages/Onboarding";
import { Upload, Actions, Search, Trends, Analytics, Settings } from "./pages/OtherPages";
import { useStore } from "./store";

// ── ARIA Context ──────────────────────────────────────────────────────────────
export const ARIAContext = createContext(null);
export const useARIA = () => useContext(ARIAContext);

const getAuthHeader = () => ({
  "Authorization": "Bearer " + (window.__ariaToken || localStorage.getItem("aria_token") || ""),
});

// ── Crawl page (inline) ───────────────────────────────────────────────────────
function Crawl() {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "var(--bg1)" }}>
      <div style={{ padding: "12px 18px", borderBottom: "1px solid var(--border)", background: "var(--bg2)" }}>
        <div style={{ fontSize: 14, fontWeight: 500, color: "var(--cyan)" }}>Web Crawler</div>
      </div>
      <div style={{ flex: 1, padding: 18 }}>
        <p style={{ fontSize: 13, color: "var(--text2)", lineHeight: 1.7 }}>
          Use the crawler from the browser UI at{" "}
          <a href="http://localhost:8000" target="_blank" rel="noopener noreferrer">localhost:8000</a>
          {" "}for live progress streaming. Full Playwright-based deep crawler with stealth mode.
        </p>
      </div>
    </div>
  );
}

// ── Nav strip ─────────────────────────────────────────────────────────────────
const NAV = [
  { to: "/",          icon: "💬", title: "Chat"      },
  { to: "/voice",     icon: "🎙️", title: "Voice"     },
  { to: "/upload",    icon: "📤", title: "Upload"    },
  { to: "/crawl",     icon: "🕷️", title: "Crawl"     },
  { to: "/search",    icon: "🌐", title: "Search"    },
  { to: "/trends",    icon: "📈", title: "Trends"    },
  { to: "/actions",   icon: "⚡", title: "Actions"   },
  { to: "/analytics", icon: "📊", title: "Analytics" },
  { to: "/settings",  icon: "⚙️", title: "Settings"  },
];

function NavStrip({ onLogout }) {
  return (
    <div className="nav-strip">
      {NAV.map(({ to, icon, title }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          title={title}
          className={({ isActive }) => isActive ? "active" : ""}
        >
          <span style={{ fontSize: 16 }}>{icon}</span>
        </NavLink>
      ))}
      <div style={{ flex: 1 }} />
      <button
        onClick={onLogout}
        title="Logout"
        style={{
          width: 36, height: 36, borderRadius: "var(--r)",
          border: "1px solid var(--border)", background: "none",
          color: "var(--text3)", fontSize: 14, cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "center",
          transition: "all .12s",
        }}
      >
        ⏏
      </button>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const { checkServer, serverOk } = useStore();
  const [authed, setAuthed]       = useState(false);
  const [token, setToken]         = useState(null);
  const [onboarded, setOnboarded] = useState(
    () => !!localStorage.getItem("aria_onboarded")
  );

  // HUD & panel state
  const [mode, setMode]         = useState("auto");
  const [agentFeed, setAgentFeed] = useState([]);
  const [tasks, setTasks]       = useState([]);
  const [signals, setSignals]   = useState({});
  const [sources, setSources]   = useState([]);

  // On mount — check saved token
  useEffect(() => {
    const t = TokenStore.get();
    if (t) { setToken(t); setAuthed(true); }
  }, []);

  const handleAuth = (t) => {
    setToken(t);
    setAuthed(true);
    TokenStore.set(t);
  };

  const handleLogout = useCallback(() => {
    TokenStore.clear();
    setToken(null);
    setAuthed(false);
  }, []);

  // Expose token globally
  useEffect(() => {
    if (token) {
      useStore.setState({ token });
      window.__ariaToken = token;
    }
  }, [token]);

  // Server health checks
  useEffect(() => {
    if (!authed) return;
    checkServer();
    const t = setInterval(checkServer, 15000);
    return () => clearInterval(t);
  }, [authed, checkServer]);

  // Notification SSE
  useEffect(() => {
    if (!authed) return;
    let es;
    const savedToken = localStorage.getItem("aria_token") || "";
    const connect = () => {
      es = new EventSource(`/api/notifications/stream?token=${encodeURIComponent(savedToken)}`);
      es.onmessage = async (event) => {
        try {
          const notif = JSON.parse(event.data);
          if (notif.type === "ping") return;
          const { addNotification } = useStore.getState();
          addNotification(notif.body || notif.title, notif.type || "info");
          try {
            const { isPermissionGranted, requestPermission, sendNotification }
              = await import("@tauri-apps/api/notification");
            let ok = await isPermissionGranted();
            if (!ok) { const perm = await requestPermission(); ok = perm === "granted"; }
            if (ok) { sendNotification({ title: `ARIA — ${notif.title}`, body: notif.body }); return; }
          } catch { /* not Tauri */ }
          if ("Notification" in window) {
            if (Notification.permission === "granted") {
              new Notification(`ARIA — ${notif.title}`, { body: notif.body, icon: "/icon-192.png", tag: notif.id });
            } else if (Notification.permission !== "denied") {
              Notification.requestPermission().then(p => {
                if (p === "granted") new Notification(`ARIA — ${notif.title}`, { body: notif.body });
              });
            }
          }
        } catch { /* ignore */ }
      };
      es.onerror = () => { es?.close(); setTimeout(connect, 5000); };
    };
    connect();
    return () => es?.close();
  }, [authed]);

  // Poll tasks every 3s
  useEffect(() => {
    if (!authed) return;
    let lastJson = "";
    const fetchTasks = async () => {
      try {
        const r = await fetch(API_BASE + "/api/tasks", { headers: getAuthHeader() });
        if (r.ok) {
          const d = await r.json();
          const next = Array.isArray(d) ? d : d.tasks || [];
          // Only call setTasks if data actually changed — avoids focus-killing re-renders
          const json = JSON.stringify(next);
          if (json !== lastJson) { lastJson = json; setTasks(next); }
        }
      } catch { /* ignore */ }
    };
    fetchTasks();
    const id = setInterval(fetchTasks, 3000);
    return () => clearInterval(id);
  }, [authed]);

  // ARIA context callbacks
  const updateAgentFeed = useCallback((name) => {
    const entry = { name, active: true, time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false }) };
    setAgentFeed(prev => [entry, ...prev].slice(0, 12));
  }, []);

  const updateTaskList = useCallback((taskData) => {
    if (Array.isArray(taskData)) {
      setTasks(taskData);
    } else if (taskData && taskData.id) {
      setTasks(prev => {
        const idx = prev.findIndex(t => t.id === taskData.id);
        if (idx >= 0) {
          const next = [...prev];
          next[idx] = { ...next[idx], ...taskData };
          return next;
        }
        return [taskData, ...prev];
      });
    }
  }, []);

  const updateSignals = useCallback((newSignals) => {
    setSignals(prev => ({ ...prev, ...newSignals }));
  }, []);

  const updateSources = useCallback((newSources) => {
    setSources(Array.isArray(newSources) ? newSources : []);
  }, []);

  const handleReset = useCallback(() => {
    setAgentFeed([]);
    setTasks([]);
    setSignals({});
    setSources([]);
  }, []);

  const ariaContextValue = {
    mode,
    onAgentFire:  updateAgentFeed,
    onTaskUpdate: updateTaskList,
    onSignals:    updateSignals,
    onSources:    updateSources,
  };

  if (!authed) return <Login onAuth={handleAuth} />;
  if (!onboarded) return <Onboarding onComplete={() => setOnboarded(true)} />;

  return (
    <ARIAContext.Provider value={ariaContextValue}>
      {/* HUD corners */}
      <div className="hud-corner tl" />
      <div className="hud-corner tr" />
      <div className="hud-corner bl" />
      <div className="hud-corner br" />

      <div className="app-shell">
        {/* Top bar */}
        <TopBar
          mode={mode}
          onModeChange={setMode}
          onReset={handleReset}
          serverOk={serverOk}
          llmOk={serverOk}
        />

        {/* Body row */}
        <div className="app-body">
          {/* Left nav strip */}
          <NavStrip onLogout={handleLogout} />

          {/* Left panel */}
          <LeftPanel agentFeed={agentFeed} />

          {/* Main content */}
          <div className="app-main">
            <Routes>
              <Route path="/"          element={<Chat />} />
              <Route path="/voice"     element={<Voice />} />
              <Route path="/upload"    element={<Upload />} />
              <Route path="/crawl"     element={<Crawl />} />
              <Route path="/search"    element={<Search />} />
              <Route path="/trends"    element={<Trends />} />
              <Route path="/actions"   element={<Actions />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/settings"  element={<Settings onLogout={handleLogout} />} />
            </Routes>
          </div>

          {/* Right panel */}
          <RightPanel tasks={tasks} signals={signals} sources={sources} />
        </div>
      </div>

      <ToastContainer />
    </ARIAContext.Provider>
  );
}
