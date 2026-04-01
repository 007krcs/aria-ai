import { create } from "zustand";

// Apply theme to :root CSS variables
function applyTheme(theme) {
  const dark = theme === "dark" ||
    (theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  const root = document.documentElement;
  if (dark) {
    root.style.setProperty("--bg",    "#0f0f11");
    root.style.setProperty("--bg2",   "#18181c");
    root.style.setProperty("--bg3",   "#222228");
    root.style.setProperty("--bg4",   "#2a2a32");
    root.style.setProperty("--border","#2e2e38");
    root.style.setProperty("--text",  "#e8e8f0");
    root.style.setProperty("--text2", "#9898b0");
    root.style.setProperty("--text3", "#5a5a72");
  } else {
    root.style.setProperty("--bg",    "#f5f5f7");
    root.style.setProperty("--bg2",   "#ffffff");
    root.style.setProperty("--bg3",   "#f0f0f3");
    root.style.setProperty("--bg4",   "#e8e8ed");
    root.style.setProperty("--border","#d0d0dc");
    root.style.setProperty("--text",  "#1a1a2e");
    root.style.setProperty("--text2", "#5a5a72");
    root.style.setProperty("--text3", "#9898b0");
  }
}

// Apply saved theme on load
const _savedTheme = localStorage.getItem("aria_theme") || "dark";
applyTheme(_savedTheme);

// Use relative paths so Vite proxy handles routing — eliminates CORS entirely.
// When built for production, set VITE_API_URL to the server address.
const API = import.meta.env.VITE_API_URL || "";

// Get token for every request
const getToken = () => window.__ariaToken || localStorage.getItem("aria_token") || "";

const authHeader = () => {
  const t = getToken();
  return t ? { "Authorization": `Bearer ${t}` } : {};
};

export const useStore = create((set, get) => ({
  serverOk:   false,
  serverInfo: {},
  token:      null,

  checkServer: async () => {
    try {
      const r = await fetch(`${API}/api/health`, {
        headers: authHeader(),
        signal: AbortSignal.timeout(2500),
      });
      if (r.ok) set({ serverOk: true, serverInfo: await r.json() });
      else set({ serverOk: false });
    } catch { set({ serverOk: false }); }
  },

  notifications: [],
  addNotification: (msg, type = "info") => {
    const id = Date.now();
    set(s => ({ notifications: [...s.notifications, { id, msg, type }] }));
    setTimeout(() => set(s => ({
      notifications: s.notifications.filter(n => n.id !== id),
    })), 4500);
  },

  messages: [],
  addMessage:    (msg) => set(s => ({ messages: [...s.messages, { id: Date.now() + Math.random(), ...msg }] })),
  clearMessages: ()    => set({ messages: [] }),

  selectedVoice: "en-IN-NeerjaNeural",
  setVoice:      (v) => set({ selectedVoice: v }),

  settings: {
    city: "", language: "en", wakeWord: true, autoSpeak: true,
    theme: "dark",  // "dark" | "light" | "system"
  },
  updateSettings: (patch) => {
    set(s => ({ settings: { ...s.settings, ...patch } }));
    // Apply theme immediately
    if (patch.theme) applyTheme(patch.theme);
  },
}));

export const api = {
  get: (path, opts = {}) =>
    fetch(`${API}${path}`, {
      ...opts,
      headers: { ...authHeader(), ...(opts.headers || {}) },
    }),

  post: (path, body, opts = {}) =>
    fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader(), ...(opts.headers || {}) },
      body: JSON.stringify(body),
      ...opts,
    }),

  postForm: (path, formData) =>
    fetch(`${API}${path}`, {
      method: "POST",
      headers: { ...authHeader() },
      body: formData,
    }),

  ws: (path) => {
    const t = getToken();
    const url = `ws://localhost:8000${path}${t ? `?token=${encodeURIComponent(t)}` : ""}`;
    return new WebSocket(url);
  },
};

export { API };
