import { useState } from "react";
import { useStore, api } from "../store";

const STEPS = [
  { id: "name",     title: "Tell me about yourself",   icon: "👤" },
  { id: "voice",    title: "Pick your voice",           icon: "🎙️" },
  { id: "android",  title: "Connect your phone",        icon: "📱" },
  { id: "alerts",   title: "What should I watch for?",  icon: "📈" },
  { id: "done",     title: "You're all set",            icon: "✅" },
];

const VOICES = [
  { id: "en-IN-NeerjaNeural",  label: "Neerja",  lang: "English (Indian)", gender: "♀" },
  { id: "en-IN-PrabhatNeural", label: "Prabhat", lang: "English (Indian)", gender: "♂" },
  { id: "hi-IN-SwaraNeural",   label: "Swara",   lang: "Hindi",            gender: "♀" },
  { id: "hi-IN-MadhurNeural",  label: "Madhur",  lang: "Hindi",            gender: "♂" },
  { id: "en-US-AriaNeural",    label: "Aria",    lang: "English (US)",     gender: "♀" },
];

export default function Onboarding({ onComplete }) {
  const { updateSettings, setVoice, addNotification } = useStore();
  const [step, setStep]     = useState(0);
  const [data, setData]     = useState({
    name: "", city: "", language: "en",
    voice: "en-IN-NeerjaNeural",
    androidIp: "", androidPort: "5555",
    watchStocks: "", watchNews: "",
  });
  const [loading, setLoad]  = useState(false);

  const update = (k, v) => setData(d => ({ ...d, [k]: v }));

  const canNext = () => {
    if (step === 0) return data.name.trim().length > 0;
    return true;
  };

  const handleNext = async () => {
    if (step === STEPS.length - 1) { finish(); return; }

    // Save step data
    if (step === 0) {
      updateSettings({ city: data.city, ownerName: data.name });
    }
    if (step === 1) {
      setVoice(data.voice);
      await api.post(`/api/voice/set-voice?voice=${encodeURIComponent(data.voice)}`, {});
    }
    if (step === 2 && data.androidIp.trim()) {
      setLoad(true);
      try {
        await api.post("/api/device/android/connect", {
          ip: data.androidIp, port: parseInt(data.androidPort) || 5555
        });
      } catch { /* non-critical */ }
      setLoad(false);
    }
    if (step === 3) {
      // Set up proactive watches
      const stocks = data.watchStocks.split(",").map(s => s.trim()).filter(Boolean);
      for (const s of stocks) {
        const parts = s.split("@");
        const sym = parts[0].trim().toUpperCase();
        const target = parseFloat(parts[1]) || 0;
        if (sym) {
          await api.post("/api/proactive/price-target", {
            symbol: sym, target: target || 100, direction: "above"
          }).catch(() => {});
        }
      }
      const topics = data.watchNews.split(",").map(t => t.trim()).filter(Boolean);
      for (const t of topics) {
        await api.post("/api/proactive/news-topic", { topic: t }).catch(() => {});
      }
    }

    setStep(s => s + 1);
  };

  const finish = () => {
    localStorage.setItem("aria_onboarded", "1");
    addNotification(`Welcome, ${data.name}! ARIA is ready.`, "success");
    onComplete();
  };

  const skip = () => {
    localStorage.setItem("aria_onboarded", "1");
    onComplete();
  };

  const current = STEPS[step];

  return (
    <div style={{
      height: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      background: "var(--bg)", padding: 24,
    }}>
      {/* Progress dots */}
      <div style={{ display: "flex", gap: 6, marginBottom: 32 }}>
        {STEPS.map((s, i) => (
          <div key={s.id} style={{
            width: i === step ? 24 : 8, height: 8,
            borderRadius: 4, transition: "all .3s",
            background: i < step ? "var(--green)"
              : i === step ? "var(--accent)" : "var(--border)",
          }} />
        ))}
      </div>

      {/* Card */}
      <div style={{
        background: "var(--bg2)", border: "1px solid var(--border)",
        borderRadius: 16, padding: "28px 32px", width: "100%", maxWidth: 420,
      }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>{current.icon}</div>
        <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 20 }}>
          {current.title}
        </div>

        {/* Step content */}
        {step === 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <Input placeholder="Your name *" value={data.name}
              onChange={v => update("name", v)} autoFocus />
            <Input placeholder="Your city (for weather)" value={data.city}
              onChange={v => update("city", v)} />
            <select value={data.language}
              onChange={e => update("language", e.target.value)}
              style={selectStyle}>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="auto">Auto-detect</option>
            </select>
          </div>
        )}

        {step === 1 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {VOICES.map(v => (
              <div key={v.id} onClick={() => update("voice", v.id)} style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "10px 14px", borderRadius: 10, cursor: "pointer",
                border: `1px solid ${data.voice === v.id ? "var(--accent)" : "var(--border)"}`,
                background: data.voice === v.id ? "rgba(124,106,247,.08)" : "var(--bg3)",
                transition: "all .15s",
              }}>
                <div style={{
                  width: 32, height: 32, borderRadius: "50%",
                  background: data.voice === v.id ? "rgba(124,106,247,.2)" : "var(--bg4)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 14, color: "var(--accent)", fontWeight: 600,
                }}>{v.gender}</div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text)" }}>{v.label}</div>
                  <div style={{ fontSize: 11, color: "var(--text3)" }}>{v.lang}</div>
                </div>
                {data.voice === v.id && (
                  <div style={{ marginLeft: "auto", color: "var(--accent)", fontSize: 14 }}>✓</div>
                )}
              </div>
            ))}
          </div>
        )}

        {step === 2 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <p style={{ fontSize: 12, color: "var(--text2)", lineHeight: 1.7, marginBottom: 4 }}>
              Enable on your phone:<br />
              <strong>Settings → Developer Options → Wireless Debugging</strong><br />
              Note the IP address and port shown on screen.
            </p>
            <Input placeholder="Phone IP (e.g. 192.168.1.5)" value={data.androidIp}
              onChange={v => update("androidIp", v)} />
            <Input placeholder="Port (default: 5555)" value={data.androidPort}
              onChange={v => update("androidPort", v)} />
            <p style={{ fontSize: 11, color: "var(--text3)" }}>
              You can also do this later from Settings. This step is optional.
            </p>
          </div>
        )}

        {step === 3 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <p style={{ fontSize: 12, color: "var(--text2)", lineHeight: 1.6 }}>
              ARIA will watch these and alert you without being asked.
            </p>
            <Input
              placeholder="Stocks to watch (e.g. AAPL@200, TCS@4000)"
              value={data.watchStocks}
              onChange={v => update("watchStocks", v)}
            />
            <p style={{ fontSize: 11, color: "var(--text3)" }}>
              Format: SYMBOL@target_price — alerts when it crosses.
            </p>
            <Input
              placeholder="News topics (e.g. SpaceX, AI regulation, cricket)"
              value={data.watchNews}
              onChange={v => update("watchNews", v)}
            />
            <p style={{ fontSize: 11, color: "var(--text3)" }}>
              Comma-separated. ARIA checks hourly and notifies on new content.
            </p>
          </div>
        )}

        {step === 4 && (
          <div style={{ textAlign: "center" }}>
            <p style={{ fontSize: 13, color: "var(--text2)", lineHeight: 1.8, marginBottom: 8 }}>
              <strong style={{ color: "var(--text)" }}>Welcome, {data.name}.</strong><br />
              ARIA is set up and ready. Voice, search, actions,<br />
              research, and device control are all active.
            </p>
            <div style={{
              display: "flex", flexDirection: "column", gap: 6,
              fontSize: 12, color: "var(--text3)", marginTop: 12,
            }}>
              {[
                "Say 'Hey ARIA' to activate voice",
                "Hold mic button in Voice tab to speak",
                "Add documents in Upload tab",
                "Monitor your analytics in Analytics tab",
              ].map(tip => (
                <div key={tip} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span style={{ color: "var(--green)" }}>→</span> {tip}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Nav buttons */}
      <div style={{ display: "flex", gap: 10, marginTop: 20, width: "100%", maxWidth: 420 }}>
        {step < STEPS.length - 1 && (
          <button onClick={skip} style={{
            flex: 1, padding: "10px", borderRadius: 10,
            border: "1px solid var(--border)", background: "transparent",
            color: "var(--text3)", cursor: "pointer", fontSize: 13,
          }}>
            Skip setup
          </button>
        )}
        <button
          onClick={handleNext}
          disabled={!canNext() || loading}
          style={{
            flex: 2, padding: "11px", borderRadius: 10, border: "none",
            background: !canNext() ? "var(--bg4)" : "var(--accent)",
            color: !canNext() ? "var(--text3)" : "#fff",
            cursor: !canNext() ? "default" : "pointer",
            fontSize: 13, fontWeight: 500,
          }}
        >
          {loading ? "Connecting…"
            : step === STEPS.length - 1 ? "Start using ARIA"
            : step === 2 && !data.androidIp.trim() ? "Skip for now"
            : "Continue →"}
        </button>
      </div>
    </div>
  );
}

function Input({ placeholder, value, onChange, autoFocus }) {
  return (
    <input
      placeholder={placeholder}
      value={value}
      autoFocus={autoFocus}
      onChange={e => onChange(e.target.value)}
      style={{
        background: "var(--bg3)", border: "1px solid var(--border)",
        borderRadius: 8, padding: "9px 12px", color: "var(--text)",
        fontSize: 13, outline: "none", fontFamily: "inherit", width: "100%",
      }}
      onFocus={e => e.target.style.borderColor = "var(--accent)"}
      onBlur={e => e.target.style.borderColor = "var(--border)"}
    />
  );
}

const selectStyle = {
  background: "var(--bg3)", border: "1px solid var(--border)",
  borderRadius: 8, padding: "9px 12px", color: "var(--text)",
  fontSize: 13, width: "100%",
};
