import { useState, useEffect, useRef } from "react";
import { api } from "../store";

// Persists token in localStorage
export const TokenStore = {
  get:   ()    => localStorage.getItem("aria_token"),
  set:   (t)   => localStorage.setItem("aria_token", t),
  clear: ()    => localStorage.removeItem("aria_token"),
};

// ── PIN grid — MUST be defined outside Login to prevent remount on every render ─
function PinGrid({ value, handlers, refs }) {
  return (
    <div style={{ display:"flex", gap:10, justifyContent:"center", margin:"18px 0" }}>
      {value.map((d, i) => (
        <input
          key={i}
          ref={el => (refs.current[i] = el)}
          type="password"
          inputMode="numeric"
          maxLength={1}
          value={d}
          autoComplete="off"
          onChange={e => handlers.onChange(i, e.target.value)}
          onKeyDown={e => handlers.onKeyDown(i, e)}
          onFocus={e => e.target.select()}
          style={{
            width:50, height:58, textAlign:"center",
            fontSize:24, fontWeight:700,
            background:"var(--bg2)",
            border:`2px solid ${d ? "var(--cyan)" : "var(--border2)"}`,
            borderRadius:10, color:"var(--text)", outline:"none",
            boxShadow: d ? "0 0 12px var(--glow)" : "none",
            transition:"border-color .15s, box-shadow .15s",
            caretColor:"transparent",
          }}
        />
      ))}
    </div>
  );
}

export default function Login({ onAuth }) {
  // screens: check | setup | setup-done | login | reset | pair
  const [screen, setScreen]         = useState("check");
  const [pin, setPin]               = useState(["","","","","",""]);
  const [newPin, setNewPin]         = useState(["","","","","",""]);
  const [name, setName]             = useState("");
  const [recoveryCode, setRecovery] = useState(""); // shown after setup
  const [resetCode, setResetCode]   = useState(""); // entered by user on reset screen
  const [error, setError]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [pairCode, setPairCode]     = useState("");
  const pinRefs    = useRef([]);
  const newPinRefs = useRef([]);

  // Check server on mount
  useEffect(() => {
    (async () => {
      try {
        const existing = TokenStore.get();
        if (existing) {
          const r = await api.get("/auth/verify", {
            headers: { Authorization: `Bearer ${existing}` },
          });
          if (r.ok) { onAuth(existing); return; }
          TokenStore.clear();
        }
        const r = await api.get("/auth/status");
        const d = await r.json();
        setScreen(d.setup ? "login" : "setup");
      } catch {
        setError("Cannot reach ARIA server. Is it running on port 8000?");
        setScreen("login");
      }
    })();
  }, [onAuth]);

  // ── PIN digit handlers ────────────────────────────────────────────────────
  const makePinHandler = (getter, setter, refs, onDone) => ({
    onChange: (i, val) => {
      if (!/^\d?$/.test(val)) return;
      const next = [...getter()]; next[i] = val; setter(next);
      if (val && i < 5) refs.current[i + 1]?.focus();
      if (!val && i > 0) refs.current[i - 1]?.focus();
    },
    onKeyDown: (i, e) => {
      if (e.key === "Backspace" && !getter()[i] && i > 0) refs.current[i - 1]?.focus();
      if (e.key === "Enter") onDone();
    },
  });

  const pinHandlers    = makePinHandler(() => pin,    setPin,    pinRefs,    () => doAuth());
  const newPinHandlers = makePinHandler(() => newPin, setNewPin, newPinRefs, () => doReset());

  const pinValue    = pin.join("");
  const newPinValue = newPin.join("");

  // ── Auth actions ──────────────────────────────────────────────────────────
  const doSetup = async () => {
    if (pinValue.length < 4) { setError("Enter at least 4 digits"); return; }
    setLoading(true); setError("");
    try {
      const r = await api.post("/auth/setup", { pin: pinValue, owner_name: name || "Owner" });
      const d = await r.json();
      if (d.success) {
        // Show recovery code before entering — user must confirm they saved it
        setRecovery(d.recovery_code || "");
        TokenStore.set(d.token);
        setScreen("setup-done");
      } else {
        setError(d.message || "Setup failed");
      }
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  const doLogin = async () => {
    if (pinValue.length < 4) { setError("Enter your PIN"); return; }
    setLoading(true); setError("");
    try {
      const r = await api.post("/auth/login", {
        pin: pinValue,
        device_name: navigator.userAgent.slice(0, 30),
      });
      const d = await r.json();
      if (r.ok && d.token) { TokenStore.set(d.token); onAuth(d.token); }
      else setError(d.detail || d.message || "Wrong PIN");
    } catch { setError("Server error"); }
    setLoading(false);
  };

  const doReset = async () => {
    if (!resetCode.trim()) { setError("Enter your recovery code"); return; }
    if (newPinValue.length < 4) { setError("Enter new PIN (at least 4 digits)"); return; }
    setLoading(true); setError("");
    try {
      const r = await api.post("/auth/reset-pin", {
        recovery_code: resetCode.trim().toUpperCase(),
        new_pin:       newPinValue,
      });
      const d = await r.json();
      if (r.ok && d.token) {
        // Show new recovery code before entering
        setRecovery(d.recovery_code || "");
        TokenStore.set(d.token);
        setScreen("setup-done");
      } else {
        setError(d.detail || d.error || "Reset failed");
      }
    } catch { setError("Server error"); }
    setLoading(false);
  };

  const doAuth = screen === "setup" ? doSetup : doLogin;

  // ── Screens ───────────────────────────────────────────────────────────────

  if (screen === "check") {
    return (
      <LoginOverlay>
        <ARIALogo />
        <div style={{ color:"var(--text3)", fontSize:13, marginTop:8,
                      letterSpacing:2, animation:"dotBlink 1.5s ease infinite" }}>
          CONNECTING…
        </div>
      </LoginOverlay>
    );
  }

  // Recovery code display — shown after setup or after reset
  if (screen === "setup-done") {
    return (
      <LoginOverlay>
        <ARIALogo />
        <AuthCard>
          <div style={{ textAlign:"center", marginBottom:8 }}>
            <span style={{ fontSize:28 }}>🔑</span>
          </div>
          <div style={{ fontSize:15, fontWeight:700, color:"var(--safe)",
                        textAlign:"center", marginBottom:6 }}>
            Save Your Recovery Code
          </div>
          <div style={{ fontSize:11, color:"var(--text2)", textAlign:"center",
                        marginBottom:16, lineHeight:1.7 }}>
            If you forget your PIN, this is the <strong style={{color:"var(--text)"}}>only</strong> way
            to regain access.<br />It won't be shown again.
          </div>
          {/* Recovery code box */}
          <div style={{
            background:"rgba(0,255,136,.05)", border:"1px solid rgba(0,255,136,.25)",
            borderRadius:8, padding:"14px 18px", textAlign:"center", marginBottom:16,
          }}>
            <div style={{ fontSize:10, color:"var(--text3)", letterSpacing:1,
                          textTransform:"uppercase", marginBottom:8 }}>
              Your Recovery Code
            </div>
            <div style={{
              fontFamily:"monospace", fontSize:22, fontWeight:700,
              color:"var(--safe)", letterSpacing:4,
              textShadow:"0 0 12px rgba(0,255,136,.4)",
              userSelect:"all",
            }}>
              {recoveryCode || "—"}
            </div>
            <div style={{ fontSize:10, color:"var(--text3)", marginTop:8 }}>
              Click to select — copy and store safely
            </div>
          </div>
          <AuthBtn onClick={() => onAuth(TokenStore.get())} loading={false} disabled={false}>
            I've Saved It — Enter ARIA
          </AuthBtn>
        </AuthCard>
      </LoginOverlay>
    );
  }

  // Reset PIN screen
  if (screen === "reset") {
    return (
      <LoginOverlay>
        <ARIALogo />
        <AuthCard>
          <div style={{ fontSize:15, fontWeight:600, textAlign:"center",
                        color:"var(--text)", marginBottom:4 }}>
            Reset PIN
          </div>
          <div style={{ fontSize:11, color:"var(--text2)", textAlign:"center",
                        marginBottom:16, lineHeight:1.7 }}>
            Enter the recovery code you saved during setup,<br />
            then set a new PIN.
          </div>

          {/* Recovery code input */}
          <div style={{ fontSize:11, color:"var(--text3)", marginBottom:6 }}>
            Recovery Code
          </div>
          <input
            value={resetCode}
            onChange={e => setResetCode(e.target.value.toUpperCase())}
            placeholder="XXXXXXXXXXXX"
            maxLength={12}
            style={{
              width:"100%", fontFamily:"monospace", textAlign:"center",
              letterSpacing:4, fontSize:16, fontWeight:700,
              background:"var(--bg2)", border:"1px solid var(--border2)",
              borderRadius:8, padding:"10px 14px", color:"var(--safe)",
              outline:"none", marginBottom:14,
              transition:"border-color .15s",
            }}
            onFocus={e => (e.target.style.borderColor = "var(--safe)")}
            onBlur={e => (e.target.style.borderColor = "var(--border2)")}
          />

          {/* New PIN */}
          <div style={{ fontSize:11, color:"var(--text3)", textAlign:"center", marginBottom:4 }}>
            New PIN (4–6 digits)
          </div>
          <PinGrid value={newPin} handlers={newPinHandlers} refs={newPinRefs} />

          {error && <ErrorMsg>{error}</ErrorMsg>}
          <AuthBtn onClick={doReset} loading={loading}
                   disabled={!resetCode.trim() || newPinValue.length < 4}>
            {loading ? "Resetting…" : "Reset PIN"}
          </AuthBtn>
          <TextBtn onClick={() => { setScreen("login"); setError(""); }}>
            Back to Login
          </TextBtn>
        </AuthCard>
      </LoginOverlay>
    );
  }

  // Pair screen
  if (screen === "pair") {
    return (
      <LoginOverlay>
        <ARIALogo />
        <AuthCard>
          <div style={{ fontSize:15, fontWeight:600, marginBottom:8,
                        textAlign:"center", color:"var(--text)" }}>
            Pair this device
          </div>
          <div style={{ fontSize:12, color:"var(--text2)", textAlign:"center",
                        marginBottom:16, lineHeight:1.6 }}>
            Enter the 6-digit code shown on your main device.<br />
            Get a code from <strong style={{color:"var(--text)"}}>Settings › Pair new device</strong>.
          </div>
          <input
            value={pairCode}
            onChange={e => setPairCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
            placeholder="000000"
            style={{
              width:"100%", textAlign:"center", letterSpacing:8, fontSize:24,
              background:"var(--bg2)", border:"1px solid var(--border2)",
              borderRadius:8, padding:"12px", color:"var(--text)",
              outline:"none", marginBottom:12,
            }}
          />
          {error && <ErrorMsg>{error}</ErrorMsg>}
          <AuthBtn
            onClick={async () => {
              setLoading(true); setError("");
              try {
                const r = await api.post("/auth/pair", {
                  code: pairCode, device_name: navigator.userAgent.slice(0, 30),
                });
                const d = await r.json();
                if (d.success) { TokenStore.set(d.token); onAuth(d.token); }
                else setError(d.error || "Invalid code");
              } catch { setError("Server error"); }
              setLoading(false);
            }}
            loading={loading}
            disabled={pairCode.length < 6}
          >
            Pair Device
          </AuthBtn>
          <TextBtn onClick={() => setScreen("login")}>Use PIN instead</TextBtn>
        </AuthCard>
      </LoginOverlay>
    );
  }

  // Setup screen
  if (screen === "setup") {
    return (
      <LoginOverlay>
        <ARIALogo />
        <AuthCard>
          <div style={{ fontSize:15, fontWeight:600, marginBottom:4,
                        textAlign:"center", color:"var(--text)" }}>
            Welcome to ARIA
          </div>
          <div style={{ fontSize:12, color:"var(--text2)", textAlign:"center",
                        marginBottom:14, lineHeight:1.6 }}>
            Set a PIN to protect your assistant.<br />
            A recovery code will be generated — save it safely.
          </div>
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Your name (optional)"
            style={{
              width:"100%", background:"var(--bg2)",
              border:"1px solid var(--border2)",
              borderRadius:8, padding:"10px 14px",
              color:"var(--text)", fontSize:13, outline:"none",
              marginBottom:4, transition:"border-color .15s",
            }}
            onFocus={e => (e.target.style.borderColor = "var(--cyan)")}
            onBlur={e => (e.target.style.borderColor = "var(--border2)")}
          />
          <div style={{ fontSize:11, color:"var(--text3)", textAlign:"center",
                        margin:"8px 0 2px" }}>
            Create a PIN (4–6 digits)
          </div>
          <PinGrid value={pin} handlers={pinHandlers} refs={pinRefs} />
          {error && <ErrorMsg>{error}</ErrorMsg>}
          <AuthBtn onClick={doSetup} loading={loading} disabled={pinValue.length < 4}>
            {loading ? "Setting up…" : "Set PIN & Start ARIA"}
          </AuthBtn>
        </AuthCard>
      </LoginOverlay>
    );
  }

  // Login screen (default)
  return (
    <LoginOverlay>
      <ARIALogo />
      <AuthCard>
        <div style={{ fontSize:15, fontWeight:600, marginBottom:4,
                      textAlign:"center", color:"var(--text)" }}>
          Welcome back
        </div>
        <div style={{ fontSize:12, color:"var(--text2)", textAlign:"center",
                      marginBottom:4 }}>
          Enter your PIN to access ARIA
        </div>
        <PinGrid value={pin} handlers={pinHandlers} refs={pinRefs} />
        {error && <ErrorMsg>{error}</ErrorMsg>}
        <AuthBtn onClick={doLogin} loading={loading} disabled={pinValue.length < 4}>
          {loading ? "Verifying…" : "Unlock ARIA"}
        </AuthBtn>
        <div style={{ display:"flex", justifyContent:"space-between", marginTop:4 }}>
          <TextBtn onClick={() => { setScreen("pair"); setError(""); }}>
            Pair new device
          </TextBtn>
          <TextBtn onClick={() => { setScreen("reset"); setError(""); setNewPin(["","","","","",""]); }}>
            Forgot PIN?
          </TextBtn>
        </div>
      </AuthCard>
    </LoginOverlay>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function LoginOverlay({ children }) {
  return (
    <div style={{
      height:"100vh", width:"100vw",
      background:"var(--bg0)",
      display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center",
      position:"relative", overflow:"hidden",
    }}>
      <div className="hud-corner tl" />
      <div className="hud-corner tr" />
      <div className="hud-corner bl" />
      <div className="hud-corner br" />
      {children}
    </div>
  );
}

function ARIALogo() {
  return (
    <div style={{ textAlign:"center", marginBottom:28 }}>
      <div style={{
        width:72, height:72, borderRadius:"50%",
        border:"2px solid var(--cyan)",
        boxShadow:"0 0 24px var(--glow), inset 0 0 16px var(--glow2)",
        display:"inline-flex", alignItems:"center", justifyContent:"center",
        marginBottom:14, animation:"ringPulse 3s ease-in-out infinite",
      }}>
        <span style={{ fontSize:26, fontWeight:800, color:"var(--cyan)", letterSpacing:1 }}>A</span>
      </div>
      <div style={{
        fontSize:34, fontWeight:800, letterSpacing:6, color:"var(--cyan)",
        textShadow:"0 0 20px rgba(0,212,255,.5)", marginBottom:4,
      }}>
        ARIA
      </div>
      <div style={{ fontSize:11, color:"var(--text3)", letterSpacing:2, textTransform:"uppercase" }}>
        Personal AI Assistant
      </div>
    </div>
  );
}

function AuthCard({ children }) {
  return (
    <div style={{
      background:"var(--bg1)", border:"1px solid var(--border2)",
      borderRadius:10, padding:"28px 28px 20px",
      width:340, display:"flex", flexDirection:"column",
      boxShadow:"0 0 40px var(--glow2), 0 4px 32px rgba(0,0,0,.5)",
    }}>
      {children}
    </div>
  );
}

function ErrorMsg({ children }) {
  return (
    <div style={{
      background:"rgba(255,51,85,.1)", border:"1px solid rgba(255,51,85,.3)",
      borderRadius:6, padding:"8px 12px",
      fontSize:12, color:"var(--danger)",
      marginBottom:10, textAlign:"center",
    }}>
      {children}
    </div>
  );
}

function AuthBtn({ children, onClick, loading, disabled }) {
  const off = disabled || loading;
  return (
    <button
      onClick={onClick}
      disabled={off}
      style={{
        width:"100%", padding:"11px",
        borderRadius:8, border:"1px solid",
        borderColor: off ? "var(--border)" : "var(--cyan)",
        background: off ? "var(--bg2)"
          : "linear-gradient(135deg, var(--cyan3) 0%, var(--cyan2) 100%)",
        color: off ? "var(--text3)" : "#fff",
        fontSize:13, fontWeight:700,
        letterSpacing:1, textTransform:"uppercase",
        cursor: off ? "default" : "pointer",
        marginTop:8,
        boxShadow: off ? "none" : "0 0 12px var(--glow)",
        transition:"all .15s",
      }}
    >
      {children}
    </button>
  );
}

function TextBtn({ children, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        background:"none", border:"none",
        color:"var(--text3)", fontSize:11,
        cursor:"pointer", marginTop:10,
        transition:"color .12s",
      }}
      onMouseEnter={e => (e.target.style.color = "var(--cyan)")}
      onMouseLeave={e => (e.target.style.color = "var(--text3)")}
    >
      {children}
    </button>
  );
}
