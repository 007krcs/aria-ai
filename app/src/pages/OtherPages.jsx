// ─────────────────────────────────────────────────────────────────────────────
// UPLOAD PAGE
// ─────────────────────────────────────────────────────────────────────────────
import { useState, useRef, useEffect } from "react";
import { useStore, api, API } from "../store";

export function Upload() {
  const { addNotification } = useStore();
  const [results, setResults]   = useState([]);
  const [loading, setLoading]   = useState(false);
  const [domain, setDomain]     = useState("general");
  const [urlVal, setUrlVal]     = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef();

  const upload = async (file) => {
    setLoading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await fetch(`${API}/api/upload?domain=${encodeURIComponent(domain)}`, {
        method: "POST", body: form,
      });
      const d = await r.json();
      if (r.ok) {
        setResults(p => [{ name: file.name, ...d }, ...p]);
        addNotification(`✓ ${file.name} — ${d.chunks || 0} chunks`, "success");
      } else {
        addNotification(d.detail || "Upload failed", "error");
      }
    } catch (e) {
      addNotification(e.message, "error");
    }
    setLoading(false);
  };

  const ingestUrl = async () => {
    if (!urlVal.trim()) return;
    setLoading(true);
    try {
      const r = await api.post("/api/ingest-url", { url: urlVal, domain });
      const d = await r.json();
      addNotification(`✓ ${d.chunks || 0} chunks from URL`, "success");
      setUrlVal("");
    } catch (e) {
      addNotification(e.message, "error");
    }
    setLoading(false);
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:"var(--bg)" }}>
      <Header title="Upload Documents" />
      <div style={{ flex:1, overflowY:"auto", padding:18 }}>
        <div
          onDragOver={e=>{e.preventDefault();setDragOver(true);}}
          onDragLeave={()=>setDragOver(false)}
          onDrop={e=>{e.preventDefault();setDragOver(false);[...e.dataTransfer.files].forEach(upload);}}
          onClick={()=>inputRef.current?.click()}
          style={{
            border: `2px dashed ${dragOver ? "var(--accent)" : "var(--border)"}`,
            borderRadius:12, padding:"32px 20px", textAlign:"center",
            cursor:"pointer", marginBottom:16, background: dragOver ? "rgba(124,106,247,.06)" : "transparent",
            transition:"all .15s",
          }}
        >
          <div style={{ fontSize:32, marginBottom:8 }}>📄</div>
          <div style={{ fontSize:13, color:"var(--text2)", marginBottom:4 }}>
            {loading ? "Uploading…" : "Drop files here or click to browse"}
          </div>
          <div style={{ fontSize:11, color:"var(--text3)" }}>
            PDF · Word · Excel · PowerPoint · CSV · JSON · Images · Audio · Video
          </div>
          <input ref={inputRef} type="file" multiple style={{ display:"none" }}
            accept=".pdf,.docx,.xlsx,.pptx,.csv,.json,.txt,.md,.jpg,.jpeg,.png,.mp3,.wav,.mp4,.mkv"
            onChange={e=>[...e.target.files].forEach(upload)} />
        </div>

        <Row style={{ marginBottom:16, gap:8 }}>
          <Select value={domain} onChange={e=>setDomain(e.target.value)}
            options={["general","technology","science","medicine","law","finance","education"]} />
        </Row>

        <div style={{ marginBottom:16 }}>
          <Label>Or ingest a URL</Label>
          <Row style={{ gap:8, marginTop:6 }}>
            <input value={urlVal} onChange={e=>setUrlVal(e.target.value)}
              onKeyDown={e=>e.key==="Enter"&&ingestUrl()}
              placeholder="https://…" style={inputStyle} />
            <Btn onClick={ingestUrl} disabled={loading}>Fetch & Learn</Btn>
          </Row>
        </div>

        {results.map((r, i) => (
          <div key={i} style={{
            background:"var(--bg2)", border:"1px solid var(--border)",
            borderRadius:10, padding:14, marginBottom:8,
          }}>
            <div style={{ fontSize:13, fontWeight:500, marginBottom:4 }}>{r.name}</div>
            <div style={{ display:"flex", gap:12, fontSize:11, color:"var(--text3)" }}>
              <span>{r.chunks} chunks</span>
              <span>{r.language}</span>
              <span>{r.domain}</span>
            </div>
            {r.summary && <div style={{ fontSize:12, color:"var(--text2)", marginTop:6, lineHeight:1.5 }}>{r.summary?.slice(0,200)}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ACTIONS PAGE — call, message, alarm, email, etc.
// ─────────────────────────────────────────────────────────────────────────────

export function Actions() {
  const { addNotification } = useStore();
  const [cmd, setCmd]     = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const EXAMPLES = [
    "Call 9876543210",
    "Set alarm at 7am tomorrow",
    "Search latest AI news on Google",
    "Play Blinding Lights on YouTube",
    "Remind me at 6pm to call mom",
    "What's the stock price of AAPL",
    "Open Chrome",
    "Take a screenshot",
    "What's the weather in Mumbai",
  ];

  const execute = async (command = cmd) => {
    if (!command.trim()) return;
    setLoading(true); setResult(null);
    try {
      const r = await api.post("/api/action", { command, device: "auto" });
      const d = await r.json();
      setResult(d);
      const ok = d.actions?.every(a => a.success !== false);
      addNotification(ok ? "✓ Action completed" : "⚠ Partial success", ok ? "success" : "warning");
    } catch (e) {
      addNotification(e.message, "error");
    }
    setLoading(false);
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:"var(--bg)" }}>
      <Header title="Actions" subtitle="Control devices · set alarms · send messages · search" />
      <div style={{ flex:1, overflowY:"auto", padding:18 }}>
        <Row style={{ gap:8, marginBottom:16 }}>
          <input value={cmd} onChange={e=>setCmd(e.target.value)}
            onKeyDown={e=>e.key==="Enter"&&execute()}
            placeholder="Say what you want ARIA to do…" style={{ ...inputStyle, flex:1 }} />
          <Btn onClick={()=>execute()} disabled={loading || !cmd.trim()}>
            {loading ? "…" : "Execute ↑"}
          </Btn>
        </Row>

        <div style={{ marginBottom:16 }}>
          <Label>Quick actions</Label>
          <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginTop:8 }}>
            {EXAMPLES.map(ex => (
              <button key={ex} onClick={()=>execute(ex)} style={{
                background:"var(--bg3)", border:"1px solid var(--border)",
                borderRadius:20, padding:"5px 12px", cursor:"pointer",
                color:"var(--text2)", fontSize:12,
              }}>{ex}</button>
            ))}
          </div>
        </div>

        {result && (
          <div style={{ background:"var(--bg2)", border:"1px solid var(--border)", borderRadius:10, padding:14 }}>
            <Label style={{ marginBottom:8 }}>Result</Label>
            {result.actions?.map((a, i) => {
              const outputText = a.output || a.answer || a.text || a.response || a.result || "";
              const isOk = a.success !== false;
              // Hide generic fallback labels — only show meaningful action types
              const HIDE_LABEL = new Set(["chat", "answer", "question", ""]);
              const showLabel = !HIDE_LABEL.has(a.action);
              return (
                <div key={i} style={{
                  display:"flex", alignItems:"flex-start", gap:8, padding:"8px 0",
                  borderBottom: i < result.actions.length-1 ? "1px solid var(--border)" : "none",
                }}>
                  <span style={{ color: isOk ? "var(--safe)" : "var(--danger)", flexShrink:0, fontSize:14, marginTop:1 }}>
                    {isOk ? "✓" : "✗"}
                  </span>
                  <div style={{ flex:1, minWidth:0 }}>
                    {showLabel && (
                      <div style={{ fontSize:10, fontWeight:700, color:"var(--cyan)",
                        textTransform:"uppercase", letterSpacing:".06em", marginBottom:5 }}>
                        {a.action.replace(/_/g, " ")}
                      </div>
                    )}
                    {outputText ? (
                      <div className="selectable" style={{ fontSize:13, color:"var(--text)",
                        lineHeight:1.7, whiteSpace:"pre-wrap", wordBreak:"break-word" }}>
                        {outputText}
                      </div>
                    ) : (
                      !a.error && (
                        <div style={{ fontSize:12, color:"var(--text3)" }}>
                          Done{showLabel ? ` — ${a.action.replace(/_/g," ")}` : ""}
                        </div>
                      )
                    )}
                    {a.error && <div style={{ fontSize:12, color:"var(--danger)", marginTop:4 }}>{a.error}</div>}
                    {a.url && (
                      <a href={a.url} target="_blank" rel="noreferrer"
                        style={{ fontSize:12, color:"var(--cyan)", marginTop:6, display:"block" }}>
                        {a.url.slice(0, 80)}
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SEARCH PAGE
// ─────────────────────────────────────────────────────────────────────────────

// Lightweight markdown renderer — same as Chat.jsx
function renderMd(text) {
  if (!text) return "";
  return text
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="lang-${lang||'text'}">${code.replace(/</g,"&lt;").replace(/>/g,"&gt;")}</code></pre>`)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g,    "<em>$1</em>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm,  "<h2>$1</h2>")
    .replace(/^# (.+)$/gm,   "<h1>$1</h1>")
    .replace(/^- (.+)$/gm,   "<li>$1</li>")
    .replace(/\n/g, "<br>");
}

export function Search() {
  const { settings }        = useStore();
  const [q, setQ]           = useState("");
  const [cards, setCards]   = useState([]);
  const [answer, setAnswer] = useState("");
  const [phase, setPhase]   = useState("idle"); // idle | searching | answering | done
  const esRef               = useRef(null);

  // Clean up SSE on unmount
  useEffect(() => () => esRef.current?.close(), []);

  const search = () => {
    if (!q.trim() || phase === "searching" || phase === "answering") return;

    // Reset state
    setCards([]);
    setAnswer("");
    setPhase("searching");

    // Close any previous stream
    esRef.current?.close();

    // POST body via fetch + SSE trick: we use fetch to POST then switch to EventSource-style reading
    // Since EventSource only supports GET, we use fetch + ReadableStream manually
    const ctrl = new AbortController();
    esRef.current = ctrl;

    // Enrich location-free weather queries with user's saved city
    const _weatherWords = ["weather","temperature","forecast","rain","hot","cold","sunny","cloudy"];
    const _locWords = [" in "," at "," for "," near "];
    const isWeather = _weatherWords.some(w => q.toLowerCase().includes(w));
    const hasLoc = _locWords.some(p => q.toLowerCase().includes(p));
    const searchQuery = (isWeather && !hasLoc && settings?.city)
      ? `${q.trim()} in ${settings.city}` : q;

    const token = window.__ariaToken || localStorage.getItem("aria_token") || "";
    fetch(`${API}/api/search/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ query: searchQuery, save_to_memory: true }),
      signal: ctrl.signal,
    }).then(async (res) => {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // Parse SSE lines
        const lines = buf.split("\n");
        buf = lines.pop(); // keep incomplete last line

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const msg = JSON.parse(line.slice(6));
            if (msg.type === "cards") {
              setCards(msg.cards || []);
              setPhase("answering");
            } else if (msg.type === "token") {
              setAnswer(prev => prev + (msg.text || ""));
            } else if (msg.type === "done") {
              setPhase("done");
            } else if (msg.type === "error") {
              setAnswer("Search failed. Please try again.");
              setPhase("done");
            }
          } catch (_) { /* ignore malformed lines */ }
        }
      }
      setPhase(p => p === "answering" ? "done" : p);
    }).catch(err => {
      if (err.name !== "AbortError") {
        setAnswer("Search failed. Please try again.");
        setPhase("done");
      }
    });
  };

  const isLoading = phase === "searching" || phase === "answering";

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:"var(--bg)" }}>
      <Header title="Web Search" />
      <div style={{ flex:1, overflowY:"auto", padding:18 }}>
        <Row style={{ gap:8, marginBottom:16 }}>
          <input value={q} onChange={e=>setQ(e.target.value)}
            onKeyDown={e=>e.key==="Enter"&&search()}
            placeholder="Search anything…" style={{ ...inputStyle, flex:1 }} />
          <Btn onClick={search} disabled={isLoading || !q.trim()}>
            {phase === "searching" ? "Fetching…" : phase === "answering" ? "Reading…" : "Search"}
          </Btn>
        </Row>

        {/* Phase indicator */}
        {phase === "searching" && (
          <div style={{ display:"flex", alignItems:"center", gap:8, padding:"8px 0", marginBottom:8,
            fontSize:12, color:"var(--text3)" }}>
            <span className="spin" style={{ display:"inline-block" }}>◌</span>
            Searching the web…
          </div>
        )}
        {phase === "answering" && (
          <div style={{ display:"flex", alignItems:"center", gap:8, padding:"4px 0", marginBottom:8,
            fontSize:12, color:"var(--text3)" }}>
            <span style={{ color:"var(--accent)", fontSize:10 }}>●</span>
            ARIA is reading results…
          </div>
        )}

        {/* Result cards — shown as soon as Phase 1 completes */}
        {cards.map((c, i) => (
          <a key={i} href={c.url} target="_blank" rel="noreferrer" style={{
            display:"block", background:"var(--bg2)", border:"1px solid var(--border)",
            borderRadius:10, padding:14, marginBottom:8, textDecoration:"none",
          }}>
            <div style={{ fontSize:13, fontWeight:500, color:"var(--accent)", marginBottom:4 }}>{c.title}</div>
            <div style={{ fontSize:12, color:"var(--text2)", lineHeight:1.6 }}>{c.snippet?.slice(0, 220)}</div>
            <div style={{ fontSize:10, color:"var(--text3)", marginTop:4 }}>{c.url?.slice(0, 70)}</div>
          </a>
        ))}

        {/* Synthesised answer — streams in after cards, rendered as markdown */}
        {answer && (
          <div style={{
            background:"var(--bg2)", border:"1px solid var(--accent)",
            borderRadius:10, padding:14, marginTop: cards.length ? 8 : 0,
          }}>
            <div style={{ fontSize:10, fontWeight:600, color:"var(--accent)", marginBottom:8,
              textTransform:"uppercase", letterSpacing:".05em" }}>ARIA's Answer</div>
            <div className="md-body selectable"
              dangerouslySetInnerHTML={{ __html: renderMd(answer) }} />
            {phase === "answering" && (
              <span style={{ display:"inline-block", width:7, height:13,
                background:"var(--accent)", marginLeft:2, animation:"blink 1s step-end infinite",
                verticalAlign:"text-bottom", borderRadius:1 }} />
            )}
          </div>
        )}

        {/* Empty state after search */}
        {phase === "done" && !cards.length && !answer && (
          <div style={{ textAlign:"center", padding:32, color:"var(--text3)", fontSize:13 }}>
            No results found. Try a different query.
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TRENDS PAGE — live market overview + intelligence feed
// ─────────────────────────────────────────────────────────────────────────────

const MARKETS = [
  { key:"india", label:"India", flag:"🇮🇳" },
  { key:"us",    label:"US",    flag:"🇺🇸" },
  { key:"uk",    label:"UK",    flag:"🇬🇧" },
  { key:"japan", label:"Japan", flag:"🇯🇵" },
];

const TREND_SECTIONS = [
  ["arxiv",           "arXiv",          "#7c6af7"],
  ["hackernews",      "Hacker News",    "#f0a040"],
  ["github_trending", "GitHub",         "#3dd68c"],
  ["reddit",          "Reddit",         "#f06060"],
];

function Sparkline({ vals, color = "var(--cyan)" }) {
  if (!vals || vals.length < 2) return null;
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = max - min || 1;
  const w = 64, h = 24;
  const pts = vals.map((v, i) =>
    `${(i / (vals.length - 1)) * w},${h - ((v - min) / range) * h}`
  ).join(" ");
  return (
    <svg width={w} height={h} style={{ flexShrink:0 }}>
      <polyline fill="none" stroke={color} strokeWidth="1.5"
        points={pts} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PctBadge({ val }) {
  if (val == null) return null;
  const up = val >= 0;
  return (
    <span style={{
      fontSize:11, fontWeight:600, padding:"2px 6px", borderRadius:4,
      background: up ? "rgba(0,255,136,.1)" : "rgba(255,51,85,.1)",
      color: up ? "var(--safe)" : "var(--danger)",
    }}>{up ? "+" : ""}{val.toFixed(2)}%</span>
  );
}

export function Trends() {
  const [tab, setTab]             = useState("market");   // "market" | "intel"
  const [market, setMarket]       = useState("india");
  const [stocks, setStocks]       = useState([]);
  const [stocksLoading, setSL]    = useState(false);
  const [stocksErr, setStocksErr] = useState(null);

  const [topic, setTopic]         = useState("artificial intelligence");
  const [intel, setIntel]         = useState(null);
  const [intelLoading, setIL]     = useState(false);

  // Auto-load market data on mount + market change
  useEffect(() => {
    let cancelled = false;
    setSL(true); setStocksErr(null);
    api.get(`/api/stocks/quick?market=${market}`)
      .then(r => r.json())
      .then(d => {
        if (!cancelled) {
          if (d.ok) { setStocks(d.top10 || []); }
          else { setStocksErr(d.error || "Scan returned no data — market may be loading"); setStocks([]); }
          setSL(false);
        }
      })
      .catch(() => { if (!cancelled) { setStocksErr("Backend offline"); setSL(false); } });
    return () => { cancelled = true; };
  }, [market]);

  const fetchIntel = () => {
    setIL(true);
    api.post("/api/trend/pulse", { topic })
      .then(r => r.json())
      .then(d => { setIntel(d); setIL(false); })
      .catch(() => setIL(false));
  };

  const tabBtn = (key, label) => (
    <button key={key} onClick={() => setTab(key)} style={{
      padding:"6px 14px", borderRadius:6, border:"none", cursor:"pointer", fontSize:12,
      background: tab === key ? "var(--cyan)" : "var(--bg3)",
      color: tab === key ? "#000" : "var(--text2)",
      fontWeight: tab === key ? 600 : 400,
    }}>{label}</button>
  );

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:"var(--bg1)" }}>
      <Header title="Trend Intelligence" subtitle="Live market overview + intelligence feed" />

      {/* Tab strip */}
      <div style={{ display:"flex", gap:6, padding:"10px 16px 0",
        borderBottom:"1px solid var(--border)", background:"var(--bg2)" }}>
        {tabBtn("market", "📈 Market")}
        {tabBtn("intel",  "🧠 Intel Feed")}
      </div>

      <div style={{ flex:1, overflowY:"auto", padding:16 }}>

        {/* ── MARKET TAB ── */}
        {tab === "market" && (
          <>
            {/* Market selector */}
            <div style={{ display:"flex", gap:6, marginBottom:14, flexWrap:"wrap" }}>
              {MARKETS.map(m => (
                <button key={m.key} onClick={() => setMarket(m.key)} style={{
                  padding:"5px 12px", borderRadius:6, border:`1px solid ${market===m.key ? "var(--cyan)" : "var(--border)"}`,
                  background: market===m.key ? "rgba(0,212,255,.1)" : "var(--bg2)",
                  color: market===m.key ? "var(--cyan)" : "var(--text2)",
                  cursor:"pointer", fontSize:12, display:"flex", alignItems:"center", gap:5,
                }}>
                  <span>{m.flag}</span> {m.label}
                </button>
              ))}
              <button onClick={() => {
                setSL(true); setStocksErr(null);
                api.get(`/api/stocks/quick?market=${market}`)
                  .then(r=>r.json()).then(d=>{ setStocks(d.ok?d.top10||[]:[]);
                    if(!d.ok) setStocksErr(d.error||"Scan returned no data"); setSL(false);})
                  .catch(()=>{setStocksErr("Backend offline"); setSL(false);});
              }} style={{
                marginLeft:"auto", padding:"5px 12px", borderRadius:6,
                border:"1px solid var(--border)", background:"var(--bg2)",
                color:"var(--text3)", cursor:"pointer", fontSize:11,
              }}>↻ Refresh</button>
            </div>

            {stocksLoading && <Loading />}
            {stocksErr && (
              <div style={{ textAlign:"center", padding:32, color:"var(--text3)", fontSize:13 }}>
                <div style={{ fontSize:28, marginBottom:8 }}>📡</div>
                {stocksErr} — start the ARIA server to load live market data
              </div>
            )}

            {!stocksLoading && !stocksErr && stocks.length === 0 && (
              <div style={{ textAlign:"center", padding:32, color:"var(--text3)", fontSize:13 }}>
                <div style={{ fontSize:28, marginBottom:8 }}>📊</div>
                No data — try refreshing or changing market
              </div>
            )}

            {/* Stock cards grid */}
            {stocks.length > 0 && (
              <>
                <div style={{ fontSize:10, color:"var(--text3)", marginBottom:10,
                  textTransform:"uppercase", letterSpacing:".08em" }}>
                  Top {stocks.length} stocks · {MARKETS.find(m=>m.key===market)?.label} market
                </div>
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(220px, 1fr))", gap:10 }}>
                  {stocks.map((s, i) => {
                    const upside = s.upside_pct ?? 0;
                    const isUp = upside >= 0;
                    return (
                      <div key={s.ticker} style={{
                        background:"var(--bg2)", border:`1px solid ${isUp ? "rgba(0,255,136,.2)" : "rgba(255,51,85,.15)"}`,
                        borderRadius:10, padding:"12px 14px", position:"relative",
                      }}>
                        {/* Rank badge */}
                        <div style={{ position:"absolute", top:8, right:10,
                          fontSize:9, color:"var(--text3)", fontWeight:700 }}>#{i+1}</div>

                        {/* Ticker + name */}
                        <div style={{ fontSize:13, fontWeight:700, color:"var(--cyan)", marginBottom:2 }}>
                          {s.ticker}
                        </div>
                        <div style={{ fontSize:11, color:"var(--text2)", marginBottom:8,
                          whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis", paddingRight:20 }}>
                          {s.name || "—"}
                        </div>

                        {/* Price + change */}
                        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                          <div style={{ fontSize:15, fontWeight:600, color:"var(--text)" }}>
                            {s.price != null ? s.price.toLocaleString() : "—"}
                          </div>
                          <PctBadge val={upside} />
                        </div>

                        {/* Score bar */}
                        <div style={{ marginTop:10 }}>
                          <div style={{ display:"flex", justifyContent:"space-between",
                            fontSize:9, color:"var(--text3)", marginBottom:3 }}>
                            <span>Score</span>
                            <span>{s.composite_score?.toFixed(0) ?? "—"}</span>
                          </div>
                          <div style={{ height:3, borderRadius:2, background:"var(--bg4)", overflow:"hidden" }}>
                            <div style={{
                              height:"100%", borderRadius:2,
                              width:`${Math.min(100, Math.max(0, (s.composite_score ?? 0) / 1200 * 100))}%`,
                              background: isUp ? "var(--safe)" : "var(--danger)",
                              transition:"width .4s ease",
                            }} />
                          </div>
                        </div>

                        {/* Signal pill + sector */}
                        <div style={{ display:"flex", alignItems:"center", gap:6, marginTop:8 }}>
                          {s.buy_signal && (
                            <span style={{
                              fontSize:9, padding:"2px 6px", borderRadius:3,
                              background:"rgba(0,212,255,.12)", color:"var(--cyan)",
                              fontWeight:600, textTransform:"uppercase",
                            }}>{s.buy_signal}</span>
                          )}
                          {s.sector && (
                            <span style={{ fontSize:9, color:"var(--text3)" }}>{s.sector}</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </>
        )}

        {/* ── INTEL FEED TAB ── */}
        {tab === "intel" && (
          <>
            <Row style={{ gap:8, marginBottom:14 }}>
              <input value={topic} onChange={e=>setTopic(e.target.value)}
                onKeyDown={e=>e.key==="Enter"&&fetchIntel()}
                placeholder="Topic to track…" style={{ ...inputStyle, flex:1 }} />
              <Btn onClick={fetchIntel} disabled={intelLoading}>
                {intelLoading ? "Loading…" : "Fetch"}
              </Btn>
            </Row>

            {!intel && !intelLoading && (
              <div style={{ textAlign:"center", padding:32, color:"var(--text3)", fontSize:13 }}>
                <div style={{ fontSize:28, marginBottom:8 }}>🧠</div>
                Enter a topic and hit Fetch to pull from arXiv, Hacker News, GitHub and Reddit
              </div>
            )}

            {intelLoading && <Loading />}

            {intel && TREND_SECTIONS.map(([key, title, color]) => {
              const items = intel[key] || [];
              if (!items.length) return null;
              return (
                <div key={key} style={{ marginBottom:20 }}>
                  <div style={{ fontSize:10, fontWeight:700, color, marginBottom:8,
                    textTransform:"uppercase", letterSpacing:".08em",
                    display:"flex", alignItems:"center", gap:6 }}>
                    <div style={{ width:6, height:6, borderRadius:"50%", background:color }} />
                    {title}
                  </div>
                  {items.slice(0,5).map((item, i) => (
                    <a key={i} href={item.url||"#"} target="_blank" rel="noreferrer" style={{
                      display:"block", background:"var(--bg2)", border:"1px solid var(--border)",
                      borderRadius:8, padding:"10px 12px", marginBottom:6, textDecoration:"none",
                      transition:"border-color .12s",
                    }}>
                      <div style={{ fontSize:12, color:"var(--text)", marginBottom:3, lineHeight:1.4 }}>
                        {item.title || item.name}
                      </div>
                      {(item.summary || item.description) && (
                        <div style={{ fontSize:11, color:"var(--text3)", lineHeight:1.5 }}>
                          {(item.summary || item.description).slice(0,140)}
                        </div>
                      )}
                    </a>
                  ))}
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ANALYTICS PAGE
// ─────────────────────────────────────────────────────────────────────────────

export function Analytics() {
  const { addNotification } = useStore();
  const [today, setToday]   = useState(null);
  const [profile, setProf]  = useState(null);
  const [loading, setLoad]  = useState(true);
  const [acting, setActing] = useState(null);

  useEffect(() => {
    Promise.all([
      api.get("/api/monitor/today").then(r => r.json()),
      api.get("/api/behaviour/profile").then(r => r.json()),
    ]).then(([t, p]) => { setToday(t); setProf(p); setLoad(false); })
     .catch(() => setLoad(false));
  }, []);

  // Actionable insight button handler
  const act = async (action, payload, label) => {
    setActing(label);
    try {
      if (action === "set_reminder") {
        await api.post("/api/task/reminder", payload);
        addNotification(`✓ ${label}`, "success");
      } else if (action === "start_focus") {
        await api.post("/api/task/alarm", { time: `in ${payload.minutes} minutes`, label: "Focus session end" });
        addNotification(`✓ Focus timer: ${payload.minutes}min`, "success");
      } else if (action === "run_backup") {
        await api.post("/api/backup/run", {});
        addNotification("✓ Backup started", "success");
      } else if (action === "research") {
        window.location.hash = `/search?q=${encodeURIComponent(payload.topic)}`;
        addNotification(`Searching: ${payload.topic}`, "info");
      }
    } catch (e) { addNotification(e.message, "error"); }
    setActing(null);
  };

  const StatCard = ({ value, label, color }) => (
    <div style={{ background:"var(--bg2)", border:"1px solid var(--border)", borderRadius:10,
      padding:"14px", textAlign:"center" }}>
      <div style={{ fontSize:26, fontWeight:700, color: color || "var(--accent)" }}>{value ?? "—"}</div>
      <div style={{ fontSize:11, color:"var(--text3)", marginTop:4 }}>{label}</div>
    </div>
  );

  // Build actionable insights from data
  const insights = [];
  if (profile) {
    const peakHours = profile.peak_hours || [];
    if (peakHours.length) {
      const h = peakHours[0];
      insights.push({
        text: `Your peak focus is ${h}:00–${h+2}:00. Block this time tomorrow.`,
        action: { type:"set_reminder", payload:{ text:"Peak focus time — close distractions", time:`tomorrow ${h}:00` }, label:"Block focus time" },
        color: "var(--accent)",
      });
    }
    if (profile.stress_level === "high") {
      insights.push({
        text: "High stress detected. A 5-minute break can reset focus.",
        action: { type:"start_focus", payload:{ minutes:5 }, label:"Set 5min break timer" },
        color: "var(--amber)",
      });
    }
    if (profile.cognitive_style === "deep_worker") {
      insights.push({
        text: "You're a deep worker. Research on flow state could help you optimise.",
        action: { type:"research", payload:{ topic:"flow state deep work productivity" }, label:"Research this" },
        color: "var(--teal)",
      });
    }
    if ((profile.avg_focus_min || 0) < 15) {
      insights.push({
        text: "Average focus sessions under 15min. Try the Pomodoro technique.",
        action: { type:"start_focus", payload:{ minutes:25 }, label:"Start 25min Pomodoro" },
        color: "var(--blue)",
      });
    }
  }

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:"var(--bg)" }}>
      <Header title="Analytics" subtitle="Behaviour · insights · actions" />
      <div style={{ flex:1, overflowY:"auto", padding:18 }}>
        {loading && <Loading />}
        {today && (
          <>
            <Label style={{ marginBottom:10 }}>Today</Label>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10, marginBottom:18 }}>
              <StatCard value={today.active_time_h + "h"} label="Active" />
              <StatCard value={today.idle_time_h + "h"} label="Idle" color="var(--text3)" />
              <StatCard value={today.focus_sessions?.length || 0} label="Focus sessions" color="var(--green)" />
            </div>
            {today.top_apps?.length > 0 && (
              <div style={{ marginBottom:18 }}>
                <Label style={{ marginBottom:8 }}>Top apps</Label>
                {today.top_apps.slice(0,6).map((a, i) => (
                  <div key={i} style={{ display:"flex", alignItems:"center", gap:8, marginBottom:6 }}>
                    <div style={{ fontSize:12, color:"var(--text2)", minWidth:120 }}>{a.app}</div>
                    <div style={{ flex:1, background:"var(--bg3)", borderRadius:3, height:6, overflow:"hidden" }}>
                      <div style={{ height:"100%", background:"var(--accent)",
                        width:`${Math.min(100,a.minutes/(today.top_apps[0]?.minutes||1)*100)}%` }} />
                    </div>
                    <div style={{ fontSize:11, color:"var(--text3)", minWidth:40, textAlign:"right" }}>{a.minutes}m</div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {insights.length > 0 && (
          <div style={{ marginBottom:18 }}>
            <Label style={{ marginBottom:8 }}>Actionable insights</Label>
            {insights.map((ins, i) => (
              <div key={i} style={{
                background:"var(--bg2)", border:"1px solid var(--border)",
                borderRadius:10, padding:"12px 14px", marginBottom:8,
                display:"flex", alignItems:"center", gap:12,
              }}>
                <div style={{ flex:1 }}>
                  <div style={{ width:3, height:"100%", background:ins.color,
                    borderRadius:2, alignSelf:"stretch", flexShrink:0 }} />
                  <div style={{ fontSize:13, color:"var(--text)", lineHeight:1.5 }}>{ins.text}</div>
                </div>
                {ins.action && (
                  <button
                    onClick={() => act(ins.action.type, ins.action.payload, ins.action.label)}
                    disabled={acting === ins.action.label}
                    style={{
                      background:"var(--bg3)", border:"1px solid var(--border)",
                      borderRadius:8, padding:"5px 10px", cursor:"pointer",
                      color:"var(--text2)", fontSize:11, whiteSpace:"nowrap", flexShrink:0,
                    }}
                  >
                    {acting === ins.action.label ? "…" : ins.action.label}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {profile && (
          <>
            <Label style={{ margin:"0 0 10px" }}>Behaviour profile</Label>
            <div style={{ background:"var(--bg2)", border:"1px solid var(--border)", borderRadius:10, padding:14, marginBottom:16 }}>
              <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10, marginBottom:12 }}>
                <StatCard value={profile.cognitive_style?.replace(/_/g," ")} label="Style" />
                <StatCard value={profile.stress_level || "—"} label="Stress"
                  color={{ low:"var(--green)", medium:"var(--amber)", high:"var(--red)" }[profile.stress_level] || "var(--text2)"} />
                <StatCard value={(profile.avg_focus_min||0)+"m"} label="Avg focus" />
              </div>
              {profile.narrative && (
                <div style={{ fontSize:12, color:"var(--text2)", lineHeight:1.7 }}>
                  {profile.narrative}
                </div>
              )}
            </div>
          </>
        )}

        {/* Quick actions row */}
        <Label style={{ marginBottom:8 }}>Quick actions</Label>
        <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
          {[
            ["Run backup", () => act("run_backup", {}, "Run backup")],
            ["25min focus", () => act("start_focus", { minutes:25 }, "Start 25min Pomodoro")],
            ["Research focus science", () => act("research", { topic:"focus productivity neuroscience" }, "Research this")],
          ].map(([label, fn]) => (
            <button key={label} onClick={fn} style={{
              background:"var(--bg2)", border:"1px solid var(--border)",
              borderRadius:8, padding:"7px 12px", cursor:"pointer",
              color:"var(--text2)", fontSize:12,
            }}>{label}</button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SETTINGS PAGE
// ─────────────────────────────────────────────────────────────────────────────

export function Settings() {
  const { settings, updateSettings, addNotification } = useStore();
  const [androidIp, setIp]   = useState("");
  const [connecting, setConn] = useState(false);

  const connectAndroid = async () => {
    if (!androidIp.trim()) return;
    setConn(true);
    try {
      const [ip, port] = androidIp.split(":");
      const r = await api.post("/api/device/android/connect", { ip, port: parseInt(port)||5555 });
      const d = await r.json();
      addNotification(d.success ? "✓ Android connected" : `✗ ${d.output}`, d.success ? "success" : "error");
    } catch (e) { addNotification(e.message, "error"); }
    setConn(false);
  };

  const runAudit = async () => {
    addNotification("Running security audit…", "info");
    const r = await api.get("/api/security/audit?force=true");
    const d = await r.json();
    addNotification(`Security: ${d.overall_risk} risk · ${d.dependency_audit?.vuln_count||0} vulns`, 
      d.overall_risk === "HIGH" ? "error" : d.overall_risk === "MEDIUM" ? "warning" : "success");
  };

  const S = ({ label, children }) => (
    <div style={{ marginBottom:16 }}>
      <Label style={{ marginBottom:8 }}>{label}</Label>
      {children}
    </div>
  );

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", background:"var(--bg)" }}>
      <Header title="Settings" />
      <div style={{ flex:1, overflowY:"auto", padding:18 }}>
        <S label="Your location (for weather)">
          <input value={settings.city} onChange={e=>updateSettings({city:e.target.value})}
            placeholder="e.g. Mumbai, New York, London…" style={inputStyle} />
        </S>
        <S label="Language preference">
          <Select value={settings.language} onChange={e=>updateSettings({language:e.target.value})}
            options={[["en","English"],["hi","Hindi"],["auto","Auto-detect"]]} />
        </S>
        <S label="Android device (ADB over WiFi)">
          <div style={{ fontSize:12, color:"var(--text3)", marginBottom:8, lineHeight:1.6 }}>
            Enable: Settings → Developer Options → Wireless Debugging<br/>
            Note the IP:PORT shown on your phone screen.
          </div>
          <Row style={{ gap:8 }}>
            <input value={androidIp} onChange={e=>setIp(e.target.value)}
              placeholder="192.168.1.X:PORT" style={{ ...inputStyle, flex:1 }} />
            <Btn onClick={connectAndroid} disabled={connecting}>
              {connecting ? "…" : "Connect"}
            </Btn>
          </Row>
        </S>
        <S label="Auto-speak responses">
          <Toggle checked={settings.autoSpeak} onChange={v=>updateSettings({autoSpeak:v})} />
        </S>
        <S label="Wake word (Hey ARIA)">
          <Toggle checked={settings.wakeWord} onChange={v=>updateSettings({wakeWord:v})} />
        </S>
        <S label="Theme">
          <div style={{ display:"flex", gap:8 }}>
            {["dark","light","system"].map(t => (
              <button key={t} onClick={() => {
                updateSettings({ theme: t });
                localStorage.setItem("aria_theme", t);
              }} style={{
                flex:1, padding:"7px", borderRadius:8, cursor:"pointer",
                border:`1px solid ${settings.theme===t ? "var(--accent)" : "var(--border)"}`,
                background: settings.theme===t ? "rgba(124,106,247,.12)" : "var(--bg3)",
                color: settings.theme===t ? "var(--accent)" : "var(--text2)",
                fontSize:12, fontWeight: settings.theme===t ? 500 : 400,
                textTransform:"capitalize",
              }}>{t}</button>
            ))}
          </div>
        </S>
        <S label="Backup">
          <div style={{ display:"flex", gap:8 }}>
            <Btn onClick={async()=>{
              addNotification("Creating backup…","info");
              const r = await api.post("/api/backup/run",{});
              const d = await r.json();
              addNotification(d.success ? `✓ Backup: ${d.size_mb}MB` : `✗ ${d.error}`,
                d.success ? "success" : "error");
            }} variant="outline">Backup Now</Btn>
            <Btn onClick={async()=>{
              const r = await api.get("/api/backup/status");
              const d = await r.json();
              addNotification(`${d.backup_count} backups · ${d.total_size_mb}MB total`, "info");
            }} variant="outline">View Backups</Btn>
          </div>
        </S>
        <S label="Security">
          <Btn onClick={runAudit} variant="outline">Run Security Audit</Btn>
        </S>
        <S label="Self-improvement">
          <Btn onClick={async()=>{
            addNotification("Running improvement cycle…","info");
            await api.post("/api/improve/run",{});
            addNotification("Improvement cycle complete","success");
          }} variant="outline">Run Improvement Cycle</Btn>
        </S>
        <div style={{ fontSize:11, color:"var(--text3)", marginTop:16, lineHeight:1.7 }}>
          ARIA v3 · Local · Private · Free<br />
          Server: http://localhost:8000
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared UI primitives
// ─────────────────────────────────────────────────────────────────────────────

function Header({ title, subtitle }) {
  return (
    <div style={{ padding:"12px 18px", borderBottom:"1px solid var(--border)",
      background:"var(--bg2)" }}>
      <div style={{ fontSize:14, fontWeight:500 }}>{title}</div>
      {subtitle && <div style={{ fontSize:11, color:"var(--text3)", marginTop:2 }}>{subtitle}</div>}
    </div>
  );
}

function Label({ children, style }) {
  return <div style={{ fontSize:11, fontWeight:600, color:"var(--text3)",
    textTransform:"uppercase", letterSpacing:".05em", ...style }}>{children}</div>;
}

function Row({ children, style }) {
  return <div style={{ display:"flex", alignItems:"center", ...style }}>{children}</div>;
}

function Btn({ children, onClick, disabled, variant }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      padding:"8px 14px", borderRadius:8, border:`1px solid ${variant==="outline" ? "var(--border)" : "var(--accent)"}`,
      background: variant==="outline" ? "transparent" : disabled ? "var(--bg4)" : "var(--accent)",
      color: disabled ? "var(--text3)" : "#fff", cursor: disabled ? "default" : "pointer",
      fontSize:12, fontWeight:500, whiteSpace:"nowrap", flexShrink:0,
    }}>{children}</button>
  );
}

function Select({ value, onChange, options }) {
  return (
    <select value={value} onChange={onChange} style={{
      background:"var(--bg3)", border:"1px solid var(--border)", borderRadius:8,
      padding:"7px 10px", color:"var(--text)", fontSize:13, width:"100%",
    }}>
      {options.map(o => Array.isArray(o)
        ? <option key={o[0]} value={o[0]}>{o[1]}</option>
        : <option key={o} value={o}>{o}</option>
      )}
    </select>
  );
}

function Toggle({ checked, onChange }) {
  return (
    <div onClick={()=>onChange(!checked)} style={{
      width:44, height:24, borderRadius:12, cursor:"pointer",
      background: checked ? "var(--accent)" : "var(--bg4)",
      position:"relative", transition:"background .2s",
    }}>
      <div style={{
        position:"absolute", top:3, left: checked ? 22 : 3,
        width:18, height:18, borderRadius:"50%", background:"#fff",
        transition:"left .2s",
      }} />
    </div>
  );
}

function Loading() {
  return (
    <div style={{ textAlign:"center", padding:24, color:"var(--text3)" }}>
      <span className="spin" style={{ display:"inline-block", fontSize:20 }}>◌</span>
    </div>
  );
}

const inputStyle = {
  background:"var(--bg3)", border:"1px solid var(--border)", borderRadius:8,
  padding:"8px 12px", color:"var(--text)", fontSize:13, outline:"none",
  fontFamily:"inherit", width:"100%",
};
