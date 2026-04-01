"""
ARIA — NiceGUI Desktop App
===========================
Replaces Flet. NiceGUI is stable, pure Python, renders in any browser.

Why NiceGUI over Flet:
- No breaking icon/padding changes every version
- Works in browser → accessible from phone/tablet automatically
- One Python file, no compilation step
- Beautiful dark theme built in
- Vue.js components under the hood — smooth and fast
- Can open as native OS window with --native flag

Install:
    pip install nicegui

Run:
    python app/aria_app.py              # opens in browser
    python app/aria_app.py --native     # opens as OS window (needs pywebview)
"""

import sys
import os
import json
import time
import base64
import asyncio
import threading
import requests
import httpx
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from nicegui import ui, app as ng_app
except ImportError:
    print("Install NiceGUI: pip install nicegui")
    sys.exit(1)

API      = "http://localhost:8000"
APP_PORT = 8080   # NiceGUI runs on 8080, ARIA API on 8000

# ─────────────────────────────────────────────────────────────────────────────
# THEME  — dark, matching ARIA's colour palette
# ─────────────────────────────────────────────────────────────────────────────

ACCENT  = "#7c6af7"
GREEN   = "#3dd68c"
RED     = "#f06060"
AMBER   = "#f0a040"
BG      = "#0f0f11"
BG2     = "#18181c"
BG3     = "#222228"
TEXT    = "#e8e8f0"
TEXT2   = "#9898b0"
BORDER  = "#2e2e38"

ui.add_head_html(f"""
<style>
  :root {{
    --accent:  {ACCENT};
    --green:   {GREEN};
    --red:     {RED};
    --amber:   {AMBER};
    --bg:      {BG};
    --bg2:     {BG2};
    --bg3:     {BG3};
    --text:    {TEXT};
    --text2:   {TEXT2};
    --border:  {BORDER};
  }}

  body, .nicegui-content {{
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  }}

  /* Sidebar */
  .sidebar {{
    background: var(--bg2);
    border-right: 1px solid var(--border);
    min-height: 100vh;
    width: 210px;
    flex-shrink: 0;
  }}

  /* Cards */
  .aria-card {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 14px;
  }}

  /* Nav items */
  .nav-btn {{
    display: flex !important;
    align-items: center;
    gap: 10px;
    padding: 9px 12px !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-size: 13px !important;
    color: var(--text2) !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer;
    text-align: left !important;
    transition: background .15s;
  }}
  .nav-btn:hover {{ background: rgba(255,255,255,.04) !important; color: var(--text) !important; }}
  .nav-btn.active {{ background: rgba(124,106,247,.15) !important; color: var(--accent) !important; }}

  /* Chat bubbles */
  .bubble-user {{
    background: rgba(124,106,247,.12);
    border: 1px solid rgba(124,106,247,.25);
    border-radius: 12px 12px 2px 12px;
    padding: 12px 14px;
    margin: 6px 0 6px 60px;
    color: var(--text);
    font-size: 13px;
    line-height: 1.65;
  }}
  .bubble-aria {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 2px;
    padding: 12px 14px;
    margin: 6px 60px 6px 0;
    color: var(--text);
    font-size: 13px;
    line-height: 1.65;
  }}
  .bubble-aria code {{
    background: rgba(255,255,255,.07);
    padding: 1px 5px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
  }}
  .bubble-aria pre {{
    background: rgba(255,255,255,.07);
    padding: 10px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 8px 0;
  }}

  /* Search cards */
  .search-card {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
    cursor: pointer;
    text-decoration: none !important;
    display: block;
    transition: border-color .2s, transform .15s;
  }}
  .search-card:hover {{
    border-color: var(--accent);
    transform: translateY(-1px);
  }}
  .search-card .card-title {{ font-size: 13px; font-weight: 600; color: var(--accent); margin-bottom: 4px; }}
  .search-card .card-snippet {{ font-size: 12px; color: var(--text2); line-height: 1.55; }}
  .search-card .card-url {{ font-size: 10px; color: var(--text2); margin-top: 6px; }}

  /* Inputs */
  .q-field__native {{ color: var(--text) !important; }}
  .q-field__control {{ background: var(--bg3) !important; border: 1px solid var(--border) !important; }}
  .q-field--focused .q-field__control {{ border-color: var(--accent) !important; }}

  /* Stat chips */
  .stat-chip {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
  }}
  .stat-num {{ font-size: 22px; font-weight: 700; color: var(--accent); }}
  .stat-lbl {{ font-size: 10px; color: var(--text2); margin-top: 2px; }}

  /* Progress / log */
  .crawl-log {{
    background: var(--bg3);
    border-radius: 8px;
    padding: 12px;
    max-height: 300px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 11px;
    line-height: 1.7;
    color: var(--text2);
  }}
  .log-ok   {{ color: var(--green); }}
  .log-skip {{ color: var(--amber); }}
  .log-err  {{ color: var(--red); }}

  /* Badge */
  .mode-badge {{
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 20px;
    background: rgba(124,106,247,.15);
    color: var(--accent);
    font-weight: 500;
    margin-top: 5px;
    display: inline-block;
  }}

  /* Override NiceGUI defaults */
  .q-page {{ background: var(--bg) !important; }}
  .q-drawer {{ background: var(--bg2) !important; }}
  a {{ color: var(--accent); }}
  .nicegui-markdown p {{ color: var(--text) !important; margin: 4px 0; }}
  .nicegui-markdown h1,.nicegui-markdown h2,.nicegui-markdown h3 {{
    color: var(--text) !important; font-size: 14px !important; font-weight: 600;
  }}
  .nicegui-markdown code {{ background: var(--bg3); padding: 1px 5px; border-radius:4px; font-size:12px; }}
  .nicegui-markdown pre {{ background: var(--bg3); padding:10px; border-radius:8px; overflow-x:auto; }}
</style>
""", shared=True)


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.active_tab    = "chat"
        self.server_status = "Connecting..."
        self.server_ok     = False
        self.token         = ""   # Bearer JWT — fetched at login

state = AppState()

# ── Desktop PIN cache — stored in data/desktop_pin.json ──────────────────────
_PIN_CACHE = PROJECT_ROOT / "data" / "desktop_pin.json"

def _load_cached_pin() -> str:
    try:
        return json.loads(_PIN_CACHE.read_text())["pin"]
    except Exception:
        return ""

def _save_cached_pin(pin: str):
    try:
        _PIN_CACHE.write_text(json.dumps({"pin": pin}))
    except Exception:
        pass

def _do_login(pin: str) -> str:
    """POST /auth/login with pin, return token or '' on failure."""
    try:
        r = requests.post(f"{API}/auth/login",
                          json={"pin": pin, "device_name": "desktop"},
                          timeout=5)
        if r.ok:
            return r.json().get("token", "")
    except Exception:
        pass
    return ""

def _auto_login():
    """Try to login using cached PIN. Sets state.token if successful."""
    pin = _load_cached_pin()
    if pin:
        tok = _do_login(pin)
        if tok:
            state.token = tok
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {state.token}"} if state.token else {}


def api_get(path: str, timeout=5) -> dict | None:
    try:
        r = requests.get(f"{API}{path}", headers=_auth_headers(), timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(path: str, data: dict, timeout=30) -> dict | None:
    try:
        r = requests.post(f"{API}{path}", json=data, headers=_auth_headers(), timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


def check_server() -> tuple[bool, str]:
    try:
        r = requests.get(f"{API}/api/health", timeout=2)
        d = r.json()
        return True, f"{d.get('model','?')} · {d.get('chunks',0)} chunks"
    except Exception:
        return False, "Server offline — run server.py"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/")
def main_page():

    # ── Auth: show PIN dialog if no token ─────────────────────────────────────
    if not state.token:
        _auto_login()   # try cached PIN first

    if not state.token:
        with ui.dialog().props("persistent") as login_dlg, ui.card().style(
            f"background:{BG2};border:1px solid {BORDER};padding:24px;min-width:300px"
        ):
            ui.label("Enter your ARIA PIN").style(
                f"font-size:14px;font-weight:600;color:{TEXT};margin-bottom:12px"
            )
            pin_input = ui.input(placeholder="PIN", password=True).style(
                f"width:100%;background:{BG3}"
            ).props("outlined dense")
            err_label = ui.label("").style(f"font-size:11px;color:{RED};min-height:16px")

            def try_login():
                pin = pin_input.value.strip()
                tok = _do_login(pin)
                if tok:
                    state.token = tok
                    _save_cached_pin(pin)
                    login_dlg.close()
                    ui.navigate.to("/")   # reload page with token
                else:
                    err_label.text = "Incorrect PIN. Try again."
                    pin_input.value = ""

            pin_input.on("keydown.enter", try_login)
            ui.button("Unlock", on_click=try_login).style(
                f"background:{ACCENT};color:white;width:100%;margin-top:8px;border-radius:8px"
            ).props("no-caps")

        login_dlg.open()
        return   # don't render the rest of the page until logged in

    # ── Left sidebar ──────────────────────────────────────────────────────────
    with ui.left_drawer(fixed=True).style(
        f"background:{BG2}; border-right:1px solid {BORDER}; padding:0; width:210px"
    ):
        # Logo
        with ui.column().style("padding:18px 14px 10px"):
            with ui.row().style("gap:0"):
                ui.label("AR").style(f"font-size:17px;font-weight:700;color:{ACCENT}")
                ui.label("IA").style(f"font-size:17px;font-weight:700;color:{GREEN}")
            ui.label("Personal AI Assistant").style(f"font-size:10px;color:{TEXT2};margin-top:2px")

        ui.separator().style(f"background:{BORDER};margin:0")

        # Nav buttons — each swaps the right panel
        tab_panels_ref = {}  # will be filled after panels are created

        def make_nav(icon_char, label, tab_id):
            btn = ui.button(f"{icon_char}  {label}").style(
                "width:100%;text-align:left;padding:9px 12px;border-radius:8px;"
                f"font-size:13px;color:{TEXT2};background:transparent;border:none;"
                "cursor:pointer;margin:1px 6px;width:calc(100% - 12px)"
            ).props("flat no-caps")
            def on_click(t=tab_id, b=btn):
                state.active_tab = t
                # show/hide panels
                for k, panel in tab_panels_ref.items():
                    panel.set_visibility(k == t)
                ui.update()
            btn.on_click(on_click)
            return btn

        with ui.column().style("padding:8px 0;gap:2px"):
            make_nav("💬", "Chat",           "chat")
            make_nav("🌐", "Web Search",     "search")
            make_nav("📤", "Upload Docs",    "upload")
            make_nav("🕷️", "Web Crawler",    "crawl")
            make_nav("📚", "Knowledge Base", "kb")
            make_nav("📈", "Trends",         "trends")
            make_nav("👁️", "Vision OCR",     "vision")
            make_nav("🔐", "Security",       "security")
            make_nav("🧠", "NOVA Engine",    "nova")

        # Status footer
        ui.separator().style(f"background:{BORDER};margin:0")
        status_label = ui.label(state.server_status).style(
            f"font-size:11px;color:{TEXT2};padding:10px 14px"
        )
        status_dot = ui.badge("●").style(
            f"font-size:8px;color:{RED};background:transparent;padding:0 4px"
        )

        async def refresh_status():
            while True:
                ok, msg = check_server()
                status_label.text = msg
                status_dot.style(
                    f"font-size:8px;color:{'#3dd68c' if ok else '#f06060'};"
                    "background:transparent;padding:0 4px"
                )
                await asyncio.sleep(15)

        ui.timer(0.1, lambda: asyncio.ensure_future(refresh_status()), once=True)

    # ── Main content area ─────────────────────────────────────────────────────
    with ui.column().style(
        f"flex:1;background:{BG};min-height:100vh;padding:0"
    ):

        # ════════════════════════════════════════════════════════════════════
        # TAB: CHAT
        # ════════════════════════════════════════════════════════════════════
        chat_panel = ui.column().style("width:100%;height:100vh;display:flex;flex-direction:column")
        tab_panels_ref["chat"] = chat_panel

        with chat_panel:
            # Header
            with ui.row().style(
                f"padding:12px 18px;border-bottom:1px solid {BORDER};"
                f"background:{BG2};align-items:center;width:100%"
            ):
                ui.label("Chat").style(f"font-size:14px;font-weight:600;color:{TEXT}")
                ui.space()
                ui.button("Clear", on_click=lambda: (
                    chat_container.clear(),
                    ui.update()
                )).props("flat no-caps").style(f"color:{TEXT2};font-size:12px")

            # Messages
            chat_container = ui.column().style(
                "flex:1;overflow-y:auto;padding:20px;gap:8px;width:100%;"
                "min-height:0;max-height:calc(100vh - 140px)"
            )

            with chat_container:
                with ui.column().style(f"max-width:680px"):
                    ui.html(
                        f'<div class="bubble-aria">'
                        f'Hi! I\'m <strong>ARIA</strong>. Ask me anything — '
                        f'I\'ll search my knowledge base and answer in markdown.'
                        f'</div>'
                    )

            # Input row
            with ui.row().style(
                f"padding:14px 18px;background:{BG2};border-top:1px solid {BORDER};"
                "align-items:flex-end;width:100%;gap:8px"
            ):
                chat_input = ui.textarea(placeholder="Ask anything...").style(
                    f"flex:1;background:{BG3};border:1px solid {BORDER};"
                    "border-radius:10px;color:{TEXT};font-size:13px;"
                    "min-height:42px;max-height:110px"
                ).props("autogrow outlined dense")

                async def do_send():
                    text = chat_input.value.strip()
                    if not text:
                        return
                    chat_input.value = ""

                    # User bubble
                    with chat_container:
                        with ui.column().style("align-items:flex-end;width:100%"):
                            ui.html(f'<div class="bubble-user">{text}</div>')

                    # ARIA thinking
                    with chat_container:
                        thinking = ui.spinner(size="sm", color=ACCENT[1:])
                        status_el = ui.label("Searching memory...").style(
                            f"font-size:11px;color:{TEXT2};font-style:italic"
                        )

                    chat_container.scroll_to(percent=100)

                    # Stream response
                    full_text  = ""
                    mode_used  = "cot"
                    aria_html  = None

                    import re as _re
                    def _md(t):
                        t = _re.sub(r'```[\w]*\n([\s\S]*?)```', r'<pre><code>\1</code></pre>', t)
                        t = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
                        t = _re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         t)
                        t = _re.sub(r'`(.+?)`',       r'<code>\1</code>',     t)
                        return t.replace('\n', '<br>')

                    try:
                        # Use async httpx so NiceGUI event loop is never blocked
                        async with httpx.AsyncClient(timeout=180) as client:
                            async with client.stream(
                                "POST", f"{API}/api/chat/stream",
                                json={"message": text, "city": ""},
                                headers=_auth_headers(),
                            ) as resp:
                                if resp.status_code == 401:
                                    # Token expired — clear and show re-login
                                    state.token = ""
                                    raise Exception("Session expired — please restart the app and re-enter your PIN.")
                                buf = ""
                                async for chunk in resp.aiter_text():
                                    buf += chunk
                                    # Process all complete SSE lines in buffer
                                    while "\n" in buf:
                                        line, buf = buf.split("\n", 1)
                                        if not line.startswith("data: "):
                                            continue
                                        try:
                                            d = json.loads(line[6:])
                                        except Exception:
                                            continue
                                        if d.get("type") == "status":
                                            status_el.text = d.get("text", "")
                                        elif d.get("type") == "mode":
                                            mode_used = d.get("mode", "cot")
                                        elif d.get("type") == "token":
                                            full_text += d["text"]
                                            if aria_html is None:
                                                thinking.delete()
                                                status_el.delete()
                                                with chat_container:
                                                    with ui.column().style("width:100%"):
                                                        aria_html = ui.html("")
                                                        mode_labels = {
                                                            "fast":   "Fast answer",
                                                            "cot":    "Chain of thought",
                                                            "verify": "Verified reasoning",
                                                        }
                                                        mode_badge = ui.html(
                                                            f'<span class="mode-badge">'
                                                            f'{mode_labels.get(mode_used, mode_used)}</span>'
                                                        )
                                            aria_html.set_content(
                                                f'<div class="bubble-aria">{_md(full_text)}</div>'
                                            )
                                            chat_container.scroll_to(percent=100)

                    except Exception as e:
                        if aria_html is None:
                            thinking.delete()
                            status_el.delete()
                        with chat_container:
                            ui.html(f'<div class="bubble-aria" style="color:{RED}">Error: {e}</div>')

                    chat_container.scroll_to(percent=100)

                send_btn = ui.button("Send", on_click=do_send).style(
                    f"background:{ACCENT};color:white;border-radius:8px;"
                    "height:42px;padding:0 16px;font-size:13px"
                ).props("no-caps")

                # Enter key to send
                ui.keyboard(on_key=lambda e: (
                    asyncio.ensure_future(do_send())
                    if e.key == "Enter" and not e.shift_key else None
                ))

        # ════════════════════════════════════════════════════════════════════
        # TAB: WEB SEARCH
        # ════════════════════════════════════════════════════════════════════
        search_panel = ui.column().style(
            f"width:100%;padding:20px;display:none"
        )
        tab_panels_ref["search"] = search_panel

        with search_panel:
            ui.label("Web Search").style(
                f"font-size:14px;font-weight:600;color:{TEXT};margin-bottom:16px"
            )

            with ui.row().style("gap:8px;align-items:center;margin-bottom:16px"):
                search_input = ui.input(placeholder="Search anything...").style(
                    f"flex:1;background:{BG3}"
                ).props("outlined dense")
                auto_learn   = ui.checkbox("Auto-learn", value=True)
                search_btn   = ui.button("Search").style(
                    f"background:{ACCENT};color:white"
                ).props("no-caps")

            search_answer = ui.html("").style("margin-bottom:12px")
            search_cards  = ui.column()

            async def do_search():
                q = search_input.value.strip()
                if not q:
                    return
                search_cards.clear()
                search_answer.set_content(
                    f'<p style="color:{TEXT2};font-size:13px">Searching...</p>'
                )

                d = api_post("/api/search", {
                    "query": q, "save_to_memory": auto_learn.value
                }, timeout=30)

                if not d:
                    search_answer.set_content(
                        f'<p style="color:{RED}">Search failed — is the server running?</p>'
                    )
                    return

                if d.get("answer"):
                    import re as _re
                    ans = d["answer"]
                    ans = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', ans)
                    ans = ans.replace('\n', '<br>')
                    search_answer.set_content(
                        f'<div style="background:{BG3};border:1px solid {BORDER};"'
                        f'border-radius:10px;padding:14px;font-size:13px;'
                        f'color:{TEXT};line-height:1.6;margin-bottom:12px">'
                        f'{ans}</div>'
                    )
                    if d.get("learned", 0) > 0:
                        search_answer.set_content(
                            search_answer.content +
                            f'<p style="font-size:11px;color:{GREEN};margin-bottom:8px">'
                            f'✓ Learned from {d["learned"]} results</p>'
                        )

                with search_cards:
                    for card in d.get("cards", []):
                        url   = card.get("url","#")
                        title = card.get("title","Result")
                        snip  = card.get("snippet","")[:200]
                        src   = card.get("source","web")
                        ui.html(
                            f'<a href="{url}" target="_blank" class="search-card">'
                            f'<div class="card-title">{title}</div>'
                            f'<div class="card-snippet">{snip}</div>'
                            f'<div class="card-url">'
                            f'<span style="background:rgba(74,158,255,.15);color:#4a9eff;'
                            f'padding:1px 7px;border-radius:20px;font-size:10px">{src}</span>'
                            f' {url[:60]}</div>'
                            f'</a>'
                        )

            search_btn.on_click(do_search)
            search_input.on("keydown.enter", do_search)

        # ════════════════════════════════════════════════════════════════════
        # TAB: UPLOAD
        # ════════════════════════════════════════════════════════════════════
        upload_panel = ui.column().style(f"width:100%;padding:20px;display:none")
        tab_panels_ref["upload"] = upload_panel

        with upload_panel:
            ui.label("Upload Documents").style(
                f"font-size:14px;font-weight:600;color:{TEXT};margin-bottom:16px"
            )

            upload_status = ui.html("")

            async def on_upload(e):
                upload_status.set_content(
                    f'<p style="color:{TEXT2};font-size:12px">Uploading {e.name}...</p>'
                )
                try:
                    import io
                    files    = {"file": (e.name, io.BytesIO(e.content), e.type)}
                    r        = requests.post(
                        f"{API}/api/upload?domain={domain_select.value}",
                        files=files, timeout=120
                    )
                    d        = r.json()
                    if r.ok:
                        upload_status.set_content(
                            f'<div style="background:{BG3};border:1px solid {BORDER};'
                            f'border-radius:8px;padding:12px;margin-top:8px">'
                            f'<p style="color:{GREEN};font-size:13px;font-weight:600">'
                            f'✓ {e.name}</p>'
                            f'<p style="color:{TEXT2};font-size:12px">'
                            f'{d.get("chunks",0)} chunks · {d.get("language","?")} · '
                            f'{d.get("domain","?")}</p>'
                            f'<p style="color:{TEXT2};font-size:12px;margin-top:6px">'
                            f'{(d.get("summary","") or "")[:200]}</p>'
                            f'</div>'
                        )
                    else:
                        upload_status.set_content(
                            f'<p style="color:{RED}">Error: {d.get("detail","Upload failed")}</p>'
                        )
                except Exception as ex:
                    upload_status.set_content(f'<p style="color:{RED}">Error: {ex}</p>')

            ui.upload(
                on_upload=on_upload,
                label="Drop file here or click to browse",
                multiple=False,
            ).props("flat").style(
                f"background:{BG3};border:2px dashed {BORDER};border-radius:10px;"
                "width:100%;padding:20px;margin-bottom:12px"
            )

            with ui.row().style("gap:8px;align-items:center;margin-bottom:12px"):
                domain_select = ui.select(
                    ["general","technology","science","medicine","law","finance","education","news"],
                    value="general", label="Domain"
                ).style(f"min-width:160px;background:{BG3}")

            ui.separator().style(f"background:{BORDER};margin:16px 0")
            ui.label("Or ingest a URL").style(
                f"font-size:13px;font-weight:600;color:{TEXT2};margin-bottom:8px"
            )

            url_status = ui.html("")
            with ui.row().style("gap:8px;align-items:center"):
                url_input = ui.input(placeholder="https://...").style(
                    f"flex:1;background:{BG3}"
                ).props("outlined dense")

                async def do_ingest_url():
                    url = url_input.value.strip()
                    if not url:
                        return
                    url_status.set_content(
                        f'<p style="color:{TEXT2};font-size:12px">Fetching...</p>'
                    )
                    d = api_post("/api/ingest-url", {"url": url, "domain": "general"}, timeout=60)
                    if d:
                        url_status.set_content(
                            f'<p style="color:{GREEN};font-size:12px">'
                            f'✓ {d.get("chunks",0)} chunks stored from URL</p>'
                        )
                    else:
                        url_status.set_content(
                            f'<p style="color:{RED};font-size:12px">Failed to fetch URL</p>'
                        )

                ui.button("Fetch & Learn", on_click=do_ingest_url).style(
                    f"background:{ACCENT};color:white"
                ).props("no-caps")

            url_status

        # ════════════════════════════════════════════════════════════════════
        # TAB: CRAWLER
        # ════════════════════════════════════════════════════════════════════
        crawl_panel = ui.column().style(f"width:100%;padding:20px;display:none")
        tab_panels_ref["crawl"] = crawl_panel

        with crawl_panel:
            ui.label("Web Crawler").style(
                f"font-size:14px;font-weight:600;color:{TEXT};margin-bottom:16px"
            )

            with ui.row().style("gap:8px;align-items:center;margin-bottom:8px"):
                crawl_url_input = ui.input(placeholder="https://docs.python.org").style(
                    f"flex:1;background:{BG3}"
                ).props("outlined dense")
                crawl_pages_input = ui.number(value=15, min=1, max=100, label="Max pages").style(
                    f"width:100px;background:{BG3}"
                ).props("outlined dense")
                crawl_domain_sel = ui.select(
                    ["general","technology","education","science"],
                    value="general", label="Domain"
                ).style(f"min-width:130px;background:{BG3}")
                crawl_btn = ui.button("Start Crawl").style(
                    f"background:{ACCENT};color:white"
                ).props("no-caps")

            ui.label("Tries 3 methods: requests → session → Playwright stealth").style(
                f"font-size:11px;color:{TEXT2};margin-bottom:16px"
            )

            # Stats row
            with ui.row().style("gap:12px;margin-bottom:12px"):
                pages_lbl  = ui.html('<div class="stat-chip"><div class="stat-num" id="cp">0</div><div class="stat-lbl">Pages crawled</div></div>')
                chunks_lbl = ui.html('<div class="stat-chip"><div class="stat-num" style="color:#3dd68c">0</div><div class="stat-lbl">Chunks stored</div></div>')
                errors_lbl = ui.html('<div class="stat-chip"><div class="stat-num" style="color:#f06060">0</div><div class="stat-lbl">Errors</div></div>')

            crawl_progress = ui.linear_progress(value=0, color=ACCENT[1:]).style(
                "margin-bottom:8px"
            )
            crawl_log = ui.html('<div class="crawl-log" id="crawl-log">Waiting to start...</div>')

            pages_done  = [0]
            total_chunks= [0]
            err_count   = [0]
            log_lines   = []

            def add_log(cls, text):
                log_lines.append(f'<div class="log-{cls}">{text}</div>')
                crawl_log.set_content(
                    f'<div class="crawl-log">{"".join(log_lines[-60:])}</div>'
                )

            async def do_crawl():
                url = crawl_url_input.value.strip()
                if not url:
                    return
                pages_done[0] = total_chunks[0] = err_count[0] = 0
                log_lines.clear()
                crawl_progress.value = 0
                max_p = int(crawl_pages_input.value or 15)
                crawl_btn.disable()

                try:
                    r = requests.post(
                        f"{API}/api/crawl/stream",
                        json={
                            "url": url,
                            "max_pages": max_p,
                            "domain": crawl_domain_sel.value,
                            "delay_s": 1.5,
                        },
                        stream=True, timeout=600,
                    )
                    for line in r.iter_lines():
                        if not line:
                            continue
                        s = line.decode() if isinstance(line, bytes) else line
                        if not s.startswith("data: "):
                            continue
                        d = json.loads(s[6:])
                        t = d.get("type","")
                        u = d.get("url","")[:60]

                        if t == "start":
                            add_log("ok", f"Starting crawl: {d.get('url','')} (max {d.get('max_pages',0)} pages)")
                        elif t == "crawling":
                            add_log("", f"→ {u}")
                            pct = d.get("done",0) / max(max_p,1)
                            crawl_progress.value = min(pct, 0.99)
                        elif t == "done":
                            pages_done[0]  = d.get("pages_done",0)
                            total_chunks[0]= d.get("total_chunks",0)
                            pages_lbl.set_content(
                                f'<div class="stat-chip"><div class="stat-num">{pages_done[0]}</div>'
                                f'<div class="stat-lbl">Pages crawled</div></div>'
                            )
                            chunks_lbl.set_content(
                                f'<div class="stat-chip"><div class="stat-num" style="color:#3dd68c">'
                                f'{total_chunks[0]}</div><div class="stat-lbl">Chunks stored</div></div>'
                            )
                            add_log("ok", f"✓ {u} — {d.get('chunks',0)} chunks")
                        elif t == "skip":
                            add_log("skip", f"skip: {u} ({d.get('reason','')})")
                        elif t == "error":
                            err_count[0] += 1
                            errors_lbl.set_content(
                                f'<div class="stat-chip"><div class="stat-num" style="color:#f06060">'
                                f'{err_count[0]}</div><div class="stat-lbl">Errors</div></div>'
                            )
                            add_log("err", f"✗ {u}: {d.get('reason','')}")
                        elif t == "finished":
                            crawl_progress.value = 1.0
                            add_log("ok",
                                f"DONE — {d.get('pages_crawled',0)} pages · "
                                f"{d.get('total_chunks',0)} chunks · {d.get('errors',0)} errors"
                            )
                except Exception as e:
                    add_log("err", f"Connection error: {e}")

                crawl_btn.enable()

            crawl_btn.on_click(do_crawl)

        # ════════════════════════════════════════════════════════════════════
        # TAB: KNOWLEDGE BASE
        # ════════════════════════════════════════════════════════════════════
        kb_panel = ui.column().style(f"width:100%;padding:20px;display:none")
        tab_panels_ref["kb"] = kb_panel

        with kb_panel:
            ui.label("Knowledge Base").style(
                f"font-size:14px;font-weight:600;color:{TEXT};margin-bottom:16px"
            )

            # Stats
            with ui.row().style("gap:12px;margin-bottom:16px"):
                kb_chunks = ui.html('<div class="stat-chip"><div class="stat-num">—</div><div class="stat-lbl">Chunks</div></div>')
                kb_sources_stat = ui.html('<div class="stat-chip"><div class="stat-num">—</div><div class="stat-lbl">Sources</div></div>')

            async def load_kb_stats():
                d = api_get("/api/kb/stats")
                if d:
                    kb_chunks.set_content(
                        f'<div class="stat-chip"><div class="stat-num">{d.get("total_chunks",0)}</div>'
                        f'<div class="stat-lbl">Chunks</div></div>'
                    )
                    kb_sources_stat.set_content(
                        f'<div class="stat-chip"><div class="stat-num">{d.get("ingested_sources",0)}</div>'
                        f'<div class="stat-lbl">Sources</div></div>'
                    )

            ui.button("Refresh stats", on_click=load_kb_stats).props("flat no-caps").style(
                f"color:{TEXT2};font-size:12px;margin-bottom:16px"
            )

            ui.label("Ask your knowledge base").style(
                f"font-size:13px;font-weight:600;color:{TEXT2};margin-bottom:8px"
            )
            kb_answer = ui.html("")
            with ui.row().style("gap:8px"):
                kb_q = ui.input(placeholder="Ask your documents...").style(
                    f"flex:1;background:{BG3}"
                ).props("outlined dense")

                async def do_kb_ask():
                    q = kb_q.value.strip()
                    if not q:
                        return
                    kb_answer.set_content(
                        f'<p style="color:{TEXT2};font-size:12px">Searching...</p>'
                    )
                    d = api_post("/api/kb/ask", {"question": q}, timeout=60)
                    if d:
                        import re as _re
                        ans = d.get("answer","No answer")
                        ans = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', ans)
                        ans = ans.replace('\n','<br>')
                        kb_answer.set_content(
                            f'<div style="background:{BG3};border:1px solid {BORDER};'
                            f'border-radius:8px;padding:14px;font-size:13px;'
                            f'color:{TEXT};line-height:1.65;margin-top:10px">{ans}</div>'
                        )
                    else:
                        kb_answer.set_content(f'<p style="color:{RED}">Search failed</p>')

                ui.button("Ask", on_click=do_kb_ask).style(
                    f"background:{ACCENT};color:white"
                ).props("no-caps")

            kb_answer
            ui.timer(0.5, load_kb_stats, once=True)

        # ════════════════════════════════════════════════════════════════════
        # REMAINING TABS — Trends, Vision, Security, NOVA (simple panels)
        # ════════════════════════════════════════════════════════════════════

        def simple_tab(tab_id, label, endpoint, input_label, btn_label):
            panel = ui.column().style(f"width:100%;padding:20px;display:none")
            tab_panels_ref[tab_id] = panel
            with panel:
                ui.label(label).style(
                    f"font-size:14px;font-weight:600;color:{TEXT};margin-bottom:16px"
                )
                result_area = ui.html("")
                with ui.row().style("gap:8px;margin-bottom:12px"):
                    inp = ui.input(placeholder=input_label).style(
                        f"flex:1;background:{BG3}"
                    ).props("outlined dense")

                    async def do_action(i=inp, r=result_area, e=endpoint):
                        val = i.value.strip()
                        if not val:
                            return
                        r.set_content(f'<p style="color:{TEXT2};font-size:12px">Loading...</p>')
                        d = api_post(e, {"query": val} if "search" in e else {"question": val})
                        if d:
                            answer = d.get("answer") or str(d)[:600]
                            import re as _re
                            answer = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', answer)
                            answer = answer.replace('\n','<br>')
                            r.set_content(
                                f'<div style="background:{BG3};border:1px solid {BORDER};'
                                f'border-radius:8px;padding:14px;font-size:13px;'
                                f'color:{TEXT};line-height:1.6;margin-top:10px">{answer}</div>'
                            )

                    ui.button(btn_label, on_click=do_action).style(
                        f"background:{ACCENT};color:white"
                    ).props("no-caps")
                result_area
            return panel

        # Trends
        trends_panel = ui.column().style(f"width:100%;padding:20px;display:none")
        tab_panels_ref["trends"] = trends_panel
        with trends_panel:
            ui.label("Trend Intelligence").style(
                f"font-size:14px;font-weight:600;color:{TEXT};margin-bottom:16px"
            )
            trend_result = ui.html("")
            with ui.row().style("gap:8px;margin-bottom:8px"):
                trend_inp = ui.input(value="artificial intelligence",
                                     placeholder="Topic...").style(
                    f"flex:1;background:{BG3}"
                ).props("outlined dense")

                async def do_trend():
                    topic = trend_inp.value.strip() or "artificial intelligence"
                    trend_result.set_content(
                        f'<p style="color:{TEXT2};font-size:12px">Fetching from arXiv, HN, GitHub, Reddit...</p>'
                    )
                    d = api_post("/api/trend/pulse", {"topic": topic}, timeout=40)
                    if not d:
                        trend_result.set_content(f'<p style="color:{RED}">Failed</p>')
                        return
                    html = ""
                    for section, key in [("arXiv Papers","arxiv"),("Hacker News","hackernews"),("GitHub Trending","github_trending")]:
                        items = d.get(key,[])
                        if not items:
                            continue
                        html += f'<p style="font-size:12px;font-weight:600;color:{ACCENT};margin:12px 0 6px">{section}</p>'
                        for item in items[:4]:
                            title = item.get("title") or item.get("name","")
                            url   = item.get("url","#")
                            desc  = (item.get("summary") or item.get("description",""))[:120]
                            html += (
                                f'<a href="{url}" target="_blank" class="search-card">'
                                f'<div class="card-title">{title}</div>'
                                f'<div class="card-snippet">{desc}</div>'
                                f'</a>'
                            )
                    trend_result.set_content(html or "<p>No results</p>")

                ui.button("Fetch Trends", on_click=do_trend).style(
                    f"background:{ACCENT};color:white"
                ).props("no-caps")
            trend_result

        simple_tab("vision",   "Vision OCR",   "/api/vision/ask",    "Paste image URL or describe...", "Analyse")
        simple_tab("security", "Security",     "/api/kb/ask",        "Ask about security...",          "Ask")
        simple_tab("nova",     "NOVA Engine",  "/api/nova/reason",   "Math, logic, or reasoning...",   "Reason")


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/learn")
def learn_page():
    with ui.column().style(f"padding:24px;background:{BG};min-height:100vh;width:100%"):
        with ui.row().style("align-items:center;margin-bottom:20px"):
            ui.label("ARIA — Learning Dashboard").style(
                f"font-size:16px;font-weight:700;color:{TEXT}"
            )
            ui.space()
            ui.link("← Back to ARIA", "/").style(f"color:{TEXT2};font-size:12px")

        d = api_get("/api/learn/stats")
        if d:
            with ui.row().style("gap:12px;margin-bottom:20px"):
                for val, lbl in [
                    (d.get("total_examples",0), "Training examples"),
                    (d.get("learning_events",0),"Learning events"),
                    (d.get("learned_today",0),  "Learned today"),
                ]:
                    ui.html(
                        f'<div class="stat-chip">'
                        f'<div class="stat-num">{val}</div>'
                        f'<div class="stat-lbl">{lbl}</div></div>'
                    )

        # Timeline
        ui.label("Learning timeline").style(
            f"font-size:13px;font-weight:600;color:{TEXT2};margin-bottom:8px"
        )
        tl = api_get("/api/learn/timeline?hours=48")
        if tl and tl.get("events"):
            for ev in tl["events"][:20]:
                color = {
                    "web_search":"#4a9eff","url_ingest":"#3dd68c",
                    "web_crawl":"#f0a040","document_upload":"#7c6af7",
                }.get(ev.get("event_type",""), TEXT2)
                ui.html(
                    f'<div style="display:flex;gap:10px;padding:8px 0;'
                    f'border-bottom:1px solid {BORDER};align-items:flex-start">'
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{color};'
                    f'margin-top:4px;flex-shrink:0"></div>'
                    f'<div>'
                    f'<div style="font-size:13px;color:{TEXT2}">{ev.get("description","")}</div>'
                    f'<div style="font-size:10px;color:{TEXT2};margin-top:2px">'
                    f'{ev.get("ts","")[:16]} · {ev.get("event_type","")}'
                    f'{" · +" + str(ev["chunks_added"]) + " chunks" if ev.get("chunks_added") else ""}'
                    f'</div></div></div>'
                )
        else:
            ui.label("No learning events yet — upload documents or search the web.").style(
                f"color:{TEXT2};font-size:12px"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true",
                        help="Open as native OS window (needs: pip install pywebview)")
    args = parser.parse_args()

    # Try to auto-login on startup using cached PIN
    _auto_login()

    print(f"\n  ARIA UI starting on http://localhost:{APP_PORT}")
    print(f"  Phone/tablet: http://YOUR-PC-IP:{APP_PORT}")
    print(f"  Learning:     http://localhost:{APP_PORT}/learn\n")

    ui.run(
        title="ARIA — Personal AI",
        favicon="🧠",
        port=APP_PORT,
        dark=True,
        reload=False,
        show=True,
        native=args.native,
    )
