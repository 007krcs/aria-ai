# ARIA Frontend Setup

## Step 1 — Install Node.js
Download from https://nodejs.org (v18 or higher)
Verify: `node --version`

## Step 2 — Install dependencies
```bash
cd C:\Users\chand\ai-remo\app
npm install
```

## Step 3 — Run in browser (easiest)
```bash
# Terminal 1: start ARIA server
cd C:\Users\chand\ai-remo
python server.py

# Terminal 2: start React dev server
cd C:\Users\chand\ai-remo\app
npm run dev
```
Open http://localhost:1420 in your browser.

## Step 4 — Run as Tauri desktop app
```bash
# Install Rust first: https://rustup.rs
# Then:
npm install @tauri-apps/cli
npm run tauri dev
```
This opens a native OS window — no browser needed.

## Step 5 — Build installers
```bash
npm run tauri build
```
Creates: `.exe` (Windows), `.dmg` (Mac), `.AppImage` (Linux)

## Step 6 — Install on Android/iOS as PWA
1. Open Chrome on your phone
2. Go to http://YOUR-PC-IP:1420
3. Tap "Add to home screen"
4. Full app — works offline, push notifications

## Access
- Desktop app:     http://localhost:1420
- From phone:      http://192.168.29.228:1420
- API:             http://localhost:8000
- Voice WebSocket: ws://localhost:8000/ws/voice
