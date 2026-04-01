"""
ARIA — Device Control Agent  (v3 — full personal assistant)
=============================================================
Controls ANY connected device as a true personal assistant:

ANDROID (Full control via ADB over WiFi — no root, no app needed):
  • Calls, SMS, WhatsApp, Telegram, Email
  • Media: play/pause, volume, YouTube, Spotify, Netflix
  • System: brightness, Wi-Fi toggle, Bluetooth, DND, flashlight
  • Files: push/pull, clipboard read/write
  • Contacts: search by name, get number
  • Notifications: read and dismiss
  • Screen: on/off, screenshot, record

DESKTOP (Windows / macOS / Linux):
  • Launch any app, type, click, scroll
  • Volume & media control
  • Screenshot & screen OCR
  • System power (sleep, shutdown, restart, lock)
  • Clipboard read/write
  • Spotlight/Alfred/dmenu integration

iOS (Limited — URL schemes + Shortcuts):
  • Deep links to any app
  • Trigger custom Shortcuts automations
  • QR codes for phone-side actions

SMART HOME (via local APIs — optional):
  • Philips Hue lights
  • Home Assistant
  • Generic REST device control

Setup Android (one-time):
  1. Settings → Developer Options → Wireless Debugging → ON
  2. Note the IP:PORT on screen
  3. Run: adb connect YOUR_PHONE_IP:PORT
"""

from __future__ import annotations

import re
import os
import sys
import time
import json
import shutil
import asyncio
import subprocess
import threading
import platform
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()
PLATFORM     = platform.system()

# ─────────────────────────────────────────────────────────────────────────────
# ANDROID AGENT
# ─────────────────────────────────────────────────────────────────────────────

# Expanded app map — common name → Android package
ANDROID_APPS: dict[str, str] = {
    "youtube":      "com.google.android.youtube",
    "whatsapp":     "com.whatsapp",
    "instagram":    "com.instagram.android",
    "spotify":      "com.spotify.music",
    "chrome":       "com.android.chrome",
    "maps":         "com.google.android.apps.maps",
    "gmail":        "com.google.android.gm",
    "camera":       "com.android.camera2",
    "settings":     "com.android.settings",
    "calculator":   "com.android.calculator2",
    "photos":       "com.google.android.apps.photos",
    "twitter":      "com.twitter.android",
    "x":            "com.twitter.android",
    "telegram":     "org.telegram.messenger",
    "netflix":      "com.netflix.mediaclient",
    "amazon":       "com.amazon.mShop.android.shopping",
    "prime":        "com.amazon.avod.thirdpartyclient",
    "facebook":     "com.facebook.katana",
    "messenger":    "com.facebook.orca",
    "snapchat":     "com.snapchat.android",
    "tiktok":       "com.zhiliaoapp.musically",
    "zoom":         "us.zoom.videomeetings",
    "meet":         "com.google.android.apps.meetings",
    "teams":        "com.microsoft.teams",
    "discord":      "com.discord",
    "reddit":       "com.reddit.frontpage",
    "linkedin":     "com.linkedin.android",
    "uber":         "com.ubercabs.android",
    "ola":          "com.olacabs.customer",
    "swiggy":       "in.swiggy.android",
    "zomato":       "com.application.zomato",
    "gpay":         "com.google.android.apps.nbu.paisa.user",
    "phonepe":      "com.phonepe.app",
    "paytm":        "net.one97.paytm",
    "clock":        "com.google.android.deskclock",
    "calendar":     "com.google.android.calendar",
    "contacts":     "com.google.android.contacts",
    "messages":     "com.google.android.apps.messaging",
    "phone":        "com.google.android.dialer",
    "files":        "com.google.android.documentsui",
    "drive":        "com.google.android.apps.docs",
    "docs":         "com.google.android.apps.docs.editors.docs",
    "sheets":       "com.google.android.apps.docs.editors.sheets",
    "translate":    "com.google.android.apps.translate",
    "news":         "com.google.android.apps.magazines",
    "gallery":      "com.google.android.apps.photos",
    "music":        "com.google.android.music",
    "podcast":      "com.google.android.apps.podcasts",
    "assistant":    "com.google.android.googlequicksearchbox",
    "chrome beta":  "com.chrome.beta",
    "firefox":      "org.mozilla.firefox",
    "brave":        "com.brave.browser",
    "vlc":          "org.videolan.vlc",
    "mx player":    "com.mxtech.videoplayer.ad",
    "hotstar":      "in.startv.hotstar",
    "jio cinema":   "com.jio.media.ondemand",
    "amazon music": "com.amazon.mp3",
    "gaana":        "com.gaana",
    "wynk":         "com.bsbportal.music",
    "phone manager":"com.miui.securitycenter",
    "mi home":      "com.xiaomi.smarthome",
    "bixby":        "com.samsung.android.bixby.agent",
    "samsung pay":  "com.samsung.android.spay",
}


class AndroidAgent:
    """
    Controls Android devices via ADB over WiFi.
    Full personal-assistant capabilities — no root, no app install needed.
    """

    def __init__(self):
        self._adb              = shutil.which("adb") or "adb"
        self._connected: list  = []
        self._keepalive_active = False
        self._keepalive_target = None

    # ── Availability ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            r = subprocess.run([self._adb, "version"],
                               capture_output=True, timeout=3)
            return r.returncode == 0
        except Exception:
            return False

    # ── Connection ──────────────────────────────────────────────────────────────

    def connect(self, ip: str, port: int = 5555) -> dict:
        try:
            r = subprocess.run(
                [self._adb, "connect", f"{ip}:{port}"],
                capture_output=True, text=True, timeout=10,
            )
            success = "connected" in r.stdout.lower()
            if success:
                target = f"{ip}:{port}"
                if target not in self._connected:
                    self._connected.append(target)
                console.print(f"  [green]Android connected:[/] {ip}:{port}")
                # Auto-start keep-alive
                self.start_keepalive(ip, port)
            return {"success": success, "output": r.stdout.strip()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_devices(self) -> list[str]:
        try:
            r = subprocess.run([self._adb, "devices"],
                               capture_output=True, text=True, timeout=5)
            lines   = r.stdout.strip().split("\n")[1:]
            return [l.split("\t")[0] for l in lines if "\tdevice" in l]
        except Exception:
            return []

    def _adb_cmd(self, *args, device: str = None, timeout: int = 30) -> dict:
        cmd = [self._adb]
        if device:
            cmd += ["-s", device]
        cmd += list(args)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return {
                "success": r.returncode == 0,
                "output":  r.stdout.strip(),
                "error":   r.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "ADB timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _first_device(self) -> Optional[str]:
        devices = self.list_devices()
        return devices[0] if devices else None

    # ── Communication ───────────────────────────────────────────────────────────

    def make_call(self, phone_number: str, device: str = None) -> dict:
        number = re.sub(r"[^\d+]", "", phone_number)
        if not number:
            return {"success": False, "error": "Invalid phone number"}
        result = self._adb_cmd(
            "shell", "am", "start",
            "-a", "android.intent.action.CALL",
            "-d", f"tel:{number}",
            device=device or self._first_device(),
        )
        console.print(f"  [green]Calling:[/] {number}")
        return result

    def end_call(self, device: str = None) -> dict:
        """End the current call."""
        return self._adb_cmd(
            "shell", "input", "keyevent", "6",  # KEYCODE_ENDCALL
            device=device or self._first_device(),
        )

    def send_sms(self, number: str, message: str, device: str = None) -> dict:
        number = re.sub(r"[^\d+]", "", number)
        return self._adb_cmd(
            "shell", "am", "start",
            "-a", "android.intent.action.SENDTO",
            "-d", f"smsto:{number}",
            "--es", "sms_body", message,
            device=device or self._first_device(),
        )

    def send_whatsapp(self, contact_number: str, message: str = "",
                      device: str = None) -> dict:
        number = re.sub(r"[^\d+]", "", contact_number)
        url    = f"https://wa.me/{number}?text={urllib.parse.quote(message)}"
        return self._adb_cmd(
            "shell", "am", "start", "-a", "android.intent.action.VIEW",
            "-d", url, device=device or self._first_device(),
        )

    def send_telegram(self, username: str, message: str = "",
                      device: str = None) -> dict:
        """Open Telegram chat with a username."""
        url = f"tg://resolve?domain={username}&text={urllib.parse.quote(message)}"
        return self._adb_cmd(
            "shell", "am", "start", "-a", "android.intent.action.VIEW",
            "-d", url, device=device or self._first_device(),
        )

    def compose_email(self, to: str, subject: str = "", body: str = "",
                      device: str = None) -> dict:
        """Open email compose screen."""
        url = f"mailto:{to}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
        return self._adb_cmd(
            "shell", "am", "start", "-a", "android.intent.action.SENDTO",
            "-d", url, device=device or self._first_device(),
        )

    # ── Contacts ────────────────────────────────────────────────────────────────

    def get_contacts(self, device: str = None) -> list[dict]:
        """Read contacts from phone (name + number)."""
        result = self._adb_cmd(
            "shell", "content", "query",
            "--uri", "content://contacts/phones/",
            "--projection", "display_name:number",
            device=device or self._first_device(),
        )
        contacts = []
        if result["success"]:
            for line in result["output"].split("\n"):
                if "display_name=" in line and "number=" in line:
                    name_m   = re.search(r"display_name=([^,]+)", line)
                    number_m = re.search(r"number=([^,\s]+)", line)
                    if name_m and number_m:
                        contacts.append({
                            "name":   name_m.group(1).strip(),
                            "number": number_m.group(1).strip(),
                        })
        return contacts

    def find_contact(self, name: str, device: str = None) -> Optional[str]:
        """Find a contact's number by name (fuzzy match)."""
        contacts = self.get_contacts(device)
        name_lower = name.lower()
        # Exact match first
        for c in contacts:
            if c["name"].lower() == name_lower:
                return c["number"]
        # Partial match
        for c in contacts:
            if name_lower in c["name"].lower():
                return c["number"]
        return None

    def call_contact(self, contact_name: str, device: str = None) -> dict:
        """Call a contact by name — auto-resolves number."""
        number = self.find_contact(contact_name, device)
        if not number:
            return {"success": False, "error": f"Contact '{contact_name}' not found"}
        return self.make_call(number, device)

    # ── Apps ────────────────────────────────────────────────────────────────────

    def open_app(self, app_name: str, device: str = None) -> dict:
        pkg = ANDROID_APPS.get(app_name.lower().strip(), app_name)
        result = self._adb_cmd(
            "shell", "monkey",
            "-p", pkg, "-c", "android.intent.category.LAUNCHER", "1",
            device=device or self._first_device(),
        )
        if not result["success"]:
            # Try as activity launch
            result = self._adb_cmd(
                "shell", "am", "start", "-n", pkg,
                device=device or self._first_device(),
            )
        console.print(f"  [green]Opened:[/] {app_name} ({pkg})")
        return result

    def close_app(self, app_name: str, device: str = None) -> dict:
        """Force-stop an app."""
        pkg = ANDROID_APPS.get(app_name.lower().strip(), app_name)
        return self._adb_cmd(
            "shell", "am", "force-stop", pkg,
            device=device or self._first_device(),
        )

    def list_installed_apps(self, device: str = None) -> list[str]:
        result = self._adb_cmd(
            "shell", "pm", "list", "packages", "-3",  # third-party apps only
            device=device or self._first_device(),
        )
        if result["success"]:
            return [l.replace("package:", "").strip()
                    for l in result["output"].split("\n") if l.strip()]
        return []

    # ── Media ────────────────────────────────────────────────────────────────────

    def play_youtube(self, query: str, device: str = None) -> dict:
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        return self._adb_cmd(
            "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url,
            device=device or self._first_device(),
        )

    def play_spotify(self, query: str, device: str = None) -> dict:
        url = f"spotify:search:{urllib.parse.quote(query)}"
        return self._adb_cmd(
            "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url,
            device=device or self._first_device(),
        )

    def media_play_pause(self, device: str = None) -> dict:
        """Toggle play/pause for any media player."""
        return self._adb_cmd(
            "shell", "input", "keyevent", "85",  # KEYCODE_MEDIA_PLAY_PAUSE
            device=device or self._first_device(),
        )

    def media_next(self, device: str = None) -> dict:
        return self._adb_cmd(
            "shell", "input", "keyevent", "87",  # KEYCODE_MEDIA_NEXT
            device=device or self._first_device(),
        )

    def media_previous(self, device: str = None) -> dict:
        return self._adb_cmd(
            "shell", "input", "keyevent", "88",  # KEYCODE_MEDIA_PREVIOUS
            device=device or self._first_device(),
        )

    def volume_up(self, steps: int = 1, device: str = None) -> dict:
        for _ in range(steps):
            self._adb_cmd("shell", "input", "keyevent", "24",
                          device=device or self._first_device())
        return {"success": True, "action": f"volume up x{steps}"}

    def volume_down(self, steps: int = 1, device: str = None) -> dict:
        for _ in range(steps):
            self._adb_cmd("shell", "input", "keyevent", "25",
                          device=device or self._first_device())
        return {"success": True, "action": f"volume down x{steps}"}

    def volume_mute(self, device: str = None) -> dict:
        return self._adb_cmd(
            "shell", "input", "keyevent", "164",  # KEYCODE_VOLUME_MUTE
            device=device or self._first_device(),
        )

    def set_volume(self, level: int, stream: int = 3, device: str = None) -> dict:
        """Set volume to absolute level (0–15). stream=3 is media."""
        return self._adb_cmd(
            "shell", "media", "volume",
            "--stream", str(stream), "--set", str(max(0, min(15, level))),
            device=device or self._first_device(),
        )

    # ── System controls ─────────────────────────────────────────────────────────

    def set_brightness(self, level: int, device: str = None) -> dict:
        """Set screen brightness 0–255. Also disables auto-brightness."""
        dev = device or self._first_device()
        # Disable auto-brightness
        self._adb_cmd("shell", "settings", "put", "system",
                      "screen_brightness_mode", "0", device=dev)
        return self._adb_cmd(
            "shell", "settings", "put", "system",
            "screen_brightness", str(max(0, min(255, level))),
            device=dev,
        )

    def screen_on(self, device: str = None) -> dict:
        """Wake up screen."""
        return self._adb_cmd(
            "shell", "input", "keyevent", "224",  # KEYCODE_WAKEUP
            device=device or self._first_device(),
        )

    def screen_off(self, device: str = None) -> dict:
        """Turn off screen."""
        return self._adb_cmd(
            "shell", "input", "keyevent", "223",  # KEYCODE_SLEEP
            device=device or self._first_device(),
        )

    def toggle_wifi(self, on: bool, device: str = None) -> dict:
        state = "enable" if on else "disable"
        return self._adb_cmd(
            "shell", "svc", "wifi", state,
            device=device or self._first_device(),
        )

    def toggle_bluetooth(self, on: bool, device: str = None) -> dict:
        state = "enable" if on else "disable"
        return self._adb_cmd(
            "shell", "svc", "bluetooth", state,
            device=device or self._first_device(),
        )

    def toggle_flashlight(self, on: bool, device: str = None) -> dict:
        """Toggle flashlight via camera2 API intent."""
        action = "1" if on else "0"
        return self._adb_cmd(
            "shell", "settings", "put", "global", "torch_power_button_gesture", action,
            device=device or self._first_device(),
        )

    def toggle_dnd(self, on: bool, device: str = None) -> dict:
        """Enable/disable Do Not Disturb."""
        mode = "2" if on else "1"  # 2=priority, 1=off
        return self._adb_cmd(
            "shell", "cmd", "notification", "set_dnd", mode,
            device=device or self._first_device(),
        )

    def set_alarm(self, hour: int, minute: int = 0,
                  label: str = "ARIA Alarm", device: str = None) -> dict:
        result = self._adb_cmd(
            "shell", "am", "start",
            "-a", "android.intent.action.SET_ALARM",
            "--ei", "android.intent.extra.alarm.HOUR", str(hour),
            "--ei", "android.intent.extra.alarm.MINUTES", str(minute),
            "--es", "android.intent.extra.alarm.MESSAGE", label,
            "--ez", "android.intent.extra.alarm.SKIP_UI", "true",
            device=device or self._first_device(),
        )
        console.print(f"  [green]Alarm set:[/] {hour:02d}:{minute:02d} — {label}")
        return result

    def set_timer(self, seconds: int, label: str = "ARIA Timer",
                  device: str = None) -> dict:
        return self._adb_cmd(
            "shell", "am", "start",
            "-a", "android.intent.action.SET_TIMER",
            "--ei", "android.intent.extra.alarm.LENGTH", str(seconds),
            "--es", "android.intent.extra.alarm.MESSAGE", label,
            "--ez", "android.intent.extra.alarm.SKIP_UI", "true",
            device=device or self._first_device(),
        )

    def open_maps(self, query: str, device: str = None) -> dict:
        url = f"geo:0,0?q={urllib.parse.quote(query)}"
        return self._adb_cmd(
            "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url,
            device=device or self._first_device(),
        )

    def navigate_to(self, destination: str, device: str = None) -> dict:
        """Start navigation in Google Maps."""
        url = f"google.navigation:q={urllib.parse.quote(destination)}"
        return self._adb_cmd(
            "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url,
            device=device or self._first_device(),
        )

    # ── Clipboard ───────────────────────────────────────────────────────────────

    def set_clipboard(self, text: str, device: str = None) -> dict:
        """Set clipboard content on Android."""
        # Requires API 29+ — uses content provider
        escaped = text.replace("'", "\\'")
        return self._adb_cmd(
            "shell", f"am start -a android.intent.action.MAIN --es extra_text '{escaped}'",
            device=device or self._first_device(),
        )

    def type_text(self, text: str, device: str = None) -> dict:
        """Type text on Android (no special chars)."""
        escaped = text.replace(" ", "%s").replace("'", "")
        return self._adb_cmd(
            "shell", "input", "text", escaped,
            device=device or self._first_device(),
        )

    def back(self, device: str = None) -> dict:
        return self._adb_cmd(
            "shell", "input", "keyevent", "4",  # KEYCODE_BACK
            device=device or self._first_device(),
        )

    def home(self, device: str = None) -> dict:
        return self._adb_cmd(
            "shell", "input", "keyevent", "3",  # KEYCODE_HOME
            device=device or self._first_device(),
        )

    def recent_apps(self, device: str = None) -> dict:
        return self._adb_cmd(
            "shell", "input", "keyevent", "187",  # KEYCODE_APP_SWITCH
            device=device or self._first_device(),
        )

    # ── Screenshot / Screen content ─────────────────────────────────────────────

    def take_screenshot(self, device: str = None, save_path: str = None) -> dict:
        remote = "/sdcard/aria_screenshot.png"
        local  = save_path or str(
            PROJECT_ROOT / "data" / "screenshots" /
            f"android_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        dev = device or self._first_device()
        self._adb_cmd("shell", "screencap", "-p", remote, device=dev)
        result = self._adb_cmd("pull", remote, local, device=dev)
        self._adb_cmd("shell", "rm", remote, device=dev)
        result["local_path"] = local
        return result

    def read_screen_text(self, device: str = None) -> str:
        """Read all visible text on screen via UI Automator dump."""
        dev     = device or self._first_device()
        remote  = "/sdcard/aria_uidump.xml"
        local   = str(PROJECT_ROOT / "data" / "screenshots" / "uidump.xml")
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self._adb_cmd("shell", "uiautomator", "dump", remote, device=dev)
        self._adb_cmd("pull", remote, local, device=dev)
        self._adb_cmd("shell", "rm", remote, device=dev)
        try:
            with open(local) as f:
                content = f.read()
            # Extract text attributes
            texts = re.findall(r'text="([^"]+)"', content)
            return " | ".join(t for t in texts if t.strip())
        except Exception:
            return ""

    # ── Notifications ───────────────────────────────────────────────────────────

    def read_notifications(self, device: str = None) -> list[dict]:
        result = self._adb_cmd(
            "shell", "dumpsys", "notification",
            device=device or self._first_device(),
        )
        notifications = []
        if result["success"]:
            lines   = result["output"].split("\n")
            current = {}
            for line in lines:
                if "pkg=" in line:
                    if current and "app" in current:
                        notifications.append(current)
                    current = {}
                    m = re.search(r"pkg=(\S+)", line)
                    if m:
                        current["app"] = m.group(1)
                elif "android.title" in line:
                    m = re.search(r'android.title=String\s+"([^"]+)"', line)
                    if not m:
                        m = re.search(r"android.title=(.+?)(?:\s+android|\s*$)", line)
                    if m:
                        current["title"] = m.group(1).strip()
                elif "android.text" in line:
                    m = re.search(r'android.text=String\s+"([^"]+)"', line)
                    if not m:
                        m = re.search(r"android.text=(.+?)(?:\s+android|\s*$)", line)
                    if m:
                        current["text"] = m.group(1).strip()
            if current and "app" in current:
                notifications.append(current)
        return notifications[:25]

    def clear_notifications(self, device: str = None) -> dict:
        """Dismiss all notifications."""
        return self._adb_cmd(
            "shell", "service", "call", "notification", "1",
            device=device or self._first_device(),
        )

    # ── Battery & Device info ────────────────────────────────────────────────────

    def get_battery(self, device: str = None) -> dict:
        result = self._adb_cmd(
            "shell", "dumpsys", "battery",
            device=device or self._first_device(),
        )
        info = {}
        if result["success"]:
            for line in result["output"].split("\n"):
                if "level:" in line:
                    m = re.search(r"level:\s+(\d+)", line)
                    if m:
                        info["level"] = int(m.group(1))
                elif "status:" in line:
                    m = re.search(r"status:\s+(\d+)", line)
                    if m:
                        status_map = {1:"unknown", 2:"charging", 3:"discharging",
                                      4:"not charging", 5:"full"}
                        info["status"] = status_map.get(int(m.group(1)), "unknown")
                elif "temperature:" in line:
                    m = re.search(r"temperature:\s+(\d+)", line)
                    if m:
                        info["temperature_c"] = int(m.group(1)) / 10
        return info

    def get_device_info(self, device: str = None) -> dict:
        """Get device model, Android version, and stats."""
        dev = device or self._first_device()
        model   = self._adb_cmd("shell", "getprop", "ro.product.model", device=dev)
        android = self._adb_cmd("shell", "getprop", "ro.build.version.release", device=dev)
        return {
            "model":   model["output"].strip(),
            "android": android["output"].strip(),
            "battery": self.get_battery(dev),
        }

    # ── File transfer ────────────────────────────────────────────────────────────

    def push_file(self, local_path: str, remote_path: str = "/sdcard/",
                  device: str = None) -> dict:
        return self._adb_cmd(
            "push", local_path, remote_path,
            device=device or self._first_device(), timeout=120,
        )

    def pull_file(self, remote_path: str, local_path: str = None,
                  device: str = None) -> dict:
        local = local_path or str(PROJECT_ROOT / "data" / "downloads" /
                                   Path(remote_path).name)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        return self._adb_cmd(
            "pull", remote_path, local,
            device=device or self._first_device(), timeout=120,
        )

    # ── Keep-alive ───────────────────────────────────────────────────────────────

    def start_keepalive(self, ip: str, port: int = 5555, interval_s: int = 25):
        if self._keepalive_active:
            return
        self._keepalive_active = True
        self._keepalive_target = (ip, port)

        def loop():
            while self._keepalive_active:
                time.sleep(interval_s)
                devices = self.list_devices()
                target  = f"{ip}:{port}"
                if target not in devices:
                    self.connect(ip, port)

        threading.Thread(target=loop, daemon=True, name="adb-keepalive").start()
        console.print(f"  [dim]ADB keep-alive active for {ip}:{port}[/]")

    def stop_keepalive(self):
        self._keepalive_active = False

    def status(self) -> dict:
        devices  = self.list_devices()
        ka       = self._keepalive_target
        return {
            "adb_available":      self.is_available(),
            "connected_devices":  devices,
            "keepalive_target":   f"{ka[0]}:{ka[1]}" if ka else None,
            "keepalive_active":   self._keepalive_active,
            "supported_actions": [
                "call", "sms", "whatsapp", "telegram", "email",
                "open_app", "close_app", "media_control",
                "volume", "brightness", "wifi", "bluetooth",
                "flashlight", "dnd", "alarm", "timer",
                "screenshot", "read_screen", "notifications",
                "battery", "push_file", "pull_file",
                "navigate", "maps", "contacts",
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# DESKTOP AGENT
# ─────────────────────────────────────────────────────────────────────────────

class DesktopAgent:
    """
    Full desktop control: Windows / macOS / Linux.
    """

    # ── App launching ───────────────────────────────────────────────────────────

    def open_app(self, app_name: str) -> dict:
        try:
            if PLATFORM == "Windows":
                return self._open_windows(app_name)
            elif PLATFORM == "Darwin":
                return self._open_mac(app_name)
            else:
                return self._open_linux(app_name)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _open_windows(self, app: str) -> dict:
        home = Path.home()
        known = {
            "chrome":        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "firefox":       r"C:\Program Files\Mozilla Firefox\firefox.exe",
            "edge":          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "notepad":       "notepad.exe",
            "calculator":    "calc.exe",
            "file explorer": "explorer.exe",
            "word":          r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
            "excel":         r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
            "powerpoint":    r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
            "outlook":       r"C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE",
            "spotify":       str(home / "AppData/Roaming/Spotify/Spotify.exe"),
            "vs code":       str(home / "AppData/Local/Programs/Microsoft VS Code/Code.exe"),
            "terminal":      "wt.exe",
            "cmd":           "cmd.exe",
            "powershell":    "powershell.exe",
            "paint":         "mspaint.exe",
            "snipping tool": "SnippingTool.exe",
            "task manager":  "taskmgr.exe",
            "control panel": "control.exe",
            "device manager":"devmgmt.msc",
            "discord":       str(home / "AppData/Local/Discord/app-*/Discord.exe"),
            "telegram":      str(home / "AppData/Roaming/Telegram Desktop/Telegram.exe"),
            "zoom":          str(home / "AppData/Roaming/Zoom/bin/Zoom.exe"),
            "vlc":           r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            "steam":         r"C:\Program Files (x86)\Steam\Steam.exe",
        }
        app_lower = app.lower().strip()
        cmd = known.get(app_lower, app)
        try:
            subprocess.Popen([cmd], shell=True)
            console.print(f"  [green]Opened:[/] {app}")
            return {"success": True, "app": app}
        except Exception as e:
            return {"success": False, "error": str(e), "tried": cmd}

    def _open_mac(self, app: str) -> dict:
        r = subprocess.run(["open", "-a", app], capture_output=True, text=True, timeout=10)
        return {"success": r.returncode == 0, "error": r.stderr.strip()}

    def _open_linux(self, app: str) -> dict:
        try:
            subprocess.Popen([app.lower()])
            return {"success": True}
        except Exception as e:
            # Try xdg-open
            try:
                subprocess.Popen(["xdg-open", app])
                return {"success": True}
            except Exception:
                return {"success": False, "error": str(e)}

    # ── Browser actions ─────────────────────────────────────────────────────────

    def open_url(self, url: str) -> dict:
        import webbrowser
        webbrowser.open(url)
        return {"success": True, "url": url}

    def open_youtube(self, query: str) -> dict:
        return self.open_url(
            f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        )

    def open_google(self, query: str) -> dict:
        return self.open_url(
            f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        )

    def open_spotify_web(self, query: str) -> dict:
        return self.open_url(
            f"https://open.spotify.com/search/{urllib.parse.quote(query)}"
        )

    def open_maps(self, query: str) -> dict:
        return self.open_url(
            f"https://www.google.com/maps/search/{urllib.parse.quote(query)}"
        )

    # ── Volume & media ──────────────────────────────────────────────────────────

    def volume_up(self, steps: int = 1) -> dict:
        if PLATFORM == "Windows":
            for _ in range(steps):
                subprocess.run(
                    ["powershell", "-c",
                     "(New-Object -com WScript.Shell).SendKeys([char]175)"],
                    capture_output=True, timeout=5,
                )
        elif PLATFORM == "Darwin":
            for _ in range(steps):
                subprocess.run(
                    ["osascript", "-e",
                     "set volume output volume (output volume of (get volume settings) + 10)"],
                    capture_output=True, timeout=5,
                )
        else:
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"],
                           capture_output=True, timeout=5)
        return {"success": True, "action": f"volume up x{steps}"}

    def volume_down(self, steps: int = 1) -> dict:
        if PLATFORM == "Windows":
            for _ in range(steps):
                subprocess.run(
                    ["powershell", "-c",
                     "(New-Object -com WScript.Shell).SendKeys([char]174)"],
                    capture_output=True, timeout=5,
                )
        elif PLATFORM == "Darwin":
            for _ in range(steps):
                subprocess.run(
                    ["osascript", "-e",
                     "set volume output volume (output volume of (get volume settings) - 10)"],
                    capture_output=True, timeout=5,
                )
        else:
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"],
                           capture_output=True, timeout=5)
        return {"success": True, "action": f"volume down x{steps}"}

    def volume_mute(self) -> dict:
        if PLATFORM == "Windows":
            subprocess.run(
                ["powershell", "-c",
                 "(New-Object -com WScript.Shell).SendKeys([char]173)"],
                capture_output=True, timeout=5,
            )
        elif PLATFORM == "Darwin":
            subprocess.run(
                ["osascript", "-e", "set volume output muted true"],
                capture_output=True, timeout=5,
            )
        else:
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                           capture_output=True, timeout=5)
        return {"success": True}

    def media_play_pause(self) -> dict:
        if PLATFORM == "Windows":
            subprocess.run(
                ["powershell", "-c",
                 "(New-Object -com WScript.Shell).SendKeys([char]179)"],
                capture_output=True, timeout=5,
            )
        elif PLATFORM == "Darwin":
            subprocess.run(
                ["osascript", "-e", "tell application \"System Events\" to key code 49"],
                capture_output=True, timeout=5,
            )
        else:
            subprocess.run(["playerctl", "play-pause"], capture_output=True, timeout=5)
        return {"success": True}

    # ── System ──────────────────────────────────────────────────────────────────

    def screenshot(self, save_path: str = None) -> dict:
        try:
            import pyautogui
            path = save_path or str(
                PROJECT_ROOT / "data" / "screenshots" /
                f"desktop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            pyautogui.screenshot(path)
            return {"success": True, "path": path}
        except ImportError:
            # Fallback to OS screenshot tool
            return self._os_screenshot(save_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _os_screenshot(self, path: str = None) -> dict:
        path = path or str(
            PROJECT_ROOT / "data" / "screenshots" /
            f"desktop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if PLATFORM == "Windows":
            subprocess.run(
                ["powershell", "-c",
                 f"Add-Type -Assembly 'System.Windows.Forms';"
                 f"[System.Windows.Forms.Screen]::PrimaryScreen | "
                 f"ForEach-Object {{ $bmp=[System.Drawing.Bitmap]::new($_.Bounds.Width,$_.Bounds.Height);"
                 f"$g=[System.Drawing.Graphics]::FromImage($bmp);"
                 f"$g.CopyFromScreen(0,0,0,0,$_.Bounds.Size);"
                 f"$bmp.Save('{path}') }}"],
                capture_output=True, timeout=10,
            )
        elif PLATFORM == "Darwin":
            subprocess.run(["screencapture", path], capture_output=True, timeout=10)
        else:
            subprocess.run(["scrot", path], capture_output=True, timeout=10)
        return {"success": True, "path": path}

    def lock_screen(self) -> dict:
        if PLATFORM == "Windows":
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"],
                           capture_output=True, timeout=5)
        elif PLATFORM == "Darwin":
            subprocess.run(
                ["pmset", "displaysleepnow"], capture_output=True, timeout=5
            )
        else:
            subprocess.run(["xdg-screensaver", "lock"], capture_output=True, timeout=5)
        return {"success": True}

    def sleep(self) -> dict:
        if PLATFORM == "Windows":
            subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
        elif PLATFORM == "Darwin":
            subprocess.run(["pmset", "sleepnow"], capture_output=True)
        else:
            subprocess.run(["systemctl", "suspend"], capture_output=True)
        return {"success": True}

    def type_text(self, text: str) -> dict:
        try:
            import pyautogui
            pyautogui.write(text, interval=0.03)
            return {"success": True}
        except ImportError:
            return {"success": False, "error": "pip install pyautogui"}

    def get_clipboard(self) -> str:
        try:
            import pyperclip
            return pyperclip.paste()
        except ImportError:
            if PLATFORM == "Windows":
                r = subprocess.run(
                    ["powershell", "-c", "Get-Clipboard"],
                    capture_output=True, text=True, timeout=5,
                )
                return r.stdout.strip()
            elif PLATFORM == "Darwin":
                r = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
                return r.stdout.strip()
            return ""

    def set_clipboard(self, text: str) -> dict:
        try:
            import pyperclip
            pyperclip.copy(text)
            return {"success": True}
        except ImportError:
            if PLATFORM == "Windows":
                subprocess.run(
                    f'echo {text} | clip', shell=True, capture_output=True, timeout=5
                )
            elif PLATFORM == "Darwin":
                subprocess.run(["pbcopy"], input=text.encode(), capture_output=True, timeout=5)
            return {"success": True}

    def set_alarm(self, hour: int, minute: int, label: str = "ARIA") -> dict:
        if PLATFORM == "Windows":
            return self._alarm_windows(hour, minute, label)
        elif PLATFORM == "Darwin":
            return self._alarm_mac(hour, minute, label)
        else:
            return self._alarm_linux(hour, minute, label)

    def _alarm_windows(self, hour: int, minute: int, label: str) -> dict:
        trigger_time = f"{hour:02d}:{minute:02d}"
        ps = (
            f"$action = New-ScheduledTaskAction -Execute 'powershell.exe' "
            f"-Argument '-WindowStyle Hidden -Command "
            f"\"[System.Media.SystemSounds]::Beep.Play();"
            f"Add-Type -AssemblyName PresentationFramework;"
            f"[System.Windows.MessageBox]::Show(\\'{label}\\')\"; ';"
            f"$trigger = New-ScheduledTaskTrigger -Once -At '{trigger_time}';"
            f"Register-ScheduledTask -Action $action -Trigger $trigger "
            f"-TaskName 'ARIA_Alarm_{hour:02d}{minute:02d}' -Force -RunLevel Highest"
        )
        r = subprocess.run(["powershell", "-Command", ps],
                           capture_output=True, text=True, timeout=10)
        console.print(f"  [green]Alarm set:[/] {trigger_time} — {label}")
        return {"success": r.returncode == 0, "time": trigger_time}

    def _alarm_mac(self, hour: int, minute: int, label: str) -> dict:
        at_time = f"{hour:02d}:{minute:02d}"
        cmd = (
            f'echo "osascript -e \'display notification \"{label}\" '
            f'with title \"ARIA Alarm\" sound name \"Ping\"\'" | at {at_time}'
        )
        subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
        return {"success": True, "time": at_time}

    def _alarm_linux(self, hour: int, minute: int, label: str) -> dict:
        at_time = f"{hour:02d}:{minute:02d}"
        cmd = f'echo "notify-send \\"ARIA Alarm\\" \\"{label}\\"" | at {at_time}'
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
        return {"success": r.returncode == 0, "time": at_time}


# ─────────────────────────────────────────────────────────────────────────────
# iOS AGENT
# ─────────────────────────────────────────────────────────────────────────────

class iOSAgent:
    """
    iOS control via URL schemes + Shortcuts.
    No jailbreak needed. Limited but useful.
    """

    URL_SCHEMES = {
        "whatsapp":   "whatsapp://send?phone={contact}&text={message}",
        "facetime":   "facetime://{contact}",
        "facetime_audio": "facetime-audio://{contact}",
        "phone":      "tel://{number}",
        "sms":        "sms://{number}&body={message}",
        "maps":       "maps://q={query}",
        "waze":       "waze://?q={query}&navigate=yes",
        "youtube":    "youtube://www.youtube.com/results?search_query={query}",
        "spotify":    "spotify:search:{query}",
        "mail":       "mailto:{email}?subject={subject}&body={body}",
        "calendar":   "calshow://",
        "clock":      "clock-alarm://",
        "settings":   "app-settings:",
        "safari":     "https://{url}",
        "shortcuts":  "shortcuts://run-shortcut?name={name}&input={input}",
        "translate":  "itms-apps://apps.apple.com/app/id1514844618",
        "appstore":   "itms-apps://search.itunes.apple.com/WebObjects/MZSearch.woa/wa/search?q={query}",
        "instagram":  "instagram://",
        "telegram":   "tg://resolve?domain={username}",
        "twitter":    "twitter://search?query={query}",
        "netflix":    "nflx://www.netflix.com/",
    }

    def __init__(self, device_ip: str = None):
        self.device_ip = device_ip

    def open_url_scheme(self, app: str, **params) -> dict:
        template = self.URL_SCHEMES.get(app.lower())
        if not template:
            return {"success": False, "error": f"No URL scheme for '{app}'"}
        url = template
        for k, v in params.items():
            url = url.replace(f"{{{k}}}", urllib.parse.quote(str(v)))
        return {
            "success":  True,
            "url":      url,
            "method":   "ios_url_scheme",
            "qr":       self._qr_b64(url),
            "note":     "Scan the QR code with your iPhone or tap the link",
        }

    def run_shortcut(self, name: str, input_text: str = "") -> dict:
        url = f"shortcuts://run-shortcut?name={urllib.parse.quote(name)}&input={urllib.parse.quote(input_text)}"
        return {
            "success": True,
            "url":     url,
            "qr":      self._qr_b64(url),
            "note":    f"Scan to run Shortcut: {name}",
        }

    def _qr_b64(self, url: str) -> str:
        try:
            import qrcode, io, base64
            qr  = qrcode.make(url)
            buf = io.BytesIO()
            qr.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            return ""

    def status(self) -> dict:
        return {
            "platform":         "iOS",
            "control_level":    "limited (URL schemes + Shortcuts)",
            "available_apps":   list(self.URL_SCHEMES.keys()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SMART HOME AGENT  (optional)
# ─────────────────────────────────────────────────────────────────────────────

class SmartHomeAgent:
    """
    Controls smart home devices via local APIs.
    Currently supports: Home Assistant, Philips Hue, generic REST.
    All local network — no cloud required.
    """

    def __init__(self, ha_url: str = None, ha_token: str = None,
                 hue_ip: str = None, hue_user: str = None):
        self._ha_url    = ha_url
        self._ha_token  = ha_token
        self._hue_ip    = hue_ip
        self._hue_user  = hue_user

    # ── Home Assistant ──────────────────────────────────────────────────────────

    def ha_call(self, domain: str, service: str, entity_id: str,
                extra: dict = None) -> dict:
        """Call a Home Assistant service."""
        if not self._ha_url or not self._ha_token:
            return {"success": False, "error": "Home Assistant not configured"}
        try:
            import requests
            url  = f"{self._ha_url}/api/services/{domain}/{service}"
            body = {"entity_id": entity_id, **(extra or {})}
            r = requests.post(
                url, json=body,
                headers={"Authorization": f"Bearer {self._ha_token}"},
                timeout=5, verify=False,
            )
            return {"success": r.status_code < 300, "status": r.status_code}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def turn_on(self, entity_id: str) -> dict:
        return self.ha_call("homeassistant", "turn_on", entity_id)

    def turn_off(self, entity_id: str) -> dict:
        return self.ha_call("homeassistant", "turn_off", entity_id)

    def set_light_brightness(self, entity_id: str, brightness: int) -> dict:
        """brightness: 0–255"""
        return self.ha_call("light", "turn_on", entity_id,
                            {"brightness": max(0, min(255, brightness))})

    def set_thermostat(self, entity_id: str, temp: float) -> dict:
        return self.ha_call("climate", "set_temperature", entity_id,
                            {"temperature": temp})

    # ── Philips Hue ─────────────────────────────────────────────────────────────

    def hue_lights(self) -> dict:
        if not self._hue_ip or not self._hue_user:
            return {}
        try:
            import requests
            r = requests.get(
                f"http://{self._hue_ip}/api/{self._hue_user}/lights",
                timeout=5,
            )
            return r.json()
        except Exception:
            return {}

    def hue_set(self, light_id: Union[int, str], on: bool = True,
                bri: int = 254, hue: int = None, sat: int = None) -> dict:
        if not self._hue_ip or not self._hue_user:
            return {"success": False, "error": "Hue not configured"}
        try:
            import requests
            body: dict = {"on": on, "bri": bri}
            if hue is not None:
                body["hue"] = hue
            if sat is not None:
                body["sat"] = sat
            r = requests.put(
                f"http://{self._hue_ip}/api/{self._hue_user}/lights/{light_id}/state",
                json=body, timeout=5,
            )
            return {"success": True, "result": r.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE MANAGER — master router
# ─────────────────────────────────────────────────────────────────────────────

class DeviceManager:
    """
    Master controller. Routes actions to the right device agent.
    Intelligently falls back (Android → Desktop → iOS).
    """

    def __init__(self, bus=None,
                 ha_url: str = None, ha_token: str = None,
                 hue_ip: str = None, hue_user: str = None):
        self.android    = AndroidAgent()
        self.desktop    = DesktopAgent()
        self.ios        = iOSAgent()
        self.smarthome  = SmartHomeAgent(ha_url, ha_token, hue_ip, hue_user)
        self.bus        = bus

        if bus:
            self._register_handlers()

    def _register_handlers(self):
        handlers = {
            "make_call":       self._handle_call,
            "end_call":        lambda e: self.android.end_call(),
            "send_message":    self._handle_message,
            "set_alarm":       self._handle_alarm,
            "set_timer":       lambda e: self.android.set_timer(
                int(e.data.get("seconds", 60)), e.data.get("label","Timer")),
            "open_app":        self._handle_open_app,
            "close_app":       lambda e: self.android.close_app(e.data.get("app","")),
            "play_youtube":    self._handle_youtube,
            "play_music":      self._handle_music,
            "web_search":      self._handle_search,
            "take_screenshot": self._handle_screenshot,
            "volume_up":       lambda e: self._do_volume("up", e.data),
            "volume_down":     lambda e: self._do_volume("down", e.data),
            "volume_mute":     lambda e: self._do_mute(),
            "media_play_pause":lambda e: self._do_media_pp(),
            "media_next":      lambda e: self._do_media_next(),
            "navigate":        self._handle_navigate,
            "get_notifications": lambda e: self.android.read_notifications(),
            "get_battery":     lambda e: self.android.get_battery(),
            "call_contact":    lambda e: self.android.call_contact(e.data.get("name","")),
        }
        for event, handler in handlers.items():
            self.bus.subscribe(event, handler)

    # ── Main execute ────────────────────────────────────────────────────────────

    def execute(self, action: str, data: dict,
                device_preference: str = "auto") -> dict:
        """Route any action to the best available device."""
        dispatch = {
            "make_call":        self._do_call,
            "end_call":         lambda d, _: self.android.end_call(),
            "send_message":     self._do_message,
            "set_alarm":        self._do_alarm,
            "set_timer":        lambda d, _: self.android.set_timer(
                int(d.get("seconds", 60)), d.get("label","Timer")),
            "open_app":         self._do_open_app,
            "close_app":        lambda d, _: self.android.close_app(d.get("app","")),
            "play_youtube":     self._do_youtube,
            "play_music":       self._do_music,
            "web_search":       self._do_search,
            "take_screenshot":  self._do_screenshot,
            "volume_up":        lambda d, _: self._do_volume("up", d),
            "volume_down":      lambda d, _: self._do_volume("down", d),
            "volume_mute":      lambda d, _: self._do_mute(),
            "media_play_pause": lambda d, _: self._do_media_pp(),
            "media_next":       lambda d, _: self._do_media_next(),
            "navigate":         lambda d, _: self._do_navigate(d),
            "get_notifications": lambda d, _: {"notifications": self.android.read_notifications()},
            "get_battery":      lambda d, _: self.android.get_battery(),
            "call_contact":     lambda d, _: self.android.call_contact(d.get("name","")),
            "lock_screen":      lambda d, _: self.desktop.lock_screen(),
            "sleep":            lambda d, _: self.desktop.sleep(),
            "get_clipboard":    lambda d, _: {"text": self.desktop.get_clipboard()},
            "set_clipboard":    lambda d, _: self.desktop.set_clipboard(d.get("text","")),
            "smart_home_on":    lambda d, _: self.smarthome.turn_on(d.get("entity_id","")),
            "smart_home_off":   lambda d, _: self.smarthome.turn_off(d.get("entity_id","")),
        }
        handler = dispatch.get(action)
        if handler:
            return handler(data, device_preference)
        return {"success": False, "error": f"Unknown action: {action}"}

    # ── Action implementations ──────────────────────────────────────────────────

    def _do_call(self, data: dict, device: str = "auto") -> dict:
        contact = data.get("contact", "").strip()
        name    = data.get("name", "").strip()
        if not contact and name:
            # Try to find in contacts
            number = self.android.find_contact(name)
            if number:
                contact = number
        if not contact:
            return {"success": False, "error": "No contact or number specified"}

        if device in ("auto", "android") and self.android.list_devices():
            return self.android.make_call(contact, self.android._first_device())
        return self.ios.open_url_scheme("phone", number=contact)

    def _do_message(self, data: dict, device: str = "auto") -> dict:
        contact = data.get("contact", "") or data.get("number", "")
        message = data.get("message", "") or data.get("text", "")
        app     = data.get("app", "whatsapp").lower()

        if self.android.list_devices():
            if "whatsapp" in app:
                return self.android.send_whatsapp(contact, message)
            if "telegram" in app:
                return self.android.send_telegram(contact, message)
            if "email" in app or "mail" in app:
                return self.android.compose_email(contact, message)
            return self.android.send_sms(contact, message)

        return self.ios.open_url_scheme(
            "whatsapp" if "whatsapp" in app else "sms",
            contact=contact, number=contact, message=message,
        )

    def _do_alarm(self, data: dict, device: str = "auto") -> dict:
        h, m = self._parse_time(data.get("time", ""))
        if h is None:
            return {"success": False, "error": f"Could not parse time: {data.get('time','')}"}
        label = data.get("label", "ARIA Alarm")
        if device in ("auto", "android") and self.android.list_devices():
            return self.android.set_alarm(h, m, label)
        return self.desktop.set_alarm(h, m, label)

    def _do_open_app(self, data: dict, device: str = "auto") -> dict:
        app = data.get("app", "").strip()
        if not app:
            return {"success": False, "error": "No app specified"}
        if device == "android" and self.android.list_devices():
            return self.android.open_app(app)
        if device in ("auto", "desktop"):
            return self.desktop.open_app(app)
        return {"success": False, "error": "No device available"}

    def _do_youtube(self, data: dict, device: str = "auto") -> dict:
        query = data.get("query", "")
        if self.android.list_devices() and device in ("auto", "android"):
            return self.android.play_youtube(query)
        return self.desktop.open_youtube(query)

    def _do_music(self, data: dict, device: str = "auto") -> dict:
        query = data.get("query", "")
        if self.android.list_devices() and device in ("auto", "android"):
            return self.android.play_spotify(query)
        return self.desktop.open_spotify_web(query)

    def _do_search(self, data: dict, device: str = "auto") -> dict:
        query = data.get("query", "") or data.get("raw_text", "")
        return self.desktop.open_google(query)

    def _do_screenshot(self, data: dict, device: str = "auto") -> dict:
        if device == "android" and self.android.list_devices():
            return self.android.take_screenshot()
        return self.desktop.screenshot()

    def _do_volume(self, direction: str, data: dict) -> dict:
        steps = int(data.get("steps", 1))
        if self.android.list_devices():
            return self.android.volume_up(steps) if direction == "up" \
                else self.android.volume_down(steps)
        return self.desktop.volume_up(steps) if direction == "up" \
            else self.desktop.volume_down(steps)

    def _do_mute(self) -> dict:
        if self.android.list_devices():
            return self.android.volume_mute()
        return self.desktop.volume_mute()

    def _do_media_pp(self) -> dict:
        if self.android.list_devices():
            return self.android.media_play_pause()
        return self.desktop.media_play_pause()

    def _do_media_next(self) -> dict:
        if self.android.list_devices():
            return self.android.media_next()
        return {"success": False, "error": "Media next not supported on desktop"}

    def _do_navigate(self, data: dict) -> dict:
        dest = data.get("destination", "") or data.get("query", "")
        if self.android.list_devices():
            return self.android.navigate_to(dest)
        return self.desktop.open_maps(dest)

    # ── Bus handlers ─────────────────────────────────────────────────────────────

    def _handle_call(self, event):
        result = self._do_call(event.data)
        if self.bus:
            try:
                from agents.agent_bus import Event
                self.bus.publish(Event(
                    "make_call_response", result, "device_manager", reply_to=event.id
                ))
            except Exception:
                pass

    def _handle_message(self, event):
        result = self._do_message(event.data)
        if self.bus:
            try:
                from agents.agent_bus import Event
                self.bus.publish(Event(
                    "send_message_response", result, "device_manager", reply_to=event.id
                ))
            except Exception:
                pass

    def _handle_alarm(self, event):
        result = self._do_alarm(event.data)
        if self.bus:
            try:
                from agents.agent_bus import Event
                self.bus.publish(Event(
                    "set_alarm_response", result, "device_manager", reply_to=event.id
                ))
            except Exception:
                pass

    def _handle_open_app(self, event):
        self._do_open_app(event.data)

    def _handle_youtube(self, event):
        self._do_youtube(event.data)

    def _handle_music(self, event):
        self._do_music(event.data)

    def _handle_search(self, event):
        self._do_search(event.data)

    def _handle_screenshot(self, event):
        self._do_screenshot(event.data)

    def _handle_navigate(self, event):
        self._do_navigate(event.data)

    # ── Time parsing ─────────────────────────────────────────────────────────────

    def _parse_time(self, time_str: str) -> tuple[Optional[int], int]:
        if not time_str:
            return None, 0
        time_str = time_str.strip().lower()

        m = re.search(r"(\d{1,2})[:.](\d{2})\s*(am|pm)?", time_str)
        if m:
            h, mn = int(m.group(1)), int(m.group(2))
            if m.group(3) == "pm" and h < 12:
                h += 12
            elif m.group(3) == "am" and h == 12:
                h = 0
            return h, mn

        m = re.search(r"(\d{1,2})\s*(am|pm)", time_str)
        if m:
            h = int(m.group(1))
            if m.group(2) == "pm" and h < 12:
                h += 12
            elif m.group(2) == "am" and h == 12:
                h = 0
            return h, 0

        m = re.search(r"(\d{1,2})", time_str)
        if m:
            return int(m.group(1)), 0

        return None, 0

    # ── Status ───────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "android":   self.android.status(),
            "desktop":   {"platform": PLATFORM, "available": True},
            "ios":       self.ios.status(),
            "smart_home": {
                "home_assistant": bool(self.smarthome._ha_url),
                "philips_hue":    bool(self.smarthome._hue_ip),
            },
        }
