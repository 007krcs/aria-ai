"""
ARIA Media Agent
Windows media and audio control — volume, playback, device enumeration.
Dependencies: pycaw, comtypes, psutil, pywin32 (win32api/win32con/win32gui)
"""

import ctypes
import ctypes.wintypes
import os
import subprocess
import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Optional pycaw / comtypes imports
# ---------------------------------------------------------------------------
try:
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import (
        AudioUtilities,
        IAudioEndpointVolume,
        ISimpleAudioVolume,
    )
    _PYCAW = True
except ImportError:
    _PYCAW = False

# ---------------------------------------------------------------------------
# Optional win32 imports
# ---------------------------------------------------------------------------
try:
    import win32api
    import win32con
    import win32gui
    _WIN32 = True
except ImportError:
    _WIN32 = False


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _send_media_key(vk_code: int) -> None:
    """Send a media key via keybd_event (works without focus)."""
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP      = 0xB2
VK_VOLUME_MUTE     = 0xAD


def _get_endpoint_volume():
    """Return the IAudioEndpointVolume COM interface for the default render device."""
    if not _PYCAW:
        raise RuntimeError("pycaw not installed")
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _ok(result: str, data=None) -> dict:
    return {"ok": True, "result": result, "data": data}


def _err(result: str, data=None) -> dict:
    return {"ok": False, "result": result, "data": data}


# ---------------------------------------------------------------------------
# Volume control
# ---------------------------------------------------------------------------

def set_volume(level: int) -> dict:
    """Set master volume 0-100."""
    try:
        level = max(0, min(100, int(level)))
        if _PYCAW:
            vol = _get_endpoint_volume()
            vol.SetMasterVolumeLevelScalar(level / 100.0, None)
            return _ok(f"Volume set to {level}%")
        # Fallback: nircmd if available
        result = subprocess.run(
            ["nircmd.exe", "setsysvolume", str(int(level * 655.35))],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return _ok(f"Volume set to {level}% via nircmd")
        return _err("pycaw not available and nircmd not found")
    except Exception as e:
        return _err(f"set_volume error: {e}\n{traceback.format_exc()}")


def get_volume() -> dict:
    """Get current master volume level (0-100)."""
    try:
        if _PYCAW:
            vol = _get_endpoint_volume()
            scalar = vol.GetMasterVolumeLevelScalar()
            level = round(scalar * 100)
            muted = bool(vol.GetMute())
            return _ok(f"Volume: {level}% {'(muted)' if muted else ''}", {"level": level, "muted": muted})
        return _err("pycaw not available")
    except Exception as e:
        return _err(f"get_volume error: {e}")


def mute() -> dict:
    """Mute master volume."""
    try:
        if _PYCAW:
            vol = _get_endpoint_volume()
            vol.SetMute(1, None)
            return _ok("Audio muted")
        _send_media_key(VK_VOLUME_MUTE)
        return _ok("Mute key sent")
    except Exception as e:
        return _err(f"mute error: {e}")


def unmute() -> dict:
    """Unmute master volume."""
    try:
        if _PYCAW:
            vol = _get_endpoint_volume()
            vol.SetMute(0, None)
            return _ok("Audio unmuted")
        _send_media_key(VK_VOLUME_MUTE)
        return _ok("Mute-toggle key sent (was muted assumption)")
    except Exception as e:
        return _err(f"unmute error: {e}")


def toggle_mute() -> dict:
    """Toggle mute state."""
    try:
        if _PYCAW:
            vol = _get_endpoint_volume()
            current = vol.GetMute()
            vol.SetMute(not current, None)
            state = "muted" if not current else "unmuted"
            return _ok(f"Audio {state}")
        _send_media_key(VK_VOLUME_MUTE)
        return _ok("Mute toggle key sent")
    except Exception as e:
        return _err(f"toggle_mute error: {e}")


# ---------------------------------------------------------------------------
# Playback controls
# ---------------------------------------------------------------------------

def play_pause() -> dict:
    """Send media Play/Pause key."""
    try:
        _send_media_key(VK_MEDIA_PLAY_PAUSE)
        return _ok("Play/Pause sent")
    except Exception as e:
        return _err(f"play_pause error: {e}")


def next_track() -> dict:
    """Send Next Track media key."""
    try:
        _send_media_key(VK_MEDIA_NEXT_TRACK)
        return _ok("Next track sent")
    except Exception as e:
        return _err(f"next_track error: {e}")


def prev_track() -> dict:
    """Send Previous Track media key."""
    try:
        _send_media_key(VK_MEDIA_PREV_TRACK)
        return _ok("Previous track sent")
    except Exception as e:
        return _err(f"prev_track error: {e}")


def stop_media() -> dict:
    """Send Stop media key."""
    try:
        _send_media_key(VK_MEDIA_STOP)
        return _ok("Media stop sent")
    except Exception as e:
        return _err(f"stop_media error: {e}")


def open_media(path: str) -> dict:
    """Open and play a media file using the default system handler."""
    try:
        if not os.path.exists(path):
            return _err(f"File not found: {path}")
        os.startfile(path)
        return _ok(f"Opened: {path}")
    except Exception as e:
        return _err(f"open_media error: {e}")


# ---------------------------------------------------------------------------
# Now-playing detection
# ---------------------------------------------------------------------------

def get_now_playing() -> dict:
    """
    Detect what is currently playing by scanning window titles for common
    media player patterns (Spotify, Windows Media Player, VLC, foobar2000,
    Chrome/Edge with media, etc.).
    """
    try:
        candidates = []

        if _WIN32:
            def _enum_cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        tl = title.lower()
                        # Common media player signatures
                        keywords = [
                            "spotify", "vlc", "foobar", "winamp",
                            "media player", "groove", "youtube",
                            "soundcloud", "deezer", "tidal", "apple music",
                            " - ", "▶", "⏸",
                        ]
                        if any(k in tl for k in keywords):
                            candidates.append(title)
                return True

            win32gui.EnumWindows(_enum_cb, None)
        else:
            # Fallback: use tasklist + wmic
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | Select-Object Name,MainWindowTitle | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import json
                procs = json.loads(result.stdout or "[]")
                media_names = {"spotify", "vlc", "wmplayer", "foobar2000",
                               "winamp", "groove", "msedge", "chrome"}
                if isinstance(procs, dict):
                    procs = [procs]
                for p in procs:
                    name = p.get("Name", "").lower()
                    title = p.get("MainWindowTitle", "")
                    if name in media_names and title:
                        candidates.append(title)

        if candidates:
            best = candidates[0]
            return _ok(f"Now playing (window title): {best}", {"candidates": candidates})
        return _ok("Nothing detected as playing", {"candidates": []})

    except Exception as e:
        return _err(f"get_now_playing error: {e}")


# ---------------------------------------------------------------------------
# Audio device enumeration
# ---------------------------------------------------------------------------

def get_audio_devices() -> dict:
    """List all audio input/output devices."""
    try:
        devices_info = []
        if _PYCAW:
            # Render (output) devices
            render_devices = AudioUtilities.GetAllDevices()
            for d in render_devices:
                devices_info.append({
                    "name": d.FriendlyName,
                    "id": d.id,
                    "type": "output",
                    "state": str(d.state),
                })

        # Also enumerate via PowerShell for inputs
        ps_cmd = (
            "Get-WmiObject Win32_SoundDevice | "
            "Select-Object Name,Status,DeviceID | ConvertTo-Json"
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            import json
            wmi_devs = json.loads(result.stdout)
            if isinstance(wmi_devs, dict):
                wmi_devs = [wmi_devs]
            for d in wmi_devs:
                devices_info.append({
                    "name": d.get("Name", "Unknown"),
                    "id": d.get("DeviceID", ""),
                    "type": "wmi",
                    "state": d.get("Status", ""),
                })

        summary = "\n".join(
            f"  [{d['type']}] {d['name']} ({d['state']})" for d in devices_info
        ) or "No devices found"
        return _ok(f"Audio devices:\n{summary}", {"devices": devices_info})
    except Exception as e:
        return _err(f"get_audio_devices error: {e}")


def take_audio_screenshot() -> dict:
    """List all audio devices and their current volume levels."""
    try:
        lines = []
        if _PYCAW:
            # Master
            vol = _get_endpoint_volume()
            master_level = round(vol.GetMasterVolumeLevelScalar() * 100)
            muted = bool(vol.GetMute())
            lines.append(f"Master: {master_level}% {'[MUTED]' if muted else ''}")

            # Per-session (per-app) volumes
            sessions = AudioUtilities.GetAllSessions()
            for s in sessions:
                if s.Process:
                    try:
                        sv = s._ctl.QueryInterface(ISimpleAudioVolume)
                        app_vol = round(sv.GetMasterVolume() * 100)
                        app_muted = bool(sv.GetMute())
                        lines.append(
                            f"  [{s.Process.name()}] {app_vol}% "
                            f"{'[MUTED]' if app_muted else ''}"
                        )
                    except Exception:
                        pass

        devices_result = get_audio_devices()
        if devices_result["ok"] and devices_result.get("data"):
            for d in devices_result["data"].get("devices", []):
                lines.append(f"  Device: {d['name']} ({d['type']}, {d['state']})")

        if not lines:
            return _err("pycaw not available; cannot capture audio snapshot")
        return _ok("\n".join(lines), {"lines": lines})
    except Exception as e:
        return _err(f"take_audio_screenshot error: {e}")


def set_app_volume(app_name: str, level: int) -> dict:
    """Set volume for a specific application by process name (0-100)."""
    try:
        if not _PYCAW:
            return _err("pycaw not available")
        level = max(0, min(100, int(level)))
        sessions = AudioUtilities.GetAllSessions()
        matched = []
        for s in sessions:
            if s.Process and app_name.lower() in s.Process.name().lower():
                sv = s._ctl.QueryInterface(ISimpleAudioVolume)
                sv.SetMasterVolume(level / 100.0, None)
                matched.append(s.Process.name())

        if not matched:
            return _err(f"No audio session found for '{app_name}'")
        return _ok(f"Set volume to {level}% for: {', '.join(matched)}")
    except Exception as e:
        return _err(f"set_app_volume error: {e}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    print("=== get_volume ===")
    print(json.dumps(get_volume(), indent=2))
    print("=== get_audio_devices ===")
    print(json.dumps(get_audio_devices(), indent=2))
    print("=== get_now_playing ===")
    print(json.dumps(get_now_playing(), indent=2))
    print("=== take_audio_screenshot ===")
    print(json.dumps(take_audio_screenshot(), indent=2))
