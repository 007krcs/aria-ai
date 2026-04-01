"""
ARIA System Agent
Deep Windows system monitoring and control agent.
Uses psutil, ctypes, and win32 APIs for full OS integration.
"""

import os
import re
import sys
import time
import socket
import subprocess
import traceback
import winreg
from typing import Any, Dict, Optional

# ------------------------------------------------------------------
# Optional imports — auto-installed on first use
# ------------------------------------------------------------------

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

try:
    import ctypes
    from ctypes import wintypes
    CTYPES_OK = True
except ImportError:
    CTYPES_OK = False

try:
    import pyperclip
    PYPERCLIP_OK = True
except ImportError:
    PYPERCLIP_OK = False


def _ok(result: str = "", data: Any = None) -> Dict:
    return {"ok": True, "result": result, "data": data}


def _err(result: str = "", data: Any = None) -> Dict:
    return {"ok": False, "result": result, "data": data}


def _auto_install(*packages: str):
    """Pip-install packages silently and reload relevant globals."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + list(packages)
    )
    import importlib
    importlib.invalidate_caches()


class SystemAgent:
    """
    Deep Windows system monitoring and control agent for ARIA.
    All methods return {"ok": bool, "result": str, "data": any}.
    """

    def __init__(self):
        self._ensure_deps()

    # ------------------------------------------------------------------
    # Dependency bootstrap
    # ------------------------------------------------------------------

    def _ensure_deps(self):
        global PSUTIL_OK, PYPERCLIP_OK, psutil, pyperclip
        missing = []
        if not PSUTIL_OK:
            missing.append("psutil")
        if not PYPERCLIP_OK:
            missing.append("pyperclip")
        if missing:
            try:
                _auto_install(*missing)
                if not PSUTIL_OK:
                    import psutil as _p
                    psutil = _p
                    PSUTIL_OK = True
                if not PYPERCLIP_OK:
                    import pyperclip as _pc
                    pyperclip = _pc
                    PYPERCLIP_OK = True
            except Exception as exc:
                pass  # Will surface as errors on individual calls

    # ------------------------------------------------------------------
    # System overview
    # ------------------------------------------------------------------

    def get_system_info(self) -> Dict:
        """CPU, RAM, disk, GPU (if available), OS, hostname, uptime."""
        try:
            import platform
            uname = platform.uname()
            boot_time = psutil.boot_time()
            uptime_secs = int(time.time() - boot_time)
            uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime_secs))

            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "freq_mhz": getattr(psutil.cpu_freq(), "current", None),
                "usage_percent": psutil.cpu_percent(interval=0.5),
            }

            mem = psutil.virtual_memory()
            mem_info = {
                "total_gb": round(mem.total / 1e9, 2),
                "used_gb": round(mem.used / 1e9, 2),
                "available_gb": round(mem.available / 1e9, 2),
                "percent": mem.percent,
            }

            disk = psutil.disk_usage("C:\\")
            disk_info = {
                "total_gb": round(disk.total / 1e9, 2),
                "used_gb": round(disk.used / 1e9, 2),
                "free_gb": round(disk.free / 1e9, 2),
                "percent": disk.percent,
            }

            # GPU via optional libraries
            gpu_info = self._get_gpu_info()

            data = {
                "hostname": uname.node,
                "os": f"{uname.system} {uname.release} {uname.version}",
                "architecture": uname.machine,
                "processor": uname.processor,
                "uptime": uptime_str,
                "uptime_seconds": uptime_secs,
                "cpu": cpu_info,
                "memory": mem_info,
                "disk_c": disk_info,
                "gpu": gpu_info,
            }
            return _ok(
                f"System: {uname.node} | CPU {cpu_info['usage_percent']}% | "
                f"RAM {mem_info['percent']}%",
                data=data,
            )
        except Exception as exc:
            return _err(str(exc), data=traceback.format_exc())

    def _get_gpu_info(self) -> Dict:
        """Try GPUtil, then nvidia-smi, then return empty."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                return {
                    "name": g.name,
                    "load_percent": round(g.load * 100, 1),
                    "memory_used_mb": g.memoryUsed,
                    "memory_total_mb": g.memoryTotal,
                    "temperature_c": g.temperature,
                }
        except Exception:
            pass
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
            parts = [p.strip() for p in result.split(",")]
            if len(parts) >= 5:
                return {
                    "name": parts[0],
                    "load_percent": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "temperature_c": float(parts[4]),
                }
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    def get_cpu_usage(self) -> Dict:
        """Per-core CPU usage percentages."""
        try:
            per_core = psutil.cpu_percent(interval=0.5, percpu=True)
            total = psutil.cpu_percent(interval=0)
            freq = psutil.cpu_freq()
            data = {
                "total_percent": total,
                "per_core_percent": per_core,
                "freq_mhz": getattr(freq, "current", None),
                "freq_max_mhz": getattr(freq, "max", None),
            }
            return _ok(f"CPU total: {total}%", data=data)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def get_memory_info(self) -> Dict:
        """RAM total/used/free/percent, plus swap."""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            data = {
                "ram": {
                    "total_gb": round(mem.total / 1e9, 2),
                    "used_gb": round(mem.used / 1e9, 2),
                    "free_gb": round(mem.available / 1e9, 2),
                    "percent": mem.percent,
                },
                "swap": {
                    "total_gb": round(swap.total / 1e9, 2),
                    "used_gb": round(swap.used / 1e9, 2),
                    "free_gb": round(swap.free / 1e9, 2),
                    "percent": swap.percent,
                },
            }
            return _ok(
                f"RAM: {mem.percent}% used ({round(mem.used/1e9,1)}/{round(mem.total/1e9,1)} GB)",
                data=data,
            )
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Disk
    # ------------------------------------------------------------------

    def get_disk_info(self, path: str = "C:\\") -> Dict:
        """Disk space for a given path, plus all mounted partitions."""
        try:
            usage = psutil.disk_usage(path)
            partitions = []
            for part in psutil.disk_partitions(all=False):
                try:
                    u = psutil.disk_usage(part.mountpoint)
                    partitions.append({
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "total_gb": round(u.total / 1e9, 2),
                        "used_gb": round(u.used / 1e9, 2),
                        "free_gb": round(u.free / 1e9, 2),
                        "percent": u.percent,
                    })
                except PermissionError:
                    pass
            data = {
                "path": path,
                "total_gb": round(usage.total / 1e9, 2),
                "used_gb": round(usage.used / 1e9, 2),
                "free_gb": round(usage.free / 1e9, 2),
                "percent": usage.percent,
                "all_partitions": partitions,
            }
            return _ok(
                f"{path}: {usage.percent}% used ({round(usage.free/1e9,1)} GB free)",
                data=data,
            )
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def get_network_stats(self) -> Dict:
        """Bytes sent/received and active connections."""
        try:
            io = psutil.net_io_counters()
            conns = psutil.net_connections(kind="inet")
            established = [
                {
                    "pid": c.pid,
                    "local": f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "",
                    "remote": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "",
                    "status": c.status,
                }
                for c in conns
                if c.status == "ESTABLISHED"
            ]
            data = {
                "bytes_sent": io.bytes_sent,
                "bytes_recv": io.bytes_recv,
                "packets_sent": io.packets_sent,
                "packets_recv": io.packets_recv,
                "mb_sent": round(io.bytes_sent / 1e6, 2),
                "mb_recv": round(io.bytes_recv / 1e6, 2),
                "established_connections": established,
                "total_connections": len(conns),
            }
            return _ok(
                f"Net: ↑{data['mb_sent']} MB ↓{data['mb_recv']} MB | "
                f"{len(established)} established connections",
                data=data,
            )
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------

    def get_battery(self) -> Dict:
        """Battery percentage and charging status."""
        try:
            batt = psutil.sensors_battery()
            if batt is None:
                return _ok("No battery detected (desktop)", data={"battery": None})
            secs_left = batt.secsleft
            time_left = (
                "Unlimited" if secs_left == psutil.POWER_TIME_UNLIMITED
                else "Unknown" if secs_left == psutil.POWER_TIME_UNKNOWN
                else str(int(secs_left // 3600)) + "h " + str(int((secs_left % 3600) // 60)) + "m"
            )
            data = {
                "percent": batt.percent,
                "plugged_in": batt.power_plugged,
                "charging": batt.power_plugged and batt.percent < 100,
                "time_left": time_left,
            }
            status = "Charging" if data["charging"] else ("Plugged in" if batt.power_plugged else "On battery")
            return _ok(f"Battery: {batt.percent}% ({status})", data=data)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Processes
    # ------------------------------------------------------------------

    def get_top_processes(self, n: int = 10) -> Dict:
        """Top N processes by CPU and RAM usage."""
        try:
            procs = []
            for p in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent", "memory_info", "status"]
            ):
                try:
                    info = p.info
                    info["memory_mb"] = round(info["memory_info"].rss / 1e6, 1) if info.get("memory_info") else 0
                    del info["memory_info"]
                    procs.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Trigger CPU measurement
            time.sleep(0.2)
            for p in psutil.process_iter(["pid", "cpu_percent"]):
                try:
                    pass
                except Exception:
                    pass

            by_cpu = sorted(procs, key=lambda x: x.get("cpu_percent", 0), reverse=True)[:n]
            by_ram = sorted(procs, key=lambda x: x.get("memory_mb", 0), reverse=True)[:n]

            return _ok(
                f"Top {n} processes by CPU/RAM",
                data={"by_cpu": by_cpu, "by_ram": by_ram},
            )
        except Exception as exc:
            return _err(str(exc))

    def kill_process(self, name_or_pid) -> Dict:
        """Kill a process by name or PID."""
        try:
            killed = []
            errors = []
            # Try as PID first
            try:
                pid = int(name_or_pid)
                p = psutil.Process(pid)
                p.kill()
                killed.append(f"PID {pid} ({p.name()})")
            except (ValueError, TypeError):
                # It's a name
                for p in psutil.process_iter(["pid", "name"]):
                    try:
                        if p.info["name"].lower() == str(name_or_pid).lower():
                            p.kill()
                            killed.append(f"PID {p.info['pid']} ({p.info['name']})")
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        errors.append(str(e))
            except psutil.NoSuchProcess:
                return _err(f"No process with PID {name_or_pid}")
            except psutil.AccessDenied:
                return _err(f"Access denied killing PID {name_or_pid}")

            if killed:
                return _ok(f"Killed: {', '.join(killed)}", data={"killed": killed, "errors": errors})
            else:
                return _err(f"No process found: {name_or_pid}", data={"errors": errors})
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Volume control (Windows Core Audio via ctypes)
    # ------------------------------------------------------------------

    def _get_volume_interface(self):
        """Return the Windows ISimpleAudioVolume COM interface via pycaw."""
        try:
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = interface.QueryInterface(IAudioEndpointVolume)
            return volume
        except Exception:
            return None

    def set_volume(self, level: int) -> Dict:
        """Set system volume 0-100."""
        level = max(0, min(100, int(level)))
        # Method 1: pycaw
        vol_if = self._get_volume_interface()
        if vol_if:
            try:
                vol_if.SetMasterVolumeLevelScalar(level / 100.0, None)
                return _ok(f"Volume set to {level}%", data={"volume": level})
            except Exception:
                pass
        # Method 2: nircmd (if available)
        try:
            nircmd_val = int(level * 655.35)
            subprocess.run(["nircmd", "setsysvolume", str(nircmd_val)], check=True,
                           capture_output=True, timeout=5)
            return _ok(f"Volume set to {level}% via nircmd", data={"volume": level})
        except Exception:
            pass
        # Method 3: PowerShell
        try:
            ps_cmd = (
                f"$obj = New-Object -ComObject WScript.Shell; "
                f"$steps = [Math]::Round({level} / 2); "
                f"$obj.SendKeys([char]174 * 50); "
                f"$obj.SendKeys([char]175 * $steps)"
            )
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True, timeout=10,
            )
            return _ok(f"Volume adjusted to ~{level}% via PowerShell", data={"volume": level})
        except Exception as exc:
            return _err(f"Could not set volume: {exc}")

    def get_volume(self) -> Dict:
        """Get current system volume (0-100)."""
        vol_if = self._get_volume_interface()
        if vol_if:
            try:
                scalar = vol_if.GetMasterVolumeLevelScalar()
                level = round(scalar * 100)
                muted = bool(vol_if.GetMute())
                return _ok(f"Volume: {level}% (muted={muted})", data={"volume": level, "muted": muted})
            except Exception:
                pass
        # Fallback: registry (approximate)
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Multimedia\Audio",
            )
            val, _ = winreg.QueryValueEx(key, "UserVolumeLevelNS")
            winreg.CloseKey(key)
            level = round(val / 0xFFFF * 100)
            return _ok(f"Volume: ~{level}% (registry)", data={"volume": level})
        except Exception as exc:
            return _err(f"Could not read volume: {exc}")

    def mute(self) -> Dict:
        """Mute system audio."""
        vol_if = self._get_volume_interface()
        if vol_if:
            try:
                vol_if.SetMute(1, None)
                return _ok("System audio muted")
            except Exception:
                pass
        try:
            import ctypes
            # VK_VOLUME_MUTE = 0xAD
            ctypes.windll.user32.keybd_event(0xAD, 0, 0, 0)
            ctypes.windll.user32.keybd_event(0xAD, 0, 2, 0)
            return _ok("Muted via keybd_event")
        except Exception as exc:
            return _err(f"Mute failed: {exc}")

    def unmute(self) -> Dict:
        """Unmute system audio."""
        vol_if = self._get_volume_interface()
        if vol_if:
            try:
                vol_if.SetMute(0, None)
                return _ok("System audio unmuted")
            except Exception:
                pass
        try:
            import ctypes
            # Press mute key again to toggle
            ctypes.windll.user32.keybd_event(0xAD, 0, 0, 0)
            ctypes.windll.user32.keybd_event(0xAD, 0, 2, 0)
            return _ok("Unmuted via keybd_event")
        except Exception as exc:
            return _err(f"Unmute failed: {exc}")

    # ------------------------------------------------------------------
    # WiFi
    # ------------------------------------------------------------------

    def get_wifi_info(self) -> Dict:
        """Current WiFi SSID, signal strength, and IP address."""
        try:
            result = subprocess.check_output(
                ["netsh", "wlan", "show", "interfaces"],
                encoding="utf-8",
                errors="ignore",
                timeout=10,
            )
            data = {}
            for line in result.splitlines():
                line = line.strip()
                if line.startswith("SSID") and "BSSID" not in line:
                    data["ssid"] = line.split(":", 1)[-1].strip()
                elif line.startswith("Signal"):
                    data["signal"] = line.split(":", 1)[-1].strip()
                elif line.startswith("State"):
                    data["state"] = line.split(":", 1)[-1].strip()
                elif line.startswith("Radio type"):
                    data["radio_type"] = line.split(":", 1)[-1].strip()

            # Get local IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                data["ip"] = s.getsockname()[0]
                s.close()
            except Exception:
                data["ip"] = "Unknown"

            ssid = data.get("ssid", "Unknown")
            signal = data.get("signal", "?")
            ip = data.get("ip", "Unknown")
            return _ok(f"WiFi: {ssid} | Signal: {signal} | IP: {ip}", data=data)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Ping
    # ------------------------------------------------------------------

    def ping(self, host: str) -> Dict:
        """Ping a host and return average latency in ms."""
        try:
            result = subprocess.check_output(
                ["ping", "-n", "4", host],
                encoding="utf-8",
                errors="ignore",
                timeout=15,
            )
            # Parse average latency
            latency = None
            for line in result.splitlines():
                line_lower = line.lower()
                if "average" in line_lower or "avg" in line_lower:
                    parts = line.split("=")
                    if len(parts) > 1:
                        val = parts[-1].strip().replace("ms", "").strip()
                        try:
                            latency = float(val)
                        except ValueError:
                            pass
                    break
            data = {
                "host": host,
                "latency_ms": latency,
                "raw": result.strip(),
                "reachable": "TTL=" in result or "ttl=" in result,
            }
            if latency is not None:
                return _ok(f"Ping {host}: {latency}ms avg", data=data)
            elif data["reachable"]:
                return _ok(f"Ping {host}: reachable (latency parse failed)", data=data)
            else:
                return _ok(f"Ping {host}: unreachable", data=data)
        except subprocess.TimeoutExpired:
            return _err(f"Ping {host}: timed out")
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # URL / browser
    # ------------------------------------------------------------------

    def open_url_default(self, url: str) -> Dict:
        """Open a URL in the system's default browser."""
        try:
            import webbrowser
            webbrowser.open(url)
            return _ok(f"Opened in default browser: {url}", data={"url": url})
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    def get_clipboard(self) -> Dict:
        """Read current clipboard content."""
        try:
            if PYPERCLIP_OK:
                text = pyperclip.paste()
                return _ok(f"Clipboard ({len(text)} chars)", data={"text": text})
            # Fallback via PowerShell
            result = subprocess.check_output(
                ["powershell", "-Command", "Get-Clipboard"],
                encoding="utf-8",
                errors="ignore",
                timeout=5,
            ).strip()
            return _ok(f"Clipboard ({len(result)} chars)", data={"text": result})
        except Exception as exc:
            return _err(str(exc))

    def set_clipboard(self, text: str) -> Dict:
        """Set clipboard content."""
        try:
            if PYPERCLIP_OK:
                pyperclip.copy(text)
                return _ok(f"Clipboard set ({len(text)} chars)")
            subprocess.run(
                ["powershell", "-Command", f"Set-Clipboard -Value '{text}'"],
                check=True,
                timeout=5,
            )
            return _ok(f"Clipboard set via PowerShell ({len(text)} chars)")
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def send_notification(self, title: str, message: str) -> Dict:
        """Send a Windows toast notification via plyer or win10toast."""
        # Method 1: plyer
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=message,
                app_name="ARIA",
                timeout=8,
            )
            return _ok(f"Notification sent: {title}", data={"title": title, "message": message})
        except Exception:
            pass

        # Method 2: win10toast
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=8, threaded=True)
            return _ok(f"Notification sent via win10toast: {title}")
        except Exception:
            pass

        # Method 3: PowerShell BurntToast / native
        try:
            ps_script = (
                f"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
                f"ContentType=WindowsRuntime] | Out-Null; "
                f"$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02; "
                f"$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template); "
                f"$xml.GetElementsByTagName('text')[0].AppendChild($xml.CreateTextNode('{title}')) | Out-Null; "
                f"$xml.GetElementsByTagName('text')[1].AppendChild($xml.CreateTextNode('{message}')) | Out-Null; "
                f"$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); "
                f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('ARIA').Show($toast);"
            )
            subprocess.run(
                ["powershell", "-Command", ps_script],
                timeout=10,
                capture_output=True,
            )
            return _ok(f"Notification sent via PowerShell: {title}")
        except Exception as exc:
            return _err(f"Notification failed: {exc}")

    # ------------------------------------------------------------------
    # Environment variables
    # ------------------------------------------------------------------

    def get_environment_vars(self) -> Dict:
        """List key environment variables."""
        try:
            important_keys = [
                "PATH", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
                "USERNAME", "COMPUTERNAME", "SYSTEMROOT", "WINDIR",
                "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES",
                "HOMEDRIVE", "HOMEPATH", "JAVA_HOME", "PYTHON", "PYTHONPATH",
                "CONDA_PREFIX", "VIRTUAL_ENV", "NODE_PATH", "GOPATH",
                "OneDrive", "OneDriveConsumer", "NUMBER_OF_PROCESSORS",
            ]
            env = dict(os.environ)
            filtered = {k: env[k] for k in important_keys if k in env}
            # Also include any that look like dev tools
            dev_keys = [k for k in env if any(x in k.upper() for x in ["HOME", "PATH", "SDK", "ROOT", "DIR"])]
            for k in dev_keys:
                filtered.setdefault(k, env[k])
            return _ok(f"Found {len(filtered)} environment variables", data=filtered)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Windows-specific controls
    # ------------------------------------------------------------------

    def restart_explorer(self) -> Dict:
        """Restart Windows Explorer (refreshes taskbar, desktop, file explorer)."""
        try:
            subprocess.run(["taskkill", "/f", "/im", "explorer.exe"],
                           capture_output=True, timeout=10)
            time.sleep(1)
            subprocess.Popen(["explorer.exe"])
            return _ok("Windows Explorer restarted")
        except Exception as exc:
            return _err(f"Failed to restart Explorer: {exc}")

    def lock_screen(self) -> Dict:
        """Lock the Windows workstation."""
        try:
            import ctypes
            ctypes.windll.user32.LockWorkStation()
            return _ok("Workstation locked")
        except Exception as exc:
            return _err(f"Could not lock screen: {exc}")

    def get_screen_resolution(self) -> Dict:
        """Get current display resolution(s)."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            primary_w = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            primary_h = user32.GetSystemMetrics(1)  # SM_CYSCREEN

            # Virtual screen (all monitors combined)
            virt_w = user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
            virt_h = user32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN
            monitors = user32.GetSystemMetrics(80)  # SM_CMONITORS

            data = {
                "primary": {"width": primary_w, "height": primary_h},
                "virtual_screen": {"width": virt_w, "height": virt_h},
                "monitor_count": monitors,
            }
            return _ok(
                f"Primary: {primary_w}x{primary_h} | Monitors: {monitors}",
                data=data,
            )
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # Windows Services
    # ------------------------------------------------------------------

    def get_running_services(self) -> Dict:
        """List top 20 running Windows services."""
        try:
            result = subprocess.check_output(
                ["sc", "query", "type=", "all", "state=", "running"],
                encoding="utf-8",
                errors="ignore",
                timeout=15,
            )
            services = []
            current = {}
            for line in result.splitlines():
                line = line.strip()
                if line.startswith("SERVICE_NAME:"):
                    if current:
                        services.append(current)
                    current = {"name": line.split(":", 1)[-1].strip()}
                elif line.startswith("DISPLAY_NAME:"):
                    current["display_name"] = line.split(":", 1)[-1].strip()
                elif line.startswith("STATE"):
                    parts = line.split()
                    current["state"] = parts[-1] if parts else "UNKNOWN"
            if current:
                services.append(current)

            top20 = services[:20]
            return _ok(
                f"Found {len(services)} running services (showing top 20)",
                data={"services": top20, "total": len(services)},
            )
        except Exception as exc:
            return _err(str(exc))


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    agent = SystemAgent()

    print("\n--- System Info ---")
    r = agent.get_system_info()
    print(r["result"])

    print("\n--- CPU Usage ---")
    r = agent.get_cpu_usage()
    print(r["result"])

    print("\n--- Memory ---")
    r = agent.get_memory_info()
    print(r["result"])

    print("\n--- Disk C:\\ ---")
    r = agent.get_disk_info("C:\\")
    print(r["result"])

    print("\n--- Network Stats ---")
    r = agent.get_network_stats()
    print(r["result"])

    print("\n--- Battery ---")
    r = agent.get_battery()
    print(r["result"])

    print("\n--- Top Processes ---")
    r = agent.get_top_processes(5)
    print(r["result"])
    if r["ok"]:
        for p in r["data"]["by_cpu"][:5]:
            print(f"  {p.get('name','?')} | CPU: {p.get('cpu_percent',0):.1f}% | RAM: {p.get('memory_mb',0)} MB")

    print("\n--- WiFi Info ---")
    r = agent.get_wifi_info()
    print(r["result"])

    print("\n--- Ping Google ---")
    r = agent.ping("google.com")
    print(r["result"])

    print("\n--- Volume ---")
    r = agent.get_volume()
    print(r["result"])

    print("\n--- Screen Resolution ---")
    r = agent.get_screen_resolution()
    print(r["result"])

    print("\n--- Running Services (top 5) ---")
    r = agent.get_running_services()
    print(r["result"])
    if r["ok"]:
        for svc in r["data"]["services"][:5]:
            print(f"  {svc.get('name')} — {svc.get('display_name','')}")
