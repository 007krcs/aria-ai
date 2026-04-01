"""
ARIA Network Agent
Network diagnostics and operations — ping, traceroute, port scan, WiFi, connections, speed.
Dependencies: requests, psutil, socket (stdlib), subprocess (stdlib)
"""

import json
import re
import socket
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(result: str, data: Any = None) -> Dict:
    return {"ok": True, "result": result, "data": data}


def _err(result: str, data: Any = None) -> Dict:
    return {"ok": False, "result": result, "data": data}


def _run(cmd: list, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------

def ping(host: str, count: int = 4) -> Dict:
    """Ping a host. Returns avg latency and packet loss."""
    try:
        result = _run(["ping", "-n", str(count), host], timeout=30)
        output = result.stdout + result.stderr
        # Parse avg latency
        avg_ms = None
        loss_pct = None

        # "Average = 23ms" pattern (Windows)
        m = re.search(r"Average\s*=\s*(\d+)ms", output)
        if m:
            avg_ms = int(m.group(1))

        # "Lost = X (Y% loss)" pattern
        m2 = re.search(r"\((\d+)%\s+loss\)", output)
        if m2:
            loss_pct = int(m2.group(1))

        if avg_ms is None and loss_pct is None:
            return _err(f"Ping failed or host unreachable: {host}", {"raw": output})

        summary = f"Host: {host} | Avg latency: {avg_ms}ms | Packet loss: {loss_pct}%"
        return _ok(summary, {"host": host, "avg_ms": avg_ms, "loss_pct": loss_pct, "raw": output})
    except Exception as e:
        return _err(f"ping error: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# get_ip_info
# ---------------------------------------------------------------------------

def get_ip_info() -> Dict:
    """Return local IP, public IP, gateway, DNS servers."""
    try:
        info = {}

        # Local IP (connect trick)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            info["local_ip"] = s.getsockname()[0]
            s.close()
        except Exception:
            info["local_ip"] = "unknown"

        # Hostname
        info["hostname"] = socket.gethostname()

        # Gateway and DNS via ipconfig
        try:
            r = _run(["ipconfig", "/all"])
            lines = r.stdout.splitlines()
            gateways, dns_servers = [], []
            for line in lines:
                ll = line.lower()
                if "default gateway" in ll:
                    parts = line.split(":")
                    if len(parts) > 1:
                        gw = parts[-1].strip()
                        if gw and re.match(r"\d+\.\d+\.\d+\.\d+", gw):
                            gateways.append(gw)
                if "dns servers" in ll or "dns server" in ll:
                    parts = line.split(":")
                    if len(parts) > 1:
                        dns = parts[-1].strip()
                        if dns and re.match(r"\d+\.\d+\.\d+\.\d+", dns):
                            dns_servers.append(dns)
            info["gateway"] = gateways[0] if gateways else "unknown"
            info["dns_servers"] = dns_servers if dns_servers else ["unknown"]
        except Exception:
            info["gateway"] = "unknown"
            info["dns_servers"] = []

        # Public IP
        pub = get_public_ip()
        info["public_ip"] = pub["data"].get("ip") if pub["ok"] else "unknown"

        summary = (
            f"Local: {info['local_ip']} | Public: {info['public_ip']} | "
            f"Gateway: {info['gateway']} | DNS: {', '.join(info['dns_servers'])}"
        )
        return _ok(summary, info)
    except Exception as e:
        return _err(f"get_ip_info error: {e}")


# ---------------------------------------------------------------------------
# traceroute
# ---------------------------------------------------------------------------

def traceroute(host: str) -> Dict:
    """Run tracert to host and return hop list."""
    try:
        result = _run(["tracert", "-d", "-w", "1000", host], timeout=60)
        output = result.stdout
        hops = []
        for line in output.splitlines():
            m = re.match(
                r"\s*(\d+)\s+(?:(\d+)\s+ms\s+(\d+)\s+ms\s+(\d+)\s+ms|(\*\s*\*\s*\*))\s+([\d.]+|\*)",
                line
            )
            if m:
                hop_num = int(m.group(1))
                ip = m.group(6) or "*"
                latencies = [m.group(2), m.group(3), m.group(4)]
                latencies = [int(l) for l in latencies if l]
                avg_lat = round(sum(latencies) / len(latencies)) if latencies else None
                hops.append({"hop": hop_num, "ip": ip, "avg_ms": avg_lat})

        if not hops:
            return _err(f"No hops found for {host}", {"raw": output})
        summary = f"Traceroute to {host}: {len(hops)} hops"
        return _ok(summary, {"host": host, "hops": hops, "raw": output})
    except Exception as e:
        return _err(f"traceroute error: {e}")


# ---------------------------------------------------------------------------
# port_scan
# ---------------------------------------------------------------------------

def port_scan(host: str, ports: Optional[List[int]] = None) -> Dict:
    """Check if ports are open on a host."""
    if ports is None:
        ports = [80, 443, 22, 21, 8080]
    try:
        results = {}

        def _check(port):
            try:
                s = socket.create_connection((host, port), timeout=2)
                s.close()
                return port, True
            except Exception:
                return port, False

        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {ex.submit(_check, p): p for p in ports}
            for f in as_completed(futures):
                port, open_ = f.result()
                results[port] = open_

        open_ports = [p for p, o in results.items() if o]
        closed_ports = [p for p, o in results.items() if not o]
        summary = (
            f"Host: {host} | Open: {sorted(open_ports)} | "
            f"Closed: {sorted(closed_ports)}"
        )
        return _ok(summary, {"host": host, "open": open_ports, "closed": closed_ports, "all": results})
    except Exception as e:
        return _err(f"port_scan error: {e}")


# ---------------------------------------------------------------------------
# get_wifi_networks
# ---------------------------------------------------------------------------

def get_wifi_networks() -> Dict:
    """Scan available WiFi networks (Windows: netsh)."""
    try:
        result = _run(["netsh", "wlan", "show", "networks", "mode=bssid"], timeout=15)
        output = result.stdout
        networks = []
        current = {}
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("SSID") and "BSSID" not in line:
                if current:
                    networks.append(current)
                m = re.match(r"SSID\s+\d*\s*:\s*(.*)", line)
                current = {"ssid": m.group(1).strip() if m else "", "bssid": [], "signal": "", "auth": ""}
            elif "BSSID" in line:
                m = re.match(r"BSSID\s+\d*\s*:\s*(.*)", line)
                if m:
                    current.setdefault("bssid", []).append(m.group(1).strip())
            elif "Signal" in line:
                m = re.match(r"Signal\s*:\s*(.*)", line)
                if m:
                    current["signal"] = m.group(1).strip()
            elif "Authentication" in line:
                m = re.match(r"Authentication\s*:\s*(.*)", line)
                if m:
                    current["auth"] = m.group(1).strip()
        if current:
            networks.append(current)

        if not networks:
            return _err("No WiFi networks found (WiFi may be off or netsh unavailable)", {"raw": output})
        summary = f"Found {len(networks)} WiFi networks"
        return _ok(summary, {"networks": networks})
    except Exception as e:
        return _err(f"get_wifi_networks error: {e}")


# ---------------------------------------------------------------------------
# get_active_connections
# ---------------------------------------------------------------------------

def get_active_connections() -> Dict:
    """List active TCP/UDP connections with process names."""
    try:
        connections = []
        if _PSUTIL:
            for conn in psutil.net_connections(kind="inet"):
                try:
                    proc_name = psutil.Process(conn.pid).name() if conn.pid else "System"
                except Exception:
                    proc_name = f"PID:{conn.pid}"
                laddr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "-"
                raddr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "-"
                connections.append({
                    "proto": conn.type.name if hasattr(conn.type, "name") else str(conn.type),
                    "local": laddr,
                    "remote": raddr,
                    "status": conn.status,
                    "pid": conn.pid,
                    "process": proc_name,
                })
        else:
            # Fallback: netstat
            result = _run(["netstat", "-ano"], timeout=15)
            for line in result.stdout.splitlines()[4:]:
                parts = line.split()
                if len(parts) >= 4:
                    connections.append({
                        "proto": parts[0],
                        "local": parts[1],
                        "remote": parts[2],
                        "status": parts[3] if len(parts) > 4 else "",
                        "pid": parts[-1],
                        "process": "",
                    })

        established = [c for c in connections if "ESTABLISHED" in c.get("status", "")]
        summary = (
            f"Total connections: {len(connections)} | "
            f"Established: {len(established)}"
        )
        return _ok(summary, {"connections": connections, "count": len(connections)})
    except Exception as e:
        return _err(f"get_active_connections error: {e}")


# ---------------------------------------------------------------------------
# get_network_speed
# ---------------------------------------------------------------------------

def get_network_speed(interval: float = 1.0) -> Dict:
    """Bytes sent/received per second over the given sample interval."""
    try:
        if not _PSUTIL:
            return _err("psutil not available for network speed")
        before = psutil.net_io_counters()
        time.sleep(interval)
        after = psutil.net_io_counters()
        sent_ps = (after.bytes_sent - before.bytes_sent) / interval
        recv_ps = (after.bytes_recv - before.bytes_recv) / interval

        def _fmt(bps):
            if bps >= 1_000_000:
                return f"{bps/1_000_000:.2f} MB/s"
            if bps >= 1_000:
                return f"{bps/1_000:.1f} KB/s"
            return f"{bps:.0f} B/s"

        summary = f"Download: {_fmt(recv_ps)} | Upload: {_fmt(sent_ps)}"
        return _ok(summary, {
            "bytes_recv_per_sec": recv_ps,
            "bytes_sent_per_sec": sent_ps,
            "recv_human": _fmt(recv_ps),
            "sent_human": _fmt(sent_ps),
        })
    except Exception as e:
        return _err(f"get_network_speed error: {e}")


# ---------------------------------------------------------------------------
# dns_lookup
# ---------------------------------------------------------------------------

def dns_lookup(domain: str) -> Dict:
    """Resolve domain to IP addresses."""
    try:
        ips = socket.getaddrinfo(domain, None)
        unique_ips = list({info[4][0] for info in ips})
        summary = f"{domain} -> {', '.join(unique_ips)}"
        return _ok(summary, {"domain": domain, "ips": unique_ips})
    except socket.gaierror as e:
        return _err(f"DNS lookup failed for '{domain}': {e}")
    except Exception as e:
        return _err(f"dns_lookup error: {e}")


# ---------------------------------------------------------------------------
# whois_lookup
# ---------------------------------------------------------------------------

def whois_lookup(domain: str) -> Dict:
    """Basic WHOIS lookup via web (whois.domaintools.com or similar)."""
    try:
        if not _REQUESTS:
            # Try whois command
            result = _run(["whois", domain], timeout=15)
            if result.returncode == 0:
                return _ok(result.stdout[:2000], {"raw": result.stdout})
            return _err("requests not available and whois command not found")

        url = f"https://www.whois.com/whois/{domain}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "ARIA/1.0"})
        if resp.status_code != 200:
            return _err(f"WHOIS lookup HTTP {resp.status_code}")

        text = resp.text
        # Extract meaningful section
        import html
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = html.unescape(clean)
        clean = re.sub(r"\s+", " ", clean)

        # Find WHOIS data block
        m = re.search(r"(Domain Name:.*?)(?:DNSSEC|--)", clean, re.DOTALL | re.IGNORECASE)
        snippet = m.group(1).strip()[:1500] if m else clean[:1500]
        return _ok(snippet, {"domain": domain, "raw_snippet": snippet})
    except Exception as e:
        return _err(f"whois_lookup error: {e}")


# ---------------------------------------------------------------------------
# check_internet
# ---------------------------------------------------------------------------

def check_internet() -> Dict:
    """Test internet connectivity by hitting multiple endpoints."""
    endpoints = [
        ("8.8.8.8", 53),
        ("1.1.1.1", 53),
        ("google.com", 80),
    ]
    connected = False
    results = []
    for host, port in endpoints:
        try:
            s = socket.create_connection((host, port), timeout=3)
            s.close()
            connected = True
            results.append({"host": f"{host}:{port}", "reachable": True})
        except Exception:
            results.append({"host": f"{host}:{port}", "reachable": False})

    if connected:
        return _ok("Internet is connected", {"checks": results})
    return _err("No internet connectivity detected", {"checks": results})


# ---------------------------------------------------------------------------
# get_public_ip
# ---------------------------------------------------------------------------

def get_public_ip() -> Dict:
    """Get public IP address."""
    try:
        if _REQUESTS:
            for url in ["https://api.ipify.org?format=json", "https://ipinfo.io/json"]:
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200:
                        data = r.json()
                        ip = data.get("ip", "")
                        if ip:
                            return _ok(f"Public IP: {ip}", {"ip": ip, "info": data})
                except Exception:
                    continue
        # Fallback: nslookup myip.opendns.com resolver1.opendns.com
        result = _run(
            ["nslookup", "myip.opendns.com", "resolver1.opendns.com"], timeout=10
        )
        m = re.search(r"Address:\s*([\d.]+)", result.stdout)
        if m:
            ip = m.group(1)
            return _ok(f"Public IP: {ip}", {"ip": ip})
        return _err("Could not determine public IP")
    except Exception as e:
        return _err(f"get_public_ip error: {e}")


# ---------------------------------------------------------------------------
# speed_test_quick
# ---------------------------------------------------------------------------

def speed_test_quick() -> Dict:
    """
    Quick bandwidth estimate by downloading a small test file from a CDN
    and measuring throughput. Falls back to network_speed() on failure.
    """
    try:
        if not _REQUESTS:
            return get_network_speed(interval=2.0)

        test_url = "https://speed.cloudflare.com/__down?bytes=5000000"  # 5 MB
        start = time.time()
        r = requests.get(test_url, timeout=30, stream=True)
        total = 0
        for chunk in r.iter_content(chunk_size=65536):
            total += len(chunk)
        elapsed = time.time() - start
        if elapsed == 0:
            return _err("Speed test completed instantly (invalid)")
        bps = total / elapsed
        mbps = bps * 8 / 1_000_000
        summary = f"Download speed: {mbps:.2f} Mbps ({total/1e6:.2f} MB in {elapsed:.1f}s)"
        return _ok(summary, {"mbps": mbps, "bytes": total, "seconds": elapsed})
    except Exception as e:
        # Fallback
        return get_network_speed(interval=2.0)


# ---------------------------------------------------------------------------
# http_request
# ---------------------------------------------------------------------------

def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    body: Optional[Any] = None,
) -> Dict:
    """Make an HTTP request and return status, headers, and body."""
    try:
        if not _REQUESTS:
            return _err("requests library not available")
        headers = headers or {}
        method = method.upper()
        kwargs = {"headers": headers, "timeout": 30}
        if body:
            if isinstance(body, dict):
                kwargs["json"] = body
            else:
                kwargs["data"] = body

        r = requests.request(method, url, **kwargs)
        content_type = r.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                resp_body = r.json()
            except Exception:
                resp_body = r.text[:4000]
        else:
            resp_body = r.text[:4000]

        summary = f"{method} {url} -> HTTP {r.status_code}"
        return _ok(summary, {
            "status_code": r.status_code,
            "headers": dict(r.headers),
            "body": resp_body,
            "url": r.url,
        })
    except Exception as e:
        return _err(f"http_request error: {e}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== check_internet ===")
    print(json.dumps(check_internet(), indent=2))
    print("=== get_ip_info ===")
    print(json.dumps(get_ip_info(), indent=2))
    print("=== ping google.com ===")
    print(json.dumps(ping("google.com", count=2), indent=2))
    print("=== dns_lookup ===")
    print(json.dumps(dns_lookup("google.com"), indent=2))
    print("=== port_scan ===")
    print(json.dumps(port_scan("google.com", ports=[80, 443]), indent=2))
