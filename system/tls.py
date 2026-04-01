"""
ARIA TLS / HTTPS Setup
======================
Generates a self-signed certificate for local HTTPS.
Enables HTTPS so ARIA API calls are encrypted on LAN.

Usage:
  from system.tls import ensure_cert
  cert, key = ensure_cert()
  uvicorn.run(app, host="0.0.0.0", port=8443, ssl_certfile=cert, ssl_keyfile=key)

The certificate is valid for 10 years and covers:
  - localhost
  - 127.0.0.1
  - The machine's local LAN IP (auto-detected)

On first run this will ask the OS to trust the cert (optional, can skip).
"""

import ipaddress
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("aria.tls")

CERT_DIR  = Path(r"C:\Users\chand\ai-remo\data\tls")
CERT_FILE = CERT_DIR / "aria.crt"
KEY_FILE  = CERT_DIR / "aria.key"

CERT_DIR.mkdir(parents=True, exist_ok=True)


def _local_ip() -> str:
    """Get the machine's primary LAN IP."""
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=2) as s:
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def generate_self_signed_cert(
    cert_path: Path = CERT_FILE,
    key_path:  Path = KEY_FILE,
    days:      int  = 3650,
) -> Tuple[str, str]:
    """
    Generate a self-signed TLS certificate using the `cryptography` library.
    Falls back to `openssl` CLI if cryptography is not installed.

    Returns (cert_path, key_path) as strings.
    """
    # Try cryptography library first (pure Python, already likely installed)
    try:
        return _generate_with_cryptography(cert_path, key_path, days)
    except ImportError:
        pass

    # Fallback: openssl CLI
    try:
        return _generate_with_openssl(cert_path, key_path, days)
    except Exception as e:
        raise RuntimeError(f"Cannot generate TLS cert: {e}. Install 'cryptography' package.")


def _generate_with_cryptography(
    cert_path: Path, key_path: Path, days: int
) -> Tuple[str, str]:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    lan_ip = _local_ip()

    # Generate RSA key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Build certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME,             "IN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME,   "Local"),
        x509.NameAttribute(NameOID.LOCALITY_NAME,            "ARIA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME,        "ARIA Local Assistant"),
        x509.NameAttribute(NameOID.COMMON_NAME,              "localhost"),
    ])

    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv4Address(lan_ip)),
    ])

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=days))
        .add_extension(san, critical=False)
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .sign(private_key, hashes.SHA256())
    )

    # Write key
    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)

    # Write cert
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"TLS cert generated: {cert_path} (valid {days} days, IPs: 127.0.0.1, {lan_ip})")
    return str(cert_path), str(key_path)


def _generate_with_openssl(
    cert_path: Path, key_path: Path, days: int
) -> Tuple[str, str]:
    lan_ip = _local_ip()
    # Write SAN config
    san_conf = CERT_DIR / "san.cnf"
    san_conf.write_text(f"""[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = localhost
O = ARIA Local Assistant
C = IN

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = {lan_ip}
""")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(key_path),
        "-out",    str(cert_path),
        "-days",   str(days),
        "-nodes",
        "-config", str(san_conf),
    ], check=True, capture_output=True)
    key_path.chmod(0o600)
    logger.info(f"TLS cert generated via openssl: {cert_path}")
    return str(cert_path), str(key_path)


def ensure_cert() -> Tuple[str, str]:
    """
    Return (cert_path, key_path), generating them if they don't exist.
    Thread-safe for multiple callers at startup.
    """
    if CERT_FILE.exists() and KEY_FILE.exists():
        # Check if still valid (not expired)
        try:
            from cryptography import x509
            cert_data = CERT_FILE.read_bytes()
            cert = x509.load_pem_x509_certificate(cert_data)
            now = datetime.now(timezone.utc)
            if cert.not_valid_after_utc > now + timedelta(days=30):
                return str(CERT_FILE), str(KEY_FILE)
            logger.info("TLS cert expiring soon, regenerating...")
        except Exception:
            return str(CERT_FILE), str(KEY_FILE)

    logger.info("Generating self-signed TLS certificate...")
    return generate_self_signed_cert()


def cert_info() -> dict:
    """Return certificate metadata."""
    if not CERT_FILE.exists():
        return {"exists": False}
    try:
        from cryptography import x509
        cert = x509.load_pem_x509_certificate(CERT_FILE.read_bytes())
        now  = datetime.now(timezone.utc)
        return {
            "exists":      True,
            "subject":     str(cert.subject),
            "not_before":  cert.not_valid_before_utc.isoformat(),
            "not_after":   cert.not_valid_after_utc.isoformat(),
            "days_left":   (cert.not_valid_after_utc - now).days,
            "cert_path":   str(CERT_FILE),
            "key_path":    str(KEY_FILE),
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}
