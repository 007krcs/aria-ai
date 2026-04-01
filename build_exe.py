"""
ARIA Build Script
=================
Creates ARIA.exe (Windows launcher) using PyInstaller.

Usage:
    python build_exe.py           # build ARIA.exe
    python build_exe.py --clean   # clean build artifacts first
    python build_exe.py --test    # build + run quick smoke test
"""

import sys
import os
import subprocess
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def run(cmd, **kwargs):
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def clean():
    print("Cleaning build artifacts...")
    for d in ["build", "dist/__pycache__"]:
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p)
            print(f"  Removed {d}/")
    # Remove stale ARIA.exe from dist (keep installer subfolder)
    exe = ROOT / "dist" / "ARIA.exe"
    if exe.exists():
        exe.unlink()
        print("  Removed dist/ARIA.exe")


def ensure_pyinstaller():
    try:
        import PyInstaller  # noqa
    except ImportError:
        print("Installing PyInstaller...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller", "--quiet"])


def build():
    print("\n" + "=" * 60)
    print("  Building ARIA.exe")
    print("=" * 60 + "\n")

    ensure_pyinstaller()

    # PyInstaller flags
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(ROOT / "ARIA.spec"),
        "--clean",
        "--noconfirm",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--log-level", "WARN",
    ]
    run(cmd)

    exe = ROOT / "dist" / "ARIA.exe"
    if exe.exists():
        size_mb = exe.stat().st_size / 1024 / 1024
        print(f"\n  [OK]  Built: dist/ARIA.exe  ({size_mb:.1f} MB)")
        return True
    else:
        print("\n  [FAIL]  Build failed -- ARIA.exe not found in dist/")
        return False


def test_exe():
    exe = ROOT / "dist" / "ARIA.exe"
    if not exe.exists():
        print("ARIA.exe not found -- run build first.")
        return False

    print("\nTesting ARIA.exe (will exit after 3 seconds)...")
    import time
    proc = subprocess.Popen(
        [str(exe)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(ROOT),
    )
    time.sleep(3)
    if proc.poll() is None:
        proc.terminate()
        print("  [OK]  ARIA.exe launched and ran for 3 seconds without crashing.")
        return True
    else:
        out, err = proc.communicate()
        print(f"  [FAIL]  Exited early with code {proc.returncode}")
        if err:
            print(f"  stderr: {err.decode()[:200]}")
        return False


def print_next_steps():
    print("""
Next steps
----------
  1. Run ARIA:         double-click  dist\\ARIA.exe
                       or:           dist\\ARIA.exe  (from terminal)

  2. Full installer:   Install Inno Setup 6 from https://jrsoftware.org/isinfo.php
                       Open  installer.iss  ->  Build  ->  dist\\installer\\ARIA_Setup_1.0.0.exe

  3. Distribute:       Share  dist\\ARIA.exe  (requires Python + .venv on target machine)
                       OR share  dist\\installer\\ARIA_Setup_1.0.0.exe  (full installer)
""")


def main():
    parser = argparse.ArgumentParser(description="Build ARIA.exe")
    parser.add_argument("--clean", action="store_true", help="Clean before build")
    parser.add_argument("--test",  action="store_true", help="Test after build")
    args = parser.parse_args()

    if args.clean:
        clean()

    ok = build()
    if ok:
        if args.test:
            test_exe()
        print_next_steps()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
