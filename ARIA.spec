# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for ARIA.exe launcher.

This builds a SMALL exe (~5-8 MB) that bootstraps the ARIA environment
and launches server.py as a subprocess. It does NOT bundle torch/chromadb
— those stay in the .venv so the exe stays lean.

Build:
    pyinstaller ARIA.spec
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH)  # noqa: F821  (SPECPATH is injected by PyInstaller)

block_cipher = None

a = Analysis(
    [str(ROOT / "aria_launcher.py")],
    # Empty pathex — do NOT scan the project root or venv
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude EVERYTHING that isn't pure stdlib
    excludes=[
        "torch", "torchvision", "torchaudio", "torch_cuda",
        "transformers", "tokenizers", "sentence_transformers",
        "chromadb", "hnswlib", "numpy", "pandas",
        "sklearn", "scipy", "matplotlib", "PIL", "cv2",
        "tensorflow", "keras", "jax",
        "langchain", "langchain_core", "langchain_community",
        "langgraph", "llama_index", "openai",
        "fastapi", "uvicorn", "starlette", "pydantic",
        "requests", "httpx", "aiohttp",
        "sqlalchemy", "chromadb", "redis",
        "rich", "click", "typer",
        "sympy", "mpmath", "numba",
        "bitsandbytes", "xformers", "safetensors", "huggingface_hub",
        "grpc", "google", "protobuf",
        "multiprocessing", "concurrent.futures",
        "email", "html", "http", "xml", "xmlrpc",
        "unittest", "doctest", "pdb", "profile",
        "tkinter", "wx", "gtk",
        "IPython", "jupyter", "notebook",
        "pkg_resources", "setuptools", "pip", "wheel",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ARIA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,          # compress with UPX if available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,      # keep console visible so users see logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Use ARIA icon if available
    icon=str(ROOT / "app" / "public" / "favicon.ico") if (ROOT / "app" / "public" / "favicon.ico").exists() else None,
    version=None,
)
