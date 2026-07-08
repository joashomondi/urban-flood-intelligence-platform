"""
bootstrap.py
============
One-shot engine bootstrap for Zerve blocks.

Zerve sandboxes often lack ``git``, so ``pip install git+https://...`` fails.
This module downloads the repo as a ZIP, installs locally (or adds to
``sys.path``), then ``from src import ...`` works everywhere.
"""
from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ZIP = (
    "https://github.com/joashomondi/urban-flood-intelligence-platform/"
    "archive/refs/heads/main.zip"
)


def engine_cache() -> Path:
    """Shared cache path so every Zerve block sees the same engine."""
    fixed = Path("/tmp/ufip_engine")
    if fixed.parent.exists():
        return fixed
    return Path(tempfile.gettempdir()) / "ufip_engine"


def ensure_engine(tag: str = "bootstrap") -> None:
    """Make ``import src`` succeed on Zerve (idempotent)."""
    try:
        import src  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    cache = engine_cache()
    src_init = cache / "src" / "__init__.py"

    if not src_init.exists():
        print(f"[{tag}] Downloading engine from GitHub (zip, no git) …")
        raw = urllib.request.urlopen(REPO_ZIP, timeout=180).read()
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(cache)
        extracted = next(cache.glob("urban-flood-intelligence-platform-*"), None)
        if extracted is None:
            raise RuntimeError("Download succeeded but repo folder not found in zip.")
        for item in extracted.iterdir():
            dest = cache / item.name
            if dest.exists():
                continue
            item.rename(dest)
        extracted.rmdir()

    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))

    try:
        import src  # noqa: F401
        print(f"[{tag}] Engine ready (cached).")
        return
    except ModuleNotFoundError:
        pass

    print(f"[{tag}] Installing engine (local pip, no git) …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", str(cache)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[{tag}] pip install failed; using sys.path fallback")
        if result.stderr:
            print(result.stderr[-500:])

    import src  # noqa: F401
    print(f"[{tag}] Engine ready.")
