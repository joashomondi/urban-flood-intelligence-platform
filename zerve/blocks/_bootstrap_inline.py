# Copy this block to the TOP of every Zerve canvas block (01–06).
# Each block runs in its own process, so ``src`` must be bootstrapped every time.
import io
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_REPO_ZIP = (
    "https://github.com/joashomondi/urban-flood-intelligence-platform/"
    "archive/refs/heads/main.zip"
)


def _engine_cache() -> Path:
    fixed = Path("/tmp/ufip_engine")
    if fixed.parent.exists():
        return fixed
    return Path(tempfile.gettempdir()) / "ufip_engine"


def _ensure_engine() -> None:
    try:
        import src  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    cache = _engine_cache()
    if not (cache / "src" / "__init__.py").exists():
        print("[engine] Downloading from GitHub (zip) …")
        raw = urllib.request.urlopen(_REPO_ZIP, timeout=180).read()
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(cache)
        extracted = next(cache.glob("urban-flood-intelligence-platform-*"), None)
        if extracted is None:
            raise RuntimeError("Repo folder not found in downloaded zip.")
        for item in extracted.iterdir():
            dest = cache / item.name
            if not dest.exists():
                item.rename(dest)
        extracted.rmdir()

    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))

    try:
        import src  # noqa: F401
        return
    except ModuleNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", str(cache)],
            capture_output=True,
        )
        if str(cache) not in sys.path:
            sys.path.insert(0, str(cache))
        import src  # noqa: F401


_ensure_engine()
