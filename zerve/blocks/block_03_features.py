# =============================================================================
# ZERVE BLOCK 03 — FEATURE ENGINEERING
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : terrain, hydro, rainfall_field
# OUTPUTS (downstream) : indicators, feature_table, labels
# =============================================================================
# Builds the six normalised flood indicators and physically-motivated flood
# labels, then flattens the raster stack into a tidy per-cell training table.
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

from src import feature_engineering

try:                          # inherited from blocks 01 + 02
    terrain; hydro; rainfall_field
except NameError:             # fallback: recompute the upstream chain
    from src.pipeline import build_state
    _s = build_state(train=False)
    terrain, hydro, rainfall_field = _s["terrain"], _s["hydro"], _s["rainfall_field"]

_feats = feature_engineering.engineer_features(terrain, hydro, rainfall_field)

indicators = _feats["indicators"]
feature_table = _feats["table"]
labels = _feats["labels"]

print(f"[03] indicators={list(indicators)}")
print(f"[03] feature_table {feature_table.shape}  |  "
      f"positive rate {100 * feature_table['flood_label'].mean():.1f}%")
