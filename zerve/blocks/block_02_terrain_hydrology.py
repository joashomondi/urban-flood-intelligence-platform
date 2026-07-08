# =============================================================================
# ZERVE BLOCK 02 — TERRAIN & HYDROLOGY ENGINE
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : dem            (from block_01_ingest)
# OUTPUTS (downstream) : terrain, hydro, slope, hillshade
# =============================================================================
# Derives slope / aspect / roughness / curvature / hillshade, then the D8
# hydrology stack: filled DEM, flow direction, flow accumulation, TWI and
# drainage density.
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

from src import terrain_analysis, hydrology

try:                          # use the DEM inherited from the upstream block
    dem
except NameError:             # fallback so the block also runs standalone
    from src import data_loader, utils
    utils.ensure_dirs()
    dem = data_loader.load_dem().elevation

terrain = terrain_analysis.analyze_terrain(dem)
hydro = hydrology.analyze_hydrology(dem, terrain["slope"])

slope = terrain["slope"]
hillshade = terrain["hillshade"]

print(f"[02] slope mean {slope.mean():.2f}°  |  TWI mean {hydro['twi'].mean():.2f}  "
      f"|  flow-acc max {hydro['flow_accumulation'].max():.0f}")
