# =============================================================================
# ZERVE BLOCK 01 — DATA INGESTION
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : none  (this is the first node in the DAG)
# OUTPUTS (downstream) : grid, dem, rainfall_field, rainfall_ts,
#                        drainage, flood_points
# =============================================================================
# Ingests the DEM + rainfall + vector layers for the study area (Nairobi).
# Uses a real SRTM GeoTIFF / CHIRPS raster from data/raw if present, else
# synthesises physically-plausible surfaces so the canvas runs anywhere.

# Bootstrap: Zerve sandboxes often lack ``git``, so ``pip install git+https://…``
# fails. Download the repo as a ZIP and install locally (or add to sys.path).
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


def _ensure_engine() -> None:
    try:
        import src  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    cache = Path("/tmp/ufip_engine") if Path("/tmp").exists() else Path(tempfile.gettempdir()) / "ufip_engine"
    if not (cache / "src" / "__init__.py").exists():
        print("[01] Downloading engine from GitHub (zip, no git) …")
        raw = urllib.request.urlopen(_REPO_ZIP, timeout=180).read()
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
        print("[01] Engine ready.")
        return
    except ModuleNotFoundError:
        pass

    print("[01] Installing engine (local pip, no git) …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", str(cache)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[01] pip install failed; using sys.path fallback")
        if str(cache) not in sys.path:
            sys.path.insert(0, str(cache))

    import src  # noqa: F401
    print("[01] Engine ready.")


_ensure_engine()

from src import data_loader, utils

utils.ensure_dirs()

_ingest = data_loader.ingest_all(base_rainfall_mm=utils.RAINFALL_REFERENCE_MM)

grid = _ingest["grid"]
dem = grid.elevation
rainfall_field = _ingest["rainfall_field"]
rainfall_ts = _ingest["rainfall_timeseries"]
drainage = _ingest["drainage"]
flood_points = _ingest["flood_points"]

print(f"[01] DEM {dem.shape}  elev [{dem.min():.0f}, {dem.max():.0f}] m")
print(f"[01] Rainfall mean {rainfall_field.mean():.1f} mm  |  "
      f"{len(drainage)} drainage lines, {len(flood_points)} flood points")
