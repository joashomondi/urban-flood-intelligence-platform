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

# Bootstrap: install the engine if the environment build skipped the git+ line.
import subprocess
import sys

_ENGINE = "git+https://github.com/joashomondi/urban-flood-intelligence-platform.git"
try:
    import src  # noqa: F401
except ModuleNotFoundError:
    print("[01] Installing flood-intelligence engine from GitHub …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", _ENGINE],
        stdout=subprocess.DEVNULL,
    )
    print("[01] Engine installed.")

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
