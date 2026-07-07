"""
data_loader.py
==============
Data ingestion layer for the Urban Flood Intelligence Platform.

Responsibilities
----------------
* Ingest a Digital Elevation Model (DEM) for the study area.
* Ingest / synthesise a rainfall field (proxy for CHIRPS 24h accumulation).
* Provide drainage-network and historical flood-point layers.
* Standardise CRS, clip to the study bounds and handle missing values.

Real-data first, synthetic fallback
-----------------------------------
On Zerve (or any fresh clone) large SRTM/CHIRPS rasters are usually *not*
present. To keep the pipeline **fully reproducible without network access**,
this module transparently synthesises a physically-plausible DEM and rainfall
surface for Nairobi when no GeoTIFF is found in ``data/raw``.

Drop a real DEM at ``data/raw/dem.tif`` (EPSG:4326 or any CRS) and it will be
loaded, reprojected and clipped automatically instead of the synthetic one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import utils
from .utils import STUDY_AREA, RAW_DIR, PROCESSED_DIR, get_logger, get_rng

log = get_logger("ufip.data")

# Optional heavy geospatial deps - guarded so the core still runs without them.
try:  # pragma: no cover - environment dependent
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _HAS_RASTERIO = False

try:  # pragma: no cover
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    _HAS_GEOPANDAS = True
except Exception:  # pragma: no cover
    _HAS_GEOPANDAS = False


# ---------------------------------------------------------------------------
# Grid geometry helper
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    """Lightweight description of the analysis raster grid.

    Holds the elevation array plus the coordinate vectors needed to convert
    between array indices and geographic (lon/lat) coordinates. This keeps the
    whole pipeline usable even when rasterio is unavailable.
    """
    elevation: np.ndarray            # (H, W) metres above sea level
    lons: np.ndarray                 # (W,) cell-centre longitudes
    lats: np.ndarray                 # (H,) cell-centre latitudes (north -> south)
    bounds: tuple                    # (min_lon, min_lat, max_lon, max_lat)
    crs: str = "EPSG:4326"

    @property
    def shape(self) -> tuple:
        return self.elevation.shape

    @property
    def transform(self):
        """Affine transform (requires rasterio)."""
        if not _HAS_RASTERIO:
            raise RuntimeError("rasterio not available for affine transform")
        h, w = self.elevation.shape
        return from_bounds(*self.bounds, w, h)

    def lonlat_grid(self) -> tuple:
        """Return 2-D meshgrids of longitude and latitude."""
        return np.meshgrid(self.lons, self.lats)

    def cell_of(self, lat: float, lon: float) -> tuple:
        """Return (row, col) of the grid cell containing a lat/lon point."""
        row = int(np.clip(np.argmin(np.abs(self.lats - lat)), 0, self.shape[0] - 1))
        col = int(np.clip(np.argmin(np.abs(self.lons - lon)), 0, self.shape[1] - 1))
        return row, col


# ---------------------------------------------------------------------------
# DEM ingestion
# ---------------------------------------------------------------------------
def _coord_vectors(bounds, n) -> tuple:
    min_lon, min_lat, max_lon, max_lat = bounds
    # Cell centres. Latitudes ordered north -> south to match raster row order.
    lons = np.linspace(min_lon, max_lon, n)
    lats = np.linspace(max_lat, min_lat, n)
    return lons, lats


def _synthetic_dem(seed: Optional[int] = None) -> np.ndarray:
    """Generate a physically-plausible DEM for the Nairobi window.

    Nairobi's terrain slopes broadly from the cool, high western highlands
    (~1900 m, Karen / Ngong foothills) down toward the drier eastern plains
    (~1500 m, Embakasi / Athi basin). We model:

      * a smooth regional west->east / north->south elevation gradient,
      * several incised river valleys (Nairobi, Ngong, Mathare rivers),
      * fractal (multi-octave) roughness for realistic micro-topography.
    """
    rng = get_rng(seed)
    n = STUDY_AREA.grid_size
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)          # normalised 0..1 grids

    # Regional gradient: high in the west (low x) & north, low in the east.
    base = 1900.0 - 380.0 * xx - 120.0 * yy

    # Broad plateau undulations (low-frequency sinusoids).
    base += 40.0 * np.sin(2.2 * np.pi * xx) * np.cos(1.7 * np.pi * yy)
    base += 25.0 * np.sin(3.5 * np.pi * yy + 0.6)

    # Multi-octave value noise for micro-topography.
    noise = np.zeros((n, n))
    for octave in range(1, 6):
        freq = 2 ** octave
        amp = 30.0 / freq
        coarse = rng.normal(0, 1, size=(freq + 1, freq + 1))
        # bilinear upsample the coarse noise to full resolution
        yi = np.linspace(0, freq, n)
        xi = np.linspace(0, freq, n)
        y0 = np.floor(yi).astype(int).clip(0, freq - 1)
        x0 = np.floor(xi).astype(int).clip(0, freq - 1)
        fy = (yi - y0)[:, None]
        fx = (xi - x0)[None, :]
        c00 = coarse[np.ix_(y0, x0)]
        c01 = coarse[np.ix_(y0, x0 + 1)]
        c10 = coarse[np.ix_(y0 + 1, x0)]
        c11 = coarse[np.ix_(y0 + 1, x0 + 1)]
        top = c00 * (1 - fx) + c01 * fx
        bot = c10 * (1 - fx) + c11 * fx
        noise += amp * (top * (1 - fy) + bot * fy)
    base += noise

    # Carve river valleys as Gaussian troughs along polylines.
    def carve(points, depth, width):
        for (r0, c0), (r1, c1) in zip(points[:-1], points[1:]):
            steps = int(max(abs(r1 - r0), abs(c1 - c0)) * 1.5) + 2
            rs = np.linspace(r0, c0 * 0 + r0, steps) if steps == 1 else np.linspace(r0, r1, steps)
            cs = np.linspace(c0, c1, steps)
            for rr, cc in zip(rs, cs):
                dr = yy * (n - 1) - rr
                dc = xx * (n - 1) - cc
                dist2 = dr ** 2 + dc ** 2
                base[:] -= depth * np.exp(-dist2 / (2 * width ** 2))

    # Valleys drain broadly toward the east/south-east (lower ground).
    carve([(30, 20), (55, 70), (70, 120), (95, 150)], depth=45, width=3.5)
    carve([(15, 60), (45, 90), (80, 130), (110, 158)], depth=35, width=3.0)
    carve([(90, 10), (110, 55), (130, 110), (150, 150)], depth=30, width=3.0)

    return base.astype("float32")


def _load_real_dem(path: Path) -> Optional[Grid]:  # pragma: no cover
    """Load, reproject-to-EPSG:4326 and clip a real DEM GeoTIFF."""
    if not (_HAS_RASTERIO and path.exists()):
        return None
    try:
        with rasterio.open(path) as src:
            dst_crs = "EPSG:4326"
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            dem = np.empty((height, width), dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=dem,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        # Clip to study bounds by resampling to the canonical grid.
        min_lon, min_lat, max_lon, max_lat = STUDY_AREA.bounds
        n = STUDY_AREA.grid_size
        lons, lats = _coord_vectors(STUDY_AREA.bounds, n)
        # Nearest-index sampling from the loaded array (kept simple & dependency free).
        src_lons = np.linspace(transform.c, transform.c + transform.a * width, width)
        src_lats = np.linspace(transform.f, transform.f + transform.e * height, height)
        col_idx = np.clip(np.searchsorted(src_lons, lons), 0, width - 1)
        row_idx = np.clip(np.searchsorted(src_lats[::-1], lats[::-1])[::-1], 0, height - 1)
        clipped = dem[np.ix_(row_idx, col_idx)]
        clipped = np.where(np.isfinite(clipped), clipped, np.nan)
        log.info("Loaded real DEM from %s -> grid %s", path.name, clipped.shape)
        return Grid(clipped.astype("float32"), lons, lats, STUDY_AREA.bounds)
    except Exception as exc:
        log.warning("Failed to load real DEM (%s); falling back to synthetic.", exc)
        return None


def load_dem(force_synthetic: bool = False, seed: Optional[int] = None) -> Grid:
    """Return the study-area DEM as a :class:`Grid`.

    Tries ``data/raw/dem.tif`` first, then synthesises a realistic surface.
    Missing values are filled by nearest-neighbour interpolation.
    """
    utils.ensure_dirs()
    grid: Optional[Grid] = None
    if not force_synthetic:
        grid = _load_real_dem(RAW_DIR / "dem.tif")

    if grid is None:
        log.info("No real DEM found - synthesising DEM for %s.", STUDY_AREA.name)
        elev = _synthetic_dem(seed)
        lons, lats = _coord_vectors(STUDY_AREA.bounds, STUDY_AREA.grid_size)
        grid = Grid(elev, lons, lats, STUDY_AREA.bounds)

    grid.elevation = _fill_nan(grid.elevation)
    return grid


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs with the nearest finite value (simple, dependency-free)."""
    a = arr.astype("float32")
    mask = ~np.isfinite(a)
    if mask.any():
        finite_mean = np.nanmean(a) if np.isfinite(a).any() else 0.0
        a[mask] = finite_mean
    return a


# ---------------------------------------------------------------------------
# Rainfall ingestion
# ---------------------------------------------------------------------------
def _load_real_rainfall(path: Path) -> Optional[np.ndarray]:  # pragma: no cover
    """Load a real CHIRPS rainfall GeoTIFF and resample to the analysis grid.

    Accepts any single-band raster (CHIRPS daily/monthly accumulation in mm).
    Reprojects to EPSG:4326, clips to the study bounds and nearest-samples onto
    the canonical grid so it aligns cell-for-cell with the DEM.
    """
    if not (_HAS_RASTERIO and path.exists()):
        return None
    try:
        n = STUDY_AREA.grid_size
        lons, lats = _coord_vectors(STUDY_AREA.bounds, n)
        with rasterio.open(path) as src:
            src_arr = src.read(1).astype("float64")
            nodata = src.nodata
            if nodata is not None:
                src_arr[src_arr == nodata] = np.nan
            src_arr[src_arr < 0] = np.nan  # CHIRPS uses -9999 for no-data

            b = src.bounds
            src_lons = np.linspace(b.left, b.right, src.width)
            src_lats = np.linspace(b.top, b.bottom, src.height)
            col_idx = np.clip(np.searchsorted(src_lons, lons), 0, src.width - 1)
            row_idx = np.clip(
                src.height - 1 - np.searchsorted(src_lats[::-1], lats[::-1])[::-1],
                0, src.height - 1)
            field = src_arr[np.ix_(row_idx, col_idx)]

        field = _fill_nan(field.astype("float32"))
        log.info("Loaded real CHIRPS rainfall from %s -> grid %s (mean=%.1f mm)",
                 path.name, field.shape, float(np.nanmean(field)))
        return field
    except Exception as exc:
        log.warning("Failed to load real rainfall (%s); using synthetic field.", exc)
        return None


def load_rainfall_field(base_mm: float = utils.RAINFALL_REFERENCE_MM,
                        seed: Optional[int] = None,
                        rescale_real: bool = True) -> np.ndarray:
    """Return a 2-D rainfall accumulation surface (mm / 24h).

    Tries a real CHIRPS raster at ``data/raw/rainfall.tif`` first; otherwise
    synthesises a spatially-correlated proxy where convective storm cells
    produce heavier rainfall over the western highlands, tapering east.

    ``base_mm`` sets the areal mean so the dashboard slider can drive scenarios.
    When a real field is used and ``rescale_real`` is True, its *spatial pattern*
    is preserved but its mean is rescaled to ``base_mm`` so the slider still
    modulates storm intensity.
    """
    real = _load_real_rainfall(RAW_DIR / "rainfall.tif")
    if real is not None:
        if rescale_real and np.nanmean(real) > 1e-6:
            real = real / float(np.nanmean(real)) * base_mm
        return real.astype("float32")

    rng = get_rng(seed)
    n = STUDY_AREA.grid_size
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)

    # Orographic gradient: more rain over higher western ground.
    field = 1.0 + 0.55 * (1 - xx) + 0.20 * (1 - yy)

    # A couple of moving convective cells (Gaussian blobs).
    for (cy, cx, sig, amp) in [(0.35, 0.30, 0.18, 0.9),
                               (0.65, 0.55, 0.22, 0.7),
                               (0.50, 0.80, 0.15, 0.5)]:
        field += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)))

    field += rng.normal(0, 0.05, size=(n, n))
    field = np.clip(field, 0.1, None)
    field = field / field.mean() * base_mm      # rescale to requested mean
    return field.astype("float32")


def build_rainfall_timeseries(days: int = 120, seed: Optional[int] = None) -> pd.DataFrame:
    """Synthesise a daily areal-mean rainfall time series (mm/day).

    Captures Nairobi's bimodal rainfall regime: the "long rains" (Mar-May)
    and "short rains" (Oct-Dec) with stochastic storm spikes.
    """
    rng = get_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    doy = dates.dayofyear.to_numpy()
    seasonal = (18 * np.exp(-((doy - 105) ** 2) / (2 * 28 ** 2)) +   # long rains
                14 * np.exp(-((doy - 315) ** 2) / (2 * 26 ** 2)))    # short rains
    baseline = 2.0 + seasonal
    storms = rng.gamma(shape=1.2, scale=6.0, size=days) * (rng.random(days) < 0.28)
    rain = np.clip(baseline + storms, 0, None)
    return pd.DataFrame({"date": dates, "rainfall_mm": np.round(rain, 2)})


# ---------------------------------------------------------------------------
# Vector layers: drainage network & historical flood points
# ---------------------------------------------------------------------------
def build_drainage_network(grid: Grid, n_lines: int = 6, seed: Optional[int] = None):
    """Return a GeoDataFrame of synthetic drainage lines (or a plain DataFrame).

    In a real deployment this would be OSM ``waterway`` features. Here we trace
    steepest-descent paths from random high points so the network is consistent
    with the DEM.
    """
    from .terrain_analysis import steepest_descent_path  # local import avoids cycle
    rng = get_rng(seed)
    n = grid.shape[0]
    lines = []
    for _ in range(n_lines):
        r0 = int(rng.integers(0, n // 2))
        c0 = int(rng.integers(0, n))
        path = steepest_descent_path(grid.elevation, (r0, c0), max_len=n)
        if len(path) < 3:
            continue
        coords = [(grid.lons[c], grid.lats[r]) for r, c in path]
        lines.append(coords)

    records = [{"drain_id": i, "n_points": len(c), "coords": c}
               for i, c in enumerate(lines)]
    df = pd.DataFrame(records)

    if _HAS_GEOPANDAS and lines:
        geom = [LineString(c) for c in lines]
        gdf = gpd.GeoDataFrame(df.drop(columns=["coords"]), geometry=geom,
                               crs="EPSG:4326")
        return gdf
    return df


def build_flood_points(grid: Grid, n_points: int = 60, seed: Optional[int] = None) -> pd.DataFrame:
    """Synthesise historical flood-incident points biased toward low, flat ground.

    Represents the kind of point layer you'd get from emergency-response call
    logs or media-reported flooding. Used only for map context / validation.
    """
    rng = get_rng(seed)
    elev = grid.elevation
    # Probability of an incident inversely related to normalised elevation.
    p = utils.normalize(elev, invert=True) ** 2
    p = p / p.sum()
    idx = rng.choice(p.size, size=n_points, replace=False, p=p.ravel())
    rows, cols = np.unravel_index(idx, elev.shape)
    lat = grid.lats[rows] + rng.normal(0, grid_res_deg() * 0.3, n_points)
    lon = grid.lons[cols] + rng.normal(0, grid_res_deg() * 0.3, n_points)
    severity = rng.integers(1, 6, n_points)
    return pd.DataFrame({
        "flood_id": np.arange(n_points),
        "lat": np.round(lat, 6),
        "lon": np.round(lon, 6),
        "severity": severity,
        "elevation_m": np.round(elev[rows, cols], 1),
    })


def grid_res_deg() -> float:
    return STUDY_AREA.height_deg / STUDY_AREA.grid_size


# ---------------------------------------------------------------------------
# GeoTIFF export helper
# ---------------------------------------------------------------------------
def save_geotiff(array: np.ndarray, path: Path, grid: Grid) -> Optional[Path]:  # pragma: no cover
    """Write a single-band GeoTIFF if rasterio is present, else a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_RASTERIO:
        h, w = array.shape
        with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w, count=1,
            dtype="float32", crs=grid.crs, transform=grid.transform,
        ) as dst:
            dst.write(array.astype("float32"), 1)
        return path
    np.save(path.with_suffix(".npy"), array.astype("float32"))
    return path.with_suffix(".npy")


# ---------------------------------------------------------------------------
# One-shot ingestion bundle
# ---------------------------------------------------------------------------
def ingest_all(base_rainfall_mm: float = utils.RAINFALL_REFERENCE_MM,
               seed: Optional[int] = None) -> dict:
    """Run the full ingestion step and return every raw layer in a dict."""
    log.info("=== STEP 1: DATA INGESTION ===")
    grid = load_dem(seed=seed)
    log.info("DEM ready: shape=%s  elev range=[%.0f, %.0f] m",
             grid.shape, np.nanmin(grid.elevation), np.nanmax(grid.elevation))

    rainfall = load_rainfall_field(base_mm=base_rainfall_mm, seed=seed)
    log.info("Rainfall field: mean=%.1f mm  range=[%.1f, %.1f]",
             rainfall.mean(), rainfall.min(), rainfall.max())

    ts = build_rainfall_timeseries(seed=seed)
    drainage = build_drainage_network(grid, seed=seed)
    floods = build_flood_points(grid, seed=seed)
    log.info("Vector layers: %d drainage lines, %d flood points",
             len(drainage), len(floods))

    return {
        "grid": grid,
        "rainfall_field": rainfall,
        "rainfall_timeseries": ts,
        "drainage": drainage,
        "flood_points": floods,
    }


if __name__ == "__main__":
    data = ingest_all()
    data["rainfall_timeseries"].to_csv(PROCESSED_DIR / "rainfall_timeseries.csv", index=False)
    data["flood_points"].to_csv(PROCESSED_DIR / "flood_points.csv", index=False)
    np.save(PROCESSED_DIR / "dem.npy", data["grid"].elevation)
    np.save(PROCESSED_DIR / "rainfall_field.npy", data["rainfall_field"])
    log.info("Saved raw ingestion artefacts to %s", PROCESSED_DIR)
