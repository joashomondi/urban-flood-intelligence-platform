"""
terrain_analysis.py
===================
Terrain Processing Engine.

Derives the primary topographic products used downstream by the hydrology and
feature-engineering stages:

    * slope (degrees)
    * aspect (degrees, compass)
    * terrain roughness (TRI - Terrain Ruggedness Index)
    * profile curvature
    * hillshade (for cartography)

All functions operate on plain NumPy 2-D arrays so they are portable and fast,
and require no GIS runtime.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .utils import STUDY_AREA, get_logger

log = get_logger("ufip.terrain")


def _gradients(dem: np.ndarray, cell_size_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dz/dx, dz/dy) in metres/metre using central differences."""
    dzdy, dzdx = np.gradient(dem.astype("float64"), cell_size_m, cell_size_m)
    return dzdx, dzdy


def slope(dem: np.ndarray, cell_size_m: float | None = None) -> np.ndarray:
    """Slope in **degrees** (0 = flat, 90 = vertical)."""
    cs = cell_size_m or STUDY_AREA.cell_size_m
    dzdx, dzdy = _gradients(dem, cs)
    rise_run = np.sqrt(dzdx ** 2 + dzdy ** 2)
    return np.degrees(np.arctan(rise_run)).astype("float32")


def aspect(dem: np.ndarray, cell_size_m: float | None = None) -> np.ndarray:
    """Aspect in **compass degrees** (0=N, 90=E, 180=S, 270=W); flat = -1."""
    cs = cell_size_m or STUDY_AREA.cell_size_m
    dzdx, dzdy = _gradients(dem, cs)
    asp = np.degrees(np.arctan2(dzdy, -dzdx))
    asp = (450.0 - asp) % 360.0
    flat = (np.abs(dzdx) < 1e-9) & (np.abs(dzdy) < 1e-9)
    asp[flat] = -1.0
    return asp.astype("float32")


def terrain_roughness(dem: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index: mean absolute elevation diff to 8 neighbours."""
    d = dem.astype("float64")
    acc = np.zeros_like(d)
    cnt = np.zeros_like(d)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(d, dr, axis=0), dc, axis=1)
            acc += np.abs(d - shifted)
            cnt += 1
    return (acc / cnt).astype("float32")


def profile_curvature(dem: np.ndarray, cell_size_m: float | None = None) -> np.ndarray:
    """Second-derivative curvature (concave < 0 tends to accumulate water)."""
    cs = cell_size_m or STUDY_AREA.cell_size_m
    d = dem.astype("float64")
    dzdx, dzdy = _gradients(d, cs)
    dxx = np.gradient(dzdx, cs, axis=1)
    dyy = np.gradient(dzdy, cs, axis=0)
    return (dxx + dyy).astype("float32")


def hillshade(dem: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0,
              cell_size_m: float | None = None) -> np.ndarray:
    """Analytical hillshade [0, 255] for cartographic relief shading."""
    cs = cell_size_m or STUDY_AREA.cell_size_m
    dzdx, dzdy = _gradients(dem, cs)
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    aspect_rad = np.arctan2(dzdy, -dzdx)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    shaded = (np.sin(alt) * np.cos(slope_rad) +
              np.cos(alt) * np.sin(slope_rad) * np.cos(az - aspect_rad))
    return (np.clip(shaded, 0, 1) * 255).astype("float32")


# ---------------------------------------------------------------------------
# Steepest-descent path (used to synthesise drainage lines)
# ---------------------------------------------------------------------------
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


def steepest_descent_path(dem: np.ndarray, start: Tuple[int, int],
                          max_len: int = 300) -> List[Tuple[int, int]]:
    """Trace a flow path following steepest descent from ``start``."""
    h, w = dem.shape
    r, c = start
    path = [(r, c)]
    visited = {(r, c)}
    for _ in range(max_len):
        best = None
        best_drop = 0.0
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                drop = dem[r, c] - dem[nr, nc]
                if drop > best_drop:
                    best_drop = drop
                    best = (nr, nc)
        if best is None:
            break
        r, c = best
        visited.add(best)
        path.append(best)
    return path


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
def analyze_terrain(dem: np.ndarray, cell_size_m: float | None = None) -> dict:
    """Compute every terrain derivative and return them in a dict."""
    log.info("=== STEP 2a: TERRAIN ANALYSIS ===")
    cs = cell_size_m or STUDY_AREA.cell_size_m
    out = {
        "elevation": dem.astype("float32"),
        "slope": slope(dem, cs),
        "aspect": aspect(dem, cs),
        "roughness": terrain_roughness(dem),
        "curvature": profile_curvature(dem, cs),
        "hillshade": hillshade(dem, cell_size_m=cs),
    }
    log.info("Slope: mean=%.2f deg max=%.2f deg | Roughness mean=%.2f m",
             out["slope"].mean(), out["slope"].max(), out["roughness"].mean())
    return out
