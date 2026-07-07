"""
hydrology.py
============
Hydrological Feature Extraction engine.

Implements the classic D8 surface-hydrology workflow on the DEM:

    * pit filling (priority-flood style, simplified)
    * D8 flow direction
    * flow accumulation (upslope contributing area)
    * Topographic Wetness Index (TWI)
    * drainage density (kernel density of the derived stream network)

These indicators quantify *where water goes* and *where it collects* - the
hydrological backbone of the flood-susceptibility model.
"""
from __future__ import annotations

import numpy as np

from .utils import STUDY_AREA, get_logger

log = get_logger("ufip.hydro")

# D8 encoding: index -> (dr, dc). ESRI-style power-of-two codes on the side.
_D8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_D8_CODE = [64, 128, 1, 2, 4, 8, 16, 32]


def fill_depressions(dem: np.ndarray, max_iter: int = 60, eps: float = 1e-3) -> np.ndarray:
    """Simplified iterative depression filling (Planchon-Darboux flavour).

    Raises interior pits to their lowest spill point so flow routing does not
    dead-end. Boundary cells act as outlets. Converges quickly for smooth DEMs.
    """
    filled = dem.astype("float64").copy()
    h, w = filled.shape
    huge = filled.max() + 1000.0
    work = np.full_like(filled, huge)
    # Boundary keeps original elevation (outlets).
    work[0, :] = filled[0, :]
    work[-1, :] = filled[-1, :]
    work[:, 0] = filled[:, 0]
    work[:, -1] = filled[:, -1]

    for _ in range(max_iter):
        changed = 0
        prev = work.copy()
        # For interior cells, water level = max(dem, min neighbour + eps).
        neigh_min = np.full_like(work, huge)
        for dr, dc in _D8:
            shifted = np.roll(np.roll(prev, dr, axis=0), dc, axis=1)
            neigh_min = np.minimum(neigh_min, shifted)
        candidate = np.maximum(filled, neigh_min + eps)
        interior = np.ones_like(work, dtype=bool)
        interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
        new_work = np.where(interior, np.minimum(prev, candidate), work)
        changed = np.abs(new_work - prev).sum()
        work = new_work
        if changed < eps:
            break
    return work.astype("float32")


def flow_direction(dem: np.ndarray) -> np.ndarray:
    """D8 flow direction as neighbour index 0..7 (-1 if a sink/no lower cell)."""
    d = dem.astype("float64")
    h, w = d.shape
    best_idx = np.full((h, w), -1, dtype="int8")
    best_slope = np.zeros((h, w))
    dist = np.array([1, np.sqrt(2)] * 4)  # cardinal vs diagonal spacing
    for k, (dr, dc) in enumerate(_D8):
        shifted = np.roll(np.roll(d, -dr, axis=0), -dc, axis=1)
        drop = (d - shifted) / dist[k]
        update = drop > best_slope
        best_slope = np.where(update, drop, best_slope)
        best_idx = np.where(update, k, best_idx)
    return best_idx


def flow_accumulation(fdir: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Upslope contributing-cell count via topological (elevation) ordering.

    Processing cells from highest to lowest guarantees every donor is handled
    before its receiver - an O(N log N) accumulation without recursion.
    """
    h, w = fdir.shape
    acc = np.ones((h, w), dtype="float64")     # each cell contributes itself
    order = np.argsort(dem.ravel())[::-1]       # high -> low
    fd = fdir.ravel()
    accf = acc.ravel()
    for flat in order:
        k = fd[flat]
        if k < 0:
            continue
        r, c = divmod(flat, w)
        dr, dc = _D8[k]
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            accf[nr * w + nc] += accf[flat]
    return acc.astype("float32")


def topographic_wetness_index(acc: np.ndarray, slope_deg: np.ndarray,
                              cell_size_m: float | None = None) -> np.ndarray:
    """TWI = ln( a / tan(beta) ) - high where large catchment meets flat ground."""
    cs = cell_size_m or STUDY_AREA.cell_size_m
    specific_area = (acc * cs)  # contributing area per unit contour width
    slope_rad = np.radians(np.maximum(slope_deg, 0.1))  # avoid div-by-zero
    twi = np.log((specific_area + 1.0) / np.tan(slope_rad))
    return twi.astype("float32")


def drainage_density(acc: np.ndarray, threshold_pct: float = 92.0,
                     radius: int = 4) -> np.ndarray:
    """Local density of channel cells (proxy for drainage density km/km^2).

    Cells whose flow accumulation exceeds ``threshold_pct`` are treated as the
    stream network; a moving-window mean of that mask gives a smooth density.
    """
    thr = np.percentile(acc, threshold_pct)
    channels = (acc >= thr).astype("float64")
    # Box-filter via cumulative sums (fast moving average).
    k = 2 * radius + 1
    padded = np.pad(channels, radius, mode="edge")
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    h, w = channels.shape
    dens = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            r2, c2 = r + k, c + k
            total = (csum[r2, c2] - csum[r, c2] - csum[r2, c] + csum[r, c])
            dens[r, c] = total / (k * k)
    return dens.astype("float32")


def analyze_hydrology(dem: np.ndarray, slope_deg: np.ndarray,
                      cell_size_m: float | None = None) -> dict:
    """Run the full hydrological extraction and return all layers."""
    log.info("=== STEP 2b: HYDROLOGICAL ANALYSIS ===")
    cs = cell_size_m or STUDY_AREA.cell_size_m
    filled = fill_depressions(dem)
    fdir = flow_direction(filled)
    acc = flow_accumulation(fdir, filled)
    twi = topographic_wetness_index(acc, slope_deg, cs)
    dens = drainage_density(acc)
    log.info("Flow accumulation: max=%.0f cells | TWI mean=%.2f | drainage density mean=%.3f",
             acc.max(), twi.mean(), dens.mean())
    return {
        "filled_dem": filled,
        "flow_direction": fdir,
        "flow_accumulation": acc,
        "twi": twi,
        "drainage_density": dens,
    }
