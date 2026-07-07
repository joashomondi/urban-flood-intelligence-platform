"""
utils.py
========
Shared configuration, path management, logging and small helpers for the
Urban Flood Intelligence Platform (UFIP).

This module is intentionally dependency-light so that *every* other module and
notebook can import it without pulling in heavy geospatial libraries.

Design goals
------------
* Single source of truth for the study-area geometry, grid resolution, CRS,
  file locations and model / scoring hyper-parameters.
* Deterministic behaviour: a global RANDOM_SEED drives every stochastic step
  so the whole pipeline is reproducible on Zerve.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random Generator seeded for reproducibility."""
    return np.random.default_rng(RANDOM_SEED if seed is None else seed)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
def _find_project_root(start: Path) -> Path:
    """Walk upward until we find the repository root (contains requirements.txt)."""
    for parent in [start, *start.parents]:
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    # Fallback: two levels up from this file (src/ -> root)
    return start.parents[1] if len(start.parents) >= 2 else start


PROJECT_ROOT: Path = _find_project_root(Path(__file__).resolve())

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MAPS_DIR = OUTPUTS_DIR / "maps"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"

ALL_DIRS = [
    RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR,
    MAPS_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR,
]


def ensure_dirs() -> None:
    """Create every project directory if it does not already exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Study-area configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudyArea:
    """Geographic definition of the analysis window.

    Defaults describe a bounding box over Nairobi, Kenya - a flood-prone
    East-African metropolis with well documented stormwater challenges.
    """
    name: str = "Nairobi, Kenya"
    # Bounding box in EPSG:4326 (lon/lat degrees): (min_lon, min_lat, max_lon, max_lat)
    bounds: Tuple[float, float, float, float] = (36.65, -1.44, 36.95, -1.16)
    crs_geographic: str = "EPSG:4326"
    crs_projected: str = "EPSG:32737"  # UTM zone 37S - metric units for Nairobi
    grid_size: int = 160               # cells per side => 160 x 160 raster grid
    # Approximate real-world resolution (metres) implied by grid + bbox.
    # ~33 km / 160 ≈ 205 m per cell.

    @property
    def min_lon(self) -> float: return self.bounds[0]
    @property
    def min_lat(self) -> float: return self.bounds[1]
    @property
    def max_lon(self) -> float: return self.bounds[2]
    @property
    def max_lat(self) -> float: return self.bounds[3]

    @property
    def center(self) -> Tuple[float, float]:
        """(lat, lon) centroid - handy for Folium map initialisation."""
        return ((self.min_lat + self.max_lat) / 2.0,
                (self.min_lon + self.max_lon) / 2.0)

    @property
    def width_deg(self) -> float: return self.max_lon - self.min_lon
    @property
    def height_deg(self) -> float: return self.max_lat - self.min_lat

    @property
    def cell_size_m(self) -> float:
        """Rough metric cell size using 111 km per degree of latitude."""
        return (self.height_deg * 111_000.0) / self.grid_size


# Named operational zones inside the study area. Coordinates are (lat, lon)
# centroids used for the zone selector and per-zone scoring.
NAIROBI_ZONES: Dict[str, Tuple[float, float]] = {
    "CBD / Central": (-1.286, 36.817),
    "Westlands": (-1.267, 36.803),
    "Kibera": (-1.312, 36.782),
    "Eastleigh": (-1.272, 36.850),
    "Embakasi": (-1.322, 36.894),
    "Karen": (-1.319, 36.712),
    "Kasarani": (-1.220, 36.898),
    "Mathare": (-1.259, 36.858),
    "Langata": (-1.363, 36.741),
    "Ruaraka": (-1.238, 36.872),
    "Dagoretti": (-1.293, 36.735),
    "Roysambu": (-1.213, 36.878),
}


# ---------------------------------------------------------------------------
# Scoring / model configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringConfig:
    """Weights and thresholds for the Flood Risk Score (FRS) engine.

    Weights are expressed on the 0-1 susceptibility indicators produced by the
    feature-engineering stage and must sum to 1.0.
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        "low_elevation_susceptibility": 0.22,
        "rainfall_intensity": 0.26,
        "drainage_stress": 0.18,
        "slope_vulnerability": 0.14,
        "runoff_potential": 0.12,
        "terrain_instability": 0.08,
    })
    # FRS category thresholds (0-100 scale).
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "Low": 25.0,
        "Moderate": 50.0,
        "High": 75.0,
        "Severe": 100.0,
    })
    # A zone is flagged "affected" once its FRS crosses this value.
    affected_threshold: float = 55.0

    def category(self, score: float) -> str:
        """Map a 0-100 FRS to an operational vulnerability class."""
        if score < self.thresholds["Low"]:
            return "Low"
        if score < self.thresholds["Moderate"]:
            return "Moderate"
        if score < self.thresholds["High"]:
            return "High"
        return "Severe"


CATEGORY_COLORS: Dict[str, str] = {
    "Low": "#2ecc71",       # green
    "Moderate": "#f1c40f",  # amber
    "High": "#e67e22",      # orange
    "Severe": "#e74c3c",    # red
}


# Baseline rainfall (mm/24h) used to normalise rainfall intensity to 0-1.
# 80 mm/24h is a heavy-storm reference for Nairobi's wet seasons.
RAINFALL_REFERENCE_MM: float = 80.0
RAINFALL_MAX_MM: float = 200.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "ufip") -> logging.Logger:
    """Return a configured, non-duplicating logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def normalize(arr: np.ndarray, invert: bool = False,
              clip_pct: Tuple[float, float] | None = (1.0, 99.0)) -> np.ndarray:
    """Min-max normalise an array to [0, 1].

    Parameters
    ----------
    arr : np.ndarray
        Input array (NaNs are ignored during range computation).
    invert : bool
        If True, return ``1 - normalized`` (useful for "lower is worse"
        indicators such as elevation or slope).
    clip_pct : (low, high) | None
        Percentile clipping to make normalisation robust to outliers.
    """
    a = np.asarray(arr, dtype="float64")
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)

    if clip_pct is not None:
        lo = np.nanpercentile(a[finite], clip_pct[0])
        hi = np.nanpercentile(a[finite], clip_pct[1])
    else:
        lo, hi = np.nanmin(a[finite]), np.nanmax(a[finite])

    if hi - lo < 1e-12:
        out = np.zeros_like(a)
    else:
        out = (a - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0)

    out = np.where(finite, out, 0.0)
    return 1.0 - out if invert else out


def save_json(obj: Dict[str, Any], path: Path | str) -> None:
    """Persist a dict as pretty JSON (numpy-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# Instantiate singletons used across the codebase.
STUDY_AREA = StudyArea()
SCORING = ScoringConfig()


def config_summary() -> Dict[str, Any]:
    """Return a serialisable snapshot of the active configuration."""
    return {
        "random_seed": RANDOM_SEED,
        "study_area": {**asdict(STUDY_AREA),
                       "center": STUDY_AREA.center,
                       "cell_size_m": round(STUDY_AREA.cell_size_m, 1)},
        "scoring": asdict(SCORING),
        "rainfall_reference_mm": RAINFALL_REFERENCE_MM,
        "rainfall_max_mm": RAINFALL_MAX_MM,
        "zones": NAIROBI_ZONES,
    }


if __name__ == "__main__":
    ensure_dirs()
    log = get_logger()
    log.info("Project root: %s", PROJECT_ROOT)
    log.info("Study area  : %s  bounds=%s", STUDY_AREA.name, STUDY_AREA.bounds)
    log.info("Grid        : %d x %d  (~%.0f m/cell)",
             STUDY_AREA.grid_size, STUDY_AREA.grid_size, STUDY_AREA.cell_size_m)
    save_json(config_summary(), PROCESSED_DIR / "config_summary.json")
    log.info("Wrote config summary -> %s", PROCESSED_DIR / "config_summary.json")
