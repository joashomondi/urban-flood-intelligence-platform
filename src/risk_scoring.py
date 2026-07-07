"""
risk_scoring.py
===============
The signature operational engine: **Flood Risk Score (FRS)**.

The FRS is a 0-100 index summarising how likely and how severe flooding is for
each grid cell / zone, given current rainfall and static terrain-hydrology
conditions. It is the product's headline metric and is designed to:

    * respond **dynamically** to rainfall intensity (dashboard slider),
    * blend a transparent weighted-indicator model with the trained ML model,
    * roll up to per-zone scores, vulnerability categories and affected-zone
      counts for operational decision-making.

Signature output
----------------
    Flood Risk Score : 82/100
    Drainage Stress  : High
    Terrain Vulnerability : Severe
    Affected Zones   : 14
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import utils
from .feature_engineering import INDICATOR_NAMES
from .utils import (NAIROBI_ZONES, SCORING, STUDY_AREA, get_logger, normalize,
                    RAINFALL_REFERENCE_MM, RAINFALL_MAX_MM)

log = get_logger("ufip.scoring")


def _stress_label(value_0_1: float) -> str:
    """Map a 0-1 stress intensity to a qualitative operational label."""
    if value_0_1 < 0.25:
        return "Low"
    if value_0_1 < 0.5:
        return "Moderate"
    if value_0_1 < 0.75:
        return "High"
    return "Severe"


@dataclass
class ScoringResult:
    """Container for a single scoring run at a given rainfall level."""
    rainfall_mm: float
    frs_surface: np.ndarray          # (H, W) 0-100
    mean_frs: float
    category: str
    drainage_stress: str
    terrain_vulnerability: str
    affected_zones: int
    zone_table: pd.DataFrame
    hotspots: pd.DataFrame

    def headline(self) -> Dict[str, object]:
        """Return the compact KPI dict shown at the top of the dashboard."""
        return {
            "flood_risk_score": round(self.mean_frs, 1),
            "category": self.category,
            "drainage_stress": self.drainage_stress,
            "terrain_vulnerability": self.terrain_vulnerability,
            "affected_zones": self.affected_zones,
            "rainfall_mm": round(self.rainfall_mm, 1),
        }


class FloodRiskScorer:
    """Reusable, stateful scoring engine.

    Parameters
    ----------
    indicators : dict of str -> 2-D array
        The six normalised indicators from feature engineering. Computed at the
        *reference* rainfall level; the engine rescales rainfall dynamically.
    lons, lats : 1-D arrays
        Cell-centre coordinates (for hotspot/zone geolocation).
    model_surface : 2-D array, optional
        ML flood-probability surface (0-1). If provided, the FRS blends the
        transparent weighted score with the model for extra signal.
    model_blend : float
        Weight on the ML surface in [0, 1]. 0 = pure weighted indicators.
    """

    def __init__(self, indicators: Dict[str, np.ndarray],
                 lons: np.ndarray, lats: np.ndarray,
                 model_surface: Optional[np.ndarray] = None,
                 model_blend: float = 0.35,
                 reference_rainfall_mm: float = RAINFALL_REFERENCE_MM):
        self.indicators = indicators
        self.lons = lons
        self.lats = lats
        self.model_surface = model_surface
        self.model_blend = float(np.clip(model_blend, 0.0, 1.0))
        self.reference_rainfall_mm = reference_rainfall_mm
        self.weights = SCORING.weights
        self.shape = indicators[INDICATOR_NAMES[0]].shape

    # -- core -------------------------------------------------------------
    def _rainfall_scaled_indicator(self, rainfall_mm: float) -> np.ndarray:
        """Rescale the rainfall-intensity indicator for a new rainfall total.

        The base indicator captures the *spatial pattern* of rainfall; here we
        modulate its overall magnitude by the ratio of requested rainfall to
        the reference, saturating toward 1.0 at extreme totals.
        """
        base = self.indicators["rainfall_intensity"]
        ratio = rainfall_mm / max(self.reference_rainfall_mm, 1e-6)
        # Saturating gain so 2x rainfall != 2x indicator (physically bounded).
        gain = ratio / (1.0 + 0.6 * (ratio - 1.0)) if ratio > 1 else ratio
        scaled = np.clip(base * gain, 0.0, 1.0)
        return scaled

    def weighted_surface(self, rainfall_mm: float) -> np.ndarray:
        """0-1 weighted-indicator susceptibility for a rainfall total."""
        rain_ind = self._rainfall_scaled_indicator(rainfall_mm)
        surface = np.zeros(self.shape, dtype="float64")
        for name in INDICATOR_NAMES:
            layer = rain_ind if name == "rainfall_intensity" else self.indicators[name]
            surface += self.weights[name] * layer
        return surface

    def compute(self, rainfall_mm: float = RAINFALL_REFERENCE_MM,
                threshold: Optional[float] = None,
                n_hotspots: int = 15) -> ScoringResult:
        """Full scoring run -> :class:`ScoringResult`."""
        weighted = self.weighted_surface(rainfall_mm)
        if self.model_surface is not None and self.model_blend > 0:
            frs01 = (1 - self.model_blend) * weighted + self.model_blend * self.model_surface
        else:
            frs01 = weighted
        frs = np.clip(frs01 * 100.0, 0, 100).astype("float32")

        mean_frs = float(frs.mean())
        category = SCORING.category(mean_frs)

        drainage_stress = _stress_label(float(self.indicators["drainage_stress"].mean()))
        terrain_vuln = _stress_label(float(
            0.5 * self.indicators["slope_vulnerability"].mean()
            + 0.5 * self.indicators["terrain_instability"].mean()))

        zone_table = self.score_zones(frs, threshold=threshold)
        thr = threshold if threshold is not None else SCORING.affected_threshold
        affected = int((zone_table["frs"] >= thr).sum())
        hotspots = self.extract_hotspots(frs, n=n_hotspots)

        return ScoringResult(
            rainfall_mm=rainfall_mm, frs_surface=frs, mean_frs=mean_frs,
            category=category, drainage_stress=drainage_stress,
            terrain_vulnerability=terrain_vuln, affected_zones=affected,
            zone_table=zone_table, hotspots=hotspots,
        )

    # -- roll-ups ---------------------------------------------------------
    def _cell_of(self, lat: float, lon: float) -> tuple:
        row = int(np.clip(np.argmin(np.abs(self.lats - lat)), 0, self.shape[0] - 1))
        col = int(np.clip(np.argmin(np.abs(self.lons - lon)), 0, self.shape[1] - 1))
        return row, col

    def score_zones(self, frs: np.ndarray, radius: int = 6,
                    threshold: Optional[float] = None) -> pd.DataFrame:
        """Aggregate the FRS surface into per-zone scores (windowed mean)."""
        rows = []
        h, w = frs.shape
        for name, (lat, lon) in NAIROBI_ZONES.items():
            r, c = self._cell_of(lat, lon)
            r0, r1 = max(0, r - radius), min(h, r + radius + 1)
            c0, c1 = max(0, c - radius), min(w, c + radius + 1)
            window = frs[r0:r1, c0:c1]
            score = float(window.mean())
            rows.append({
                "zone": name, "lat": lat, "lon": lon,
                "frs": round(score, 1),
                "category": SCORING.category(score),
                "peak_frs": round(float(window.max()), 1),
            })
        df = pd.DataFrame(rows).sort_values("frs", ascending=False).reset_index(drop=True)
        return df

    def extract_hotspots(self, frs: np.ndarray, n: int = 15) -> pd.DataFrame:
        """Return the top-N highest-risk cells as a point table for mapping."""
        flat = frs.ravel()
        idx = np.argsort(flat)[::-1][:n]
        rows, cols = np.unravel_index(idx, frs.shape)
        return pd.DataFrame({
            "rank": np.arange(1, len(idx) + 1),
            "lat": np.round(self.lats[rows], 6),
            "lon": np.round(self.lons[cols], 6),
            "frs": np.round(flat[idx], 1),
            "category": [SCORING.category(v) for v in flat[idx]],
        })

    # -- scenario sweep ---------------------------------------------------
    def rainfall_response_curve(self,
                                mm_values: Optional[List[float]] = None) -> pd.DataFrame:
        """Sweep rainfall totals and record mean FRS + affected zones.

        Powers the dashboard's "how does risk escalate with rainfall?" chart.
        """
        if mm_values is None:
            mm_values = list(np.linspace(10, RAINFALL_MAX_MM, 20))
        records = []
        for mm in mm_values:
            res = self.compute(rainfall_mm=mm)
            records.append({
                "rainfall_mm": round(mm, 1),
                "mean_frs": round(res.mean_frs, 2),
                "affected_zones": res.affected_zones,
                "category": res.category,
            })
        return pd.DataFrame(records)


def build_scorer_from_artifacts(indicators: Dict[str, np.ndarray],
                                model_surface: Optional[np.ndarray] = None,
                                model_blend: float = 0.35) -> FloodRiskScorer:
    """Convenience constructor that infers lon/lat vectors from the study area."""
    n = indicators[INDICATOR_NAMES[0]].shape[0]
    lons = np.linspace(STUDY_AREA.min_lon, STUDY_AREA.max_lon, n)
    lats = np.linspace(STUDY_AREA.max_lat, STUDY_AREA.min_lat, n)
    return FloodRiskScorer(indicators, lons, lats,
                           model_surface=model_surface, model_blend=model_blend)


if __name__ == "__main__":
    log.info("risk_scoring is a library module; run scripts/run_pipeline.py instead.")
