"""
feature_engineering.py
=====================
Operational Flood Indicator engine.

Transforms raw terrain + hydrology + rainfall layers into a set of normalised
(0-1) flood-susceptibility indicators, then assembles a per-cell tabular
dataset ready for modelling. Where ground-truth flood labels are unavailable,
it generates *physically-motivated* synthetic labels from the indicators so the
supervised model has a defensible target.

Indicators produced
--------------------
    low_elevation_susceptibility : lower ground floods first
    rainfall_intensity           : normalised rainfall load
    drainage_stress              : high wetness + high drainage density
    slope_vulnerability          : flat ground drains poorly
    runoff_potential             : upslope contributing area (flow accumulation)
    terrain_instability          : ruggedness / curvature driven instability
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from . import utils
from .utils import STUDY_AREA, get_logger, get_rng, normalize

log = get_logger("ufip.features")

INDICATOR_NAMES = [
    "low_elevation_susceptibility",
    "rainfall_intensity",
    "drainage_stress",
    "slope_vulnerability",
    "runoff_potential",
    "terrain_instability",
]


def build_indicators(terrain: Dict[str, np.ndarray],
                     hydro: Dict[str, np.ndarray],
                     rainfall_field: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the six normalised flood indicators from raw layers."""
    log.info("=== STEP 3: FEATURE ENGINEERING ===")

    elevation = terrain["elevation"]
    slope_deg = terrain["slope"]
    roughness = terrain["roughness"]
    curvature = terrain["curvature"]
    acc = hydro["flow_accumulation"]
    twi = hydro["twi"]
    dens = hydro["drainage_density"]

    # Lower elevation => higher susceptibility (invert).
    low_elev = normalize(elevation, invert=True)

    # Rainfall load, normalised over the field.
    rain = normalize(rainfall_field)

    # Drainage stress: catchments that concentrate water (TWI) AND dense
    # channel networks that can be overwhelmed.
    drainage_stress = normalize(0.6 * normalize(twi) + 0.4 * normalize(dens))

    # Flatter ground drains poorly => higher slope vulnerability.
    slope_vuln = normalize(slope_deg, invert=True)

    # Runoff potential from upslope contributing area (log-scaled).
    runoff = normalize(np.log1p(acc))

    # Terrain instability: rugged + strongly concave micro-topography.
    concavity = normalize(-curvature)   # concave (negative curvature) -> high
    instability = normalize(0.5 * normalize(roughness) + 0.5 * concavity)

    indicators = {
        "low_elevation_susceptibility": low_elev.astype("float32"),
        "rainfall_intensity": rain.astype("float32"),
        "drainage_stress": drainage_stress.astype("float32"),
        "slope_vulnerability": slope_vuln.astype("float32"),
        "runoff_potential": runoff.astype("float32"),
        "terrain_instability": instability.astype("float32"),
    }
    for k, v in indicators.items():
        log.info("  indicator %-30s mean=%.3f", k, float(v.mean()))
    return indicators


def generate_synthetic_labels(indicators: Dict[str, np.ndarray],
                              seed: int | None = None) -> np.ndarray:
    """Create binary flood-occurrence labels from a weighted indicator blend.

    A logistic response with additive noise turns the continuous susceptibility
    surface into realistic, non-separable labels (~20-30% positive rate) so the
    classifier has a genuine learning task rather than a trivial threshold.
    """
    rng = get_rng(seed)
    w = utils.SCORING.weights
    latent = sum(w[name] * indicators[name] for name in INDICATOR_NAMES)
    latent = normalize(latent)
    # Logistic transform centred so ~25% of cells are positive.
    z = 7.5 * (latent - 0.62) + rng.normal(0, 0.6, size=latent.shape)
    prob = 1.0 / (1.0 + np.exp(-z))
    labels = (rng.random(prob.shape) < prob).astype("int8")
    log.info("Synthetic flood labels: positive rate = %.1f%%", 100 * labels.mean())
    return labels


def build_feature_table(indicators: Dict[str, np.ndarray],
                        extra_layers: Dict[str, np.ndarray],
                        labels: np.ndarray,
                        lons: np.ndarray, lats: np.ndarray) -> pd.DataFrame:
    """Flatten the raster stack into a tidy per-cell DataFrame."""
    h, w = labels.shape
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    data = {
        "row": np.repeat(np.arange(h), w),
        "col": np.tile(np.arange(w), h),
        "lon": lon_grid.ravel(),
        "lat": lat_grid.ravel(),
    }
    for name in INDICATOR_NAMES:
        data[name] = indicators[name].ravel()
    for name, layer in extra_layers.items():
        data[name] = layer.ravel()
    data["flood_label"] = labels.ravel()
    df = pd.DataFrame(data)
    log.info("Feature table: %d rows x %d cols", df.shape[0], df.shape[1])
    return df


def engineer_features(terrain, hydro, rainfall_field, seed: int | None = None) -> dict:
    """End-to-end feature engineering: indicators + labels + tabular dataset."""
    indicators = build_indicators(terrain, hydro, rainfall_field)
    labels = generate_synthetic_labels(indicators, seed=seed)
    extra = {
        "elevation_m": terrain["elevation"],
        "slope_deg": terrain["slope"],
        "twi": hydro["twi"],
        "flow_accumulation": hydro["flow_accumulation"],
        "drainage_density": hydro["drainage_density"],
    }
    # need lon/lat vectors - reconstruct from study area
    n = terrain["elevation"].shape[0]
    lons = np.linspace(STUDY_AREA.min_lon, STUDY_AREA.max_lon, n)
    lats = np.linspace(STUDY_AREA.max_lat, STUDY_AREA.min_lat, n)
    table = build_feature_table(indicators, extra, labels, lons, lats)
    return {"indicators": indicators, "labels": labels, "table": table}
