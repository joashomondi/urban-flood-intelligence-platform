"""
pipeline.py
===========
Single reusable entry point that assembles the full analytical state in memory.

This is the glue used by:
  * ``scripts/run_pipeline.py``    (batch run + artefact export)
  * ``zerve/blocks/*.py``          (fallback when a block is run out of order)
  * ``zerve/app/main.py`` & the Streamlit app (cold-start fallback)

Everything here is pure in-memory computation (no file writes), so it works on
Zerve's serverless blocks where state is passed between blocks rather than via
the local filesystem.
"""
from __future__ import annotations

from typing import Optional

from . import (data_loader, feature_engineering, hydrology, modeling,
               risk_scoring, terrain_analysis, utils)
from .utils import RAINFALL_REFERENCE_MM, get_logger

log = get_logger("ufip.pipeline_core")


def build_state(rainfall_mm: float = RAINFALL_REFERENCE_MM,
                model_blend: float = 0.35,
                train: bool = True,
                seed: Optional[int] = None) -> dict:
    """Run the whole analytical chain and return every artefact in one dict.

    Parameters
    ----------
    rainfall_mm : float
        Rainfall scenario (mm/24h) used for ingestion + initial scoring.
    model_blend : float
        Weight on the ML probability surface inside the FRS (0..1).
    train : bool
        If True, train the classifier and blend its surface into the FRS.
        Set False for a fast, model-free state (e.g. upstream-only fallback).
    """
    data = data_loader.ingest_all(base_rainfall_mm=rainfall_mm, seed=seed)
    grid = data["grid"]

    terrain = terrain_analysis.analyze_terrain(grid.elevation)
    hydro = hydrology.analyze_hydrology(grid.elevation, terrain["slope"])

    feats = feature_engineering.engineer_features(
        terrain, hydro, data["rainfall_field"], seed=seed)
    indicators = feats["indicators"]

    model_out = None
    model_surface = None
    if train:
        model_out = modeling.train_and_evaluate(feats["table"])
        model_surface = modeling.predict_surface(
            model_out["best_model"], indicators, model_out["feature_columns"])

    scorer = risk_scoring.build_scorer_from_artifacts(
        indicators, model_surface=model_surface, model_blend=model_blend)
    result = scorer.compute(rainfall_mm=rainfall_mm)

    return {
        # raw / ingestion
        "grid": grid,
        "dem": grid.elevation,
        "rainfall_field": data["rainfall_field"],
        "rainfall_ts": data["rainfall_timeseries"],
        "drainage": data["drainage"],
        "flood_points": data["flood_points"],
        # terrain + hydrology
        "terrain": terrain,
        "hydro": hydro,
        "hillshade": terrain["hillshade"],
        # features
        "indicators": indicators,
        "feature_table": feats["table"],
        "labels": feats["labels"],
        # model
        "model_out": model_out,
        "model_surface": model_surface,
        # scoring
        "scorer": scorer,
        "result": result,
        "frs_surface": result.frs_surface,
        "zone_table": result.zone_table,
        "hotspots": result.hotspots,
        "headline": result.headline(),
    }
