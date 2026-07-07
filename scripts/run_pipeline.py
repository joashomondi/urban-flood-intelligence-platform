"""
run_pipeline.py
===============
End-to-end orchestration of the Urban Flood Intelligence Platform.

Runs every engine in sequence and materialises the artefacts that the
Streamlit app and notebooks consume:

    data/processed/  - DEM, indicators, feature table, scoring cache
    outputs/models/  - persisted best model
    outputs/figures/ - static maps + plotly HTML
    outputs/maps/    - interactive Folium maps
    outputs/reports/ - metrics + operational summary

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --rainfall 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow "python scripts/run_pipeline.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (data_loader, feature_engineering, hydrology, modeling,
                 risk_scoring, terrain_analysis, utils, visualization)
from src.utils import (FIGURES_DIR, MAPS_DIR, PROCESSED_DIR, REPORTS_DIR,
                       RAINFALL_REFERENCE_MM, get_logger, save_json)

log = get_logger("ufip.pipeline")


def main(rainfall_mm: float = RAINFALL_REFERENCE_MM, quick: bool = False) -> dict:
    t0 = time.time()
    utils.ensure_dirs()
    log.info("################  URBAN FLOOD INTELLIGENCE PLATFORM  ################")
    save_json(utils.config_summary(), PROCESSED_DIR / "config_summary.json")

    # 1) Ingestion -------------------------------------------------------
    data = data_loader.ingest_all(base_rainfall_mm=rainfall_mm)
    grid = data["grid"]
    data["rainfall_timeseries"].to_csv(PROCESSED_DIR / "rainfall_timeseries.csv", index=False)
    data["flood_points"].to_csv(PROCESSED_DIR / "flood_points.csv", index=False)
    np.save(PROCESSED_DIR / "dem.npy", grid.elevation)
    np.save(PROCESSED_DIR / "rainfall_field.npy", data["rainfall_field"])

    # 2) Terrain + hydrology --------------------------------------------
    terrain = terrain_analysis.analyze_terrain(grid.elevation)
    hydro = hydrology.analyze_hydrology(grid.elevation, terrain["slope"])
    for name, arr in {**terrain, **hydro}.items():
        if isinstance(arr, np.ndarray) and arr.dtype != object:
            np.save(PROCESSED_DIR / f"layer_{name}.npy", arr.astype("float32"))

    # 3) Feature engineering --------------------------------------------
    feats = feature_engineering.engineer_features(
        terrain, hydro, data["rainfall_field"])
    indicators = feats["indicators"]
    feats["table"].to_parquet(PROCESSED_DIR / "feature_table.parquet") \
        if _parquet_ok() else feats["table"].to_csv(
            PROCESSED_DIR / "feature_table.csv", index=False)
    for name, arr in indicators.items():
        np.save(PROCESSED_DIR / f"indicator_{name}.npy", arr)

    # 4) Modelling -------------------------------------------------------
    model_out = modeling.train_and_evaluate(feats["table"])
    modeling.save_model(model_out["best_model"], model_out["best_model_name"],
                        model_out["feature_columns"])
    model_surface = modeling.predict_surface(
        model_out["best_model"], indicators, model_out["feature_columns"])
    np.save(PROCESSED_DIR / "model_surface.npy", model_surface)

    # 5) Risk scoring ----------------------------------------------------
    scorer = risk_scoring.build_scorer_from_artifacts(
        indicators, model_surface=model_surface, model_blend=0.35)
    result = scorer.compute(rainfall_mm=rainfall_mm)
    response_curve = scorer.rainfall_response_curve()
    np.save(PROCESSED_DIR / "frs_surface.npy", result.frs_surface)
    result.zone_table.to_csv(PROCESSED_DIR / "zone_scores.csv", index=False)
    result.hotspots.to_csv(PROCESSED_DIR / "hotspots.csv", index=False)
    response_curve.to_csv(PROCESSED_DIR / "rainfall_response.csv", index=False)

    log.info("SIGNATURE METRIC -> %s", result.headline())

    # 6) Visualization ---------------------------------------------------
    if not quick:
        _render_visuals(grid, terrain, hydro, data, result, model_out,
                        response_curve, scorer)

    # 7) Reports ---------------------------------------------------------
    _write_reports(result, model_out, rainfall_mm)

    elapsed = time.time() - t0
    log.info("################  PIPELINE COMPLETE in %.1fs  ################", elapsed)
    return {"result": result, "model_out": model_out, "scorer": scorer}


def _render_visuals(grid, terrain, hydro, data, result, model_out,
                    response_curve, scorer) -> None:
    log.info("=== STEP 6: VISUALIZATION ===")
    viz = visualization
    viz.plot_elevation(grid.elevation, hillshade=terrain["hillshade"])
    viz.plot_slope(terrain["slope"])
    viz.plot_twi(hydro["twi"])
    viz.plot_flow_accumulation(hydro["flow_accumulation"])
    viz.plot_frs(result.frs_surface, hillshade=terrain["hillshade"])

    viz.save_plotly(viz.kpi_gauge(result.mean_frs), "kpi_gauge.html")
    viz.save_plotly(viz.roc_figure(model_out["results"]), "roc_curves.html")
    best = model_out["results"][model_out["best_model_name"]]
    viz.save_plotly(viz.confusion_figure(best["confusion_matrix"]), "confusion_matrix.html")
    viz.save_plotly(viz.feature_importance_figure(best["feature_importance"]),
                    "feature_importance.html")
    viz.save_plotly(viz.rainfall_response_figure(response_curve), "rainfall_response.html")
    viz.save_plotly(viz.zone_bar_figure(result.zone_table), "zone_scores.html")
    viz.save_plotly(viz.rainfall_timeseries_figure(data["rainfall_timeseries"]),
                    "rainfall_timeseries.html")

    viz.folium_risk_map(result.frs_surface, scorer.lons, scorer.lats,
                        hotspots=result.hotspots, zones=result.zone_table,
                        save_path=MAPS_DIR / "flood_risk_map.html")
    viz.folium_drainage_map(data["drainage"], data["flood_points"],
                            save_path=MAPS_DIR / "drainage_map.html")


def _write_reports(result, model_out, rainfall_mm) -> None:
    best_name = model_out["best_model_name"]
    best = model_out["results"][best_name]
    metrics_payload = {
        "signature_metric": result.headline(),
        "best_model": best_name,
        "model_metrics": {k: v["metrics"] for k, v in model_out["results"].items()},
        "feature_importance": best["feature_importance"],
    }
    save_json(metrics_payload, REPORTS_DIR / "model_metrics.json")

    lines = [
        "# Urban Flood Intelligence Platform - Operational Summary", "",
        f"- **Study area:** {utils.STUDY_AREA.name}",
        f"- **Rainfall scenario:** {rainfall_mm:.0f} mm / 24h", "",
        "## Signature Metric", "",
        f"| Flood Risk Score | {result.mean_frs:.1f} / 100 |",
        "|---|---|",
        f"| Overall Category | {result.category} |",
        f"| Drainage Stress | {result.drainage_stress} |",
        f"| Terrain Vulnerability | {result.terrain_vulnerability} |",
        f"| Affected Zones | {result.affected_zones} |", "",
        "## Highest-Risk Zones", "",
        _df_to_markdown(result.zone_table.head(6)),
        "", f"## Best Model: {best_name}", "",
        f"- ROC-AUC: {best['metrics']['roc_auc']:.3f}",
        f"- F1: {best['metrics']['f1']:.3f}",
        f"- Accuracy: {best['metrics']['accuracy']:.3f}",
    ]
    report_path = REPORTS_DIR / "operational_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    result.zone_table.to_csv(REPORTS_DIR / "risk_table.csv", index=False)
    log.info("Reports written to %s", REPORTS_DIR)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table without the tabulate dep."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = ["| " + " | ".join(str(v) for v in rec) + " |"
                for rec in df.itertuples(index=False, name=None)]
        return "\n".join([header, sep, *rows])


def _parquet_ok() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the UFIP pipeline.")
    ap.add_argument("--rainfall", type=float, default=RAINFALL_REFERENCE_MM,
                    help="Rainfall scenario in mm/24h (default: reference).")
    ap.add_argument("--quick", action="store_true",
                    help="Skip heavy visualisation exports.")
    args = ap.parse_args()
    main(rainfall_mm=args.rainfall, quick=args.quick)
