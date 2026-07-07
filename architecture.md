# System Architecture — Urban Flood Intelligence Platform (UFIP)

> A Real-Time Geospatial Risk Scoring and Stormwater Analysis Pipeline

This document describes the system design, data flow, module contracts and
deployment topology of the platform. It is written for engineers and reviewers
who need to understand *how the system is built*, not just *what it does*.

---

## 1. Design principles

| Principle | How it is realised |
| --- | --- |
| **Modular engines** | Each stage is an independent, importable module in `src/` with a single responsibility and a clean function contract. |
| **Reproducibility** | A single `RANDOM_SEED` drives every stochastic step. The pipeline runs deterministically end-to-end. |
| **Real-data first, synthetic fallback** | Real DEM/rainfall rasters are used when present; otherwise physically-plausible surfaces are synthesised so the system runs on a cold clone with no network access. |
| **Operational output** | Every stage emits artefacts (`data/processed`, `outputs/`) that the dashboard and reports consume — nothing lives only in notebook memory. |
| **Graceful degradation** | Heavy dependencies (rasterio, geopandas, folium, xgboost, joblib) are optional; the core pipeline still runs and falls back sensibly. |

---

## 2. High-level workflow

```
          ┌───────────────────────────────────────────────────────────┐
          │                     RAW INPUTS (data/raw)                   │
          │   SRTM DEM · CHIRPS rainfall · OSM drainage · flood points  │
          └───────────────────────────────┬───────────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   1. DATA INGESTION       │  data_loader.py
                              │   CRS · clip · fill NaN   │
                              └────────────┬────────────┘
                                           │  DEM grid + rainfall field
                    ┌──────────────────────▼──────────────────────┐
                    │        2. TERRAIN PROCESSING ENGINE           │  terrain_analysis.py
                    │   slope · aspect · roughness · curvature      │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │     2b. HYDROLOGICAL FEATURE EXTRACTION       │  hydrology.py
                    │  fill · D8 flow dir · flow acc · TWI · dens   │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │          3. FEATURE ENGINEERING               │  feature_engineering.py
                    │   6 normalised indicators + synthetic labels  │
                    └──────────────────────┬──────────────────────┘
                                           │  feature table (tidy per-cell)
                    ┌──────────────────────▼──────────────────────┐
                    │        4. FLOOD SUSCEPTIBILITY MODEL          │  modeling.py
                    │   LogReg · RandomForest · XGBoost + metrics   │
                    └──────────────────────┬──────────────────────┘
                                           │  probability surface
                    ┌──────────────────────▼──────────────────────┐
                    │        5. FLOOD RISK SCORE ENGINE (FRS)       │  risk_scoring.py
                    │   0–100 score · zones · categories · hotspots │
                    └──────────────────────┬──────────────────────┘
                                           │
              ┌────────────────────────────▼────────────────────────────┐
              │            6. VISUALIZATION & INTELLIGENCE OUTPUT         │  visualization.py
              │   Folium maps · Plotly KPIs · static maps · reports      │
              └────────────────────────────┬────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │      7. OPERATIONAL MONITORING DASHBOARD      │  streamlit_app/app.py
                    │   live rainfall · dynamic FRS · downloads     │
                    └───────────────────────────────────────────────┘
```

---

## 3. Module contracts

| Module | Key entry points | Consumes | Produces |
| --- | --- | --- | --- |
| `utils` | `STUDY_AREA`, `SCORING`, `normalize`, `get_logger` | — | config, paths, helpers |
| `data_loader` | `ingest_all`, `load_dem`, `load_rainfall_field` | `data/raw/*` (optional) | `Grid`, rainfall field, vector layers |
| `terrain_analysis` | `analyze_terrain` | DEM array | slope, aspect, roughness, curvature, hillshade |
| `hydrology` | `analyze_hydrology` | DEM, slope | filled DEM, flow dir/acc, TWI, drainage density |
| `feature_engineering` | `engineer_features` | terrain, hydro, rainfall | 6 indicators, labels, feature table |
| `modeling` | `train_and_evaluate`, `predict_surface`, `save_model` | feature table | fitted model, metrics, probability surface |
| `risk_scoring` | `FloodRiskScorer`, `build_scorer_from_artifacts` | indicators, model surface | FRS surface, zone table, hotspots |
| `visualization` | `folium_risk_map`, `kpi_gauge`, `roc_figure`, … | any artefact | maps + figures |

Each module logs through a namespaced logger (`ufip.<module>`).

---

## 4. The FRS scoring model

The Flood Risk Score is a weighted blend of six normalised indicators,
optionally fused with the ML probability surface:

```
FRS(cell) = 100 · [ (1 − λ) · Σ wᵢ · indicatorᵢ(cell)  +  λ · P_model(cell) ]
```

* `wᵢ` — indicator weights (sum to 1.0), defined in `utils.ScoringConfig`.
* `λ` — the ML blend weight (dashboard-adjustable, default 0.35).
* Rainfall enters through a **saturating gain** on the rainfall indicator, so
  the score responds dynamically to the rainfall slider without growing
  unbounded at extreme totals.

Categories: **Low** (<25) · **Moderate** (25–50) · **High** (50–75) · **Severe** (≥75).

---

## 5. Artefact layout

```
data/processed/     dem.npy · rainfall_field.npy · indicator_*.npy ·
                    layer_*.npy · feature_table.{parquet,csv} ·
                    model_surface.npy · frs_surface.npy · zone_scores.csv ·
                    hotspots.csv · rainfall_response.csv · config_summary.json
outputs/models/     flood_model.joblib
outputs/figures/    *.png (static maps) · *.html (Plotly)
outputs/maps/       flood_risk_map.html · drainage_map.html
outputs/reports/    model_metrics.json · operational_summary.md · risk_table.csv
```

---

## 6. Deployment topology

```
┌────────────────────┐     python scripts/run_pipeline.py      ┌──────────────────┐
│  Batch pipeline     │ ─────────────────────────────────────▶ │ data/processed + │
│  (Zerve / CI / cron)│                                         │ outputs/         │
└────────────────────┘                                         └────────┬─────────┘
                                                                        │ reads artefacts
                                                              ┌─────────▼──────────┐
                                                              │  Streamlit service │
                                                              │  streamlit run …   │
                                                              └────────────────────┘
```

* **Batch layer** — the pipeline can run on a schedule (each new rainfall feed)
  to refresh scores and reports.
* **Serving layer** — the Streamlit app reads pre-computed artefacts for instant
  load, and recomputes the FRS live as the user changes rainfall / thresholds.
* **Cold-start safe** — if artefacts are missing, the app bootstraps the base
  pipeline itself, so a fresh deploy still works.

---

## 7. Scalability path

1. **Real data** — drop SRTM `dem.tif` + CHIRPS rasters into `data/raw`; the
   loader reprojects and clips automatically.
2. **Larger AOIs** — raise `StudyArea.grid_size`; hydrology is `O(N log N)`.
3. **True live feeds** — replace `build_rainfall_timeseries` with an API client
   (e.g. TAHMO / OpenWeather) and schedule the batch layer.
4. **Tiled serving** — export the FRS surface as XYZ raster tiles for web maps.
5. **Model registry** — persist versioned models and metrics per run for MLOps
   traceability.
```
