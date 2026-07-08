# Running the Urban Flood Intelligence Platform on Zerve

This folder maps the platform onto **Zerve's canvas + deployment model** for the
[Zerve × HackerEarth $10,000 Data Challenge](https://zerve.hackerearth.com/)
(**Track 04 — Climate & Energy**: extreme-weather prediction).

Zerve runs code as a **DAG of independent blocks** that pass state left-to-right
(each block inherits upstream variables; outputs are cached/serialized). A
**deployment** (Streamlit/FastAPI) then loads any block output with
`from zerve import variable`. This folder provides exactly that shape:

```
zerve/
├── blocks/                     # one file per canvas node (the DAG)
│   ├── block_01_ingest.py      #  -> grid, dem, rainfall_field, rainfall_ts, ...
│   ├── block_02_terrain_hydrology.py  # dem            -> terrain, hydro, slope
│   ├── block_03_features.py    # terrain,hydro,rain    -> indicators, feature_table
│   ├── block_04_model.py       # feature_table         -> model, model_surface, ...
│   ├── block_05_scoring.py     # indicators,surface    -> scorer, frs_result (FRS!)
│   └── block_06_visuals.py     # scorer,results        -> figures, risk_map, summary_md
├── app/
│   └── main.py                 # Streamlit deployment entry (uses zerve.variable)
├── requirements.txt            # canvas / deployment environment
└── README.md                   # this file
```

## DAG

```
block_01_ingest ─┬─> block_02_terrain_hydrology ─┐
                 │                                 ├─> block_03_features ─> block_04_model ─┐
                 └────────────────(rainfall_field)─┘                                        │
                                                                                            v
                                    block_06_visuals <── block_05_scoring <─────────────────┘
                                            │
                                            v
                                   Streamlit deployment (app/main.py)
```

## Build it (≈ 5 minutes)

1. **Create a canvas** in Zerve (or let the agent scaffold one) and open
   **Environments → Requirements**. Paste the contents of
   [`requirements.txt`](requirements.txt) and build the environment. Block 01
   bootstraps the engine automatically (GitHub ZIP download — no `git` CLI
   needed), so `from src import ...` works in every block after the first run.

2. **Create six Python code blocks**, left to right, and paste the matching file
   from `blocks/` into each. Name the blocks exactly:

   | Block name | File |
   | --- | --- |
   | `block_01_ingest` | `blocks/block_01_ingest.py` |
   | `block_02_terrain_hydrology` | `blocks/block_02_terrain_hydrology.py` |
   | `block_03_features` | `blocks/block_03_features.py` |
   | `block_04_model` | `blocks/block_04_model.py` |
   | `block_05_scoring` | `blocks/block_05_scoring.py` |
   | `block_06_visuals` | `blocks/block_06_visuals.py` |

   > Block names matter: the deployment references `variable("block_05_scoring",
   > "scorer")` etc. Each block also has a self-contained fallback, so it still
   > runs if executed out of order.

3. **Connect the blocks** to match the DAG above and hit **Run All**. Watch the
   `block_05_scoring` output print the **signature metric**:

   ```
   [05]   flood_risk_score:      47.1
   [05]   category:              Moderate
   [05]   drainage_stress:       Moderate
   [05]   terrain_vulnerability: High
   [05]   affected_zones:        6
   ```

   The Plotly figures and the Folium map from `block_06_visuals` appear in the
   **Output Gallery** — drop them straight into an Agentic Report.

## Deploy the dashboard

1. **Deploy → Streamlit.** Upload / point to `app/main.py` and set
   **App Script Name = `main.py`**.
2. Use the same `requirements.txt`.
3. Deploy. The app pulls `scorer`, `rainfall_ts` and `model_results` from the
   canvas via `zerve.variable(...)` and stays fully interactive — moving the
   rainfall slider recomputes the Flood Risk Score live (the scorer object holds
   the indicators + ML surface, so no pipeline rebuild is needed).

Every deploy gets a `*.zerve.app` URL you can share in the submission.

## Real data on Zerve (optional)

The blocks synthesise a physically-plausible Nairobi DEM + rainfall by default,
so the canvas runs with zero setup. To use real data, run these once in a block
(or a terminal) before `Run All`:

```python
import subprocess
subprocess.run(["python", "-m", "scripts.download_dem"])       # NASA SRTM 30 m
subprocess.run(["python", "-m", "scripts.download_rainfall"])  # CHIRPS v2.0
```

…or upload `dem.tif` / `rainfall.tif` into `data/raw/`. The loaders auto-detect
them and the FRS becomes fully terrain-driven from real SRTM + CHIRPS.

## Why this fits the challenge

| Judging signal | This project |
| --- | --- |
| Deploys a real output (app / API / report) | Streamlit `*.zerve.app` + gallery figures + downloadable report |
| Signature operational metric | **Flood Risk Score (0-100)** — like the track's "Grid stress: 0.73" sample |
| Live / auto-refresh behaviour | Rainfall slider recomputes FRS; live-monitoring toggle |
| Rigorous & documented | Modular DAG, README, architecture, data sources, notebooks |
| Real public datasets | NASA SRTM + CHIRPS (Track 04 climate data) |
