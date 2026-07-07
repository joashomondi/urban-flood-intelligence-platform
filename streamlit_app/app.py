"""
Urban Flood Intelligence Platform - Streamlit Operational Dashboard
===================================================================
A deployable, live-like flood intelligence console.

Run
---
    streamlit run streamlit_app/app.py

The app loads pre-computed pipeline artefacts from ``data/processed`` when
available and otherwise builds them on the fly, so it works both after a full
pipeline run and on a cold clone.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- make ``src`` importable regardless of launch directory ----------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (data_loader, feature_engineering, hydrology, modeling,
                 risk_scoring, terrain_analysis, utils, visualization)
from src.feature_engineering import INDICATOR_NAMES
from src.utils import (CATEGORY_COLORS, NAIROBI_ZONES, PROCESSED_DIR,
                       RAINFALL_MAX_MM, RAINFALL_REFERENCE_MM, STUDY_AREA)

try:
    from streamlit_folium import st_folium
    _HAS_ST_FOLIUM = True
except Exception:
    _HAS_ST_FOLIUM = False


# ==========================================================================
# Page config + styling
# ==========================================================================
st.set_page_config(
    page_title="Urban Flood Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div { padding-top: 1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 14px; padding: 18px 20px; color: #fff;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    .kpi-label { font-size: 0.82rem; text-transform: uppercase;
                 letter-spacing: 0.06em; opacity: 0.85; }
    .kpi-value { font-size: 2.0rem; font-weight: 700; line-height: 1.1; }
    .kpi-sub   { font-size: 0.85rem; opacity: 0.9; }
    .status-pill { display:inline-block; padding:3px 12px; border-radius:20px;
                   font-weight:600; color:#fff; font-size:0.8rem; }
    .block-title { font-size:1.15rem; font-weight:700; margin:0.2rem 0 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Data / model loading (cached)
# ==========================================================================
@st.cache_resource(show_spinner="Booting flood intelligence engines…")
def bootstrap():
    """Load artefacts if present, else compute the base pipeline once."""
    utils.ensure_dirs()

    ind_path = PROCESSED_DIR / f"indicator_{INDICATOR_NAMES[0]}.npy"
    if ind_path.exists():
        indicators = {n: np.load(PROCESSED_DIR / f"indicator_{n}.npy")
                      for n in INDICATOR_NAMES}
        dem = np.load(PROCESSED_DIR / "dem.npy")
        hillshade = _safe_load(PROCESSED_DIR / "layer_hillshade.npy")
        model_surface = _safe_load(PROCESSED_DIR / "model_surface.npy")
        try:
            ts = pd.read_csv(PROCESSED_DIR / "rainfall_timeseries.csv",
                             parse_dates=["date"])
        except Exception:
            ts = data_loader.build_rainfall_timeseries()
        drainage = None
        flood_points = _safe_csv(PROCESSED_DIR / "flood_points.csv")
        source = "precomputed"
    else:
        data = data_loader.ingest_all()
        dem = data["grid"].elevation
        terrain = terrain_analysis.analyze_terrain(dem)
        hydro = hydrology.analyze_hydrology(dem, terrain["slope"])
        feats = feature_engineering.engineer_features(
            terrain, hydro, data["rainfall_field"])
        indicators = feats["indicators"]
        hillshade = terrain["hillshade"]
        ts = data["rainfall_timeseries"]
        drainage = data["drainage"]
        flood_points = data["flood_points"]
        model_surface = None
        source = "computed"

    # Model (optional).
    model_payload = None
    try:
        model_payload = modeling.load_model()
        if model_surface is None:
            model_surface = modeling.predict_surface(
                model_payload["model"], indicators,
                model_payload["feature_columns"])
    except Exception:
        model_payload = None

    scorer = risk_scoring.build_scorer_from_artifacts(
        indicators, model_surface=model_surface, model_blend=0.35)

    return {
        "dem": dem, "hillshade": hillshade, "indicators": indicators,
        "scorer": scorer, "rainfall_ts": ts, "drainage": drainage,
        "flood_points": flood_points, "model_payload": model_payload,
        "source": source,
    }


def _safe_load(path: Path):
    return np.load(path) if path.exists() else None


def _safe_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _frs_matplotlib(frs: np.ndarray, hillshade):
    """Static FRS surface figure for the map tab."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    extent = [STUDY_AREA.min_lon, STUDY_AREA.max_lon,
              STUDY_AREA.min_lat, STUDY_AREA.max_lat]
    if hillshade is not None:
        ax.imshow(hillshade, cmap="gray", extent=extent, alpha=0.5)
    im = ax.imshow(frs, cmap=visualization.FLOOD_CMAP, extent=extent,
                   alpha=0.7, vmin=0, vmax=100)
    ax.set_title("Flood Risk Score surface")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.8, label="FRS (0-100)")
    fig.tight_layout()
    return fig


def _build_summary_md(headline: dict, zone_table: pd.DataFrame,
                      rainfall: float) -> str:
    top = zone_table.head(5)
    lines = [
        f"### Flood Intelligence Summary — {STUDY_AREA.name}",
        "",
        f"- **Scenario rainfall:** {rainfall:.0f} mm / 24h",
        f"- **Flood Risk Score:** **{headline['flood_risk_score']:.0f}/100** "
        f"({headline['category']})",
        f"- **Drainage stress:** {headline['drainage_stress']}",
        f"- **Terrain vulnerability:** {headline['terrain_vulnerability']}",
        f"- **Affected zones:** {headline['affected_zones']} of "
        f"{len(NAIROBI_ZONES)}",
        "",
        "**Highest-risk zones**",
        "",
        "| Zone | FRS | Category |",
        "| --- | --- | --- |",
    ]
    for _, r in top.iterrows():
        lines.append(f"| {r['zone']} | {r['frs']:.0f} | {r['category']} |")
    return "\n".join(lines)


BOOT = bootstrap()
SCORER: risk_scoring.FloodRiskScorer = BOOT["scorer"]


# ==========================================================================
# Sidebar - operational controls
# ==========================================================================
st.sidebar.title("🌊 Control Center")
st.sidebar.caption("Real-time stormwater scenario controls")

rainfall = st.sidebar.slider(
    "Rainfall intensity (mm / 24h)", min_value=5.0, max_value=float(RAINFALL_MAX_MM),
    value=float(RAINFALL_REFERENCE_MM), step=5.0,
    help="Drive the live flood scenario. Higher rainfall escalates the FRS.")

zone_names = ["All Zones"] + list(NAIROBI_ZONES.keys())
selected_zone = st.sidebar.selectbox("Focus zone", zone_names, index=0)

threshold = st.sidebar.slider(
    "Affected-zone FRS threshold", 30, 90, int(utils.SCORING.affected_threshold), 5,
    help="A zone is flagged 'affected' once its FRS crosses this value.")

model_blend = st.sidebar.slider(
    "ML model blend", 0.0, 1.0, 0.35, 0.05,
    help="Weight on the ML flood-probability surface vs. the transparent "
         "weighted-indicator score.")
SCORER.model_blend = model_blend if SCORER.model_surface is not None else 0.0

live_mode = st.sidebar.toggle("🛰️ Live monitoring simulation", value=False,
                              help="Auto-refresh with fluctuating rainfall to "
                                   "mimic an operational feed.")
if live_mode:
    jitter = np.random.default_rng(int(time.time())).normal(0, 8)
    rainfall = float(np.clip(rainfall + jitter, 5, RAINFALL_MAX_MM))
    st.sidebar.metric("Live rainfall feed", f"{rainfall:.0f} mm",
                      delta=f"{jitter:+.0f} mm")

st.sidebar.divider()
st.sidebar.caption(f"Study area: **{STUDY_AREA.name}**")
st.sidebar.caption(f"Grid: {STUDY_AREA.grid_size}×{STUDY_AREA.grid_size} "
                   f"(~{STUDY_AREA.cell_size_m:.0f} m/cell)")
st.sidebar.caption(f"Data source: `{BOOT['source']}`")
if BOOT["model_payload"]:
    st.sidebar.caption(f"Model: `{BOOT['model_payload']['name']}`")


# ==========================================================================
# Compute scoring for current scenario
# ==========================================================================
result = SCORER.compute(rainfall_mm=rainfall, threshold=float(threshold))
zone_table = result.zone_table
headline = result.headline()


# ==========================================================================
# Header
# ==========================================================================
st.title("Urban Flood Intelligence Platform")
st.markdown(
    "**A Real-Time Geospatial Risk Scoring and Stormwater Analysis Pipeline** — "
    f"operational flood monitoring for *{STUDY_AREA.name}*.")

if live_mode:
    st.markdown(
        "<span class='status-pill' style='background:#e74c3c;'>● LIVE</span> "
        "Operational monitoring feed active — scores refresh with the incoming "
        "rainfall stream.", unsafe_allow_html=True)


# ==========================================================================
# KPI panels
# ==========================================================================
def kpi(col, label, value, sub, color="#2a5298"):
    col.markdown(
        f"""<div class="kpi-card" style="background:linear-gradient(135deg,{color} 0%, #2a5298 100%);">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""",
        unsafe_allow_html=True)


cat_color = CATEGORY_COLORS.get(headline["category"], "#2a5298")
c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Flood Risk Score", f"{headline['flood_risk_score']:.0f}/100",
    headline["category"], color=cat_color)
kpi(c2, "Drainage Stress", headline["drainage_stress"], "network load")
kpi(c3, "Terrain Vulnerability", headline["terrain_vulnerability"], "topographic")
kpi(c4, "Affected Zones", f"{headline['affected_zones']}",
    f"of {len(NAIROBI_ZONES)} monitored",
    color="#c0392b" if headline["affected_zones"] > 5 else "#2a5298")
kpi(c5, "Rainfall Intensity", f"{headline['rainfall_mm']:.0f} mm", "per 24h")

st.divider()


# ==========================================================================
# Main tabs
# ==========================================================================
tab_map, tab_analytics, tab_model, tab_zones, tab_report = st.tabs(
    ["🗺️ Risk Map", "📈 Analytics", "🤖 Model", "🏙️ Zones", "📄 Report"])


# ---- Tab 1: Interactive risk map -----------------------------------------
with tab_map:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("<div class='block-title'>Live Flood Risk Heatmap</div>",
                    unsafe_allow_html=True)
        if _HAS_ST_FOLIUM and visualization._HAS_FOLIUM:
            fmap = visualization.folium_risk_map(
                result.frs_surface, SCORER.lons, SCORER.lats,
                hotspots=result.hotspots, zones=zone_table, step=3)
            st_folium(fmap, height=520, use_container_width=True,
                      returned_objects=[])
        else:
            st.info("Interactive map needs `folium` + `streamlit-folium`. "
                    "Showing static FRS surface instead.")
            fig = visualization.kpi_gauge(result.mean_frs)
            st.plotly_chart(fig, use_container_width=True)
        st.pyplot(_frs_matplotlib(result.frs_surface, BOOT["hillshade"]))
    with right:
        st.markdown("<div class='block-title'>Top Flood Hotspots</div>",
                    unsafe_allow_html=True)
        st.dataframe(result.hotspots, use_container_width=True, height=300,
                     hide_index=True)
        st.markdown("<div class='block-title'>Vulnerability Legend</div>",
                    unsafe_allow_html=True)
        for cat, col in CATEGORY_COLORS.items():
            st.markdown(
                f"<span class='status-pill' style='background:{col};'>{cat}</span>",
                unsafe_allow_html=True)


# ---- Tab 2: Analytics -----------------------------------------------------
with tab_analytics:
    a1, a2 = st.columns(2)
    with a1:
        curve = SCORER.rainfall_response_curve()
        st.plotly_chart(visualization.rainfall_response_figure(curve),
                        use_container_width=True)
    with a2:
        st.plotly_chart(visualization.zone_bar_figure(zone_table),
                        use_container_width=True)
    st.plotly_chart(
        visualization.rainfall_timeseries_figure(BOOT["rainfall_ts"]),
        use_container_width=True)
    st.markdown("<div class='block-title'>Flood Indicator Composition</div>",
                unsafe_allow_html=True)
    icols = st.columns(3)
    for i, name in enumerate(INDICATOR_NAMES):
        icols[i % 3].metric(name.replace("_", " ").title(),
                            f"{BOOT['indicators'][name].mean():.2f}")


# ---- Tab 3: Model ---------------------------------------------------------
with tab_model:
    metrics_path = utils.REPORTS_DIR / "model_metrics.json"
    if metrics_path.exists():
        metrics = utils.load_json(metrics_path)
        st.markdown(f"**Best model:** `{metrics.get('best_model', 'n/a')}`")
        mm = metrics.get("model_metrics", {})
        if mm:
            st.dataframe(pd.DataFrame(mm).T.round(3), use_container_width=True)
        fi = metrics.get("feature_importance", {})
        if fi:
            st.plotly_chart(visualization.feature_importance_figure(fi),
                            use_container_width=True)
    else:
        st.info("Run `python scripts/run_pipeline.py` to generate full model "
                "evaluation artefacts (ROC, confusion matrix, importances).")
        if BOOT["model_payload"]:
            st.success(f"Live model loaded: {BOOT['model_payload']['name']}")


# ---- Tab 4: Zones ---------------------------------------------------------
with tab_zones:
    if selected_zone != "All Zones":
        row = zone_table[zone_table["zone"] == selected_zone].iloc[0]
        z1, z2, z3 = st.columns(3)
        z1.metric("Zone FRS", f"{row['frs']:.0f}/100")
        z2.metric("Peak FRS", f"{row['peak_frs']:.0f}/100")
        z3.metric("Category", row["category"])
    st.dataframe(
        zone_table.style.background_gradient(subset=["frs"], cmap="OrRd"),
        use_container_width=True, hide_index=True, height=440)


# ---- Tab 5: Report / downloads -------------------------------------------
with tab_report:
    st.markdown("<div class='block-title'>Operational Summary</div>",
                unsafe_allow_html=True)
    summary_md = _build_summary_md(headline, zone_table, rainfall)
    st.markdown(summary_md)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Summary report (Markdown)", summary_md,
                       file_name="flood_summary.md", mime="text/markdown")
    d2.download_button("⬇️ Zone risk table (CSV)",
                       zone_table.to_csv(index=False),
                       file_name="zone_risk_table.csv", mime="text/csv")
    d3.download_button("⬇️ Hotspots (CSV)",
                       result.hotspots.to_csv(index=False),
                       file_name="flood_hotspots.csv", mime="text/csv")

st.caption("Urban Flood Intelligence Platform · built for operational "
           "decision-making · synthetic + real-data ready.")
