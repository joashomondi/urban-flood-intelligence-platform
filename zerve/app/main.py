"""
Zerve Deployment Entry Point — Streamlit
========================================
Urban Flood Intelligence Platform · operational flood monitoring console.

On Zerve
--------
This app loads the artefacts produced by the canvas blocks via
``from zerve import variable`` (no pipeline rebuild — outputs are already
computed and cached by the DAG). It stays interactive because the ``scorer``
object recomputes the Flood Risk Score live as the user moves the sliders.

Off Zerve (local / any host)
----------------------------
If the ``zerve`` runtime is unavailable it transparently falls back to
``src.pipeline.build_state`` so the exact same file runs with:

    streamlit run zerve/app/main.py

Deployment config (Zerve → Deploy → Streamlit):
    App Script Name : main.py
    Instance Type   : any
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make ``src`` importable when run locally from the repo.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import visualization as viz
from src.feature_engineering import INDICATOR_NAMES
from src.utils import (CATEGORY_COLORS, NAIROBI_ZONES, RAINFALL_MAX_MM,
                       RAINFALL_REFERENCE_MM, STUDY_AREA, SCORING)

try:
    from streamlit_folium import st_folium
    _HAS_ST_FOLIUM = True
except Exception:
    _HAS_ST_FOLIUM = False


# --------------------------------------------------------------------------
# Load canvas outputs (Zerve) or compute locally (fallback)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading flood intelligence state…")
def load_state() -> dict:
    """Pull block outputs from the Zerve canvas, else build the state locally."""
    try:
        from zerve import variable  # available only inside a Zerve deployment
        scorer = variable("block_05_scoring", "scorer")
        state = {
            "scorer": scorer,
            "rainfall_ts": variable("block_01_ingest", "rainfall_ts"),
            "model_results": variable("block_04_model", "model_results"),
            "source": "zerve-canvas",
        }
        return state
    except Exception:
        from src.pipeline import build_state
        s = build_state(train=True)
        return {
            "scorer": s["scorer"],
            "rainfall_ts": s["rainfall_ts"],
            "model_results": s["model_out"]["results"] if s["model_out"] else {},
            "source": "local-fallback",
        }


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Urban Flood Intelligence Platform",
                   page_icon="🌊", layout="wide")

st.markdown(
    """
    <style>
    .kpi-card{background:linear-gradient(135deg,#1e3c72,#2a5298);border-radius:14px;
      padding:16px 18px;color:#fff;box-shadow:0 4px 14px rgba(0,0,0,.18);}
    .kpi-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;opacity:.85;}
    .kpi-value{font-size:1.9rem;font-weight:700;line-height:1.1;}
    .kpi-sub{font-size:.82rem;opacity:.9;}
    .pill{display:inline-block;padding:3px 12px;border-radius:20px;color:#fff;
      font-weight:600;font-size:.8rem;margin-right:4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

STATE = load_state()
SCORER = STATE["scorer"]

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.title("🌊 Control Center")
st.sidebar.caption("Real-time stormwater scenario controls")

rainfall = st.sidebar.slider("Rainfall intensity (mm / 24h)", 5.0,
                             float(RAINFALL_MAX_MM), float(RAINFALL_REFERENCE_MM), 5.0)
selected_zone = st.sidebar.selectbox("Focus zone",
                                     ["All Zones"] + list(NAIROBI_ZONES))
threshold = st.sidebar.slider("Affected-zone FRS threshold", 30, 90,
                              int(SCORING.affected_threshold), 5)
if SCORER.model_surface is not None:
    SCORER.model_blend = st.sidebar.slider("ML model blend", 0.0, 1.0, 0.35, 0.05)

st.sidebar.divider()
st.sidebar.caption(f"Study area: **{STUDY_AREA.name}**")
st.sidebar.caption(f"State source: `{STATE['source']}`")

# --------------------------------------------------------------------------
# Live scoring for the current scenario
# --------------------------------------------------------------------------
result = SCORER.compute(rainfall_mm=rainfall, threshold=float(threshold))
h = result.headline()
zone_table = result.zone_table

st.title("Urban Flood Intelligence Platform")
st.caption("A Real-Time Geospatial Risk Scoring and Stormwater Analysis "
           f"Pipeline — operational flood monitoring for {STUDY_AREA.name}.")


def kpi(col, label, value, sub, color="#2a5298"):
    col.markdown(
        f"<div class='kpi-card' style='background:linear-gradient(135deg,{color},#2a5298);'>"
        f"<div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)


c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Flood Risk Score", f"{h['flood_risk_score']:.0f}/100", h["category"],
    CATEGORY_COLORS.get(h["category"], "#2a5298"))
kpi(c2, "Drainage Stress", h["drainage_stress"], "network load")
kpi(c3, "Terrain Vulnerability", h["terrain_vulnerability"], "topographic")
kpi(c4, "Affected Zones", f"{h['affected_zones']}", f"of {len(NAIROBI_ZONES)}",
    "#c0392b" if h["affected_zones"] > 5 else "#2a5298")
kpi(c5, "Rainfall Intensity", f"{h['rainfall_mm']:.0f} mm", "per 24h")

st.divider()

tab_map, tab_an, tab_model, tab_zone, tab_rep = st.tabs(
    ["🗺️ Risk Map", "📈 Analytics", "🤖 Model", "🏙️ Zones", "📄 Report"])

with tab_map:
    left, right = st.columns([3, 2])
    with left:
        if _HAS_ST_FOLIUM and viz._HAS_FOLIUM:
            fmap = viz.folium_risk_map(result.frs_surface, SCORER.lons, SCORER.lats,
                                       hotspots=result.hotspots, zones=zone_table, step=3)
            st_folium(fmap, height=520, use_container_width=True, returned_objects=[])
        else:
            st.plotly_chart(viz.kpi_gauge(result.mean_frs), use_container_width=True)
    with right:
        st.markdown("**Top flood hotspots**")
        st.dataframe(result.hotspots, hide_index=True, use_container_width=True, height=320)
        for cat, col in CATEGORY_COLORS.items():
            st.markdown(f"<span class='pill' style='background:{col};'>{cat}</span>",
                        unsafe_allow_html=True)

with tab_an:
    a1, a2 = st.columns(2)
    a1.plotly_chart(viz.rainfall_response_figure(SCORER.rainfall_response_curve()),
                    use_container_width=True)
    a2.plotly_chart(viz.zone_bar_figure(zone_table), use_container_width=True)
    st.plotly_chart(viz.rainfall_timeseries_figure(STATE["rainfall_ts"]),
                    use_container_width=True)

with tab_model:
    mr = STATE["model_results"]
    if mr:
        best = max(mr, key=lambda k: mr[k]["metrics"]["roc_auc"])
        st.markdown(f"**Best model:** `{best}`")
        st.dataframe(pd.DataFrame({k: v["metrics"] for k, v in mr.items()}).T.round(3),
                     use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.roc_figure(mr), use_container_width=True)
        c2.plotly_chart(viz.feature_importance_figure(mr[best]["feature_importance"]),
                        use_container_width=True)
    else:
        st.info("No model metrics in state (block_04 not run).")

with tab_zone:
    if selected_zone != "All Zones":
        row = zone_table[zone_table["zone"] == selected_zone].iloc[0]
        z1, z2, z3 = st.columns(3)
        z1.metric("Zone FRS", f"{row['frs']:.0f}/100")
        z2.metric("Peak FRS", f"{row['peak_frs']:.0f}/100")
        z3.metric("Category", row["category"])
    st.dataframe(zone_table, hide_index=True, use_container_width=True, height=440)

with tab_rep:
    top = zone_table.head(5)
    md = [f"### Flood Intelligence Summary — {STUDY_AREA.name}", "",
          f"- **Scenario rainfall:** {rainfall:.0f} mm / 24h",
          f"- **Flood Risk Score:** {h['flood_risk_score']:.0f}/100 ({h['category']})",
          f"- **Drainage stress:** {h['drainage_stress']}",
          f"- **Terrain vulnerability:** {h['terrain_vulnerability']}",
          f"- **Affected zones:** {h['affected_zones']} of {len(NAIROBI_ZONES)}", "",
          "| Zone | FRS | Category |", "| --- | --- | --- |"]
    md += [f"| {r['zone']} | {r['frs']:.0f} | {r['category']} |" for _, r in top.iterrows()]
    summary_md = "\n".join(md)
    st.markdown(summary_md)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Summary (MD)", summary_md, "flood_summary.md", "text/markdown")
    d2.download_button("⬇️ Zone table (CSV)", zone_table.to_csv(index=False),
                       "zone_risk_table.csv", "text/csv")
    d3.download_button("⬇️ Hotspots (CSV)", result.hotspots.to_csv(index=False),
                       "flood_hotspots.csv", "text/csv")

st.caption("Urban Flood Intelligence Platform · deployed on Zerve · "
           "live Flood Risk Score engine.")
