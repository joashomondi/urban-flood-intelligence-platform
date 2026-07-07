"""
visualization.py
================
Interactive Geospatial Visualization layer.

Produces the polished, operational visuals the platform is judged on:

    * Matplotlib static maps  : elevation, slope, hillshade, FRS, hydrology
    * Folium interactive maps : FRS heatmap, hotspot markers, drainage overlay
    * Plotly figures          : KPI gauge, ROC, confusion matrix, feature
                                importance, rainfall response, zone bars

Every function returns the figure/object *and* can persist it to ``outputs/``
so the notebooks, pipeline and Streamlit app all share one visual language.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import utils
from .utils import (CATEGORY_COLORS, FIGURES_DIR, MAPS_DIR, STUDY_AREA,
                    get_logger)

log = get_logger("ufip.viz")

import matplotlib
matplotlib.use("Agg")  # headless-safe for Zerve / CI
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    _HAS_FOLIUM = True
except Exception:  # pragma: no cover
    _HAS_FOLIUM = False


# Shared color ramp for flood risk (green -> amber -> red).
FLOOD_CMAP = LinearSegmentedColormap.from_list(
    "flood_risk", ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"])

_EXTENT = [STUDY_AREA.min_lon, STUDY_AREA.max_lon,
           STUDY_AREA.min_lat, STUDY_AREA.max_lat]


# ---------------------------------------------------------------------------
# Matplotlib static maps
# ---------------------------------------------------------------------------
def _static_map(array: np.ndarray, title: str, cmap: str | object,
                cbar_label: str, fname: Optional[str] = None,
                hillshade: Optional[np.ndarray] = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    if hillshade is not None:
        ax.imshow(hillshade, cmap="gray", extent=_EXTENT, alpha=0.55)
        im = ax.imshow(array, cmap=cmap, extent=_EXTENT, alpha=0.65)
    else:
        im = ax.imshow(array, cmap=cmap, extent=_EXTENT)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    if fname:
        path = FIGURES_DIR / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved figure -> %s", path)
        return path
    return None


def plot_elevation(dem: np.ndarray, hillshade=None, save=True) -> Path | None:
    return _static_map(dem, f"Elevation - {STUDY_AREA.name}", "terrain",
                       "Elevation (m)", "elevation_map.png" if save else None,
                       hillshade=hillshade)


def plot_slope(slope: np.ndarray, save=True) -> Path | None:
    return _static_map(slope, "Slope", "YlOrBr", "Slope (degrees)",
                       "slope_map.png" if save else None)


def plot_twi(twi: np.ndarray, save=True) -> Path | None:
    return _static_map(twi, "Topographic Wetness Index (Hydrological Heatmap)",
                       "YlGnBu", "TWI", "twi_heatmap.png" if save else None)


def plot_flow_accumulation(acc: np.ndarray, save=True) -> Path | None:
    return _static_map(np.log1p(acc), "Flow Accumulation (log scale)",
                       "Blues", "log(1 + upslope cells)",
                       "flow_accumulation.png" if save else None)


def plot_frs(frs: np.ndarray, hillshade=None, save=True,
             fname="frs_map.png") -> Path | None:
    return _static_map(frs, "Flood Risk Score (FRS)", FLOOD_CMAP,
                       "FRS (0-100)", fname if save else None,
                       hillshade=hillshade)


# ---------------------------------------------------------------------------
# Folium interactive maps
# ---------------------------------------------------------------------------
def folium_risk_map(frs: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                    hotspots: Optional[pd.DataFrame] = None,
                    zones: Optional[pd.DataFrame] = None,
                    step: int = 2, save_path: Optional[Path] = None):
    """Interactive Folium map: FRS heat layer + hotspot & zone markers."""
    if not _HAS_FOLIUM:
        log.warning("folium not installed - skipping interactive map.")
        return None

    m = folium.Map(location=list(STUDY_AREA.center), zoom_start=11,
                   tiles="CartoDB positron", control_scale=True)

    # Heat layer (sub-sampled for performance).
    heat = []
    h, w = frs.shape
    for r in range(0, h, step):
        for c in range(0, w, step):
            heat.append([float(lats[r]), float(lons[c]), float(frs[r, c]) / 100.0])
    HeatMap(heat, radius=9, blur=12, min_opacity=0.35,
            gradient={0.3: "#1a9850", 0.5: "#fee08b",
                      0.7: "#fc8d59", 0.9: "#d73027"},
            name="FRS Heat").add_to(m)

    # Zone markers.
    if zones is not None:
        fg = folium.FeatureGroup(name="Zone Risk").add_to(m)
        for _, z in zones.iterrows():
            color = CATEGORY_COLORS.get(z["category"], "#333")
            folium.CircleMarker(
                [z["lat"], z["lon"]], radius=9, color=color,
                fill=True, fill_color=color, fill_opacity=0.85,
                tooltip=f"{z['zone']}: FRS {z['frs']} ({z['category']})",
                popup=folium.Popup(
                    f"<b>{z['zone']}</b><br>FRS: {z['frs']}/100<br>"
                    f"Peak: {z['peak_frs']}<br>Class: {z['category']}",
                    max_width=220),
            ).add_to(fg)

    # Hotspot markers.
    if hotspots is not None:
        cluster = MarkerCluster(name="Top Hotspots").add_to(m)
        for _, hp in hotspots.iterrows():
            folium.Marker(
                [hp["lat"], hp["lon"]],
                icon=folium.Icon(color="red", icon="tint", prefix="fa"),
                tooltip=f"#{int(hp['rank'])} FRS {hp['frs']} ({hp['category']})",
            ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        log.info("Saved interactive map -> %s", save_path)
    return m


def folium_drainage_map(drainage, flood_points: Optional[pd.DataFrame] = None,
                        save_path: Optional[Path] = None):
    """Interactive map of the drainage network and historical flood points."""
    if not _HAS_FOLIUM:
        return None
    m = folium.Map(location=list(STUDY_AREA.center), zoom_start=11,
                   tiles="CartoDB dark_matter", control_scale=True)
    # Drainage lines (works for GeoDataFrame or the fallback DataFrame).
    try:
        if hasattr(drainage, "geometry"):
            for _, row in drainage.iterrows():
                coords = [(y, x) for x, y in row.geometry.coords]
                folium.PolyLine(coords, color="#3498db", weight=2.5,
                                opacity=0.8).add_to(m)
        else:
            for _, row in drainage.iterrows():
                coords = [(lat, lon) for lon, lat in row["coords"]]
                folium.PolyLine(coords, color="#3498db", weight=2.5,
                                opacity=0.8).add_to(m)
    except Exception as exc:  # pragma: no cover
        log.warning("Could not render drainage lines: %s", exc)

    if flood_points is not None:
        for _, fp in flood_points.iterrows():
            folium.CircleMarker(
                [fp["lat"], fp["lon"]], radius=3 + float(fp["severity"]),
                color="#e74c3c", fill=True, fill_opacity=0.6,
                tooltip=f"Flood incident (sev {int(fp['severity'])})").add_to(m)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        log.info("Saved drainage map -> %s", save_path)
    return m


# ---------------------------------------------------------------------------
# Plotly KPI / analytics figures
# ---------------------------------------------------------------------------
def kpi_gauge(score: float, title: str = "Flood Risk Score") -> go.Figure:
    """Speedometer-style FRS gauge for the dashboard header."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": " / 100", "font": {"size": 40}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#2c3e50"},
            "steps": [
                {"range": [0, 25], "color": CATEGORY_COLORS["Low"]},
                {"range": [25, 50], "color": CATEGORY_COLORS["Moderate"]},
                {"range": [50, 75], "color": CATEGORY_COLORS["High"]},
                {"range": [75, 100], "color": CATEGORY_COLORS["Severe"]},
            ],
            "threshold": {"line": {"color": "black", "width": 4},
                          "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def roc_figure(results: Dict[str, dict]) -> go.Figure:
    """Overlaid ROC curves for every trained model."""
    fig = go.Figure()
    for name, res in results.items():
        roc = res["roc"]
        auc = res["metrics"]["roc_auc"]
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                 name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"),
                             name="Chance", showlegend=True))
    fig.update_layout(title="ROC Curves - Flood Susceptibility Models",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      height=420, legend=dict(x=0.55, y=0.1))
    return fig


def confusion_figure(cm, labels=("No Flood", "Flood")) -> go.Figure:
    """Annotated confusion-matrix heatmap."""
    cm = np.asarray(cm)
    fig = go.Figure(go.Heatmap(
        z=cm, x=list(labels), y=list(labels), colorscale="Blues",
        text=cm, texttemplate="%{text}", showscale=True))
    fig.update_layout(title="Confusion Matrix (best model)",
                      xaxis_title="Predicted", yaxis_title="Actual",
                      height=380)
    fig.update_yaxes(autorange="reversed")
    return fig


def feature_importance_figure(importance: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of model feature importances."""
    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [k.replace("_", " ").title() for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                           marker_color="#2980b9"))
    fig.update_layout(title="Feature Importance", xaxis_title="Relative importance",
                      height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def rainfall_response_figure(curve: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: mean FRS and affected zones vs rainfall."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=curve["rainfall_mm"], y=curve["mean_frs"],
                             name="Mean FRS", mode="lines+markers",
                             line=dict(color="#d73027", width=3)),
                  secondary_y=False)
    fig.add_trace(go.Bar(x=curve["rainfall_mm"], y=curve["affected_zones"],
                         name="Affected Zones", marker_color="#4575b4",
                         opacity=0.45),
                  secondary_y=True)
    fig.update_layout(title="Rainfall Response Curve",
                      xaxis_title="Rainfall (mm / 24h)", height=380,
                      legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="Mean FRS", secondary_y=False)
    fig.update_yaxes(title_text="Affected Zones", secondary_y=True)
    return fig


def zone_bar_figure(zone_table: pd.DataFrame) -> go.Figure:
    """Per-zone FRS bar chart, coloured by vulnerability category."""
    colors = [CATEGORY_COLORS.get(c, "#888") for c in zone_table["category"]]
    fig = go.Figure(go.Bar(
        x=zone_table["frs"], y=zone_table["zone"], orientation="h",
        marker_color=colors, text=zone_table["category"],
        textposition="outside"))
    fig.update_layout(title="Flood Risk Score by Zone",
                      xaxis_title="FRS (0-100)", height=440,
                      yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def rainfall_timeseries_figure(ts: pd.DataFrame) -> go.Figure:
    """Daily rainfall time-series with wet-season context."""
    fig = go.Figure(go.Scatter(
        x=ts["date"], y=ts["rainfall_mm"], mode="lines",
        fill="tozeroy", line=dict(color="#2980b9")))
    fig.update_layout(title="Daily Rainfall (proxy CHIRPS)",
                      xaxis_title="Date", yaxis_title="Rainfall (mm)",
                      height=320)
    return fig


def save_plotly(fig: go.Figure, fname: str) -> Path:
    """Persist a Plotly figure as a self-contained HTML file."""
    path = FIGURES_DIR / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("Saved plotly figure -> %s", path)
    return path
