# =============================================================================
# ZERVE BLOCK 06 — VISUALIZATION & REPORT ARTIFACTS  (Output Gallery)
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : scorer, frs_result, zone_table, hotspots,
#                       model_results, rainfall_ts
# OUTPUTS (downstream) : fig_gauge, fig_zone, fig_response, fig_roc,
#                        fig_importance, fig_rainfall, risk_map, summary_md
# =============================================================================
# Every figure produced here flows into the Zerve Output Gallery and can be
# dropped straight into an Agentic Report or referenced by the deployment.
import io
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_REPO_ZIP = (
    "https://github.com/joashomondi/urban-flood-intelligence-platform/"
    "archive/refs/heads/main.zip"
)


def _engine_cache() -> Path:
    fixed = Path("/tmp/ufip_engine")
    if fixed.parent.exists():
        return fixed
    return Path(tempfile.gettempdir()) / "ufip_engine"


def _ensure_engine() -> None:
    try:
        import src  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    cache = _engine_cache()
    if not (cache / "src" / "__init__.py").exists():
        print("[engine] Downloading from GitHub (zip) …")
        raw = urllib.request.urlopen(_REPO_ZIP, timeout=180).read()
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(cache)
        extracted = next(cache.glob("urban-flood-intelligence-platform-*"), None)
        if extracted is None:
            raise RuntimeError("Repo folder not found in downloaded zip.")
        for item in extracted.iterdir():
            dest = cache / item.name
            if not dest.exists():
                item.rename(dest)
        extracted.rmdir()
    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))
    try:
        import src  # noqa: F401
        return
    except ModuleNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", str(cache)],
            capture_output=True,
        )
        if str(cache) not in sys.path:
            sys.path.insert(0, str(cache))
        import src  # noqa: F401


_ensure_engine()

from src import visualization as viz

try:                          # inherited from blocks 04 + 05
    scorer; frs_result; zone_table; hotspots; model_results; rainfall_ts
except NameError:             # fallback: full recompute
    from src.pipeline import build_state
    _s = build_state(train=True)
    scorer, frs_result = _s["scorer"], _s["result"]
    zone_table, hotspots = _s["zone_table"], _s["hotspots"]
    model_results = _s["model_out"]["results"]
    rainfall_ts = _s["rainfall_ts"]

_best = max(model_results, key=lambda k: model_results[k]["metrics"]["roc_auc"])

# Plotly figures -> Output Gallery
fig_gauge = viz.kpi_gauge(frs_result.mean_frs)
fig_zone = viz.zone_bar_figure(zone_table)
fig_response = viz.rainfall_response_figure(scorer.rainfall_response_curve())
fig_roc = viz.roc_figure(model_results)
fig_importance = viz.feature_importance_figure(
    model_results[_best]["feature_importance"])
fig_rainfall = viz.rainfall_timeseries_figure(rainfall_ts)

# Interactive Folium map (renders inline / saved to gallery)
risk_map = viz.folium_risk_map(
    frs_result.frs_surface, scorer.lons, scorer.lats,
    hotspots=hotspots, zones=zone_table, step=3)

# One-page operational brief (Markdown) for an Agentic Report
_h = frs_result.headline()
summary_md = (
    f"### Flood Intelligence Summary — Nairobi, Kenya\n\n"
    f"- **Flood Risk Score:** {_h['flood_risk_score']}/100 ({_h['category']})\n"
    f"- **Drainage stress:** {_h['drainage_stress']}\n"
    f"- **Terrain vulnerability:** {_h['terrain_vulnerability']}\n"
    f"- **Affected zones:** {_h['affected_zones']}\n"
)

print("[06] figures + risk_map + summary_md ready for gallery / deployment")
