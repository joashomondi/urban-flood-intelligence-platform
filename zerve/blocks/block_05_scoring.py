# =============================================================================
# ZERVE BLOCK 05 — FLOOD RISK SCORE ENGINE  (signature metric)
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : indicators, model_surface
# OUTPUTS (downstream) : scorer, frs_result, frs_surface,
#                        zone_table, hotspots, headline
# =============================================================================
# Produces the platform's headline metric — the Flood Risk Score (0-100) —
# blending the transparent weighted indicators with the ML surface, then rolls
# it up to per-zone scores, categories, affected-zone counts and hotspots.
from src import risk_scoring, utils

try:                          # inherited from block 03
    indicators
except NameError:
    from src.pipeline import build_state
    _s = build_state(train=True)
    indicators = _s["indicators"]
    model_surface = _s["model_surface"]

try:                          # inherited from block 04 (optional)
    model_surface
except NameError:
    model_surface = None

scorer = risk_scoring.build_scorer_from_artifacts(
    indicators, model_surface=model_surface, model_blend=0.35)

frs_result = scorer.compute(rainfall_mm=utils.RAINFALL_REFERENCE_MM)
frs_surface = frs_result.frs_surface
zone_table = frs_result.zone_table
hotspots = frs_result.hotspots
headline = frs_result.headline()

print("[05] ===== SIGNATURE METRIC =====")
for _k, _v in headline.items():
    print(f"[05]   {_k:>22}: {_v}")
