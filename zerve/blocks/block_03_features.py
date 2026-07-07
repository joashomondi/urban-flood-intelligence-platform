# =============================================================================
# ZERVE BLOCK 03 — FEATURE ENGINEERING
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : terrain, hydro, rainfall_field
# OUTPUTS (downstream) : indicators, feature_table, labels
# =============================================================================
# Builds the six normalised flood indicators and physically-motivated flood
# labels, then flattens the raster stack into a tidy per-cell training table.
from src import feature_engineering

try:                          # inherited from blocks 01 + 02
    terrain; hydro; rainfall_field
except NameError:             # fallback: recompute the upstream chain
    from src.pipeline import build_state
    _s = build_state(train=False)
    terrain, hydro, rainfall_field = _s["terrain"], _s["hydro"], _s["rainfall_field"]

_feats = feature_engineering.engineer_features(terrain, hydro, rainfall_field)

indicators = _feats["indicators"]
feature_table = _feats["table"]
labels = _feats["labels"]

print(f"[03] indicators={list(indicators)}")
print(f"[03] feature_table {feature_table.shape}  |  "
      f"positive rate {100 * feature_table['flood_label'].mean():.1f}%")
