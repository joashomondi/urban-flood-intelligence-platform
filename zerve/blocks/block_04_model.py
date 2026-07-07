# =============================================================================
# ZERVE BLOCK 04 — FLOOD SUSCEPTIBILITY MODEL
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : feature_table, indicators
# OUTPUTS (downstream) : model, model_name, model_results,
#                        feature_columns, model_surface
# =============================================================================
# Trains LogisticRegression / RandomForest / XGBoost, keeps the best by
# ROC-AUC, and predicts a full flood-probability surface for the scorer.
from src import modeling

try:                          # inherited from block 03
    feature_table; indicators
except NameError:             # fallback: recompute upstream
    from src.pipeline import build_state
    _s = build_state(train=False)
    feature_table, indicators = _s["feature_table"], _s["indicators"]

_out = modeling.train_and_evaluate(feature_table)

model = _out["best_model"]
model_name = _out["best_model_name"]
model_results = _out["results"]
feature_columns = _out["feature_columns"]

# Persist so a FastAPI/download deployment can also load it via joblib.
modeling.save_model(model, model_name, feature_columns)

model_surface = modeling.predict_surface(model, indicators, feature_columns)

_auc = model_results[model_name]["metrics"]["roc_auc"]
print(f"[04] best model = {model_name}  ROC-AUC = {_auc:.3f}")
