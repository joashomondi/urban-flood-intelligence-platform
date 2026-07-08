# =============================================================================
# ZERVE BLOCK 04 — FLOOD SUSCEPTIBILITY MODEL
# -----------------------------------------------------------------------------
# INPUTS  (inherited) : feature_table, indicators
# OUTPUTS (downstream) : model, model_name, model_results,
#                        feature_columns, model_surface
# =============================================================================
# Trains LogisticRegression / RandomForest / XGBoost, keeps the best by
# ROC-AUC, and predicts a full flood-probability surface for the scorer.
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
