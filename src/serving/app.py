from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from joblib import load
from pathlib import Path
import os
import time
import pandas as pd

from ..utils.io import ROOT, read_json

app = FastAPI(title="Steam PTB & Reco API", version="0.1.0")

REG_PATH = ROOT / "models" / "registry.json"

MODEL = None
MODEL_PATH = None


def _load_model():
    """
    Load the best model from registry.json. 
    Stores the fitted pipeline object globally in MODEL.
    """
    global MODEL, MODEL_PATH
    if not REG_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REG_PATH}")

    reg = read_json(REG_PATH)
    path_str = reg.get("best_model_path")
    if not path_str:
        raise RuntimeError("best_model_path not found in registry.json")

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    if not path.is_absolute():
        path = ROOT / path
    MODEL = load(path)
    MODEL_PATH = path
    return MODEL


# Load model at startup
try:
    _load_model()
except Exception as e:
    # Delay actual raising to endpoints; allow /health to report error
    MODEL = None
    MODEL_PATH = None


class PredictRequest(BaseModel):
    developer: Optional[str] = ""
    publisher: Optional[str] = ""
    pos_ratio: Optional[float] = 0.5
    release_year: Optional[int] = 2018
    desc: Optional[str] = ""


class PredictResponse(BaseModel):
    pred_class: str
    proba: Dict[str, float]


@app.get("/health")
def health():
    """
    Health probe with basic model info if loaded.
    """
    ok = MODEL is not None
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "model_path": str(MODEL_PATH) if MODEL_PATH else None,
    }


@app.get("/version")
def version():
    """
    Return minimal model metadata (filename and modified time).
    """
    if MODEL_PATH and MODEL_PATH.exists():
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(MODEL_PATH.stat().st_mtime))
        return {
            "artifact": MODEL_PATH.name,
            "modified_at": mtime,
        }
    return {"artifact": None, "modified_at": None}


@app.post("/reload")
def reload_model():
    """
    Reload model from registry.json (use after retraining).
    """
    try:
        _load_model()
        return {"status": "reloaded", "model_path": str(MODEL_PATH)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.post("/predict_ptb", response_model=PredictResponse)
def predict_ptb(req: PredictRequest):
    """
    Predict PTB class and probabilities for a single item.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Ensure feature order matches training
    X = pd.DataFrame([{
        "developer": req.developer or "",
        "publisher": req.publisher or "",
        "pos_ratio": req.pos_ratio if req.pos_ratio is not None else 0.5,
        "release_year": req.release_year if req.release_year is not None else 2018,
        "desc": req.desc or ""
    }])

    # Predict probabilities (OvR order)
    try:
        proba = MODEL.predict_proba(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_proba failed: {e}")

    # Prefer human-readable label names from training
    class_names = getattr(MODEL, "class_names_", None)
    if class_names is None:
        class_names = getattr(MODEL, "classes_", None)
    if class_names is None:
        # Fallback to indices
        class_names = list(range(len(proba)))
    else:
        class_names = list(class_names)

    pred_idx = int(proba.argmax())
    pred_label = str(class_names[pred_idx]) if pred_idx < len(class_names) else str(pred_idx)

    return PredictResponse(
        pred_class=pred_label,
        proba={str(c): float(p) for c, p in zip(class_names, proba)}
    )
