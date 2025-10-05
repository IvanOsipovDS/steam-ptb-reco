from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
from joblib import load
from pathlib import Path
import os
import time
import pandas as pd
from fastapi.responses import RedirectResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.io import ROOT, read_json

app = FastAPI(title="Steam PTB & Reco API", version="0.1.0")

REG_PATH = ROOT / "models" / "registry.json"
CATALOG_PATH = Path(__file__).resolve().parent / "catalog.csv"
MODEL = None
MODEL_PATH = None

# ---- catalog for recommendations ----
DF = None                 # catalog dataframe
TFIDF = None              # fitted vectorizer
DESC_MTX = None           # TF-IDF matrix (sparse)
APPIDX = None             # appid -> row index
PTB_HIGH = None           # vector of P(high) for each row

# --- add near other globals ---
CATALOG_ERROR = None  # last catalog load error (string)


def _load_catalog():
    """
    Load catalog CSV and build TF-IDF. Be robust to missing columns.
    """
    global DF, TFIDF, DESC_MTX, APPIDX, CATALOG_ERROR
    CATALOG_ERROR = None
    try:
        if not CATALOG_PATH.exists():
            raise FileNotFoundError(f"Catalog not found: {CATALOG_PATH}")

        # robust read
        df = pd.read_csv(CATALOG_PATH, low_memory=False)

        # normalize column names (strip / lower)
        df.columns = [c.strip() for c in df.columns]

        # required base columns (we can synthesize desc if missing)
        base_needed = {"appid", "name", "developer", "publisher", "pos_ratio", "release_year"}
        missing_base = [c for c in base_needed if c not in df.columns]
        for c in missing_base:
            # fill sensible defaults if absent
            if c == "pos_ratio":
                df[c] = 0.5
            elif c == "release_year":
                df[c] = 2018
            elif c in {"developer", "publisher"}:
                df[c] = ""
            elif c == "name":
                df[c] = ""
            elif c == "appid":
                raise RuntimeError("Column 'appid' is required in catalog.csv")

        # ensure integer appid/year where possible
        df["appid"] = pd.to_numeric(df["appid"], errors="coerce").astype("Int64")
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2018).astype(int)
        df["pos_ratio"] = pd.to_numeric(df["pos_ratio"], errors="coerce").fillna(0.5)

        # build/sanitize desc
        if "desc" not in df.columns:
            # synthesize from text-y fields
            df["desc"] = (
                df["name"].fillna("").astype(str) + " "
                + df["developer"].fillna("").astype(str) + " "
                + df["publisher"].fillna("").astype(str)
            )
        else:
            df["desc"] = df["desc"].fillna("").astype(str)

        # drop rows without appid/name
        df = df.dropna(subset=["appid", "name"]).copy()
        df["appid"] = df["appid"].astype(int)

        # finally assign globals
        DF = df.reset_index(drop=True)
        TFIDF = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        DESC_MTX = TFIDF.fit_transform(DF["desc"])
        APPIDX = {int(a): i for i, a in enumerate(DF["appid"])}

    except Exception as e:
        # expose error in health
        DF = None
        TFIDF = None
        DESC_MTX = None
        APPIDX = None
        CATALOG_ERROR = f"{type(e).__name__}: {e}"

def _precompute_ptb_high():
    """
    Precompute P(high) for the catalog using the trained MODEL (one-time).
    """
    global PTB_HIGH
    if MODEL is None:
        PTB_HIGH = None
        return
    X = pd.DataFrame({
        "developer": DF["developer"].fillna(""),
        "publisher": DF["publisher"].fillna(""),
        "pos_ratio": DF["pos_ratio"].fillna(0.5),
        "release_year": DF["release_year"].fillna(2018).astype(int),
        "desc": DF["desc"].fillna(""),
    })
    proba = MODEL.predict_proba(X)
    class_names = getattr(MODEL, "class_names_", None) or getattr(MODEL, "classes_", None)
    class_names = list(class_names) if class_names is not None else list(range(proba.shape[1]))
    try:
        idx_high = class_names.index("high")
    except Exception:
        # fallback: if label names unknown — use argmax per row is meaningless here; pick last col
        idx_high = min(proba.shape[1]-1, 0)
    PTB_HIGH = pd.Series(proba[:, idx_high], index=DF.index)

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


# Load model & catalog at startup
try:
    _load_model()
except Exception as e:
    # Delay actual raising to endpoints; allow /health to report error
    MODEL = None
    MODEL_PATH = None

# Load catalog + build TF-IDF + precompute PTB(high)
try:
    _load_catalog()
    _precompute_ptb_high()
except Exception:
    # if anything fails, expose via /health and keep API alive
    DF = None
    TFIDF = None
    DESC_MTX = None
    APPIDX = None
    PTB_HIGH = None

class PredictRequest(BaseModel):
    developer: Optional[str] = ""
    publisher: Optional[str] = ""
    pos_ratio: Optional[float] = 0.5
    release_year: Optional[int] = 2018
    desc: Optional[str] = ""


class PredictResponse(BaseModel):
    pred_class: str
    proba: Dict[str, float]

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH else None,
        "catalog_loaded": DF is not None,
        "catalog_path": str(CATALOG_PATH),
        "catalog_exists": CATALOG_PATH.exists(),
        "catalog_rows": int(DF.shape[0]) if DF is not None else 0,
        "catalog_error": CATALOG_ERROR,
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
        # after model reload, recompute PTB(high) over the catalog if it is loaded
        if DF is not None:
            _precompute_ptb_high()
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

@app.get("/recommend")
def recommend(
    appid: int = Query(..., description="Source appid"),
    k: int = Query(10, ge=1, le=50),
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="blend: similarity vs PTB(high)"),
    same_developer: bool = Query(False, description="exclude same developer as source"),
    year_from: int | None = Query(None, description="optional min release year"),
):
    """
    Content-based recommendations with PTB re-ranking.
    score = alpha * cosine(desc, desc_i) + (1 - alpha) * P(high)_i
    """
    if DF is None or DESC_MTX is None or APPIDX is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded")

    if appid not in APPIDX:
        raise HTTPException(status_code=404, detail=f"appid {appid} not found in catalog")

    src_idx = APPIDX[appid]
    sim = cosine_similarity(DESC_MTX[src_idx], DESC_MTX).ravel()
    sim[src_idx] = 0.0  # exclude itself

    # filters
    mask = pd.Series(True, index=DF.index)
    if same_developer:
        src_dev = str(DF.iloc[src_idx]["developer"])
        mask &= DF["developer"].astype(str).ne(src_dev)
    if year_from is not None:
        mask &= DF["release_year"].fillna(0).astype(int).ge(int(year_from))

    # blended score
    if PTB_HIGH is None:
        score = alpha * sim
    else:
        score = alpha * sim + (1.0 - alpha) * PTB_HIGH.values

    score_masked = score.copy()
    score_masked[~mask.values] = -1e9

    top_idx = score_masked.argsort()[::-1][:k]
    items = []
    for i in top_idx:
        r = DF.iloc[i]
        items.append({
            "appid": int(r["appid"]),
            "name": str(r.get("name","")),
            "developer": str(r.get("developer","")),
            "publisher": str(r.get("publisher","")),
            "release_year": int(r.get("release_year")) if pd.notna(r.get("release_year")) else None,
            "similarity": float(sim[i]),
            "ptb_high": float(PTB_HIGH.iloc[i]) if PTB_HIGH is not None else None,
            "score": float(score[i]),
        })

    src = DF.iloc[src_idx][["appid","name","developer","publisher","release_year"]].to_dict()
    src["appid"] = int(src["appid"])
    src["release_year"] = int(src["release_year"]) if pd.notna(src["release_year"]) else None

    return {"source": src, "k": k, "alpha": alpha,
            "filters": {"same_developer": same_developer, "year_from": year_from},
            "items": items}
