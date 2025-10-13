from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
from joblib import load
from pathlib import Path
import os
import time
import pandas as pd
import numpy as np
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

PRECOMPUTE_ERROR = None  # last error during _precompute_ptb_high

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
    Precompute P(high) for all items in DF using the loaded MODEL.
    Non-fatal: raise to caller; caller will put error into PRECOMPUTE_ERROR.
    """
    if MODEL is None or DF is None:
        raise RuntimeError("MODEL or DF is not available")

    feat_cols = ["developer", "publisher", "pos_ratio", "release_year", "desc"]
    X = DF[feat_cols]
    p_high = _proba_high(MODEL, X)

    if p_high.shape[0] != DF.shape[0]:
        raise ValueError(f"p_high shape mismatch: {p_high.shape} vs DF {DF.shape}")

    # строго 1D float ndarray
    globals()["PTB_HIGH"] = np.asarray(p_high, dtype=float).ravel()

def _proba_high(model, X):
    """
    Return probability for the 'high' tier class.
    Works whether classes_ are ['high','low','mid'] or [0,1,2].
    """
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim == 1:          # guard
        return proba.ravel()     # <— добавил ravel()

    high_idx = None
    classes = getattr(model, "classes_", None)
    if classes is not None:
        for i, c in enumerate(classes):
            if isinstance(c, str) and c.lower() == "high":
                high_idx = i
                break
        if high_idx is None:
            try:
                high_idx = int(np.argmax(classes))
            except Exception:
                high_idx = proba.shape[1] - 1
    else:
        high_idx = proba.shape[1] - 1

    return proba[:, high_idx].ravel()

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

# ---- load model at startup ----
try:
    _load_model()
except Exception:
    MODEL = None
    MODEL_PATH = None

# ---- load catalog (robust), but DO NOT clear it if precompute fails ----
CATALOG_ERROR = None
try:
    _load_catalog()  # только загрузка CSV + TF-IDF
except Exception as e:
    DF = None
    TFIDF = None
    DESC_MTX = None
    APPIDX = None
    CATALOG_ERROR = f"{type(e).__name__}: {e}"

# --- precompute PTB (non fatal) ---
PRECOMPUTE_ERROR = None
try:
    if DF is not None:
        _precompute_ptb_high()
except Exception as e:
    PTB_HIGH = None
    PRECOMPUTE_ERROR = f"{type(e).__name__}: {e}"

class PredictRequest(BaseModel):
    developer: Optional[str] = ""
    publisher: Optional[str] = ""
    pos_ratio: Optional[float] = 0.5
    release_year: Optional[int] = 2018
    desc: Optional[str] = ""


class PredictResponse(BaseModel):
    pred_class: str
    proba: Dict[str, float]

class RecoItem(BaseModel):
    appid: int
    name: str
    developer: Optional[str] = None
    publisher: Optional[str] = None
    release_year: Optional[int] = None
    similarity: float
    ptb_high: Optional[float] = None
    score: float

class RecoSource(BaseModel):
    appid: int
    name: str
    developer: Optional[str] = None
    publisher: Optional[str] = None
    release_year: Optional[int] = None

class RecoFilters(BaseModel):
    same_developer: bool
    year_from: Optional[int] = None
    min_score: Optional[float] = None

class RecoResponse(BaseModel):
    source: RecoSource
    k: int
    alpha: float
    use_ptb: bool
    filters: RecoFilters
    count: int
    items: List[RecoItem]

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
        "precompute_error": PRECOMPUTE_ERROR,
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
    global PRECOMPUTE_ERROR
    try:
        _load_model()
        PRECOMPUTE_ERROR = None
        if DF is not None:
            try:
                _precompute_ptb_high()
            except Exception as e:
                globals()["PTB_HIGH"] = None
                PRECOMPUTE_ERROR = f"{type(e).__name__}: {e}"
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

@app.get("/recommend", response_model=RecoResponse)
def recommend(
    appid: int = Query(..., description="Source appid"),
    k: int = Query(10, ge=1, le=50),
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="blend: similarity vs PTB(high)"),
    same_developer: bool = Query(False, description="exclude same developer as source"),
    year_from: int | None = Query(None, description="optional min release year"),
    use_ptb: bool = Query(True, description="use PTB(high) for re-ranking"),
    min_score: float | None = Query(None, description="threshold on blended score"),
    fields: Optional[List[str]] = Query(None, description="return only these item fields"),
    debug: bool = Query(False, description="include debug metadata"),
):
    if DF is None or DESC_MTX is None or APPIDX is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded")
    if appid not in APPIDX:
        raise HTTPException(status_code=404, detail=f"appid {appid} not found in catalog")

    src_idx = APPIDX[appid]
    sim = cosine_similarity(DESC_MTX[src_idx], DESC_MTX).ravel()
    sim[src_idx] = 0.0

    # filters
    mask = pd.Series(True, index=DF.index)
    if same_developer:
        src_dev = str(DF.iloc[src_idx]["developer"])
        mask &= DF["developer"].astype(str).ne(src_dev)
    if year_from is not None:
        mask &= DF["release_year"].fillna(0).astype(int).ge(int(year_from))

    # blended score
    if not use_ptb or PTB_HIGH is None:
        score = alpha * sim
        used_alpha = 1.0  # фактически чистый sim
    else:
        score = alpha * sim + (1.0 - alpha) * PTB_HIGH
        used_alpha = alpha

    # пороги + маска
    score_masked = score.copy()
    score_masked[~mask.values] = -1e9
    if min_score is not None:
        score_masked[score_masked < float(min_score)] = -1e9

    # top-k
    top_idx = score_masked.argsort()[::-1][:k]

    # сбор ответа
    def pick_fields(row_dict: dict) -> dict:
        if fields:
            return {k: row_dict.get(k) for k in fields if k in row_dict}
        return row_dict

    items = []
    for i in top_idx:
        r = DF.iloc[i]
        item = {
            "appid": int(r["appid"]),
            "name": str(r.get("name", "")),
            "developer": str(r.get("developer", "")),
            "publisher": str(r.get("publisher", "")),
            "release_year": int(r.get("release_year")) if pd.notna(r.get("release_year")) else None,
            "similarity": float(sim[i]),
            "ptb_high": float(PTB_HIGH[i]) if (use_ptb and PTB_HIGH is not None) else None,
            "score": float(score[i]),
        }
        items.append(pick_fields(item))

    src = DF.iloc[src_idx][["appid","name","developer","publisher","release_year"]].to_dict()
    src["appid"] = int(src["appid"])
    src["release_year"] = int(src["release_year"]) if pd.notna(src["release_year"]) else None

    payload = {
        "source": src,
        "k": k,
        "alpha": used_alpha,
        "use_ptb": use_ptb and (PTB_HIGH is not None),
        "filters": {"same_developer": same_developer, "year_from": year_from, "min_score": min_score},
        "count": len(items),
        "items": items
    }
    if debug:
        payload["debug"] = {
            "catalog_rows": int(DF.shape[0]),
            "ptb_available": PTB_HIGH is not None,
        }
    return payload
