# Steam PTB & Reco — Content-Based Game Recommendations with PTB Re-Ranking

A production-ready FastAPI service that:

1. **Predicts PTB** (“potential to be high”) for a game using text + metadata.  
2. **Returns content-based recommendations** (TF-IDF cosine similarity), **re-ranked** by the predicted probability of *high* PTB.

**Live API (Swagger):** `https://steam-ptb-reco.onrender.com`  
**Tech stack:** Python, scikit-learn, XGBoost, FastAPI, Uvicorn, Render

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start (Local)](#quick-start-local)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Smoke Tests](#smoke-tests)
- [Deployment (Render)](#deployment-render)
- [Testing & CI](#testing--ci)
- [Metrics](#metrics)
- [Limitations & Next Steps](#limitations--next-steps)
- [Data Sources & License](#data-sources--license)

---

## Overview

This project demonstrates an end-to-end ML workflow:

- **Ingestion & enrichment** of Steam metadata (based on SteamSpy).  
- **Feature engineering** (TF-IDF for descriptions, numeric/categorical features).  
- **Target definition**: PTB tiers (*low/mid/high*) using owners as a proxy; probability of *high* is used for re-ranking.  
- **Training**: multiple models with metrics and artifact registry.  
- **Serving**: a FastAPI app that exposes:
  - `/predict_ptb` — predict class + probabilities,
  - `/recommend` — content-based recommendations with PTB re-ranking,
  - `/health`, `/version`.

---

## Repository Structure

```
data/
  raw/               # source CSVs (ingested)
  processed/         # cleaned/engineered dataset (dataset.csv)
models/
  artifacts/         # trained artifacts (e.g., best_xgb.joblib)
  registry.json      # pointer to best model path
notebooks/           # (optional) analysis notebooks
src/
  data/
    ingest.py                 # fetch raw data
    enrich_release_dates.py   # enrich with release dates (web)
  features/
    build_features.py         # create dataset.csv
  models/
    train.py                  # train, evaluate, save best model
  serving/
    app.py          # FastAPI app (endpoints live here)
    catalog.csv     # lightweight catalog used by /recommend
  utils/
    io.py           # paths, I/O helpers
tests/
  test_app.py       # minimal API tests (unittest)
tools/
  smoke.py          # quick smoke test for /health, /predict_ptb, /recommend
Dockerfile
requirements.txt
README.md
```

---

## Quick Start (Local)

**Requirements:** Python 3.12+

```bash
git clone <your-repo-url>
cd steam-ptb-reco

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Run the API:

```bash
uvicorn src.serving.app:app --reload
# Swagger: http://127.0.0.1:8000/docs
```

> The repo already ships a small `src/serving/catalog.csv` and a trained artifact (`models/artifacts/...`).  
> You can re-build data and re-train any time (see next sections).

---

## Data Pipeline

Recreate raw → processed → dataset (optional):

```bash
# 1) Download raw data (SteamSpy top & details)
python -m src.data.ingest

# 2) Enrich release dates (web scrape)
python -m src.data.enrich_release_dates

# 3) Build features and dataset.csv
python -m src.features.build_features
```

Artifacts:

- `data/raw/` — initial CSV dumps,  
- `data/processed/dataset.csv` — final supervised dataset for training.

---

## Model Training

Train, evaluate and store the best model:

```bash
python -m src.models.train
```

Outputs:

- Best artifact saved to `models/artifacts/best_*.joblib`,  
- Registry updated at `models/registry.json` with the `best_model_path`.

---

## API Endpoints

### `GET /health`

Returns model/catalog status.

```bash
curl -s https://steam-ptb-reco.onrender.com/health | jq
```

Example:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/models/artifacts/best_xgb.joblib",
  "catalog_loaded": true,
  "catalog_path": "/app/src/serving/catalog.csv",
  "catalog_exists": true,
  "catalog_rows": 1000,
  "catalog_error": null,
  "precompute_error": null
}
```

### `POST /predict_ptb`

Predict PTB class and probabilities for a single game.

```bash
curl -s -X POST https://steam-ptb-reco.onrender.com/predict_ptb   -H "Content-Type: application/json"   -d '{
        "developer": "Valve",
        "publisher": "Valve",
        "pos_ratio": 0.8,
        "release_year": 2015,
        "desc": "FPS action shooter with multiplayer"
      }' | jq
```

Response (example):
```json
{
  "pred_class": "high",
  "proba": {
    "high": 0.59,
    "low": 0.03,
    "mid": 0.37
  }
}
```

### `GET /recommend`

Content-based recommendations with PTB re-ranking.  
Blended score:

```
score = alpha * cosine(desc_i, desc_src) + (1 - alpha) * P(high)_i
```

**Query params:**

- `appid` *(int, required)* — source game id from the catalog,  
- `k` *(int, default=10, [1..50])* — how many items to return,  
- `alpha` *(float, default=0.7, [0..1])* — balance between cosine and PTB,  
- `same_developer` *(bool, default=false)* — exclude same developer as source,  
- `year_from` *(int, optional)* — filter by min release year,  
- `min_score` *(float, optional)* — threshold on blended score.

```bash
curl -s "https://steam-ptb-reco.onrender.com/recommend?appid=1623730&k=5&alpha=0.7" | jq
```

Fragment:
```json
{
  "source": {
    "appid": 1623730,
    "name": "Palworld",
    "developer": "Pocketpair",
    "publisher": "Pocketpair",
    "release_year": 2024
  },
  "k": 5,
  "alpha": 0.7,
  "use_ptb": true,
  "filters": {
    "same_developer": false,
    "year_from": null,
    "min_score": null
  },
  "count": 5,
  "items": [
    {
      "appid": 1203620,
      "name": "Enshrouded",
      "developer": "Keen Games GmbH",
      "publisher": "Keen Games GmbH",
      "release_year": 2024,
      "similarity": 0.3179,
      "ptb_high": 0.9149,
      "score": 0.4970
    }
  ]
}
```

---

## Smoke Tests

Local smoke check (defaults to `http://127.0.0.1:8000`):

```bash
python tools/smoke.py
```

Against production:

```powershell
# PowerShell
$env:API_BASE = "https://steam-ptb-reco.onrender.com"
python tools/smoke.py
```

---

## Deployment (Render)

**Service type:** Web Service (Python)  
**Start command:**
```
uvicorn src.serving.app:app --host 0.0.0.0 --port $PORT
```

Notes:
- Auto-Deploy: **On Commit**.  
- Make sure `src/serving/catalog.csv` is present in the repo (it is).  
- Health checks: `GET /health`.

---

## Testing & CI

Run unit tests locally:

```bash
python -m unittest tests/test_app.py
```

Minimal CI (GitHub Actions) example is included in `.github/workflows/ci.yml`:
- Installs dependencies,
- Runs the API unit tests.

---

## Metrics

Validation on ~1000 samples (SteamSpy slice):

- **XGBoost**: **AUC ≈ 0.716**, **F1 ≈ 0.498** (best)
- **Logistic Regression**: AUC ≈ 0.663, F1 ≈ 0.464

Train again:

```bash
python -m src.models.train
```

The script logs the scores, saves the best artifact and updates `models/registry.json`.

---

## Limitations & Next Steps

**Limitations**
- PTB is a proxy target derived from SteamSpy owners, not a business KPI.
- Catalog and training slice are limited; real-world bias may exist.
- Text features are TF-IDF only (no deep embeddings).

**Next Steps**
- Expand text features (genres/tags), experiment with BM25 or sentence embeddings.
- Add Approximate Nearest Neighbors (FAISS/ScaNN) for scalable retrieval.
- AB-tune `alpha` and filtering strategies.
- Add model card & more thorough monitoring.

---

## Data Sources & License

- **Data:** SteamSpy aggregate datasets (used for demo/educational purposes).
- **License:** MIT — see `LICENSE`.

---

**Contact / Feedback**

Issues and PRs are welcome!
