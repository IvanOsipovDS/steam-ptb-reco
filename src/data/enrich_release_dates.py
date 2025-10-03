# src/data/enrich_release_dates.py
import time
import requests
import pandas as pd
from pathlib import Path
from ..utils.io import ROOT, ensure_dir

RAW_DIR = ROOT / "data" / "raw"
OUT_CSV = RAW_DIR / "steam_release_dates.csv"
CHECKPOINT = RAW_DIR / "steam_release_dates.partial.csv"

STORE_API = "https://store.steampowered.com/api/appdetails"

def fetch_release_date(appid: int, timeout=20):
    # oficial Store API
    params = {"appids": str(appid), "l": "en", "cc": "us"}
    r = requests.get(STORE_API, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if not js or str(appid) not in js:
        return None
    entry = js[str(appid)]
    if not entry.get("success"):
        return None
    data = entry.get("data", {})
    rel = data.get("release_date", {})
    return rel.get("date")

def main(limit=None, sleep=0.25):
    ensure_dir(RAW_DIR)
    base = pd.read_csv(RAW_DIR / "steamspy_details.csv")
    if limit:
        base = base.head(limit)

    # checkpoint
    done = set()
    rows = []
    if CHECKPOINT.exists():
        prev = pd.read_csv(CHECKPOINT)
        rows = prev.to_dict(orient="records")
        done = set(prev["appid"].tolist())

    for i, appid in enumerate(base["appid"]):
        appid = int(appid)
        if appid in done:
            continue
        try:
            date_str = fetch_release_date(appid)
            rows.append({"appid": appid, "release_date_str": date_str})
        except Exception:
            rows.append({"appid": appid, "release_date_str": None})
        # checkpoint each 100
        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(CHECKPOINT, index=False)
        time.sleep(sleep) 

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    # final snapshot instead partial
    if CHECKPOINT.exists():
        CHECKPOINT.unlink(missing_ok=True)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()