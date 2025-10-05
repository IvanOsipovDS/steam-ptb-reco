# src/tools/export_catalog.py
"""
Export a lightweight catalog for the /recommend endpoint.
Reads data/processed/dataset.csv and writes data/catalog/catalog.csv
with only the columns we need in production.
"""

from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
IN_PATH = ROOT / "data" / "processed" / "dataset.csv"
OUT_DIR = ROOT / "data" / "catalog"
OUT_PATH = OUT_DIR / "catalog.csv"

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    cols = ["appid", "name", "developer", "publisher", "pos_ratio", "release_year", "desc"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in dataset: {missing}")

    out = df[cols].dropna(subset=["appid", "name"]).copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[export_catalog] wrote {OUT_PATH} with shape={out.shape}")

if __name__ == "__main__":
    main()
