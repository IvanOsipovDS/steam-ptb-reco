import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.preprocessing import OneHotEncoder              
from sklearn.compose import ColumnTransformer               
from sklearn.pipeline import Pipeline                       
from dateutil import parser

from ..utils.io import ROOT, ensure_dir
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

def parse_owners(value) -> float:
    """
    Parse SteamSpy 'owners' into a single numeric proxy.
    Expected formats:
      - '100000..200000' (typical range) -> mid-point
      - '200000' (single number) -> itself
      - any string with digits (we'll extract digits)
    Returns np.nan if cannot parse.
    """
    if pd.isna(value):
        return np.nan

    s = str(value)
    # Split by '..' if present
    parts = s.split("..")
    nums = []
    for p in parts:
        digits = "".join(ch for ch in p if ch.isdigit())
        if digits:
            try:
                nums.append(int(digits))
            except Exception:
                pass

    if len(nums) == 2:
        return (nums[0] + nums[1]) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def to_year(s):
    """Parse year from messy strings like 'Nov 10, 2015', '10 Nov, 2015', '2020'."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s.isdigit():
        y = int(s)
        return y if 1970 <= y <= 2035 else np.nan

    # Try pandas
    y = pd.to_datetime(s, errors="coerce").year
    if not pd.isna(y):
        return int(y)

    # Try dateutil
    try:
        return parser.parse(s, fuzzy=True).year
    except Exception:
        return np.nan


def main():
    ensure_dir(PROC_DIR)

    raw_path = RAW_DIR / "steamspy_details.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"[build_features] Loaded raw: {raw_path}, shape={df.shape}, columns={list(df.columns)[:12]}...")

    # --- Basic sanity checks ---
    for required in ["appid", "name"]:
        if required not in df.columns:
            raise ValueError(f"Required column '{required}' not found in raw data.")

    # --- Owners proxy ---
    owners_col = "owners"
    if owners_col not in df.columns:
        print(f"[WARN] Column '{owners_col}' not found. Will attempt a fallback popularity proxy.")
        df["owners_mid"] = np.nan
    else:
        df["owners_mid"] = df[owners_col].apply(parse_owners)

    # Diagnostics for owners
    n_total = len(df)
    n_owners_ok = int(df["owners_mid"].notna().sum())
    print(f"[build_features] owners_mid non-null: {n_owners_ok}/{n_total}")

    # Fallback if owners_mid is entirely missing
    if n_owners_ok == 0:
        # Use a simple popularity proxy if available: positive + negative reviews
        pos = df["positive"] if "positive" in df.columns else 0
        neg = df["negative"] if "negative" in df.columns else 0
        if isinstance(pos, pd.Series) and isinstance(neg, pd.Series):
            df["owners_mid"] = (pos.fillna(0) + neg.fillna(0)).astype(float)
            n_owners_ok = int(df["owners_mid"].notna().sum())
            print(f"[build_features] Fallback owners_mid via reviews -> non-null: {n_owners_ok}/{n_total}")
        else:
            raise ValueError(
                "Could not parse 'owners' and no 'positive/negative' columns to fallback on. "
                "Check your raw file schema."
            )

    # --- Positive ratio ---
    if "positive" in df.columns and "negative" in df.columns:
        denom = (df["positive"].fillna(0) + df["negative"].fillna(0)).replace(0, np.nan)
        df["pos_ratio"] = df["positive"] / denom
    else:
        print("[WARN] 'positive'/'negative' not found. Using neutral pos_ratio=0.5")
        df["pos_ratio"] = 0.5

    # --- Release year enrichment (optional) ---
    rel_path = RAW_DIR / "steam_release_dates.csv"
    if rel_path.exists():
        rel = pd.read_csv(rel_path)
        if "appid" in rel.columns and "release_date_str" in rel.columns:
            df = df.merge(rel, on="appid", how="left")
            df["release_year"] = df["release_date_str"].apply(to_year)
        else:
            print(f"[WARN] {rel_path} has unexpected schema; skipping merge.")
            df["release_year"] = np.nan
    else:
        print(f"[build_features] No release dates file at {rel_path}. Setting release_year=NaN.")
        df["release_year"] = np.nan

    # --- Target: Low/Mid/High bins by owners_mid quantiles ---
    df_non_null = df[df["owners_mid"].notna()].copy()
    if df_non_null.empty:
        raise ValueError("All rows have NaN owners_mid even after fallback. Cannot build target tiers.")

    q1, q2 = df_non_null["owners_mid"].quantile([0.33, 0.66])

    def bin_target(x):
        if x <= q1:
            return "low"
        if x <= q2:
            return "mid"
        return "high"

    df["target_tier"] = df["owners_mid"].apply(bin_target)

    # --- Text field (robust to missing columns) ---
    genre = df["genre"] if "genre" in df.columns else ""
    tags = df["tags"] if "tags" in df.columns else ""
    name = df["name"] if "name" in df.columns else ""
    df["desc"] = genre.fillna("") + " " + tags.fillna("") + " " + name.fillna("")

    cols_keep = [
        "appid", "name",
        "owners_mid", "target_tier",
        "pos_ratio", "release_year",
        "developer", "publisher",  # if missing, pandas will fill with NaN
        "desc"
    ]
    # Some columns might be absent: reindex with fill
    out = df.reindex(columns=cols_keep)
    ensure_dir(PROC_DIR)
    out_path = PROC_DIR / "dataset.csv"
    out.to_csv(out_path, index=False)
    print(f"[build_features] Saved: {out_path}, shape={out.shape}")


if __name__ == "__main__":
    main()