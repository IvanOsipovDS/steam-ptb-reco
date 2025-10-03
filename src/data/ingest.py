import requests, time
import pandas as pd
from pathlib import Path
from ..utils.io import ROOT, ensure_dir

STEAMSPY_BASE = "https://steamspy.com/api.php"
RAW_DIR = ROOT / "data" / "raw"

def fetch_toplist(n=5000):
    # top sales of all time (can be changed to 'top100in2weeks'/'top100owned')
    r = requests.get(STEAMSPY_BASE, params={"request":"top100owned"})
    r.raise_for_status()
    items = list(r.json().values())
    df = pd.DataFrame(items)
    # we'll get the rest (steamspy doesn't have paged; we'll expand it using all + filter)
    r2 = requests.get(STEAMSPY_BASE, params={"request":"all"})
    r2.raise_for_status()
    all_items = pd.DataFrame(list(r2.json().values()))
    all_items = all_items.sort_values("owners", ascending=False).head(n)
    df = pd.concat([df, all_items], ignore_index=True).drop_duplicates("appid")
    return df

def fetch_details(appids):
    rows = []
    for appid in appids:
        time.sleep(0.2)  # no spam
        r = requests.get(STEAMSPY_BASE, params={"request":"appdetails","appid":appid})
        if r.status_code != 200:
            continue
        rows.append(r.json())
    return pd.DataFrame(rows)

def main():
    ensure_dir(RAW_DIR)
    toplist = fetch_toplist(n=5000)
    toplist.to_csv(RAW_DIR / "steamspy_top.csv", index=False)
    details = fetch_details(toplist["appid"].tolist())
    details.to_csv(RAW_DIR / "steamspy_details.csv", index=False)
    print("Saved:", RAW_DIR)

if __name__ == "__main__":
    main()
