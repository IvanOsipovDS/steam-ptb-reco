import os, json, requests
BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
print("Using base:", BASE)

def ok(r):
    r.raise_for_status()
    print(r.status_code, r.url)
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))

# 1) health
r = requests.get(f"{BASE}/health", timeout=30); ok(r)

# 2) predict_ptb
payload = {
    "developer": "Valve",
    "publisher": "Valve",
    "pos_ratio": 0.8,
    "release_year": 2015,
    "desc": "multiplayer fps shooter",
}
r = requests.post(f"{BASE}/predict_ptb", json=payload, timeout=30); ok(r)

# 3) recommend
r = requests.get(f"{BASE}/recommend",
                 params={"appid": 1623730, "k": 5, "alpha": 0.7},
                 timeout=30)
ok(r)
