import os, requests, json, sys

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
print("Using base:", BASE)

def ok(resp):
    resp.raise_for_status()
    print(resp.status_code, resp.url)
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))

try:
    r = requests.get(f"{BASE}/health", timeout=10); ok(r)
except requests.exceptions.ConnectionError:
    print("❌ Can't connect. Start API locally (`uvicorn src.serving.app:app --reload`) "
          "or set API_BASE to your Render URL.")
    sys.exit(1)
