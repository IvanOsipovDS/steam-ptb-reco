from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def read_json(path: Path):
    return json.loads(path.read_text())
