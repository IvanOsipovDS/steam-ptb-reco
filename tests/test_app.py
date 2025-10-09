import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import unittest
from fastapi.testclient import TestClient
from src.serving.app import app

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        js = r.json()
        self.assertIs(True, js["model_loaded"])
        self.assertIs(True, js["catalog_loaded"])
        self.assertGreater(js["catalog_rows"], 0)

    def test_predict_ptb_minimal(self):
        payload = {
            "developer": "Valve",
            "publisher": "Valve",
            "pos_ratio": 0.8,
            "release_year": 2015,
            "desc": "multiplayer fps shooter"
        }
        r = self.client.post("/predict_ptb", json=payload)
        self.assertEqual(r.status_code, 200)
        js = r.json()
        self.assertIn("pred_class", js)
        self.assertIn("proba", js)

    def test_recommend_required(self):
        appid = 1623730  # Palworld (из каталога)
        r = self.client.get(f"/recommend?appid={appid}&k=5&alpha=0.7")
        self.assertEqual(r.status_code, 200)
        js = r.json()
        self.assertLessEqual(js["k"], 5)
        self.assertIsInstance(js["items"], list)

if __name__ == "__main__":
    unittest.main()