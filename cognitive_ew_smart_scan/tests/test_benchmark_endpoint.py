"""Test dynamic benchmark API endpoints."""

import unittest
from fastapi.testclient import TestClient
from src.deployment.api import app


class DynamicBenchmarkApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_benchmark_scenarios(self):
        res = self.client.get("/benchmark/scenarios")
        self.assertEqual(res.status_code, 200)
        scenarios = res.json()
        self.assertIsInstance(scenarios, list)
        self.assertGreater(len(scenarios), 0)
        ids = [s["id"] for s in scenarios]
        self.assertIn("AG-04", ids)
        self.assertIn("AG-01", ids)

    def test_post_benchmark_evaluate(self):
        payload = {
            "scenario": "AG-04",
            "n_steps": 25,
            "snr_db": 15.0,
            "seed": 42,
        }
        res = self.client.post("/benchmark/evaluate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["scenario"], "AG-04")
        self.assertEqual(len(data["columns"]), 7)
        self.assertEqual(len(data["rows"]), 5)
        # Check all 5 required rows exist
        dim_names = [r[0] for r in data["rows"]]
        self.assertIn("Intercept Rate", dim_names)
        self.assertIn("Mean Detect Latency", dim_names)
        self.assertIn("False-Alarm Rate", dim_names)
        self.assertIn("Revisit Compliance", dim_names)
        self.assertIn("Agile Track Continuity", dim_names)

    def test_get_benchmark_latest(self):
        res = self.client.get("/benchmark/latest")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("rows", data)
        self.assertIn("columns", data)
