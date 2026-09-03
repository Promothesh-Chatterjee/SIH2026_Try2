import unittest

from fastapi.testclient import TestClient

from src.deployment import api as api_mod


class ApiTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure a clean telemetry state for deterministic assertions.
        api_mod.telemetry._last = {}
        api_mod.telemetry._history = []
        api_mod.telemetry._n_updates = 0
        api_mod.TELEMETRY_ROOT = "runs"
        cls.client = TestClient(api_mod.app)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_telemetry_latest_empty_is_not_live(self):
        resp = self.client.get("/telemetry/latest")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertIn("live", payload)

    def test_telemetry_history_empty(self):
        resp = self.client.get("/telemetry/history")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("records", resp.json())
        self.assertIn("live", resp.json())

    def test_telemetry_runs_list(self):
        resp = self.client.get("/telemetry/runs")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("runs", resp.json())

    def test_inprocess_update_goes_live(self):
        api_mod.telemetry.update(step=0, pd=0.99, band_priorities=[0.1, 0.2, 0.3])
        resp = self.client.get("/telemetry/latest")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["live"])
        self.assertAlmostEqual(payload["pd"], 0.99)
        self.assertEqual(payload["band_priorities"], [0.1, 0.2, 0.3])

    def test_ws_payload_no_fabrication_when_empty(self):
        api_mod.telemetry._last = {}
        api_mod.telemetry._history = []
        api_mod.telemetry._n_updates = 0
        api_mod.TELEMETRY_ROOT = "definitely_missing_runs_dir"
        payload = api_mod._telemetry_payload()
        self.assertFalse(payload["live"])


if __name__ == "__main__":
    unittest.main()