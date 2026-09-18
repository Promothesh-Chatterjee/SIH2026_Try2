"""Tests for Phase 4 API endpoints: /api/v1/metrics and /api/v1/spectrum."""

import unittest
import numpy as np
from fastapi.testclient import TestClient

from ew_core.deployment.api import (
    app,
    update_latest_evaluation_data,
    _latest_episode_metrics,
    _latest_spectrum_data,
)
from ew_core.metrics.ew_metrics import EWMetrics


class TestPhase4APIEndpoints(unittest.TestCase):
    """Test suite for /api/v1/metrics and /api/v1/spectrum."""

    def setUp(self):
        self.client = TestClient(app)
        # Reset state before each test
        import ew_core.deployment.api as api_mod
        api_mod._latest_episode_metrics = None
        api_mod._latest_spectrum_data = None

    def tearDown(self):
        import ew_core.deployment.api as api_mod
        api_mod._latest_episode_metrics = None
        api_mod._latest_spectrum_data = None

    def test_metrics_endpoint_uninitialized(self):
        """Uninitialized endpoint must return meaningful defaults (not 500 or null JSON)."""
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("status"), "uninitialized")
        self.assertEqual(data.get("sensitivity_dbm"), -140.0)
        self.assertEqual(data.get("pd"), 0.0)
        self.assertEqual(data.get("pfa"), 0.0)
        self.assertEqual(data.get("avg_intercept_rate"), 0.0)
        self.assertEqual(data.get("n_receiver_dwells"), 0)

    def test_metrics_endpoint_populated(self):
        """Populated endpoint returns exact serialized EWMetrics fields."""
        metrics = EWMetrics(
            pd=0.92,
            pfa=0.01,
            sensitivity_dbm=-140.0,
            avg_intercept_rate=0.85,
            avg_reward=0.78,
            pct_correct_predictions=91.4,
            avg_intercept_time_error_us=1.5,
            n_intercepts=450,
            n_false_alarms=5,
            n_total_transmissions=500,
            n_receiver_dwells=500,
            n_missed_dwells=39,
        )
        update_latest_evaluation_data(metrics, {})

        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertAlmostEqual(data["pd"], 0.92)
        self.assertAlmostEqual(data["pfa"], 0.01)
        self.assertEqual(data["sensitivity_dbm"], -140.0)
        self.assertAlmostEqual(data["avg_intercept_rate"], 0.85)
        self.assertEqual(data["n_intercepts"], 450)
        self.assertEqual(data["n_receiver_dwells"], 500)

    def test_spectrum_endpoint_uninitialized(self):
        """Uninitialized spectrum endpoint returns default lists."""
        response = self.client.get("/api/v1/spectrum")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("status"), "uninitialized")
        self.assertEqual(data.get("truth_matrix"), [])
        self.assertEqual(data.get("receiver_positions"), [])
        self.assertEqual(data.get("emitter_ids"), [])

    def test_spectrum_endpoint_populated(self):
        """Populated spectrum endpoint serializes numpy matrices into JSON-compliant lists."""
        n_bands = 36
        t_steps = 100
        truth = np.zeros((n_bands, t_steps), dtype=bool)
        truth[5, 10:20] = True
        receiver_pos = np.arange(t_steps) % n_bands
        emitter_ids = [[1] if 10 <= t < 20 else [] for t in range(t_steps)]

        update_latest_evaluation_data(
            metrics={},
            spectrum_data={
                "truth_matrix": truth,
                "receiver_positions": receiver_pos,
                "emitter_ids": emitter_ids,
            },
        )

        response = self.client.get("/api/v1/spectrum")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ready")
        self.assertEqual(len(data["truth_matrix"]), n_bands)
        self.assertEqual(len(data["truth_matrix"][0]), t_steps)
        self.assertEqual(len(data["receiver_positions"]), t_steps)
        self.assertEqual(len(data["emitter_ids"]), t_steps)
        self.assertTrue(data["truth_matrix"][5][15])


def test_conftest_synthetic_pdw_stream_fixture(synthetic_pdw_stream):
    """Verify synthetic_pdw_stream fixture in conftest.py generates valid PulseDescriptorWord instances."""
    assert len(synthetic_pdw_stream) == 100
    pdw = synthetic_pdw_stream[0]
    assert hasattr(pdw, "toa_us")
    assert hasattr(pdw, "frequency_mhz")
    assert hasattr(pdw, "pulse_width_us")
    assert hasattr(pdw, "amplitude_db")
    assert pdw.toa_us >= 0.0


def test_conftest_small_env_fixture(small_env):
    """Verify small_env fixture in conftest.py returns a valid SpectrumEnvironment(8, 50)."""
    assert small_env.n_bands == 8
    assert small_env.t_steps == 50
    obs, info = small_env.reset()
    assert obs is not None
    assert isinstance(info, dict)


if __name__ == "__main__":
    unittest.main()

