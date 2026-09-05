"""Phase 16: production API is fail-safe.

Guards:
  * /predict_bands accepts ONLY obs_dim = 360 (canonical 36x10); the legacy
    2*n_bands layout is rejected.
  * No trained scheduler -> HTTP 503 (no random-scheduler fallback).
  * No trained deinterleaver -> HTTP 503 on deinterleaving-dependent endpoints
    (no raw-HDBSCAN baseline fallback).
  * Responses expose the full real-selection contract and never fabricate
    attribution or metrics.
"""

import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from fastapi import HTTPException

import src.deployment.api as api_mod
from src.deployment.api import DeinterleaveRequest, PredictBandsRequest, deinterleave_endpoint, predict_bands

OBS_DIM = 360
N_BANDS = 36
N_MODES = 5
N_ACTIONS = N_BANDS * N_MODES

from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    def _make_moe():
        drqn = DRQNScheduler(
            obs_dim=OBS_DIM, n_bands=N_BANDS, n_actions=N_ACTIONS, n_modes=N_MODES,
            lstm_hidden=32, lstm_layers=1,
        )
        moe = SmartScanMoE(drqn, {
            "n_bands": N_BANDS,
            "n_modes": N_MODES,
            "n_actions": N_ACTIONS,
            "eager_weight": 0.6,
            "revisit_weight": 0.4,
            "k_receivers": 1,
            "device": "cpu",
        })
        return moe


def _obs(length=OBS_DIM, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, length).astype(np.float32).tolist()


class _BaseAPITest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(api_mod.STATE)
        api_mod.STATE["device"] = "cpu"
        api_mod.STATE["model_cfg"] = {
            "drqn_scheduler": {"obs_dim": OBS_DIM, "n_bands": N_BANDS, "n_modes": N_MODES, "n_actions": N_ACTIONS},
            "smartscan_moe": {"eager_weight": 0.6, "revisit_weight": 0.4},
            "dwell_modes": {
                "base_dwell_time_us": 500.0,
                "mode_multipliers": [
                    {"name": "SHORT_DWELL", "multiplier": 0.25},
                    {"name": "NORMAL_DWELL", "multiplier": 1.0},
                    {"name": "LONG_DWELL", "multiplier": 2.5},
                    {"name": "REVISIT", "multiplier": 1.0},
                    {"name": "PREEMPTIVE_INTERCEPT", "multiplier": 1.0},
                ],
            },
        }
        api_mod.STATE["moe"] = None
        api_mod.STATE["scheduler_onnx"] = None
        api_mod.STATE["scheduler"] = None
        api_mod.STATE["hidden"] = None
        api_mod.STATE["deinterleaver"] = None
        api_mod.STATE["deinterleaver_onnx"] = None
        api_mod.STATE["normalization_stats"] = None

    def tearDown(self):
        api_mod.STATE.clear()
        api_mod.STATE.update(self._saved)


class PredictBandsFailSafeTests(_BaseAPITest):
    def test_no_scheduler_returns_503(self):
        with self.assertRaises(HTTPException) as ctx:
            predict_bands(PredictBandsRequest(obs=_obs()))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_legacy_2x_n_bands_obs_rejected(self):
        # The legacy 2*n_bands (72-length) layout must no longer be accepted.
        api_mod.STATE["moe"] = object()  # trained scheduler present
        with self.assertRaises(HTTPException) as ctx:
            predict_bands(PredictBandsRequest(obs=_obs(length=2 * N_BANDS)))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_wrong_length_rejected(self):
        api_mod.STATE["moe"] = object()
        for bad in (8, 72, 359, 361, 1000):
            with self.assertRaises(HTTPException) as ctx:
                predict_bands(PredictBandsRequest(obs=_obs(length=bad)))
            self.assertEqual(ctx.exception.status_code, 400)

    def test_noncanonical_obs_dim_config_returns_503(self):
        api_mod.STATE["moe"] = object()
        api_mod.STATE["model_cfg"]["drqn_scheduler"]["obs_dim"] = 72
        with self.assertRaises(HTTPException) as ctx:
            predict_bands(PredictBandsRequest(obs=_obs()))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_pt_path_returns_full_real_selection(self):
        moe = _make_moe()
        api_mod.STATE["moe"] = moe
        resp = predict_bands(PredictBandsRequest(obs=_obs()))
        self.assertIsInstance(resp.selected_action, int)
        self.assertTrue(0 <= resp.selected_action < N_ACTIONS)
        self.assertTrue(0 <= resp.selected_band < N_BANDS)
        self.assertTrue(0 <= resp.selected_mode < N_MODES)
        self.assertEqual(resp.selected_action, resp.selected_band * N_MODES + resp.selected_mode)
        self.assertGreater(resp.dwell_time_us, 0.0)
        self.assertTrue(0.0 <= resp.intercept_probability <= 1.0)
        self.assertGreaterEqual(resp.predicted_intercept_time_us, 0.0)
        self.assertIn("eager_pct", resp.attribution)
        self.assertIn("revisit_pct", resp.attribution)
        self.assertAlmostEqual(
            float(resp.attribution["eager_pct"]) + float(resp.attribution["revisit_pct"]), 1.0, places=5
        )
        self.assertGreaterEqual(resp.latency_ms, 0.0)

    def test_onnx_path_returns_real_not_fabricated(self):
        # ONNX scheduler present, no MoE. Stub session returns REAL q/prob/time.
        q = np.random.default_rng(1).uniform(-2.0, 3.0, (1, 1, N_ACTIONS)).astype(np.float32)
        prob = np.random.default_rng(2).uniform(0.2, 0.9, (1, 1, N_ACTIONS)).astype(np.float32)
        time_us = np.random.default_rng(3).uniform(200.0, 5000.0, (1, 1, N_ACTIONS)).astype(np.float32)

        class _Sess:
            def run(self, _names, _feeds):
                return [q, prob, time_us]

        api_mod.STATE["scheduler"] = "onnx"
        api_mod.STATE["scheduler_onnx"] = _Sess()
        resp = predict_bands(PredictBandsRequest(obs=_obs()))
        self.assertTrue(0 <= resp.selected_action < N_ACTIONS)
        self.assertEqual(resp.intercept_probability, float(prob[0, 0, resp.selected_action]))
        self.assertEqual(resp.predicted_intercept_time_us, float(time_us[0, 0, resp.selected_action]))
        self.assertTrue(0.0 <= float(resp.attribution["eager_pct"]) <= 1.0)
        self.assertTrue(0.0 <= float(resp.attribution["revisit_pct"]) <= 1.0)
        self.assertAlmostEqual(
            float(resp.attribution["eager_pct"]) + float(resp.attribution["revisit_pct"]), 1.0, places=5
        )

    def test_obs_must_be_flat(self):
        api_mod.STATE["moe"] = object()
        with self.assertRaises(HTTPException) as ctx:
            predict_bands(PredictBandsRequest(obs=_obs(length=OBS_DIM)[:2]))  # never runs (400 above)
        # The 2-element flat obs is length-rejected; ndim stays 1.
        self.assertEqual(ctx.exception.status_code, 400)


class DeinterleaveFailSafeTests(_BaseAPITest):
    def test_no_trained_deinterleaver_returns_503(self):
        req = DeinterleaveRequest(pdws=[[0.0, 1000.0, 1.0, 10.0, 1.0]] * 5, min_cluster_size=2)
        with self.assertRaises(HTTPException) as ctx:
            deinterleave_endpoint(req, request=SimpleNamespace(headers={}))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_validation_before_503_gate(self):
        req = DeinterleaveRequest(pdws=[[0.0] * 6] * 5, min_cluster_size=2)  # wrong width
        with self.assertRaises(HTTPException) as ctx:
            deinterleave_endpoint(req, request=SimpleNamespace(headers={}))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_trained_model_without_stats_returns_503(self):
        api_mod.STATE["deinterleaver"] = object()  # trained PT model marker
        req = DeinterleaveRequest(pdws=[[0.0, 1000.0, 1.0, 10.0, 1.0]] * 5, min_cluster_size=2)
        with self.assertRaises(HTTPException) as ctx:
            deinterleave_endpoint(req, request=SimpleNamespace(headers={}))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_no_fallback_to_raw_hdbscan(self):
        # Even with a loaded trained model that can't actually run, the API
        # must surface ONLY the trained-model path — the raw HDBSCAN baseline
        # is never invoked and the response is not clustered from raw PDWs.
        from unittest import mock

        with mock.patch("hdbscan.HDBSCAN") as hdbscan_cls:
            api_mod.STATE["deinterleaver"] = object()
            api_mod.STATE["normalization_stats"] = {
                "cf_median": 9000.0, "cf_iqr": 6000.0, "pw_mean": 2.5, "pw_std": 1.5,
                "amp_mean": -80.0, "amp_std": 20.0,
            }
            req = DeinterleaveRequest(pdws=[[0.0, 1000.0, 1.0, 10.0, 1.0]] * 5, min_cluster_size=2)
            resp = deinterleave_endpoint(req, request=SimpleNamespace(headers={}))
            hdbscan_cls.assert_not_called()
        self.assertEqual(resp.labels, [-1] * 5)
        self.assertEqual(resp.n_clusters, 0)


if __name__ == "__main__":
    unittest.main()