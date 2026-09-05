"""Phase 14: normalisation consistency — train-only fit, persisted inference.

Guards:
  * /deinterleave must NEVER fit stats from request data when a trained model
    is serving; it must reuse checkpoints/deinterleaver/normalization_stats.json.
  * A normalization metadata hash makes artifacts agree on the exact stats.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.preprocessing.normalise import (
    load_normalization_stats,
    normalise_pdws,
    normalization_stats_hash,
    save_normalization_stats,
)

SAMPLE_STATS = {
    "cf_median": 9000.0,
    "cf_iqr": 6000.0,
    "pw_mean": 2.5,
    "pw_std": 1.5,
    "amp_mean": -80.0,
    "amp_std": 20.0,
    "fitted_sample_size": 12345,
}


class NormalizationStatsHashTests(unittest.TestCase):
    def test_hash_deterministic(self):
        self.assertEqual(normalization_stats_hash(SAMPLE_STATS), normalization_stats_hash(SAMPLE_STATS))

    def test_hash_order_independent(self):
        shuffled = dict(list(SAMPLE_STATS.items())[::-1])
        self.assertEqual(normalization_stats_hash(SAMPLE_STATS), normalization_stats_hash(shuffled))

    def test_hash_changes_with_values(self):
        changed = dict(SAMPLE_STATS)
        changed["cf_median"] = 1234.5
        self.assertNotEqual(normalization_stats_hash(SAMPLE_STATS), normalization_stats_hash(changed))

    def test_save_load_roundtrip_hash_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "normalization_stats.json"
            save_normalization_stats(SAMPLE_STATS, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("stats_hash", payload)
            self.assertIn("stats_version", payload)
            # Stored hash must agree with the canonical hash of the payload
            # (self-referential stats_hash key excluded).
            canonical = {k: v for k, v in payload.items() if k != "stats_hash"}
            self.assertEqual(payload["stats_hash"], normalization_stats_hash(canonical))
            loaded = load_normalization_stats(path)
            self.assertEqual(loaded["stats_hash"], payload["stats_hash"])

    def test_persisted_stats_used_verbatim(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "normalization_stats.json"
            save_normalization_stats(SAMPLE_STATS, path)
            loaded = load_normalization_stats(path)
            # Recompute normalisation with the loaded stats must use THEM, not
            # fresh per-call values: two different batches yield identical
            # normalised CF/PW/Amp values given the same stats.
            a = np.array([[0.0, 3000.0, 1.0, 10.0, -70.0]] * 8, dtype=np.float32)
            b = np.array([[0.0, 15000.0, 30.0, 200.0, -110.0]] * 8, dtype=np.float32)
            na, stats_used = normalise_pdws(a, loaded)
            nb, stats_used2 = normalise_pdws(b, loaded)
            # CF/PW/Amp columns identical under persisted stats (ToA always windowed).
            self.assertAlmostEqual(na[0, 1], (3000.0 - loaded["cf_median"]) / loaded["cf_iqr"], places=5)
            self.assertAlmostEqual(nb[0, 1], (15000.0 - loaded["cf_median"]) / loaded["cf_iqr"], places=5)
            self.assertEqual(stats_used["cf_median"], loaded["cf_median"])
            self.assertEqual(stats_used2["amp_std"], loaded["amp_std"])


class ApiInferenceNormalisationTests(unittest.TestCase):
    def setUp(self):
        from src.deployment import api as api_mod

        self.api_mod = api_mod
        self._saved_state = dict(api_mod.STATE)
        # Normalise away so endpoint-state mutations don't leak.
        api_mod.STATE["deinterleaver_onnx"] = None
        api_mod.STATE["deinterleaver"] = None
        api_mod.STATE["normalization_stats"] = None
        api_mod.STATE["normalization_stats_path"] = None

    def tearDown(self):
        self.api_mod.STATE.clear()
        self.api_mod.STATE.update(self._saved_state)

    def _pdws(self):
        return np.array([[0.0, 9000.0, 2.0, 45.0, -70.0]] * 4, dtype=np.float32)

    def test_no_model_fits_per_request(self):
        # Baseline-only mode (raw HDBSCAN): fitting from the request is allowed.
        out = self.api_mod._normalise_for_inference(self._pdws())
        self.assertEqual(out.shape, (4, 6))

    def test_trained_model_requires_persisted_stats(self):
        self.api_mod.STATE["deinterleaver"] = object()  # nn.Module sentinel
        self.api_mod.STATE["normalization_stats"] = None
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            self.api_mod._normalise_for_inference(self._pdws())
        self.assertEqual(ctx.exception.status_code, 503)

    def test_trained_model_uses_persisted_stats_not_none(self):
        self.api_mod.STATE["deinterleaver"] = object()
        self.api_mod.STATE["normalization_stats"] = SAMPLE_STATS
        seen = {}

        def _fake_normalise(pdws, fit_stats):
            seen["fit_stats"] = fit_stats
            return np.zeros((pdws.shape[0], 6), dtype=np.float32), {"fit_stats": fit_stats}

        with mock.patch("src.preprocessing.normalise.normalise_pdws", side_effect=_fake_normalise):
            out = self.api_mod._normalise_for_inference(self._pdws())
        self.assertEqual(out.shape, (4, 6))
        self.assertEqual(seen["fit_stats"], SAMPLE_STATS)

    def test_onnx_model_also_uses_persisted_stats(self):
        self.api_mod.STATE["deinterleaver"] = "onnx"
        self.api_mod.STATE["deinterleaver_onnx"] = object()  # InferenceSession sentinel
        self.api_mod.STATE["normalization_stats"] = SAMPLE_STATS
        seen = {}

        def _fake_normalise(pdws, fit_stats):
            seen["fit_stats"] = fit_stats
            return np.zeros((pdws.shape[0], 6), dtype=np.float32), {"fit_stats": fit_stats}

        with mock.patch("src.preprocessing.normalise.normalise_pdws", side_effect=_fake_normalise):
            self.api_mod._normalise_for_inference(self._pdws())
        self.assertEqual(seen["fit_stats"], SAMPLE_STATS)


class ExportNormalizationMetaTests(unittest.TestCase):
    def test_resolve_prefers_checkpoint_stamped_hash(self):
        from src.deployment.export_onnx import _resolve_normalization_meta
        hash_str, path = _resolve_normalization_meta(
            {"normalization_stats_hash": "abc123", "normalization_stats_path": "/x/normalization_stats.json"},
            "checkpoints/deinterleaver/best.pt",
        )
        self.assertEqual((hash_str, path), ("abc123", "/x/normalization_stats.json"))

    def test_resolve_computes_from_sidecar_stats(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            stats_path = ckpt.parent / "normalization_stats.json"
            save_normalization_stats(SAMPLE_STATS, stats_path)
            from src.deployment.export_onnx import _resolve_normalization_meta
            hash_str, path = _resolve_normalization_meta({}, ckpt, stats_candidates=[stats_path])
            # Resolver hashes what it LOADED (the stamped file), which must also
            # equal the stored hash and stay stable vs the pre-stamp payload.
            self.assertEqual(hash_str, normalization_stats_hash(SAMPLE_STATS))
            self.assertEqual(hash_str, load_normalization_stats(stats_path)["stats_hash"])
            self.assertEqual(Path(path).name, "normalization_stats.json")

    def test_resolve_unknown_when_no_stats(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            ckpt.touch()
            from src.deployment.export_onnx import _resolve_normalization_meta
            hash_str, path = _resolve_normalization_meta({}, ckpt, stats_candidates=[])
            self.assertEqual((hash_str, path), ("unknown", None))


if __name__ == "__main__":
    unittest.main()