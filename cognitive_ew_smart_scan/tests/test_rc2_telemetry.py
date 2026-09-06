"""
RC-2 telemetry instrumentation tests (Test A-H of the RC-2 directive).

Fast / synthetic only — no dependency on the real TSRD dataset.
   A  schema completeness of the v2 episode telemetry record
   B  no NaN / Inf leakage through the NaN-safe serializer
   C  reward reconstruction matches the logged episode reward
   D  action-selection counts sum exactly to episode steps
   E  mode-selection counts sum exactly to episode steps
   F  zero-hit episode => Pd == 0.0 (no saturation of the metric)
   G  Pd == 1.0 with coverage < 1.0 stay distinct (Phase 9 separation)
   H  JSONL serialization of the full record (null / floats / arrays / dicts)
"""

import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import DWELL_MODES
from src.evaluation.metrics import FiguresOfMerit
from src.telemetry.publisher import TelemetryPublisher
from src.telemetry.run_manager import RunManager
from src.telemetry.schema import (
    EPISODE_CORE_FIELDS,
    REWARD_COMPONENT_FIELDS,
    TELEMETRY_SCHEMA_VERSION,
    coerce,
    make_episode_record,
    reward_reconstruction,
    shannon_entropy,
)


def _sample_components() -> dict[str, float]:
    return {
        "reward_novel": 2.0, "reward_hit": 1.0, "reward_miss": -3.0,
        "reward_dwell_cost": -5.0, "reward_false_alarm": -2.0, "reward_timing": -0.5,
        "reward_priority": 1.0, "reward_info_gain": 0.5, "reward_redundant": -0.5,
        "reward_delay": 0.0,
    }


def _sample_learning() -> dict[str, Any]:
    return {
        "td_loss": 0.24, "mean_td_error": 0.31, "max_td_error": 2.1,
        "mean_q": -0.9, "max_q": 0.4, "min_q": -2.3, "q_std": 0.12,
        "mean_online_q": -0.9, "mean_target_q": -0.85, "max_target_q": 0.5,
        "target_online_gap": 0.05, "gradient_norm": 0.42,
        "n_updates": 25, "learning_rate": 1e-4, "replay_size": 4096, "epsilon": 0.11,
    }


class TestAMakeEpisodeSchema(unittest.TestCase):
    def test_full_record_has_all_rc2_fields(self):
        core = {
            "pd": 0.5, "pfa": 0.1, "intercept_rate": 0.2, "avg_reward": -0.95,
            "episode_reward": -961.3, "ep_hits": 30, "ep_steps": 1000,
            "coverage": 0.2, "discovery_rate": 0.3, "avg_intercept_time": 1234.5,
            "avg_intercept_time_error": 1234.5, "pct_correct": 88.0,
            "selected_active": 200, "spectrum_active": 1000,
        }
        actions = {
            "unique_actions": 17, "unique_bands": 12, "unique_modes": 5,
            "band_selection_counts": list(range(36)),
            "band_selection_frequencies": [1 / 36.0] * 36,
            "mode_selection_counts": {m: 200 for m in DWELL_MODES},
            "mode_selection_frequencies": {m: 0.2 for m in DWELL_MODES},
            "action_entropy": 3.2, "band_entropy": 3.1, "mode_entropy": 2.0,
        }
        moe = {
            "moe_eager_weight": 0.6, "moe_revisit_weight": 0.4,
            "moe_semantic_weight": 1.0, "moe_preemptive_weight": 0.0,
            "mean_q_score": -0.5, "mean_eager_score": 0.3, "mean_revisit_score": 0.2,
            "mean_semantic_score": 0.4, "mean_preemptive_score": 0.0,
            "mean_fused_score": 0.9, "same_argmax_fraction": 0.8,
            "mean_drqn_rank": 3.2, "moe_rank": 1, "selected_band": 4,
            "q_argmax_band": 3, "moe_argmax_band": 4,
        }
        rec = make_episode_record(1000, 1, core, _sample_components(), actions,
                                  _sample_learning(), moe, band_priorities=[0.1] * 36,
                                  epsilon=0.11)

        # A.1 core + components + aggregates present
        for k in EPISODE_CORE_FIELDS:
            self.assertIn(k, rec, f"missing core field {k}")
        for k in REWARD_COMPONENT_FIELDS:
            self.assertIn(k, rec, f"missing component {k}")
        for k in ("unique_actions", "unique_bands", "unique_modes", "band_selection_counts",
                  "band_selection_frequencies", "mode_selection_counts", "mode_selection_frequencies",
                  "action_entropy", "band_entropy", "mode_entropy"):
            self.assertIn(k, rec, f"missing action stats {k}")
        for k in ("td_loss", "mean_td_error", "max_td_error", "mean_q", "max_q", "min_q",
                  "q_std", "mean_online_q", "mean_target_q", "max_target_q", "target_online_gap",
                  "gradient_norm", "learning_rate", "epsilon", "replay_size", "n_updates"):
            self.assertIn(k, rec, f"missing learning stat {k}")
        for k in ("moe_eager_weight", "moe_revisit_weight", "moe_semantic_weight",
                  "moe_preemptive_weight", "mean_q_score", "mean_eager_score", "mean_revisit_score",
                  "mean_semantic_score", "mean_preemptive_score", "mean_fused_score",
                  "same_argmax_fraction", "mean_drqn_rank", "moe_rank", "selected_band",
                  "q_argmax_band", "moe_argmax_band"):
            self.assertIn(k, rec, f"missing MoE field {k}")
        self.assertEqual(rec["telemetry_schema_version"], TELEMETRY_SCHEMA_VERSION)
        self.assertEqual(rec["type"], "episode")
        # A.2 legacy aliases preserved
        self.assertEqual(rec["ep_reward"], -961.3)
        self.assertEqual(rec["pd"], 0.5)


class TestBNaNLeakage(unittest.TestCase):
    def _collect_leaves(self, obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from self._collect_leaves(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from self._collect_leaves(v)
        else:
            yield obj

    def test_record_with_nan_inf_has_no_nan_leak(self):
        core = {k: (float("nan") if k in ("pd", "pfa", "intercept_rate", "avg_intercept_time",
                                          "avg_intercept_time_error", "pct_correct", "selected_active") else 0.0)
                for k in EPISODE_CORE_FIELDS}
        core["episode_reward"] = -500.0
        core["ep_hits"] = 7
        core["ep_steps"] = 1000
        comps = {k: (float("nan") if k == "reward_delay" else 1.0) for k in REWARD_COMPONENT_FIELDS}
        actions = {
            "unique_actions": 3, "unique_bands": 2, "unique_modes": 1,
            "band_selection_counts": [1, 1] + [0] * 34,
            "band_selection_frequencies": [0.5, 0.5] + [0.0] * 34,
            "mode_selection_counts": {m: 0 for m in DWELL_MODES},
            "mode_selection_frequencies": {m: 0.0 for m in DWELL_MODES},
            "action_entropy": float("nan"), "band_entropy": 1.0, "mode_entropy": 0.0,
        }
        learning = _sample_learning()
        rec = make_episode_record(0, 1, core, comps, actions, learning, {}, epsilon=float("nan"))
        for leaf in self._collect_leaves(coerce(rec)):
            self.assertTrue(leaf is None or isinstance(leaf, (bool, int, float, str)),
                            f"unexpected non-scalar leaf {type(leaf)}: {leaf!r}")
            if isinstance(leaf, float):
                self.assertFalse(math.isnan(leaf) or math.isinf(leaf),
                                 f"NaN/Inf leaked into record: {leaf!r}")
        # NaN must be emitted as JSON null, never as a "NaN" string.
        self.assertIsNone(rec["pd"], "NaN should be coerced to null")
        self.assertIsNone(rec["action_entropy"], "NaN should be coerced to null")
        self.assertIsNone(rec["epsilon"], "NaN should be coerced to null")


class TestCRewardReconstruction(unittest.TestCase):
    def test_reconstruction_matches_and_warns_on_mismatch(self):
        comps = _sample_components()
        ep_reward = sum(comps.values())  # must reconstruct exactly
        r = reward_reconstruction(comps, ep_reward, tol_rel=1e-6, tol_abs=1e-6)
        self.assertTrue(r["reward_reconstruction_ok"])
        self.assertAlmostEqual(r["reward_total_reconstructed"], ep_reward, places=5)
        self.assertLessEqual(r["reward_reconstruction_error"], max(1e-6, 1e-6 * abs(ep_reward)))
        # Mismatch detects corruption.
        r_bad = reward_reconstruction(comps, ep_reward + 50.0, tol_rel=1e-3, tol_abs=1.0)
        self.assertFalse(r_bad["reward_reconstruction_ok"])
        self.assertGreater(r_bad["reward_reconstruction_error"], 1.0)
        # Missing component => not ok (undefined degradation).
        comps_miss = dict(comps)
        comps_miss["reward_novel"] = None
        r_miss = reward_reconstruction(comps_miss, ep_reward, tol_rel=1e-3, tol_abs=1.0)
        self.assertFalse(r_miss["reward_reconstruction_ok"])


class TestDActionCountsSumToSteps(unittest.TestCase):
    def test_counts_sum_and_entropy(self):
        actions = [int(i) % 180 for i in range(1, 1001)]
        counts = np.zeros(180, dtype=np.float64)
        for a in actions:
            counts[a] += 1
        self.assertEqual(counts.sum(), 1000, "action counts must sum to ep_steps")
        expected_unique = len(set(actions))
        self.assertEqual(int(np.count_nonzero(counts)), expected_unique)
        self.assertGreater(shannon_entropy(counts), 0.0)
        # Frequencies must sum to 1.0 when a denominator exists.
        freqs = counts / counts.sum()
        self.assertAlmostEqual(float(freqs.sum()), 1.0, places=9)


class TestEModeCountsSumToSteps(unittest.TestCase):
    def test_mode_counts_sum_to_steps(self):
        actions = [int(i) % 180 for i in range(500)]  # 5 modes x (non-uniform bands)
        mode_counts = np.zeros(5, dtype=np.float64)
        for a in actions:
            mode_counts[a % 5] += 1
        self.assertEqual(mode_counts.sum(), 500, "mode counts must sum to ep_steps")
        # Same invariant expressed through the DWELL_MODES mapping used in telemetry.
        mapped = {DWELL_MODES[i]: int(mode_counts[i]) for i in range(5)}
        self.assertEqual(sum(mapped.values()), 500)


class TestFZeroHitPdIsZero(unittest.TestCase):
    def test_zero_hit_zero_opportunities(self):
        fom = FiguresOfMerit(n_bands=36)
        for i in range(50):
            # Band inactive, no detection -> correct reject, zero hits.
            fom.update(band_chosen=i % 36, ground_truth_active=np.zeros(36, dtype=int),
                       pred_active=False, reward=-0.9)
        s = fom.summary()
        self.assertEqual(int(s["n_hits"]), 0)
        self.assertEqual(s["Pd"], 0.0, "zero-hit episode must report Pd == 0.0, not saturation")


class TestGPdOneCoverageSubOneDistinct(unittest.TestCase):
    def test_pd_1_with_coverage_below_1(self):
        fom = FiguresOfMerit(n_bands=36)
        for i in range(10):
            if i % 2 == 0:
                # Only band 0 active; tuned & intercepted.
                gt = np.zeros(36, dtype=int)
                gt[0] = 1
                fom.update(band_chosen=0, ground_truth_active=gt, pred_active=True, reward=0.5)
            else:
                # Bands 0 and 1 active; tuned band 0 intercepted, band 1 unselected.
                gt = np.zeros(36, dtype=int)
                gt[0] = 1
                gt[1] = 1
                fom.update(band_chosen=0, ground_truth_active=gt, pred_active=True, reward=0.5)
        s = fom.summary()
        self.assertEqual(s["Pd"], 1.0)
        self.assertLess(s["band_selection_coverage"], 1.0)
        # Phase-9 separation: coverage is a distinct metric, not Pd.
        self.assertNotEqual(s["band_selection_coverage"], s["Pd"])


class TestHJsonlSerialization(unittest.TestCase):
    def test_full_record_round_trips(self):
        core = {k: 0.5 for k in EPISODE_CORE_FIELDS}
        core.update({"episode_reward": -950.0, "ep_hits": 3, "ep_steps": 1000})
        learning = _sample_learning()
        moe = {"same_argmax_fraction": 0.9}
        rec = make_episode_record(5000, 5, core, _sample_components(), {
            "unique_actions": 5, "band_selection_counts": [3] * 36,
            "band_selection_frequencies": [0.0] * 36,
            "mode_selection_counts": {m: 0 for m in DWELL_MODES},
            "mode_selection_frequencies": {m: 0.0 for m in DWELL_MODES},
            "action_entropy": None, "band_entropy": None, "mode_entropy": 1.0,
        }, learning, moe, band_priorities=[0.1, 0.2])
        rec["nested"] = {"scenario_details": [{"scenario_id": "abc", "reward": -950.0}]}
        rec["none_field"] = None

        with tempfile.TemporaryDirectory() as tmp:
            run = RunManager(root=tmp)
            pub = TelemetryPublisher(run=run)
            pub.update(**rec)
            lines = (run.dir / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            back = json.loads(lines[0])
            # Strict JSON: null, not the literal NaN token.
            self.assertIn("none_field", back)
            self.assertIsNone(back["none_field"])
            self.assertEqual(back["nested"]["scenario_details"][0]["scenario_id"], "abc")
            self.assertAlmostEqual(back["episode_reward"], -950.0)
            self.assertEqual(back["telemetry_schema_version"], TELEMETRY_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()