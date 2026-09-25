"""Phase 4 Qualification Test Suite: Reward v2 & Learning-Signal Correction.

Formal Verifications:
Task 4.1: Hard Dominance Invariant (interception > shaping across all dwell durations & ToA offsets).
Task 4.2: Time-Normalized Reward Analysis & Mission-Time Telemetry (RewardTracker & FiguresOfMerit).
Task 4.3: Deterministic Canonical Reward Table (7 exact benchmark cases).
Task 4.4: Zero Proxy Contamination (action score, future data, threat class, ID permutation, heuristic ranking).
Task 4.5: Frozen Checkpoint SHA-256 Invariance.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_BAND_FEATURES, CANONICAL_OBS_DIM
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.radio_environment import PulseRecord
from ew_core.evaluation.metrics import FiguresOfMerit
from ew_core.training.diagnostics.reward_tracker import RewardTracker
from ew_core.training.reward import (
    receiver_reward_components_v2,
    validate_reward_v2_dominance,
)

FROZEN_CHECKPOINT_PATH = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
EXPECTED_CHECKPOINT_SHA256 = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"


def _make_observation(dwell_us: float = 500.0, detections=None):
    if detections is None:
        detections = []
    return SimpleNamespace(
        dwell_time_us=dwell_us,
        dwell_interval_us=[0.0, dwell_us],
        detections=detections,
        center_frequency_mhz=2500.0,
    )


class TestTask41RewardDominance:
    """Task 4.1: Interception must remain strictly dominant over secondary shaping."""

    def test_dominance_invariant_across_dwell_durations_and_toa(self):
        """Proof: Shaping can NEVER overpower repeat (+8.0) or novel (+10.0) interception."""
        dwell_durations = [125.0, 500.0, 1250.0]
        # Fine grid of arrival fractions within dwell
        hit_fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        for dwell_us in dwell_durations:
            for frac in hit_fractions:
                t_hit = frac * dwell_us
                obs = _make_observation(dwell_us, [SimpleNamespace(time_us=t_hit)])

                # Maximum possible shaping: agile=True, predicted=True, instant arrival
                res_repeat = receiver_reward_components_v2(
                    observation=obs,
                    ground_truth_active=True,
                    novel_emitter=False,
                    detected=True,
                    intercept_time_us=t_hit,
                    is_agile=True,
                    is_predicted=True,
                    strict_dominance=True,
                )

                interception = res_repeat["interception_reward"]  # +8.0
                shaping = (
                    abs(res_repeat["latency_reward"])
                    + abs(res_repeat["agility_bonus"])
                    + abs(res_repeat["prediction_bonus"])
                    + abs(res_repeat["redundant_penalty"])
                    + abs(res_repeat["dwell_cost"])
                )

                assert interception == 8.0
                assert shaping <= 7.5, f"Shaping ({shaping}) exceeded 7.5 for dwell {dwell_us}, t_hit {t_hit}"
                assert shaping < interception, f"Dominance violated: shaping {shaping} >= {interception}"
                assert res_repeat["dominance_warning"] is False

    def test_dominance_validation_rejects_invalid_configurations(self):
        """Proof: validate_reward_v2_dominance raises ValueError on non-dominant weights."""
        # Valid baseline config
        assert validate_reward_v2_dominance(
            w_hit_repeat=8.0,
            w_latency=5.0,
            w_agile_bonus=2.0,
            w_prediction=0.5,
        ) is True

        # Invalid config: latency (6.0) + agile (2.5) = 8.5 >= 8.0
        with pytest.raises(ValueError, match="Reward v2 dominance violated"):
            validate_reward_v2_dominance(
                w_hit_repeat=8.0,
                w_latency=6.0,
                w_agile_bonus=2.5,
                w_prediction=0.5,
            )

    def test_strict_dominance_mode_raises_on_violation(self):
        """Proof: receiver_reward_components_v2 raises ValueError under strict_dominance."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=0.0)])
        # Artificially force shaping to exceed interception
        with pytest.raises(ValueError, match="REWARD_OBJECTIVE_DOMINANCE_VIOLATION"):
            receiver_reward_components_v2(
                observation=obs,
                ground_truth_active=True,
                detected=True,
                w_hit_repeat=5.0,
                w_latency=5.0,
                w_agile_bonus=2.0,
                is_agile=True,
                strict_dominance=True,
            )


class TestTask42TimeNormalizedReward:
    """Task 4.2: Time-normalized reward analysis & diagnostic telemetry."""

    def test_controlled_reward_scaling_across_dwell_durations(self):
        """Proof: For identical normalized hit timing, SHORT dwell yields higher reward per ms than LONG dwell."""
        # Same normalized arrival fraction: t_hit / dwell = 0.1
        res_short = receiver_reward_components_v2(
            observation=_make_observation(125.0, [SimpleNamespace(time_us=12.5)]),
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=12.5,
        )
        res_norm = receiver_reward_components_v2(
            observation=_make_observation(500.0, [SimpleNamespace(time_us=50.0)]),
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=50.0,
        )
        res_long = receiver_reward_components_v2(
            observation=_make_observation(1250.0, [SimpleNamespace(time_us=125.0)]),
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=125.0,
        )

        # With mode-independent reference dwell (500us), earlier physical arrivals yield higher latency rewards:
        assert res_short["reward"] == pytest.approx(12.875, abs=1e-4)
        assert res_norm["reward"] == pytest.approx(12.5, abs=1e-4)
        assert res_long["reward"] == pytest.approx(11.75, abs=1e-4)

        # Reward per millisecond scales with arrival speed:
        # SHORT (0.125 ms): 12.875 / 0.125 = 103.0 / ms
        # NORMAL (0.500 ms): 12.5 / 0.500 = 25.0 / ms
        # LONG (1.250 ms): 11.75 / 1.250 = 9.4 / ms
        assert res_short["reward_per_ms"] == pytest.approx(103.0, abs=1e-2)
        assert res_norm["reward_per_ms"] == pytest.approx(25.0, abs=1e-2)
        assert res_long["reward_per_ms"] == pytest.approx(9.4, abs=1e-2)

        # SHORT > NORMAL > LONG in reward per unit mission time:
        assert res_short["reward_per_ms"] > res_norm["reward_per_ms"] > res_long["reward_per_ms"]

    def test_reward_tracker_mission_time_telemetry(self):
        """Proof: RewardTracker accurately accumulates mission time and computes per-ms telemetry."""
        rt = RewardTracker(gamma=0.99)

        # Step 1: Hit on SHORT dwell (125 us)
        info1 = {
            "mission_dwell_us": 125.0,
            "hit": True,
            "reward_components": {
                "interception_reward": 8.0,
                "latency_reward": 4.5,
                "agility_bonus": 0.0,
                "prediction_bonus": 0.0,
                "miss_penalty": 0.0,
                "false_alarm_penalty": 0.0,
                "redundant_penalty": 0.0,
                "dwell_cost": 0.0,
            }
        }
        rt.step(12.5, info1)

        # Step 2: False alarm on LONG dwell (1250 us)
        info2 = {
            "mission_dwell_us": 1250.0,
            "hit": False,
            "reward_components": {
                "interception_reward": 0.0,
                "latency_reward": 0.0,
                "agility_bonus": 0.0,
                "prediction_bonus": 0.0,
                "miss_penalty": 0.0,
                "false_alarm_penalty": -1.0,
                "redundant_penalty": 0.0,
                "dwell_cost": -0.025,
            }
        }
        rt.step(-1.025, info2)

        summary = rt.get_episode_summary()
        expected_time_ms = (125.0 + 1250.0) / 1000.0  # 1.375 ms
        assert summary["total_mission_ms"] == pytest.approx(expected_time_ms, abs=1e-5)
        assert summary["total_reward"] == pytest.approx(12.5 - 1.025, abs=1e-5)
        assert summary["reward_per_dwell"] == pytest.approx((12.5 - 1.025) / 2.0, abs=1e-5)
        assert summary["reward_per_ms"] == pytest.approx((12.5 - 1.025) / expected_time_ms, abs=1e-5)
        assert summary["hit_reward_per_ms"] == pytest.approx(12.5 / expected_time_ms, abs=1e-5)
        assert summary["penalty_per_ms"] == pytest.approx(-1.025 / expected_time_ms, abs=1e-5)

        # Direct reconstruction check: reward_per_ms == hit_reward_per_ms + penalty_per_ms
        assert summary["reward_per_ms"] == pytest.approx(
            summary["hit_reward_per_ms"] + summary["penalty_per_ms"],
            abs=1e-6,
        )

    def test_figures_of_merit_time_aware_telemetry(self):
        """Proof: FiguresOfMerit integrates dwell time and exposes per-ms properties."""
        fom = FiguresOfMerit(n_bands=36)
        comp1 = {
            "dwell_time_us": 500.0,
            "hit_term": 8.0,
            "latency_bonus": 4.0,
            "agility_bonus": 2.0,
            "prediction_bonus": 0.5,
            "miss_penalty": 0.0,
            "false_alarm_penalty": 0.0,
            "redundant_penalty": 0.0,
            "dwell_cost": 0.0,
        }
        fom.record_reward_components(comp1)
        fom.update(band_chosen=2, ground_truth_active=True, pred_active=True, reward=14.5)

        summ = fom.summary()
        assert summ["total_mission_time_ms"] == pytest.approx(0.5, abs=1e-5)
        assert summ["reward_per_dwell"] == pytest.approx(14.5, abs=1e-5)
        assert summ["reward_per_ms"] == pytest.approx(14.5 / 0.5, abs=1e-5)
        assert summ["hit_reward_per_ms"] == pytest.approx(14.5 / 0.5, abs=1e-5)
        assert summ["penalty_per_ms"] == pytest.approx(0.0, abs=1e-5)


class TestTask43CanonicalRewardTable:
    """Task 4.3: Automated regressions for the 7 canonical deterministic reward cases."""

    def test_case_1_fast_novel_intercept(self):
        """Case 1: Novel emitter intercepted at t=10us in NORMAL (500us) dwell -> +14.90."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=10.0)])
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=True,
            detected=True,
            intercept_time_us=10.0,
        )
        # 10.0 + 5.0 * (1 - 10/500) = 10.0 + 4.90 = 14.90
        assert res["reward"] == pytest.approx(14.90, abs=1e-4)
        assert res["interception_reward"] == 10.0
        assert res["latency_reward"] == pytest.approx(4.90, abs=1e-4)

    def test_case_2_fast_agile_intercept(self):
        """Case 2: Repeat agile emitter intercepted at t=20us in NORMAL dwell -> +14.80."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=20.0)])
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=20.0,
            is_agile=True,
        )
        # 8.0 + 5.0 * (1 - 20/500) + 2.0 = 8.0 + 4.80 + 2.0 = 14.80
        assert res["reward"] == pytest.approx(14.80, abs=1e-4)
        assert res["interception_reward"] == 8.0
        assert res["latency_reward"] == pytest.approx(4.80, abs=1e-4)
        assert res["agility_bonus"] == 2.0

    def test_case_3_late_intercept(self):
        """Case 3: Repeat known emitter intercepted at t=450us in NORMAL dwell -> +8.50."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=450.0)])
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=450.0,
        )
        # 8.0 + 5.0 * (1 - 450/500) = 8.0 + 0.50 = 8.50
        assert res["reward"] == pytest.approx(8.50, abs=1e-4)
        assert res["interception_reward"] == 8.0
        assert res["latency_reward"] == pytest.approx(0.50, abs=1e-4)

    def test_case_4_empty_dwell(self):
        """Case 4: Inactive band, age > 1 in NORMAL dwell -> -1.01."""
        obs = _make_observation(500.0, [])
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=False,
            detected=False,
            band_age=10.0,
        )
        # -1.0 (false alarm) - 0.01 (dwell cost) = -1.01
        assert res["reward"] == pytest.approx(-1.01, abs=1e-4)
        assert res["false_alarm_penalty"] == -1.0
        assert res["dwell_cost"] == -0.01

    def test_case_5_active_miss(self):
        """Case 5: Active emitter present but receiver failed to detect in NORMAL dwell -> -4.01."""
        obs = _make_observation(500.0, [])
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            detected=False,
        )
        # -4.0 (miss penalty) - 0.01 (dwell cost) = -4.01
        assert res["reward"] == pytest.approx(-4.01, abs=1e-4)
        assert res["miss_penalty"] == -4.0
        assert res["dwell_cost"] == -0.01

    def test_case_6_redundant_dwell(self):
        """Case 6: Inactive band re-scanned with age <= 1 in NORMAL dwell -> -1.26."""
        obs = _make_observation(500.0, [])
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=False,
            detected=False,
            band_age=0.5,
        )
        # -1.0 (false alarm) - 0.25 (redundant) - 0.01 (dwell cost) = -1.26
        assert res["reward"] == pytest.approx(-1.26, abs=1e-4)
        assert res["false_alarm_penalty"] == -1.0
        assert res["redundant_penalty"] == -0.25
        assert res["dwell_cost"] == -0.01

    def test_case_7_periodic_prediction_hit(self):
        """Case 7: Repeat emitter intercepted at t=0us on predicted periodic arrival -> +13.50."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=0.0)])
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=0.0,
            is_predicted=True,
        )
        # 8.0 (repeat) + 5.0 (latency) + 0.5 (prediction) = 13.50
        assert res["reward"] == pytest.approx(13.50, abs=1e-4)
        assert res["interception_reward"] == 8.0
        assert res["latency_reward"] == 5.0
        assert res["prediction_bonus"] == 0.5


class TestTask44ZeroProxyContamination:
    """Task 4.4: Hard regressions verifying zero proxy reward contamination."""

    def test_action_score_mutation_invariance(self):
        """Proof: Changing hand-coded action_score produces bitwise identical reward."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=50.0)])
        res_baseline = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=True,
            detected=True,
            intercept_time_us=50.0,
        )
        res_mutated = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=True,
            detected=True,
            intercept_time_us=50.0,
            action_score=99.9,  # Hand-coded proxy score
        )
        assert res_baseline == res_mutated

    def test_future_pulse_arrival_isolation(self):
        """Proof: Pulses arriving strictly after dwell_end do not affect current dwell reward."""
        # Dwell window [0.0, 500.0]
        rec_in_window = PulseRecord(toa_us=250.0, frequency_mhz=2500.0, pulse_width_us=1.0, amplitude_db=-30.0, aoa_deg=45.0, emitter_id=1)
        rec_future = PulseRecord(toa_us=650.0, frequency_mhz=2500.0, pulse_width_us=1.0, amplitude_db=-30.0, aoa_deg=45.0, emitter_id=1)

        env1 = CognitiveRFScanEnv(
            {"dwell_time_us": 500.0, "reward": {"version": "v2"}},
            records=[rec_in_window],
            seed=42,
        )
        env1.reset()
        _, r1, _, _, _ = env1.step(action=5)

        env2 = CognitiveRFScanEnv(
            {"dwell_time_us": 500.0, "reward": {"version": "v2"}},
            records=[rec_in_window, rec_future],
            seed=42,
        )
        env2.reset()
        _, r2, _, _, _ = env2.step(action=5)

        assert r1 == pytest.approx(r2, abs=1e-7)

    def test_hidden_threat_class_invariance(self):
        """Proof: Modifying hidden threat class has zero effect on training reward."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=30.0, threat_class="LETHAL_SAM")])
        res1 = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=30.0,
        )
        obs_mutated = _make_observation(500.0, [SimpleNamespace(time_us=30.0, threat_class="BENIGN_WEATHER")])
        res2 = receiver_reward_components_v2(
            observation=obs_mutated,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=30.0,
        )
        assert res1 == res2

    def test_emitter_id_permutation_invariance(self):
        """Proof: Permuting ground truth emitter IDs (preserving novel status) yields identical reward."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=40.0, emitter_id=42)])
        res_id42 = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=True,
            detected=True,
            intercept_time_us=40.0,
        )
        obs_permuted = _make_observation(500.0, [SimpleNamespace(time_us=40.0, emitter_id=9999)])
        res_id9999 = receiver_reward_components_v2(
            observation=obs_permuted,
            ground_truth_active=True,
            novel_emitter=True,
            detected=True,
            intercept_time_us=40.0,
        )
        assert res_id42 == res_id9999

    def test_external_heuristic_ranking_invariance(self):
        """Proof: External priority / heuristic rankings do not leak into pure v2 reward."""
        obs = _make_observation(500.0, [SimpleNamespace(time_us=50.0)])
        res_baseline = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=50.0,
        )
        res_heuristic = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=50.0,
            heuristic_rank=1,
            thompson_score=0.99,
        )
        assert res_baseline == res_heuristic


class TestTask45FrozenBaselineCheckpoint:
    """Task 4.5: Ensure production baseline checkpoint is strictly immutable."""

    def test_production_baseline_checkpoint_sha256_unmodified(self):
        """Proof: Production baseline checkpoint matches SHA-256 bit-for-bit."""
        assert FROZEN_CHECKPOINT_PATH.exists(), f"Frozen checkpoint missing at {FROZEN_CHECKPOINT_PATH}"
        hasher = hashlib.sha256()
        with open(FROZEN_CHECKPOINT_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        observed_hash = hasher.hexdigest()
        assert observed_hash == EXPECTED_CHECKPOINT_SHA256, (
            f"Production baseline hash altered!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nObserved: {observed_hash}"
        )
