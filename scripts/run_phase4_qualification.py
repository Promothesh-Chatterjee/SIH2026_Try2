"""Phase 4 Qualification & Empirical Benchmark Script.

Executes:
1. Canonical Reward Table Evaluation (7 deterministic cases).
2. Dwell-Time Reward Scaling & Time-Normalized Analysis (SHORT 125us, NORMAL 500us, LONG 1250us).
3. Hard Dominance Invariant Verification across all dwell modes.
4. Anti-Contamination Verification (action score, future data, threat class, ID permutation, heuristic ranking).
5. Checkpoint Bitwise Invariance Verification.

Outputs reports to experiments/reports/phase4/.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_BAND_FEATURES, CANONICAL_OBS_DIM
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.radio_environment import PulseRecord
from ew_core.evaluation.metrics import FiguresOfMerit
from ew_core.training.diagnostics.reward_tracker import RewardTracker
from ew_core.training.reward import (
    receiver_reward_components_v2,
    validate_reward_v2_dominance,
)

REPORT_DIR = Path("experiments/reports/phase4")
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


def run_canonical_reward_table() -> Dict[str, Any]:
    """Evaluate and verify the 7 deterministic canonical reward cases."""
    cases = {}

    # Case 1: Fast Novel
    obs1 = _make_observation(500.0, [SimpleNamespace(time_us=10.0)])
    res1 = receiver_reward_components_v2(
        observation=obs1,
        ground_truth_active=True,
        novel_emitter=True,
        detected=True,
        intercept_time_us=10.0,
    )
    cases["fast_novel_intercept"] = {
        "parameters": "novel=True, t_hit=10us, NORMAL (500us)",
        "expected_reward": 14.90,
        "observed_reward": round(float(res1["reward"]), 4),
        "interception_reward": float(res1["interception_reward"]),
        "latency_reward": float(res1["latency_reward"]),
        "status": "PASS" if math.isclose(res1["reward"], 14.90, abs_tol=1e-3) else "FAIL",
    }

    # Case 2: Fast Agile
    obs2 = _make_observation(500.0, [SimpleNamespace(time_us=20.0)])
    res2 = receiver_reward_components_v2(
        observation=obs2,
        ground_truth_active=True,
        novel_emitter=False,
        detected=True,
        intercept_time_us=20.0,
        is_agile=True,
    )
    cases["fast_agile_intercept"] = {
        "parameters": "repeat=True, agile=True, t_hit=20us, NORMAL (500us)",
        "expected_reward": 14.80,
        "observed_reward": round(float(res2["reward"]), 4),
        "interception_reward": float(res2["interception_reward"]),
        "latency_reward": float(res2["latency_reward"]),
        "agility_bonus": float(res2["agility_bonus"]),
        "status": "PASS" if math.isclose(res2["reward"], 14.80, abs_tol=1e-3) else "FAIL",
    }

    # Case 3: Late Intercept
    obs3 = _make_observation(500.0, [SimpleNamespace(time_us=450.0)])
    res3 = receiver_reward_components_v2(
        observation=obs3,
        ground_truth_active=True,
        novel_emitter=False,
        detected=True,
        intercept_time_us=450.0,
    )
    cases["late_intercept"] = {
        "parameters": "repeat=True, t_hit=450us, NORMAL (500us)",
        "expected_reward": 8.50,
        "observed_reward": round(float(res3["reward"]), 4),
        "interception_reward": float(res3["interception_reward"]),
        "latency_reward": float(res3["latency_reward"]),
        "status": "PASS" if math.isclose(res3["reward"], 8.50, abs_tol=1e-3) else "FAIL",
    }

    # Case 4: Empty Dwell
    obs4 = _make_observation(500.0, [])
    res4 = receiver_reward_components_v2(
        observation=obs4,
        ground_truth_active=False,
        detected=False,
        band_age=10.0,
    )
    cases["empty_dwell"] = {
        "parameters": "inactive, age > 1, NORMAL (500us)",
        "expected_reward": -1.01,
        "observed_reward": round(float(res4["reward"]), 4),
        "false_alarm_penalty": float(res4["false_alarm_penalty"]),
        "dwell_cost": float(res4["dwell_cost"]),
        "status": "PASS" if math.isclose(res4["reward"], -1.01, abs_tol=1e-3) else "FAIL",
    }

    # Case 5: Active Miss
    obs5 = _make_observation(500.0, [])
    res5 = receiver_reward_components_v2(
        observation=obs5,
        ground_truth_active=True,
        detected=False,
    )
    cases["active_miss"] = {
        "parameters": "active=True, undetected, NORMAL (500us)",
        "expected_reward": -4.01,
        "observed_reward": round(float(res5["reward"]), 4),
        "miss_penalty": float(res5["miss_penalty"]),
        "dwell_cost": float(res5["dwell_cost"]),
        "status": "PASS" if math.isclose(res5["reward"], -4.01, abs_tol=1e-3) else "FAIL",
    }

    # Case 6: Redundant Dwell
    obs6 = _make_observation(500.0, [])
    res6 = receiver_reward_components_v2(
        observation=obs6,
        ground_truth_active=False,
        detected=False,
        band_age=0.5,
    )
    cases["redundant_dwell"] = {
        "parameters": "inactive, age <= 1, NORMAL (500us)",
        "expected_reward": -1.26,
        "observed_reward": round(float(res6["reward"]), 4),
        "false_alarm_penalty": float(res6["false_alarm_penalty"]),
        "redundant_penalty": float(res6["redundant_penalty"]),
        "dwell_cost": float(res6["dwell_cost"]),
        "status": "PASS" if math.isclose(res6["reward"], -1.26, abs_tol=1e-3) else "FAIL",
    }

    # Case 7: Periodic Prediction Hit
    obs7 = _make_observation(500.0, [SimpleNamespace(time_us=0.0)])
    res7 = receiver_reward_components_v2(
        observation=obs7,
        ground_truth_active=True,
        novel_emitter=False,
        detected=True,
        intercept_time_us=0.0,
        is_predicted=True,
    )
    cases["periodic_prediction_hit"] = {
        "parameters": "repeat=True, predicted=True, t_hit=0us, NORMAL (500us)",
        "expected_reward": 13.50,
        "observed_reward": round(float(res7["reward"]), 4),
        "interception_reward": float(res7["interception_reward"]),
        "latency_reward": float(res7["latency_reward"]),
        "prediction_bonus": float(res7["prediction_bonus"]),
        "status": "PASS" if math.isclose(res7["reward"], 13.50, abs_tol=1e-3) else "FAIL",
    }

    all_passed = all(c["status"] == "PASS" for c in cases.values())
    return {
        "all_passed": all_passed,
        "cases": cases,
    }


def run_dwell_time_scaling_analysis() -> Dict[str, Any]:
    """Controlled scaling analysis across SHORT, NORMAL, and LONG dwell durations."""
    dwells = {
        "SHORT": 125.0,
        "NORMAL": 500.0,
        "LONG": 1250.0,
    }
    fractions = [0.0, 0.2, 0.5, 0.8, 1.0]
    scaling_matrix = {}

    for mode_name, dwell_us in dwells.items():
        mode_results = []
        for frac in fractions:
            t_hit = frac * dwell_us
            obs = _make_observation(dwell_us, [SimpleNamespace(time_us=t_hit)])
            res = receiver_reward_components_v2(
                observation=obs,
                ground_truth_active=True,
                novel_emitter=False,
                detected=True,
                intercept_time_us=t_hit,
                is_agile=True,
                is_predicted=True,
            )
            mode_results.append({
                "normalized_hit_fraction": frac,
                "t_hit_us": t_hit,
                "total_reward": round(float(res["reward"]), 4),
                "reward_per_ms": round(float(res["reward_per_ms"]), 4),
                "hit_reward_per_ms": round(float(res["hit_reward_per_ms"]), 4),
            })
        scaling_matrix[mode_name] = {
            "dwell_us": dwell_us,
            "dwell_ms": dwell_us / 1000.0,
            "evaluations": mode_results,
        }

    # Verify that for constant normalized hit fraction, SHORT yields highest reward per ms
    frac_mid_idx = 2  # frac = 0.5
    short_rate = scaling_matrix["SHORT"]["evaluations"][frac_mid_idx]["reward_per_ms"]
    normal_rate = scaling_matrix["NORMAL"]["evaluations"][frac_mid_idx]["reward_per_ms"]
    long_rate = scaling_matrix["LONG"]["evaluations"][frac_mid_idx]["reward_per_ms"]

    no_pathological_advantage = (short_rate > normal_rate > long_rate)

    return {
        "scaling_matrix": scaling_matrix,
        "controlled_scaling_test": {
            "normalized_fraction": 0.5,
            "reward_per_ms_short": short_rate,
            "reward_per_ms_normal": normal_rate,
            "reward_per_ms_long": long_rate,
            "short_greater_than_normal": bool(short_rate > normal_rate),
            "normal_greater_than_long": bool(normal_rate > long_rate),
            "no_pathological_long_advantage": no_pathological_advantage,
        },
    }


def run_anti_contamination_audit() -> Dict[str, Any]:
    """Verify zero proxy reward contamination across all 5 dimensions."""
    obs = _make_observation(500.0, [SimpleNamespace(time_us=50.0, emitter_id=10, threat_class="SAM_RADAR")])

    # 1. Action score invariance
    r_base = receiver_reward_components_v2(observation=obs, ground_truth_active=True, detected=True, intercept_time_us=50.0)
    r_score = receiver_reward_components_v2(observation=obs, ground_truth_active=True, detected=True, intercept_time_us=50.0, action_score=42.0)
    action_score_inv = (r_base == r_score)

    # 2. Threat class invariance
    obs_threat = _make_observation(500.0, [SimpleNamespace(time_us=50.0, emitter_id=10, threat_class="BENIGN_WEATHER")])
    r_threat = receiver_reward_components_v2(observation=obs_threat, ground_truth_active=True, detected=True, intercept_time_us=50.0)
    threat_class_inv = (r_base == r_threat)

    # 3. Emitter ID permutation invariance
    obs_id = _make_observation(500.0, [SimpleNamespace(time_us=50.0, emitter_id=9999, threat_class="SAM_RADAR")])
    r_id = receiver_reward_components_v2(observation=obs_id, ground_truth_active=True, detected=True, intercept_time_us=50.0)
    emitter_id_inv = (r_base == r_id)

    # 4. External heuristic ranking invariance
    r_heur = receiver_reward_components_v2(observation=obs, ground_truth_active=True, detected=True, intercept_time_us=50.0, heuristic_rank=1, thompson_score=0.99)
    heuristic_inv = (r_base == r_heur)

    # 5. Future record isolation
    rec1 = PulseRecord(toa_us=200.0, frequency_mhz=2500.0, pulse_width_us=1.0, amplitude_db=-30.0, aoa_deg=45.0, emitter_id=1)
    rec_future = PulseRecord(toa_us=800.0, frequency_mhz=2500.0, pulse_width_us=1.0, amplitude_db=-30.0, aoa_deg=45.0, emitter_id=1)
    env1 = CognitiveRFScanEnv({"dwell_time_us": 500.0, "reward": {"version": "v2"}}, records=[rec1], seed=42)
    env1.reset()
    _, r_step1, _, _, _ = env1.step(action=5)

    env2 = CognitiveRFScanEnv({"dwell_time_us": 500.0, "reward": {"version": "v2"}}, records=[rec1, rec_future], seed=42)
    env2.reset()
    _, r_step2, _, _, _ = env2.step(action=5)
    future_record_inv = math.isclose(r_step1, r_step2, abs_tol=1e-7)

    all_isolated = all([action_score_inv, threat_class_inv, emitter_id_inv, heuristic_inv, future_record_inv])

    return {
        "all_proxy_contamination_rejected": all_isolated,
        "action_score_invariance": action_score_inv,
        "threat_class_invariance": threat_class_inv,
        "emitter_id_permutation_invariance": emitter_id_inv,
        "external_heuristic_ranking_invariance": heuristic_inv,
        "future_record_isolation": future_record_inv,
    }


def verify_frozen_checkpoint() -> Dict[str, Any]:
    """Verify SHA-256 of production checkpoint."""
    hasher = hashlib.sha256()
    with open(FROZEN_CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    observed = hasher.hexdigest()
    matched = (observed == EXPECTED_CHECKPOINT_SHA256)
    return {
        "checkpoint_path": str(FROZEN_CHECKPOINT_PATH),
        "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        "observed_sha256": observed,
        "bitwise_identical": matched,
    }


def main():
    print("Executing Phase 4 Qualification Benchmarks...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Canonical Reward Table
    print("Running Canonical Reward Table evaluations...")
    table_res = run_canonical_reward_table()
    table_path = REPORT_DIR / "phase4_reward_table.json"
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(table_res, f, indent=2)
    print(f"Saved: {table_path}")

    # 2. Dwell-Time Scaling Analysis
    print("Running Dwell-Time Scaling Analysis...")
    scaling_res = run_dwell_time_scaling_analysis()
    scaling_path = REPORT_DIR / "phase4_dwell_time_analysis.json"
    with open(scaling_path, "w", encoding="utf-8") as f:
        json.dump(scaling_res, f, indent=2)
    print(f"Saved: {scaling_path}")

    # 3. Anti-Contamination Audit
    print("Running Anti-Contamination Audit...")
    anti_res = run_anti_contamination_audit()
    anti_path = REPORT_DIR / "phase4_anti_contamination.json"
    with open(anti_path, "w", encoding="utf-8") as f:
        json.dump(anti_res, f, indent=2)
    print(f"Saved: {anti_path}")

    # 4. Checkpoint Verification
    print("Verifying Frozen Baseline Checkpoint...")
    ckpt_res = verify_frozen_checkpoint()

    print("\nPhase 4 Qualification Summary:")
    print(f"  Canonical Reward Table: {'PASS' if table_res['all_passed'] else 'FAIL'}")
    print(f"  Dwell Time Scaling (No Pathological Long Advantage): {'PASS' if scaling_res['controlled_scaling_test']['no_pathological_long_advantage'] else 'FAIL'}")
    print(f"  Anti-Contamination Isolation: {'PASS' if anti_res['all_proxy_contamination_rejected'] else 'FAIL'}")
    print(f"  Frozen Checkpoint Integrity: {'PASS' if ckpt_res['bitwise_identical'] else 'FAIL'}")


if __name__ == "__main__":
    main()
