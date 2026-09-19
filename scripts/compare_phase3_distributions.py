"""Pre-Phase-3 vs Post-Phase-3 Observation Distribution Comparison.

Cognitive EW Smart Scan Scheduler v2.
Runs 1,000 environment steps comparing pre-Phase-3 legacy feature calculation
against post-Phase-3 unified canonical belief state.

Outputs:
  - Pre & Post mean / std
  - Pre & Post quantiles (25%, 50%, 75%)
  - Two-sample Kolmogorov-Smirnov (KS) statistic and p-value
  - Fraction at bounds [0.0, 1.0]
  - NaN / Inf counts
  - Explicit predefined tolerance verification:
      1. Boundedness: strictly in [0.0, 1.0]
      2. Finiteness: exactly 0 NaNs, 0 Infs
      3. Non-degeneracy: non-zero variance (no mode collapse)
      4. Mean drift: |mean_post - mean_pre| <= 0.35 (operates within DRQN dynamic range)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.stats

from ew_core.contracts import CANONICAL_BAND_FEATURES, CANONICAL_N_BANDS
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.radio_environment import PulseRecord


CANONICAL_FEATURE_NAMES = [
    "occupancy_prob",
    "detection_rate",
    "miss_rate",
    "uncertainty",
    "revisit_age",
    "emitter_count",
    "deint_confidence",
    "pri_stability",
    "agility",
    "priority",
]

OUTPUT_PATH = Path("experiments/reports/phase3/phase3_distribution_comparison.json")


def generate_synthetic_records(n_pulses: int = 1500) -> List[PulseRecord]:
    """Generate deterministic synthetic pulses across diverse radar behaviors."""
    records = []
    # Emitter 1: Stable pulsed radar in Band 4 (2250 MHz)
    for i in range(500):
        records.append(PulseRecord(
            emitter_id=1,
            frequency_mhz=2250.0 + 0.1 * (i % 3),
            pulse_width_us=10.0,
            amplitude_db=-60.0,
            aoa_deg=45.0,
            toa_us=i * 200.0,
        ))
    # Emitter 2: Periodic hopper across Bands 10, 15, 20 (5250, 7750, 10250 MHz)
    hop_freqs = [5250.0, 7750.0, 10250.0]
    for i in range(500):
        records.append(PulseRecord(
            emitter_id=2,
            frequency_mhz=hop_freqs[i % 3],
            pulse_width_us=5.0,
            amplitude_db=-55.0,
            aoa_deg=135.0,
            toa_us=i * 150.0 + 20.0,
        ))
    # Emitter 3: Agile radar across Bands 25..30 (12500 - 15000 MHz)
    for i in range(500):
        f = 12500.0 + float((i * 37) % 2500)
        records.append(PulseRecord(
            emitter_id=3,
            frequency_mhz=f,
            pulse_width_us=15.0,
            amplitude_db=-65.0,
            aoa_deg=270.0,
            toa_us=i * 180.0 + 50.0,
        ))
    records.sort(key=lambda r: r.toa_us)
    return records


def run_legacy_pre_step(
    env: CognitiveRFScanEnv,
    band: int,
    hit: bool,
    detections: List[Any],
    legacy_belief: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute legacy (pre-Phase-3) feature representations."""
    n = CANONICAL_N_BANDS
    # Legacy occupancy: naive EMA without asymmetric miss decay
    alpha = 0.3
    target = 1.0 if hit else 0.0
    legacy_belief["occupancy"][band] = (1.0 - alpha) * legacy_belief["occupancy"][band] + alpha * target
    legacy_belief["visits"][band] += 1
    if hit:
        legacy_belief["hits"][band] += 1
    legacy_belief["age"] += 1
    legacy_belief["age"][band] = 0

    obs_legacy = np.zeros((n, CANONICAL_BAND_FEATURES), dtype=np.float32)
    for b in range(n):
        p = float(np.clip(legacy_belief["occupancy"][b], 0.0, 1.0))
        dwells = int(legacy_belief["visits"][b])
        hits = int(legacy_belief["hits"][b])
        det = float(hits / dwells) if dwells > 0 else 0.0
        miss = 1.0 - det
        unc = 1.0 - abs(2.0 * p - 1.0) if dwells > 0 else 1.0
        age = float(min(float(legacy_belief["age"][b]), 50.0) / 50.0)

        # Legacy emitter count: AoA bin slicing heuristic
        if b == band and detections:
            aoas = [float(getattr(d, "aoa_deg", 0.0)) for d in detections]
            unique_bearings = len(set(int(a / 15.0) for a in aoas))
            emit_cnt = float(np.clip(unique_bearings / 5.0, 0.1, 1.0))
            deint_conf = float(np.clip(0.6 + 0.08 * min(len(detections), 5), 0.0, 1.0))
            freqs = [float(getattr(d, "frequency_mhz", 0.0)) for d in detections]
            agil = float(np.clip(np.std(freqs) / 100.0, 0.0, 1.0)) if len(freqs) >= 2 else 0.0
            toas = [float(getattr(d, "time_us", 0.0)) for d in detections]
            if len(toas) >= 2:
                pris = np.diff(sorted(toas))
                cv = float(np.std(pris) / max(float(np.mean(pris)), 1e-6))
                pri_stab = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
            else:
                pri_stab = 0.5
        else:
            emit_cnt = 0.0
            deint_conf = 0.0
            agil = 0.0
            pri_stab = 0.0

        prio = float(np.clip(0.35 * age + 0.25 * p + 0.20 * unc, 0.0, 1.0))

        obs_legacy[b] = [p, det, miss, unc, age, emit_cnt, deint_conf, pri_stab, agil, prio]

    return obs_legacy


def main():
    print("Initializing environment for 1,000-step distribution comparison...")
    records = generate_synthetic_records(1500)
    env = CognitiveRFScanEnv(
        config={"max_steps": 1000},
        records_provider=lambda: list(records),
    )
    env.reset(seed=42)

    n_steps = 1000
    pre_features = [[] for _ in range(CANONICAL_BAND_FEATURES)]
    post_features = [[] for _ in range(CANONICAL_BAND_FEATURES)]

    legacy_belief = {
        "occupancy": np.full(CANONICAL_N_BANDS, 0.5, dtype=np.float32),
        "visits": np.zeros(CANONICAL_N_BANDS, dtype=np.int64),
        "hits": np.zeros(CANONICAL_N_BANDS, dtype=np.int64),
        "age": np.ones(CANONICAL_N_BANDS, dtype=np.int64),
    }

    # Action schedule cycling exploration and targeted revisits
    for step in range(n_steps):
        # Round-robin with periodic revisits to active bands
        if step % 3 == 0:
            band = 4  # Stable
        elif step % 3 == 1:
            band = [10, 15, 20][(step // 3) % 3]  # Hopper
        else:
            band = (step % CANONICAL_N_BANDS)  # Exploration
        action = band * 5 + 1  # Standard mode

        obs_post, reward, term, trunc, info = env.step(action)
        detections = info.get("detections", [])
        hit = bool(info.get("hit", False))

        obs_post_mat = obs_post.reshape(CANONICAL_N_BANDS, CANONICAL_BAND_FEATURES)
        obs_pre_mat = run_legacy_pre_step(env, band, hit, detections, legacy_belief)

        for f_idx in range(CANONICAL_BAND_FEATURES):
            pre_features[f_idx].extend(obs_pre_mat[:, f_idx].tolist())
            post_features[f_idx].extend(obs_post_mat[:, f_idx].tolist())

        if term or trunc:
            env.reset(seed=42 + step)

    print(f"Collected {len(pre_features[0])} observations per feature across {n_steps} steps.")

    comparison_results: Dict[str, Any] = {}
    tolerances = {
        "max_mean_drift": 0.35,
        "require_bounded_in_0_1": True,
        "require_finite": True,
        "min_variance": 1e-5,
    }

    all_passed = True
    print("\nFeature Distribution Comparison (Pre-Phase-3 vs Post-Phase-3):")
    print(f"{'Feature Name':<18} {'Pre Mean±Std':<16} {'Post Mean±Std':<16} {'KS Stat (p)':<16} {'Bound% (0/1)':<14} {'Status'}")
    print("-" * 92)

    for idx, name in enumerate(CANONICAL_FEATURE_NAMES):
        pre_arr = np.asarray(pre_features[idx], dtype=np.float64)
        post_arr = np.asarray(post_features[idx], dtype=np.float64)

        pre_mean = float(np.mean(pre_arr))
        pre_std = float(np.std(pre_arr))
        post_mean = float(np.mean(post_arr))
        post_std = float(np.std(post_arr))

        pre_q = [float(q) for q in np.percentile(pre_arr, [25, 50, 75])]
        post_q = [float(q) for q in np.percentile(post_arr, [25, 50, 75])]

        # Kolmogorov-Smirnov test
        ks_res = scipy.stats.ks_2samp(pre_arr, post_arr)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        # Bound & finiteness checks
        nan_count = int(np.sum(np.isnan(post_arr)))
        inf_count = int(np.sum(np.isinf(post_arr)))
        frac_at_0 = float(np.mean(post_arr == 0.0))
        frac_at_1 = float(np.mean(post_arr == 1.0))
        in_bounds = bool(np.all(post_arr >= 0.0) and np.all(post_arr <= 1.0))

        drift = abs(post_mean - pre_mean)
        is_finite = bool(nan_count == 0 and inf_count == 0)
        has_variance = bool(post_std >= tolerances["min_variance"])
        drift_ok = bool(drift <= tolerances["max_mean_drift"])

        status = "COMPATIBLE_WITHIN_TOLERANCE" if (in_bounds and is_finite and has_variance and drift_ok) else "OUT_OF_TOLERANCE"
        if status != "COMPATIBLE_WITHIN_TOLERANCE":
            all_passed = False

        comparison_results[name] = {
            "pre_mean": round(pre_mean, 4),
            "pre_std": round(pre_std, 4),
            "post_mean": round(post_mean, 4),
            "post_std": round(post_std, 4),
            "pre_quantiles_25_50_75": [round(q, 4) for q in pre_q],
            "post_quantiles_25_50_75": [round(q, 4) for q in post_q],
            "mean_drift": round(drift, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": ks_pval,
            "frac_at_lower_bound_0": round(frac_at_0, 4),
            "frac_at_upper_bound_1": round(frac_at_1, 4),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "in_bounds_0_1": in_bounds,
            "status": status,
        }

        print(f"{name:<18} {pre_mean:.3f}±{pre_std:.3f}     {post_mean:.3f}±{post_std:.3f}     {ks_stat:.3f} (p={ks_pval:.2e})  {frac_at_0:.2f}/{frac_at_1:.2f}     {status}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "num_steps": n_steps,
            "total_observations_evaluated": len(pre_features[0]),
            "tolerances": tolerances,
            "all_features_compatible": all_passed,
            "features": comparison_results,
        }, f, indent=2)

    print(f"\nDistribution comparison successfully saved to {OUTPUT_PATH}")
    assert all_passed, "One or more features fell outside defined compatibility tolerances."


if __name__ == "__main__":
    main()
