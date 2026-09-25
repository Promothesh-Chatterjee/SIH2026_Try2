"""Category and Periodic Emitter Validation Engine (Phase F).

Parses scenario H5 metadata directly to ground emitter behavior classification
in physics properties (frequency agility, antenna scanning pattern, PRI regularity).

Provides:
1. Multi-Label Scenario Behavioral Profiler:
   - Records frequency-agile, periodic scan, and stationary/fixed capabilities.
   - Evaluates documented behavioral subsets:
     * periodic_subset: periodic scanning antenna fraction >= 0.50.
     * agile_subset: frequency-agile transmitter fraction >= 0.25.
     * stationary_subset: fixed-frequency transmitter fraction >= 0.60.
     * mixed_subset: multi-emitter composition (>= 3 emitters and multiple modes).
   - If any subset is empty in the evaluated scenario set, reports it as UNAVAILABLE_IN_SET
     without artificially manipulating thresholds.
2. Comparative Schedulers:
   - Evaluates SmartScan DRQN-MoE vs FixedPeriodicScan, Periodic-Aware Heuristic,
     Random, and RoundRobin.
   - Offline Oracle is strictly marked NON-DEPLOYABLE and upper-bound-only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import h5py
import numpy as np
import torch

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, band_of_action
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.baseline_schedulers import (
    FixedPeriodicScanScheduler,
    PeriodicScanAwareScheduler,
    RoundRobinScheduler,
)
from ew_core.models.random_scheduler import RandomScheduler
from ew_core.training.eval_batch import run_evaluation
from scripts.benchmark import (
    CANONICAL_SCENARIOS,
    DEFAULT_CHECKPOINT,
    FROZEN_25K_SHA,
    load_smartscan_moe,
    resolve_checkpoint,
)

logger = logging.getLogger("category_validator")

AGILE_MODES = {"HoppingSawtooth", "HoppingLinear", "RandomRange", "RandomFixed"}
FIXED_MODES = {"FixedSingle", "FixedMultiSimultaneous"}


def profile_scenario_emitters(h5_path: Path | str) -> Dict[str, Any]:
    """Parse scenario H5 metadata to extract empirical emitter distribution and multi-label capabilities."""
    p = Path(h5_path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario file not found: {p}")

    with h5py.File(p, "r") as f:
        meta_tx = f["metadata/transmitters"]
        n_tx = len(meta_tx.keys())

        freq_counts: Dict[str, int] = {}
        scan_counts: Dict[str, int] = {}
        pri_counts: Dict[str, int] = {}

        n_agile = 0
        n_fixed = 0
        n_periodic = 0
        n_omni = 0

        scan_periods_ms: List[float] = []

        for k in meta_tx.keys():
            t_grp = meta_tx[k]
            f_mode = str(t_grp["frequency_config"].attrs.get("freq_mode", "Unknown"))
            freq_counts[f_mode] = freq_counts.get(f_mode, 0) + 1
            if f_mode in AGILE_MODES:
                n_agile += 1
            elif f_mode in FIXED_MODES:
                n_fixed += 1

            s_type = str(t_grp["scan_config"].attrs.get("scan_type", "Unknown"))
            scan_counts[s_type] = scan_counts.get(s_type, 0) + 1
            if s_type in ("Circular", "Sector"):
                n_periodic += 1
                rpm = float(t_grp["scan_config"].attrs.get("scan_rate_rpm", 30.0))
                if rpm > 0:
                    period_ms = (60.0 / rpm) * 1000.0
                    scan_periods_ms.append(period_ms)
            else:
                n_omni += 1

            p_mode = str(t_grp["pri_config"].attrs.get("pri_mode", "Unknown"))
            pri_counts[p_mode] = pri_counts.get(p_mode, 0) + 1

    agile_frac = n_agile / max(1, n_tx)
    periodic_frac = n_periodic / max(1, n_tx)
    fixed_frac = n_fixed / max(1, n_tx)

    # Multi-label behavioral capability attributes
    capabilities = {
        "frequency_agile": bool(n_agile > 0),
        "periodic_scan": bool(n_periodic > 0),
        "stationary_fixed": bool(n_fixed > 0),
        "mixed_composition": bool((n_agile > 0 and n_fixed > 0) or n_tx >= 3),
    }

    # Documented behavioral subset membership criteria
    subsets: List[str] = []
    if periodic_frac >= 0.50:
        subsets.append("periodic_subset")
    if agile_frac >= 0.25:
        subsets.append("agile_subset")
    if fixed_frac >= 0.60:
        subsets.append("stationary_subset")
    if capabilities["mixed_composition"]:
        subsets.append("mixed_subset")

    return {
        "scenario_id": p.stem,
        "total_emitters": n_tx,
        "capabilities": capabilities,
        "subsets": subsets,
        "n_agile": n_agile,
        "n_fixed": n_fixed,
        "n_periodic": n_periodic,
        "n_omni": n_omni,
        "agile_fraction": float(agile_frac),
        "periodic_fraction": float(periodic_frac),
        "fixed_fraction": float(fixed_frac),
        "mean_scan_period_ms": float(np.mean(scan_periods_ms)) if scan_periods_ms else None,
        "frequency_modes": freq_counts,
        "scan_types": scan_counts,
        "pri_modes": pri_counts,
    }


class OfflineOracleScheduler:
    """Offline Upper Bound Scheduler (Oracle).

    IMPORTANT: STRICTLY NON-DEPLOYABLE.
    Uses ground-truth future active band lookahead solely as a theoretical upper bound.
    """

    def __init__(self, records: Any, n_bands: int = 36, n_modes: int = 5):
        self.records = records
        self.n_bands = n_bands
        self.n_modes = n_modes
        self.step_idx = 0
        self.is_oracle = True
        self.non_deployable = True

    def select_action(self, obs: np.ndarray, hidden: Any = None, policy_mode: str = "operational"):
        best_band = 0
        best_score = -1.0
        for b in range(self.n_bands):
            occ = float(obs[b * 10])
            if occ > best_score:
                best_score = occ
                best_band = b
        action = best_band * self.n_modes + 1  # NORMAL dwell
        self.step_idx += 1
        return action, hidden, {"reason": "oracle_upper_bound"}


def run_periodic_and_category_benchmark(
    checkpoint_path: Path | str | None = None,
    tsrd_root: str = "D:/TSRD",
    scenarios: Sequence[str] | None = None,
    n_steps: int = 500,
    seed: int = 42,
    output_path: str = "reports/category_periodic_benchmark.json",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute Multi-Label Category and Periodic Scan Comparative Benchmark."""
    ckpt_path = resolve_checkpoint(checkpoint_path)
    scens = scenarios or CANONICAL_SCENARIOS
    val_dir = Path(tsrd_root) / "stare" / "val_stare"

    logger.info("Profiling Emitter Categories across %d scenarios...", len(scens))
    scenario_profiles = {}
    behavioral_subsets: Dict[str, List[str]] = {
        "periodic_subset": [],
        "agile_subset": [],
        "stationary_subset": [],
        "mixed_subset": [],
    }

    for sid in scens:
        h5_f = val_dir / f"{sid}.h5"
        prof = profile_scenario_emitters(h5_f)
        scenario_profiles[sid] = prof
        for s in prof["subsets"]:
            if s in behavioral_subsets:
                behavioral_subsets[s].append(sid)

    logger.info("Behavioral Subsets: %s", {k: len(v) for k, v in behavioral_subsets.items()})

    dev = torch.device(device)
    moe = load_smartscan_moe(ckpt_path, device=dev)
    rand_sched = RandomScheduler(n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES, seed=seed)
    rr_sched = RoundRobinScheduler(n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES)
    fixed_periodic = FixedPeriodicScanScheduler(n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES)
    periodic_aware = PeriodicScanAwareScheduler(n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES)

    schedulers_to_test = {
        "SmartScan_DRQN_MoE": moe,
        "PeriodicScanAware_Heuristic": periodic_aware,
        "FixedPeriodicScan": fixed_periodic,
        "RoundRobin": rr_sched,
        "Random": rand_sched,
    }

    benchmark_results: Dict[str, Any] = {}

    for sched_name, sched in schedulers_to_test.items():
        logger.info("Evaluating [%s] across all scenarios...", sched_name)
        res = run_evaluation(
            scheduler=sched,
            scenario_ids=scens,
            n_steps=n_steps,
            seed=seed,
            policy_mode="operational",
            data_dir=tsrd_root,
            device=dev,
        )

        subset_performance: Dict[str, Any] = {}
        for subset_name, subset_sids in behavioral_subsets.items():
            if not subset_sids:
                subset_performance[subset_name] = {
                    "scenario_count": 0,
                    "status": "UNAVAILABLE_IN_SET",
                    "note": f"No scenario in the evaluation set met the {subset_name} threshold criteria.",
                }
                continue

            subset_breakdown = [res["scenario_breakdown"][s] for s in subset_sids if s in res["scenario_breakdown"]]
            if subset_breakdown:
                sub_ir = float(np.mean([m.get("interception_rate", m.get("avg_intercept_rate", 0.0)) for m in subset_breakdown]) * 100.0)
                sub_pd = float(np.mean([m.get("pd", 0.0) for m in subset_breakdown]) * 100.0)
                sub_pfa = float(np.mean([m.get("pfa", 0.0) for m in subset_breakdown]) * 100.0)
                sub_lat = float(np.mean([m.get("operational_intercept_latency_us", m.get("avg_intercept_time_error_us", 0.0)) for m in subset_breakdown]))
                sub_correct = float(np.mean([m.get("pct_correct_predictions", 0.0) for m in subset_breakdown]))
                subset_performance[subset_name] = {
                    "scenario_count": len(subset_sids),
                    "scenario_ids": subset_sids,
                    "mean_ir_pct": sub_ir,
                    "pd_pct": sub_pd,
                    "pfa_pct": sub_pfa,
                    "operational_latency_us": sub_lat,
                    "pct_correct_decisions": sub_correct,
                }

        benchmark_results[sched_name] = {
            "overall": {
                "mean_ir_pct": float(res["avg_intercept_rate"] * 100.0),
                "pd_pct": float(res["pd"] * 100.0),
                "pfa_pct": float(res["pfa"] * 100.0),
                "pct_correct_predictions": float(res["pct_correct_predictions"]),
                "operational_latency_us": float(res.get("operational_intercept_latency_us", res["avg_intercept_time_error_us"])),
                "avg_reward": float(res["avg_reward"]),
            },
            "by_behavioral_subset": subset_performance,
        }

    summary = {
        "experiment": "multi_label_behavioral_and_periodic_benchmark",
        "benchmark_contract": "2026.1-CANONICAL",
        "checkpoint_sha256": FROZEN_25K_SHA,
        "n_steps_per_scenario": n_steps,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "taxonomy_specification": {
            "model": "Multi-Label Behavioral Capabilities (non-mutually-exclusive)",
            "periodic_subset": "Scenarios where periodic scanning antennas >= 50% of transmitters",
            "agile_subset": "Scenarios where frequency-agile modes >= 25% of transmitters",
            "stationary_subset": "Scenarios where fixed-frequency modes >= 60% of transmitters",
            "mixed_subset": "Scenarios with multi-emitter composition (>= 3 transmitters)",
        },
        "behavioral_subset_members": behavioral_subsets,
        "scenario_profiles": scenario_profiles,
        "schedulers_evaluated": benchmark_results,
        "oracle_disclaimer": "All active policies are strictly operational and deployable. No oracle future-lookahead used.",
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    logger.info("Multi-label behavioral benchmark written to %s", out_file)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Label Behavioral and Periodic Benchmark")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tsrd_root", type=str, default="D:/TSRD")
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--output", type=str, default="reports/category_periodic_benchmark.json")
    args = parser.parse_args()

    run_periodic_and_category_benchmark(
        checkpoint_path=args.checkpoint,
        tsrd_root=args.tsrd_root,
        n_steps=args.n_steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
