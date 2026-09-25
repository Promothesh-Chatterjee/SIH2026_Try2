#!/usr/bin/env python3
"""Dedicated evaluation runner for the held-out test set (Phase G & H).

Evaluates candidate models and comparison baselines against the immutable
held-out test set scenarios defined in `experiments/test_set/TEST_SET_MANIFEST.json`.

CONTRACT RULES:
1. Evaluates directly against `stare/test_stare` with zero fallback guessing.
2. Cryptographically verifies each test scenario file against its manifest SHA-256 before running.
3. Evaluates SmartScan alongside comparison baselines (Random, RoundRobin, HighestOccupancy)
   on the exact same held-out scenarios to provide fair comparative figures.
4. Held-out scenarios are strictly for generalization evaluation; never for model selection.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES
from ew_core.models.baseline_schedulers import (
    HighestOccupancyScheduler,
    RoundRobinScheduler,
)
from ew_core.models.random_scheduler import RandomScheduler
from ew_core.training.eval_batch import run_evaluation
from scripts.benchmark import (
    DEFAULT_CHECKPOINT,
    FROZEN_25K_SHA,
    load_smartscan_moe,
    resolve_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_held_out_test_set")


def verify_scenario_hashes(test_data_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Verify existence and SHA-256 of all declared held-out test scenarios."""
    verified_files: dict[str, str] = {}
    expected_scenarios = manifest.get("held_out_test_scenarios", {})

    for sid, info in expected_scenarios.items():
        fname = info["file"]
        expected_sha = info["sha256"]
        fpath = test_data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"FATAL: Held-out test scenario file missing: {fpath} "
                f"(expected from TEST_SET_MANIFEST.json)"
            )
        sha256 = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_sha = sha256.hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"FATAL: SHA-256 mismatch for held-out scenario {fname}!\n"
                f"Expected: {expected_sha}\n"
                f"Actual:   {actual_sha}"
            )
        verified_files[sid] = actual_sha

    logger.info("Successfully verified SHA-256 for all %d held-out test scenarios.", len(verified_files))
    return verified_files


def run_test_set_evaluation(
    checkpoint_path: Path | str | None = None,
    tsrd_root: str = "D:/TSRD",
    n_steps: int = 500,
    seed: int = 42,
    output_path: str = "reports/held_out_test_set_results.json",
    device: str = "cpu",
    evaluate_baselines: bool = True,
) -> Dict[str, Any]:
    """Execute evaluation on the held-out test set with comparative baselines."""
    manifest_path = repo_root / "experiments" / "test_set" / "TEST_SET_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Test set manifest missing: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_data_dir = Path(tsrd_root) / "stare" / "test_stare"
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Held-out test stare directory not found: {test_data_dir}")

    # Cryptographic pre-flight verification
    verified_hashes = verify_scenario_hashes(test_data_dir, manifest)
    test_scenarios = list(verified_hashes.keys())

    ckpt_path = resolve_checkpoint(checkpoint_path)
    dev = torch.device(device)
    moe = load_smartscan_moe(ckpt_path, device=dev)

    schedulers_to_evaluate: dict[str, Any] = {
        "SmartScan_DRQN_MoE": moe,
    }

    if evaluate_baselines:
        schedulers_to_evaluate["HighestOccupancy"] = HighestOccupancyScheduler(
            n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES
        )
        schedulers_to_evaluate["RoundRobin"] = RoundRobinScheduler(
            n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES
        )
        schedulers_to_evaluate["Random"] = RandomScheduler(
            n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES, seed=seed
        )

    all_results: dict[str, Any] = {}

    for sched_name, sched in schedulers_to_evaluate.items():
        logger.info("Evaluating [%s] on Held-Out Test Set (500 steps x %d scenarios)...", sched_name, len(test_scenarios))
        res = run_evaluation(
            scheduler=sched,
            scenario_ids=test_scenarios,
            n_steps=n_steps,
            seed=seed,
            policy_mode="operational",
            data_dir=tsrd_root,
            device=dev,
            dataset_subset="test_stare",
            scenario_dir=test_data_dir,
        )

        op_lat = float(res.get("operational_intercept_latency_us", res.get("avg_intercept_time_error_us", 0.0)))
        all_results[sched_name] = {
            "mean_ir_pct": float(res["avg_intercept_rate"] * 100.0),
            "pd_pct": float(res["pd"] * 100.0),
            "pfa_pct": float(res["pfa"] * 100.0),
            "pct_correct_predictions": float(res["pct_correct_predictions"]),
            "operational_intercept_latency_us": op_lat,
            "avg_intercept_time_error_us_deprecated": float(res["avg_intercept_time_error_us"]),
            "avg_reward": float(res["avg_reward"]),
            "tp": int(res["tp"]),
            "fn": int(res["fn"]),
            "fp": int(res["fp"]),
            "tn": int(res["tn"]),
            "scenario_breakdown": res.get("scenario_breakdown", {}),
        }

    smartscan_res = all_results["SmartScan_DRQN_MoE"]
    summary = {
        "evaluation_type": "HELD_OUT_TEST_SET_GENERALIZATION",
        "benchmark_contract": "2026.1-CANONICAL",
        "checkpoint_sha256": FROZEN_25K_SHA,
        "n_steps_per_scenario": n_steps,
        "total_dwells_per_policy": len(test_scenarios) * n_steps,
        "seed": seed,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "manifest_used": "experiments/test_set/TEST_SET_MANIFEST.json",
        "test_dataset_directory": str(test_data_dir.resolve()).replace("\\", "/"),
        "verified_scenario_hashes": verified_hashes,
        "smartscan_results": {
            "mean_ir_pct": smartscan_res["mean_ir_pct"],
            "pd_pct": smartscan_res["pd_pct"],
            "pfa_pct": smartscan_res["pfa_pct"],
            "pct_correct_predictions": smartscan_res["pct_correct_predictions"],
            "operational_intercept_latency_us": smartscan_res["operational_intercept_latency_us"],
            "avg_reward": smartscan_res["avg_reward"],
            "tp": smartscan_res["tp"],
            "fn": smartscan_res["fn"],
            "fp": smartscan_res["fp"],
            "tn": smartscan_res["tn"],
        },
        "all_schedulers": {
            k: {
                "mean_ir_pct": v["mean_ir_pct"],
                "pd_pct": v["pd_pct"],
                "pfa_pct": v["pfa_pct"],
                "pct_correct_predictions": v["pct_correct_predictions"],
                "operational_intercept_latency_us": v["operational_intercept_latency_us"],
                "avg_reward": v["avg_reward"],
                "tp": v["tp"],
                "fn": v["fn"],
                "fp": v["fp"],
                "tn": v["tn"],
            }
            for k, v in all_results.items()
        },
        "scenario_breakdown": smartscan_res["scenario_breakdown"],
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    logger.info("Held-out test results with comparative baselines written to %s", out_file)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Held-Out Test Set Evaluator")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tsrd_root", type=str, default="D:/TSRD")
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reports/held_out_test_set_results.json")
    args = parser.parse_args()

    run_test_set_evaluation(
        checkpoint_path=args.checkpoint,
        tsrd_root=args.tsrd_root,
        n_steps=args.n_steps,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
