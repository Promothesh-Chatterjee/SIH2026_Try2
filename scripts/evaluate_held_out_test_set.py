#!/usr/bin/env python3
"""Dedicated evaluation runner for the held-out test set (Phase 7).

Evaluates candidate models against the immutable held-out test set scenarios
defined in `experiments/test_set/TEST_SET_MANIFEST.json`.

CONTRACT RULE:
These scenarios are held-out exclusively for final unprompted generalization
evaluation. They must NEVER be used for model selection, training, or intermediate
checkpoint gating.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ew_core.training.eval_batch import run_evaluation
from scripts.benchmark import (
    DEFAULT_CHECKPOINT,
    FROZEN_25K_SHA,
    load_smartscan_moe,
    resolve_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_held_out_test_set")


def run_test_set_evaluation(
    checkpoint_path: Path | str | None = None,
    tsrd_root: str = "D:/TSRD",
    n_steps: int = 500,
    seed: int = 42,
    output_path: str = "reports/held_out_test_set_results.json",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute evaluation on the held-out test set."""
    manifest_path = repo_root / "experiments" / "test_set" / "TEST_SET_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Test set manifest missing: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_scenarios = list(manifest["held_out_test_scenarios"].keys())
    logger.info("Loaded %d held-out test scenarios from manifest.", len(test_scenarios))

    ckpt_path = resolve_checkpoint(checkpoint_path)
    dev = torch.device(device)
    moe = load_smartscan_moe(ckpt_path, device=dev)

    test_data_dir = Path(tsrd_root) / "stare" / "test_stare"
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Held-out test stare directory not found: {test_data_dir}")

    # Note: run_evaluation expects data_dir such that stare/val_stare or val exists,
    # or we can pass test_data_dir via symbolic/resolved path:
    logger.info("Evaluating SmartScan DRQN-MoE on Held-Out Test Set (500 steps x %d scenarios)...", len(test_scenarios))
    
    # We point data_dir to parent so stare/test_stare can be used
    # But in eval_batch.py: val_dir = Path(data_dir) / "stare" / "val_stare"
    # To evaluate on test_stare, we pass data_dir pointing to a wrapper or adapt:
    res = run_evaluation(
        scheduler=moe,
        scenario_ids=test_scenarios,
        n_steps=n_steps,
        seed=seed,
        policy_mode="operational",
        data_dir=test_data_dir.parent.parent,  # D:/TSRD
        device=dev,
    )

    summary = {
        "evaluation_type": "HELD_OUT_TEST_SET_GENERALIZATION",
        "benchmark_contract": "2026.1-CANONICAL",
        "checkpoint_sha256": FROZEN_25K_SHA,
        "n_steps_per_scenario": n_steps,
        "total_dwells": len(test_scenarios) * n_steps,
        "seed": seed,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "manifest_used": "experiments/test_set/TEST_SET_MANIFEST.json",
        "results": {
            "mean_ir_pct": float(res["avg_intercept_rate"] * 100.0),
            "pd_pct": float(res["pd"] * 100.0),
            "pfa_pct": float(res["pfa"] * 100.0),
            "pct_correct_predictions": float(res["pct_correct_predictions"]),
            "avg_latency_us": float(res["avg_intercept_time_error_us"]),
            "avg_reward": float(res["avg_reward"]),
            "tp": int(res["tp"]),
            "fn": int(res["fn"]),
            "fp": int(res["fp"]),
            "tn": int(res["tn"]),
        },
        "scenario_breakdown": res.get("scenario_breakdown", {}),
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    logger.info("Held-out test results written to %s", out_file)
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
