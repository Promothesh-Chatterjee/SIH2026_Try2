"""
CLI runner for the Gate 75k Deep Stress & Diagnostic Evaluation Suite.

Evaluates the frozen Gate 75k checkpoint across 5 stress axes and outputs:
- Per-scenario intercept rate, fallback rate, and dwell mode usage
- Category-level aggregations (sparse, dense, intermittent, agile)
- Systematic gap analysis: HighestOccupancy vs DRQN+MoE
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
import yaml

from src.evaluation.stress_suite_evaluator import StressSuiteEvaluator, CURATED_STRESS_SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_gate75k_stress_suite")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gate 75k Stress Test Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scheduler/checkpoint_gate_75000.pt")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--output", type=str, default="results/gate_75000_stress_report.json")
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)

    env_cfg = dict(train_cfg.get("environment", {}))
    evaluator = StressSuiteEvaluator(
        checkpoint_path=args.checkpoint,
        env_config=env_cfg,
        model_config=model_cfg,
        device=args.device,
    )

    logger.info("Starting Gate 75k Stress Suite across all 4 categories (16 scenario-evals per policy)...")
    battery_results = evaluator.run_stress_battery(n_steps=args.n_steps)
    results = battery_results["results"]

    # 1. Print Per-Scenario Detailed Table
    print("\n" + "=" * 165)
    print("GATE 75k DEEP STRESS SUITE: PER-SCENARIO DETAILED SCORECARD")
    print("=" * 165)
    print(f"{'Category':<15} | {'Scenario':<18} | {'Policy':<18} | {'Intercept':<12} | {'Fallback':<10} | {'EmptyEsc':<10} | {'Distinct':<10} | {'Discov':<9} | {'TimingErr':<11} | {'Reward':<8}")
    print("-" * 165)

    for r in results:
        pol_display = "DRQN+MoE" if r["policy"] == "full_moe" else (r["policy"].upper() if r["policy"] == "drqn" else r["policy"].title().replace("_", ""))
        ir_str = f"{r['intercept_rate']*100:5.2f}% ({r['hits']:3d})"
        fb_str = f"{r['fallback_rate_pct']:5.1f}%" if r["fallback_rate_pct"] is not None else "N/A"
        empty_str = f"{r['empty_escape_rate']*100:5.1f}%"
        dist_str = f"{r['distinct_bands']:2d}/36"
        disc_str = f"{r['discovery_rate']*100:5.1f}%"
        terr_str = f"{r['avg_timing_error_us']:5.1f}us"
        rew_str = f"{r['total_reward']:7.1f}"
        print(f"{r['category']:<15} | {r['scenario_id']:<18} | {pol_display:<18} | {ir_str:<12} | {fb_str:<10} | {empty_str:<10} | {dist_str:<10} | {disc_str:<9} | {terr_str:<11} | {rew_str:<8}")

    print("=" * 165)

    # 2. Print Category-Level Summary Table
    categories = sorted(list(set(r["category"] for r in results)))
    policies = ["highest_occupancy", "highest_uncertainty", "round_robin", "drqn", "full_moe"]

    print("\n" + "=" * 140)
    print("CATEGORY-LEVEL PERFORMANCE SUMMARY")
    print("=" * 140)
    print(f"{'Category':<20} | {'Policy':<18} | {'Intercept Rate':<16} | {'Fallback Rate':<15} | {'Empty Escape':<14} | {'Distinct Bands':<15} | {'Discovery Rate'}")
    print("-" * 140)

    cat_summary: dict[str, dict[str, Any]] = {}
    for cat in categories:
        cat_summary[cat] = {}
        for pol in policies:
            subset = [r for r in results if r["category"] == cat and r["policy"] == pol]
            if not subset:
                continue
            mean_ir = float(np.mean([r["intercept_rate"] for r in subset]))
            mean_hits = float(np.mean([r["hits"] for r in subset]))
            fbs = [r["fallback_rate_pct"] for r in subset if r["fallback_rate_pct"] is not None]
            mean_fb = float(np.mean(fbs)) if fbs else None
            mean_empty = float(np.mean([r["empty_escape_rate"] for r in subset]))
            mean_dist = float(np.mean([r["distinct_bands"] for r in subset]))
            mean_disc = float(np.mean([r["discovery_rate"] for r in subset]))

            cat_summary[cat][pol] = {
                "mean_intercept_rate": mean_ir,
                "mean_hits": mean_hits,
                "mean_fallback_rate": mean_fb,
                "mean_empty_escape": mean_empty,
                "mean_distinct_bands": mean_dist,
                "mean_discovery_rate": mean_disc,
            }

            pol_display = "DRQN+MoE" if pol == "full_moe" else (pol.upper() if pol == "drqn" else pol.title().replace("_", ""))
            fb_str = f"{mean_fb:5.1f}%" if mean_fb is not None else "N/A"
            print(f"{cat:<20} | {pol_display:<18} | {mean_ir*100:6.2f}% ({mean_hits:4.1f}) | {fb_str:<15} | {mean_empty*100:6.1f}%        | {mean_dist:4.1f}/36         | {mean_disc*100:5.1f}%")
        print("-" * 140)
    print("=" * 140)

    # 3. Head-to-head Diagnosis: HighestOccupancy vs DRQN+MoE
    print("\n" + "=" * 115)
    print("HEAD-TO-HEAD DIAGNOSIS: HighestOccupancy (Camping) vs DRQN+MoE (Cognitive)")
    print("=" * 115)
    print(f"{'Scenario':<22} | {'Occupancy Intercept':<20} | {'DRQN+MoE Intercept':<20} | {'DRQN Primary %':<16} | {'Empty Escape (Occ vs MoE)'}")
    print("-" * 115)
    for scen_id in sorted(list(set(r["scenario_id"] for r in results))):
        occ_r = next(r for r in results if r["scenario_id"] == scen_id and r["policy"] == "highest_occupancy")
        moe_r = next(r for r in results if r["scenario_id"] == scen_id and r["policy"] == "full_moe")
        drqn_prim = f"{moe_r['drqn_primary_rate_pct']:5.1f}%" if moe_r['drqn_primary_rate_pct'] is not None else "N/A"
        esc_comp = f"{occ_r['empty_escape_rate']*100:4.1f}% vs {moe_r['empty_escape_rate']*100:4.1f}%"
        print(f"{scen_id:<22} | {occ_r['intercept_rate']*100:5.2f}% ({occ_r['hits']:3d} hits)   | {moe_r['intercept_rate']*100:5.2f}% ({moe_r['hits']:3d} hits)   | {drqn_prim:<16} | {esc_comp}")
    print("=" * 115 + "\n")

    # 4. Save structured report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "elapsed_seconds": battery_results["elapsed_seconds"],
        "category_summary": cat_summary,
        "scenario_evaluations": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved stress suite report to: %s", out_path)


if __name__ == "__main__":
    main()
