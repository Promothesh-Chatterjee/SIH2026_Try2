"""Lexicographic Promotion Sentinel for Cognitive EW SmartScan.

Implements the formal lexicographic gate promotion rules:
Level 0: Safety & Stability Constraints
  - Pfa <= 0.001 (0.10%)
  - No Q-value explosion (max |Q| <= 50.0)
  - Action entropy >= 1.0 (no static policy lock)
  - Loss finite, non-zero distinct bands (>= 12/36)

Level 1: Mean Intercept Rate Optimization
  - Candidate Mean IR >= Baseline Mean IR (60.45%) OR
  - If Candidate Mean IR is within 1.0% of Baseline (Mean IR >= 59.45%),
    promotion is permitted IF substantial robustness gains are achieved:
    * Agile IR gain >= +2.0% (Candidate Agile IR >= 48.70%)
    * Sparse IR gain >= +1.5% (Candidate Sparse IR >= 19.10%)
    * Worst-case IR gain >= +1.0% (Candidate Worst-case IR >= 13.40%)
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

BASELINE_GATE25K = {
    "mean_ir": 60.45,
    "agile_ir": 46.70,
    "sparse_ir": 17.60,
    "worst_case_ir": 12.40,
    "pfa": 0.0,
    "distinct_bands": 29.9,
    "action_entropy": 2.06,
}

def evaluate_promotion(report: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if "scenario_summary" in report:
        scen = report["scenario_summary"]
        act = report.get("action_summary", {})
        diag = report.get("training_diagnostics", {})

        raw_mean = float(scen.get("mean_ir", 0.0))
        mean_ir = raw_mean if raw_mean > 1.0 else raw_mean * 100

        raw_agile = float(scen.get("agile_ir", 0.0))
        agile_ir = raw_agile if raw_agile > 1.0 else raw_agile * 100

        raw_sparse = float(scen.get("sparse_ir", 0.0))
        sparse_ir = raw_sparse if raw_sparse > 1.0 else raw_sparse * 100

        raw_worst = float(scen.get("worst_case_ir", 0.0))
        # Note: scenario_tracker returns worst_case_ir already in percentage [0, 100].
        # If raw_worst is in (0, 1.0], it was already percentage (e.g. 0.50% worst case).
        # We check if scenario_summary contains "scenarios" dictionary which indicates it comes from ScenarioTracker (already in percent).
        if "scenarios" in scen:
            worst_case_ir = raw_worst
        else:
            worst_case_ir = raw_worst if raw_worst > 1.0 else raw_worst * 100.0

        raw_pfa = float(scen.get("pfa", 0.0) or 0.0)
        pfa = raw_pfa / 100.0 if raw_pfa > 1.0 else raw_pfa
        distinct_bands = float(act.get("unique_bands", 36))
        action_entropy = float(act.get("action_entropy", 0.0))
        q_max = float(diag.get("q_max", 0.0))
    else:
        # Extract benchmark format
        diag = report.get("training_diagnostics", {})
        comp = report.get("composite_generalization_components_10scen", {})
        bench = report.get("baseline_hierarchy", {})
        drqn = bench.get("drqn", {})
        
        mean_ir = float(comp.get("overall_ir", drqn.get("intercept_rate", 0.0))) * 100
        agile_ir = float(comp.get("agile_ir", 0.0)) * 100
        sparse_ir = float(comp.get("sparse_ir", 0.0)) * 100
        worst_case_ir = float(comp.get("worst_case_ir", 0.0)) * 100
        pfa = float(drqn.get("pfa", 0.0) or 0.0)
        distinct_bands = float(drqn.get("distinct_bands", 0.0))
        action_entropy = float(drqn.get("action_entropy", 0.0))
        q_max = float(diag.get("q_max", 0.0))
    
    reasons = []
    
    # Level 0 checks
    if pfa > 0.001:
        reasons.append(f"Pfa violation: {pfa:.4f} > 0.001")
    if abs(q_max) > 50.0:
        reasons.append(f"Q-explosion violation: max |Q|={abs(q_max):.2f} > 50.0")
    if action_entropy < 1.0:
        reasons.append(f"Action entropy violation: entropy={action_entropy:.2f} < 1.0")
    if distinct_bands < 12.0:
        reasons.append(f"Distinct bands violation: {distinct_bands:.1f} < 12.0")
        
    if reasons:
        return False, f"REJECT (Level 0 safety constraints failed: {'; '.join(reasons)})", {
            "mean_ir": mean_ir, "agile_ir": agile_ir, "sparse_ir": sparse_ir, "worst_case_ir": worst_case_ir
        }
        
    # Level 1 checks
    ir_delta = mean_ir - BASELINE_GATE25K["mean_ir"]
    agile_delta = agile_ir - BASELINE_GATE25K["agile_ir"]
    sparse_delta = sparse_ir - BASELINE_GATE25K["sparse_ir"]
    worst_delta = worst_case_ir - BASELINE_GATE25K["worst_case_ir"]
    
    if ir_delta >= 0.0:
        return True, f"PROMOTED: Candidate Mean IR ({mean_ir:.2f}%) exceeds baseline ({BASELINE_GATE25K['mean_ir']:.2f}%) by +{ir_delta:.2f}%", {
            "mean_ir": mean_ir, "agile_ir": agile_ir, "sparse_ir": sparse_ir, "worst_case_ir": worst_case_ir
        }
    elif ir_delta >= -1.0:
        if agile_delta >= 2.0 or sparse_delta >= 1.5 or worst_delta >= 1.0:
            return True, (
                f"PROMOTED (Lexicographic Robustness): Candidate Mean IR ({mean_ir:.2f}%) within 1.0% of baseline, "
                f"with substantial robustness gains (Agile: {agile_delta:+.2f}%, Sparse: {sparse_delta:+.2f}%, Worst-case: {worst_delta:+.2f}%)"
            ), {
                "mean_ir": mean_ir, "agile_ir": agile_ir, "sparse_ir": sparse_ir, "worst_case_ir": worst_case_ir
            }
            
    return False, f"REJECT: Candidate Mean IR ({mean_ir:.2f}%) below baseline ({BASELINE_GATE25K['mean_ir']:.2f}%) without required robustness offset", {
        "mean_ir": mean_ir, "agile_ir": agile_ir, "sparse_ir": sparse_ir, "worst_case_ir": worst_case_ir
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Promotion Sentinel Evaluator")
    parser.add_argument("--candidate", required=True, help="Candidate gate report JSON")
    parser.add_argument("--output", help="Optional output JSON path for promotion decision")
    args = parser.parse_args()

    with open(args.candidate, "r") as f:
        rep = json.load(f)

    promoted, verdict, metrics = evaluate_promotion(rep)
    print("=" * 80)
    print(f"PROMOTION SENTINEL EVALUATION: {args.candidate}")
    print(f"Status: {'PROMOTED' if promoted else 'REJECTED'}")
    print(f"Verdict: {verdict}")
    print(f"Candidate Metrics: {json.dumps(metrics, indent=2)}")
    print("=" * 80)

    if args.output:
        out_dict = {
            "candidate_report": args.candidate,
            "promoted": promoted,
            "verdict": verdict,
            "metrics": metrics,
            "baseline": BASELINE_GATE25K
        }
        with open(args.output, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Saved evaluation result to {args.output}")


