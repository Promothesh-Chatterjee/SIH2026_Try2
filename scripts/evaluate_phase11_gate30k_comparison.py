"""
Analyzes Gate 30,000 evaluation results and prints the direct side-by-side comparison
table against the Authoritative 25k Frozen Baseline, as well as the 27,500 candidate.
Evaluates the Decision Tree for Gate 30k.
"""

import json
from pathlib import Path
import numpy as np

def main():
    base_file = Path("experiments/reports/phase11_baseline_25k_frozen_eval.json")
    with open(base_file) as f:
        base_data = json.load(f)
    base_drqn = base_data["drqn_standalone"]

    gate275_file = Path("experiments/reports/phase11_gate_27500_eval.json")
    with open(gate275_file) as f:
        gate275_data = json.load(f)
    g275_drqn = gate275_data["drqn_standalone"]

    gate30k_file = Path("experiments/checkpoints/phase11_controlled/gate_30000_report.json")
    with open(gate30k_file) as f:
        gate30k_data = json.load(f)

    # Extract scenario results
    policies = gate30k_data.get("baseline_hierarchy", {})
    drqn_res = policies.get("drqn", {})
    scens = drqn_res.get("scenario_breakdown", [])

    irs = [s["intercept_rate"] for s in scens]
    latencies = [s["avg_intercept_time_us"] for s in scens if s.get("avg_intercept_time_us") is not None]

    agile_ids = ("config_119", "config_241", "config_29", "config_195")
    sparse_ids = ("config_143", "config_119")

    agile_irs = [s["intercept_rate"] for s in scens if any(s["scenario_id"].startswith(aid) for aid in agile_ids)]
    sparse_irs = [s["intercept_rate"] for s in scens if any(s["scenario_id"].startswith(sid) for sid in sparse_ids)]

    mode_fractions = {
        "SHORT": float(drqn_res.get("short_fraction", 0.0)),
        "NORMAL": float(drqn_res.get("normal_fraction", 0.0)),
        "LONG": float(drqn_res.get("long_fraction", 0.0)),
        "REVISIT": float(drqn_res.get("revisit_fraction", 0.0)),
        "PREEMPTIVE": float(drqn_res.get("preemptive_fraction", 0.0)),
    }

    cand_mean_ir = float(np.mean(irs)) if irs else float(drqn_res.get("intercept_rate", 0.0))
    cand_med_ir = float(np.median(irs)) if irs else 0.0
    cand_worst_ir = float(np.min(irs)) if irs else 0.0
    cand_agile_ir = float(np.mean(agile_irs)) if agile_irs else 0.0
    cand_sparse_ir = float(np.mean(sparse_irs)) if sparse_irs else 0.0
    cand_pd = float(drqn_res.get("decision_level_pd", 0.0) or 0.0)
    cand_pfa = float(drqn_res.get("pfa", 0.0) or 0.0)
    cand_mean_lat = float(np.mean(latencies)) if latencies else 0.0
    cand_med_lat = float(np.median(latencies)) if latencies else 0.0
    cand_distinct_bands = float(drqn_res.get("distinct_bands", 0.0))
    cand_act_ent = float(drqn_res.get("action_entropy", 0.0))
    cand_mode_ent = float(drqn_res.get("mode_entropy", 0.0))

    composite_score = (
        cand_mean_ir
        + 0.5 * cand_agile_ir
        + 0.5 * cand_sparse_ir
        + 0.25 * cand_worst_ir
        + 0.10 * float(drqn_res.get("discovery_rate", 0.0))
        - 0.0005 * cand_mean_lat
    )

    print("\n" + "=" * 135)
    print("  PHASE 11 GATE 30,000 vs. 27,500 CANDIDATE vs. 25K FROZEN BASELINE")
    print("=" * 135)
    print(f"  Parent Checkpoint : experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    print(f"  Parent SHA-256    : 7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0")
    print(f"  Candidate Step    : 30,000 ({len(scens)} held-out validation scenarios)")
    print("=" * 135)
    print(f"  {'Metric':<30} | {'25k Baseline':<15} | {'27.5k Step':<15} | {'30k Step':<15} | {'Delta vs 25k':<18} | {'Gate 30k Spec'}")
    print("-" * 135)

    def row(name, b_val, g275_val, c_val, fmt, suffix="", higher_better=True, req_str=""):
        delta = c_val - b_val
        arrow = "+" if delta >= 0 else ""
        sym = "[+]" if ((delta > 0 and higher_better) or (delta < 0 and not higher_better)) else ("[-]" if delta != 0 else "[=]")
        b_str = f"{b_val:{fmt}}{suffix}"
        g_str = f"{g275_val:{fmt}}{suffix}"
        c_str = f"{c_val:{fmt}}{suffix}"
        d_str = f"{arrow}{delta:{fmt}}{suffix} {sym}"
        print(f"  {name:<30} | {b_str:<15} | {g_str:<15} | {c_str:<15} | {d_str:<18} | {req_str}")

    row("Mean Interception Rate (IR)", base_drqn["mean_ir"]*100, g275_drqn["mean_ir"]*100, cand_mean_ir*100, "6.2f", "%", higher_better=True, req_str=">= 63.44%")
    row("Median Interception Rate", base_drqn["median_ir"]*100, g275_drqn["median_ir"]*100, cand_med_ir*100, "6.2f", "%", higher_better=True, req_str=">= 62.60%")
    row("Worst-Case IR", base_drqn["worst_case_ir"]*100, g275_drqn["worst_case_ir"]*100, cand_worst_ir*100, "6.2f", "%", higher_better=True, req_str=">= 17.10%")
    row("Agile Scenario IR", base_drqn["agile_ir"]*100, g275_drqn["agile_ir"]*100, cand_agile_ir*100, "6.2f", "%", higher_better=True, req_str=">= 53.10%")
    row("Sparse Scenario IR", base_drqn["sparse_ir"]*100, g275_drqn["sparse_ir"]*100, cand_sparse_ir*100, "6.2f", "%", higher_better=True, req_str=">= 26.60%")
    row("Decision Pd", base_drqn["pd"]*100, g275_drqn["pd"]*100, cand_pd*100, "6.2f", "%", higher_better=True, req_str=">= 99.86%")
    row("Decision Pfa", base_drqn["pfa"]*100, g275_drqn["pfa"]*100, cand_pfa*100, "6.4f", "%", higher_better=False, req_str="= 0.0000%")
    row("Mean Latency Error", base_drqn["mean_latency_us"], g275_drqn["mean_latency_us"], cand_mean_lat, "6.1f", " us", higher_better=False, req_str="<= 305.45 us")
    row("Median Latency Error", base_drqn["median_latency_us"], g275_drqn["median_latency_us"], cand_med_lat, "6.1f", " us", higher_better=False, req_str="<= 263.11 us")
    row("Distinct Bands Visited", base_drqn["distinct_bands"], g275_drqn["distinct_bands"], cand_distinct_bands, "4.1f", " / 36", higher_better=True, req_str=">= 18.0 / 36")
    row("Action Entropy", base_drqn["action_entropy"], g275_drqn["action_entropy"], cand_act_ent, "6.4f", "", higher_better=True, req_str="Monitor")
    row("Mode Entropy", base_drqn["mode_entropy"], g275_drqn["mode_entropy"], cand_mode_ent, "6.4f", "", higher_better=True, req_str="Monitor")
    row("Composite Score", base_drqn["composite_generalization_score"], g275_drqn["composite_generalization_score"], composite_score, "6.4f", "", higher_better=True, req_str="Monitor")

    print("-" * 135)
    base_mf = base_drqn["mode_fractions"]
    g275_mf = g275_drqn["mode_fractions"]
    for m in ("SHORT", "NORMAL", "LONG", "REVISIT", "PREEMPTIVE"):
        b_pct = base_mf[m] * 100
        g_pct = g275_mf[m] * 100
        c_pct = mode_fractions[m] * 100
        d_pct = c_pct - b_pct
        sign = "+" if d_pct >= 0 else ""
        print(f"  {('Mode: ' + m):<30} | {b_pct:5.1f}%          | {g_pct:5.1f}%          | {c_pct:5.1f}%          | {sign}{d_pct:5.1f}%             |")

    print("=" * 135)
    print("\nScenario Breakdown Comparison (25k Baseline vs 27.5k vs 30k IR):")
    base_scens = {s["scenario_id"]: s["intercept_rate"] for s in base_data["raw_results"]["policies"]["drqn"]["scenario_breakdown"]}
    g275_scens = {s["scenario_id"]: s["intercept_rate"] for s in gate275_data["raw_results"]["policies"]["drqn"]["scenario_breakdown"]}
    c30k_scens = {s["scenario_id"]: s["intercept_rate"] for s in scens}
    for sid in sorted(c30k_scens):
        b_ir = base_scens.get(sid, 0.0) * 100
        g_ir = g275_scens.get(sid, 0.0) * 100
        c_ir = c30k_scens.get(sid, 0.0) * 100
        d_ir = c_ir - b_ir
        arrow = "+" if d_ir >= 0 else ""
        print(f"  {sid:<15} : 25k={b_ir:5.1f}%  ->  27.5k={g_ir:5.1f}%  ->  30k={c_ir:5.1f}% (vs 25k: {arrow}{d_ir:5.1f}%)")
    print("=" * 135)

    # Decision tree evaluation
    collapse_diag = gate30k_data.get("collapse_diagnostics", {})
    print(f"\nGate 30k Verdict: {gate30k_data.get('verdict')} ({gate30k_data.get('verdict_notes')})")
    print(f"Collapse Severity: {collapse_diag.get('severity')} | Tag: {collapse_diag.get('tag')}")
    if collapse_diag.get("reasons"):
        print(f"Collapse Reasons: {collapse_diag.get('reasons')}")

    # Save summary json
    summary = {
        "gate": 30000,
        "global_step": 30000,
        "checkpoint_file": "experiments/checkpoints/phase11_controlled/checkpoint_phase11_step_30000.pt",
        "parent_checkpoint": "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt",
        "parent_sha256": "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
        "scenarios_evaluated": len(scens),
        "drqn_standalone": {
            "mean_ir": cand_mean_ir,
            "median_ir": cand_med_ir,
            "worst_case_ir": cand_worst_ir,
            "agile_ir": cand_agile_ir,
            "sparse_ir": cand_sparse_ir,
            "pd": cand_pd,
            "pfa": cand_pfa,
            "mean_latency_us": cand_mean_lat,
            "median_latency_us": cand_med_lat,
            "mode_fractions": mode_fractions,
            "mode_entropy": cand_mode_ent,
            "action_entropy": cand_act_ent,
            "distinct_bands": cand_distinct_bands,
            "composite_generalization_score": float(composite_score),
        },
        "baseline_comparison": {
            "delta_mean_ir": cand_mean_ir - base_drqn["mean_ir"],
            "delta_median_ir": cand_med_ir - base_drqn["median_ir"],
            "delta_worst_case_ir": cand_worst_ir - base_drqn["worst_case_ir"],
            "delta_agile_ir": cand_agile_ir - base_drqn["agile_ir"],
            "delta_sparse_ir": cand_sparse_ir - base_drqn["sparse_ir"],
            "delta_pd": cand_pd - base_drqn["pd"],
            "delta_pfa": cand_pfa - base_drqn["pfa"],
            "delta_mean_latency_us": cand_mean_lat - base_drqn["mean_latency_us"],
            "delta_median_latency_us": cand_med_lat - base_drqn["median_latency_us"],
            "delta_distinct_bands": cand_distinct_bands - base_drqn["distinct_bands"],
            "delta_composite_score": composite_score - base_drqn["composite_generalization_score"],
        },
        "scenario_comparison": {
            sid: {
                "base_ir": base_scens.get(sid, 0.0),
                "step27500_ir": g275_scens.get(sid, 0.0),
                "step30000_ir": c30k_scens.get(sid, 0.0),
                "delta_vs_25k": c30k_scens.get(sid, 0.0) - base_scens.get(sid, 0.0),
            }
            for sid in c30k_scens
        },
        "gate_pass": {
            "mean_ir_pass": bool(cand_mean_ir >= base_drqn["mean_ir"]),
            "worst_case_ir_pass": bool(cand_worst_ir >= base_drqn["worst_case_ir"]),
            "agile_ir_pass": bool(cand_agile_ir >= base_drqn["agile_ir"]),
            "sparse_ir_pass": bool(cand_sparse_ir >= base_drqn["sparse_ir"]),
            "pfa_pass": bool(cand_pfa == 0.0),
            "pd_pass": bool(cand_pd >= base_drqn["pd"] - 0.001),
            "latency_pass": bool(cand_mean_lat <= base_drqn["mean_latency_us"]),
        }
    }
    out_file = Path("experiments/reports/phase11_gate_30000_eval.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved Gate 30k summary report to: {out_file}\n")

if __name__ == "__main__":
    main()
