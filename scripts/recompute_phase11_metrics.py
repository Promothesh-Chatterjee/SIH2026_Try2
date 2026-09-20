"""
Recomputes and finalizes the metrics for:
  - 25k Frozen Baseline (experiments/reports/phase11_baseline_25k_frozen_eval.json)
  - Phase 11 Step 27500 Candidate (experiments/checkpoints/phase11_controlled/gate_27500_report.json and experiments/reports/phase11_gate_27500_eval.json)
Prints a clean, complete side-by-side comparison table.
"""

import json
from pathlib import Path
import numpy as np

def compute_metrics_from_raw(data):
    policies = data.get("raw_results", {}).get("policies", {})
    drqn_res = policies.get("drqn", {})
    full_moe_res = policies.get("full_moe", {})
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

    mean_ir = float(np.mean(irs)) if irs else 0.0
    med_ir = float(np.median(irs)) if irs else 0.0
    worst_ir = float(np.min(irs)) if irs else 0.0
    agile_ir = float(np.mean(agile_irs)) if agile_irs else 0.0
    sparse_ir = float(np.mean(sparse_irs)) if sparse_irs else 0.0
    pd = float(drqn_res.get("decision_level_pd", 0.0) or 0.0)
    pfa = float(drqn_res.get("pfa", 0.0) or 0.0)
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    med_lat = float(np.median(latencies)) if latencies else 0.0
    distinct_bands = float(drqn_res.get("distinct_bands", 0.0))
    act_ent = float(drqn_res.get("action_entropy", 0.0))
    mode_ent = float(drqn_res.get("mode_entropy", 0.0))

    composite_score = (
        mean_ir
        + 0.5 * agile_ir
        + 0.5 * sparse_ir
        + 0.25 * worst_ir
        + 0.10 * float(drqn_res.get("discovery_rate", 0.0))
        - 0.0005 * mean_lat
    )

    data["scenarios_evaluated"] = len(scens)
    data["drqn_standalone"] = {
        "mean_ir": mean_ir,
        "median_ir": med_ir,
        "worst_case_ir": worst_ir,
        "agile_ir": agile_ir,
        "sparse_ir": sparse_ir,
        "pd": pd,
        "pfa": pfa,
        "mean_latency_us": mean_lat,
        "median_latency_us": med_lat,
        "mode_fractions": mode_fractions,
        "mode_entropy": mode_ent,
        "action_entropy": act_ent,
        "distinct_bands": distinct_bands,
        "composite_generalization_score": float(composite_score),
    }
    data["full_moe"] = {
        "mean_ir": float(full_moe_res.get("intercept_rate", 0.0)),
        "pd": float(full_moe_res.get("decision_level_pd", 0.0) or 0.0),
        "pfa": float(full_moe_res.get("pfa", 0.0) or 0.0),
        "mean_latency_us": float(full_moe_res.get("avg_intercept_time_us", 0.0) or 0.0),
    }
    return data

def main():
    base_file = Path("experiments/reports/phase11_baseline_25k_frozen_eval.json")
    with open(base_file) as f:
        base_data = json.load(f)
    base_data = compute_metrics_from_raw(base_data)
    with open(base_file, "w") as f:
        json.dump(base_data, f, indent=2)

    cand_file = Path("experiments/reports/phase11_gate_27500_eval.json")
    with open(cand_file) as f:
        cand_data = json.load(f)
    cand_data = compute_metrics_from_raw(cand_data)

    # Compute deltas
    base_m = base_data["drqn_standalone"]
    cand_m = cand_data["drqn_standalone"]
    cand_data["baseline_comparison"] = {
        "delta_mean_ir": cand_m["mean_ir"] - base_m["mean_ir"],
        "delta_median_ir": cand_m["median_ir"] - base_m["median_ir"],
        "delta_worst_case_ir": cand_m["worst_case_ir"] - base_m["worst_case_ir"],
        "delta_agile_ir": cand_m["agile_ir"] - base_m["agile_ir"],
        "delta_sparse_ir": cand_m["sparse_ir"] - base_m["sparse_ir"],
        "delta_pd": cand_m["pd"] - base_m["pd"],
        "delta_pfa": cand_m["pfa"] - base_m["pfa"],
        "delta_mean_latency_us": cand_m["mean_latency_us"] - base_m["mean_latency_us"],
        "delta_median_latency_us": cand_m["median_latency_us"] - base_m["median_latency_us"],
        "delta_distinct_bands": cand_m["distinct_bands"] - base_m["distinct_bands"],
        "delta_composite_score": cand_m["composite_generalization_score"] - base_m["composite_generalization_score"],
    }

    with open(cand_file, "w") as f:
        json.dump(cand_data, f, indent=2)

    gate_file = Path("experiments/checkpoints/phase11_controlled/gate_27500_report.json")
    with open(gate_file, "w") as f:
        json.dump(cand_data, f, indent=2)

    print("\n" + "=" * 115)
    print("  PHASE 11 STAGE 2 — GATE 27,500 vs. AUTHORITATIVE 25K BASELINE")
    print("=" * 115)
    print(f"  Parent Checkpoint : {cand_data.get('parent_checkpoint')} (SHA: {cand_data.get('parent_sha256')[:16]}...)")
    print(f"  Candidate Step    : {cand_data.get('global_step')} (SHA: {cand_data.get('sha256')[:16]}...)")
    print(f"  Scenarios Eval'd  : {cand_data.get('scenarios_evaluated')} fixed held-out validation scenarios")
    print("=" * 115)
    print(f"  {'Metric':<32} | {'25k Frozen Baseline':<20} | {'Phase-11 Step 27,500':<22} | {'Delta':<16}")
    print("-" * 115)

    def row(name, b_val, c_val, fmt, suffix="", higher_better=True):
        delta = c_val - b_val
        arrow = "+" if delta >= 0 else ""
        sym = "[+]" if ((delta > 0 and higher_better) or (delta < 0 and not higher_better)) else ("[-]" if delta != 0 else "[=]")
        b_str = f"{b_val:{fmt}}{suffix}"
        c_str = f"{c_val:{fmt}}{suffix}"
        d_str = f"{arrow}{delta:{fmt}}{suffix} {sym}"
        print(f"  {name:<32} | {b_str:<20} | {c_str:<22} | {d_str:<16}")

    row("Mean Interception Rate (IR)", base_m["mean_ir"]*100, cand_m["mean_ir"]*100, "6.2f", "%", higher_better=True)
    row("Median Interception Rate", base_m["median_ir"]*100, cand_m["median_ir"]*100, "6.2f", "%", higher_better=True)
    row("Worst-Case IR", base_m["worst_case_ir"]*100, cand_m["worst_case_ir"]*100, "6.2f", "%", higher_better=True)
    row("Agile Scenario IR", base_m["agile_ir"]*100, cand_m["agile_ir"]*100, "6.2f", "%", higher_better=True)
    row("Sparse Scenario IR", base_m["sparse_ir"]*100, cand_m["sparse_ir"]*100, "6.2f", "%", higher_better=True)
    row("Decision Pd", base_m["pd"]*100, cand_m["pd"]*100, "6.2f", "%", higher_better=True)
    row("Decision Pfa", base_m["pfa"]*100, cand_m["pfa"]*100, "6.4f", "%", higher_better=False)
    row("Mean Latency Error", base_m["mean_latency_us"], cand_m["mean_latency_us"], "6.1f", " us", higher_better=False)
    row("Median Latency Error", base_m["median_latency_us"], cand_m["median_latency_us"], "6.1f", " us", higher_better=False)
    row("Distinct Bands Visited", base_m["distinct_bands"], cand_m["distinct_bands"], "4.1f", " / 36", higher_better=True)
    row("Action Entropy", base_m["action_entropy"], cand_m["action_entropy"], "6.4f", "", higher_better=True)
    row("Mode Entropy", base_m["mode_entropy"], cand_m["mode_entropy"], "6.4f", "", higher_better=True)
    row("Composite Score", base_m["composite_generalization_score"], cand_m["composite_generalization_score"], "6.4f", "", higher_better=True)

    print("-" * 115)
    base_mf = base_m["mode_fractions"]
    cand_mf = cand_m["mode_fractions"]
    for m_name in ("SHORT", "NORMAL", "LONG", "REVISIT", "PREEMPTIVE"):
        b_pct = base_mf[m_name] * 100
        c_pct = cand_mf[m_name] * 100
        d_pct = c_pct - b_pct
        sign = "+" if d_pct >= 0 else ""
        print(f"  {('Mode: ' + m_name):<32} | {b_pct:5.1f}%              | {c_pct:5.1f}%                | {sign}{d_pct:5.1f}%")

    print("=" * 115)
    print("\nScenario Breakdown (Candidate vs. Baseline IR):")
    base_scens = {s["scenario_id"]: s["intercept_rate"] for s in base_data["raw_results"]["policies"]["drqn"]["scenario_breakdown"]}
    cand_scens = {s["scenario_id"]: s["intercept_rate"] for s in cand_data["raw_results"]["policies"]["drqn"]["scenario_breakdown"]}
    for sid in sorted(cand_scens):
        b_ir = base_scens.get(sid, 0.0) * 100
        c_ir = cand_scens.get(sid, 0.0) * 100
        d_ir = c_ir - b_ir
        arrow = "+" if d_ir >= 0 else ""
        print(f"  {sid:<15} : Base={b_ir:5.1f}% -> Cand={c_ir:5.1f}% (Delta: {arrow}{d_ir:5.1f}%)")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
