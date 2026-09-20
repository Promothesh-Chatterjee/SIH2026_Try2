"""
Evaluates the Phase 11 Gate 27,500 candidate checkpoint on the 10 fixed validation scenarios.
Generates:
  - experiments/checkpoints/phase11_controlled/gate_27500_report.json
  - experiments/reports/phase11_gate_27500_eval.json
Prints direct side-by-side comparison with the authoritative 25k frozen baseline.
"""

import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import torch
import yaml

from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.training.staged_gate_evaluator import StagedGateEvaluator, coerce
from ew_core.training.val_set import FixedValidationSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def evaluate_gate_27500():
    ckpt_path = Path("experiments/checkpoints/phase11_controlled/checkpoint_phase11_step_27500.pt")
    assert ckpt_path.exists(), f"Missing gate checkpoint: {ckpt_path}"
    
    sha256_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    logger.info("Phase 11 Step 27500 Checkpoint SHA-256: %s", sha256_hash)

    # Load baseline reference
    baseline_file = Path("experiments/reports/phase11_baseline_25k_frozen_eval.json")
    assert baseline_file.exists(), f"Missing baseline file: {baseline_file}"
    with open(baseline_file) as f:
        baseline_eval = json.load(f)
    base_drqn = baseline_eval["drqn_standalone"]

    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/training_phase11_controlled.yaml") as f:
        train_cfg = yaml.safe_load(f)

    env_cfg = train_cfg.get("environment", {})
    val_cfg = train_cfg.get("validation", {})
    data_dir = "D:/TSRD"
    seed = int(train_cfg.get("seed", 42))

    val_set = FixedValidationSet(
        data_root=data_dir,
        subset=str(val_cfg.get("subset", "val")),
        mode="stare",
        n_files=int(val_cfg.get("n_files", 10)),
        seed=seed,
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)),
        allow_synthetic_fallback=False,
    )

    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    drqn.load_state_dict(state, strict=True)
    drqn.eval()

    output_dir = Path("experiments/checkpoints/phase11_controlled")
    evaluator = StagedGateEvaluator(
        output_dir=output_dir,
        gates=[27500],
        val_files=val_set.files_used,
        env_config=env_cfg,
        model_config=model_cfg,
        train_config=train_cfg,
        seed=seed,
        device="cpu",
        parent_checkpoint="experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt",
    )

    logger.info("Running Gate 27500 evaluation on 10 held-out scenarios...")
    bench = evaluator.evaluate_baseline_hierarchy(
        online_drqn=drqn,
        moe=None,
        n_steps=1000,
        policies=["drqn", "full_moe", "random", "round_robin"],
    )

    policies = bench.get("policies", {})
    drqn_res = policies.get("drqn", {})
    full_moe_res = policies.get("full_moe", {})

    drqn_scens = bench.get("scenario_results", {}).get("drqn", [])
    irs = [r["intercept_rate"] for r in drqn_scens]
    latencies = [r["avg_intercept_time_us"] for r in drqn_scens if r.get("avg_intercept_time_us") is not None]

    agile_ids = ("config_119", "config_241", "config_29", "config_195")
    sparse_ids = ("config_143", "config_119")

    agile_irs = [r["intercept_rate"] for r in drqn_scens if any(r["scenario_id"].startswith(aid) for aid in agile_ids)]
    sparse_irs = [r["intercept_rate"] for r in drqn_scens if any(r["scenario_id"].startswith(sid) for sid in sparse_ids)]

    cand_mean_ir = float(np.mean(irs)) if irs else 0.0
    cand_med_ir = float(np.median(irs)) if irs else 0.0
    cand_worst_ir = float(np.min(irs)) if irs else 0.0
    cand_agile_ir = float(np.mean(agile_irs)) if agile_irs else 0.0
    cand_sparse_ir = float(np.mean(sparse_irs)) if sparse_irs else 0.0
    cand_pd = float(drqn_res.get("decision_level_pd", 0.0) or 0.0)
    cand_pfa = float(drqn_res.get("pfa", 0.0) or 0.0)
    cand_mean_lat = float(np.mean(latencies)) if latencies else 0.0
    cand_med_lat = float(np.median(latencies)) if latencies else 0.0
    cand_bands = float(drqn_res.get("distinct_bands", 0.0))
    cand_act_ent = float(drqn_res.get("action_entropy", 0.0))
    cand_mode_ent = float(drqn_res.get("mode_entropy", 0.0))

    mode_fractions = {
        "SHORT": float(drqn_res.get("short_fraction", 0.0)),
        "NORMAL": float(drqn_res.get("normal_fraction", 0.0)),
        "LONG": float(drqn_res.get("long_fraction", 0.0)),
        "REVISIT": float(drqn_res.get("revisit_fraction", 0.0)),
        "PREEMPTIVE": float(drqn_res.get("preemptive_fraction", 0.0)),
    }

    # Evaluate policy collapse detector
    collapse_diag = evaluator.collapse_detector.evaluate_eval_run(
        step=27500,
        distinct_bands=cand_bands,
        top_band_fraction=0.0,
        top_action_fraction=0.0,
        action_entropy=cand_act_ent,
        scenario_irs={s.get("scenario_id", f"scen_{i}"): float(s.get("intercept_rate", 0.0)) for i, s in enumerate(drqn_scens)},
        agile_ir=cand_agile_ir,
        sparse_ir=cand_sparse_ir,
        q_max=None,
        q_std=None,
        td_error_p90=None,
        pd=cand_pd,
        pfa=cand_pfa,
        latency_us=cand_mean_lat,
        mode_entropy=cand_mode_ent,
        mode_fractions=mode_fractions,
    )

    # Composite score
    composite_score = (
        cand_mean_ir
        + 0.5 * cand_agile_ir
        + 0.5 * cand_sparse_ir
        + 0.25 * cand_worst_ir
        + 0.10 * float(drqn_res.get("discovery_rate", 0.0))
        - 0.0005 * cand_mean_lat
    )

    report = {
        "gate": 27500,
        "global_step": 27500,
        "checkpoint_file": str(ckpt_path),
        "sha256": sha256_hash,
        "parent_checkpoint": "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt",
        "parent_sha256": "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
        "scenarios_evaluated": len(drqn_scens),
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
            "distinct_bands": cand_bands,
            "composite_generalization_score": float(composite_score),
        },
        "full_moe": {
            "mean_ir": float(full_moe_res.get("intercept_rate", 0.0)),
            "pd": float(full_moe_res.get("decision_level_pd", 0.0) or 0.0),
            "pfa": float(full_moe_res.get("pfa", 0.0) or 0.0),
            "mean_latency_us": float(full_moe_res.get("avg_intercept_time_us", 0.0) or 0.0),
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
            "delta_distinct_bands": cand_bands - base_drqn["distinct_bands"],
        },
        "collapse_diagnostics": collapse_diag.to_dict(),
        "raw_results": bench,
    }

    # Save to gate report and report dir
    gate_report_path = output_dir / "gate_27500_report.json"
    with open(gate_report_path, "w", encoding="utf-8") as f:
        json.dump(coerce(report), f, indent=2)
    logger.info("Saved gate report to %s", gate_report_path)

    summary_report_path = Path("experiments/reports/phase11_gate_27500_eval.json")
    with open(summary_report_path, "w", encoding="utf-8") as f:
        json.dump(coerce(report), f, indent=2)
    logger.info("Saved summary report to %s", summary_report_path)

    # Print Comparison Table
    print("\n" + "=" * 115)
    print("  PHASE 11 GATE 27,500 vs. FROZEN 25K BASELINE COMPARISON TABLE")
    print("=" * 115)
    print(f"  {'Metric':<30} | {'25k Frozen Baseline':<20} | {'Phase-11 Step 27,500':<22} | {'Delta':<15}")
    print("-" * 115)

    def row(name, base_val, cand_val, fmt, suffix="", higher_better=True):
        delta = cand_val - base_val
        arrow = "+" if delta >= 0 else ""
        sym = "▲" if ((delta > 0 and higher_better) or (delta < 0 and not higher_better)) else ("▼" if delta != 0 else "=")
        b_str = f"{base_val:{fmt}}{suffix}"
        c_str = f"{cand_val:{fmt}}{suffix}"
        d_str = f"{arrow}{delta:{fmt}}{suffix} {sym}"
        print(f"  {name:<30} | {b_str:<20} | {c_str:<22} | {d_str:<15}")

    row("Mean Interception Rate (IR)", base_drqn["mean_ir"]*100, cand_mean_ir*100, "6.2f", "%", higher_better=True)
    row("Median Interception Rate", base_drqn["median_ir"]*100, cand_med_ir*100, "6.2f", "%", higher_better=True)
    row("Worst-Case IR", base_drqn["worst_case_ir"]*100, cand_worst_ir*100, "6.2f", "%", higher_better=True)
    row("Agile Scenario IR", base_drqn["agile_ir"]*100, cand_agile_ir*100, "6.2f", "%", higher_better=True)
    row("Sparse Scenario IR", base_drqn["sparse_ir"]*100, cand_sparse_ir*100, "6.2f", "%", higher_better=True)
    row("Decision Pd", base_drqn["pd"]*100, cand_pd*100, "6.2f", "%", higher_better=True)
    row("Decision Pfa", base_drqn["pfa"]*100, cand_pfa*100, "6.4f", "%", higher_better=False)
    row("Mean Latency Error", base_drqn["mean_latency_us"], cand_mean_lat, "6.1f", " us", higher_better=False)
    row("Median Latency Error", base_drqn["median_latency_us"], cand_med_lat, "6.1f", " us", higher_better=False)
    row("Distinct Bands Visited", base_drqn["distinct_bands"], cand_bands, "4.1f", " / 36", higher_better=True)
    row("Action Entropy", base_drqn["action_entropy"], cand_act_ent, "6.4f", "", higher_better=True)
    
    print("-" * 115)
    base_mf = base_drqn["mode_fractions"]
    print(f"  {'Mode: SHORT':<30} | {base_mf['SHORT']*100:5.1f}%              | {mode_fractions['SHORT']*100:5.1f}%                | {(mode_fractions['SHORT']-base_mf['SHORT'])*100:+5.1f}%")
    print(f"  {'Mode: NORMAL':<30} | {base_mf['NORMAL']*100:5.1f}%              | {mode_fractions['NORMAL']*100:5.1f}%                | {(mode_fractions['NORMAL']-base_mf['NORMAL'])*100:+5.1f}%")
    print(f"  {'Mode: LONG':<30} | {base_mf['LONG']*100:5.1f}%              | {mode_fractions['LONG']*100:5.1f}%                | {(mode_fractions['LONG']-base_mf['LONG'])*100:+5.1f}%")
    print(f"  {'Mode: REVISIT':<30} | {base_mf['REVISIT']*100:5.1f}%              | {mode_fractions['REVISIT']*100:5.1f}%                | {(mode_fractions['REVISIT']-base_mf['REVISIT'])*100:+5.1f}%")
    print(f"  {'Mode: PREEMPTIVE':<30} | {base_mf['PREEMPTIVE']*100:5.1f}%              | {mode_fractions['PREEMPTIVE']*100:5.1f}%                | {(mode_fractions['PREEMPTIVE']-base_mf['PREEMPTIVE'])*100:+5.1f}%")
    print("=" * 115)
    print(f"  Collapse Severity: {collapse_diag.severity.value} | Tag: {collapse_diag.tag}")
    if collapse_diag.reasons:
        print(f"  Reasons: {collapse_diag.reasons}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    evaluate_gate_27500()
