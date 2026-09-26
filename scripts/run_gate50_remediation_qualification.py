#!/usr/bin/env python3
"""Gate-50k Anti-Collapse Remediation: 3,000-Step Controlled Qualification Runner (Phase E).

Lineage & Contracts:
  - Parent Checkpoint: experiments/checkpoints/scheduler_v2_continuation_100k/checkpoint_gate_50000.pt
    SHA-256: f3aab6b824faf8207194594fe52c096ec5b3b00990e37e7c321d4794bfb9d519
  - Frozen Baseline Reference: experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt
    SHA-256: 7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0
  - Execution Horizon: Step 50,000 -> Step 53,000 (3,000 continuation steps)
  - Semantics: In-flight optimizer/target/RNG/global-step continuation with fresh replay
  - Isolated Output Dir: experiments/checkpoints/scheduler_v2_gate50_remediation_r1/
"""

import os
import sys
import json
import copy
import time
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import yaml
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gate50_qualification_runner")

ROOT = Path(__file__).resolve().parent.parent

CANONICAL_GATE25_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
CANONICAL_GATE50_SHA = "f3aab6b824faf8207194594fe52c096ec5b3b00990e37e7c321d4794bfb9d519"

GATE25_PATH = ROOT / "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"
GATE50_PATH = ROOT / "experiments/checkpoints/scheduler_v2_continuation_100k/checkpoint_gate_50000.pt"

TRAIN_CFG_PATH = ROOT / "configs/training_config_gate50_remediation.yaml"
MODEL_CFG_PATH = ROOT / "configs/model_config.yaml"
OUTPUT_DIR = ROOT / "experiments/checkpoints/scheduler_v2_gate50_remediation_r1"


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_preflight_invariants() -> dict:
    logger.info("Verifying Pre-Flight Execution Invariants...")

    # 1. Gate-25k Frozen Baseline Verification
    if not GATE25_PATH.exists():
        raise FileNotFoundError(f"Gate-25k frozen baseline not found at {GATE25_PATH}")
    gate25_sha = compute_sha256(GATE25_PATH)
    if gate25_sha != CANONICAL_GATE25_SHA:
        raise RuntimeError(
            f"FATAL: Gate-25k frozen baseline SHA-256 mismatch!\n"
            f"Expected: {CANONICAL_GATE25_SHA}\nGot:      {gate25_sha}"
        )
    logger.info("Invariant 1 PASS: Gate-25k frozen baseline SHA-256 verified bit-exact: %s", gate25_sha)

    # 2. Gate-50k Candidate Lineage Verification
    if not GATE50_PATH.exists():
        raise FileNotFoundError(f"Gate-50k checkpoint not found at {GATE50_PATH}")
    gate50_sha = compute_sha256(GATE50_PATH)
    if gate50_sha != CANONICAL_GATE50_SHA:
        raise RuntimeError(
            f"FATAL: Gate-50k checkpoint SHA-256 mismatch!\n"
            f"Expected: {CANONICAL_GATE50_SHA}\nGot:      {gate50_sha}"
        )
    logger.info("Invariant 2 PASS: Gate-50k candidate SHA-256 verified bit-exact: %s", gate50_sha)

    # 3. Output Directory Isolation Check
    forbidden = [
        (ROOT / "experiments/checkpoints/production_baseline").resolve(),
        (ROOT / "experiments/checkpoints/scheduler_v2_operational_candidate").resolve(),
    ]
    resolved_out = OUTPUT_DIR.resolve()
    for f_dir in forbidden:
        if resolved_out == f_dir or f_dir in resolved_out.parents:
            raise RuntimeError(f"FATAL: Output dir {resolved_out} inside forbidden baseline dir {f_dir}!")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Invariant 3 PASS: Output dir isolated at %s", resolved_out)

    # 4. Config & Git State
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_sha = "unknown"

    train_cfg_sha = compute_sha256(TRAIN_CFG_PATH)
    model_cfg_sha = compute_sha256(MODEL_CFG_PATH)

    with open(TRAIN_CFG_PATH) as f:
        train_cfg = yaml.safe_load(f)

    # Verify weights_only == False at both levels
    top_wo = train_cfg.get("weights_only")
    sched_wo = train_cfg.get("scheduler", {}).get("weights_only")
    if top_wo is not False or sched_wo is not False:
        raise ValueError(
            f"FATAL: Dual weights_only: false contract violated! top={top_wo}, sched={sched_wo}"
        )
    logger.info("Invariant 4 PASS: Dual weights_only: false verified (in-flight continuation contract active).")

    preflight_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_sha,
        "gate25_frozen_sha256": gate25_sha,
        "gate50_parent_sha256": gate50_sha,
        "train_cfg_sha256": train_cfg_sha,
        "model_cfg_sha256": model_cfg_sha,
        "start_step": 50000,
        "stop_step": 53000,
        "qualification_steps": 3000,
        "output_dir": str(resolved_out),
        "candidate_r1_hyperparameters": {
            "q_reg_coef": train_cfg["scheduler"].get("q_reg_coef"),
            "lambda_entropy": train_cfg["scheduler"].get("lambda_entropy"),
            "band_diversity_penalty_coef": train_cfg["scheduler"].get("band_diversity_penalty_coef"),
            "band_diversity_threshold": train_cfg["scheduler"].get("band_diversity_threshold"),
            "mode_diversity_penalty_coef": train_cfg["scheduler"].get("mode_diversity_penalty_coef"),
            "mode_diversity_threshold": train_cfg["scheduler"].get("mode_diversity_threshold"),
            "mode_collapse_rate_threshold": train_cfg["scheduler"].get("mode_collapse_rate_threshold"),
        },
        "reward_invariants_frozen": train_cfg.get("reward", {}),
    }

    manifest_path = OUTPUT_DIR / "qualification_preflight_manifest.json"
    manifest_path.write_text(json.dumps(preflight_manifest, indent=2), encoding="utf-8")
    logger.info("Saved preflight manifest to %s", manifest_path)

    return preflight_manifest


def evaluate_decision_tree(report_path: Path) -> dict:
    """Evaluate Case A-E decision tree on Gate-53k report."""
    with open(report_path) as f:
        rep = json.load(f)

    # Locate DRQN policy results
    policies = rep.get("policies", {})
    drqn_res = policies.get("drqn", {})
    cd = rep.get("collapse_diagnostics", {})
    metrics = rep.get("metrics", cd.get("metrics", {}))
    scenarios = rep.get("scenarios", {})

    # Extract Key Metrics
    mean_ir = float(drqn_res.get("intercept_rate", metrics.get("mean_ir", 0.0))) * 100.0
    action_entropy = float(drqn_res.get("action_entropy", metrics.get("action_entropy", 0.0)))
    mode_entropy = float(drqn_res.get("mode_entropy", metrics.get("mode_entropy", 0.0)))
    top_band_fraction = float(drqn_res.get("top_band_fraction", metrics.get("top_band_fraction", 0.0)))
    pd_rate = float(drqn_res.get("decision_level_pd", metrics.get("pd", 0.0))) * 100.0
    pfa_rate = float(drqn_res.get("pfa", metrics.get("pfa", 0.0)))

    # Q Metrics
    q_max = float(metrics.get("q_max", rep.get("training_diagnostics", {}).get("q_max", 0.0)))
    q_std = float(metrics.get("q_std", rep.get("training_diagnostics", {}).get("q_std", 0.0)))
    target_online_gap = float(metrics.get("target_online_gap", 0.0))

    # Scenario IRs
    sparse_ids = ["config_143", "config_119", "config_141", "config_117"]
    agile_ids = ["config_241", "config_242", "config_244", "config_245"]

    scen_irs = {}
    if "scenario_irs" in metrics:
        for k, v in metrics["scenario_irs"].items():
            scen_irs[k] = float(v) * 100.0
    else:
        for scen_name, s_data in scenarios.items():
            if isinstance(s_data, dict):
                s_ir = float(s_data.get("drqn", {}).get("intercept_rate", s_data.get("intercept_rate", 0.0)))
                scen_irs[scen_name] = s_ir * 100.0

    sparse_mean_ir = float(metrics.get("sparse_ir", 0.0)) * 100.0 if "sparse_ir" in metrics else (float(np.mean([scen_irs[s] for s in sparse_ids if s in scen_irs])) if scen_irs else 0.0)
    agile_mean_ir = float(metrics.get("agile_ir", 0.0)) * 100.0 if "agile_ir" in metrics else (float(np.mean([scen_irs[s] for s in agile_ids if s in scen_irs])) if scen_irs else 0.0)

    c143_ir = scen_irs.get("config_143", 0.0)
    c119_ir = scen_irs.get("config_119", 0.0)
    c241_ir = scen_irs.get("config_241", 0.0)

    collapse_verdict = str(cd.get("severity", rep.get("policy_collapse", {}).get("severity", "UNKNOWN")))
    checkpoint_tag = rep.get("checkpoint_tag", "candidate")

    # Evaluate Case A criteria
    c1_verdict = collapse_verdict != "CRITICAL"
    c2_tag = checkpoint_tag != "quarantined_collapsed"
    c3_qmax = q_max <= 100.0
    c4_qstd = q_std < 60.0
    c5_gap = target_online_gap < 30.0
    c6_act_ent = action_entropy >= 1.000
    c7_top_band = top_band_fraction < 0.850
    c8_mode_ent = mode_entropy >= 0.400  # or justified
    c9_sparse = sparse_mean_ir >= 50.0
    c10_c143 = c143_ir >= 50.0
    c11_c119 = c119_ir >= 50.0
    c12_c241 = c241_ir >= 50.0
    c13_agile = agile_mean_ir >= 55.0
    c14_pd = pd_rate >= 98.0
    c15_pfa = pfa_rate <= 0.0005

    case_a_passed = all([
        c1_verdict, c2_tag, c3_qmax, c4_qstd, c5_gap,
        c6_act_ent, c7_top_band, (c8_mode_ent or True), # check justification if < 0.4
        c9_sparse, c10_c143, c11_c119, c12_c241, c13_agile,
        c14_pd, c15_pfa
    ])

    if case_a_passed and c8_mode_ent:
        decision_case = "CASE_A_QUALIFIED"
        recommendation = "QUALIFIED: Satisfies all Case A criteria. Recommended for Gate-75k authorization review."
    elif not c1_verdict or not c2_tag:
        decision_case = "CASE_C_STOP"
        recommendation = "CRITICAL COLLAPSE: Stop. Do not continue to Gate-75k. Escalate for architectural review."
    elif c3_qmax and (not c6_act_ent or not c7_top_band or not c8_mode_ent):
        decision_case = "CASE_D_CONTINUE_REMEDIATION"
        recommendation = "Value drift resolved (Qmax <= 100), but mode/band concentration remains. Continue remediation."
    elif not c9_sparse or not c13_agile:
        decision_case = "CASE_B_REQUIRES_REVISION"
        recommendation = "Collapse metrics improved, but sparse/agile performance regressed significantly. Requires revision."
    else:
        decision_case = "CASE_E_REJECT"
        recommendation = "Artificial diversity without operational efficacy. Reject."

    verdict_summary = {
        "decision_case": decision_case,
        "recommendation": recommendation,
        "case_a_criteria_checks": {
            "c1_collapse_not_critical": {"passed": c1_verdict, "value": collapse_verdict, "threshold": "!= CRITICAL"},
            "c2_no_quarantined_collapsed_tag": {"passed": c2_tag, "value": checkpoint_tag, "threshold": "!= quarantined_collapsed"},
            "c3_q_max_le_100": {"passed": c3_qmax, "value": q_max, "threshold": "<= 100.0"},
            "c4_q_std_lt_60": {"passed": c4_qstd, "value": q_std, "threshold": "< 60.0"},
            "c5_target_online_gap_lt_30": {"passed": c5_gap, "value": target_online_gap, "threshold": "< 30.0"},
            "c6_action_entropy_ge_1": {"passed": c6_act_ent, "value": action_entropy, "threshold": ">= 1.000"},
            "c7_top_band_fraction_lt_0_85": {"passed": c7_top_band, "value": top_band_fraction, "threshold": "< 0.850"},
            "c8_mode_entropy_ge_0_40": {"passed": c8_mode_ent, "value": mode_entropy, "threshold": ">= 0.400"},
            "c9_sparse_mean_ir_ge_50": {"passed": c9_sparse, "value": sparse_mean_ir, "threshold": ">= 50.0%"},
            "c10_config_143_ir_ge_50": {"passed": c10_c143, "value": c143_ir, "threshold": ">= 50.0%"},
            "c11_config_119_ir_ge_50": {"passed": c11_c119, "value": c119_ir, "threshold": ">= 50.0%"},
            "c12_config_241_ir_ge_50": {"passed": c12_c241, "value": c241_ir, "threshold": ">= 50.0%"},
            "c13_agile_mean_ir_ge_55": {"passed": c13_agile, "value": agile_mean_ir, "threshold": ">= 55.0%"},
            "c14_pd_ge_98": {"passed": c14_pd, "value": pd_rate, "threshold": ">= 98.0%"},
            "c15_pfa_le_0_0005": {"passed": c15_pfa, "value": pfa_rate, "threshold": "<= 0.0005"},
        },
        "measured_metrics": {
            "mean_ir": mean_ir,
            "sparse_mean_ir": sparse_mean_ir,
            "agile_mean_ir": agile_mean_ir,
            "config_143_ir": c143_ir,
            "config_119_ir": c119_ir,
            "config_241_ir": c241_ir,
            "action_entropy": action_entropy,
            "mode_entropy": mode_entropy,
            "top_band_fraction": top_band_fraction,
            "pd": pd_rate,
            "pfa": pfa_rate,
            "q_max": q_max,
            "q_std": q_std,
            "target_online_gap": target_online_gap,
        }
    }
    return verdict_summary


def main():
    logger.info("=" * 80)
    logger.info("STARTING PHASE E: 3,000-STEP CONTROLLED QUALIFICATION EXPERIMENT (50k -> 53k)")
    logger.info("=" * 80)

    # 1. Preflight Invariants Check
    preflight = verify_preflight_invariants()

    # 2. Launch Training via train_scheduler
    from ew_core.training.train_scheduler import train_scheduler

    logger.info("Launching train_scheduler from step 50,000 to step 53,000...")
    t0 = time.time()
    try:
        train_scheduler(
            model_cfg_path=str(MODEL_CFG_PATH),
            train_cfg_path=str(TRAIN_CFG_PATH),
            output_dir_override=str(OUTPUT_DIR),
            stop_at_step=53000,
            staged_gates=[53000],
            expected_parent_sha=CANONICAL_GATE50_SHA,
            resume_checkpoint=str(GATE50_PATH),
        )
    except Exception as exc:
        logger.error("Qualification run execution failed: %s", exc, exc_info=True)
        raise

    elapsed_s = time.time() - t0
    logger.info("Continuation training step 50,000 -> 53,000 completed in %.1f seconds (%.2f min).", elapsed_s, elapsed_s / 60.0)

    # 3. Post-run Artifact Audit & Decision Tree Evaluation
    report_53k = OUTPUT_DIR / "gate_53000_report.json"
    ckpt_53k = OUTPUT_DIR / "checkpoint_gate_53000.pt"

    if not report_53k.exists():
        # Check quarantine dir in case it collapsed
        q_dir = OUTPUT_DIR / "quarantine"
        q_rep = q_dir / "gate_53000_report.json"
        if q_rep.exists():
            report_53k = q_rep
        else:
            raise FileNotFoundError(f"Neither {report_53k} nor {q_rep} was produced!")

    logger.info("Evaluating Gate-53k report from %s...", report_53k)
    decision = evaluate_decision_tree(report_53k)

    qualification_result = {
        "preflight": preflight,
        "execution": {
            "started_at_utc": preflight["timestamp_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_s,
            "target_steps": 3000,
            "gate_evaluated": 53000,
            "checkpoint_path": str(ckpt_53k) if ckpt_53k.exists() else None,
            "report_path": str(report_53k),
        },
        "decision_tree": decision,
    }

    # Emit final qualification report
    out_qual_path = ROOT / "reports/gate50_remediation_qualification.json"
    out_qual_path.write_text(json.dumps(qualification_result, indent=2), encoding="utf-8")
    logger.info("Saved qualification evaluation to %s", out_qual_path)

    copy_to_out = OUTPUT_DIR / "gate50_remediation_qualification.json"
    copy_to_out.write_text(json.dumps(qualification_result, indent=2), encoding="utf-8")

    logger.info("=" * 80)
    logger.info("PHASE E DECISION: %s", decision["decision_case"])
    logger.info("RECOMMENDATION:  %s", decision["recommendation"])
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
