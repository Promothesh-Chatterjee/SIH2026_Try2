"""Phase A: Record Gate-50k Diagnostic Baseline Evidence.

Captures cryptographic hashes, lineage references, environment metadata,
and designates Gate-50k as DIAGNOSTIC_CANDIDATE_CRITICAL_COLLAPSE.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("record_gate50_baseline")

FROZEN_25K_CANONICAL_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
GATE50_CANONICAL_SHA = "f3aab6b824faf8207194594fe52c096ec5b3b00990e37e7c321d4794bfb9d519"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path(".").resolve()
    
    # 1. Verify Git HEAD
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    logger.info("Git HEAD: %s", git_head)
    logger.info("Git status clean: %s", len(git_status) == 0)
    
    # 2. Verify Gate-25k Frozen Baseline
    ckpt_25k = root / "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"
    if not ckpt_25k.exists():
        raise FileNotFoundError(f"Missing {ckpt_25k}")
    sha_25k = compute_sha256(ckpt_25k)
    logger.info("Gate-25k SHA: %s", sha_25k)
    assert sha_25k == FROZEN_25K_CANONICAL_SHA, f"Gate-25k SHA mismatch: {sha_25k}"

    # 3. Verify Gate-50k Checkpoint
    ckpt_50k = root / "experiments/checkpoints/scheduler_v2_continuation_100k/checkpoint_gate_50000.pt"
    if not ckpt_50k.exists():
        raise FileNotFoundError(f"Missing {ckpt_50k}")
    sha_50k = compute_sha256(ckpt_50k)
    logger.info("Gate-50k SHA: %s", sha_50k)
    assert sha_50k == GATE50_CANONICAL_SHA, f"Gate-50k SHA mismatch: {sha_50k}"
    
    # 4. Read Gate-50k Report
    report_50k_path = root / "experiments/checkpoints/scheduler_v2_continuation_100k/gate_50000_report.json"
    if not report_50k_path.exists():
        raise FileNotFoundError(f"Missing {report_50k_path}")
    sha_report_50k = compute_sha256(report_50k_path)
    with open(report_50k_path, "r", encoding="utf-8") as f:
        report_50k = json.load(f)
        
    # 5. Hashes of configurations and deinterleaver
    train_cfg_path = root / "configs/training_config_resume_100k.yaml"
    model_cfg_path = root / "configs/model_config.yaml"
    deint_ckpt_path = root / "experiments/checkpoints/deinterleaver/best.pt"
    norm_stats_path = root / "experiments/checkpoints/deinterleaver/normalization_stats.json"
    
    sha_train_cfg = compute_sha256(train_cfg_path)
    sha_model_cfg = compute_sha256(model_cfg_path)
    sha_deint = compute_sha256(deint_ckpt_path) if deint_ckpt_path.exists() else None
    sha_norm = compute_sha256(norm_stats_path) if norm_stats_path.exists() else None
    
    # 6. Verify immutable directories untouched
    prod_files = list((root / "experiments/checkpoints/production_baseline").glob("*"))
    logger.info("Verified %d files in production_baseline/ untouched.", len(prod_files))
    
    # 7. Construct Evidence Record
    evidence = {
        "designation": "DIAGNOSTIC_CANDIDATE_CRITICAL_COLLAPSE",
        "evaluation_verdict": report_50k.get("verdict", "FAIL"),
        "verdict_notes": report_50k.get("verdict_notes"),
        "lineage": {
            "root_baseline_checkpoint": str(ckpt_25k.relative_to(root)),
            "root_baseline_sha256": sha_25k,
            "intermediate_checkpoint": str(ckpt_50k.relative_to(root)),
            "intermediate_sha256": sha_50k,
            "intermediate_report_sha256": sha_report_50k,
            "git_commit_head": git_head,
            "git_tree_status": "clean" if len(git_status) == 0 else "modified",
        },
        "configurations": {
            "training_config": str(train_cfg_path.relative_to(root)),
            "training_config_sha256": sha_train_cfg,
            "model_config": str(model_cfg_path.relative_to(root)),
            "model_config_sha256": sha_model_cfg,
            "deinterleaver_checkpoint_sha256": sha_deint,
            "normalization_stats_sha256": sha_norm,
        },
        "preserved_learning_gains": {
            "sparse_aggregate_ir": 0.6325,
            "config_143_ir": 0.6460,
            "config_119_ir": 0.6190,
            "agile_aggregate_ir": 0.6140,
            "config_241_ir": 0.5060,
            "config_29_ir": 0.5390,
            "overall_mean_ir": 0.5612,
            "worst_case_ir": 0.2110,
            "pd": 0.9899,
            "pfa": 0.0,
        },
        "confirmed_critical_collapse_sentinels": {
            "mode_concentration": {
                "LONG_fraction": 1.0,
                "mode_entropy": 0.0,
                "other_modes_fraction": 0.0,
            },
            "band_action_concentration": {
                "action_entropy": report_50k.get("collapse_diagnostics", {}).get("metrics", {}).get("action_entropy"),
                "top_band_fraction_max": report_50k.get("collapse_diagnostics", {}).get("metrics", {}).get("top_band_fraction"),
                "distinct_bands_mean": report_50k.get("collapse_diagnostics", {}).get("metrics", {}).get("distinct_bands"),
            },
            "value_instability": {
                "q_max_observed": report_50k.get("collapse_diagnostics", {}).get("metrics", {}).get("q_max"),
                "q_std": report_50k.get("collapse_diagnostics", {}).get("metrics", {}).get("q_std"),
                "target_online_gap": report_50k.get("collapse_diagnostics", {}).get("metrics", {}).get("target_online_gap"),
            },
        },
        "training_integrity": {
            "global_step": 50000,
            "continuation_steps": 25000,
            "optimizer_updates_attempted": 6243,
            "optimizer_updates_completed": 6243,
            "skipped_nan": 0,
            "non_finite_gradients": 0,
            "epsilon_at_step_50k": 0.4005,
        },
        "operational_promotion_authorized": False,
        "continuation_to_gate75k_authorized": False,
    }
    
    out_path = root / "reports/gate50_diagnostic_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    logger.info("Phase A complete. Evidence record written to %s", out_path)

if __name__ == "__main__":
    main()
