"""
Evaluates the Authoritative 25k Frozen Baseline on the 10 fixed validation scenarios.
Records:
  - Mean IR, Median IR, Worst-case IR, Agile IR, Sparse IR
  - Decision Pd, Pfa
  - Median interception latency, Mean interception latency, P95/P99 cycle latency
  - Mode fractions: SHORT, NORMAL, LONG, REVISIT, PREEMPTIVE
  - Action entropy, unique-band coverage
"""

import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import torch
import yaml

from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.training.staged_gate_evaluator import StagedGateEvaluator
from ew_core.training.val_set import FixedValidationSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def evaluate_baseline():
    ckpt_path = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    assert ckpt_path.exists(), f"Missing baseline checkpoint: {ckpt_path}"
    
    # Verify SHA-256
    sha256_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    expected_sha = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
    assert sha256_hash == expected_sha, f"SHA mismatch! Expected {expected_sha}, got {sha256_hash}"
    logger.info("Authoritative Baseline SHA-256 verified: %s", sha256_hash)

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

    evaluator = StagedGateEvaluator(
        output_dir="experiments/reports/phase11",
        gates=[25000],
        val_files=val_set.files_used,
        env_config=env_cfg,
        model_config=model_cfg,
        train_config=train_cfg,
        seed=seed,
        device="cpu",
        parent_checkpoint=str(ckpt_path),
    )

    logger.info("Running baseline evaluation on 10 held-out scenarios...")
    results = evaluator.evaluate_baseline_hierarchy(
        online_drqn=drqn,
        moe=None,
        n_steps=1000,
        policies=["drqn", "full_moe", "random", "round_robin"],
    )

    policies = results.get("policies", {})
    drqn_res = policies.get("drqn", {})
    full_moe_res = policies.get("full_moe", {})

    drqn_scens = results.get("scenario_results", {}).get("drqn", [])
    irs = [r["intercept_rate"] for r in drqn_scens]
    latencies = [r["avg_intercept_time_us"] for r in drqn_scens if r.get("avg_intercept_time_us") is not None]

    agile_ids = ("config_119", "config_241", "config_29", "config_195")
    sparse_ids = ("config_143", "config_119")

    agile_irs = [r["intercept_rate"] for r in drqn_scens if any(r["scenario_id"].startswith(aid) for aid in agile_ids)]
    sparse_irs = [r["intercept_rate"] for r in drqn_scens if any(r["scenario_id"].startswith(sid) for sid in sparse_ids)]

    summary = {
        "checkpoint": str(ckpt_path),
        "sha256": sha256_hash,
        "scenarios_evaluated": len(drqn_scens),
        "drqn_standalone": {
            "mean_ir": float(np.mean(irs)) if irs else 0.0,
            "median_ir": float(np.median(irs)) if irs else 0.0,
            "worst_case_ir": float(np.min(irs)) if irs else 0.0,
            "agile_ir": float(np.mean(agile_irs)) if agile_irs else 0.0,
            "sparse_ir": float(np.mean(sparse_irs)) if sparse_irs else 0.0,
            "pd": float(drqn_res.get("decision_level_pd", 0.0) or 0.0),
            "pfa": float(drqn_res.get("pfa", 0.0) or 0.0),
            "mean_latency_us": float(np.mean(latencies)) if latencies else 0.0,
            "median_latency_us": float(np.median(latencies)) if latencies else 0.0,
            "mode_fractions": {
                "SHORT": float(drqn_res.get("short_fraction", 0.0)),
                "NORMAL": float(drqn_res.get("normal_fraction", 0.0)),
                "LONG": float(drqn_res.get("long_fraction", 0.0)),
                "REVISIT": float(drqn_res.get("revisit_fraction", 0.0)),
                "PREEMPTIVE": float(drqn_res.get("preemptive_fraction", 0.0)),
            },
            "action_entropy": float(drqn_res.get("action_entropy", 0.0)),
            "band_entropy": float(drqn_res.get("band_entropy", 0.0)),
            "distinct_bands": float(drqn_res.get("distinct_bands", 0.0)),
        },
        "full_moe": {
            "mean_ir": float(full_moe_res.get("intercept_rate", 0.0)),
            "pd": float(full_moe_res.get("decision_level_pd", 0.0) or 0.0),
            "pfa": float(full_moe_res.get("pfa", 0.0) or 0.0),
            "mean_latency_us": float(full_moe_res.get("avg_intercept_time_us", 0.0) or 0.0),
        },
        "raw_results": results,
    }

    out_file = Path("experiments/reports/phase11_baseline_25k_frozen_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved baseline evaluation report to %s", out_file)
    
    print("\n" + "=" * 80)
    print("  AUTHORITATIVE 25K FROZEN BASELINE EVALUATION SUMMARY")
    print(f"  Checkpoint: {ckpt_path} (SHA: {sha256_hash[:16]}...)")
    print("=" * 80)
    print(f"  Mean IR         : {summary['drqn_standalone']['mean_ir']*100:6.2f}%")
    print(f"  Median IR       : {summary['drqn_standalone']['median_ir']*100:6.2f}%")
    print(f"  Worst-Case IR   : {summary['drqn_standalone']['worst_case_ir']*100:6.2f}%")
    print(f"  Agile IR        : {summary['drqn_standalone']['agile_ir']*100:6.2f}%")
    print(f"  Sparse IR       : {summary['drqn_standalone']['sparse_ir']*100:6.2f}%")
    print(f"  Decision Pd     : {summary['drqn_standalone']['pd']*100:6.2f}%")
    print(f"  Decision Pfa    : {summary['drqn_standalone']['pfa']*100:6.4f}%")
    print(f"  Mean Latency    : {summary['drqn_standalone']['mean_latency_us']:6.1f} us")
    print(f"  Median Latency  : {summary['drqn_standalone']['median_latency_us']:6.1f} us")
    print(f"  Distinct Bands  : {summary['drqn_standalone']['distinct_bands']:4.1f} / 36")
    print(f"  Mode Fractions  : SHORT={summary['drqn_standalone']['mode_fractions']['SHORT']*100:4.1f}%, NORMAL={summary['drqn_standalone']['mode_fractions']['NORMAL']*100:4.1f}%, LONG={summary['drqn_standalone']['mode_fractions']['LONG']*100:4.1f}%, REVISIT={summary['drqn_standalone']['mode_fractions']['REVISIT']*100:4.1f}%, PREEMPT={summary['drqn_standalone']['mode_fractions']['PREEMPTIVE']*100:4.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_baseline()
