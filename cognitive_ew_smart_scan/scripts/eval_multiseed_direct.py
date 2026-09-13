"""Direct Multi-Seed Evaluator for Baseline Hierarchy and Model Candidates.

Directly evaluates seeds 42, 123, 999 on the 4 key policies:
  1. Random
  2. RoundRobin
  3. HighestOccupancy heuristic baseline
  4. Gate-25k-Frozen (operational candidate)
Outputs full statistics into cognitive_ew_smart_scan/reports/benchmark_v2_multiseed_summary.json.
"""

import copy
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("cognitive_ew_smart_scan").resolve()))

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.training.staged_gate_evaluator import StagedGateEvaluator
from cognitive_ew_smart_scan.src.training.val_set import FixedValidationSet
from cognitive_ew_smart_scan.src.data.tsrd_root import resolve_tsrd_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("direct_evaluator")

def evaluate_candidates(seeds=[42, 123, 999]):
    train_cfg_path = Path("cognitive_ew_smart_scan/configs/training_config.yaml")
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    model_cfg_path = Path("cognitive_ew_smart_scan/configs/model_config.yaml")
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    data_dir = resolve_tsrd_root(None, train_cfg)
    frozen_ckpt_path = Path("cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt")
    
    env_cfg = copy.deepcopy(train_cfg.get("env", {}))
    reward_cfg = copy.deepcopy(model_cfg.get("reward", {}))
    full_env_cfg = {**env_cfg, **reward_cfg, "n_bands": 36, "n_modes": 5, "n_actions": 180}
    full_env_cfg.setdefault("belief", {})
    full_env_cfg["belief"]["ema_alpha_miss_confirmed"] = 0.20

    ckpt = torch.load(frozen_ckpt_path, map_location="cpu", weights_only=False)
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()

    val_set = FixedValidationSet(
        data_root=data_dir,
        subset="val",
        mode="stare",
        n_files=10,
        seed=42,
        allow_synthetic_fallback=False,
    )

    policies = ["random", "round_robin", "highest_occupancy", "drqn"]
    scores = {p: [] for p in policies}

    for s in seeds:
        logger.info("Evaluating seed %d ...", s)
        evaluator = StagedGateEvaluator(
            output_dir=Path("cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate"),
            gates=[],
            val_files=val_set.files_used,
            env_config=full_env_cfg,
            model_config=model_cfg,
            train_config=train_cfg,
            seed=s,
            device=torch.device("cpu"),
        )
        bench = evaluator.evaluate_baseline_hierarchy(
            drqn,
            moe=None,
            n_steps=1000,
            policies=policies,
        )
        for p in policies:
            scores[p].append(float(bench["policies"][p]["intercept_rate"] * 100))

    summary = {}
    name_map = {
        "random": "Random",
        "round_robin": "RoundRobin",
        "highest_occupancy": "HighestOccupancy heuristic baseline",
        "drqn": "Gate-25k-Frozen",
    }
    for p, vals in scores.items():
        arr = np.array(vals)
        label = name_map[p]
        summary[label] = {
            "per_seed": {str(seeds[i]): round(float(vals[i]), 2) for i in range(len(seeds))},
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
        }

    out_file = Path("cognitive_ew_smart_scan/reports/benchmark_v2_multiseed_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary to %s", out_file)
    print("\n" + "="*80)
    print("BENCHMARK V2 MULTI-SEED STATISTICAL SUMMARY")
    print("="*80)
    print(f"{'Policy':<38} | {'Mean ± Std':<18} | {'Median':<8} | {'Min':<8} | {'Max':<8}")
    print("-"*80)
    for pol, st in summary.items():
        mean_std = f"{st['mean']:.2f} ± {st['std']:.2f}%"
        print(f"{pol:<38} | {mean_std:<18} | {st['median']:<8.2f} | {st['min']:<8.2f} | {st['max']:<8.2f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    evaluate_candidates()
