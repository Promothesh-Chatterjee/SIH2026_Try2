"""Phase B Extension: Forced-Mode Counterfactual Evaluation.

Evaluates each of the 5 canonical dwell modes (SHORT, NORMAL, LONG, REVISIT, PREEMPTIVE)
counterfactually on identical seeds and scenarios with DRQN band selection to answer:
Is LONG dwell intrinsically superior on a time-normalized basis (hits/ms),
or did the learned policy create a self-reinforcing trajectory?

Read-only diagnostic: preserves checkpoint_gate_50000.pt untouched.
Emits reports/gate50_forced_mode_counterfactuals.json.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

# Ensure project root in sys.path
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("ew_core").resolve()))

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    RF_BASE_DWELL_TIME_US,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.baseline_suite import build_baseline
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.training.val_set import FixedValidationSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_forced_mode_counterfactuals")


def main():
    root = Path(".").resolve()
    
    with open(root / "configs/training_config_resume_100k.yaml", "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    with open(root / "configs/model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
        
    ckpt_path = root / "experiments/checkpoints/scheduler_v2_continuation_100k/checkpoint_gate_50000.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    n_bands = CANONICAL_N_BANDS
    n_modes = CANONICAL_N_MODES
    n_actions = n_bands * n_modes
    obs_dim = 360
    
    drqn = DRQNScheduler(
        obs_dim=obs_dim,
        n_bands=n_bands,
        n_actions=n_actions,
        n_modes=n_modes,
        lstm_hidden=int(model_cfg.get("drqn_scheduler", {}).get("lstm_hidden", 64)),
        lstm_layers=int(model_cfg.get("drqn_scheduler", {}).get("lstm_layers", 1)),
    )
    drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()
    
    data_dir = Path("D:/TSRD")
    val_set = FixedValidationSet(
        data_root=data_dir,
        subset="val",
        mode="stare",
        n_files=10,
        seed=42,
        allow_synthetic_fallback=False,
        chunk_mode="first",
    )
    
    n_steps = 1000
    
    # Store results per mode
    mode_aggregate_results = {
        m_idx: {
            "mode_name": DWELL_MODES[m_idx],
            "multiplier": DEFAULT_DWELL_MULTIPLIERS[m_idx],
            "dwell_us": RF_BASE_DWELL_TIME_US * DEFAULT_DWELL_MULTIPLIERS[m_idx],
            "total_steps": 0,
            "total_hits": 0,
            "total_dwell_ms": 0.0,
            "total_rewards": 0.0,
            "scenarios": {},
        }
        for m_idx in range(n_modes)
    }
    
    for item in val_set.files_used:
        file_path = Path(item[0] if isinstance(item, (list, tuple)) else item)
        scen_id = file_path.stem
        records = load_h5_records(
            file_path,
            freq_min_mhz=0.0,
            freq_max_mhz=18000.0,
            time_horizon_us=30000000.0,
            max_pulses=50000,
        )
        if not records:
            continue
            
        logger.info("Evaluating forced-mode counterfactuals on %s...", scen_id)
        
        for m_idx in range(n_modes):
            mode_name = DWELL_MODES[m_idx]
            dwell_mult = DEFAULT_DWELL_MULTIPLIERS[m_idx]
            dwell_us = RF_BASE_DWELL_TIME_US * dwell_mult
            
            env_cfg = copy.deepcopy(train_cfg.get("environment", {}))
            env_cfg["semantic_memory_enabled"] = False
            env = CognitiveRFScanEnv(env_cfg, records=records, seed=42, semantic_memory_path=":memory:")
            obs, _ = env.reset(seed=42)
            
            hidden = drqn.init_hidden(1, "cpu")
            ep_hits = 0
            ep_reward = 0.0
            first_hit_step = None
            
            for st in range(n_steps):
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
                with torch.inference_mode():
                    q_out, _, hidden = drqn(obs_t, hidden)
                    q_vals = q_out[0, -1].cpu().numpy()
                    
                    # Band selected by DRQN's highest-valued band across modes
                    band_max = [np.max(q_vals[b * n_modes : (b + 1) * n_modes]) for b in range(n_bands)]
                    best_band = int(np.argmax(band_max))
                    
                    # Force the chosen mode m_idx
                    forced_action = best_band * n_modes + m_idx
                    
                obs, reward, term, trunc, info = env.step(forced_action)
                hit = bool(info.get("hit", False))
                ep_reward += float(reward)
                if hit:
                    ep_hits += 1
                    if first_hit_step is None:
                        first_hit_step = st
                        
                if term or trunc:
                    break
                    
            fom = env.get_fom()
            steps_done = st + 1
            dwell_ms = (steps_done * dwell_us) / 1000.0
            hits_per_ms = ep_hits / max(1e-6, dwell_ms)
            ir = ep_hits / max(1, steps_done)
            first_hit_latency_us = (first_hit_step * dwell_us) if first_hit_step is not None else None
            
            mode_aggregate_results[m_idx]["total_steps"] += steps_done
            mode_aggregate_results[m_idx]["total_hits"] += ep_hits
            mode_aggregate_results[m_idx]["total_dwell_ms"] += dwell_ms
            mode_aggregate_results[m_idx]["total_rewards"] += ep_reward
            
            mode_aggregate_results[m_idx]["scenarios"][scen_id] = {
                "steps": steps_done,
                "hits": ep_hits,
                "ir": ir,
                "dwell_ms": dwell_ms,
                "hits_per_ms": hits_per_ms,
                "avg_reward": ep_reward / max(1, steps_done),
                "pd": float(fom.get("Pd", 0.0)),
                "pfa": float(fom.get("Pfa", 0.0)),
                "first_hit_latency_us": first_hit_latency_us,
            }

    # Summary table across all 10 scenarios
    summary_table = {}
    for m_idx in range(n_modes):
        agg = mode_aggregate_results[m_idx]
        tot_steps = agg["total_steps"]
        tot_hits = agg["total_hits"]
        tot_dwell_ms = agg["total_dwell_ms"]
        tot_rewards = agg["total_rewards"]
        
        ir = tot_hits / max(1, tot_steps)
        hits_per_ms = tot_hits / max(1e-6, tot_dwell_ms)
        avg_reward = tot_rewards / max(1, tot_steps)
        
        # Scenario averages
        scen_irs = [s["ir"] for s in agg["scenarios"].values()]
        scen_pds = [s["pd"] for s in agg["scenarios"].values()]
        
        # Sparse scenarios (config_143, config_119)
        sparse_irs = [agg["scenarios"][sid]["ir"] for sid in ("config_143", "config_119") if sid in agg["scenarios"]]
        agile_irs = [agg["scenarios"][sid]["ir"] for sid in ("config_29", "config_241", "config_195") if sid in agg["scenarios"]]
        
        summary_table[agg["mode_name"]] = {
            "mode_index": m_idx,
            "multiplier": agg["multiplier"],
            "dwell_duration_us": agg["dwell_us"],
            "mean_ir": float(np.mean(scen_irs)),
            "sparse_ir": float(np.mean(sparse_irs)) if sparse_irs else 0.0,
            "agile_ir": float(np.mean(agile_irs)) if agile_irs else 0.0,
            "mean_pd": float(np.mean(scen_pds)),
            "total_dwell_ms": tot_dwell_ms,
            "time_normalized_efficiency_hits_per_ms": hits_per_ms,
            "avg_reward_per_step": avg_reward,
        }

    output = {
        "analysis": "Forced-Mode Counterfactual Evaluation across 10 Held-Out Scenarios (1,000 steps each)",
        "checkpoint": str(ckpt_path.relative_to(root)),
        "checkpoint_sha256": "f3aab6b824faf8207194594fe52c096ec5b3b00990e37e7c321d4794bfb9d519",
        "methodology": (
            "DRQN band selection was kept identical (argmax over bands), while the dwell mode was counterfactually "
            "forced to each of the 5 canonical modes on identical seeds and scenario records."
        ),
        "summary_table": summary_table,
        "detailed_scenarios": mode_aggregate_results,
        "causal_finding": {
            "is_long_intrinsically_superior_hits_per_ms": bool(
                summary_table["LONG_DWELL"]["time_normalized_efficiency_hits_per_ms"]
                > summary_table["NORMAL_DWELL"]["time_normalized_efficiency_hits_per_ms"]
            ),
            "ratio_long_to_normal_efficiency": (
                summary_table["LONG_DWELL"]["time_normalized_efficiency_hits_per_ms"]
                / max(1e-6, summary_table["NORMAL_DWELL"]["time_normalized_efficiency_hits_per_ms"])
            ),
            "sparse_ir_comparison": {
                m: summary_table[m]["sparse_ir"] for m in summary_table
            },
        },
    }

    out_file = root / "reports/gate50_forced_mode_counterfactuals.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Forced-mode counterfactual evaluation complete. Written to %s", out_file)


if __name__ == "__main__":
    main()
