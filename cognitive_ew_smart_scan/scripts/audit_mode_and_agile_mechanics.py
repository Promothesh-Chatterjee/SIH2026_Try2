"""Diagnostic script to audit mode selection, Q-values, and action distributions
between baseline Gate-25k and Trial C candidate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler

def audit():
    device = torch.device("cpu")
    
    # 1. Load Baseline Model
    base_ckpt_path = Path("cognitive_ew_smart_scan/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    base_model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    base_model.load_state_dict(base_ckpt["state_dict"])
    base_model.eval()

    # 2. Load Candidate Model
    cand_ckpt_path = Path("cognitive_ew_smart_scan/checkpoints/safe_continuation_candidate/checkpoint_gate_27500.pt")
    cand_ckpt = torch.load(cand_ckpt_path, map_location=device, weights_only=False)
    cand_model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    cand_model.load_state_dict(cand_ckpt["online_drqn"])
    cand_model.eval()

    scenarios = [
        ("config_29", "Fast Hopper (Agile)"),
        ("config_241", "Agile Hopper (Agile)"),
        ("config_119", "Sparse Agile (Sparse)"),
        ("config_143", "Sparse Radar (Sparse)"),
        ("config_195", "Dense High-PRF (Dense)"),
    ]

    report = {}

    for scen_id, desc in scenarios:
        h5_path = Path(f"D:/TSRD/stare/val_stare/{scen_id}.h5")
        if not h5_path.exists():
            print(f"Skipping {scen_id}, not found")
            continue
        records = load_h5_records(h5_path)

        scen_report = {"description": desc}

        for model_name, model in [("baseline", base_model), ("candidate", cand_model)]:
            env = CognitiveRFScanEnv(
                config={
                    "n_bands": 36,
                    "dwell_modes": 5,
                    "features_per_band": 10,
                    "ema_alpha": 0.30,
                    "ema_alpha_miss_confirmed": 0.20,
                    "semantic_memory_enabled": False,
                },
                records=records,
                seed=42,
            )
            obs, _ = env.reset(seed=42)
            hx = None
            hits = 0
            actions = []
            modes = []
            bands = []
            mode_q_sums = np.zeros(5, dtype=np.float64)
            mode_q_maxs = np.zeros(5, dtype=np.float64)
            mode_q_counts = 0

            n_steps = 1000
            for step in range(n_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
                with torch.no_grad():
                    q_vals, aux, hx = model(obs_t, hx)
                    q_np = q_vals.squeeze(0).squeeze(0).numpy() # (180,)
                    q_by_mode = q_np.reshape(36, 5) # (36, 5)
                    
                    mode_q_sums += q_by_mode.mean(axis=0)
                    mode_q_maxs = np.maximum(mode_q_maxs, q_by_mode.max(axis=0))
                    mode_q_counts += 1

                    act = int(np.argmax(q_np))

                actions.append(act)
                bands.append(act // 5)
                modes.append(act % 5)

                obs, r, term, trunc, info = env.step(act)
                hits += int(info.get("hit", False))
                if term or trunc:
                    break

            mode_dist = dict(Counter(modes))
            band_dist = dict(Counter(bands))
            scen_report[model_name] = {
                "intercept_rate": hits / float(n_steps),
                "hits": hits,
                "total_steps": n_steps,
                "mode_distribution": {str(k): v for k, v in mode_dist.items()},
                "unique_modes": len(mode_dist),
                "unique_bands": len(band_dist),
                "mean_q_by_mode": (mode_q_sums / mode_q_counts).tolist(),
                "max_q_by_mode": mode_q_maxs.tolist(),
                "top_5_actions": Counter(actions).most_common(5),
            }

        report[scen_id] = scen_report

    out_path = Path("cognitive_ew_smart_scan/reports/diagnostic_mode_agile_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Audit written to:", out_path)

if __name__ == "__main__":
    audit()
