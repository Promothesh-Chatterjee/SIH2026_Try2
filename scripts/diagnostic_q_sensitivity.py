"""
Diagnostic 6: Feature Sensitivity Audit of the 25k Frozen Baseline.
Evaluates how the 25k DRQN scheduler responds to perturbations of belief features:
  - revisit_age (feature 4)
  - agility (feature 8)
  - miss_rate (feature 2)
  - uncertainty (feature 3)
  - priority (feature 9)
  - occupancy (feature 0)
Measures:
  - Mean Delta Q across all 180 actions
  - Max Delta Q on the target band's actions
  - Delta Q for each dwell mode (SHORT, NORMAL, LONG, REVISIT, PREEMPTIVE)
  - Action change frequency (does the network switch its selected action?)
"""

import json
from pathlib import Path
import numpy as np
import torch
import yaml

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, DWELL_MODES
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.training.val_set import FixedValidationSet

def run_sensitivity_audit():
    ckpt_path = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    
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

    # Load 25k frozen DRQN
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    drqn.load_state_dict(state, strict=True)
    drqn.eval()

    # Collect sample observations across target scenarios
    target_scenarios = ["config_119", "config_143", "config_241", "config_29"]
    freq_min = float(env_cfg.get("freq_min_mhz", 0.0))
    freq_max = float(env_cfg.get("freq_max_mhz", 18000.0))
    time_horizon = float(env_cfg.get("time_horizon_us", 0.0)) or None
    max_pulses = int(env_cfg.get("max_pulses", 50000))

    collected_obs = []
    for item in val_set.files_used:
        fpath = Path(item[0] if isinstance(item, (list, tuple)) else item)
        if fpath.stem in target_scenarios:
            recs = load_h5_records(fpath, freq_min_mhz=freq_min, freq_max_mhz=freq_max, time_horizon_us=time_horizon, max_pulses=max_pulses)
            val_env_cfg = copy_cfg = dict(env_cfg)
            copy_cfg["semantic_memory_enabled"] = False
            env = CognitiveRFScanEnv(copy_cfg, records=recs, seed=seed, semantic_memory_path=":memory:")
            obs, _ = env.reset(seed=seed)
            hidden = drqn.init_hidden(1, "cpu")
            for step in range(50):  # Collect warm observations
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
                with torch.inference_mode():
                    q_vals, _aux, hidden = drqn(obs_t, hidden)
                    act = int(torch.argmax(q_vals[0, -1]).item())
                if step >= 10:
                    collected_obs.append((np.asarray(obs, dtype=np.float32).copy(), hidden))
                obs, _, done, _, _ = env.step(act)
                if done:
                    break

    print(f"Collected {len(collected_obs)} active observations for feature sensitivity audit.")

    features_to_audit = {
        "occupancy": 0,
        "detection_rate": 1,
        "miss_rate": 2,
        "uncertainty": 3,
        "revisit_age": 4,
        "emitter_count": 5,
        "agility": 8,
        "priority": 9,
    }

    audit_results = {}

    for feat_name, feat_idx in features_to_audit.items():
        delta_q_all = []
        delta_q_target_band = []
        delta_q_modes = {m: [] for m in DWELL_MODES}
        action_changed_cnt = 0
        band_changed_cnt = 0
        mode_changed_cnt = 0

        for base_obs, hidden in collected_obs:
            obs_tensor_base = torch.from_numpy(base_obs).unsqueeze(0).unsqueeze(0)
            with torch.inference_mode():
                q_base, _, _ = drqn(obs_tensor_base, hidden)
                q_base_np = q_base[0, -1].cpu().numpy()
                act_base = int(np.argmax(q_base_np))
                band_base = act_base // 5
                mode_base = act_base % 5

            # Perturb feature: add +0.5 (or set to 1.0) on the target band (band_base) and across all bands
            perturbed_obs = base_obs.copy()
            # In 360-D observation, band b feature f is at index b * 10 + f
            perturbed_obs[band_base * 10 + feat_idx] = min(1.0, float(perturbed_obs[band_base * 10 + feat_idx] + 0.5))

            obs_tensor_pert = torch.from_numpy(perturbed_obs).unsqueeze(0).unsqueeze(0)
            with torch.inference_mode():
                q_pert, _, _ = drqn(obs_tensor_pert, hidden)
                q_pert_np = q_pert[0, -1].cpu().numpy()
                act_pert = int(np.argmax(q_pert_np))
                band_pert = act_pert // 5
                mode_pert = act_pert % 5

            diff_q = q_pert_np - q_base_np
            delta_q_all.append(float(np.mean(np.abs(diff_q))))
            
            # Target band Q differences
            target_slice = diff_q[band_base * 5 : (band_base + 1) * 5]
            delta_q_target_band.append(float(np.mean(target_slice)))

            for m_idx, m_name in enumerate(DWELL_MODES):
                delta_q_modes[m_name].append(float(target_slice[m_idx]))

            if act_pert != act_base:
                action_changed_cnt += 1
            if band_pert != band_base:
                band_changed_cnt += 1
            if mode_pert != mode_base:
                mode_changed_cnt += 1

        n_samples = len(collected_obs)
        audit_results[feat_name] = {
            "mean_abs_delta_q_all_actions": float(np.mean(delta_q_all)),
            "mean_delta_q_target_band": float(np.mean(delta_q_target_band)),
            "per_mode_delta_q": {m: float(np.mean(delta_q_modes[m])) for m in DWELL_MODES},
            "action_change_pct": float(action_changed_cnt / n_samples) * 100.0,
            "band_change_pct": float(band_changed_cnt / n_samples) * 100.0,
            "mode_change_pct": float(mode_changed_cnt / n_samples) * 100.0,
        }

    # Print Sensitivity Table
    print("\n" + "=" * 130)
    print("  DIAGNOSTIC 6: 25K FROZEN BASELINE FEATURE SENSITIVITY AUDIT")
    print("=" * 130)
    print(f"  {'Feature':<16} | {'Mean |dQ| All':<14} | {'dQ Target Band':<15} | {'Act Change%':<12} | {'Mode Change%':<13} | {'Mode Q Sensitivity (S/N/L/R/P)'}")
    print("-" * 130)

    for feat_name, res in audit_results.items():
        modes_str = "/".join(f"{res['per_mode_delta_q'][m]:+.2f}" for m in DWELL_MODES)
        print(f"  {feat_name:<16} | {res['mean_abs_delta_q_all_actions']:12.4f}  | {res['mean_delta_q_target_band']:+13.4f}  | {res['action_change_pct']:10.1f}% | {res['mode_change_pct']:11.1f}% | {modes_str}")
    print("=" * 130)

    out_path = Path("experiments/reports/phase11_diagnostic_q_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nSaved Diagnostic 6 report to: {out_path}\n")

if __name__ == "__main__":
    run_sensitivity_audit()
