"""
Diagnostic 1: Determine whether the failure is DRQN or Arbitration.
Replays validation trajectories for config_119, config_143, config_241, config_29
across 25k, 27.5k, and 30k checkpoints.
Measures:
  - Raw DRQN IR vs Final Selected Action IR
  - Raw DRQN Mode distribution vs Final Mode distribution
  - Override rate and Decision Source
  - Q statistics (Q_selected, Q_max, Q_mean, Q_std)
"""

import copy
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, DWELL_MODES
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.models.smartscan_moe import SmartScanMoE
from ew_core.training.val_set import FixedValidationSet

def run_diagnostic():
    checkpoints = {
        "25k_frozen": Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"),
        "27.5k": Path("experiments/checkpoints/phase11_controlled/quarantine/checkpoint_phase11_step_27500.pt"),
        "30k": Path("experiments/checkpoints/phase11_controlled/quarantine/checkpoint_phase11_step_30000.pt"),
    }

    target_scenarios = ["config_119", "config_143", "config_241", "config_29"]

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

    # Filter val_set files for the target scenarios and preload records
    target_files = []
    freq_min = float(env_cfg.get("freq_min_mhz", 0.0))
    freq_max = float(env_cfg.get("freq_max_mhz", 18000.0))
    time_horizon = float(env_cfg.get("time_horizon_us", 0.0)) or None
    max_pulses = int(env_cfg.get("max_pulses", 50000))

    for item in val_set.files_used:
        fpath = Path(item[0] if isinstance(item, (list, tuple)) else item)
        sc_id = fpath.stem
        if sc_id in target_scenarios:
            recs = load_h5_records(fpath, freq_min_mhz=freq_min, freq_max_mhz=freq_max, time_horizon_us=time_horizon, max_pulses=max_pulses)
            target_files.append((fpath, sc_id, recs))
    
    print(f"Loaded {len(target_files)} target validation scenarios: {[s for _, s, _ in target_files]}")

    results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\nEvaluating Checkpoint: {ckpt_name} ({ckpt_path.name})...")
        drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = payload["state_dict"] if "state_dict" in payload else payload
        drqn.load_state_dict(state, strict=True)
        drqn.eval()

        results[ckpt_name] = {}

        for fpath, scen_id, recs in target_files:
            # 1. Run Pure Raw DRQN (No MoE / No Arbitration)
            val_env_cfg = copy.deepcopy(env_cfg)
            val_env_cfg["semantic_memory_enabled"] = False
            env_raw = CognitiveRFScanEnv(val_env_cfg, records=recs, seed=seed, semantic_memory_path=":memory:")
            obs, _ = env_raw.reset(seed=seed)
            hidden = drqn.init_hidden(1, "cpu")
            raw_hits = 0
            raw_modes = np.zeros(5, dtype=int)
            q_selected_list = []
            q_max_list = []
            q_mean_list = []
            q_std_list = []

            for step in range(1000):
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
                with torch.inference_mode():
                    q_vals, _aux, hidden = drqn(obs_t, hidden)
                    q_np = q_vals.squeeze(0).squeeze(0).cpu().numpy()
                    raw_action = int(np.argmax(q_np))
                
                raw_mode = raw_action % 5
                raw_modes[raw_mode] += 1
                q_selected_list.append(float(q_np[raw_action]))
                q_max_list.append(float(np.max(q_np)))
                q_mean_list.append(float(np.mean(q_np)))
                q_std_list.append(float(np.std(q_np)))

                obs, reward, done, _, info = env_raw.step(raw_action)
                if info.get("intercepted", False) or reward > 0:
                    raw_hits += 1
                if done:
                    break

            raw_ir = raw_hits / 1000.0

            # 2. Run Full MoE Arbitration
            env_moe = CognitiveRFScanEnv(val_env_cfg, records=recs, seed=seed, semantic_memory_path=":memory:")
            obs, _ = env_moe.reset(seed=seed)
            from ew_core.models.baseline_suite import build_baseline
            moe_agent = build_baseline(
                "full_moe",
                n_bands=36,
                n_modes=5,
                drqn=drqn,
                config={
                    "mode_selection_mode": "flat_argmax",
                    "tau": 0.0,
                    "confidence_margin_threshold": 0.02,
                    "background_margin_threshold": 0.02,
                    "uncertainty_margin_threshold": 0.02,
                },
                seed=seed,
                device="cpu",
            )
            moe_agent.reset()
            hidden_moe = drqn.init_hidden(1, "cpu")
            moe_hits = 0
            moe_modes = np.zeros(5, dtype=int)
            override_count = 0
            agreement_count = 0
            decision_sources = {}

            for step in range(1000):
                action, hidden_moe, attr = moe_agent.select_action(obs, hidden_moe)
                action = int(action)
                raw_act = int(attr.get("q_argmax_action", attr.get("q_argmax", action)))
                
                if action != raw_act:
                    override_count += 1
                else:
                    agreement_count += 1

                source = str(attr.get("decision_source", attr.get("reason", "unknown")))
                decision_sources[source] = decision_sources.get(source, 0) + 1

                moe_modes[action % 5] += 1
                obs, reward, done, _, info = env_moe.step(action)
                if info.get("intercepted", False) or reward > 0:
                    moe_hits += 1
                if done:
                    break

            moe_ir = moe_hits / 1000.0

            results[ckpt_name][scen_id] = {
                "raw_drqn_ir": raw_ir,
                "final_moe_ir": moe_ir,
                "raw_modes": {DWELL_MODES[i]: int(raw_modes[i]) for i in range(5)},
                "final_modes": {DWELL_MODES[i]: int(moe_modes[i]) for i in range(5)},
                "raw_mode_pct": {DWELL_MODES[i]: float(raw_modes[i]/10.0) for i in range(5)},
                "final_mode_pct": {DWELL_MODES[i]: float(moe_modes[i]/10.0) for i in range(5)},
                "override_rate_pct": float(override_count / 10.0),
                "agreement_rate_pct": float(agreement_count / 10.0),
                "decision_sources": decision_sources,
                "q_stats": {
                    "mean_q_selected": float(np.mean(q_selected_list)),
                    "mean_q_max": float(np.mean(q_max_list)),
                    "mean_q_mean": float(np.mean(q_mean_list)),
                    "mean_q_std": float(np.mean(q_std_list)),
                }
            }

    # Print Side-by-Side Diagnostic Table
    print("\n" + "=" * 140)
    print("  DIAGNOSTIC 1: RAW DRQN vs. FINAL ARBITRATION BREAKDOWN")
    print("=" * 140)
    print(f"  {'Checkpoint':<12} | {'Scenario':<12} | {'Raw DRQN IR':<13} | {'Final MoE IR':<13} | {'Override%':<10} | {'Raw Modes (S/N/L/R/P)':<25} | {'Final Modes (S/N/L/R/P)':<25} | {'Decision Source'}")
    print("-" * 140)

    for ckpt_name in ["25k_frozen", "27.5k", "30k"]:
        for scen_id in target_scenarios:
            d = results[ckpt_name][scen_id]
            raw_m = [d['raw_modes'][m] for m in DWELL_MODES]
            fin_m = [d['final_modes'][m] for m in DWELL_MODES]
            raw_m_str = "/".join(str(x) for x in raw_m)
            fin_m_str = "/".join(str(x) for x in fin_m)
            top_src = max(d["decision_sources"].items(), key=lambda x: x[1])[0] if d["decision_sources"] else "None"
            print(f"  {ckpt_name:<12} | {scen_id:<12} | {d['raw_drqn_ir']*100:6.2f}%       | {d['final_moe_ir']*100:6.2f}%       | {d['override_rate_pct']:5.1f}%     | {raw_m_str:<25} | {fin_m_str:<25} | {top_src}")
        print("-" * 140)

    # Save output
    out_path = Path("experiments/reports/phase11_diagnostic_drqn_vs_arbitration.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Diagnostic 1 report to: {out_path}\n")

if __name__ == "__main__":
    run_diagnostic()
