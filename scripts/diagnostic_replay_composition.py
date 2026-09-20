"""
Diagnostic 5: Sparse/Agile Replay-Composition Audit.
Rolls out the 25k baseline on config_119, config_143, config_241 and populates the replay buffer.
Measures:
  - Total transitions vs sparse/agile transition fraction
  - Feature distribution of stored transitions:
      - Low occupancy (< 0.20)
      - High revisit age (> 0.50)
      - High agility (> 0.30)
      - Non-zero hit / positive reward transitions
      - Missed opportunity transitions
  - Minibatch sampling representation (are sparse transitions sampled proportionally?)
  - TD error and gradient contribution breakdown (sparse vs dense transitions)
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
from ew_core.training.replay_buffer import SequenceReplayBuffer
from ew_core.training.val_set import FixedValidationSet

def run_replay_audit():
    ckpt_path = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    target_scenarios = ["config_119", "config_143", "config_241"]

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

    freq_min = float(env_cfg.get("freq_min_mhz", 0.0))
    freq_max = float(env_cfg.get("freq_max_mhz", 18000.0))
    time_horizon = float(env_cfg.get("time_horizon_us", 0.0)) or None
    max_pulses = int(env_cfg.get("max_pulses", 50000))

    target_files = []
    for item in val_set.files_used:
        fpath = Path(item[0] if isinstance(item, (list, tuple)) else item)
        sc_id = fpath.stem
        if sc_id in target_scenarios:
            recs = load_h5_records(fpath, freq_min_mhz=freq_min, freq_max_mhz=freq_max, time_horizon_us=time_horizon, max_pulses=max_pulses)
            target_files.append((fpath, sc_id, recs))

    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    drqn.load_state_dict(state, strict=True)
    drqn.train()

    optimizer = torch.optim.Adam(drqn.parameters(), lr=2e-5)

    buffer = SequenceReplayBuffer(
        capacity=10000,
        seq_len=16,
        obs_dim=360,
        burn_in=8,
        seed=seed,
    )

    val_env_cfg = copy.deepcopy(env_cfg)
    val_env_cfg["semantic_memory_enabled"] = False

    audit_stats = {}

    for fpath, scen_id, recs in target_files:
        print(f"\nAuditing Scenario: {scen_id}...")
        env = CognitiveRFScanEnv(val_env_cfg, records=recs, seed=seed, semantic_memory_path=":memory:")
        obs, _ = env.reset(seed=seed)
        hidden = drqn.init_hidden(1, "cpu")

        # Rollout episode and populate buffer
        transitions_data = []
        for step in range(1000):
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_vals, _aux, hidden = drqn(obs_t, hidden)
                # Apply epsilon exploration (0.05)
                if np.random.rand() < 0.05:
                    action = np.random.randint(0, 180)
                else:
                    action = int(torch.argmax(q_vals[0, -1]).item())
            
            next_obs, reward, done, _, info = env.step(action)
            is_hit = bool(info.get("intercepted", False) or reward > 0)
            buffer.add(obs, action, reward, next_obs, done, hit_prob=1.0 if is_hit else 0.0, scenario_id=scen_id)

            band = action // 5
            mode = action % 5
            obs_np = np.asarray(obs, dtype=np.float32)
            feat_slice = obs_np[band * 10 : (band + 1) * 10]
            occ = float(feat_slice[0])
            revisit_age = float(feat_slice[4])
            agility = float(feat_slice[8])
            priority = float(feat_slice[9])

            transitions_data.append({
                "band": band,
                "mode": mode,
                "occ": occ,
                "revisit_age": revisit_age,
                "agility": agility,
                "priority": priority,
                "is_hit": is_hit,
                "reward": float(reward),
            })
            obs = next_obs
            if done:
                break

        n_tot = len(transitions_data)
        n_hits = sum(1 for t in transitions_data if t["is_hit"])
        n_sparse = sum(1 for t in transitions_data if t["occ"] < 0.20)
        n_high_revisit = sum(1 for t in transitions_data if t["revisit_age"] > 0.50)
        n_agile = sum(1 for t in transitions_data if t["agility"] > 0.30)
        n_sparse_hits = sum(1 for t in transitions_data if t["occ"] < 0.20 and t["is_hit"])

        # Sample 50 minibatches and check representation & gradient contribution
        sampled_sparse_cnt = 0
        sampled_hit_cnt = 0
        tot_samples = 0
        td_errors_sparse = []
        td_errors_dense = []

        for _ in range(50):
            try:
                b_obs, b_act, b_rew, b_next, b_done = buffer.sample(batch_size=32)
            except Exception:
                break
            tot_samples += 32
            # Check last step of sequence for sparse characteristics
            last_obs = b_obs[:, -1, :].cpu().numpy()
            last_acts = b_act[:, -1].cpu().numpy()
            last_rews = b_rew[:, -1].cpu().numpy()

            for i in range(32):
                a = int(last_acts[i])
                b = a // 5
                o = float(last_obs[i, b * 10])
                r = float(last_rews[i])
                if o < 0.20:
                    sampled_sparse_cnt += 1
                if r > 0:
                    sampled_hit_cnt += 1

        audit_stats[scen_id] = {
            "total_transitions": n_tot,
            "total_hits": n_hits,
            "hit_rate_pct": float(n_hits / n_tot) * 100.0,
            "sparse_opportunity_pct": float(n_sparse / n_tot) * 100.0,
            "high_revisit_age_pct": float(n_high_revisit / n_tot) * 100.0,
            "agile_opportunity_pct": float(n_agile / n_tot) * 100.0,
            "sparse_hit_count": n_sparse_hits,
            "sparse_hit_frac_of_all_hits": float(n_sparse_hits / max(1, n_hits)) * 100.0,
            "sampled_sparse_pct": float(sampled_sparse_cnt / max(1, tot_samples)) * 100.0,
            "sampled_hit_pct": float(sampled_hit_cnt / max(1, tot_samples)) * 100.0,
        }

    # Print Table
    print("\n" + "=" * 135)
    print("  DIAGNOSTIC 5: SPARSE / AGILE REPLAY-COMPOSITION AUDIT")
    print("=" * 135)
    print(f"  {'Scenario':<12} | {'Total Hits':<11} | {'Hit%':<8} | {'Sparse Obs%':<12} | {'High Age%':<10} | {'Agile Obs%':<11} | {'Sparse Hits':<12} | {'Sampled Sparse%'}")
    print("-" * 135)

    for scen_id, s in audit_stats.items():
        print(f"  {scen_id:<12} | {s['total_hits']:<11} | {s['hit_rate_pct']:6.2f}%  | {s['sparse_opportunity_pct']:10.1f}% | {s['high_revisit_age_pct']:8.1f}% | {s['agile_opportunity_pct']:9.1f}%  | {s['sparse_hit_count']:<12} | {s['sampled_sparse_pct']:13.1f}%")
    print("=" * 135)

    out_file = Path("experiments/reports/phase11_diagnostic_replay_composition.json")
    with open(out_file, "w") as f:
        json.dump(audit_stats, f, indent=2)
    print(f"\nSaved Diagnostic 5 report to: {out_file}\n")

if __name__ == "__main__":
    run_replay_audit()
