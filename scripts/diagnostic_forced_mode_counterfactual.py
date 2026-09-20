"""
Diagnostic 2: Forced-Mode Counterfactual Study.
Holding frequency band selection constant across identical recorded trajectories
for config_119, config_143, config_241, config_29:
Forces each of the 5 modes:
  - SHORT (0)
  - NORMAL (1)
  - LONG (2)
  - REVISIT (3)
  - PREEMPTIVE (4)
Measures:
  - Hit probability / Interception Rate (IR)
  - First detection latency (us)
  - Total physical mission time (ms)
  - Interception rate per ms of mission time (hits / ms)
  - Total reward
  - Reward per ms
  - Miss probability
"""

import copy
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    DWELL_MODES,
    SHORT_DWELL,
    NORMAL_DWELL,
    LONG_DWELL,
    REVISIT,
    PREEMPTIVE_INTERCEPT,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import load_h5_records
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.training.val_set import FixedValidationSet

def run_counterfactual():
    ckpt_path = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
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

    # Load 25k frozen DRQN to generate baseline band sequence
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    drqn.load_state_dict(state, strict=True)
    drqn.eval()

    val_env_cfg = copy.deepcopy(env_cfg)
    val_env_cfg["semantic_memory_enabled"] = False

    results = {}

    for fpath, scen_id, recs in target_files:
        print(f"\nProcessing Scenario: {scen_id}...")
        results[scen_id] = {}

        # 1. Record baseline band sequence from 25k policy rollouts
        env_base = CognitiveRFScanEnv(val_env_cfg, records=recs, seed=seed, semantic_memory_path=":memory:")
        obs, _ = env_base.reset(seed=seed)
        hidden = drqn.init_hidden(1, "cpu")
        band_sequence = []

        for step in range(1000):
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
            with torch.inference_mode():
                q_vals, _aux, hidden = drqn(obs_t, hidden)
                q_np = q_vals.squeeze(0).squeeze(0).cpu().numpy()
                act = int(np.argmax(q_np))
            band = act // 5
            band_sequence.append(band)
            obs, _, done, _, _ = env_base.step(act)
            if done:
                break

        print(f"  Recorded {len(band_sequence)} baseline band decisions (unique bands: {len(set(band_sequence))})")

        # 2. For each of the 5 modes, replay with identical band sequence
        mode_names = ["SHORT (125us)", "NORMAL (500us)", "LONG (1250us)", "REVISIT (500us+boost)", "PREEMPTIVE (500us+align)"]

        for mode_idx in range(5):
            env_mode = CognitiveRFScanEnv(val_env_cfg, records=recs, seed=seed, semantic_memory_path=":memory:")
            obs, _ = env_mode.reset(seed=seed)
            
            hits = 0
            misses = 0
            rewards = []
            first_hit_latency_us = None
            start_time_us = env_mode.receiver.current_time_us

            for step, band in enumerate(band_sequence):
                action = band * 5 + mode_idx
                obs, reward, done, _, info = env_mode.step(action)
                rewards.append(float(reward))

                is_hit = bool(info.get("intercepted", False) or reward > 0)
                if is_hit:
                    hits += 1
                    if first_hit_latency_us is None:
                        first_hit_latency_us = float(info.get("intercept_latency_us", env_mode.receiver.current_time_us - start_time_us))
                else:
                    misses += 1

                if done:
                    break

            end_time_us = env_mode.receiver.current_time_us
            elapsed_ms = float((end_time_us - start_time_us) / 1000.0)
            n_steps_actual = len(rewards)
            ir = float(hits / max(1, n_steps_actual))
            miss_prob = float(misses / max(1, n_steps_actual))
            total_rew = float(np.sum(rewards))
            rew_per_ms = float(total_rew / max(1e-3, elapsed_ms))
            hits_per_ms = float(hits / max(1e-3, elapsed_ms))

            results[scen_id][DWELL_MODES[mode_idx]] = {
                "mode_name": mode_names[mode_idx],
                "steps": n_steps_actual,
                "hits": hits,
                "misses": misses,
                "intercept_rate": ir,
                "miss_probability": miss_prob,
                "first_hit_latency_us": first_hit_latency_us,
                "elapsed_mission_ms": elapsed_ms,
                "hits_per_ms": hits_per_ms,
                "total_reward": total_rew,
                "reward_per_ms": rew_per_ms,
            }

    # Print Counterfactual Comparison Table
    print("\n" + "=" * 145)
    print("  DIAGNOSTIC 2: FORCED-MODE COUNTERFACTUAL STUDY (CONSTANT BAND SEQUENCE)")
    print("=" * 145)
    print(f"  {'Scenario':<12} | {'Mode':<24} | {'IR':<9} | {'Hits':<6} | {'First Hit':<12} | {'Mission Time':<14} | {'Hits / ms':<11} | {'Reward':<10} | {'Reward / ms'}")
    print("-" * 145)

    for scen_id in target_scenarios:
        for mode_idx in range(5):
            m_key = DWELL_MODES[mode_idx]
            d = results[scen_id][m_key]
            first_hit_str = f"{d['first_hit_latency_us']:6.1f} us" if d['first_hit_latency_us'] is not None else "None"
            print(f"  {scen_id:<12} | {d['mode_name']:<24} | {d['intercept_rate']*100:6.2f}% | {d['hits']:<6} | {first_hit_str:<12} | {d['elapsed_mission_ms']:7.1f} ms    | {d['hits_per_ms']:6.3f}     | {d['total_reward']:8.1f}   | {d['reward_per_ms']:6.2f}")
        print("-" * 145)

    # Save to report
    out_file = Path("experiments/reports/phase11_diagnostic_forced_mode_counterfactual.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Diagnostic 2 report to: {out_file}\n")

if __name__ == "__main__":
    run_counterfactual()
