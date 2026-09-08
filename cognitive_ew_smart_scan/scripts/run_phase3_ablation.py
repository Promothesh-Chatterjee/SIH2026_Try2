"""
Phase 3 4-Policy Ablation Study:
Compares:
  A: Greedy DRQN (tau=0, no fallback)
  B: Boltzmann DRQN across tau in [0.05, 0.10, 0.15, 0.20, 0.25] (no fallback)
  C: Greedy DRQN + Confidence Fallback (tau=0)
  D: Boltzmann DRQN + Confidence Fallback across tau in [0.05, 0.10, 0.15, 0.20, 0.25]

Evaluated on the exact 10 held-out validation scenarios, identical seeds, and checkpoint_gate_10000.pt.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, band_of_action, mode_of_action
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import load_h5_records
from src.telemetry.schema import shannon_entropy
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("Phase3Ablation")


def run_rollout(
    policy_type: str,
    tau: float,
    drqn: DRQNScheduler,
    moe: SmartScanMoE,
    env: CognitiveRFScanEnv,
    n_steps: int = 1000,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    n_bands = env.n_bands
    n_modes = env.n_modes
    n_actions = n_bands * n_modes

    rng = np.random.default_rng(seed)
    hidden = drqn.init_hidden(1, device)
    moe.reset()

    band_counts = np.zeros(n_bands, dtype=int)
    action_counts = np.zeros(n_actions, dtype=int)
    q_margins: list[float] = []

    consecutive_empty = 0
    last_band = -1
    empty_escape_opps = 0
    empty_escapes = 0
    stale_escape_opps = 0
    stale_escapes = 0
    fallback_triggers = 0
    ep_hits = 0
    ep_reward = 0.0

    for step in range(n_steps):
        # 1. Action selection according to policy
        obs_1d = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs_t = torch.from_numpy(obs_1d).to(device).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            q_out, _aux, hidden = drqn(obs_t, hidden)
            q_vals = q_out[0, -1].detach().cpu().numpy()

        order = np.argsort(q_vals)[::-1]
        q_top1 = float(q_vals[order[0]])
        q_top2 = float(q_vals[order[1]]) if len(order) > 1 else q_top1
        margin = q_top1 - q_top2
        q_margins.append(margin)

        # Policy branching
        if policy_type == "greedy":
            # Policy A: Pure greedy DRQN
            action = int(order[0])
            is_fallback = False

        elif policy_type == "boltzmann":
            # Policy B: Pure Boltzmann DRQN (bounded to top 5)
            top_k = min(5, len(order))
            top_idx = order[:top_k]
            top_q = q_vals[top_idx]
            probs = np.exp((top_q - np.max(top_q)) / max(1e-5, tau))
            probs = probs / np.sum(probs)
            action = int(rng.choice(top_idx, p=probs))
            is_fallback = False

        elif policy_type in ("drqn_fallback", "boltzmann_fallback"):
            # Policy C or D: MoE with confidence gating
            # Confidence check: margin > 0.10, unc < 0.80, rev < 0.95
            q_argmax_band = band_of_action(int(order[0]), n_modes)
            fpb = max(1, len(obs_1d) // n_bands)
            unc = float(np.clip(obs_1d[3::fpb][q_argmax_band], 0.0, 1.0))
            rev = float(np.clip(obs_1d[4::fpb][q_argmax_band], 0.0, 1.0))
            is_confident = (margin > 0.10) and (unc < 0.80) and (rev < 0.95)

            if is_confident:
                if policy_type == "boltzmann_fallback" and tau > 0.0:
                    top_k = min(5, len(order))
                    top_idx = order[:top_k]
                    top_q = q_vals[top_idx]
                    probs = np.exp((top_q - np.max(top_q)) / max(1e-5, tau))
                    probs = probs / np.sum(probs)
                    action = int(rng.choice(top_idx, p=probs))
                else:
                    action = int(order[0])
                is_fallback = False
            else:
                # Heuristic safety fallback
                if hasattr(env, "belief") and env.belief is not None:
                    moe.set_periodic_urgency_vector(env.belief.periodic_urgency)
                action, _, attr = moe.select_action(obs_1d, None, tau=tau)
                action = int(action)
                is_fallback = True
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        if is_fallback:
            fallback_triggers += 1

        band = int(action // n_modes)
        action_counts[action] += 1
        band_counts[band] += 1

        # Check escape opportunities from previous step
        if last_band >= 0:
            if consecutive_empty >= 2:
                empty_escape_opps += 1
                if band != last_band:
                    empty_escapes += 1
            stale_escape_opps += 1
            if band != last_band:
                stale_escapes += 1

        # Step environment
        obs, reward, term, trunc, info = env.step(action)
        hit = bool(info.get("hit", False))
        ep_reward += float(reward)
        ep_hits += int(hit)

        if not hit:
            if last_band == band or last_band == -1:
                consecutive_empty += 1
            else:
                consecutive_empty = 1
        else:
            consecutive_empty = 0
        last_band = band

        moe.update(action)

        if term or trunc:
            break

    steps_done = max(1, step + 1)
    empty_escape_rate = float(empty_escapes / max(1, empty_escape_opps)) if empty_escape_opps > 0 else 1.0
    stale_escape_rate = float(stale_escapes / max(1, stale_escape_opps)) if stale_escape_opps > 0 else 1.0

    return {
        "intercept_rate": float(ep_hits / steps_done),
        "hits": int(ep_hits),
        "steps": int(steps_done),
        "distinct_bands": int(np.count_nonzero(band_counts)),
        "band_entropy": float(shannon_entropy(band_counts)),
        "action_entropy": float(shannon_entropy(action_counts)),
        "empty_band_escape_rate": empty_escape_rate,
        "stale_band_escape_rate": stale_escape_rate,
        "mean_q_margin": float(np.mean(q_margins)),
        "total_reward": float(ep_reward),
        "avg_reward": float(ep_reward / steps_done),
        "fallback_rate": float(fallback_triggers / steps_done),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 4-Policy Ablation Study")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scheduler/checkpoint_gate_10000.pt")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--val-json", type=str, default="runs/20260908-001102-2e2f42/validation_set.json")
    parser.add_argument("--out", type=str, default="phase3_ablation_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-steps", type=int, default=1000)
    args = parser.parse_args()

    # Load configs
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    drqn_cfg = cfg.get("drqn_scheduler", {})
    moe_cfg = cfg.get("smartscan_moe", {})
    reward_cfg = cfg.get("reward", {})
    env_cfg = cfg.get("environment", {})
    env_config = {**env_cfg, **reward_cfg, **cfg.get("belief", {})}
    env_config["n_bands"] = int(drqn_cfg.get("n_bands", CANONICAL_N_BANDS))
    env_config["n_modes"] = int(drqn_cfg.get("n_modes", CANONICAL_N_MODES))
    env_config["n_actions"] = env_config["n_bands"] * env_config["n_modes"]
    env_config["semantic_memory_enabled"] = False

    # Load validation scenario paths
    val_json_path = Path(args.val_json)
    if not val_json_path.exists():
        # Search for any validation_set.json
        cand = list(Path("runs").rglob("validation_set.json"))
        if cand:
            val_json_path = cand[-1]
            logger.info("Using validation set from %s", val_json_path)

    with open(val_json_path, "r", encoding="utf-8") as f:
        val_meta = json.load(f)
    val_files = val_meta.get("scenario_files", [])
    logger.info("Found %d validation scenario files.", len(val_files))

    # Preload scenarios into memory
    preloaded = []
    for fp in val_files:
        p = Path(fp)
        if not p.exists():
            # Try finding locally
            local_cand = list(Path("data").rglob(p.name))
            if local_cand:
                p = local_cand[0]
        if p.exists():
            records = load_h5_records(p, freq_min_mhz=0.0, freq_max_mhz=18000.0, max_pulses=50000)
            preloaded.append((p.stem, records))
            logger.info("Loaded scenario %s (%d pulses)", p.stem, len(records))

    if not preloaded:
        logger.error("No validation scenarios could be loaded. Aborting.")
        return

    # Instantiate model and load checkpoint
    device = torch.device(args.device)
    drqn = DRQNScheduler(
        obs_dim=int(drqn_cfg.get("obs_dim", 360)),
        n_bands=env_config["n_bands"],
        n_actions=env_config["n_actions"],
        lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
        lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    logger.info("Loading DRQN checkpoint from %s...", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    drqn.load_state_dict(state.get("state_dict", state))
    drqn.eval()

    moe = SmartScanMoE(
        drqn,
        {**moe_cfg, "n_bands": env_config["n_bands"], "n_modes": env_config["n_modes"], "n_actions": env_config["n_actions"], "device": str(device)},
    ).to(device)

    # Define ablation matrix
    # A: Greedy DRQN
    # B: Boltzmann DRQN (tau in [0.05, 0.10, 0.15, 0.20, 0.25])
    # C: Greedy DRQN + Fallback
    # D: Boltzmann DRQN + Fallback (tau in [0.05, 0.10, 0.15, 0.20, 0.25])
    ablation_policies = [
        ("A: Greedy DRQN", "greedy", 0.0),
        ("B: Boltzmann (tau=0.05)", "boltzmann", 0.05),
        ("B: Boltzmann (tau=0.10)", "boltzmann", 0.10),
        ("B: Boltzmann (tau=0.15)", "boltzmann", 0.15),
        ("B: Boltzmann (tau=0.20)", "boltzmann", 0.20),
        ("B: Boltzmann (tau=0.25)", "boltzmann", 0.25),
        ("C: Greedy + Fallback", "drqn_fallback", 0.0),
        ("D: Boltzmann + Fallback (tau=0.05)", "boltzmann_fallback", 0.05),
        ("D: Boltzmann + Fallback (tau=0.10)", "boltzmann_fallback", 0.10),
        ("D: Boltzmann + Fallback (tau=0.15)", "boltzmann_fallback", 0.15),
        ("D: Boltzmann + Fallback (tau=0.20)", "boltzmann_fallback", 0.20),
        ("D: Boltzmann + Fallback (tau=0.25)", "boltzmann_fallback", 0.25),
    ]

    all_results = {}
    summary_rows = []

    print("\n" + "=" * 145)
    print(f"{'Policy':<36} | {'Intercept':<10} | {'Distinct':<8} | {'Entropy':<8} | {'EmptyEscape':<11} | {'StaleEscape':<11} | {'Q-Margin':<9} | {'Fallback':<9} | {'Reward'}")
    print("-" * 145)

    for label, pol_type, tau in ablation_policies:
        pol_res = []
        for scen_id, records in preloaded:
            env = CognitiveRFScanEnv(copy.deepcopy(env_config), records=records, seed=42, semantic_memory_path=":memory:")
            res = run_rollout(
                policy_type=pol_type,
                tau=tau,
                drqn=drqn,
                moe=moe,
                env=env,
                n_steps=args.n_steps,
                seed=42,
                device=args.device,
            )
            pol_res.append(res)

        # Aggregate across scenarios
        agg = {
            "intercept_rate": float(np.mean([r["intercept_rate"] for r in pol_res])),
            "distinct_bands": float(np.mean([r["distinct_bands"] for r in pol_res])),
            "band_entropy": float(np.mean([r["band_entropy"] for r in pol_res])),
            "action_entropy": float(np.mean([r["action_entropy"] for r in pol_res])),
            "empty_band_escape_rate": float(np.mean([r["empty_band_escape_rate"] for r in pol_res])),
            "stale_band_escape_rate": float(np.mean([r["stale_band_escape_rate"] for r in pol_res])),
            "mean_q_margin": float(np.mean([r["mean_q_margin"] for r in pol_res])),
            "total_reward": float(np.mean([r["total_reward"] for r in pol_res])),
            "fallback_rate": float(np.mean([r["fallback_rate"] for r in pol_res])),
        }
        all_results[label] = {"summary": agg, "per_scenario": pol_res}

        print(
            f"{label:<36} | {agg['intercept_rate']*100:6.2f}%    | {agg['distinct_bands']:4.1f}/36  | {agg['band_entropy']:6.2f}   | "
            f"{agg['empty_band_escape_rate']*100:6.1f}%     | {agg['stale_band_escape_rate']*100:6.1f}%     | {agg['mean_q_margin']:6.3f}    | "
            f"{agg['fallback_rate']*100:5.1f}%    | {agg['total_reward']:8.1f}"
        )

    print("=" * 145 + "\n")

    out_file = Path(args.out)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Ablation study complete. Saved results to %s", out_file)


if __name__ == "__main__":
    main()
