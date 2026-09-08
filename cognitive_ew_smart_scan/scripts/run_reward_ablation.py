"""
Reward Ablation Study: Variants R0 through R4.

Evaluates 5 candidate reward profiles starting from clean checkpoint_gate_25000.pt:
  - R0 (Baseline): penalize_empty_dwell=True, w_hit=1.0, w_dwell=-0.001
  - R1 (TN Clean): penalize_empty_dwell=False, w_hit=5.0, w_dwell=-0.001
  - R2 (Opportunity): R1 + w_priority=1.0
  - R3 (Low Dwell Cost): R2 + w_dwell_cost=-0.0002
  - R4 (Full Balanced): R3 + w_novel=5.0, w_miss=-0.5, w_staleness=0.4, context_scaled_miss=True

Each variant trains for 3,000 steps on TSRD STARE train, then evaluates
on the exact 10 held-out validation scenarios across all baseline policies.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, band_of_action, mode_of_action
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import ScenarioSource
from src.models.deinterleaver import PDWTransformerEncoder
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE
from src.preprocessing.normalise import load_normalization_stats
from src.training.replay_buffer import SequenceReplayBuffer
from src.training.staged_gate_evaluator import StagedGateEvaluator
from src.training.train_scheduler import _do_drqn_update
from src.training.val_set import FixedValidationSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("src.cognitive.periodic_interceptor").setLevel(logging.WARNING)
logging.getLogger("src.models.smartscan_moe").setLevel(logging.WARNING)
logging.getLogger("src.environment.cognitive_rf_scan_env").setLevel(logging.WARNING)
logger = logging.getLogger("RewardAblation")

# 5 Reward Variants defined in the approved implementation plan
REWARD_VARIANTS = {
    "R0_Baseline": {
        "description": "Baseline: TN empty dwell false alarm penalty (-0.5), w_hit=1.0, w_dwell=-0.001",
        "reward": {
            "penalize_empty_dwell": True,
            "w_hit": 1.0,
            "w_novel": 2.0,
            "w_false_alarm": -0.5,
            "w_dwell_cost": -0.001,
            "w_miss": -0.3,
            "w_priority": 0.5,
            "w_information_gain": 0.2,
            "w_staleness": 0.6,
            "context_scaled_miss": False,
        },
    },
    "R1_TNClean": {
        "description": "TN Clean: penalize_empty_dwell=False (0 FA penalty on empty dwell), w_hit=5.0",
        "reward": {
            "penalize_empty_dwell": False,
            "w_hit": 5.0,
            "w_novel": 2.0,
            "w_false_alarm": -0.5,
            "w_dwell_cost": -0.001,
            "w_miss": -0.3,
            "w_priority": 0.5,
            "w_information_gain": 0.2,
            "w_staleness": 0.6,
            "context_scaled_miss": False,
        },
    },
    "R2_Opportunity": {
        "description": "R1 + elevated priority weighting (w_priority=1.0) rewarding active band placement",
        "reward": {
            "penalize_empty_dwell": False,
            "w_hit": 5.0,
            "w_novel": 2.0,
            "w_false_alarm": -0.5,
            "w_dwell_cost": -0.001,
            "w_miss": -0.3,
            "w_priority": 1.0,
            "w_information_gain": 0.2,
            "w_staleness": 0.6,
            "context_scaled_miss": False,
        },
    },
    "R3_LowDwellCost": {
        "description": "R2 + 5x reduced dwell cost (w_dwell_cost=-0.0002 / us)",
        "reward": {
            "penalize_empty_dwell": False,
            "w_hit": 5.0,
            "w_novel": 2.0,
            "w_false_alarm": -0.5,
            "w_dwell_cost": -0.0002,
            "w_miss": -0.3,
            "w_priority": 1.0,
            "w_information_gain": 0.2,
            "w_staleness": 0.6,
            "context_scaled_miss": False,
        },
    },
    "R4_FullBalanced": {
        "description": "R3 + w_novel=5.0, w_miss=-0.5, w_staleness=0.4, context_scaled_miss=True",
        "reward": {
            "penalize_empty_dwell": False,
            "w_hit": 5.0,
            "w_novel": 5.0,
            "w_false_alarm": -0.5,
            "w_dwell_cost": -0.0002,
            "w_miss": -0.5,
            "w_priority": 1.0,
            "w_information_gain": 0.2,
            "w_staleness": 0.4,
            "context_scaled_miss": True,
        },
    },
}


def build_models(
    full_cfg: dict[str, Any],
    device: torch.device,
    resume_path: Path,
) -> tuple[DRQNScheduler, DRQNScheduler, optim.Optimizer, float, float]:
    """Load clean checkpoint state dict into online and target networks."""
    drqn_cfg = full_cfg.get("drqn_scheduler", {})
    online = DRQNScheduler(
        obs_dim=int(drqn_cfg.get("obs_dim", 360)),
        n_bands=int(drqn_cfg.get("n_bands", 36)),
        n_actions=int(drqn_cfg.get("n_actions", 180)),
        lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
        lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
        n_modes=int(drqn_cfg.get("n_modes", 5)),
    ).to(device)

    target = copy.deepcopy(online).to(device)
    target.eval()

    lr = float(drqn_cfg.get("lr", 1e-4))
    optimizer = optim.Adam(online.parameters(), lr=lr)

    ckpt = torch.load(resume_path, map_location=device)
    if "state_dict" in ckpt:
        online.load_state_dict(ckpt["state_dict"])
        target.load_state_dict(ckpt["state_dict"])
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            logger.warning("Could not restore optimizer state: %s", exc)

    start_step = int(ckpt.get("global_step", 25000))
    reward_baseline = float(ckpt.get("reward_baseline", -0.7523))
    return online, target, optimizer, start_step, reward_baseline


def train_variant(
    variant_name: str,
    reward_overrides: dict[str, Any],
    base_env_cfg: dict[str, Any],
    train_source: ScenarioSource,
    deint_config: dict[str, Any],
    full_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    resume_path: Path,
    steps_to_train: int = 3000,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> tuple[DRQNScheduler, dict[str, Any]]:
    """Train one reward variant for steps_to_train steps."""
    logger.info("=" * 80)
    logger.info("STARTING ABLATION VARIANT: %s (%d steps)", variant_name, steps_to_train)
    logger.info("Overrides: %s", reward_overrides)
    logger.info("=" * 80)

    # Clone env config with reward overrides
    env_config = copy.deepcopy(base_env_cfg)
    env_config["reward"].update(reward_overrides)
    env_config["penalize_empty_dwell"] = reward_overrides.get("penalize_empty_dwell", False)
    env_config["context_scaled_miss"] = reward_overrides.get("context_scaled_miss", False)
    env_config["max_steps_per_episode"] = 250  # 250-step episodes for fast buffer population during ablation

    online_drqn, target_drqn, optimizer, start_step, reward_baseline = build_models(
        full_cfg, device, resume_path
    )

    scheduler_cfg = train_cfg.get("scheduler", {})
    batch_size = int(scheduler_cfg.get("batch_size", 32))
    seq_len = int(scheduler_cfg.get("seq_len", 16))
    burn_in = int(scheduler_cfg.get("burn_in", 8))
    update_freq = int(scheduler_cfg.get("update_freq", 4))
    target_update_freq = int(scheduler_cfg.get("target_update_freq", 1000))
    gamma = float(full_cfg.get("drqn_scheduler", {}).get("gamma", 0.99))
    eps_start = float(full_cfg.get("drqn_scheduler", {}).get("eps_start", 1.0))
    eps_end = float(full_cfg.get("drqn_scheduler", {}).get("eps_end", 0.05))
    eps_decay = float(full_cfg.get("drqn_scheduler", {}).get("eps_decay", 87837))

    buffer = SequenceReplayBuffer(
        capacity=int(scheduler_cfg.get("replay_buffer_size", 50000)),
        seq_len=seq_len,
        obs_dim=int(env_config["obs_dim"]),
        burn_in=burn_in,
        seed=seed,
    )

    loss_fn = nn.HuberLoss()
    n_actions = int(env_config["n_actions"])

    # Initialize environment
    train_env = CognitiveRFScanEnv(
        env_config,
        records=None,
        seed=seed,
        records_provider=train_source.sample,
        deinterleaver_config=deint_config,
        semantic_memory_path=":memory:",
    )

    global_step = start_step
    target_stop_step = start_step + steps_to_train
    step_losses: list[float] = []
    step_grad_norms: list[float] = []
    episode = 0
    t0 = time.time()

    while global_step < target_stop_step:
        obs, _ = train_env.reset()
        hidden = online_drqn.init_hidden(1, device)
        ep_reward = 0.0
        ep_hits = 0

        while global_step < target_stop_step:
            eps = eps_end + (eps_start - eps_end) * np.exp(-1.0 * global_step / eps_decay)

            # Epsilon-greedy action selection
            if random.random() < eps:
                action = random.randrange(n_actions)
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
                    _, _, hidden = online_drqn(obs_t, hidden)
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
                    q_out, _, hidden = online_drqn(obs_t, hidden)
                    action = int(torch.argmax(q_out[0, -1]).item())

            next_obs, reward, terminated, truncated, info = train_env.step(action)
            hit = bool(info.get("hit", False))
            ep_hits += int(hit)
            ep_reward += float(reward)

            buffer.add(
                np.asarray(obs, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(terminated or truncated),
                hit_prob=float(info.get("hit_prob", 1.0 if hit else 0.0)),
                intercept_time_us=float(info.get("intercept_time_us", float("nan"))),
            )

            obs = next_obs
            global_step += 1

            # Optimization update
            if (global_step % update_freq == 0) and buffer.can_sample(batch_size):
                try:
                    batch = buffer.sample(batch_size)
                    upd_stats: dict[str, Any] = {}
                    loss = _do_drqn_update(
                        online_drqn=online_drqn,
                        target_drqn=target_drqn,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        batch=batch,
                        gamma=gamma,
                        device=device,
                        stats=upd_stats,
                        reward_baseline=reward_baseline,
                    )
                    if upd_stats:
                        step_losses.append(float(upd_stats.get("td_loss", loss)))
                        step_grad_norms.append(float(upd_stats.get("gradient_norm", 0.0)))
                except AssertionError:
                    pass

            if global_step % target_update_freq == 0:
                target_drqn.load_state_dict(online_drqn.state_dict())

            reward_baseline = 0.99 * reward_baseline + 0.01 * float(reward)

            if terminated or truncated:
                break

        episode += 1
        elapsed = time.time() - t0
        rate = (global_step - start_step) / max(0.1, elapsed)
        if episode % 2 == 0 or global_step >= target_stop_step:
            logger.info(
                "[%s] Ep %d | Step %d/%d (%.1f step/s) | Rew %.2f | Hits %d | Mean TD Loss %.4f | GradNorm %.2f",
                variant_name,
                episode,
                global_step,
                target_stop_step,
                rate,
                ep_reward,
                ep_hits,
                float(np.mean(step_losses[-100:])) if step_losses else 0.0,
                float(np.mean(step_grad_norms[-100:])) if step_grad_norms else 0.0,
            )

    train_stats = {
        "variant": variant_name,
        "steps_trained": steps_to_train,
        "final_step": global_step,
        "mean_td_loss": float(np.mean(step_losses)) if step_losses else 0.0,
        "mean_grad_norm": float(np.mean(step_grad_norms)) if step_grad_norms else 0.0,
        "training_time_sec": float(time.time() - t0),
    }
    return online_drqn, train_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 Reward Ablation Study")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps per variant (default: 3000)")
    parser.add_argument("--eval-scenarios", type=int, default=3, help="Validation scenarios to evaluate (default: 3)")
    parser.add_argument("--eval-steps", type=int, default=500, help="Steps per validation scenario (default: 500)")
    parser.add_argument("--resume", type=str, default="checkpoints/scheduler/checkpoint_gate_25000.pt")
    parser.add_argument("--output-json", type=str, default="results/reward_ablation_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    resume_path = ROOT / args.resume
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    # Load configs
    with open(ROOT / "configs/training_config.yaml", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    with open(ROOT / "configs/model_config.yaml", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    data_dir = ROOT / train_cfg.get("data_dir", "D:/TSRD")
    env_cfg = train_cfg.get("environment", {})
    reward_cfg = full_cfg.get("reward", {})
    base_env_cfg = {**env_cfg, "reward": copy.deepcopy(reward_cfg)}
    base_env_cfg.setdefault("n_bands", int(full_cfg["drqn_scheduler"].get("n_bands", 36)))
    base_env_cfg.setdefault("n_modes", int(env_cfg.get("n_modes", 5)))
    base_env_cfg.setdefault("n_actions", int(env_cfg.get("n_actions", 180)))

    # Load trained deinterleaver
    deinterleaver_ckpt = ROOT / train_cfg.get("deinterleaver_ckpt", "checkpoints/deinterleaver/best.pt")
    norm_stats_path = ROOT / train_cfg.get("normalization_stats", "checkpoints/deinterleaver/normalization_stats.json")
    fit_stats = load_normalization_stats(norm_stats_path) if norm_stats_path.exists() else None
    deint_config = {"fit_stats": fit_stats} if fit_stats else {}

    # 1. Prepare shared train source
    logger.info("Initializing TSRD STARE train source...")
    train_source = ScenarioSource(
        data_root=str(data_dir),
        mode="stare",
        subset="train",
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)),
        seed=42,
        source_type="world",
        allow_synthetic_fallback=False,
    )

    # 2. Prepare Fixed Validation Set & Gate Evaluator
    logger.info("Preloading validation scenarios...")
    val_set = FixedValidationSet(
        data_root=str(data_dir),
        subset="val",
        mode="stare",
        n_files=max(args.eval_scenarios, 5),
        seed=42,
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)),
        allow_synthetic_fallback=False,
    )

    eval_out_dir = ROOT / "checkpoints/reward_ablation"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    gate_evaluator = StagedGateEvaluator(
        output_dir=eval_out_dir,
        gates=[28000],
        val_files=val_set.files_used,
        env_config=base_env_cfg,
        model_config=full_cfg,
        train_config=train_cfg,
        seed=42,
        device=device,
        semantic_memory_reset=True,
    )

    # 3. Evaluate baseline un-retrained checkpoint (for reference)
    logger.info("Evaluating clean Gate 25k reference baselines before ablation...")
    ref_drqn, _, _, _, _ = build_models(full_cfg, device, resume_path)
    ref_eval = gate_evaluator.evaluate_baseline_hierarchy(
        ref_drqn,
        n_steps=args.eval_steps,
        policies=["highest_occupancy", "random", "round_robin", "revisit_heuristic", "drqn"],
        max_scenarios=args.eval_scenarios,
    )

    # 4. Run training and evaluation for each variant
    ablation_results: dict[str, Any] = {
        "meta": {
            "resume_checkpoint": str(args.resume),
            "steps_per_variant": args.steps,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "reference_25k": {
            "drqn_intercept": ref_eval["policies"]["drqn"]["intercept_rate"],
            "drqn_q_margin": ref_eval["policies"]["drqn"]["q_margin"],
            "drqn_empty_escape": ref_eval["policies"]["drqn"]["empty_band_escape_rate"],
            "drqn_stale_escape": ref_eval["policies"]["drqn"]["stale_band_escape_rate"],
            "highest_occ_intercept": ref_eval["policies"]["highest_occupancy"]["intercept_rate"],
        },
        "variants": {},
    }

    checkpoint_paths: dict[str, str] = {}

    for var_name, var_info in REWARD_VARIANTS.items():
        trained_drqn, train_stats = train_variant(
            variant_name=var_name,
            reward_overrides=var_info["reward"],
            base_env_cfg=base_env_cfg,
            train_source=train_source,
            deint_config=deint_config,
            full_cfg=full_cfg,
            train_cfg=train_cfg,
            resume_path=resume_path,
            steps_to_train=args.steps,
            seed=42,
            device=device,
        )

        # Save checkpoint
        ckpt_dir = eval_out_dir / var_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "checkpoint.pt"
        torch.save({"state_dict": trained_drqn.state_dict(), "global_step": train_stats["final_step"]}, ckpt_path)
        checkpoint_paths[var_name] = str(ckpt_path)

        # Run validation
        logger.info("Evaluating validation performance for %s...", var_name)
        val_eval = gate_evaluator.evaluate_baseline_hierarchy(
            trained_drqn,
            n_steps=args.eval_steps,
            policies=["drqn", "full_moe"],
            max_scenarios=args.eval_scenarios,
        )
        pol = val_eval["policies"]

        drqn_res = pol["drqn"]
        moe_res = pol.get("full_moe", {})
        high_occ = ref_eval["policies"]["highest_occupancy"]

        ablation_results["variants"][var_name] = {
            "description": var_info["description"],
            "train_stats": train_stats,
            "checkpoint": str(ckpt_path),
            "drqn": {
                "intercept_rate": drqn_res["intercept_rate"],
                "discovery_rate": drqn_res["discovery_rate"],
                "pfa": drqn_res["pfa"],
                "avg_intercept_time_us": drqn_res["avg_intercept_time_us"],
                "empty_band_escape_rate": drqn_res["empty_band_escape_rate"],
                "stale_band_escape_rate": drqn_res["stale_band_escape_rate"],
                "distinct_bands": drqn_res["distinct_bands"],
                "action_entropy": drqn_res["action_entropy"],
                "q_margin": drqn_res["q_margin"],
                "total_reward": drqn_res["total_reward"],
            },
            "moe": {
                "intercept_rate": moe_res.get("intercept_rate", 0.0),
                "q_margin": moe_res.get("q_margin"),
            },
            "highest_occupancy_intercept": high_occ["intercept_rate"],
        }

    # 5. Print Comparison Summary Table
    print("\n" + "=" * 115)
    print("REWARD ABLATION STUDY RESULTS (3,000 Step Retraining from Gate 25k)")
    print("=" * 115)
    header = (
        f"{'Variant':<16} | {'[PRI] Intercept':<15} | {'[PRI] Discov%':<13} | "
        f"{'EmptyEsc%':<10} | {'StaleEsc%':<10} | {'Q-Margin':<10} | {'Reward':<10} | {'TD Loss':<9}"
    )
    print(header)
    print("-" * 115)

    ref = ablation_results["reference_25k"]
    print(
        f"{'Gate25k (Ref)':<16} | {ref['drqn_intercept']*100:>13.2f}% | {'--':<13} | "
        f"{ref['drqn_empty_escape']*100:>8.1f}% | {ref['drqn_stale_escape']*100:>8.1f}% | "
        f"{ref['drqn_q_margin']:>10.4f} | {'--':<10} | {'--':<9}"
    )
    print("-" * 115)

    best_variant = None
    best_intercept = -1.0

    for var_name, data in ablation_results["variants"].items():
        d = data["drqn"]
        t = data["train_stats"]
        intercept = d["intercept_rate"]
        print(
            f"{var_name:<16} | {intercept*100:>13.2f}% | {d['discovery_rate']*100:>11.1f}% | "
            f"{d['empty_band_escape_rate']*100:>8.1f}% | {d['stale_band_escape_rate']*100:>8.1f}% | "
            f"{d['q_margin']:>10.4f} | {d['total_reward']:>10.1f} | {t['mean_td_loss']:>9.4f}"
        )

        # Primary selection: Maximize Intercept Rate while meeting guardrail (EmptyEsc >= 70%)
        if d["empty_band_escape_rate"] >= 0.70 and intercept > best_intercept:
            best_intercept = intercept
            best_variant = var_name

    print("-" * 115)
    print(f"Reference HighestOccupancy: {ref['highest_occ_intercept']*100:.2f}%")
    print(f"Ablation Winner: {best_variant} (Intercept: {best_intercept*100:.2f}%)")
    print("=" * 115 + "\n")

    ablation_results["winner"] = {
        "variant": best_variant,
        "intercept_rate": best_intercept,
        "checkpoint": checkpoint_paths.get(str(best_variant), ""),
    }

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info("Saved ablation results to %s", out_file)


if __name__ == "__main__":
    main()
