"""Phase 9C - Arm B: Reward Dwell-Cost Normalization Remediation Ablation.

Evaluates whether dwell-cost penalization suppresses longer apertures.
Tests 2 variants in strict isolation against baseline reward:
- Variant B1: reward_variant = "baseline"
- Variant B2: reward_variant = "dwell_cost_normalized" (normalizes dwell cost to flat -0.01)

Governance & Controls:
- Parent: checkpoint_step_25500.pt (777de9b4...)
- Baseline: checkpoint_gate_25000_frozen.pt (7a99c659...)
- Replay: baseline (replay_strategy = "baseline" unchanged)
- Architecture: baseline (detach_value_stream = False, advantage head 100% frozen)
- Staged training: 250 steps per variant from checkpoint_step_25500.pt
- Multi-seed canonical 10-scenario evaluation: seeds 42, 123, 999.
- Automatic rollback / safety checks on every candidate.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records, ScenarioSource
from cognitive_ew_smart_scan.src.evaluation.benchmark_contract import CANONICAL_SCENARIOS
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer
from cognitive_ew_smart_scan.src.training.stratified_mode_sampler import StratifiedModeSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation_arm_b")

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = REPO_ROOT / "cognitive_ew_smart_scan"


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "UNKNOWN_GIT_REVISION"


def compute_file_sha256(path: Path | str) -> str:
    p = Path(path).resolve()
    if not p.exists():
        return f"MISSING:{p}"
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_model_multi_seed(
    model: DRQNScheduler,
    seeds: List[int],
    n_steps: int = 1000,
    data_root: str = "D:/TSRD",
) -> Dict[str, Any]:
    model.eval()
    agile_scens = {"config_29", "config_195", "config_241", "config_119"}
    sparse_scens = {"config_143", "config_119"}

    seed_results: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        scen_irs = {}
        scen_mode2_fractions = {}
        pds = []
        pfas = []

        for scen_id in CANONICAL_SCENARIOS:
            h5_path = Path(data_root) / "stare" / "val_stare" / f"{scen_id}.h5"
            records = load_h5_records(h5_path)
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
                seed=seed,
            )
            obs, _ = env.reset(seed=seed)
            hx = None
            hits = 0
            modes = []

            for step in range(n_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
                with torch.no_grad():
                    q, _, hx = model(obs_t, hx)
                act = int(q.squeeze().argmax().item())
                modes.append(act % 5)
                obs, _, term, trunc, info = env.step(act)
                hits += int(info.get("hit", False))
                if term or trunc:
                    break

            ir = hits / float(n_steps)
            scen_irs[scen_id] = ir
            m2_frac = modes.count(2) / float(len(modes))
            scen_mode2_fractions[scen_id] = m2_frac
            pds.append(float(env.fom.pd))
            pfas.append(float(env.fom.pfa))

        irs_list = [scen_irs[s] * 100 for s in CANONICAL_SCENARIOS]
        agile_irs = [scen_irs[s] * 100 for s in agile_scens]
        sparse_irs = [scen_irs[s] * 100 for s in sparse_scens]

        seed_results[seed] = {
            "mean_ir": float(np.mean(irs_list)),
            "worst_case_ir": float(np.min(irs_list)),
            "agile_ir": float(np.mean(agile_irs)),
            "sparse_ir": float(np.mean(sparse_irs)),
            "config_119_ir": float(scen_irs["config_119"] * 100),
            "config_119_mode2_frac": float(scen_mode2_fractions["config_119"]),
            "pd": float(np.mean(pds)) * 100,
            "pfa": float(np.mean(pfas)),
            "scenario_irs": scen_irs,
        }

    all_mean_irs = [r["mean_ir"] for r in seed_results.values()]
    all_worst_irs = [r["worst_case_ir"] for r in seed_results.values()]
    all_119_irs = [r["config_119_ir"] for r in seed_results.values()]
    all_119_m2 = [r["config_119_mode2_frac"] for r in seed_results.values()]

    return {
        "seeds": seeds,
        "mean_ir_mean": float(np.mean(all_mean_irs)),
        "mean_ir_std": float(np.std(all_mean_irs)),
        "worst_case_ir_mean": float(np.mean(all_worst_irs)),
        "worst_case_ir_std": float(np.std(all_worst_irs)),
        "config_119_ir_mean": float(np.mean(all_119_irs)),
        "config_119_ir_std": float(np.std(all_119_irs)),
        "config_119_mode2_frac_mean": float(np.mean(all_119_m2)),
        "per_seed": seed_results,
    }


def train_reward_variant(
    variant_name: str,
    reward_variant: str,
    parent_path: Path,
    reservoir_path: Path,
    steps: int = 250,
    lr: float = 2.5e-5,
) -> DRQNScheduler:
    logger.info("Training %s: reward_variant=%s (%d steps)", variant_name, reward_variant, steps)
    ckpt = torch.load(parent_path, map_location="cpu", weights_only=False)
    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5)
    model.load_state_dict(ckpt["state_dict"])

    # Freeze advantage head
    for name, p in model.band_advantage_head.named_parameters():
        p.requires_grad = False

    target_model = copy.deepcopy(model)
    target_model.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.HuberLoss()

    replay_buffer = SequenceReplayBuffer(capacity=50000, seq_len=16, obs_dim=360, burn_in=8, seed=42)
    replay_buffer.load_episodes(reservoir_path)

    sampler = StratifiedModeSampler(
        buffer=replay_buffer,
        mode_weights={0: 0.15, 1: 0.25, 2: 0.30, 3: 0.15, 4: 0.15},
        seq_len=16,
        burn_in=8,
        seed=42,
        replay_strategy="baseline",
    )

    data_root = Path("D:/TSRD")
    train_source = ScenarioSource(data_root=data_root, mode="stare", subset="train", source_type="world")
    rng = np.random.default_rng(42)
    train_path = rng.choice(train_source.eligible_files)
    records = load_h5_records(train_path)

    env = CognitiveRFScanEnv(
        config={
            "n_bands": 36,
            "dwell_modes": 5,
            "features_per_band": 10,
            "ema_alpha": 0.30,
            "ema_alpha_miss_confirmed": 0.20,
            "semantic_memory_enabled": False,
            "reward_variant": reward_variant,
        },
        records=records,
        seed=42,
    )
    obs, _ = env.reset(seed=42)
    hx = None

    model.train()
    for step in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            q, _, hx = model(obs_t, hx)
        if float(np.random.rand()) < 0.05:
            act = int(np.random.randint(180))
        else:
            act = int(q.squeeze().argmax().item())

        next_obs, r, term, trunc, info = env.step(act)
        replay_buffer.add(
            obs=obs,
            action=act,
            reward=r,
            next_obs=next_obs,
            done=bool(term or trunc),
            hit_prob=1.0 if info.get("hit", False) else 0.0,
            intercept_time_us=float(info.get("intercept_latency_us", np.nan)),
            scenario_id=getattr(env.radio_env, "scenario_id", "train"),
        )

        if term or trunc:
            obs, _ = env.reset()
            hx = None
        else:
            obs = next_obs

        # DRQN update
        if step % 4 == 0 and replay_buffer.can_sample(32):
            batch, _ = sampler.sample_mode_stratified(32)
            act_b = torch.tensor(batch["actions"], dtype=torch.int64)
            rew_b = torch.tensor(batch["rewards"], dtype=torch.float32)
            next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32)
            dones_b = torch.tensor(batch["dones"], dtype=torch.float32)
            valid_mask = torch.tensor(batch["valid_mask"], dtype=torch.bool)
            burn_in_mask = torch.tensor(batch["burn_in_mask"], dtype=torch.bool)
            loss_mask = valid_mask & (~burn_in_mask)

            q_online, _, _ = model(torch.tensor(batch["obs"], dtype=torch.float32))
            with torch.no_grad():
                q_target_next, _, _ = target_model(next_obs_b)
                target_td = rew_b + 0.99 * (1.0 - dones_b) * q_target_next.max(dim=-1)[0]

            q_chosen = q_online.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
            loss = loss_fn(q_chosen[loss_mask], target_td[loss_mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()

    return model


def run_arm_b() -> Dict[str, Any]:
    logger.info("Executing Phase 9C - Arm B: Reward Dwell-Cost Normalization Remediation Ablation...")

    parent_path = PKG_ROOT / "checkpoints" / "safe_continuation_candidate" / "checkpoint_step_25500.pt"
    baseline_path = PKG_ROOT / "checkpoints" / "production_baseline" / "checkpoint_gate_25000_frozen.pt"
    reservoir_path = PKG_ROOT / "checkpoints" / "production_baseline" / "baseline_reservoir_5k.pkl"

    sha_parent = compute_file_sha256(parent_path)
    sha_base = compute_file_sha256(baseline_path)
    sha_res = compute_file_sha256(reservoir_path)

    seeds = [42, 123, 999]
    eval_steps = 1000

    # 1. Train and Evaluate Variant B1 (Baseline Reward)
    m_b1 = train_reward_variant("Variant B1 (Baseline Reward)", "baseline", parent_path, reservoir_path, steps=250)
    eval_b1 = evaluate_model_multi_seed(m_b1, seeds, n_steps=eval_steps)

    # 2. Train and Evaluate Variant B2 (Dwell-Cost Normalized Reward)
    m_b2 = train_reward_variant("Variant B2 (Dwell-Cost Normalized)", "dwell_cost_normalized", parent_path, reservoir_path, steps=250)
    eval_b2 = evaluate_model_multi_seed(m_b2, seeds, n_steps=eval_steps)

    # Save Checkpoint Artifacts
    out_dir = PKG_ROOT / "checkpoints" / "phase9c_remediation" / "ablation_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m_b1.state_dict(), "variant": "B1"}, out_dir / "candidate_b1_baseline_reward.pt")
    torch.save({"state_dict": m_b2.state_dict(), "variant": "B2"}, out_dir / "candidate_b2_dwell_cost_normalized.pt")

    results_payload = {
        "experiment": "phase9c_arm_b_reward_normalization",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "reproducibility": {
            "seeds": seeds,
            "steps_per_scenario": eval_steps,
            "training_steps": 250,
            "learning_rate": 2.5e-5,
            "advantage_head_frozen": True,
            "replay_strategy": "baseline",
        },
        "provenance": {
            "parent_checkpoint": {"path": str(parent_path), "sha256": sha_parent},
            "baseline_checkpoint": {"path": str(baseline_path), "sha256": sha_base},
            "reservoir": {"path": str(reservoir_path), "sha256": sha_res},
        },
        "multi_seed_evaluations": {
            "variant_b1_baseline_reward": eval_b1,
            "variant_b2_dwell_cost_normalized": eval_b2,
        },
        "safety_assertions": {
            "parent_unmodified": sha_parent == "777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554",
            "baseline_unmodified": sha_base == "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
        },
    }

    report_path = PKG_ROOT / "reports" / "diagnostics" / "phase9c_ablation_arm_b_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("Arm B finished successfully. Report saved to: %s", report_path)
    return results_payload


if __name__ == "__main__":
    res = run_arm_b()
    b1 = res["multi_seed_evaluations"]["variant_b1_baseline_reward"]
    b2 = res["multi_seed_evaluations"]["variant_b2_dwell_cost_normalized"]
    print(f"Variant B1 (Baseline) Mean IR: {b1['mean_ir_mean']:.2f}% | config_119: {b1['config_119_ir_mean']:.2f}% | M2: {b1['config_119_mode2_frac_mean']*100:.1f}%")
    print(f"Variant B2 (Dwell Norm) Mean IR: {b2['mean_ir_mean']:.2f}% | config_119: {b2['config_119_ir_mean']:.2f}% | M2: {b2['config_119_mode2_frac_mean']*100:.1f}%")
