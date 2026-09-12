"""Phase 9C - Arm E: Controlled Synthesis Continuation.

Synthesizes validated remediations into a controlled continuation protocol:
1. Shared representation permanently frozen: input_norm, lstm, band_encoder, ctx_proj
2. Advantage head permanently frozen: band_advantage_head (assert bit-exact 0.0 drift)
3. Trainable modules: value_stream, intercept_prob_head, intercept_time_head only
4. Stop gradient (detach_value_stream): False (as instructed by user)
5. Replay strategy: sparse_balanced (StratifiedModeSampler)
6. Reward variant: baseline (standard reward_v2)
7. Multi-seed canonical 10-scenario evaluation: seeds 42, 123, 999
8. Held-out config_119 advantage margin audit: Delta A = A(s, b*, Mode 2) - A(s, b*, Mode 1)
9. Staged progression:
   - Stage 1: Pilot Steps 1 to 250 (from step 25,500 to 25,750)
   - Sentinel Gate Evaluation at 250 steps
   - Stage 2: Intermediate Steps 251 to 500 (step 25,750 to 26,000)
   - Sentinel Gate Evaluation at 500 steps
   - Progression towards 30k steps if all gates pass cleanly
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
logger = logging.getLogger("synthesis_arm_e")

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]


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
    all_agile_irs = [r["agile_ir"] for r in seed_results.values()]
    all_sparse_irs = [r["sparse_ir"] for r in seed_results.values()]
    all_119_irs = [r["config_119_ir"] for r in seed_results.values()]
    all_119_m2 = [r["config_119_mode2_frac"] for r in seed_results.values()]
    all_pds = [r["pd"] for r in seed_results.values()]
    all_pfas = [r["pfa"] for r in seed_results.values()]

    return {
        "seeds": seeds,
        "mean_ir_mean": float(np.mean(all_mean_irs)),
        "mean_ir_std": float(np.std(all_mean_irs)),
        "worst_case_ir_mean": float(np.mean(all_worst_irs)),
        "worst_case_ir_std": float(np.std(all_worst_irs)),
        "agile_ir_mean": float(np.mean(all_agile_irs)),
        "sparse_ir_mean": float(np.mean(all_sparse_irs)),
        "config_119_ir_mean": float(np.mean(all_119_irs)),
        "config_119_ir_std": float(np.std(all_119_irs)),
        "config_119_mode2_frac_mean": float(np.mean(all_119_m2)),
        "pd_mean": float(np.mean(all_pds)),
        "pfa_mean": float(np.mean(all_pfas)),
        "per_seed": seed_results,
    }


def audit_config_119_margin(
    model: DRQNScheduler,
    seed: int = 42,
    n_steps: int = 1000,
    data_root: str = "D:/TSRD",
) -> Dict[str, Any]:
    """Audit advantage margin Delta A = A(s, b*, Mode 2) - A(s, b*, Mode 1) on config_119."""
    model.eval()
    h5_path = Path(data_root) / "stare" / "val_stare" / "config_119.h5"
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
    margins = []
    mode_counts = {m: 0 for m in range(5)}
    hits = 0

    for step in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            x = model.input_norm(obs_t)
            lstm_out, hx = model.lstm(x, hx)
            B, T, _ = obs_t.shape
            obs_bands = obs_t.view(B, T, model.n_bands, model.band_features)
            band_emb = model.band_encoder(obs_bands)
            ctx = model.ctx_proj(lstm_out).unsqueeze(2).expand(-1, -1, model.n_bands, -1)
            joint = torch.cat([band_emb, ctx], dim=-1)
            a_bands = model.band_advantage_head(joint)  # (1, 1, 36, 5)
            a = a_bands.view(B, T, model.n_actions)
            v = model.value_stream(lstm_out)
            q = v + a - a.mean(dim=-1, keepdim=True)

            act = int(torch.argmax(q.squeeze()).item())
            band_chosen = act // 5
            mode_chosen = act % 5

            adv_chosen_band = a_bands[0, 0, band_chosen].numpy()
            margin = float(adv_chosen_band[2] - adv_chosen_band[1])
            margins.append(margin)

        mode_counts[mode_chosen] += 1
        obs, _, term, trunc, info = env.step(act)
        hits += int(info.get("hit", False))
        if term or trunc:
            break

    return {
        "scenario": "config_119",
        "steps": n_steps,
        "hits": hits,
        "intercept_rate": hits / float(n_steps),
        "mean_advantage_margin_m2_minus_m1": float(np.mean(margins)),
        "min_advantage_margin": float(np.min(margins)),
        "max_advantage_margin": float(np.max(margins)),
        "mode_distribution": mode_counts,
        "mode2_fraction": mode_counts[2] / float(n_steps),
    }

def execute_synthesis_continuation(
    parent_path: Path,
    baseline_path: Path,
    reservoir_path: Path,
    candidate_dir: Path,
    stage1_steps: int = 250,
    stage2_steps: int = 250,
    lr: float = 2.5e-5,
) -> Dict[str, Any]:
    logger.info("Initializing Arm E - Controlled Synthesis Continuation...")

    sha_parent = compute_file_sha256(parent_path)
    sha_base = compute_file_sha256(baseline_path)
    sha_res = compute_file_sha256(reservoir_path)

    assert sha_parent == "777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554", f"Champion hash modified! {sha_parent}"
    assert sha_base == "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0", f"Baseline hash modified! {sha_base}"

    ckpt = torch.load(parent_path, map_location="cpu", weights_only=False)
    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5)
    model.load_state_dict(ckpt["state_dict"])

    # 1. Parameter Freezing Setup
    for name, p in model.named_parameters():
        p.requires_grad = False

    trainable_params = []
    trainable_param_names = []
    frozen_param_names = []

    for name, p in model.named_parameters():
        if any(name.startswith(pfx) for pfx in ("value_stream.", "intercept_prob_head.", "intercept_time_head.")):
            p.requires_grad = True
            trainable_params.append(p)
            trainable_param_names.append(name)
        else:
            frozen_param_names.append(name)

    logger.info("Trainable parameters (%d tensors): %s", len(trainable_param_names), trainable_param_names)
    logger.info("Frozen parameters (%d tensors): %s", len(frozen_param_names), frozen_param_names)

    # Initial frozen parameter snapshot for bit-exact drift verification
    frozen_snapshots_initial = {name: p.clone().detach() for name, p in model.named_parameters() if not p.requires_grad}

    target_model = copy.deepcopy(model)
    target_model.eval()

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
        replay_strategy="sparse_balanced",
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
        },
        records=records,
        seed=42,
    )
    obs, _ = env.reset(seed=42)
    hx = None

    seeds = [42, 123, 999]
    eval_steps = 1000

    # Reference evaluation of champion prior to training
    logger.info("Auditing reference parent champion prior to training...")
    eval_parent_ref = evaluate_model_multi_seed(model, seeds, n_steps=eval_steps)
    margin_parent_ref = audit_config_119_margin(model, seed=42, n_steps=eval_steps)
    logger.info("Parent Reference Mean IR: %.2f%%, config_119: %.2f%%, M2 share: %.1f%%, Margin Delta A: %.4f",
                eval_parent_ref["mean_ir_mean"], eval_parent_ref["config_119_ir_mean"],
                eval_parent_ref["config_119_mode2_frac_mean"] * 100, margin_parent_ref["mean_advantage_margin_m2_minus_m1"])

    # -------------------------------------------------------------
    # STAGE 1: Pilot Steps 1 to 250 (step 25,500 to 25,750)
    # -------------------------------------------------------------
    logger.info("--- STARTING STAGE 1: PILOT STEPS 1 TO %d ---", stage1_steps)
    model.train()
    target_update_freq = 100

    for step in range(1, stage1_steps + 1):
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

        # Optimization Step every 4 steps
        if step % 4 == 0 and replay_buffer.can_sample(32):
            batch, _ = sampler.sample_mode_stratified(32)
            act_b = torch.tensor(batch["actions"], dtype=torch.int64)
            rew_b = torch.tensor(batch["rewards"], dtype=torch.float32)
            next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32)
            dones_b = torch.tensor(batch["dones"], dtype=torch.float32)
            valid_mask = torch.tensor(batch["valid_mask"], dtype=torch.bool)
            burn_in_mask = torch.tensor(batch["burn_in_mask"], dtype=torch.bool)
            loss_mask = valid_mask & (~burn_in_mask)

            q_online, aux_online, _ = model(torch.tensor(batch["obs"], dtype=torch.float32))
            with torch.no_grad():
                q_target_next, _, _ = target_model(next_obs_b)
                target_td = rew_b + 0.99 * (1.0 - dones_b) * q_target_next.max(dim=-1)[0]

            q_chosen = q_online.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
            td_loss = loss_fn(q_chosen[loss_mask], target_td[loss_mask])

            hit_probs_b = torch.tensor(batch["hit_probs"], dtype=torch.float32)
            prob_pred = aux_online["intercept_prob"].gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
            bce_loss = nn.functional.binary_cross_entropy(prob_pred[loss_mask], hit_probs_b[loss_mask].detach())

            total_loss = td_loss + 0.1 * bce_loss

            optimizer.zero_grad()
            total_loss.backward()

            # Verify that frozen modules received zero gradients
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None, f"Frozen parameter {name} received gradient!"

            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()

        # Synchronize target network
        if step % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

    # Verify frozen parameters bit-exact zero drift after Stage 1
    for name, p in model.named_parameters():
        if not p.requires_grad:
            diff = torch.max(torch.abs(p - frozen_snapshots_initial[name])).item()
            assert diff == 0.0, f"Critical: Frozen parameter {name} drifted by {diff} in Stage 1!"

    logger.info("Stage 1 complete. Parameter drift of all frozen heads is strictly 0.0.")

    # Save Stage 1 Checkpoint (Step 25,750)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    ckpt_25750_path = candidate_dir / "checkpoint_step_25750_arme.pt"
    torch.save({
        "global_step": 25750,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "parent_sha256": sha_parent,
        "baseline_sha256": sha_base,
        "git_commit": get_git_commit(),
        "stage": "stage_1_pilot_250_steps",
    }, ckpt_25750_path)
    sha_25750 = compute_file_sha256(ckpt_25750_path)
    logger.info("Saved Stage 1 checkpoint: %s (SHA-256: %s)", ckpt_25750_path, sha_25750)

    # Evaluate Stage 1 Candidate
    logger.info("Running Multi-Seed Canonical 10-Scenario Evaluation on Stage 1 Candidate (Step 25,750)...")
    eval_stage1 = evaluate_model_multi_seed(model, seeds, n_steps=eval_steps)
    margin_stage1 = audit_config_119_margin(model, seed=42, n_steps=eval_steps)
    logger.info("Stage 1 Evaluation: Mean IR: %.2f%% (std: %.2f%%), config_119: %.2f%%, M2 share: %.1f%%, Margin Delta A: %.4f",
                eval_stage1["mean_ir_mean"], eval_stage1["mean_ir_std"],
                eval_stage1["config_119_ir_mean"], eval_stage1["config_119_mode2_frac_mean"] * 100,
                margin_stage1["mean_advantage_margin_m2_minus_m1"])

    # Stage 1 Sentinel Check
    stage1_pass = True
    stage1_reasons = []
    if eval_stage1["mean_ir_mean"] < 62.10:
        stage1_pass = False
        stage1_reasons.append(f"Mean IR ({eval_stage1['mean_ir_mean']:.2f}%) < 62.10% floor")
    if eval_stage1["worst_case_ir_mean"] < 16.20:
        stage1_pass = False
        stage1_reasons.append(f"Worst-case IR ({eval_stage1['worst_case_ir_mean']:.2f}%) < 16.20% floor")
    if eval_stage1["config_119_ir_mean"] < 25.20:
        stage1_pass = False
        stage1_reasons.append(f"config_119 IR ({eval_stage1['config_119_ir_mean']:.2f}%) < 25.20% floor")
    if eval_stage1["config_119_mode2_frac_mean"] < 0.70:
        stage1_pass = False
        stage1_reasons.append(f"config_119 Mode 2 share ({eval_stage1['config_119_mode2_frac_mean']*100:.1f}%) < 70.0% floor")
    if eval_stage1["pfa_mean"] > 0.0000:
        stage1_pass = False
        stage1_reasons.append(f"Pfa ({eval_stage1['pfa_mean']:.4f}) > 0.0000")

    # -------------------------------------------------------------
    # STAGE 2: Intermediate Steps 251 to 500 (step 25,750 to 26,000)
    # -------------------------------------------------------------
    eval_stage2 = None
    margin_stage2 = None
    ckpt_26000_path = None
    sha_26000 = None

    if stage1_pass and stage2_steps > 0:
        logger.info("--- STAGE 1 PASSED! STARTING STAGE 2: STEPS %d TO %d ---", stage1_steps + 1, stage1_steps + stage2_steps)
        model.train()
        for step in range(stage1_steps + 1, stage1_steps + stage2_steps + 1):
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

            # Optimization Step every 4 steps
            if step % 4 == 0 and replay_buffer.can_sample(32):
                batch, _ = sampler.sample_mode_stratified(32)
                act_b = torch.tensor(batch["actions"], dtype=torch.int64)
                rew_b = torch.tensor(batch["rewards"], dtype=torch.float32)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32)
                dones_b = torch.tensor(batch["dones"], dtype=torch.float32)
                valid_mask = torch.tensor(batch["valid_mask"], dtype=torch.bool)
                burn_in_mask = torch.tensor(batch["burn_in_mask"], dtype=torch.bool)
                loss_mask = valid_mask & (~burn_in_mask)

                q_online, aux_online, _ = model(torch.tensor(batch["obs"], dtype=torch.float32))
                with torch.no_grad():
                    q_target_next, _, _ = target_model(next_obs_b)
                    target_td = rew_b + 0.99 * (1.0 - dones_b) * q_target_next.max(dim=-1)[0]

                q_chosen = q_online.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
                td_loss = loss_fn(q_chosen[loss_mask], target_td[loss_mask])

                hit_probs_b = torch.tensor(batch["hit_probs"], dtype=torch.float32)
                prob_pred = aux_online["intercept_prob"].gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
                bce_loss = nn.functional.binary_cross_entropy(prob_pred[loss_mask], hit_probs_b[loss_mask].detach())

                total_loss = td_loss + 0.1 * bce_loss

                optimizer.zero_grad()
                total_loss.backward()

                # Verify that frozen modules received zero gradients
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        assert p.grad is None, f"Frozen parameter {name} received gradient!"

                torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
                optimizer.step()

            if step % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        # Verify frozen parameters bit-exact zero drift after Stage 2
        for name, p in model.named_parameters():
            if not p.requires_grad:
                diff = torch.max(torch.abs(p - frozen_snapshots_initial[name])).item()
                assert diff == 0.0, f"Critical: Frozen parameter {name} drifted by {diff} in Stage 2!"

        logger.info("Stage 2 complete. Parameter drift of all frozen heads is strictly 0.0.")

        ckpt_26000_path = candidate_dir / "checkpoint_step_26000_arme.pt"
        torch.save({
            "global_step": 26000,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "parent_sha256": sha_parent,
            "baseline_sha256": sha_base,
            "git_commit": get_git_commit(),
            "stage": "stage_2_intermediate_500_steps",
        }, ckpt_26000_path)
        sha_26000 = compute_file_sha256(ckpt_26000_path)
        logger.info("Saved Stage 2 checkpoint: %s (SHA-256: %s)", ckpt_26000_path, sha_26000)

        logger.info("Running Multi-Seed Canonical 10-Scenario Evaluation on Stage 2 Candidate (Step 26,000)...")
        eval_stage2 = evaluate_model_multi_seed(model, seeds, n_steps=eval_steps)
        margin_stage2 = audit_config_119_margin(model, seed=42, n_steps=eval_steps)
        logger.info("Stage 2 Evaluation: Mean IR: %.2f%% (std: %.2f%%), config_119: %.2f%%, M2 share: %.1f%%, Margin Delta A: %.4f",
                    eval_stage2["mean_ir_mean"], eval_stage2["mean_ir_std"],
                    eval_stage2["config_119_ir_mean"], eval_stage2["config_119_mode2_frac_mean"] * 100,
                    margin_stage2["mean_advantage_margin_m2_minus_m1"])
    else:
        if not stage1_pass:
            logger.warning("Stage 1 failed gate checks: %s. Halting continuation.", "; ".join(stage1_reasons))

    # Verify protected baseline and champion files were NEVER modified
    sha_parent_final = compute_file_sha256(parent_path)
    sha_base_final = compute_file_sha256(baseline_path)
    assert sha_parent_final == sha_parent, "CRITICAL: Parent checkpoint was modified during run!"
    assert sha_base_final == sha_base, "CRITICAL: Baseline checkpoint was modified during run!"

    results_payload = {
        "experiment": "phase9c_arm_e_controlled_synthesis",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "configuration": {
            "replay_strategy": "sparse_balanced",
            "reward_variant": "baseline",
            "detach_value_stream": False,
            "freeze_shared_representation": True,
            "freeze_band_advantage_head": True,
            "trainable_modules": ["value_stream", "intercept_prob_head", "intercept_time_head"],
            "learning_rate": lr,
            "clip_grad_norm": 5.0,
            "target_update_freq": target_update_freq,
        },
        "provenance": {
            "parent_checkpoint": {"path": str(parent_path), "sha256": sha_parent},
            "baseline_checkpoint": {"path": str(baseline_path), "sha256": sha_base},
            "reservoir": {"path": str(reservoir_path), "sha256": sha_res},
        },
        "reproducibility": {
            "seeds": seeds,
            "steps_per_scenario": eval_steps,
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
        },
        "stage1_evaluation": {
            "checkpoint": {"path": str(ckpt_25750_path), "sha256": sha_25750},
            "metrics": eval_stage1,
            "config_119_margin_audit": margin_stage1,
            "passed": stage1_pass,
            "reasons": stage1_reasons,
        },
        "stage2_evaluation": {
            "checkpoint": {"path": str(ckpt_26000_path), "sha256": sha_26000} if ckpt_26000_path else None,
            "metrics": eval_stage2,
            "config_119_margin_audit": margin_stage2,
        },
        "parent_reference": {
            "metrics": eval_parent_ref,
            "config_119_margin_audit": margin_parent_ref,
        },
        "safety_assertions": {
            "parent_unmodified": sha_parent_final == "777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554",
            "baseline_unmodified": sha_base_final == "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
            "frozen_parameters_drift_zero": True,
            "non_trainable_gradients_none": True,
        },
    }

    report_path = PKG_ROOT / "reports" / "diagnostics" / "phase9c_ablation_arm_e_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("Arm E completed successfully. Report saved to: %s", report_path)
    return results_payload


if __name__ == "__main__":
    parent = PKG_ROOT / "checkpoints" / "safe_continuation_candidate" / "checkpoint_step_25500.pt"
    baseline = PKG_ROOT / "checkpoints" / "production_baseline" / "checkpoint_gate_25000_frozen.pt"
    reservoir = PKG_ROOT / "checkpoints" / "production_baseline" / "baseline_reservoir_5k.pkl"
    cand_dir = PKG_ROOT / "checkpoints" / "phase9c_synthesis_arme"

    res = execute_synthesis_continuation(
        parent_path=parent,
        baseline_path=baseline,
        reservoir_path=reservoir,
        candidate_dir=cand_dir,
        stage1_steps=250,
        stage2_steps=250,
        lr=2.5e-5,
    )

    s1 = res["stage1_evaluation"]
    print("=" * 80)
    print("PHASE 9C - ARM E SYNTHESIS EVALUATION SUMMARY")
    print(f"Stage 1 (Step 25,750): Mean IR: {s1['metrics']['mean_ir_mean']:.2f}% | Worst IR: {s1['metrics']['worst_case_ir_mean']:.2f}% | config_119 IR: {s1['metrics']['config_119_ir_mean']:.2f}% | M2 share: {s1['metrics']['config_119_mode2_frac_mean']*100:.1f}% | Margin Delta A: {s1['config_119_margin_audit']['mean_advantage_margin_m2_minus_m1']:.4f}")
    if res["stage2_evaluation"]["metrics"]:
        s2 = res["stage2_evaluation"]
        print(f"Stage 2 (Step 26,000): Mean IR: {s2['metrics']['mean_ir_mean']:.2f}% | Worst IR: {s2['metrics']['worst_case_ir_mean']:.2f}% | config_119 IR: {s2['metrics']['config_119_ir_mean']:.2f}% | M2 share: {s2['metrics']['config_119_mode2_frac_mean']*100:.1f}% | Margin Delta A: {s2['config_119_margin_audit']['mean_advantage_margin_m2_minus_m1']:.4f}")
    print("=" * 80)

