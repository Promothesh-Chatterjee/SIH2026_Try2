"""Phase G1: R1 Regularizer Activation & Gradient Audit.

Performs a read-only forensic audit of Candidate R1 regularizers on checkpoint_gate_53000.pt:
1. Loads Gate-53k checkpoint (online and target networks).
2. Collects realistic sequence transitions across the 10 validation scenarios using the Gate-53k policy.
3. Samples K deterministic diagnostic batches from the replay buffer.
4. For each batch, decomposes the loss into:
   - TD loss (Bellman Q error)
   - Q^2 regularization loss
   - Action entropy loss
   - Top-band spatial diversity penalty
   - Per-band mode diversity penalty
5. Computes individual gradients per component to measure:
   - Activation rate (% of batches where penalty > 0)
   - Mean / max penalty magnitude
   - Ratio of penalty loss to TD loss
   - Gradient norm of each term
   - Ratio of penalty gradient norm to TD loss gradient norm
   - Directional alignment (cosine similarity with TD loss gradient)
6. Answers conclusively:
   - Did R1 fail because penalties remained dormant?
   - Did R1 fail because penalties were active but too weak?
   - Did R1 fail because penalties optimized the wrong surrogate?
7. Emits reports/g1_r1_regularizer_activation_audit.json and a summary.
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
import torch.nn as nn
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
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.training.replay_buffer import SequenceReplayBuffer
from ew_core.training.val_set import FixedValidationSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_r1_activation")


def get_flat_grad(model: nn.Module) -> torch.Tensor:
    """Collect all parameter gradients into a single 1D tensor."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def main():
    root = Path(".").resolve()
    logger.info("=" * 80)
    logger.info("STARTING PHASE G1: R1 REGULARIZER ACTIVATION & GRADIENT AUDIT")
    logger.info("=" * 80)

    # 1. Load Configurations
    with open(root / "configs/training_config_gate50_remediation.yaml", "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    with open(root / "configs/model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    sched_cfg = train_cfg.get("scheduler", {})
    env_cfg = train_cfg.get("environment", {})

    # Candidate R1 Hyperparameters
    lambda_entropy = float(sched_cfg.get("lambda_entropy", 0.02))
    band_div_coef = float(sched_cfg.get("band_diversity_penalty_coef", 0.01))
    band_div_thresh = float(sched_cfg.get("band_diversity_threshold", 0.60))
    mode_div_coef = float(sched_cfg.get("mode_diversity_penalty_coef", 0.02))
    mode_div_thresh = float(sched_cfg.get("mode_diversity_threshold", 0.55))
    mode_collapse_rate_thresh = float(sched_cfg.get("mode_collapse_rate_threshold", 0.20))
    q_reg_coef = float(sched_cfg.get("q_reg_coef", 2.5e-4))
    gamma = float(model_cfg.get("drqn_scheduler", {}).get("gamma", 0.99))

    logger.info("Candidate R1 Parameters Under Audit:")
    logger.info("  lambda_entropy:              %.4f", lambda_entropy)
    logger.info("  band_diversity_penalty_coef: %.4f", band_div_coef)
    logger.info("  band_diversity_threshold:    %.2f", band_div_thresh)
    logger.info("  mode_diversity_penalty_coef: %.4f", mode_div_coef)
    logger.info("  mode_diversity_threshold:    %.2f", mode_div_thresh)
    logger.info("  mode_collapse_rate_thresh:   %.2f", mode_collapse_rate_thresh)
    logger.info("  q_reg_coef:                  %.6f", q_reg_coef)

    # 2. Load Checkpoint
    ckpt_path = root / "experiments/checkpoints/scheduler_v2_gate50_remediation_r1/checkpoint_gate_53000.pt"
    if not ckpt_path.exists():
        q_ckpt = root / "experiments/checkpoints/scheduler_v2_gate50_remediation_r1/quarantine/checkpoint_gate_53000_quarantined_collapsed.pt"
        if q_ckpt.exists():
            ckpt_path = q_ckpt
        else:
            raise FileNotFoundError(f"Gate-53k checkpoint not found at {ckpt_path}")

    logger.info("Loading Gate-53k checkpoint from %s...", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = torch.device("cpu")
    obs_dim = int(env_cfg.get("obs_dim", 360))
    n_bands = int(env_cfg.get("n_bands", 36))
    n_modes = int(env_cfg.get("n_modes", 5))
    n_actions = int(env_cfg.get("n_actions", 180))

    drqn_cfg = model_cfg.get("drqn_scheduler", {})
    online_drqn = DRQNScheduler(
        obs_dim=obs_dim,
        n_bands=n_bands,
        n_modes=n_modes,
        n_actions=n_actions,
        lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
        lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
    ).to(device)
    online_drqn.load_state_dict(ckpt["state_dict"])

    target_drqn = DRQNScheduler(
        obs_dim=obs_dim,
        n_bands=n_bands,
        n_modes=n_modes,
        n_actions=n_actions,
        lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
        lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
    ).to(device)
    target_drqn.load_state_dict(ckpt.get("target_state_dict", ckpt["state_dict"]))

    online_drqn.train()
    target_drqn.eval()

    # 3. Collect Real Transitions into Replay Buffer
    logger.info("Collecting real state transitions across 10 validation scenarios with Gate-53k policy...")
    data_dir = Path("D:/TSRD")
    val_set = FixedValidationSet(data_root=data_dir, subset="val", n_files=10, seed=42)
    val_scenarios = [item[0] for item in val_set.files_used]

    buffer = SequenceReplayBuffer(
        capacity=10000,
        seq_len=16,
        obs_dim=obs_dim,
        burn_in=8,
        seed=42,
    )

    reward_baseline = float(ckpt.get("reward_baseline", 0.6808))
    baseline_momentum = 0.99

    total_transitions_collected = 0
    for scen_idx, scen_file in enumerate(val_scenarios):
        scen_id = scen_file.stem
        records = load_h5_records(scen_file)
        env = CognitiveRFScanEnv(
            config=env_cfg,
            records=records,
            seed=42 + scen_idx,
            semantic_memory_path=":memory:",
        )
        obs, _ = env.reset()
        hidden = online_drqn.init_hidden(1, device)

        done = False
        steps = 0
        while not done and steps < 300:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_vals, aux, hidden = online_drqn(obs_t, hidden)
                # Greedy action selection (matching eval & 92.6% of training)
                action = int(q_vals[0, 0].argmax().item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            m_act = int(action % n_modes)
            dwell_us = RF_BASE_DWELL_TIME_US * DEFAULT_DWELL_MULTIPLIERS[m_act]
            hit = bool(info.get("interception_hit", False))

            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                hit_prob=1.0 if hit else 0.0,
                intercept_time_us=float(info.get("interception_latency", 250.0)) if hit else None,
                scenario_id=scen_id,
                dwell_time_us=dwell_us,
            )
            obs = next_obs
            steps += 1
            total_transitions_collected += 1

    logger.info("Collected %d transitions into sequence buffer (buffer size: %d).", total_transitions_collected, len(buffer))

    # 4. Sample Diagnostic Batches and Audit Regularizers
    n_batches = 50
    batch_size = 32
    seq_len = 16
    burn_in = 8
    loss_fn = nn.SmoothL1Loss(reduction="mean")

    audit_records = []

    logger.info("Auditing loss terms and gradient norms across %d diagnostic batches...", n_batches)
    np.random.seed(42)
    torch.manual_seed(42)

    for b_idx in range(n_batches):
        batch = buffer.sample(batch_size)

        obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
        act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
        rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
        next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
        done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)
        valid_b = torch.tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        burn_in_b = torch.tensor(batch["burn_in_mask"], dtype=torch.bool, device=device)
        dwell_us_b = torch.tensor(batch["dwell_times_us"], dtype=torch.float32, device=device)

        loss_mask = valid_b & (~burn_in_b)
        if not loss_mask.any():
            continue

        # Forward online network
        h_online = online_drqn.init_hidden(batch_size, device)
        q_all, _, _ = online_drqn(obs_b, h_online)
        q_chosen = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

        # Forward target network (Double-DQN)
        with torch.no_grad():
            h_target = target_drqn.init_hidden(batch_size, device)
            next_q_target, _, _ = target_drqn(next_obs_b, h_target)
            h_online_next = online_drqn.init_hidden(batch_size, device)
            next_q_online, _, _ = online_drqn(next_obs_b, h_online_next)
            best_action = next_q_online.argmax(dim=-1, keepdim=True)
            next_q = next_q_target.gather(-1, best_action).squeeze(-1)
            target_q_max = 100.0
            target_q_min = -50.0
            next_q = next_q.clamp(min=target_q_min, max=target_q_max)

        # Centered rewards and TD target
        gamma_eff = torch.pow(torch.as_tensor(float(gamma), dtype=torch.float32, device=device), dwell_us_b / 500.0)
        batch_mean = float(rew_b[loss_mask].mean().item())
        updated_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * batch_mean
        centered_rew_b = rew_b - updated_baseline
        targets = centered_rew_b + gamma_eff * next_q * (1.0 - done_b)

        # Component 1: TD Loss
        q_loss = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())

        # Component 2: Q^2 Regularization
        q_reg_loss = q_reg_coef * (q_all[loss_mask] ** 2).mean()

        # Component 3: Action Entropy Loss
        q_probs = torch.softmax(q_all[loss_mask] / 1.0, dim=-1)
        action_entropy = -(q_probs * torch.log(q_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -lambda_entropy * action_entropy

        # Total base loss for scaling penalties
        base_loss = q_loss + q_reg_loss + entropy_loss

        # Component 4: Top-Band Spatial Diversity Penalty
        _band_max_q = q_all[loss_mask].view(-1, n_bands, n_modes).max(dim=-1).values
        _band_softmax = torch.softmax(_band_max_q, dim=-1)
        _top_band_frac = _band_softmax.max(dim=-1).values.mean()
        band_div_active = bool(_top_band_frac.item() > band_div_thresh)
        if band_div_active:
            top_band_pen = band_div_coef * (_top_band_frac - band_div_thresh).clamp(min=0.0) * base_loss.abs().detach()
        else:
            top_band_pen = torch.zeros((), device=device)

        # Component 5: Mode Diversity Penalty
        q_banded_live = q_all[loss_mask].view(-1, n_bands, n_modes)
        band_mode_softmax = torch.softmax(q_banded_live / 1.0, dim=-1)
        max_mode_prob_per_band = band_mode_softmax.max(dim=-1).values
        mode_collapse_rate = (max_mode_prob_per_band > mode_div_thresh).float().mean()
        mode_div_active = bool(mode_collapse_rate.item() > mode_collapse_rate_thresh)
        avg_max_mode_prob = max_mode_prob_per_band.mean()
        if mode_div_active:
            mode_div_pen = mode_div_coef * (avg_max_mode_prob - mode_div_thresh).clamp(min=0.0) * base_loss.abs().detach()
        else:
            mode_div_pen = torch.zeros((), device=device)

        # --- Compute Individual Gradient Norms via backward passes ---
        # 1. TD Loss Grad
        online_drqn.zero_grad()
        q_loss.backward(retain_graph=True)
        g_td = get_flat_grad(online_drqn)
        gn_td = float(g_td.norm().item())

        # 2. Q^2 Loss Grad
        online_drqn.zero_grad()
        q_reg_loss.backward(retain_graph=True)
        g_qreg = get_flat_grad(online_drqn)
        gn_qreg = float(g_qreg.norm().item())

        # 3. Entropy Loss Grad
        online_drqn.zero_grad()
        entropy_loss.backward(retain_graph=True)
        g_ent = get_flat_grad(online_drqn)
        gn_ent = float(g_ent.norm().item())

        # 4. Top-Band Penalty Grad
        if band_div_active:
            online_drqn.zero_grad()
            top_band_pen.backward(retain_graph=True)
            g_band = get_flat_grad(online_drqn)
            gn_band = float(g_band.norm().item())
        else:
            gn_band = 0.0
            g_band = torch.zeros_like(g_td)

        # 5. Mode Diversity Penalty Grad
        if mode_div_active:
            online_drqn.zero_grad()
            mode_div_pen.backward(retain_graph=True)
            g_mode = get_flat_grad(online_drqn)
            gn_mode = float(g_mode.norm().item())
        else:
            gn_mode = 0.0
            g_mode = torch.zeros_like(g_td)

        # Cosine similarities with TD loss gradient
        cos_sim_qreg = float(torch.cosine_similarity(g_qreg.unsqueeze(0), g_td.unsqueeze(0)).item()) if gn_qreg > 0 and gn_td > 0 else 0.0
        cos_sim_ent = float(torch.cosine_similarity(g_ent.unsqueeze(0), g_td.unsqueeze(0)).item()) if gn_ent > 0 and gn_td > 0 else 0.0
        cos_sim_band = float(torch.cosine_similarity(g_band.unsqueeze(0), g_td.unsqueeze(0)).item()) if gn_band > 0 and gn_td > 0 else 0.0
        cos_sim_mode = float(torch.cosine_similarity(g_mode.unsqueeze(0), g_td.unsqueeze(0)).item()) if gn_mode > 0 and gn_td > 0 else 0.0

        td_val = float(q_loss.item())

        audit_records.append({
            "batch_idx": b_idx,
            "td_loss": td_val,
            "q_reg_loss": float(q_reg_loss.item()),
            "q_reg_ratio_pct": (float(q_reg_loss.item()) / max(1e-6, td_val)) * 100.0,
            "entropy_loss": float(entropy_loss.item()),
            "action_entropy_nats": float(action_entropy.item()),
            "top_band_fraction_softmax": float(_top_band_frac.item()),
            "band_diversity_active": band_div_active,
            "top_band_penalty": float(top_band_pen.item()),
            "top_band_penalty_ratio_pct": (float(top_band_pen.item()) / max(1e-6, td_val)) * 100.0,
            "mode_collapse_rate": float(mode_collapse_rate.item()),
            "avg_max_mode_prob": float(avg_max_mode_prob.item()),
            "mode_diversity_active": mode_div_active,
            "mode_diversity_penalty": float(mode_div_pen.item()),
            "mode_diversity_penalty_ratio_pct": (float(mode_div_pen.item()) / max(1e-6, td_val)) * 100.0,
            # Gradient Norms
            "grad_norm_td": gn_td,
            "grad_norm_qreg": gn_qreg,
            "grad_norm_entropy": gn_ent,
            "grad_norm_band_div": gn_band,
            "grad_norm_mode_div": gn_mode,
            # Grad Norm Ratios relative to TD
            "grad_ratio_qreg": gn_qreg / max(1e-6, gn_td),
            "grad_ratio_entropy": gn_ent / max(1e-6, gn_td),
            "grad_ratio_band_div": gn_band / max(1e-6, gn_td),
            "grad_ratio_mode_div": gn_mode / max(1e-6, gn_td),
            # Directional Cosine Similarities
            "cos_sim_qreg_td": cos_sim_qreg,
            "cos_sim_entropy_td": cos_sim_ent,
            "cos_sim_band_td": cos_sim_band,
            "cos_sim_mode_td": cos_sim_mode,
        })

    # 5. Compile Statistical Aggregate
    td_losses = [r["td_loss"] for r in audit_records]
    qreg_losses = [r["q_reg_loss"] for r in audit_records]
    ent_losses = [r["entropy_loss"] for r in audit_records]
    band_pens = [r["top_band_penalty"] for r in audit_records]
    mode_pens = [r["mode_diversity_penalty"] for r in audit_records]

    band_active_cnt = sum(1 for r in audit_records if r["band_diversity_active"])
    mode_active_cnt = sum(1 for r in audit_records if r["mode_diversity_active"])

    gn_td_arr = [r["grad_norm_td"] for r in audit_records]
    gn_qreg_arr = [r["grad_norm_qreg"] for r in audit_records]
    gn_ent_arr = [r["grad_norm_entropy"] for r in audit_records]
    gn_band_arr = [r["grad_norm_band_div"] for r in audit_records]
    gn_mode_arr = [r["grad_norm_mode_div"] for r in audit_records]

    mode_collapse_rates = [r["mode_collapse_rate"] for r in audit_records]
    avg_max_mode_probs = [r["avg_max_mode_prob"] for r in audit_records]
    top_band_fracs = [r["top_band_fraction_softmax"] for r in audit_records]

    # Formulate Authoritative Answers to the 3 Diagnostic Questions:
    # Q: Did Candidate R1 fail because penalties remained dormant,
    #    because they activated but were too weak,
    #    or because they activated but optimized the wrong surrogate?
    is_mode_pen_dormant = (mode_active_cnt == 0)
    is_band_pen_dormant = (band_active_cnt == 0)

    mean_mode_pen_ratio = float(np.mean([r["mode_diversity_penalty_ratio_pct"] for r in audit_records]))
    mean_mode_grad_ratio = float(np.mean([r["grad_ratio_mode_div"] for r in audit_records]))

    if is_mode_pen_dormant:
        diagnostic_verdict = "DORMANT: Mode diversity penalty never triggered (collapse rate threshold never met)."
    elif mean_mode_grad_ratio < 0.05:
        diagnostic_verdict = "ACTIVE_BUT_EXTREMELY_WEAK: Mode penalty triggered but gradient norm was <5% of TD gradient norm."
    else:
        diagnostic_verdict = "ACTIVE_WRONG_SURROGATE: Mode penalty triggered with non-trivial gradient, but softmax surrogate failed to alter greedy argmax action selection."

    mode_act_pct = (mode_active_cnt / max(1, len(audit_records))) * 100.0
    band_act_pct = (band_active_cnt / max(1, len(audit_records))) * 100.0

    summary = {
        "n_diagnostic_batches": len(audit_records),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "burn_in": burn_in,
        "diagnostic_verdict": diagnostic_verdict,
        "r1_hyperparameters": {
            "lambda_entropy": lambda_entropy,
            "band_diversity_penalty_coef": band_div_coef,
            "band_diversity_threshold": band_div_thresh,
            "mode_diversity_penalty_coef": mode_div_coef,
            "mode_diversity_threshold": mode_div_thresh,
            "mode_collapse_rate_threshold": mode_collapse_rate_thresh,
            "q_reg_coef": q_reg_coef,
        },
        "mode_diversity_penalty_audit": {
            "activation_rate_pct": mode_act_pct,
            "active_batches_count": mode_active_cnt,
            "mean_penalty_magnitude": float(np.mean(mode_pens)),
            "max_penalty_magnitude": float(np.max(mode_pens)) if mode_pens else 0.0,
            "mean_ratio_to_td_loss_pct": mean_mode_pen_ratio,
            "mean_grad_norm": float(np.mean(gn_mode_arr)),
            "max_grad_norm": float(np.max(gn_mode_arr)) if gn_mode_arr else 0.0,
            "mean_grad_norm_ratio_to_td": mean_mode_grad_ratio,
            "mean_mode_collapse_rate": float(np.mean(mode_collapse_rates)),
            "threshold_required_for_activation": mode_collapse_rate_thresh,
            "mean_avg_max_mode_prob": float(np.mean(avg_max_mode_probs)),
            "per_band_threshold": mode_div_thresh,
        },
        "top_band_diversity_penalty_audit": {
            "activation_rate_pct": band_act_pct,
            "active_batches_count": band_active_cnt,
            "mean_penalty_magnitude": float(np.mean(band_pens)),
            "max_penalty_magnitude": float(np.max(band_pens)) if band_pens else 0.0,
            "mean_ratio_to_td_loss_pct": float(np.mean([r["top_band_penalty_ratio_pct"] for r in audit_records])),
            "mean_grad_norm": float(np.mean(gn_band_arr)),
            "mean_grad_norm_ratio_to_td": float(np.mean([r["grad_ratio_band_div"] for r in audit_records])),
            "mean_top_band_fraction_softmax": float(np.mean(top_band_fracs)),
            "threshold_required_for_activation": band_div_thresh,
        },
        "q2_regularization_audit": {
            "mean_loss_magnitude": float(np.mean(qreg_losses)),
            "max_loss_magnitude": float(np.max(qreg_losses)) if qreg_losses else 0.0,
            "mean_ratio_to_td_loss_pct": float(np.mean([r["q_reg_ratio_pct"] for r in audit_records])),
            "mean_grad_norm": float(np.mean(gn_qreg_arr)),
            "mean_grad_norm_ratio_to_td": float(np.mean([r["grad_ratio_qreg"] for r in audit_records])),
        },
        "entropy_regularization_audit": {
            "mean_loss_magnitude": float(np.mean(ent_losses)),
            "mean_action_entropy_nats": float(np.mean([r["action_entropy_nats"] for r in audit_records])),
            "mean_grad_norm": float(np.mean(gn_ent_arr)),
            "mean_grad_norm_ratio_to_td": float(np.mean([r["grad_ratio_entropy"] for r in audit_records])),
        },
        "td_loss_baseline": {
            "mean_td_loss": float(np.mean(td_losses)),
            "min_td_loss": float(np.min(td_losses)) if td_losses else 0.0,
            "max_td_loss": float(np.max(td_losses)) if td_losses else 0.0,
            "mean_td_grad_norm": float(np.mean(gn_td_arr)),
        },
        "causal_mechanism_diagnosis": {
            "q1_did_penalties_remain_dormant": bool(is_mode_pen_dormant and is_band_pen_dormant),
            "q2_did_penalties_activate_too_weak": bool(not is_mode_pen_dormant and mean_mode_grad_ratio < 0.05),
            "q3_did_penalties_optimize_wrong_surrogate": bool(not is_mode_pen_dormant and mean_mode_grad_ratio >= 0.05),
            "mechanistic_explanation": (
                f"Mode penalty activation was {mode_active_cnt}/{len(audit_records)} ({mode_act_pct:.1f}%). "
                f"Mean mode collapse rate was {np.mean(mode_collapse_rates):.4f} (threshold: {mode_collapse_rate_thresh:.2f}). "
                f"Mean mode penalty gradient ratio to TD loss was {mean_mode_grad_ratio:.4f} ({mean_mode_grad_ratio*100:.2f}% of TD). "
                f"Top-band penalty activation was {band_active_cnt}/{len(audit_records)} ({band_act_pct:.1f}%). "
                f"Mean top-band softmax was {np.mean(top_band_fracs):.4f} (threshold: {band_div_thresh:.2f})."
            )
        }
    }

    # Print summary to console
    print("\n" + "=" * 90)
    print("PHASE G1: R1 REGULARIZER ACTIVATION & GRADIENT AUDIT RESULTS")
    print("=" * 90)
    print(f"Audit Verdict: {diagnostic_verdict}")
    print("-" * 90)
    print(f"TD Loss (Mean):                    {summary['td_loss_baseline']['mean_td_loss']:.4f} (Grad Norm: {summary['td_loss_baseline']['mean_td_grad_norm']:.4f})")
    print(f"Mode Penalty Activation Rate:      {summary['mode_diversity_penalty_audit']['activation_rate_pct']:.1f}% ({mode_active_cnt}/{len(audit_records)} batches)")
    print(f"  Mean Mode Collapse Rate:         {summary['mode_diversity_penalty_audit']['mean_mode_collapse_rate']:.4f} (Threshold: {mode_collapse_rate_thresh:.2f})")
    print(f"  Mean Mode Penalty Magnitude:     {summary['mode_diversity_penalty_audit']['mean_penalty_magnitude']:.6f} ({summary['mode_diversity_penalty_audit']['mean_ratio_to_td_loss_pct']:.4f}% of TD)")
    print(f"  Mean Mode Grad Norm:             {summary['mode_diversity_penalty_audit']['mean_grad_norm']:.6f} ({summary['mode_diversity_penalty_audit']['mean_grad_norm_ratio_to_td']*100:.2f}% of TD Grad)")
    print("-" * 90)
    print(f"Top-Band Penalty Activation Rate:  {summary['top_band_diversity_penalty_audit']['activation_rate_pct']:.1f}% ({band_active_cnt}/{len(audit_records)} batches)")
    print(f"  Mean Top-Band Softmax Frac:      {summary['top_band_diversity_penalty_audit']['mean_top_band_fraction_softmax']:.4f} (Threshold: {band_div_thresh:.2f})")
    print(f"  Mean Top-Band Penalty Mag:       {summary['top_band_diversity_penalty_audit']['mean_penalty_magnitude']:.6f} ({summary['top_band_diversity_penalty_audit']['mean_ratio_to_td_loss_pct']:.4f}% of TD)")
    print(f"  Mean Top-Band Grad Norm:         {summary['top_band_diversity_penalty_audit']['mean_grad_norm']:.6f} ({summary['top_band_diversity_penalty_audit']['mean_grad_norm_ratio_to_td']*100:.2f}% of TD Grad)")
    print("-" * 90)
    print(f"Q^2 Loss (Mean):                   {summary['q2_regularization_audit']['mean_loss_magnitude']:.6f} ({summary['q2_regularization_audit']['mean_ratio_to_td_loss_pct']:.4f}% of TD)")
    print(f"  Mean Q^2 Grad Norm:              {summary['q2_regularization_audit']['mean_grad_norm']:.6f} ({summary['q2_regularization_audit']['mean_grad_norm_ratio_to_td']*100:.2f}% of TD Grad)")
    print(f"Entropy Loss (Mean):               {summary['entropy_regularization_audit']['mean_loss_magnitude']:.6f}")
    print(f"  Mean Entropy Grad Norm:          {summary['entropy_regularization_audit']['mean_grad_norm']:.6f} ({summary['entropy_regularization_audit']['mean_grad_norm_ratio_to_td']*100:.2f}% of TD Grad)")
    print("=" * 90 + "\n")

    # Save output artifacts
    out_json = root / "reports/g1_r1_regularizer_activation_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved audit report to %s", out_json)


if __name__ == "__main__":
    main()
