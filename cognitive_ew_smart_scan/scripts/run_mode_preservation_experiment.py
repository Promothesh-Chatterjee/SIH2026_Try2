"""Diagnostic Runner for Controlled Mode-Preservation Experiment (Trial D Preparation).

Compares:
- Baseline (No-Training Diagnostic Probe)
- Arm 1: Frozen Advantage Head (requires_grad=False, excluded from optimizer)
- Arm 2: Regularized Advantage Head (scaled squared anchor loss L_anchor = lambda * ||W - W_base||^2)
- Arm 3: Unrestricted Head with Stratified Sequence Replay (anchored on Modes 0-4)

Each arm runs a controlled 500-step continuation from the frozen Gate-25k baseline.
Logs:
- Per-mode Bellman loss and isolated gradient norms
- Anchor loss magnitude relative to DRQN and Aux losses
- Weight drift on band_advantage_head.2.weight
- Mode distributions and scenario evaluations on agile, sparse, and dense scenarios.
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records
from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer
from cognitive_ew_smart_scan.src.training.stratified_mode_sampler import StratifiedModeSampler
from cognitive_ew_smart_scan.src.training.diagnostics.mode_telemetry import compute_mode_isolated_gradients
from cognitive_ew_smart_scan.scripts.compare_identical_state_traces import collect_identical_state_trace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mode_preservation_experiment")


def evaluate_arm_scenarios(model: DRQNScheduler, device: torch.device) -> Dict[str, Any]:
    scenarios = [
        ("config_29", "agile"),
        ("config_241", "agile"),
        ("config_119", "sparse"),
        ("config_143", "sparse"),
        ("config_195", "dense"),
    ]
    res: Dict[str, Any] = {}
    model.eval()

    for scen_id, cat in scenarios:
        h5_path = Path(f"D:/TSRD/stare/val_stare/{scen_id}.h5")
        if not h5_path.exists():
            continue
        recs = load_h5_records(h5_path)
        env = CognitiveRFScanEnv(
            config={
                "n_bands": 36,
                "dwell_modes": 5,
                "features_per_band": 10,
                "ema_alpha": 0.30,
                "ema_alpha_miss_confirmed": 0.20,
                "semantic_memory_enabled": False,
            },
            records=recs,
            seed=42,
        )
        obs, _ = env.reset(seed=42)
        hx = None
        hits = 0
        modes = []
        n_steps = 1000

        for _ in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            with torch.no_grad():
                q, _, hx = model(obs_t, hx)
                act = int(torch.argmax(q.squeeze()).item())
            modes.append(act % 5)
            obs, _, term, trunc, info = env.step(act)
            hits += int(info.get("hit", False))
            if term or trunc:
                break

        mode_counts = {str(m): int(modes.count(m)) for m in range(5)}
        res[scen_id] = {
            "category": cat,
            "intercept_rate": hits / float(n_steps),
            "mode_distribution": mode_counts,
            "mode2_fraction": mode_counts.get("2", 0) / float(n_steps),
        }

    agile_irs = [res[s]["intercept_rate"] for s in res if res[s]["category"] == "agile"]
    sparse_irs = [res[s]["intercept_rate"] for s in res if res[s]["category"] == "sparse"]
    dense_irs = [res[s]["intercept_rate"] for s in res if res[s]["category"] == "dense"]

    return {
        "scenario_results": res,
        "mean_ir": float(np.mean([r["intercept_rate"] for r in res.values()])),
        "agile_ir": float(np.mean(agile_irs)) if agile_irs else 0.0,
        "sparse_ir": float(np.mean(sparse_irs)) if sparse_irs else 0.0,
        "dense_ir": float(np.mean(dense_irs)) if dense_irs else 0.0,
    }


def run_probe_arm(
    arm_name: str,
    base_state_dict: dict,
    reservoir_episodes: list,
    steps: int = 500,
    lambda_anchor: float = 0.01,
    use_stratified_replay: bool = False,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("STARTING ARM: %s (Steps: %d, Stratified: %s, Lambda: %s)", arm_name, steps, use_stratified_replay, lambda_anchor)
    logger.info("=" * 80)

    device = torch.device(device_str)
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialise model from baseline
    model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2).to(device)
    model.load_state_dict(copy.deepcopy(base_state_dict))
    target_model = copy.deepcopy(model).to(device)

    # Reference weights for anchor
    base_head_weights = {
        k: v.clone().detach().to(device)
        for k, v in model.band_advantage_head.named_parameters()
    }

    # Configure frozen head if arm is frozen
    if arm_name == "Arm 1 (Frozen Head)":
        for p in model.band_advantage_head.parameters():
            p.requires_grad = False
        optim_params = [p for p in model.parameters() if p.requires_grad]
    else:
        optim_params = list(model.parameters())

    optimizer = optim.Adam(optim_params, lr=5.0e-5)
    loss_fn = nn.SmoothL1Loss()

    # Replay buffer loaded with reservoir
    buffer = SequenceReplayBuffer(capacity=20000, seq_len=16, obs_dim=360, burn_in=8, seed=42)
    for ep in reservoir_episodes:
        buffer._episodes.append(ep)
        buffer._total += int(ep["length"])

    strat_sampler = StratifiedModeSampler(buffer, seq_len=16, burn_in=8, seed=42)

    # Telemetry accumulators
    drqn_losses = []
    anchor_losses = []
    mode_telemetry_history: List[Dict[str, Any]] = []

    model.train()
    for step in range(steps):
        # Sample batch
        if use_stratified_replay:
            batch, sample_meta = strat_sampler.sample_mode_stratified(32)
        else:
            batch = buffer.sample(32, target_hit_seq_fraction=0.40)
            sample_meta = {}

        valid = torch.tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        burn_in = torch.tensor(batch["burn_in_mask"], dtype=torch.bool, device=device)
        loss_mask = valid & ~burn_in

        obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
        act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
        rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
        next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
        done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

        q_all, aux, _ = model(obs_b)
        q_chosen = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

        with torch.inference_mode():
            next_q_online, _, _ = model(next_obs_b)
            best_actions = next_q_online.argmax(dim=-1, keepdim=True)
            next_q_target, _, _ = target_model(next_obs_b)
            next_q = next_q_target.gather(-1, best_actions).squeeze(-1)

        # Baseline reward centering
        batch_mean = float(rew_b[loss_mask].mean().item())
        centered_rew_b = rew_b - batch_mean
        targets = centered_rew_b + 0.99 * next_q * (1.0 - done_b)

        # Mode-specific gradient telemetry
        if step % 50 == 0 or step == steps - 1:
            m_telem = compute_mode_isolated_gradients(
                online_drqn=model,
                loss_fn=loss_fn,
                q_chosen=q_chosen,
                targets=targets,
                act_b=act_b,
                loss_mask=loss_mask,
            )
            mode_telemetry_history.append({"step": step, "modes": m_telem})

        q_loss = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())
        q_reg_loss = 5.0e-4 * (q_all[loss_mask] ** 2).mean()

        total_loss = q_loss + q_reg_loss

        # Arm 2 anchor loss
        anchor_loss = torch.zeros((), device=device)
        if arm_name == "Arm 2 (Regularized Head)" and lambda_anchor > 0:
            for k, p in model.band_advantage_head.named_parameters():
                anchor_loss = anchor_loss + torch.sum((p - base_head_weights[k]) ** 2)
            total_loss = total_loss + lambda_anchor * anchor_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(optim_params, 5.0)
        optimizer.step()

        drqn_losses.append(float(q_loss.item()))
        anchor_losses.append(float(anchor_loss.item()) if isinstance(anchor_loss, torch.Tensor) else 0.0)

        # Sync target every 250 steps
        if (step + 1) % 250 == 0:
            target_model.load_state_dict(model.state_dict())

    # Measure weight drift on band_advantage_head.2.weight
    curr_w2 = model.band_advantage_head[2].weight.data.clone().cpu()
    base_w2 = base_state_dict["band_advantage_head.2.weight"].clone().cpu()
    weight_drift = float((curr_w2 - base_w2).abs().max().item())
    mode2_weight_val = float(curr_w2[2, 0].item())  # Mode 2 weight

    # Evaluation
    eval_metrics = evaluate_arm_scenarios(model, device)

    return {
        "arm_name": arm_name,
        "mean_drqn_loss": float(np.mean(drqn_losses)),
        "mean_anchor_loss": float(np.mean(anchor_losses)),
        "weight_drift_max": weight_drift,
        "mode2_weight_val": mode2_weight_val,
        "eval_metrics": eval_metrics,
        "mode_telemetry_sample": mode_telemetry_history[-1] if mode_telemetry_history else {},
    }


def main():
    device_str = "cpu"
    base_ckpt_path = Path("cognitive_ew_smart_scan/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
    base_ckpt = torch.load(base_ckpt_path, map_location=device_str, weights_only=False)
    base_state_dict = base_ckpt["state_dict"]

    res_path = Path("cognitive_ew_smart_scan/checkpoints/production_baseline/baseline_reservoir_5k.pkl")
    res_data = pickle.load(open(res_path, "rb"))
    reservoir_episodes = res_data["episodes"]

    report: Dict[str, Any] = {}

    # 1. No-Training Baseline Diagnostic Probe
    logger.info("=" * 80)
    logger.info("RUNNING NO-TRAINING BASELINE DIAGNOSTIC PROBE")
    logger.info("=" * 80)
    base_model = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    base_model.load_state_dict(base_state_dict)
    base_eval = evaluate_arm_scenarios(base_model, torch.device(device_str))
    base_trace_29 = collect_identical_state_trace(base_model, "D:/TSRD/stare/val_stare/config_29.h5", n_steps=50)
    base_trace_143 = collect_identical_state_trace(base_model, "D:/TSRD/stare/val_stare/config_143.h5", n_steps=50)

    report["baseline_no_training"] = {
        "eval_metrics": base_eval,
        "mode2_weight_val": float(base_state_dict["band_advantage_head.2.weight"][2, 0].item()),
    }

    # 2. Arm 1: Frozen Head
    arm1 = run_probe_arm(
        arm_name="Arm 1 (Frozen Head)",
        base_state_dict=base_state_dict,
        reservoir_episodes=reservoir_episodes,
        steps=500,
        lambda_anchor=0.0,
        use_stratified_replay=False,
    )
    report["arm_1_frozen_head"] = arm1

    # 3. Arm 2: Regularized Head (scaled lambda=0.01)
    arm2 = run_probe_arm(
        arm_name="Arm 2 (Regularized Head)",
        base_state_dict=base_state_dict,
        reservoir_episodes=reservoir_episodes,
        steps=500,
        lambda_anchor=0.01,
        use_stratified_replay=False,
    )
    report["arm_2_regularized_head"] = arm2

    # 4. Arm 3: Unrestricted Head + Stratified Replay
    arm3 = run_probe_arm(
        arm_name="Arm 3 (Stratified Replay)",
        base_state_dict=base_state_dict,
        reservoir_episodes=reservoir_episodes,
        steps=500,
        lambda_anchor=0.0,
        use_stratified_replay=True,
    )
    report["arm_3_stratified_replay"] = arm3

    # Save final report
    out_path = Path("cognitive_ew_smart_scan/reports/mode_preservation_experiment_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Mode Preservation Experiment Report saved to: %s", out_path)

    # Save sample identical state traces
    traces_path = Path("cognitive_ew_smart_scan/reports/identical_state_traces_sample.json")
    with open(traces_path, "w") as f:
        json.dump({
            "config_29_baseline_50steps": base_trace_29,
            "config_143_baseline_50steps": base_trace_143,
        }, f, indent=2)
    logger.info("Identical State Traces saved to: %s", traces_path)


if __name__ == "__main__":
    main()
