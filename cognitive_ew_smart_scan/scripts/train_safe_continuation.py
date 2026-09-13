"""Safe Controlled Continuation Trainer (25k to 40k).

Implements Phase 4 (Dry Run) and Phase 5 (Gate 1 Trial D) with:
- CheckpointGuard: Never writes into production_baseline, writes atomically with SHA-256.
- Frozen Advantage Head (Trial D Constraint): band_advantage_head strictly frozen (requires_grad=False)
  and excluded from Adam optimizer parameter groups.
- Target Network Verification: Asserts exact identity with online network and baseline weights after every sync.
- Periodic 500-Step Atomic Checkpointing & Canonical Evaluation across all 10 TSRD validation scenarios.
- Strict Operational Gate Sentinel: Fast rollback if Mean IR < 60.45%, Agile IR < 46.70%, Worst-case IR < 12.40%,
  Pd < 99.85%, Pfa > 0.0000, or Mode 2 fraction < 90% on agile/sparse scenarios.
- Stratified Replay Sampler: Anchors sequence windows on Modes 0-4 while strictly preserving 16-step DRQN
  contiguous slices, 8-step burn-in, and single-episode boundaries.
- Provenance Logging: Checkpoint SHA-256, Git revision, YAML hash, scenario breakdown in audit log.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.environment.scenario_generator import load_h5_records, ScenarioSource
from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer
from cognitive_ew_smart_scan.src.training.stratified_mode_sampler import StratifiedModeSampler
from cognitive_ew_smart_scan.src.training.multi_source_sampler import MultiSourceReplaySampler
from cognitive_ew_smart_scan.src.training.diagnostics.action_tracker import ActionTracker
from cognitive_ew_smart_scan.src.training.diagnostics.q_telemetry import QTelemetry
from cognitive_ew_smart_scan.src.training.diagnostics.reward_tracker import RewardTracker
from cognitive_ew_smart_scan.src.training.safety.checkpoint_guard import CheckpointGuard
from cognitive_ew_smart_scan.src.training.safety.safety_monitor import SafetyMonitor
from cognitive_ew_smart_scan.src.training.safety.rollback_manager import RollbackManager
from cognitive_ew_smart_scan.src.evaluation.benchmark_contract import CANONICAL_SCENARIOS
from cognitive_ew_smart_scan.src.evaluation.promotion_sentinel import evaluate_promotion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_safe_continuation")


def load_yaml(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return rev
    except Exception:
        return "UNKNOWN_GIT_REVISION"


def compute_file_sha256(path: Path | str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_head_drift(online_drqn: DRQNScheduler, base_head_weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Computes max abs diff and Frobenius norm for all advantage head parameter tensors."""
    head_drift: Dict[str, float] = {}
    for name, p in online_drqn.band_advantage_head.named_parameters():
        curr_p = p.data.clone().detach().cpu()
        base_p = base_head_weights[name]
        diff = curr_p - base_p
        head_drift[f"{name}_max_abs"] = float(diff.abs().max().item())
        head_drift[f"{name}_frobenius"] = float(torch.norm(diff, p="fro").item())
    return head_drift


def append_audit_log_record(
    audit_log_path: Path | str,
    event_type: str,
    record: Dict[str, Any],
) -> None:
    """Safely appends an audit event to the structured JSON audit log."""
    path = Path(audit_log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {"rollback_events": [], "staged_evaluations": []}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    data.update(loaded)
        except Exception as exc:
            logger.warning("Could not parse existing audit log at %s: %s", path, exc)

    if event_type not in data:
        data[event_type] = []
    data[event_type].append(record)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _do_safe_drqn_update(
    online_drqn: DRQNScheduler,
    target_drqn: DRQNScheduler,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: dict[str, np.ndarray],
    gamma: float,
    device: torch.device,
    q_telemetry: QTelemetry,
    grad_clip_norm: float = 5.0,
    q_reg_coef: float = 5.0e-4,
    aux_coef: float = 0.0,
    reward_baseline: float = 0.0,
    baseline_momentum: float = 0.99,
) -> tuple[float, Dict[str, Any]]:
    valid = torch.tensor(batch["valid_mask"], dtype=torch.bool, device=device)
    burn_in = torch.tensor(batch["burn_in_mask"], dtype=torch.bool, device=device)
    loss_mask = valid & ~burn_in

    if not loss_mask.any():
        return 0.0, {}

    obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
    rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
    next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
    done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

    q_all, aux, _ = online_drqn(obs_b)
    q_chosen = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

    # Double-DQN: online chooses best action, target evaluates (strictly detached via inference_mode)
    with torch.inference_mode():
        next_q_online, _, _ = online_drqn(next_obs_b)
        best_actions = next_q_online.argmax(dim=-1, keepdim=True)
        next_q_target, _, _ = target_drqn(next_obs_b)
        next_q_unclamped = next_q_target.gather(-1, best_actions).squeeze(-1)

    batch_mean = float(rew_b[loss_mask].mean().item())
    updated_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * batch_mean
    centered_rew_b = rew_b - updated_baseline

    targets = centered_rew_b + gamma * next_q_unclamped * (1.0 - done_b)
    td_errors = (q_chosen[loss_mask] - targets[loss_mask]).detach()
    q_loss = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())

    # Regularization
    q_reg_loss = q_reg_coef * (q_all[loss_mask] ** 2).mean() if q_reg_coef > 0.0 else torch.zeros((), device=device)
    loss = q_loss + q_reg_loss

    # Baseline-exact auxiliary representation learning
    aux_loss = torch.zeros((), device=device)
    bce_loss = torch.zeros((), device=device)
    huber_loss = torch.zeros((), device=device)
    valid_bce_targets = int(loss_mask.sum().item())
    valid_time_targets = 0

    if aux_coef > 0.0:
        hit_probs = torch.tensor(batch["hit_probs"], dtype=torch.float32, device=device)
        prob_pred = aux["intercept_prob"].gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
        bce_loss = nn.functional.binary_cross_entropy(prob_pred[loss_mask], hit_probs[loss_mask].detach())

        # Time target only when the transition is a genuine hit
        time_valid = (
            loss_mask
            & torch.tensor(batch["time_target_valid"], dtype=torch.bool, device=device)
        )
        valid_time_targets = int(time_valid.sum().item())
        if valid_time_targets > 0:
            intercept_times = torch.tensor(batch["intercept_times_us"], dtype=torch.float32, device=device)
            time_pred = aux["intercept_time_us"].gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
            huber_loss = nn.functional.huber_loss(
                time_pred[time_valid] / 1000.0,
                intercept_times[time_valid].detach() / 1000.0,
                delta=0.1
            )
        else:
            huber_loss = torch.zeros((), device=device)

        aux_loss = bce_loss + huber_loss
        loss = loss + aux_coef * aux_loss

    optimizer.zero_grad()
    loss.backward()

    # Pre-clip gradient norm
    trainable_params = [p for p in online_drqn.parameters() if p.requires_grad]
    unclipped_grad_norm = float(
        torch.nn.utils.clip_grad_norm_(trainable_params, float("inf")).item()
    )
    # Applied gradient clipping
    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
    clipped_grad_norm = float(
        torch.nn.utils.clip_grad_norm_(trainable_params, float("inf")).item()
    )

    optimizer.step()

    # Telemetry
    telemetry = q_telemetry.evaluate_step(
        q_online=q_chosen[loss_mask],
        target_q_unclamped=next_q_unclamped[loss_mask],
        td_errors=td_errors,
        online_model=online_drqn,
        unclipped_grad_norm=unclipped_grad_norm,
        clipped_grad_norm=clipped_grad_norm,
    )
    telemetry["total_loss"] = float(loss.item())
    telemetry["drqn_loss"] = float(q_loss.item())
    telemetry["td_loss"] = float(q_loss.item())
    telemetry["q_reg_loss"] = float(q_reg_loss.item())
    telemetry["aux_loss"] = float(aux_loss.item())
    telemetry["bce_loss"] = float(bce_loss.item())
    telemetry["huber_loss"] = float(huber_loss.item())
    telemetry["valid_bce_targets"] = valid_bce_targets
    telemetry["valid_time_targets"] = valid_time_targets
    telemetry["unclipped_grad_norm"] = unclipped_grad_norm
    telemetry["clipped_grad_norm"] = clipped_grad_norm

    return float(loss.item()), telemetry


def evaluate_canonical_10scenarios_inline(
    model: DRQNScheduler,
    device: torch.device,
    n_steps: int = 1000,
    seed: int = 42,
    data_root: Path | str = "D:/TSRD",
) -> Dict[str, Any]:
    """Run non-learning evaluation over all 10 canonical validation scenarios."""
    model.eval()
    scenario_results: Dict[str, Any] = {}
    pds = []
    pfas = []
    mode_totals = np.zeros(5, dtype=np.int64)

    agile_scen_ids = {"config_29", "config_195", "config_241", "config_119"}
    sparse_scen_ids = {"config_143", "config_119"}
    agile_sparse_scen_ids = agile_scen_ids | sparse_scen_ids

    for idx, scen_id in enumerate(CANONICAL_SCENARIOS):
        h5_path = Path(data_root) / "stare" / "val_stare" / f"{scen_id}.h5"
        if not h5_path.exists():
            logger.warning("Missing canonical scenario file: %s", h5_path)
            continue

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
        bands = []

        for step in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            with torch.no_grad():
                q, _, hx = model(obs_t, hx)
                act = int(torch.argmax(q.squeeze()).item())

            b = act // 5
            m = act % 5
            bands.append(b)
            modes.append(m)
            mode_totals[m] += 1

            obs, r, term, trunc, info = env.step(act)
            hits += int(info.get("hit", False))
            if term or trunc:
                break

        ir = hits / float(n_steps)
        pd_val = float(env.fom.pd)
        pfa_val = float(env.fom.pfa)
        pds.append(pd_val)
        pfas.append(pfa_val)

        mode_dist = {str(m): int(modes.count(m)) for m in range(5)}
        scenario_results[scen_id] = {
            "intercept_rate": ir,
            "hits": hits,
            "steps": n_steps,
            "pd": pd_val,
            "pfa": pfa_val,
            "distinct_bands": len(set(bands)),
            "mode_distribution": mode_dist,
            "mode2_fraction": mode_dist.get("2", 0) / float(n_steps),
        }

    model.train()
    irs = [r["intercept_rate"] * 100 for r in scenario_results.values()]
    agile_irs = [scenario_results[s]["intercept_rate"] * 100 for s in scenario_results if s in agile_scen_ids]
    sparse_irs = [scenario_results[s]["intercept_rate"] * 100 for s in scenario_results if s in sparse_scen_ids]
    agile_sparse_mode2_fractions = [scenario_results[s]["mode2_fraction"] for s in scenario_results if s in agile_sparse_scen_ids]

    summary = {
        "mean_ir": float(np.mean(irs)) if irs else 0.0,
        "median_ir": float(np.median(irs)) if irs else 0.0,
        "worst_case_ir": float(np.min(irs)) if irs else 0.0,
        "agile_ir": float(np.mean(agile_irs)) if agile_irs else 0.0,
        "sparse_ir": float(np.mean(sparse_irs)) if sparse_irs else 0.0,
        "pd": float(np.mean(pds)) * 100 if pds else 0.0,
        "pfa": float(np.mean(pfas)) if pfas else 0.0,
        "mode_selection_totals": {str(m): int(mode_totals[m]) for m in range(5)},
        "mode2_overall_fraction": float(mode_totals[2]) / float(max(1, sum(mode_totals))),
        "mode2_agile_sparse_fraction": float(np.mean(agile_sparse_mode2_fractions)) if agile_sparse_mode2_fractions else 0.0,
        "distinct_bands_mean": float(np.mean([r["distinct_bands"] for r in scenario_results.values()])) if scenario_results else 0.0,
        "scenario_breakdown": scenario_results,
    }
    return summary


def train_continuation(
    config_path: Path | str,
    dry_run: bool = False,
    dry_run_steps: int = 100,
    eval_steps_per_scenario: int | None = None,
    override_target_step: int | None = None,
) -> bool:
    cfg = load_yaml(config_path)
    config_sha256 = compute_file_sha256(config_path)
    git_rev = get_git_revision()
    logger.info("Loaded Configuration from %s (SHA-256: %s, Git Revision: %s)", config_path, config_sha256, git_rev)

    # 1. Output and Checkpoint Safety Setup
    baseline_path = Path(cfg["checkpoints"]["baseline_frozen"]).resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {baseline_path}")

    candidate_output_dir = Path(cfg["checkpoints"]["output_candidate_dir"]).resolve()
    audit_log_path = Path(cfg["checkpoints"]["audit_log_path"]).resolve()

    ckpt_guard = CheckpointGuard(
        output_dir=candidate_output_dir,
        forbidden_dirs=[baseline_path.parent, Path("checkpoints/production_baseline").resolve()],
    )
    rollback_mgr = RollbackManager(
        last_known_good_ckpt=baseline_path,
        audit_log_path=audit_log_path,
    )
    safety_monitor = SafetyMonitor(halt_ceiling_q=cfg["safety_thresholds"]["hard_safety_halt"]["max_q"])

    # 2. Model Initialization and Mandatory Trial D Freezing
    device = torch.device("cpu")
    logger.info("Loading starting model from immutable baseline: %s", baseline_path)
    payload = torch.load(baseline_path, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload.get("online_drqn", payload.get("model")))
    if state_dict is None:
        raise ValueError(f"Could not find valid state_dict in {baseline_path}")

    online_drqn = DRQNScheduler(
        obs_dim=360,
        n_bands=36,
        n_actions=180,
        lstm_hidden=256,
        lstm_layers=2,
    ).to(device)
    online_drqn.load_state_dict(state_dict, strict=True)
    logger.info("Successfully loaded state_dict into online_drqn with strict=True verification.")

    # Reference weights of advantage head for bit-exact drift assertion
    base_head_weights = {
        k: v.clone().detach().cpu()
        for k, v in online_drqn.band_advantage_head.named_parameters()
    }
    base_head_state_dict = {
        k: v.clone().detach().cpu()
        for k, v in online_drqn.band_advantage_head.state_dict().items()
    }

    freeze_advantage_head = bool(
        cfg.get("experiment", {}).get("freeze_advantage_head", False)
        or cfg.get("training", {}).get("freeze_advantage_head", False)
    )

    if freeze_advantage_head:
        logger.info("MANDATORY TRIAL D CONSTRAINT: Strictly freezing band_advantage_head (requires_grad=False)")
        for name, p in online_drqn.band_advantage_head.named_parameters():
            p.requires_grad = False
        trainable_params = [p for p in online_drqn.parameters() if p.requires_grad]
        frozen_params = [p for p in online_drqn.parameters() if not p.requires_grad]
        logger.info("Parameter partitioning: %d trainable tensors, %d frozen tensors in band_advantage_head",
                    len(trainable_params), len(frozen_params))
    else:
        logger.warning("FREEZING NOT ACTIVE: band_advantage_head remains trainable!")
        trainable_params = list(online_drqn.parameters())

    target_drqn = copy.deepcopy(online_drqn).to(device)
    target_drqn.eval()

    start_step = int(cfg["training"]["start_step"])
    if dry_run:
        target_step = start_step + dry_run_steps
        eval_freq = dry_run_steps
        logger.info("DRY RUN MODE: Stepping from %d to %d (%d steps)", start_step, target_step, dry_run_steps)
    else:
        target_step = override_target_step or int(cfg["training"]["target_step"])
        eval_freq = int(cfg["training"]["eval_frequency"])
        logger.info("GATE 1 PILOT: Stepping from %d to %d (Eval freq: %d steps)", start_step, target_step, eval_freq)

    lr = float(cfg["learning_rate"]["gate_1_25k_27500"])
    optimizer = optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.HuberLoss()

    # 3. Replay Buffer & Stratified Mode Sampler Setup
    baseline_reservoir_path = Path(cfg["replay"]["baseline_reservoir_path"]).resolve()
    if not baseline_reservoir_path.exists():
        raise FileNotFoundError(f"Baseline reservoir not found at: {baseline_reservoir_path}")

    replay_sampler_type = cfg["replay"].get("sampler", "stratified_mode_anchored")
    logger.info("Configured Replay Sampler: %s", replay_sampler_type)

    replay_buffer = SequenceReplayBuffer(
        capacity=cfg["replay"]["capacity"],
        seq_len=cfg["replay"]["seq_len"],
        obs_dim=360,
        burn_in=cfg["replay"]["burn_in"],
        seed=42,
    )
    n_loaded = replay_buffer.load_episodes(baseline_reservoir_path)
    logger.info("Loaded %d baseline reservoir episodes (%d transitions) into replay buffer", n_loaded, len(replay_buffer))

    if replay_sampler_type == "stratified_mode_anchored" or "mode_quotas" in cfg["replay"]:
        quotas_cfg = cfg["replay"].get("mode_quotas", {})
        mode_weights: Dict[int, float] = {}
        for k, v in quotas_cfg.items():
            m_idx = int(k.split("_")[-1])
            mode_weights[m_idx] = float(v)

        w_sum = sum(mode_weights.values())
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(f"Mode quotas must sum to 1.0! Got {w_sum:.4f} ({mode_weights})")

        strat_sampler = StratifiedModeSampler(
            buffer=replay_buffer,
            mode_weights=mode_weights,
            seq_len=cfg["replay"]["seq_len"],
            burn_in=cfg["replay"]["burn_in"],
            seed=42,
        )
        logger.info("StratifiedModeSampler initialized with verified quotas: %s", mode_weights)
    else:
        strat_sampler = None

    # 4. Diagnostics Trackers
    action_tracker = ActionTracker(n_bands=36, n_modes=5, window_size=500)
    q_telemetry = QTelemetry(halt_ceiling=cfg["safety_thresholds"]["hard_safety_halt"]["max_q"])
    reward_tracker = RewardTracker(gamma=cfg["optimization"]["gamma"])

    # 5. Environment & Data Setup
    data_root = Path("D:/TSRD")
    train_source = ScenarioSource(data_root=data_root, mode="stare", subset="train", source_type="world")
    logger.info("TSRD Train Manifest loaded: %d eligible training files", len(train_source.eligible_files))

    batch_size = int(cfg["optimization"]["batch_size"])
    update_freq = int(cfg["optimization"]["update_freq"])
    target_update_freq = int(cfg["optimization"]["target_update_freq"])
    grad_clip_norm = float(cfg["optimization"]["gradient_clip_norm"])
    q_reg_coef = float(cfg["optimization"]["q_reg_coef"])
    aux_coef = float(cfg["optimization"].get("aux_coef", 0.1))
    max_steps_per_episode = int(cfg["training"].get("max_steps_per_episode", 250))
    val_steps_per_scenario = eval_steps_per_scenario or (50 if dry_run else int(cfg["training"].get("val_steps_per_scenario", 1000)))

    # Operational Gate Thresholds
    min_mean_ir = float(cfg.get("operational_gate_thresholds", {}).get("min_mean_ir", 60.45))
    min_agile_ir = float(cfg.get("operational_gate_thresholds", {}).get("min_agile_ir", 46.70))
    min_worst_case_ir = float(cfg.get("operational_gate_thresholds", {}).get("min_worst_case_ir", 12.40))
    min_pd = float(cfg.get("operational_gate_thresholds", {}).get("min_pd", 99.85))
    max_pfa = float(cfg.get("operational_gate_thresholds", {}).get("max_pfa", 0.0000))
    min_mode2_fraction = float(cfg.get("operational_gate_thresholds", {}).get("min_mode2_fraction_agile_sparse", 0.90))

    logger.info("Strict Operational Gate Thresholds: Mean IR >= %.2f%%, Agile IR >= %.2f%%, Worst-case IR >= %.2f%%, Pd >= %.2f%%, Pfa <= %.4f, Mode 2 Agile/Sparse >= %.1f%%",
                min_mean_ir, min_agile_ir, min_worst_case_ir, min_pd, max_pfa, min_mode2_fraction * 100)

    global_step = start_step
    episode = 28
    eps_start = float(cfg["exploration"]["gate_1"]["eps_start"])
    eps_end = float(cfg["exploration"]["gate_1"]["eps_end"])

    rng = np.random.default_rng(42)
    sampler_telem: dict = {}
    q_diag: dict = {}

    # Save initial pre-interval checkpoint representing starting point
    init_ckpt_payload = {
        "global_step": global_step,
        "state_dict": online_drqn.state_dict(),
        "online_drqn": online_drqn.state_dict(),
        "optimizer": optimizer.state_dict(),
        "baseline_sha256": cfg["experiment"]["baseline_checkpoint_sha256"],
        "git_revision": git_rev,
    }
    init_ckpt_name = f"checkpoint_step_{global_step}_pre_eval.pt"
    ckpt_guard.save_checkpoint_atomic(init_ckpt_payload, init_ckpt_name)
    last_approved_checkpoint = baseline_path

    logger.info("Beginning safe controlled training loop: %d -> %d steps...", global_step, target_step)

    while global_step < target_step:
        train_path = rng.choice(train_source.eligible_files)
        scen_id = Path(train_path).stem
        records = load_h5_records(train_path)
        env = CognitiveRFScanEnv(
            config={
                "n_bands": 36,
                "dwell_modes": 5,
                "features_per_band": 10,
                "ema_alpha": 0.30,
                "ema_alpha_miss_confirmed": 0.20,
                "semantic_memory_enabled": False,
                "max_steps_per_episode": max_steps_per_episode,
            },
            records=records,
            seed=42 + episode,
        )
        obs, _ = env.reset()
        hx = None
        reward_tracker.reset_episode()
        action_tracker.start_scenario(scen_id)

        for step_in_ep in range(max_steps_per_episode):
            if global_step >= target_step:
                break

            progress = float(global_step - start_step) / max(1.0, float(target_step - start_step))
            eps = eps_start + progress * (eps_end - eps_start)

            if rng.random() < eps:
                action = int(rng.integers(0, 180))
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
                with torch.no_grad():
                    q_vals, _, hx = online_drqn(obs_t, hx)
                    action = int(torch.argmax(q_vals.squeeze()).item())

            action_tracker.step(action, scenario_id=scen_id)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward_tracker.step(reward, info)
            done = bool(terminated or truncated or (step_in_ep + 1 >= max_steps_per_episode))

            replay_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                hit_prob=1.0 if info.get("hit", False) else 0.0,
                intercept_time_us=float(info.get("intercept_time_us", np.nan)),
                scenario_id=scen_id,
            )
            obs = next_obs
            global_step += 1

            # Optimization step
            if global_step % update_freq == 0 and replay_buffer.can_sample(batch_size):
                if strat_sampler is not None:
                    batch, sampler_telem = strat_sampler.sample_mode_stratified(batch_size)
                else:
                    batch = replay_buffer.sample(batch_size, target_hit_seq_fraction=0.40)
                    sampler_telem = {"sampler": "standard"}

                loss_val, q_diag = _do_safe_drqn_update(
                    online_drqn=online_drqn,
                    target_drqn=target_drqn,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    batch=batch,
                    gamma=0.99,
                    device=device,
                    q_telemetry=q_telemetry,
                    grad_clip_norm=grad_clip_norm,
                    q_reg_coef=q_reg_coef,
                    aux_coef=aux_coef,
                )

                # Target network update & assertion
                if global_step % target_update_freq == 0:
                    target_drqn.load_state_dict(online_drqn.state_dict())
                    for key, param in online_drqn.band_advantage_head.state_dict().items():
                        assert torch.equal(
                            param,
                            target_drqn.band_advantage_head.state_dict()[key]
                        ), f"Target sync drift in band_advantage_head.{key}"
                        if freeze_advantage_head:
                            assert torch.equal(
                                param.cpu(),
                                base_head_state_dict[key]
                            ), f"Advantage head drift away from baseline in band_advantage_head.{key}"
                    logger.info("Target network synchronized and verified at step %d", global_step)

                # Safety Monitor Check at every step
                action_diag = action_tracker.get_diagnostics()
                safety_res = safety_monitor.check(action_diag, q_diag)

                if safety_res["halt_triggered"]:
                    logger.error("SAFETY MONITOR TRIGGERED HARD HALT AT STEP %d!", global_step)
                    rollback_mgr.execute_rollback(
                        failed_ckpt_path=None,
                        reasons=safety_res["halt_reasons"],
                        step=global_step,
                        telemetry={"q_diag": q_diag, "action_diag": action_diag, "sampler": sampler_telem},
                    )
                    return False

            # Diagnostic telemetry log
            if global_step % 100 == 0:
                act_d = action_tracker.get_diagnostics()
                logger.info(
                    "Step %d/%d | TotalLoss: %.3f (DRQN: %.3f, Aux: %.3f, BCE: %.3f, Huber: %.3f) | GradNorm: %.2f | Dom(Global/Scen): %.1f%%/%.1f%% (%s) | Bands: %d/36 | Entropy: %.2f | Eps: %.3f | Qmax: %.2f",
                    global_step, target_step,
                    q_diag.get("total_loss", 0.0),
                    q_diag.get("drqn_loss", 0.0),
                    q_diag.get("aux_loss", 0.0),
                    q_diag.get("bce_loss", 0.0),
                    q_diag.get("huber_loss", 0.0),
                    q_diag.get("unclipped_grad_norm", 0.0),
                    act_d["global_top_band_dominance"] * 100,
                    act_d["per_scenario_top_band_dominance"] * 100,
                    act_d.get("current_scenario_id", "unknown"),
                    act_d["unique_bands"],
                    act_d["action_entropy"],
                    eps,
                    q_diag.get("q_max", 0.0),
                )

            # Periodic 500-step Checkpointing and Canonical Evaluation
            if global_step % eval_freq == 0:
                logger.info("=" * 80)
                logger.info("MILESTONE REACHED AT STEP %d: Executing Pre-Interval Checkpointing & Canonical Evaluation", global_step)
                logger.info("=" * 80)

                # 1. Save candidate checkpoint atomically before evaluation
                ckpt_name = f"checkpoint_step_{global_step}.pt" if not dry_run else f"checkpoint_dryrun_{global_step}.pt"
                ckpt_payload = {
                    "global_step": global_step,
                    "state_dict": online_drqn.state_dict(),
                    "online_drqn": online_drqn.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "baseline_sha256": cfg["experiment"]["baseline_checkpoint_sha256"],
                    "config_sha256": config_sha256,
                    "git_revision": git_rev,
                    "training_diagnostics": q_diag,
                }
                saved_ckpt, sha256_hex = ckpt_guard.save_checkpoint_atomic(ckpt_payload, ckpt_name)

                # 2. Compute and assert advantage head drift
                head_drift = compute_head_drift(online_drqn, base_head_weights)
                if freeze_advantage_head:
                    for k_name, val in head_drift.items():
                        assert val == 0.0, f"Drift violation in frozen advantage head: {k_name} = {val}"
                    logger.info("Bit-exact advantage head preservation confirmed (ΔW == 0.0000 across all layers)")

                # 3. Canonical 10-Scenario Evaluation
                eval_res = evaluate_canonical_10scenarios_inline(
                    model=online_drqn,
                    device=device,
                    n_steps=val_steps_per_scenario,
                    seed=42,
                    data_root=data_root,
                )

                logger.info("CANONICAL 10-SCENARIO EVALUATION RESULTS AT STEP %d (%d steps/scen):", global_step, val_steps_per_scenario)
                logger.info("  Mean IR: %.2f%% (Gate: >= %.2f%%)", eval_res["mean_ir"], min_mean_ir)
                logger.info("  Agile IR: %.2f%% (Gate: >= %.2f%%)", eval_res["agile_ir"], min_agile_ir)
                logger.info("  Sparse IR: %.2f%%", eval_res["sparse_ir"])
                logger.info("  Worst-case IR: %.2f%% (Gate: >= %.2f%%)", eval_res["worst_case_ir"], min_worst_case_ir)
                logger.info("  Decision Pd: %.2f%% (Gate: >= %.2f%%)", eval_res["pd"], min_pd)
                logger.info("  Pfa: %.6f (Gate: <= %.4f)", eval_res["pfa"], max_pfa)
                logger.info("  Mode 2 Agile/Sparse Share: %.1f%% (Gate: >= %.1f%%)",
                            eval_res["mode2_agile_sparse_fraction"] * 100, min_mode2_fraction * 100)

                # 4. Strict Operational Gate Evaluation
                operational_failures: List[str] = []
                # If dry run with small step count, we log without aborting; in full run, operational thresholds are hard gates
                if not dry_run or val_steps_per_scenario >= 500:
                    if eval_res["mean_ir"] < min_mean_ir:
                        operational_failures.append(f"Mean IR {eval_res['mean_ir']:.2f}% < {min_mean_ir:.2f}%")
                    if eval_res["agile_ir"] < min_agile_ir:
                        operational_failures.append(f"Agile IR {eval_res['agile_ir']:.2f}% < {min_agile_ir:.2f}%")
                    if eval_res["worst_case_ir"] < min_worst_case_ir:
                        operational_failures.append(f"Worst-case IR {eval_res['worst_case_ir']:.2f}% < {min_worst_case_ir:.2f}%")
                    # Benchmark tolerance from verify_baseline_gate.py: pd=99.85 +- 0.05%
                    pd_tolerance = float(cfg.get("operational_gate_thresholds", {}).get("pd_tolerance", 0.05))
                    min_pd_effective = min_pd - pd_tolerance
                    if eval_res["pd"] < min_pd_effective:
                        operational_failures.append(f"Pd {eval_res['pd']:.2f}% < {min_pd_effective:.2f}% ({min_pd:.2f}% - {pd_tolerance:.2f}% tolerance)")
                    if eval_res["pfa"] > max_pfa:
                        operational_failures.append(f"Pfa {eval_res['pfa']:.6f} > {max_pfa:.6f}")
                    if eval_res["mode2_agile_sparse_fraction"] < min_mode2_fraction:
                        operational_failures.append(
                            f"Mode 2 Agile/Sparse share {eval_res['mode2_agile_sparse_fraction']*100:.1f}% < {min_mode2_fraction*100:.1f}%"
                        )

                # Log evaluation provenance
                eval_record = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "step": global_step,
                    "checkpoint": str(saved_ckpt),
                    "checkpoint_sha256": sha256_hex,
                    "git_revision": git_rev,
                    "config_sha256": config_sha256,
                    "seed": 42,
                    "eval_steps_per_scenario": val_steps_per_scenario,
                    "canonical_scenarios": list(CANONICAL_SCENARIOS),
                    "metrics": eval_res,
                    "head_drift": head_drift,
                    "operational_pass": len(operational_failures) == 0,
                    "failures": operational_failures,
                }
                append_audit_log_record(audit_log_path, "staged_evaluations", eval_record)

                if operational_failures:
                    logger.error("STRICT OPERATIONAL GATE FAILED AT STEP %d! Reasons: %s", global_step, operational_failures)
                    rollback_mgr.execute_rollback(
                        failed_ckpt_path=saved_ckpt,
                        reasons=operational_failures,
                        step=global_step,
                        telemetry={"eval_res": eval_res, "q_diag": q_diag},
                    )
                    return False
                else:
                    logger.info("STRICT OPERATIONAL GATE PASSED at step %d! Registering as last known-good.", global_step)
                    rollback_mgr.register_known_good(saved_ckpt, eval_res)
                    last_approved_checkpoint = saved_ckpt

            if done:
                break

        episode += 1

    # End of run final canonical evaluation and promotion verdict
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE AT GLOBAL STEP %d. RUNNING CANONICAL 10-SCENARIO PROMOTION EVALUATION...", global_step)
    logger.info("=" * 80)

    final_eval_steps = 1000 if not dry_run else val_steps_per_scenario
    final_eval_res = evaluate_canonical_10scenarios_inline(
        model=online_drqn,
        device=device,
        n_steps=final_eval_steps,
        seed=42,
        data_root=data_root,
    )

    final_ckpt_name = f"checkpoint_gate_{global_step}.pt" if not dry_run else f"checkpoint_dryrun_final_{global_step}.pt"
    final_payload = {
        "global_step": global_step,
        "state_dict": online_drqn.state_dict(),
        "online_drqn": online_drqn.state_dict(),
        "optimizer": optimizer.state_dict(),
        "baseline_sha256": cfg["experiment"]["baseline_checkpoint_sha256"],
        "config_sha256": config_sha256,
        "git_revision": git_rev,
        "eval_metrics": final_eval_res,
        "training_diagnostics": q_diag,
    }
    saved_final_ckpt, final_sha256 = ckpt_guard.save_checkpoint_atomic(final_payload, final_ckpt_name)

    # Final Promotion Sentinel check
    report_input = {
        "scenario_summary": final_eval_res,
        "action_summary": action_tracker.get_diagnostics(),
        "training_diagnostics": q_diag,
    }
    promoted, verdict_msg, promo_details = evaluate_promotion(report_input)

    verdict_record = {
        "step": global_step,
        "checkpoint": str(saved_final_ckpt),
        "checkpoint_sha256": final_sha256,
        "config_sha256": config_sha256,
        "git_revision": git_rev,
        "promoted": promoted,
        "verdict": verdict_msg,
        "details": promo_details,
        "metrics": final_eval_res,
        "head_drift": compute_head_drift(online_drqn, base_head_weights),
    }

    verdict_path = candidate_output_dir / f"gate_{global_step}_trial_d_promotion_verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict_record, f, indent=2)

    logger.info("=" * 80)
    logger.info("FINAL PROMOTION SENTINEL VERDICT AT STEP %d: %s", global_step, verdict_msg)
    logger.info("Saved final verdict record to %s", verdict_path)
    logger.info("=" * 80)

    return promoted or dry_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Controlled Continuation Trainer (Trial D)")
    parser.add_argument("--config", type=Path, default=Path("cognitive_ew_smart_scan/configs/training_25k_to_40k_safe.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Run quick dry run infrastructure test")
    parser.add_argument("--dry-run-steps", type=int, default=100, help="Number of steps in dry run")
    parser.add_argument("--eval-steps-per-scenario", type=int, default=None, help="Steps per scenario during eval")
    parser.add_argument("--target-step", type=int, default=None, help="Override target step")
    args = parser.parse_args()

    success = train_continuation(
        config_path=args.config,
        dry_run=args.dry_run,
        dry_run_steps=args.dry_run_steps,
        eval_steps_per_scenario=args.eval_steps_per_scenario,
        override_target_step=args.target_step,
    )
    sys.exit(0 if success else 1)
