"""
DRQN Scheduler Training Loop with Thompson Sampling warmup and BPTT.

Uses RFScanEnv, SequenceReplayBuffer, ThompsonSamplingExplorer, and SmartScanMoE evaluation.
Fixed 2026-09-02:
 - Single canonical DRQN update step (no duplicated blocks).
 - Clean MoE / non-MoE action selection flow.
 - Replay buffer now episode-based (contiguous BPTT sequences, no cross-episode).
"""

import copy
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ..contracts import DWELL_MODES
from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ..environment.scenario_generator import ScenarioSource
from ..data.tsrd_manifest import dataset_fingerprint
from ..models.deinterleaver import PDWTransformerEncoder
from ..models.drqn_scheduler import DRQNScheduler
from ..models.smartscan_moe import SmartScanMoE
from ..preprocessing.normalise import load_normalization_stats, normalization_stats_hash
from ..telemetry.publisher import TelemetryPublisher
from ..telemetry.run_manager import RunManager
from ..telemetry.schema import TELEMETRY_SCHEMA_VERSION, coerce, make_episode_record, make_val_record, reward_reconstruction, shannon_entropy
from ..training.replay_buffer import SequenceReplayBuffer
from ..training.thompson_sampling import ThompsonSamplingExplorer
from ..training.training_gate import require_training_gate
from ..training.val_set import FixedValidationSet

logger = logging.getLogger(__name__)


def _observable_priorities(obs: np.ndarray, features_per_band: int = 10) -> np.ndarray:
    """Extract per-band observable priority (occupancy, feature index 0) from a flat obs.

    Avoids ground truth entirely: feature 0 is the EMA occupancy from belief,
    which is a legitimate receiver-observable signal used for prioritisation.
    """
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.ndim != 1:
        raise ValueError(f"observable_priorities expects flat obs, got shape {obs_arr.shape}")
    if features_per_band <= 0:
        raise ValueError(f"features_per_band must be > 0, got {features_per_band}")
    n_bands = obs_arr.shape[0] // features_per_band
    if n_bands * features_per_band != obs_arr.shape[0]:
        raise ValueError(f"obs length {obs_arr.shape[0]} not a multiple of {features_per_band}")
    return obs_arr[::features_per_band]


# Maps MoE-per-decision attribution keys -> episode telemetry aggregate names.
_SCORE_KEYS = {
    "q_score": "mean_q_score",
    "eager_score": "mean_eager_score",
    "revisit_score": "mean_revisit_score",
    "semantic_score": "mean_semantic_score",
    "preemptive_score": "mean_preemptive_score",
    "fused_score": "mean_fused_score",
    "drqn_rank": "drqn_rank",
}


def _do_drqn_update(
    online_drqn: DRQNScheduler,
    target_drqn: DRQNScheduler,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: dict[str, np.ndarray],
    gamma: float,
    device: torch.device,
    aux_coef: float = 0.1,
    stats: dict | None = None,
    reward_baseline: float = -0.39,
    baseline_momentum: float = 0.99,
) -> float:
    """One Double-DQN BPTT update on a sampled batch. Returns loss value.

    When ``stats`` (a dict) is provided it is filled with RC-2 learning
    telemetry: TD losses, Q statistics (online vs target network), the gradient
    norm after clipping and the number of graded transitions. The return value
    stays a scalar float so existing callers/tests are unaffected.

    Phase 7/8 target-and-mask semantics:
      * ``hit_probs`` are binary (1.0 = the action intercepted).
      * ``intercept_times_us`` are genuine dwell-relative times, NaN for misses
        and padding — never a fabricated 500µs target.
      * ``valid_mask`` excludes padded transitions from every loss.
      * ``burn_in_mask`` marks leading window steps that only warm the LSTM
        hidden state and are excluded from every loss.
      * Time Huber is applied only where ``valid_mask & ~burn_in_mask &
        time_target_valid``; probability BCE and Q loss only where
        ``valid_mask & ~burn_in_mask``.
    """
    valid = torch.tensor(batch["valid_mask"], dtype=torch.bool, device=device)
    burn_in = torch.tensor(batch["burn_in_mask"], dtype=torch.bool, device=device)
    loss_mask = valid & ~burn_in  # real graded transitions only

    # Pure burn-in window (no graded transitions): warm-up only, no update.
    if not loss_mask.any():
        return 0.0

    # The full window is fed through the LSTM: burn-in columns warm the hidden
    # state, graded columns carry the recurrence forward for the losses.
    obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
    rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
    next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
    done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

    q_all, aux, _ = online_drqn(obs_b)
    q_chosen = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

    with torch.inference_mode():
        next_q_online, _, _ = online_drqn(next_obs_b)
        best_actions = next_q_online.argmax(dim=-1, keepdim=True)
        next_q_target, _, _ = target_drqn(next_obs_b)
        next_q = next_q_target.gather(-1, best_actions).squeeze(-1)

    # Track D: reward centering strictly applied to TD target computation
    if loss_mask.any():
        batch_mean = float(rew_b[loss_mask].mean().item())
    else:
        batch_mean = float(rew_b.mean().item())
    updated_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * batch_mean
    centered_rew_b = rew_b - updated_baseline

    targets = centered_rew_b + gamma * next_q * (1.0 - done_b)
    q_loss = loss_fn(q_chosen[loss_mask], targets[loss_mask].detach())

    loss: torch.Tensor = q_loss
    aux_loss: torch.Tensor = torch.zeros((), device=device)
    if aux_coef > 0:
        hit_probs = torch.tensor(batch["hit_probs"], dtype=torch.float32, device=device)
        prob_pred = aux["intercept_prob"].gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
        bce = nn.functional.binary_cross_entropy(prob_pred[loss_mask], hit_probs[loss_mask].detach())

        # Time target only when the transition is a genuine hit.
        # Targets are normalized by 1000.0 (milliseconds) to keep Huber loss bounded [0, 1]
        time_valid = (
            loss_mask
            & torch.tensor(batch["time_target_valid"], dtype=torch.bool, device=device)
        )
        if time_valid.any():
            intercept_times = torch.tensor(batch["intercept_times_us"], dtype=torch.float32, device=device)
            time_pred = aux["intercept_time_us"].gather(-1, act_b.unsqueeze(-1)).squeeze(-1)
            # Delta 0.1 ms (= 100 µs), predictions and targets in ms (/ 1000.0)
            huber = nn.functional.huber_loss(time_pred[time_valid] / 1000.0, intercept_times[time_valid].detach() / 1000.0, delta=0.1)
        else:
            huber = torch.zeros((), device=device)
        aux_loss = bce + huber
        loss = loss + aux_coef * aux_loss

    optimizer.zero_grad()
    loss.backward()
    pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(online_drqn.parameters(), 1.0)
    optimizer.step()
    if stats is not None:
        loss_mask_t = loss_mask
        qm = q_all[loss_mask_t].detach()
        td_err = (targets[loss_mask_t] - q_chosen[loss_mask_t]).detach().abs()
        # Double-DQN target values: target-network Q at online-argmax next actions.
        best_next = next_q[loss_mask_t]
        target_table = next_q_target[loss_mask_t]
        # stats["td_loss"] tracks the genuine Bellman Q loss
        stats["td_loss"] = float(q_loss.item())
        stats["total_loss"] = float(loss.item())
        stats["q_loss"] = float(q_loss.item())
        stats["aux_loss"] = float(aux_loss.item())
        stats["mean_td_error"] = float(td_err.mean().item())
        stats["max_td_error"] = float(td_err.max().item())
        stats["mean_q"] = float(qm.mean().item())
        stats["max_q"] = float(qm.max().item())
        stats["min_q"] = float(qm.min().item())
        stats["q_std"] = float(qm.std().item())
        stats["mean_online_q"] = float(qm.mean().item())
        stats["mean_target_q"] = float(best_next.mean().item())
        stats["max_target_q"] = float(target_table.max().item())
        stats["target_online_gap"] = float(best_next.mean().item() - qm.mean().item())
        pre_gn = float(pre_clip_grad_norm.item()) if torch.is_tensor(pre_clip_grad_norm) else float(pre_clip_grad_norm)
        stats["gradient_norm"] = min(1.0, pre_gn)
        stats["pre_clip_gradient_norm"] = pre_gn
        stats["n_graded_steps"] = int(loss_mask_t.sum().item())
        stats["reward_baseline"] = float(updated_baseline)
        stats["centered_reward_mean"] = float(centered_rew_b[loss_mask_t].mean().item())
        stats["raw_reward_mean"] = float(batch_mean)
    return float(loss.item())


def train_scheduler(
    model_cfg_path: str,
    train_cfg_path: str,
    data_dir_override: str | None = None,
    output_dir_override: str | None = None,
    reset_semantic_memory: bool = False,
    staged_gates: list[int] | None = None,
    stop_at_step: int | None = None,
    resume_checkpoint: str | None = None,
    override_epsilon: float | None = None,
    disable_latency_reward: bool = False,
) -> None:
    """Full DRQN training with Thompson warmup, BPTT, target network, and MoE eval.

    Args:
        model_cfg_path: Path to model_config.yaml.
        train_cfg_path: Path to training_config.yaml.
        data_dir_override: CLI override for dataset root (CLI > YAML > default).
        output_dir_override: CLI override for checkpoint output dir
            (CLI > YAML > default).
        reset_semantic_memory: If True, resets SQLite database before training.
        staged_gates: Step numbers at which to execute quantitative gate evaluation.
        stop_at_step: Step number at which to cleanly stop training.
        resume_checkpoint: Path to checkpoint .pt to resume training from.
        override_epsilon: Optional exploration floor override for diagnostic retraining.
        disable_latency_reward: Optional ablation switch to zero out latency bonus.
    """
    with open(model_cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_cuda = torch.cuda.is_available() and os.getenv("DEVICE", "cuda") != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("Training DRQN on %s seed=%d", device, seed)

    drqn_cfg = full_cfg.get("drqn_scheduler", {})
    moe_cfg = full_cfg.get("smartscan_moe", {})
    reward_cfg = full_cfg.get("reward", {})
    env_cfg = train_cfg.get("environment", {})
    sched_cfg = train_cfg.get("scheduler", {})

    if reset_semantic_memory:
        db_path = Path("data/semantic_memory.db")
        if db_path.exists():
            try:
                db_path.unlink()
                logger.info("Reset semantic memory database at %s", db_path)
            except Exception as exc:
                logger.warning("Failed to unlink semantic memory database: %s", exc)

    n_bands = int(drqn_cfg.get("n_bands", 36))
    n_modes = int(drqn_cfg.get("n_modes", env_cfg.get("n_modes", 5)))
    n_actions = int(drqn_cfg.get("n_actions", n_bands * n_modes))
    band_features = int(env_cfg.get("band_features", 10))
    obs_dim = int(env_cfg.get("obs_dim", n_bands * band_features))

    # Merge reward weights into env config
    env_config = {**env_cfg, **reward_cfg}
    if disable_latency_reward:
        env_config["disable_latency_reward"] = True
    env_config.setdefault("n_bands", n_bands)
    env_config.setdefault("n_modes", n_modes)
    env_config.setdefault("n_actions", n_actions)

    training_mode = train_cfg.get("training_mode", "real_tsrd")
    from ..data.tsrd_root import resolve_tsrd_root

    canonical_root = resolve_tsrd_root(cli_value=data_dir_override, config=train_cfg)
    if training_mode == "real_tsrd":
        require_training_gate(
            data_root=canonical_root,
            deinterleaver_checkpoint=train_cfg.get("deinterleaver_ckpt", "checkpoints/deinterleaver/best.pt"),
            normalization_stats=train_cfg.get("normalization_stats", "checkpoints/deinterleaver/normalization_stats.json"),
            environment_config=env_config,
            model_config=full_cfg,
        )

    # Load trained deinterleaver and normalization stats for perception
    deinterleaver_ckpt = train_cfg.get("deinterleaver_ckpt", "checkpoints/deinterleaver/best.pt")
    norm_stats_path = train_cfg.get("normalization_stats", "checkpoints/deinterleaver/normalization_stats.json")
    
    deinterleaver_model = None
    fit_stats = None
    
    if Path(deinterleaver_ckpt).exists():
        logger.info("Loading trained deinterleaver from %s", deinterleaver_ckpt)
        d_cfg = full_cfg.get("deinterleaver", {})
        deinterleaver_model = PDWTransformerEncoder(
            pdw_dim=d_cfg.get("pdw_dim", 6),
            d_model=d_cfg.get("d_model", 128),
            nhead=d_cfg.get("nhead", 8),
            num_layers=d_cfg.get("num_layers", 4),
            dim_feedforward=d_cfg.get("dim_feedforward", 512),
            dropout=d_cfg.get("dropout", 0.1),
            embed_dim=d_cfg.get("embed_dim", 64),
        )
        state = torch.load(deinterleaver_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        deinterleaver_model.load_state_dict(state, strict=False)
        deinterleaver_model.eval()
        
        # Load normalization stats
        if Path(norm_stats_path).exists():
            fit_stats = load_normalization_stats(norm_stats_path)
            logger.info("Loaded normalization stats from %s", norm_stats_path)
        else:
            logger.warning("Normalization stats not found at %s; perception may be degraded", norm_stats_path)
    else:
        logger.warning("Deinterleaver checkpoint not found at %s; perception disabled", deinterleaver_ckpt)

    # Build the receiver-driven cognitive env from a TSRD/synthetic scenario.
    data_dir = canonical_root
    subset = train_cfg.get("subset", "train")
    world_mode = train_cfg.get("world_mode", "stare")
    observation_mode = train_cfg.get("observation_mode", "scan")

    # For scheduler training: RF world uses STARE (latent truth), receiver observes through IBW
    if world_mode == "stare":
        train_source = ScenarioSource(
            data_root=data_dir,
            mode="stare",  # overridden by source_type
            subset=subset,
            freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
            freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
            time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
            max_pulses=int(env_config.get("max_pulses", 50000)),
            seed=seed,
            source_type="world",
            allow_synthetic_fallback=training_mode == "synthetic",
        )
        logger.info("Scheduler training: RF world source = TSRD STARE (latent truth)")
    else:
        # Fallback for compatibility
        train_source = ScenarioSource(
            data_root=data_dir,
            mode=world_mode,
            subset=subset,
            freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
            freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
            time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
            max_pulses=int(env_config.get("max_pulses", 50000)),
            seed=seed,
            allow_synthetic_fallback=training_mode == "synthetic",
        )
        logger.warning("Scheduler training: RF world source = %s (non-standard)", world_mode)

    # One env; reset() draws a fresh random TSRD file each episode.
    # Semantic memory explicitly disabled during training for clean DRQN ablation baseline
    train_env_config = copy.deepcopy(env_config)
    train_env_config["semantic_memory_enabled"] = False
    env = CognitiveRFScanEnv(
        train_env_config, 
        records=None, 
        seed=seed, 
        records_provider=train_source.sample,
        deinterleaver_model=deinterleaver_model,
        deinterleaver_config={"fit_stats": fit_stats} if fit_stats else {},
        semantic_memory_path=":memory:"
    )
    env.reset()  # populate first episode's records so obs_dim/action checks are valid
    assert env.obs_dim == obs_dim, f"env obs_dim {env.obs_dim} != configured {obs_dim}"
    assert env.action_space.n == n_actions, f"env action space {env.action_space.n} != n_actions {n_actions}"
    assert env.action_space.n == n_bands * n_modes, f"env action space must be n_bands*n_modes = {n_bands * n_modes}"
    
    if env.perception_enabled:
        logger.info("Perception pipeline ENABLED: trained deinterleaver + EmitterTracker active")
    else:
        logger.warning("Perception pipeline DISABLED: no trained deinterleaver loaded")
    if training_mode == "real_tsrd":
        assert env.perception_enabled, "Strict TSRD training requires perception_enabled=True"
        assert env.emitter_tracker is not None, "Strict TSRD training requires EmitterTracker"

    lstm_hidden = int(drqn_cfg.get("lstm_hidden", 256))
    lstm_layers = int(drqn_cfg.get("lstm_layers", 2))

    online_drqn = DRQNScheduler(
        obs_dim=obs_dim,
        n_bands=n_bands,
        n_actions=n_actions,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
    ).to(device)
    target_drqn = copy.deepcopy(online_drqn).to(device)
    target_drqn.eval()

    moe = SmartScanMoE(
        online_drqn,
        {**moe_cfg, "n_bands": n_bands, "n_modes": n_modes, "n_actions": n_actions, "device": str(device)},
    ).to(device)

    optimizer = optim.Adam(online_drqn.parameters(), lr=float(drqn_cfg.get("lr", 1e-4)))
    loss_fn = nn.HuberLoss()

    # WandB optional
    use_wandb = False
    try:
        import wandb  # type: ignore

        wandb.init(project=os.getenv("WANDB_PROJECT", "cognitive-ew-sih"), config={**drqn_cfg, **sched_cfg})
        use_wandb = True
    except Exception as exc:
        logger.info("WandB not available: %s", exc)

    ts_sampler = ThompsonSamplingExplorer(n_bands=n_bands, n_modes=n_modes, seed=seed)
    ts_warmup = int(sched_cfg.get("thompson_warmup_steps", 5000))
    eps_start = float(drqn_cfg.get("eps_start", 1.0))
    eps_end = float(drqn_cfg.get("eps_end", 0.05))
    eps_decay = float(drqn_cfg.get("eps_decay", 10000))
    gamma = float(drqn_cfg.get("gamma", 0.99))
    seq_len = int(sched_cfg.get("seq_len", 16))
    burn_in = int(sched_cfg.get("burn_in", 8))
    batch_size = int(sched_cfg.get("batch_size", 32))
    update_freq = int(sched_cfg.get("update_freq", 4))
    target_update_freq = int(sched_cfg.get("target_update_freq", 1000))
    total_steps = int(stop_at_step) if stop_at_step is not None else int(sched_cfg.get("total_timesteps", 500000))

    buffer = SequenceReplayBuffer(
        capacity=int(sched_cfg.get("replay_buffer_size", 50000)),
        seq_len=seq_len,
        obs_dim=obs_dim,
        burn_in=burn_in,
        seed=seed,
    )

    # Phase 17: canonical layout — never resolve to the ambiguous root
    # (config output_dir="checkpoints" is replaced by the canonical subdir).
    from ..utils.checkpoint_paths import SCHEDULER_DIR, resolve_checkpoint_dir

    output_dir = resolve_checkpoint_dir(
        output_dir_override,
        train_cfg.get("output_dir"),
        SCHEDULER_DIR,
        role="scheduler",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # P0-9: reproducible run directory + telemetry publisher (real metrics only).
    world_mode = train_cfg.get("world_mode", "stare")
    source_files = list(getattr(train_source, "files", []))
    data_fingerprint = dataset_fingerprint(source_files, data_dir, world_mode)
    run = RunManager(
        root=train_cfg.get("runs_dir", "runs"),
        config={**full_cfg, **train_cfg},
        extras={
            "split": subset,
            "mode": world_mode,
            "seed": seed,
            "device": str(device),
            "dataset_root": str(data_dir),
            "dataset_fingerprint": data_fingerprint,
            "semantic_memory_reset": reset_semantic_memory,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        },
    )
    run.write_git_revision()
    telemetry = TelemetryPublisher(run=run)

    # RC-2.9: fixed, documented validation scenario set (reproducible numbers).
    val_cfg = train_cfg.get("validation", {})
    val_set = FixedValidationSet(
        data_root=data_dir,
        subset=str(val_cfg.get("subset", "val")),
        mode="stare",
        n_files=int(val_cfg.get("n_files", 2)),
        seed=int(val_cfg.get("seed", 42)),
        freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_config.get("max_pulses", 50000)),
        allow_synthetic_fallback=bool(val_cfg.get("allow_synthetic_fallback", False)),
    )
    (run.dir / "validation_set.json").write_text(
        json.dumps(coerce(val_set.manifest()), indent=2), encoding="utf-8"
    )
    logger.info("Run %s at %s", run.run_id, run.dir)

    from .staged_gate_evaluator import StagedGateEvaluator

    gate_evaluator = StagedGateEvaluator(
        output_dir=output_dir,
        gates=staged_gates,
        val_files=val_set.files_used,
        env_config=env_config,
        model_config=full_cfg,
        train_config=train_cfg,
        seed=seed,
        device=device,
        semantic_memory_reset=reset_semantic_memory,
        parent_checkpoint=resume_checkpoint,
    )

    global_step = 0
    episode = 0
    best_reward = -float("inf")
    eps = eps_start
    reward_baseline = -0.39

    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(resume_path, map_location=device)
            if "state_dict" in ckpt:
                online_drqn.load_state_dict(ckpt["state_dict"])
                target_drqn.load_state_dict(ckpt.get("target_state_dict", ckpt["state_dict"]))
            if "optimizer_state_dict" in ckpt and optimizer is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as exc:
                    logger.warning("Could not restore optimizer state: %s", exc)
            if "rng_state" in ckpt:
                try:
                    torch.set_rng_state(ckpt["rng_state"])
                except Exception:
                    pass
            if "np_rng_state" in ckpt:
                try:
                    np.random.set_state(ckpt["np_rng_state"])
                except Exception:
                    pass
            global_step = int(ckpt.get("global_step", 0))
            episode = int(ckpt.get("episode", 0)) + 1
            eps = float(ckpt.get("epsilon", eps_start))
            if "reward_baseline" in ckpt:
                reward_baseline = float(ckpt["reward_baseline"])
            elif "reward_baseline" in ckpt.get("metadata", {}).get("extra", {}):
                reward_baseline = float(ckpt["metadata"]["extra"]["reward_baseline"])
            else:
                reward_baseline = -0.39
            best_reward = float(ckpt.get("metadata", {}).get("metrics", {}).get("best_episode_reward", -float("inf")))
            logger.info("Resumed state: global_step=%d, episode=%d, eps=%.4f, reward_baseline=%.4f", global_step, episode, eps, reward_baseline)
            for g in gate_evaluator.gates:
                if g <= global_step:
                    gate_evaluator.completed_gates.add(g)
        else:
            logger.warning("Resume checkpoint not found: %s — starting fresh", resume_path)

    from .eval_batch import get_or_create_fixed_eval_batch, evaluate_q_diagnostics
    fixed_eval_batch = get_or_create_fixed_eval_batch(data_dir=str(canonical_root), device=device)

    while global_step < total_steps:
        obs, _ = env.reset()
        obs_arr = np.asarray(obs)
        assert np.all(np.isfinite(obs_arr)), f"Non-finite obs at ep start: {obs_arr[~np.isfinite(obs_arr)]}"
        assert np.all(obs_arr >= 0), f"Negative obs at ep start: band indices {np.where(obs_arr < 0)}"
        try:
            hidden = online_drqn.init_hidden(1, device)
        except Exception:
            hidden = None
        moe.reset()
        if hidden is not None:
            moe.eager_agent.hidden = hidden

        done = False
        ep_reward = 0.0
        ep_hits = 0
        ep_steps = 0
        # --- RC-2 per-episode telemetry accumulators ---
        ep_band_counts = np.zeros(n_bands, dtype=np.float64)
        ep_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_action_counts = np.zeros(n_actions, dtype=np.float64)
        ep_ns = {"same_argmax": 0, "attributed": 0, "scores": {
            "mean_q_score": 0.0, "mean_eager_score": 0.0, "mean_revisit_score": 0.0,
            "mean_semantic_score": 0.0, "mean_preemptive_score": 0.0, "mean_fused_score": 0.0,
            "drqn_rank": 0.0,
        }}
        ep_q_argmax_band_counts = np.zeros(n_bands, dtype=np.float64)
        ep_learn = {"n_updates": 0, "td_loss": 0.0, "mean_td_error": 0.0, "max_td_error": -1e9,
                    "mean_q": 0.0, "max_q": -1e9, "min_q": 1e9, "q_std": 0.0,
                    "mean_online_q": 0.0, "mean_target_q": 0.0, "max_target_q": -1e9,
                    "target_online_gap": 0.0, "gradient_norm": 0.0}

        consecutive_empty = 0
        last_band = -1
        ep_explore_mode_counts = np.zeros(n_modes, dtype=np.float64)
        ep_greedy_mode_counts = np.zeros(n_modes, dtype=np.float64)

        while not done and global_step < total_steps:
            # ---- Action selection ----
            # Feed the MoE the observable periodic-imminent-arrival urgency so
            # PREEMPTIVE_INTERCEPT selection is driven by the actual prediction
            # pipeline (Phase 5 / Phase 3 no-GT-leakage constraint).
            if hasattr(moe, "set_periodic_urgency_vector") and getattr(env, "belief", None) is not None:
                try:
                    moe.set_periodic_urgency_vector(np.asarray(env.belief.periodic_urgency, dtype=np.float32))
                except Exception:
                    pass
            if override_epsilon is not None:
                eps = float(override_epsilon)
            else:
                eps = eps_end + (eps_start - eps_end) * float(np.exp(-global_step / eps_decay))
            if global_step < ts_warmup:
                use_ts = True
                action = ts_sampler.select_action()
                moe_attr = None
                act_source = "thompson"
            else:
                use_ts = False
                if random.random() < eps:
                    b_rand = random.randint(0, n_bands - 1)
                    m_rand = int(np.random.choice([0, 1, 2], p=[0.10, 0.70, 0.20]))
                    action = b_rand * n_modes + m_rand
                    ep_explore_mode_counts[m_rand] += 1
                    moe_attr = None
                    act_source = "random"
                else:
                    online_drqn.eval()
                    with torch.inference_mode():
                        obs_np = np.asarray(obs, dtype=np.float32)
                        obs_t = torch.from_numpy(obs_np).to(device)
                        action, hidden = online_drqn.act(
                            obs_t,
                            hidden,
                            mode_selection="band_first_decoupled",
                            consecutive_empty=consecutive_empty,
                            tau=0.15,
                        )
                        ep_greedy_mode_counts[int(action % n_modes)] += 1
                        # Diagnostic MoE query for passive telemetry only (MoE quarantined from action selection)
                        try:
                            _, _, moe_attr = moe.select_action(obs_np, hidden)
                        except Exception:
                            moe_attr = None
                    online_drqn.train()
                    act_source = "greedy"

            # ---- Step env ----
            dec_telem = getattr(online_drqn, "last_decision_telemetry", None)
            if act_source in ("thompson", "random") or dec_telem is None:
                dec_telem = {
                    "raw_drqn_action": dec_telem.get("raw_drqn_action") if dec_telem else None,
                    "raw_drqn_band": dec_telem.get("raw_drqn_band") if dec_telem else None,
                    "raw_drqn_mode": dec_telem.get("raw_drqn_mode") if dec_telem else None,
                    "final_action": int(action),
                    "final_band": int(action // n_modes),
                    "final_mode": int(action % n_modes),
                    "action_was_overridden": True,
                    "override_source": act_source,
                    "exploration_source": act_source,
                    "q_selected": dec_telem.get("q_selected") if dec_telem else None,
                    "q_max": dec_telem.get("q_max") if dec_telem else None,
                    "q_mean": dec_telem.get("q_mean") if dec_telem else None,
                    "q_std": dec_telem.get("q_std") if dec_telem else None,
                }
            mode_ctx = {
                "action_score": float(moe_attr.get("action_score", 1.0)) if moe_attr else 1.0,
                "reason": str(moe_attr.get("reason", "mode_preset")) if moe_attr else "mode_preset",
                **dec_telem,
            }
            next_obs, reward, terminated, truncated, info = env.step(action, mode_context=mode_ctx)

            done = bool(terminated or truncated)

            ep_steps += 1
            ep_band_counts[int(action // n_modes)] += 1
            ep_mode_counts[int(action % n_modes)] += 1
            ep_action_counts[int(action)] += 1
            if moe_attr is not None:
                ep_ns["attributed"] += 1
                ep_ns["same_argmax"] += int(bool(moe_attr.get("same_argmax", False)))
                q_arg_band = moe_attr.get("q_argmax_band")
                if q_arg_band is not None and 0 <= int(q_arg_band) < n_bands:
                    ep_q_argmax_band_counts[int(q_arg_band)] += 1
                for moe_key, tel_key in _SCORE_KEYS.items():
                    val = moe_attr.get(moe_key)
                    ep_ns["scores"][tel_key] += float(val) if val is not None and float(val) == float(val) else 0.0

            band = int(action // n_modes)
            hit = bool(info.get("hit", False))
            if hit:
                consecutive_empty = 0
            else:
                if last_band == band or last_band == -1:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 1
            last_band = band

            ts_sampler.update(action, hit)
            buffer.add(
                np.asarray(obs, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                done,
                hit_prob=float(info.get("hit_prob", 1.0 if info["hit"] else 0.0)),
                intercept_time_us=float(info.get("intercept_time_us", float("nan"))),
            )
            obs = next_obs
            ep_reward += float(reward)
            ep_hits += int(info["hit"])
            moe.update(action)
            global_step += 1

            # ---- Learning update ----
            if global_step % update_freq == 0 and buffer.can_sample(batch_size):
                try:
                    batch = buffer.sample(batch_size)
                    upd_stats: dict = {}
                    loss_val = _do_drqn_update(
                        online_drqn,
                        target_drqn,
                        optimizer,
                        loss_fn,
                        batch,
                        gamma,
                        device,
                        stats=upd_stats,
                        reward_baseline=reward_baseline,
                        baseline_momentum=0.99,
                    )
                    if upd_stats:
                        if "reward_baseline" in upd_stats:
                            reward_baseline = float(upd_stats["reward_baseline"])
                        ep_learn["n_updates"] += 1
                        ep_learn["td_loss"] += float(upd_stats["td_loss"])
                        ep_learn["mean_td_error"] += float(upd_stats["mean_td_error"])
                        ep_learn["max_td_error"] = max(ep_learn["max_td_error"], float(upd_stats["max_td_error"]))
                        ep_learn["mean_q"] += float(upd_stats["mean_q"])
                        ep_learn["max_q"] = max(ep_learn["max_q"], float(upd_stats["max_q"]))
                        ep_learn["min_q"] = min(ep_learn["min_q"], float(upd_stats["min_q"]))
                        ep_learn["q_std"] += float(upd_stats["q_std"])
                        ep_learn["mean_online_q"] += float(upd_stats["mean_online_q"])
                        ep_learn["mean_target_q"] += float(upd_stats["mean_target_q"])
                        ep_learn["max_target_q"] = max(ep_learn["max_target_q"], float(upd_stats["max_target_q"]))
                        ep_learn["target_online_gap"] += float(upd_stats["target_online_gap"])
                        ep_learn["gradient_norm"] += float(upd_stats["gradient_norm"])
                        gate_evaluator.record_step_diagnostics(
                            td_loss=float(upd_stats["td_loss"]),
                            q_mean=float(upd_stats["mean_q"]),
                            q_std=float(upd_stats["q_std"]),
                            q_min=float(upd_stats["min_q"]),
                            q_max=float(upd_stats["max_q"]),
                            grad_norm=float(upd_stats["gradient_norm"]),
                            action_source=act_source,
                            obs_finite=bool(np.all(np.isfinite(next_obs))),
                            reward_finite=bool(np.isfinite(reward)),
                        )
                    if use_wandb and global_step % 100 == 0:
                        try:
                            import wandb

                            wandb.log({"train/loss": loss_val, "train/eps": float(eps), "step": global_step})
                        except Exception:
                            pass
                except AssertionError:
                    pass
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        logger.warning("OOM in DRQN update — skipping")
                        torch.cuda.empty_cache()
                    else:
                        raise

            # ---- Periodic Q-margin and stuck-state diagnostics ----
            if global_step % 500 == 0:
                try:
                    q_diag = evaluate_q_diagnostics(online_drqn, fixed_eval_batch, device)
                    logger.info(
                        "Step %d | QDiag: margin=%.5f spread=%.4f rew_base=%.4f | StuckState: Pick B%d-M%d (margin=%.5f, 2nd B%d)",
                        global_step,
                        q_diag["q_margin_mean"],
                        q_diag["q_spread_mean"],
                        reward_baseline,
                        q_diag["stuck_state_top1_band"],
                        q_diag["stuck_state_top1_mode"],
                        q_diag["stuck_state_margin"],
                        q_diag["stuck_state_top2_band"],
                    )
                except Exception as exc:
                    logger.warning("Failed evaluating QDiag at step %d: %s", global_step, exc)

            # ---- Target update ----
            if global_step % target_update_freq == 0:
                target_drqn.load_state_dict(online_drqn.state_dict())

            # ---- Staged Promotion Gate Evaluation ----
            gate_evaluator.check_and_run(
                global_step=global_step,
                episode=episode,
                online_drqn=online_drqn,
                optimizer=optimizer,
                buffer=buffer,
                eps=eps,
                moe=moe,
                reward_baseline=reward_baseline,
                override_epsilon=override_epsilon,
                target_drqn=target_drqn,
            )

            if stop_at_step is not None and global_step >= stop_at_step:
                break

        episode += 1
        fom = env.get_fom()
        intercept_rate = ep_hits / max(1, getattr(env, "current_step", ep_steps))
        exp_tot = max(1e-5, float(np.sum(ep_explore_mode_counts)))
        grd_tot = max(1e-5, float(np.sum(ep_greedy_mode_counts)))
        logger.info(
            "Ep %d | step %d/%d | rew %.2f hits %d ir %.3f eps %.3f | Q-Modes: N=%.1f%% L=%.1f%% S=%.1f%% | Exp-Modes: N=%.1f%% L=%.1f%% S=%.1f%%",
            episode, global_step, total_steps, ep_reward, ep_hits, intercept_rate, eps,
            (ep_greedy_mode_counts[1] / grd_tot) * 100, (ep_greedy_mode_counts[2] / grd_tot) * 100, (ep_greedy_mode_counts[0] / grd_tot) * 100,
            (ep_explore_mode_counts[1] / exp_tot) * 100, (ep_explore_mode_counts[2] / exp_tot) * 100, (ep_explore_mode_counts[0] / exp_tot) * 100,
        )
        if use_wandb:
            try:
                import wandb

                wandb.log({"episode/reward": ep_reward, "episode/hits": ep_hits, "episode/intercept_rate": intercept_rate, "episode": episode, "step": global_step})
            except Exception:
                pass

        # P0-9 / RC-2: publish real episode telemetry (band priorities = observable occupancy).
        band_priorities = [float(v) for v in _observable_priorities(obs)]

        # --- RC-2 core metrics from FiguresOfMerit summary ---
        avg_intercept = fom.get("avg_intercept_time_error_us")
        core = {
            "pd": fom.get("Pd"),
            "pfa": fom.get("Pfa"),
            "intercept_rate": fom.get("avg_intercept_rate"),
            "avg_reward": fom.get("avg_reward"),
            "episode_reward": float(ep_reward),
            "ep_hits": int(ep_hits),
            "ep_steps": int(ep_steps),
            "coverage": fom.get("band_selection_coverage"),
            "discovery_rate": fom.get("discovery_rate"),
            "avg_intercept_time": avg_intercept,
            "avg_intercept_time_error": avg_intercept,
            "pct_correct": fom.get("pct_correct_predictions"),
            "selected_active": fom.get("selected_active_opportunities"),
            "spectrum_active": fom.get("spectrum_active_opportunities"),
        }
        # --- RC-2 reward decomposition totals (avg/step * steps) ---
        _avg_to_total = {
            "reward_novel": "avg_reward_novel_term",
            "reward_hit": "avg_reward_hit_term",
            "reward_miss": "avg_reward_miss_penalty",
            "reward_dwell_cost": "avg_reward_dwell_cost",
            "reward_false_alarm": "avg_reward_false_alarm_penalty",
            "reward_timing": "avg_reward_timing_penalty",
            "reward_priority": "avg_reward_priority_term",
            "reward_info_gain": "avg_reward_info_gain_term",
            "reward_redundant": "avg_reward_redundant_penalty",
            "reward_delay": "avg_reward_delay_penalty",
            "reward_staleness": "avg_reward_staleness_bonus",
        }
        reward_components: dict = {}
        for canon, avg_key in _avg_to_total.items():
            avg_v = fom.get(avg_key)
            reward_components[canon] = (float(avg_v) * ep_steps) if avg_v is not None else None

        reward_breakdown: dict = {}
        for canon, val in reward_components.items():
            if val is not None:
                reward_breakdown[canon] = {
                    "points": float(round(val, 2)),
                    "pct": float(round((abs(val) / max(1e-6, abs(ep_reward))) * 100, 2)),
                }
        reconstructed_sum = float(sum(v for v in reward_components.values() if v is not None))
        logger.info(
            "Ep %d reward breakdown: %s (sum=%.2f, ep_reward=%.2f)",
            episode,
            {k: f"{v['points']:+.1f}pts ({v['pct']:.1f}%)" for k, v in reward_breakdown.items() if abs(v['points']) > 0.01},
            reconstructed_sum,
            ep_reward,
        )

        # --- RC-2 action / mode / band statistics ---
        n_steps_ep = float(ep_steps) if ep_steps else None
        mode_freq = [float(c / n_steps_ep) for c in ep_mode_counts] if n_steps_ep else None
        band_freq = [float(c / n_steps_ep) for c in ep_band_counts] if n_steps_ep else None
        actions = {
            "unique_actions": int(np.count_nonzero(ep_action_counts)),
            "unique_bands": int(np.count_nonzero(ep_band_counts)),
            "unique_modes": int(np.count_nonzero(ep_mode_counts)),
            "band_selection_counts": [int(c) for c in ep_band_counts],
            "band_selection_frequencies": band_freq,
            "mode_selection_counts": {DWELL_MODES[i]: int(ep_mode_counts[i]) for i in range(n_modes)},
            "mode_selection_frequencies": ({DWELL_MODES[i]: float(ep_mode_counts[i] / n_steps_ep) for i in range(n_modes)}
                                           if n_steps_ep else None),
            "action_entropy": shannon_entropy(ep_action_counts),
            "band_entropy": shannon_entropy(ep_band_counts),
            "mode_entropy": shannon_entropy(ep_mode_counts),
        }

        # --- RC-2 learning statistics (averaged over the episode's updates) ---
        n_upd = ep_learn["n_updates"]
        if n_upd > 0:
            learning = {
                "td_loss": ep_learn["td_loss"] / n_upd,
                "mean_td_error": ep_learn["mean_td_error"] / n_upd,
                "max_td_error": ep_learn["max_td_error"],
                "mean_q": ep_learn["mean_q"] / n_upd,
                "max_q": ep_learn["max_q"],
                "min_q": ep_learn["min_q"],
                "q_std": ep_learn["q_std"] / n_upd,
                "mean_online_q": ep_learn["mean_online_q"] / n_upd,
                "mean_target_q": ep_learn["mean_target_q"] / n_upd,
                "max_target_q": ep_learn["max_target_q"],
                "target_online_gap": ep_learn["target_online_gap"] / n_upd,
                "gradient_norm": ep_learn["gradient_norm"] / n_upd,
                "n_updates": n_upd,
                "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                "replay_size": int(len(buffer)),
                "epsilon": float(eps),
            }
        else:
            learning = {"n_updates": 0, "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                        "replay_size": int(len(buffer)), "epsilon": float(eps)}

        # --- RC-2 MoE attribution aggregates (over greedy MoE decisions only) ---
        n_attr = ep_ns["attributed"]
        if n_attr > 0:
            sq = ep_ns["scores"]
            modal_q_arg = int(np.argmax(ep_q_argmax_band_counts))
            modal_band = int(np.argmax(ep_band_counts)) if ep_steps else None
            moe_agg = {
                "moe_eager_weight": float(moe.eager_weight),
                "moe_revisit_weight": float(moe.revisit_weight),
                "moe_semantic_weight": float(moe.semantic_weight),
                "moe_preemptive_weight": float(moe.preemptive_weight),
                "mean_q_score": sq["mean_q_score"] / n_attr,
                "mean_eager_score": sq["mean_eager_score"] / n_attr,
                "mean_revisit_score": sq["mean_revisit_score"] / n_attr,
                "mean_semantic_score": sq["mean_semantic_score"] / n_attr,
                "mean_preemptive_score": sq["mean_preemptive_score"] / n_attr,
                "mean_fused_score": sq["mean_fused_score"] / n_attr,
                "same_argmax_fraction": float(ep_ns["same_argmax"] / n_attr),
                "mean_drqn_rank": sq["drqn_rank"] / n_attr,
                "moe_rank": 1,
                "selected_band": modal_band,
                "q_argmax_band": modal_q_arg,
                "moe_argmax_band": modal_band,
            }
        else:
            moe_agg = {}

        record = make_episode_record(
            step=global_step,
            episode=episode,
            core=core,
            reward_components=reward_components,
            actions=actions,
            learning=learning,
            moe=moe_agg,
            band_priorities=band_priorities,
            epsilon=float(eps),
        )
        record["avg_intercept_rate"] = float(ep_hits / max(1, getattr(env, "current_step", ep_steps)))
        record["band_selection_entropy"] = float(fom.get("band_selection_coverage", 0.0) or 0.0)
        record["reward_breakdown"] = reward_breakdown
        record["reconstructed_reward_sum"] = reconstructed_sum
        telemetry.update(**record)

        # Periodic MoE evaluation on fixed val scenarios every 5000 steps
        if episode > 0 and global_step % 5000 == 0 and val_set.files_used:
            try:
                val_env = CognitiveRFScanEnv(
                    env_config,
                    records=None,
                    seed=seed,
                    records_provider=val_set.sample,
                    deinterleaver_model=deinterleaver_model,
                    deinterleaver_config={"fit_stats": fit_stats} if fit_stats else {},
                )
                scenario_details = []
                agg = {
                    "rewards": [], "hits": [], "steps": [],
                    "band_counts": None, "mode_counts": None,
                    "action_entropy": [], "band_entropy": [], "mode_entropy": [],
                    "components": {},
                }
                for _ in range(len(val_set.files_used)):
                    scenario_id = val_set.current_scenario_id()
                    obs_v, _ = val_env.reset()
                    try:
                        hidden_v = online_drqn.init_hidden(1, device)
                    except Exception:
                        hidden_v = None
                    moe.reset()
                    if hidden_v is not None:
                        moe.eager_agent.hidden = hidden_v
                    done_v = False
                    r_sum = 0.0
                    h_sum = 0
                    s_sum = 0
                    band_c = np.zeros(n_bands, dtype=np.float64)
                    mode_c = np.zeros(n_modes, dtype=np.float64)
                    while not done_v:
                        with torch.inference_mode():
                            a_v, hidden_v, _ = moe.select_action(obs_v, hidden_v)
                        obs_v, rew_v, term_v, trunc_v, _ = val_env.step(a_v)
                        r_sum += float(rew_v)
                        band_c[int(a_v // n_modes)] += 1
                        mode_c[int(a_v % n_modes)] += 1
                        moe.update(a_v)
                        s_sum += 1
                        done_v = bool(term_v or trunc_v)
                    fs = val_env.get_fom()
                    h_sum = int(fs.get("n_hits", 0))
                    fs = val_env.get_fom()
                    scenario_details.append({
                        "scenario_id": scenario_id,
                        "reward": float(r_sum),
                        "hits": h_sum,
                        "steps": s_sum,
                        "pd": fs.get("Pd"),
                        "coverage": fs.get("band_selection_coverage"),
                        "band_entropy": shannon_entropy(band_c),
                        "mode_entropy": shannon_entropy(mode_c),
                    })
                    agg["rewards"].append(float(r_sum))
                    agg["hits"].append(h_sum)
                    agg["steps"].append(s_sum)
                    agg["band_entropy"].append(float("nan") if shannon_entropy(band_c) is None else shannon_entropy(band_c))
                    agg["mode_entropy"].append(float("nan") if shannon_entropy(mode_c) is None else shannon_entropy(mode_c))
                    if agg["band_counts"] is None:
                        agg["band_counts"] = band_c.copy()
                        agg["mode_counts"] = mode_c.copy()
                    else:
                        agg["band_counts"] += band_c
                        agg["mode_counts"] += mode_c
                    for canon, avg_key in _avg_to_total.items():
                        avg_v = fs.get(avg_key)
                        total_v = (float(avg_v) * s_sum) if avg_v is not None else None
                        agg["components"].setdefault(canon, []).append(total_v)

                n_val = len(agg["rewards"])
                val_components: dict = {}
                for canon in _avg_to_total:
                    vals = [v for v in agg["components"].get(canon, []) if v is not None]
                    val_components[canon] = (float(np.mean(vals)) if vals else None)

                def mean_step(arr):
                    finite = [v for v in arr if v is not None and float(v) == float(v)]
                    return float(np.mean(finite)) if finite else None
                core_val = {
                    "val_reward": float(np.mean(agg["rewards"])),
                    "val_hits": int(np.sum(agg["hits"])),
                    "val_steps": int(np.sum(agg["steps"])),
                    "val_n_scenarios": n_val,
                    "val_pd": mean_step([d.get("pd") for d in scenario_details]),
                    "val_coverage": mean_step([d.get("coverage") for d in scenario_details]),
                }
                entropy_val = {
                    "val_action_entropy": None,
                    "val_band_entropy": mean_step(agg["band_entropy"]),
                    "val_mode_entropy": mean_step(agg["mode_entropy"]),
                    "val_band_selection_counts": [int(c) for c in (agg["band_counts"] if agg["band_counts"] is not None else [])],
                    "val_mode_selection_counts": {DWELL_MODES[i]: int(agg["mode_counts"][i]) for i in range(n_modes)}
                                                  if agg["mode_counts"] is not None else None,
                }
                val_record = make_val_record(
                    step=global_step,
                    episode=episode,
                    core_val=core_val,
                    reward_components=val_components,
                    entropy=entropy_val,
                    validation_set_id=val_set.validation_set_id,
                    validation_files=[str(p) for p, _, _ in val_set.files_used],
                )
                val_record["scenario_details"] = coerce(scenario_details)
                telemetry.update(**val_record)
                avg_val = core_val["val_reward"]
                logger.info("  Val MoE avg_reward %.2f (scenarios=%s)", avg_val,
                            [d["scenario_id"] for d in scenario_details])
                if use_wandb:
                    try:
                        import wandb

                        wandb.log({"val/moe_reward": avg_val, "step": global_step})
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("Val MoE eval skipped at step %d: %s", global_step, exc)

        if ep_reward > best_reward:
            best_reward = ep_reward
            from ..utils.checkpoint_meta import build_train_metadata, save_state

            meta = build_train_metadata(
                split=subset,
                n_bands=n_bands,
                arch="DRQNScheduler+SmartScanMoE",
                seed=seed,
                metrics={"best_episode_reward": float(ep_reward)},
                extra={"mode": world_mode, "obs_dim": int(env_config.get("obs_dim", obs_dim))},
            )
            save_state(online_drqn, output_dir / "best.pt", meta)
            logger.info("  New best reward %.2f — saved best.pt", ep_reward)

        if stop_at_step is not None and global_step >= stop_at_step:
            logger.info("Reached stop_at_step=%d — concluding training run", stop_at_step)
            break

    final_path = output_dir / "final.pt"
    from ..utils.checkpoint_meta import build_train_metadata, save_state, write_checkpoint_metadata

    final_meta = build_train_metadata(
        split=subset,
        n_bands=n_bands,
        arch="DRQNScheduler+SmartScanMoE",
        seed=seed,
        metrics={"best_episode_reward": float(best_reward)},
        extra={"mode": world_mode, "obs_dim": int(env_config.get("obs_dim", obs_dim))},
    )
    save_state(online_drqn, final_path, final_meta)
    # Phase 17: human-readable metadata.json sidecar (contract artifact).
    write_checkpoint_metadata(output_dir / "metadata.json", final_meta, artifacts=["best.pt", "final.pt"])
    from ..utils.experiment_manifest import write_experiment_manifest

    manifest = write_experiment_manifest(
        run.dir / "experiment_manifest.json",
        dataset_fingerprint=data_fingerprint,
        dataset_root=data_dir,
        dataset_mode=world_mode,
        split=subset,
        seed=seed,
        model_configuration=full_cfg,
        training_configuration=train_cfg,
        normalization_stats_hash=(
            normalization_stats_hash(fit_stats) if fit_stats is not None else None
        ),
        checkpoint_metadata=final_meta,
        device=str(device),
        metrics={"best_episode_reward": float(best_reward)},
    )
    write_experiment_manifest(output_dir / "experiment_manifest.json", **{
        key: manifest[key]
        for key in (
            "dataset_fingerprint", "dataset_root", "dataset_mode", "split", "seed",
            "model_configuration", "training_configuration", "normalization_stats_hash",
            "checkpoint_metadata", "device", "metrics",
        )
    })
    logger.info("Scheduler training complete. Final: %s Best: %.2f", final_path, best_reward)
    telemetry.update(step=global_step, episode=episode, type="done",
                     best_reward=float(best_reward), telemetry_schema_version=TELEMETRY_SCHEMA_VERSION)
    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Train DRQN scheduler")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset root (CLI > YAML).")
    parser.add_argument("--output-dir", type=str, default=None, help="Override checkpoint output dir (CLI > YAML).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reset-semantic-memory", action="store_true", help="Reset semantic memory DB before training.")
    parser.add_argument("--staged-gates", type=str, default="1000,5000,25000,100000,200000,300000,500000", help="Comma-separated step gates.")
    parser.add_argument("--stop-at-step", type=int, default=None, help="Stop after reaching this step.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from.")
    parser.add_argument("--override-epsilon", type=float, default=None, help="Hold exploration epsilon at a fixed floor.")
    parser.add_argument("--disable-latency-reward", action="store_true", help="Ablate latency reward bonus.")
    args = parser.parse_args()
    if args.device:
        os.environ["DEVICE"] = args.device
    gates_list = [int(g.strip()) for g in args.staged_gates.split(",") if g.strip()]
    train_scheduler(
        args.model_config,
        args.config,
        data_dir_override=args.data_dir,
        output_dir_override=args.output_dir,
        reset_semantic_memory=args.reset_semantic_memory,
        staged_gates=gates_list,
        stop_at_step=args.stop_at_step,
        resume_checkpoint=args.resume,
        override_epsilon=args.override_epsilon,
        disable_latency_reward=args.disable_latency_reward,
    )