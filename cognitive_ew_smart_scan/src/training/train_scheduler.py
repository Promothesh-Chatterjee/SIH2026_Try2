"""
DRQN Scheduler Training Loop with Thompson Sampling warmup and BPTT.

Uses RFScanEnv, SequenceReplayBuffer, ThompsonSamplingExplorer, and SmartScanMoE evaluation.
Fixed 2026-09-02:
 - Single canonical DRQN update step (no duplicated blocks).
 - Clean MoE / non-MoE action selection flow.
 - Replay buffer now episode-based (contiguous BPTT sequences, no cross-episode).
"""

import copy
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ..environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ..environment.scenario_generator import ScenarioSource
from ..models.deinterleaver import PDWTransformerEncoder
from ..models.drqn_scheduler import DRQNScheduler
from ..models.smartscan_moe import SmartScanMoE
from ..preprocessing.normalise import load_normalization_stats
from ..telemetry.publisher import TelemetryPublisher
from ..telemetry.run_manager import RunManager
from ..training.replay_buffer import SequenceReplayBuffer
from ..training.thompson_sampling import ThompsonSamplingExplorer

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


def _do_drqn_update(
    online_drqn: DRQNScheduler,
    target_drqn: DRQNScheduler,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: dict[str, np.ndarray],
    gamma: float,
    device: torch.device,
) -> float:
    """One Double-DQN BPTT update on a sampled batch. Returns loss value."""
    obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
    rew_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
    next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
    done_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

    q_all, _ = online_drqn(obs_b)
    q_chosen = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)

    with torch.inference_mode():
        next_q_online, _ = online_drqn(next_obs_b)
        best_actions = next_q_online.argmax(dim=-1, keepdim=True)
        next_q_target, _ = target_drqn(next_obs_b)
        next_q = next_q_target.gather(-1, best_actions).squeeze(-1)

    targets = rew_b + gamma * next_q * (1.0 - done_b)
    loss = loss_fn(q_chosen, targets.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_drqn.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def train_scheduler(
    model_cfg_path: str,
    train_cfg_path: str,
    data_dir_override: str | None = None,
    output_dir_override: str | None = None,
) -> None:
    """Full DRQN training with Thompson warmup, BPTT, target network, and MoE eval.

    Args:
        model_cfg_path: Path to model_config.yaml.
        train_cfg_path: Path to training_config.yaml.
        data_dir_override: CLI override for dataset root (CLI > YAML > default).
        output_dir_override: CLI override for checkpoint output dir
            (CLI > YAML > default).
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

    n_bands = int(drqn_cfg.get("n_bands", 36))
    band_features = int(env_cfg.get("band_features", 10))
    obs_dim = int(env_cfg.get("obs_dim", n_bands * band_features))

    # Merge reward weights into env config
    env_config = {**env_cfg, **reward_cfg}
    env_config.setdefault("n_bands", n_bands)

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
    data_dir = data_dir_override if data_dir_override is not None else train_cfg.get("data_dir", "data")
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
            allow_synthetic_fallback=False,  # No silent fallback for real TSRD training
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
            allow_synthetic_fallback=False,
        )
        logger.warning("Scheduler training: RF world source = %s (non-standard)", world_mode)

    # One env; reset() draws a fresh random TSRD file each episode.
    env = CognitiveRFScanEnv(
        env_config, 
        records=None, 
        seed=seed, 
        records_provider=train_source.sample,
        deinterleaver_model=deinterleaver_model,
        deinterleaver_config={"fit_stats": fit_stats} if fit_stats else {},
    )
    env.reset()  # populate first episode's records so obs_dim/action checks are valid
    assert env.obs_dim == obs_dim, f"env obs_dim {env.obs_dim} != configured {obs_dim}"
    assert env.action_space.n == n_bands, f"env action space {env.action_space.n} != n_bands {n_bands}"
    
    if env.perception_enabled:
        logger.info("Perception pipeline ENABLED: trained deinterleaver + EmitterTracker active")
    else:
        logger.warning("Perception pipeline DISABLED: no trained deinterleaver loaded")

    lstm_hidden = int(drqn_cfg.get("lstm_hidden", 256))
    lstm_layers = int(drqn_cfg.get("lstm_layers", 2))

    online_drqn = DRQNScheduler(obs_dim=obs_dim, n_bands=n_bands, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers).to(device)
    target_drqn = copy.deepcopy(online_drqn).to(device)
    target_drqn.eval()

    moe = SmartScanMoE(online_drqn, {**moe_cfg, "n_bands": n_bands, "device": str(device)}).to(device)

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

    ts_sampler = ThompsonSamplingExplorer(n_bands=n_bands, seed=seed)
    ts_warmup = int(sched_cfg.get("thompson_warmup_steps", 5000))
    eps_start = float(drqn_cfg.get("eps_start", 1.0))
    eps_end = float(drqn_cfg.get("eps_end", 0.05))
    eps_decay = float(drqn_cfg.get("eps_decay", 10000))
    gamma = float(drqn_cfg.get("gamma", 0.99))
    seq_len = int(sched_cfg.get("seq_len", 16))
    batch_size = int(sched_cfg.get("batch_size", 32))
    update_freq = int(sched_cfg.get("update_freq", 4))
    target_update_freq = int(sched_cfg.get("target_update_freq", 1000))
    total_steps = int(sched_cfg.get("total_timesteps", 500000))

    buffer = SequenceReplayBuffer(capacity=int(sched_cfg.get("replay_buffer_size", 50000)), seq_len=seq_len, obs_dim=obs_dim, seed=seed)

    output_dir = Path(output_dir_override) if output_dir_override is not None else Path(train_cfg.get("output_dir", "checkpoints/scheduler"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # P0-9: reproducible run directory + telemetry publisher (real metrics only).
    world_mode = train_cfg.get("world_mode", "stare")
    run = RunManager(
        root=train_cfg.get("runs_dir", "runs"),
        config={**full_cfg, **train_cfg},
        extras={"split": subset, "mode": world_mode, "seed": seed, "device": str(device)},
    )
    run.write_git_revision()
    telemetry = TelemetryPublisher(run=run)
    logger.info("Run %s at %s", run.run_id, run.dir)

    global_step = 0
    episode = 0
    best_reward = -float("inf")
    eps = eps_start

    while global_step < total_steps:
        obs, _ = env.reset()
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

        while not done and global_step < total_steps:
            # ---- Action selection ----
            if global_step < ts_warmup:
                use_ts = True
                action = ts_sampler.select_band()
            else:
                eps = eps_end + (eps_start - eps_end) * float(np.exp(-global_step / eps_decay))
                use_ts = False
                if random.random() < eps:
                    action = int(env.action_space.sample())
                else:
                    online_drqn.eval()
                    with torch.inference_mode():
                        obs_np = np.asarray(obs, dtype=np.float32)
                        bands, hidden_out, _ = moe.select_bands(obs_np, hidden)
                        action = int(bands[0])
                        hidden = hidden_out if hidden_out is not None else hidden
                    online_drqn.train()

            # ---- Step env ----
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ts_sampler.update(action, bool(info["hit"]))
            buffer.add(np.asarray(obs, dtype=np.float32), action, float(reward), np.asarray(next_obs, dtype=np.float32), done)
            obs = next_obs
            ep_reward += float(reward)
            ep_hits += int(info["hit"])
            moe.update(action)
            global_step += 1

            # ---- Learning update ----
            if not use_ts and global_step % update_freq == 0 and buffer.can_sample(batch_size):
                try:
                    batch = buffer.sample(batch_size)
                    loss_val = _do_drqn_update(online_drqn, target_drqn, optimizer, loss_fn, batch, gamma, device)
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

            # ---- Target update ----
            if global_step % target_update_freq == 0:
                target_drqn.load_state_dict(online_drqn.state_dict())

        episode += 1
        fom = env.get_fom()
        logger.info("Ep %d | step %d/%d | rew %.2f hits %d Pd %.3f Pfa %.3f eps %.3f", episode, global_step, total_steps, ep_reward, ep_hits, fom["Pd"], fom["Pfa"], eps)
        if use_wandb:
            try:
                import wandb

                wandb.log({"episode/reward": ep_reward, "episode/hits": ep_hits, "episode/Pd": fom["Pd"], "episode/Pfa": fom["Pfa"], "episode": episode, "step": global_step})
            except Exception:
                pass

        # P0-9: publish real episode telemetry (band priorities = observable occupancy).
        band_priorities = [float(v) for v in _observable_priorities(obs)]
        telemetry.update(
            step=global_step,
            episode=episode,
            type="episode",
            pd=float(fom["Pd"]),
            pfa=float(fom["Pfa"]),
            avg_reward=float(fom["avg_reward"]),
            ep_reward=float(ep_reward),
            ep_hits=int(ep_hits),
            epsilon=float(eps),
            band_priorities=band_priorities,
        )

        # Periodic MoE evaluation on val scenarios every 5000 steps
        if episode > 0 and global_step % 5000 == 0:
            try:
                val_source = ScenarioSource(
                    data_root=data_dir,
                    mode="stare",
                    subset="val",
                    freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
                    freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
                    time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
                    max_pulses=int(env_config.get("max_pulses", 50000)),
                    seed=seed,
                    source_type="world",
                    allow_synthetic_fallback=False,
                )
                val_env = CognitiveRFScanEnv(
                    env_config, 
                    records=None, 
                    seed=seed, 
                    records_provider=val_source.sample,
                    deinterleaver_model=deinterleaver_model,
                    deinterleaver_config={"fit_stats": fit_stats} if fit_stats else {},
                )
                val_rewards = []
                for _ in range(min(10, 2)):  # keep quick; expand to 10 when data present
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
                    while not done_v:
                        bands_v, hidden_v, _ = moe.select_bands(obs_v, hidden_v)
                        a_v = int(bands_v[0])
                        obs_v, rew_v, term_v, trunc_v, _ = val_env.step(a_v)
                        r_sum += float(rew_v)
                        moe.update(a_v)
                        done_v = bool(term_v or trunc_v)
                    val_rewards.append(r_sum)
                avg_val = float(np.mean(val_rewards)) if val_rewards else 0.0
                logger.info("  Val MoE avg_reward %.2f", avg_val)
                telemetry.update(step=global_step, episode=episode, type="val", val_reward=float(avg_val))
                if use_wandb:
                    try:
                        import wandb

                        wandb.log({"val/moe_reward": avg_val, "step": global_step})
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Val MoE eval skipped: %s", exc)

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

    final_path = output_dir / "final.pt"
    torch.save(online_drqn.state_dict(), final_path)
    logger.info("Scheduler training complete. Final: %s Best: %.2f", final_path, best_reward)
    telemetry.update(step=global_step, episode=episode, type="done", best_reward=float(best_reward))
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
    args = parser.parse_args()
    if args.device:
        os.environ["DEVICE"] = args.device
    train_scheduler(args.model_config, args.config, data_dir_override=args.data_dir, output_dir_override=args.output_dir)