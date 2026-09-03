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
from ..models.drqn_scheduler import DRQNScheduler
from ..models.smartscan_moe import SmartScanMoE
from ..training.replay_buffer import SequenceReplayBuffer
from ..training.thompson_sampling import ThompsonSamplingExplorer

logger = logging.getLogger(__name__)


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


def train_scheduler(model_cfg_path: str, train_cfg_path: str) -> None:
    """Full DRQN training with Thompson warmup, BPTT, target network, and MoE eval.

    Args:
        model_cfg_path: Path to model_config.yaml.
        train_cfg_path: Path to training_config.yaml.
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

    n_bands = int(drqn_cfg.get("n_bands", 180))
    band_features = int(env_cfg.get("band_features", 9))
    obs_dim = int(env_cfg.get("obs_dim", n_bands * band_features))

    # Merge reward weights into env config
    env_config = {**env_cfg, **reward_cfg}
    env_config.setdefault("n_bands", n_bands)

    # Build the receiver-driven cognitive env from a TSRD/synthetic scenario.
    data_dir = train_cfg.get("data_dir", "data")
    subset = train_cfg.get("subset", "train")
    mode = train_cfg.get("mode", "scan")

    train_source = ScenarioSource(
        data_root=data_dir,
        mode=mode,
        subset=subset,
        freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_config.get("max_pulses", 50000)),
        seed=seed,
    )
    # One env; reset() draws a fresh random TSRD file each episode.
    env = CognitiveRFScanEnv(env_config, records=None, seed=seed, records_provider=train_source.sample)
    env.reset()  # populate first episode's records so obs_dim/action checks are valid
    assert env.obs_dim == obs_dim, f"env obs_dim {env.obs_dim} != configured {obs_dim}"
    assert env.action_space.n == n_bands, f"env action space {env.action_space.n} != n_bands {n_bands}"

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

    output_dir = Path(train_cfg.get("output_dir", "checkpoints/scheduler"))
    output_dir.mkdir(parents=True, exist_ok=True)

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

        # Periodic MoE evaluation on val scenarios every 5000 steps
        if episode > 0 and global_step % 5000 == 0:
            try:
                val_source = ScenarioSource(
                    data_root=data_dir,
                    mode=mode,
                    subset="val",
                    freq_min_mhz=float(env_config.get("freq_min_mhz", 0.0)),
                    freq_max_mhz=float(env_config.get("freq_max_mhz", 18000.0)),
                    time_horizon_us=float(env_config.get("time_horizon_us", 0.0)) or None,
                    max_pulses=int(env_config.get("max_pulses", 50000)),
                    seed=seed,
                )
                val_env = CognitiveRFScanEnv(env_config, records=None, seed=seed, records_provider=val_source.sample)
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
            torch.save(online_drqn.state_dict(), output_dir / "best.pt")
            logger.info("  New best reward %.2f — saved best.pt", ep_reward)

    final_path = output_dir / "final.pt"
    torch.save(online_drqn.state_dict(), final_path)
    logger.info("Scheduler training complete. Final: %s Best: %.2f", final_path, best_reward)
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
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/scheduler")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    if args.device:
        os.environ["DEVICE"] = args.device
    train_scheduler(args.model_config, args.config)