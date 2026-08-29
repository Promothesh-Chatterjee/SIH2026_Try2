"""
DRQN Scheduler Training Loop.

Trains the SmartScan MoE system end-to-end using the RFScanEnv Gymnasium 
environment, with Thompson Sampling warmup transitioning to DRQN exploitation.
"""

import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ..environment.rf_scan_env import RFScanEnv
from ..models.drqn_scheduler import DRQNScheduler
from ..models.smartscan_moe import SmartScanMoE
from ..training.replay_buffer import EpisodicReplayBuffer
from ..training.thompson_sampling import ThompsonSampler

logger = logging.getLogger(__name__)


def train_scheduler(model_cfg_path: str, train_cfg_path: str) -> None:
    """
    Full DRQN training loop with Thompson Sampling warmup and target-network 
    soft updates using episodic replay.

    Args:
        model_cfg_path: Path to configs/model_config.yaml.
        train_cfg_path: Path to configs/training_config.yaml.
    """
    with open(model_cfg_path) as f:
        full_cfg = yaml.safe_load(f)

    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)

    drqn_cfg    = full_cfg["drqn_scheduler"]
    moe_cfg     = full_cfg["smartscan_moe"]
    env_cfg     = train_cfg["environment"]
    sched_cfg   = train_cfg["scheduler"]

    seed             = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # -----------------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------------
    env_config = {**env_cfg, **full_cfg["reward"]}
    env = RFScanEnv(env_config, data_dir="data", subset="train")

    n_bands      = drqn_cfg["n_bands"]
    obs_dim      = drqn_cfg["obs_dim"]
    lstm_hidden  = drqn_cfg["lstm_hidden"]
    lstm_layers  = drqn_cfg["lstm_layers"]

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    online_drqn = DRQNScheduler(
        obs_dim     = obs_dim,
        n_bands     = n_bands,
        lstm_hidden = lstm_hidden,
        lstm_layers = lstm_layers,
    ).to(device)

    target_drqn = copy.deepcopy(online_drqn).to(device)
    target_drqn.eval()

    moe = SmartScanMoE(online_drqn, {**moe_cfg, "n_bands": n_bands}).to(device)

    optimizer  = optim.Adam(online_drqn.parameters(), lr=drqn_cfg["lr"])
    loss_fn    = nn.HuberLoss()

    # -----------------------------------------------------------------------
    # Exploration
    # -----------------------------------------------------------------------
    ts_sampler  = ThompsonSampler(n_bands=n_bands, seed=seed)
    ts_warmup   = sched_cfg["thompson_warmup_steps"]
    eps_start   = drqn_cfg["eps_start"]
    eps_end     = drqn_cfg["eps_end"]
    eps_decay   = drqn_cfg["eps_decay"]
    gamma       = drqn_cfg["gamma"]
    seq_len     = sched_cfg["seq_len"]
    batch_size  = sched_cfg["batch_size"]
    update_freq         = sched_cfg["update_freq"]
    target_update_freq  = sched_cfg["target_update_freq"]
    total_steps         = sched_cfg["total_timesteps"]

    buffer = EpisodicReplayBuffer(
        capacity = sched_cfg["replay_buffer_size"],
        seq_len  = seq_len,
        obs_dim  = obs_dim,
        seed     = seed,
    )

    os.makedirs("checkpoints", exist_ok=True)

    global_step = 0
    episode     = 0
    best_ep_reward = -np.inf

    while global_step < total_steps:
        obs, _ = env.reset()
        buffer.start_episode()
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        ep_reward  = 0.0
        ep_hits    = 0
        done       = False

        while not done:
            # --- Action Selection ---
            if global_step < ts_warmup:
                # Thompson Sampling exploration
                action = ts_sampler.sample_action()
            else:
                eps = eps_end + (eps_start - eps_end) * np.exp(-global_step / eps_decay)
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    online_drqn.eval()
                    with torch.inference_mode():
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                        fused, hidden, _ = moe(obs_t, hidden)
                        action = int(fused[0, 0].argmax().cpu())
                    online_drqn.train()

            next_obs, reward, done, _, info = env.step(action)

            # Update Thompson Sampler posterior
            ts_sampler.update(action, info["hit"])

            buffer.add(obs, action, reward, next_obs, done)
            obs        = next_obs
            ep_reward += reward
            ep_hits   += int(info["hit"])
            global_step += 1

            # --- Learning Step ---
            if global_step >= ts_warmup and global_step % update_freq == 0 and buffer.can_sample(batch_size):
                batch = buffer.sample(batch_size)

                obs_b      = torch.tensor(batch["obs"],      dtype=torch.float32, device=device)
                act_b      = torch.tensor(batch["actions"],  dtype=torch.long,    device=device)
                rew_b      = torch.tensor(batch["rewards"],  dtype=torch.float32, device=device)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                done_b     = torch.tensor(batch["dones"],    dtype=torch.float32, device=device)

                # Online Q-values at chosen actions
                q_all, _ = online_drqn(obs_b)
                q_chosen  = q_all.gather(-1, act_b.unsqueeze(-1)).squeeze(-1)  # (B, T)

                # Double DQN target: online selects action, target evaluates
                with torch.inference_mode():
                    next_q_online, _ = online_drqn(next_obs_b)
                    best_actions = next_q_online.argmax(dim=-1, keepdim=True)
                    next_q_target, _ = target_drqn(next_obs_b)
                    next_q = next_q_target.gather(-1, best_actions).squeeze(-1)

                targets = rew_b + gamma * next_q * (1.0 - done_b)
                loss    = loss_fn(q_chosen, targets.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_drqn.parameters(), 1.0)
                optimizer.step()

            # --- Target Network Sync ---
            if global_step % target_update_freq == 0:
                target_drqn.load_state_dict(online_drqn.state_dict())

        buffer.end_episode()
        episode += 1

        logger.info(
            f"Episode {episode} | Step {global_step}/{total_steps} "
            f"| Reward: {ep_reward:.2f} | Hits: {ep_hits}"
        )

        if ep_reward > best_ep_reward:
            best_ep_reward = ep_reward
            torch.save(online_drqn.state_dict(), "checkpoints/drqn_best.pt")

    torch.save(online_drqn.state_dict(), "checkpoints/drqn_final.pt")
    logger.info("Scheduler training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_scheduler("configs/model_config.yaml", "configs/training_config.yaml")
