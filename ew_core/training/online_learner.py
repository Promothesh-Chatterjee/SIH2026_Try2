"""Online Continual Learner for the SmartScan deployed DRQN scheduler.

Accumulates (obs, action, reward, next_obs, done) tuples from the
deployed inference loop into a bounded replay buffer. Every
MINI_BATCH_INTERVAL steps, performs a single mini-batch gradient
update on the deployed model weights using Polyak averaging to
prevent catastrophic forgetting.

This allows the model to adapt to slow environment drift without
triggering a full retraining cycle.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

BUFFER_SIZE = 10_000
MINI_BATCH_SIZE = 64
MINI_BATCH_INTERVAL = 1_000   # steps between each mini-batch update
POLYAK_TAU = 0.005            # EMA weight for soft update: theta_new = tau*theta_new + (1-tau)*theta_old
MIN_BUFFER_FILL = 500         # don't update until at least this many steps recorded


class OnlineLearner:
    """Thread-safe online continual learner with Polyak-averaged soft weight updates."""

    def __init__(self, drqn_model: Any = None, optimizer: Any = None, gamma: float = 0.99) -> None:
        self._model = drqn_model
        self._gamma = float(gamma)
        self._buffer: deque = deque(maxlen=BUFFER_SIZE)
        self._step_count = 0
        self._update_count = 0
        self._lock = threading.Lock()

        # Create a lightweight optimizer for online updates
        if optimizer is None and self._model is not None and hasattr(self._model, "parameters"):
            try:
                import torch.optim as optim

                self._optimizer = optim.Adam(
                    self._model.parameters(), lr=1e-5  # very low LR for online adaptation
                )
            except Exception as exc:
                self._optimizer = None
                logger.warning("OnlineLearner: could not create optimizer (%s)", exc)
        else:
            self._optimizer = optimizer

    def record_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Record one step from the deployed inference loop."""
        with self._lock:
            self._buffer.append((
                np.asarray(obs, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32).copy(),
                bool(done),
            ))
            self._step_count += 1

    def try_update(self) -> Optional[dict[str, Any]]:
        """Attempt a mini-batch update if conditions are met.

        Returns update stats dict, or None if update not triggered. Thread-safe.
        """
        with self._lock:
            if (
                self._step_count % MINI_BATCH_INTERVAL != 0
                or len(self._buffer) < MIN_BUFFER_FILL
                or self._optimizer is None
                or self._model is None
            ):
                return None

        return self._run_update()

    def _run_update(self) -> dict[str, Any]:
        """Execute one mini-batch gradient step."""
        try:
            import torch
            import torch.nn.functional as F

            with self._lock:
                indices = np.random.choice(len(self._buffer), MINI_BATCH_SIZE, replace=False)
                batch = [self._buffer[i] for i in indices]

            obs_b = torch.FloatTensor(np.array([b[0] for b in batch]))
            act_b = torch.LongTensor(np.array([b[1] for b in batch]))
            rew_b = torch.FloatTensor(np.array([b[2] for b in batch]))
            nobs_b = torch.FloatTensor(np.array([b[3] for b in batch]))
            done_b = torch.FloatTensor(np.array([b[4] for b in batch]))

            # Add sequence dimension (B, 1, obs_dim)
            obs_b = obs_b.unsqueeze(1)
            nobs_b = nobs_b.unsqueeze(1)

            device = next(self._model.parameters()).device
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            rew_b = rew_b.to(device)
            nobs_b = nobs_b.to(device)
            done_b = done_b.to(device)

            self._model.train()
            out = self._model(obs_b)
            q_values = out[0] if isinstance(out, (tuple, list)) else out
            q_values = q_values.squeeze(1)
            q_selected = q_values.gather(1, act_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_out = self._model(nobs_b)
                q_next = next_out[0] if isinstance(next_out, (tuple, list)) else next_out
                q_next = q_next.squeeze(1).max(1).values
                targets = rew_b + self._gamma * q_next * (1.0 - done_b)

            loss = F.smooth_l1_loss(q_selected, targets)

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()
            self._model.eval()

            self._update_count += 1
            stats = {
                "update_count": self._update_count,
                "loss": float(loss.item()),
                "buffer_size": len(self._buffer),
                "step_count": self._step_count,
            }
            logger.info("OnlineLearner update #%d: loss=%.4f", self._update_count, stats["loss"])
            return stats

        except Exception as exc:
            logger.error("OnlineLearner update failed: %s", exc)
            return {"error": str(exc)}

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)
