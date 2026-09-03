"""
SmartScan Mixture of Experts (MoE) with EagerAgent + RevisitAgent.

EagerAgent wraps DRQN + min-max norm; RevisitAgent is algorithmic exp decay.
Fused = eager_weight * eager_norm + revisit_weight * revisit_norm; top-K selection.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .drqn_scheduler import DRQNScheduler

logger = logging.getLogger(__name__)


class SmartScanMoE(nn.Module):
    """Mixture of Experts fusing DRQN and revisitation heuristic.

    Contains inner classes EagerAgent and RevisitAgent per spec.
    """

    class EagerAgent:
        """Wraps DRQN, returns normalised Q-values; maintains LSTM hidden state."""

        def __init__(self, drqn: DRQNScheduler, device: torch.device | str = "cpu") -> None:
            """Initialise eager agent.

            Args:
                drqn: DRQN scheduler instance.
                device: Device for inference.
            """
            self.drqn = drqn
            self.device = torch.device(device) if isinstance(device, str) else device
            self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
            self.n_bands = drqn.n_bands

        def reset(self, batch_size: int = 1) -> None:
            """Reset LSTM hidden state.

            Args:
                batch_size: Batch size for hidden init.
            """
            try:
                self.hidden = self.drqn.init_hidden(batch_size, self.device)
            except Exception:
                self.hidden = None

        def get_q(self, obs: torch.Tensor) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor] | None]:
            """Run DRQN forward, return Q-values and next hidden.

            Args:
                obs: (B,T,obs_dim) or (obs_dim,) tensor.

            Returns:
                Tuple (q_values_np (n_bands,), hidden).
            """
            # Ensure (1,1,obs_dim) for single step
            if obs.dim() == 1:
                obs_b = obs.unsqueeze(0).unsqueeze(0)
            elif obs.dim() == 2:
                obs_b = obs.unsqueeze(1)
            else:
                obs_b = obs
            obs_b = obs_b.to(self.device)
            try:
                self.drqn.to(self.device)
            except Exception:
                pass
            with torch.inference_mode():
                q, h = self.drqn(obs_b, self.hidden)
                self.hidden = h
                # Last timestep, first batch
                q_last = q[0, -1].detach().cpu().numpy()
            return q_last, h

        def normalised_scores(self, q_values: np.ndarray) -> np.ndarray:
            """Min-max normalise Q-values to [0,1].

            Args:
                q_values: (n_bands,) raw Q.

            Returns:
                (n_bands,) normalised.
            """
            q_min = float(np.min(q_values))
            q_max = float(np.max(q_values))
            if q_max - q_min < 1e-8:
                return np.zeros_like(q_values, dtype=np.float32)
            return ((q_values - q_min) / (q_max - q_min + 1e-8)).astype(np.float32)

    class RevisitAgent:
        """Algorithmic revisit urgency: exp(decay_rate * (t - last_visit)).

        Ensures no band ignored > max_revisit_gap slots.
        """

        def __init__(self, n_bands: int = 180, decay_rate: float = 0.05, max_revisit_gap: int | None = None) -> None:
            """Initialise revisit agent.

            Args:
                n_bands: Number of bands.
                decay_rate: Exponential rate.
                max_revisit_gap: Max slots before forced revisit.
            """
            self.n_bands = n_bands
            self.decay_rate = decay_rate
            self.max_revisit_gap = int(max_revisit_gap if max_revisit_gap is not None else 200)
            self.last_visit_time = np.zeros(n_bands, dtype=np.float64)
            self.current_t: int = 0

        def update(self, selected_band: int) -> None:
            """Update timestamp for selected band.

            Args:
                selected_band: Band tuned this step.
            """
            self.last_visit_time[int(selected_band)] = float(self.current_t)
            self.current_t += 1

        def scores(self) -> np.ndarray:
            """Return urgency vector exp(decay * (t - last_visit)), normalised to [0,1].

            Returns:
                (n_bands,) float32 in [0,1].
            """
            raw = np.exp(self.decay_rate * (self.current_t - self.last_visit_time))
            # Min-max to [0,1] for fusion
            r_min, r_max = float(np.min(raw)), float(np.max(raw))
            if r_max - r_min < 1e-8:
                return np.zeros_like(raw, dtype=np.float32)
            norm = (raw - r_min) / (r_max - r_min + 1e-8)
            # Enforce max gap: if any band exceeds gap, boost to 1
            overdue = (self.current_t - self.last_visit_time) > self.max_revisit_gap
            norm[overdue] = 1.0
            return norm.astype(np.float32)

        def reset(self) -> None:
            """Reset visit times."""
            self.last_visit_time.fill(0.0)
            self.current_t = 0

    def __init__(self, drqn_agent: DRQNScheduler, config: dict[str, Any] | None = None) -> None:
        """Initialise MoE fusion.

        Args:
            drqn_agent: DRQN scheduler instance.
            config: Dict with eager_weight, revisit_weight, k_receivers, decay_rate, n_bands.
        """
        super().__init__()
        config = config or {}
        self.n_bands: int = int(config.get("n_bands", getattr(drqn_agent, "n_bands", 36)))
        self.eager_weight: float = float(config.get("eager_weight", 0.6))
        self.revisit_weight: float = float(config.get("revisit_weight", 0.4))
        self.k_receivers: int = int(config.get("k_receivers", 1))
        self.decay_rate: float = float(config.get("decay_rate", 0.05))
        self.max_revisit_gap: int = int(config.get("max_revisit_gap", 200))

        # Inner agents per spec
        device = config.get("device", "cpu")
        self.eager_agent = SmartScanMoE.EagerAgent(drqn_agent, device=device)
        self.revisit_agent = SmartScanMoE.RevisitAgent(
            n_bands=self.n_bands,
            decay_rate=self.decay_rate,
            max_revisit_gap=self.max_revisit_gap,
        )

        # Keep direct refs for torch MoE forward
        self.drqn = drqn_agent
        self._config = config
        logger.info("SmartScanMoE eager=%.1f revisit=%.1f K=%d", self.eager_weight, self.revisit_weight, self.k_receivers)

    def select_bands(
        self, obs: np.ndarray | torch.Tensor, eager_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[list[int], tuple[torch.Tensor, torch.Tensor] | None, dict[str, float]]:
        """Select top-K bands via fused scores.

        Args:
            obs: Observation vector (2*n_bands,) numpy or (B,T,obs_dim) torch.
            eager_hidden: Optional LSTM hidden state (if torch obs).

        Returns:
            Tuple (selected_band_indices List[int] len K, hidden, attribution_dict {eager_pct, revisit_pct}).
        """
        # Handle numpy obs path (most common for env)
        if isinstance(obs, np.ndarray):
            # Eager via DRQN
            obs_t = torch.from_numpy(obs.astype(np.float32))
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0).unsqueeze(0)
            elif obs_t.dim() == 2:
                obs_t = obs_t.unsqueeze(1)
            q_raw, hidden = self.eager_agent.get_q(obs_t.squeeze(0) if obs_t.shape[0] == 1 else obs_t)
            eager_norm = self.eager_agent.normalised_scores(q_raw)
            revisit_norm = self.revisit_agent.scores()
        else:
            # Torch path
            if isinstance(obs, torch.Tensor):
                q, hidden = self.drqn(obs, eager_hidden)
                # Store hidden for next call
                self.eager_agent.hidden = hidden
                q_last = q[0, -1].detach().cpu().numpy() if q.dim() == 3 else q.detach().cpu().numpy()
                if q_last.ndim > 1:
                    q_last = q_last[0]
                eager_norm = self.eager_agent.normalised_scores(q_last)
                revisit_norm = self.revisit_agent.scores()
            else:
                raise TypeError(f"Unsupported obs type {type(obs)}")

        fused = self.eager_weight * eager_norm + self.revisit_weight * revisit_norm
        # Top-K
        k = min(self.k_receivers, self.n_bands)
        top_k = np.argsort(fused)[-k:][::-1].tolist()

        # Attribution for explainability
        eager_contrib = float(np.sum(self.eager_weight * eager_norm[top_k]))
        revisit_contrib = float(np.sum(self.revisit_weight * revisit_norm[top_k]))
        total = eager_contrib + revisit_contrib + 1e-8
        attribution = {"eager_pct": float(eager_contrib / total), "revisit_pct": float(revisit_contrib / total)}

        logger.debug("MoE fused top=%s attribution=%s", top_k, attribution)
        return top_k, hidden if "hidden" in locals() else eager_hidden, attribution

    def update(self, selected_band: int) -> None:
        """Update revisit agent after action.

        Args:
            selected_band: Band tuned.
        """
        self.revisit_agent.update(int(selected_band))

    def reset(self) -> None:
        """Reset both agents' episodic state."""
        self.eager_agent.reset()
        self.revisit_agent.reset()

    # Torch forward for batched training (keeps old API)
    def forward(
        self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        """Batched fused scores for training.

        Args:
            obs: (B,T,2*n_bands).
            hidden: LSTM hidden.

        Returns:
            Tuple (fused_scores (B,T,n_bands), next_hidden, attribution dict tensors).
        """
        q_values, next_hidden = self.drqn(obs, hidden)
        q_min = q_values.min(dim=-1, keepdim=True)[0]
        q_max = q_values.max(dim=-1, keepdim=True)[0]
        q_norm = (q_values - q_min) / (q_max - q_min + 1e-8)
        # Revisit urgency from obs time_since half
        n = self.n_bands
        time_since = obs[:, :, n:]
        urgency = 1.0 - torch.exp(-self.decay_rate * time_since * 100.0)
        eager_contrib = self.eager_weight * q_norm
        revisit_contrib = self.revisit_weight * urgency
        fused = eager_contrib + revisit_contrib
        attribution = {"eager_contribution": eager_contrib.detach(), "revisit_contribution": revisit_contrib.detach(), "raw_q_values": q_values.detach()}
        return fused, next_hidden, attribution
