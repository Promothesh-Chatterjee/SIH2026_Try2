"""
Deep Recurrent Q-Network (DRQN) Scheduler with Dueling architecture.

The action space is the canonical time-frequency joint space: ``n_actions =
n_bands * n_modes`` where ``action = band * n_modes + mode``. For backward
compatibility ``n_actions`` defaults to ``n_bands`` (band-only, NORMAL dwell).

Two auxiliary prediction heads make the scheduler a true dynamic time-frequency
decision system:
  * ``intercept_prob``   (sigmoid)  — P(interception) for the selected action
  * ``intercept_time_us``(softplus) — expected time-to-interception (µs)

LSTM hidden state is the episodic memory of the cognitive EW system.
"""

import logging

import torch
import torch.nn as nn

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_OBS_DIM,
    band_of_action,
    mode_of_action,
    n_actions_for,
)

logger = logging.getLogger(__name__)


class DRQNScheduler(nn.Module):
    """Dueling DRQN for RF time-frequency scheduling.

    Input: (B, T, obs_dim) for training (T=seq_len) or (B,1,obs_dim) for inference.
    Architecture: LayerNorm → LSTM(2×256) → Dueling Q-head (V(s) + A(s,a) - mean A)
    plus auxiliary interception-probability and intercept-time prediction heads.

    Attributes:
        input_norm: LayerNorm on obs_dim.
        lstm: LSTM batch_first, 2 layers.
        value_stream: V(s) head.
        advantage_stream: A(s,a) head.
        intercept_prob_head: P(intercept | s,a) sigmoid head.
        intercept_time_head: expected intercept time (µs) softplus head.
    """

    def __init__(
        self,
        obs_dim: int = CANONICAL_OBS_DIM,
        n_bands: int = CANONICAL_N_BANDS,
        n_actions: int | None = None,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        n_modes: int | None = None,
    ) -> None:
        """Initialise DRQN.

        Args:
            obs_dim: Observation dim (n_bands * features_per_band).
            n_bands: Number of frequency bands.
            n_actions: Action space size. Defaults to ``n_bands * n_modes`` when
                both given, else ``n_bands`` (band-only backward compat).
            lstm_hidden: LSTM hidden units.
            lstm_layers: LSTM depth.
            n_modes: Number of dwell modes (used only when n_actions is not given).
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.n_bands = n_bands
        if n_modes is None:
            n_modes = CANONICAL_N_MODES
        self.n_modes = n_modes
        self.n_actions = int(n_actions if n_actions is not None else n_actions_for(n_bands, n_modes))
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.input_norm = nn.LayerNorm(obs_dim)
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
        )

        # Also keep q_head alias for compatibility (combined)
        self.q_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
        )

        # Auxiliary prediction heads (time-frequency decision system).
        self.intercept_prob_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
            nn.Sigmoid(),
        )
        self.intercept_time_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # shared prediction, see forward
        )

        logger.info(
            "DRQNScheduler(d=%d, bands=%d, modes=%d, actions=%d, hidden=%d, layers=%d) dueling+aux",
            obs_dim, n_bands, n_modes, self.n_actions, lstm_hidden, lstm_layers,
        )

    def forward(
        self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, dict, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning Q-values, aux predictions and hidden state.

        Q(s,a) = V(s) + A(s,a) - mean_a A(s,a) for stability.

        Args:
            obs: (B, T, obs_dim) or (B, 1, obs_dim).
            hidden: Optional (h, c) each (lstm_layers, B, hidden).

        Returns:
            Tuple:
              q_values (B,T,n_actions),
              aux dict with "intercept_prob" (B,T,n_actions) and
                    "intercept_time_us" (B,T,1),
              (h_n, c_n) hidden.
        """
        if obs.size(-1) != self.obs_dim:
            raise ValueError(
                f"Input observation feature dim {obs.size(-1)} does not match expected obs_dim {self.obs_dim}"
            )
        # LayerNorm over last dim
        x = self.input_norm(obs)
        lstm_out, hidden_out = self.lstm(x, hidden)

        # Dueling combination
        v = self.value_stream(lstm_out)  # (B,T,1)
        a = self.advantage_stream(lstm_out)  # (B,T,n_actions)
        q_values = v + a - a.mean(dim=-1, keepdim=True)

        # Aux heads
        intercept_prob = self.intercept_prob_head(lstm_out)  # (B,T,n_actions)
        intercept_time_us = self.intercept_time_head(lstm_out).squeeze(-1)  # (B,T)

        aux = {
            "intercept_prob": intercept_prob,
            "intercept_time_us": intercept_time_us,
        }
        return q_values, aux, hidden_out

    @staticmethod
    def decode_action(action: int, n_bands: int, n_modes: int) -> tuple[int, int]:
        """Decode a flat time-frequency action into (band, mode).

        Args:
            action: Flat action index in [0, n_bands*n_modes).
            n_bands: Number of bands.
            n_modes: Number of dwell modes.

        Returns:
            (band, mode) tuple.
        """
        return band_of_action(action, n_modes), mode_of_action(action, n_modes)

    def init_hidden(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialise LSTM hidden state for episode reset.

        Args:
            batch_size: Batch size.
            device: Target device.

        Returns:
            Tuple (h0, c0) each (lstm_layers, batch_size, lstm_hidden) zeros.
        """
        dev = torch.device(device) if isinstance(device, str) else device
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        return (h0, c0)

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step greedy action for deployment (sub-ms).

        Args:
            obs: (obs_dim,) or (1, obs_dim) or (1,1,obs_dim).
            hidden: Current hidden state.

        Returns:
            Tuple (action int, next_hidden).
        """
        self.eval()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)
        # Ensure batch 1
        q, _aux, h = self.forward(obs, hidden)
        action = int(torch.argmax(q[0, -1]).item())
        return action, h
