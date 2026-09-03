"""
Deep Recurrent Q-Network (DRQN) Scheduler with Dueling architecture.

LSTM hidden state is the episodic memory of the cognitive EW system.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DRQNScheduler(nn.Module):
    """Dueling DRQN for RF band scheduling.

    Input: (B, T, obs_dim) for training (T=seq_len) or (B,1,obs_dim) for inference.
    Architecture: LayerNorm → LSTM(2×256) → Dueling Q-head (V(s) + A(s,a) - mean A).

    Attributes:
        input_norm: LayerNorm on obs_dim.
        lstm: LSTM batch_first, 2 layers.
        value_stream: V(s) head.
        advantage_stream: A(s,a) head.
    """

    def __init__(
        self,
        obs_dim: int = 360,
        n_bands: int = 36,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
    ) -> None:
        """Initialise DRQN.

        Args:
            obs_dim: Observation dim (n_bands * features_per_band).
            n_bands: Action space size (default 36).
            lstm_hidden: LSTM hidden units.
            lstm_layers: LSTM depth.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.n_bands = n_bands
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
            nn.Linear(256, n_bands),
        )

        # Also keep q_head alias for compatibility (combined)
        self.q_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, n_bands),
        )
        logger.info("DRQNScheduler(d=%d, bands=%d, hidden=%d, layers=%d) dueling", obs_dim, n_bands, lstm_hidden, lstm_layers)

    def forward(
        self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning Q-values and hidden state.

        Q(s,a) = V(s) + A(s,a) - mean_a A(s,a) for stability.

        Args:
            obs: (B, T, obs_dim) or (B, 1, obs_dim).
            hidden: Optional (h, c) each (lstm_layers, B, hidden).

        Returns:
            Tuple (q_values (B,T,n_bands), (h_n, c_n)).
            Hidden is carried across steps/episodes.
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
        a = self.advantage_stream(lstm_out)  # (B,T,n_bands)
        q_values = v + a - a.mean(dim=-1, keepdim=True)

        return q_values, hidden_out

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
        q, h = self.forward(obs, hidden)
        action = int(torch.argmax(q[0, -1]).item())
        return action, h
