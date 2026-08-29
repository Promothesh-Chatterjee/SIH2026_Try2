"""
Deep Recurrent Q-Network (DRQN) Scheduler.

A Dueling architecture equipped with an LSTM layer to maintain the episodic 
memory of the RF environment's state, used by the MoE Eager Agent.
"""

import torch
import torch.nn as nn

class DRQNScheduler(nn.Module):
    """
    DRQN model utilizing a Dueling network architecture for RL scheduling.
    
    Observes the spectrum occupancy and temporal visitation states to predict 
    the Q-values of monitoring each frequency band.
    """
    def __init__(
        self, 
        obs_dim: int = 360, 
        n_bands: int = 180, 
        lstm_hidden: int = 256, 
        lstm_layers: int = 2
    ) -> None:
        """
        Initializes the DRQN Scheduler.
        
        Args:
            obs_dim: Total size of the input observation vector.
            n_bands: Number of discrete frequency bands (action space).
            lstm_hidden: Hidden size of the LSTM network.
            lstm_layers: Number of LSTM stacked layers.
        """
        super().__init__()
        self.n_bands = n_bands
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # Dense feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, lstm_hidden),
            nn.ReLU()
        )
        
        # Episodic Memory
        self.lstm = nn.LSTM(
            input_size=lstm_hidden, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        
        # Dueling DQN Heads
        # Value Stream estimates the value of being in a given state
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream estimates the relative advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_bands)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the DRQN.
        
        Args:
            x: Observation tensor of shape (Batch, Seq_Len, obs_dim).
            hidden: Tuple of (h_n, c_n) hidden states for the LSTM.
            
        Returns:
            Tuple containing:
            - q_values: Predicted Q-values of shape (Batch, Seq_Len, n_bands).
            - next_hidden: Tuple of updated (h_n, c_n) hidden states.
        """
        features = self.feature_extractor(x)
        
        lstm_out, next_hidden = self.lstm(features, hidden)
        
        # Dueling Networks splitting
        val = self.value_stream(lstm_out)           # (Batch, Seq_Len, 1)
        adv = self.advantage_stream(lstm_out)       # (Batch, Seq_Len, n_bands)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = val + (adv - adv.mean(dim=-1, keepdim=True))
        
        return q_values, next_hidden
