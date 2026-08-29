"""
SmartScan Mixture of Experts (MoE) Module.

Combines the Eager Agent (DRQN output) with a heuristic Revisit Agent to 
produce a fused set of final scheduling action probabilities or scores.
"""

import torch
import torch.nn as nn
from typing import Any

from .drqn_scheduler import DRQNScheduler

class RevisitExpert(nn.Module):
    """
    Algorithmic heuristic agent that generates urgency scores based purely on 
    how long a frequency band has remained unmonitored.
    """
    def __init__(self, n_bands: int = 180, decay_rate: float = 0.05) -> None:
        """
        Initializes the Revisit Expert.
        
        Args:
            n_bands: Number of frequency bands.
            decay_rate: Rate at which urgency increases over unvisited time.
        """
        super().__init__()
        self.n_bands = n_bands
        self.decay_rate = decay_rate
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Calculates urgency scores based on temporal neglect.
        
        Args:
            obs: Observation tensor of shape (Batch, Seq_Len, 2 * n_bands).
            
        Returns:
            Tensor of shape (Batch, Seq_Len, n_bands) with urgency scores in [0, 1].
        """
        # The environment outputs [occupancy (0:n_bands), time_since (n_bands:2*n_bands)]
        time_since = obs[:, :, self.n_bands:]
        
        # Exponential decay heuristic: 1 - exp(-k * time)
        # Urgency approaches 1 as time_since increases
        urgency = 1.0 - torch.exp(-self.decay_rate * time_since * 100.0) 
        return urgency


class SmartScanMoE(nn.Module):
    """
    Mixture of Experts architecture fusing the DRQN (Eager) and Heuristic (Revisit) models.
    """
    def __init__(self, drqn_agent: DRQNScheduler, config: dict[str, Any]) -> None:
        """
        Initializes the MoE system.
        
        Args:
            drqn_agent: Instance of the DRQNScheduler.
            config: Configuration dictionary for MoE hyperparameters.
        """
        super().__init__()
        self.eager_agent = drqn_agent
        self.n_bands = config.get("n_bands", 180)
        
        self.revisit_agent = RevisitExpert(
            n_bands=self.n_bands, 
            decay_rate=config.get("decay_rate", 0.05)
        )
        
        self.eager_weight = config.get("eager_weight", 0.6)
        self.revisit_weight = config.get("revisit_weight", 0.4)
        
    def forward(
        self, 
        obs: torch.Tensor, 
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Computes the fused action scores.
        
        Args:
            obs: Observation tensor of shape (Batch, Seq_Len, 2 * n_bands).
            hidden: LSTM hidden state tuple.
            
        Returns:
            Tuple containing:
            - fused_scores: Final action scores (Batch, Seq_Len, n_bands).
            - next_hidden: Updated LSTM hidden states.
            - attribution: Dictionary mapping agent name to its score contribution tensor.
        """
        # 1. Eager Agent (DRQN)
        q_values, next_hidden = self.eager_agent(obs, hidden)
        
        # Normalize Q-values strictly for fair MoE weighting (min-max norm over actions)
        q_min = q_values.min(dim=-1, keepdim=True)[0]
        q_max = q_values.max(dim=-1, keepdim=True)[0]
        # Adding epsilon to prevent div by zero
        q_norm = (q_values - q_min) / (q_max - q_min + 1e-8)
        
        # 2. Revisit Agent (Heuristic)
        revisit_urgency = self.revisit_agent(obs)
        
        # 3. Static MoE Fusion
        eager_contribution = self.eager_weight * q_norm
        revisit_contribution = self.revisit_weight * revisit_urgency
        fused_scores = eager_contribution + revisit_contribution
        
        attribution = {
            "eager_contribution": eager_contribution.detach(),
            "revisit_contribution": revisit_contribution.detach(),
            "raw_q_values": q_values.detach()
        }
        
        return fused_scores, next_hidden, attribution
