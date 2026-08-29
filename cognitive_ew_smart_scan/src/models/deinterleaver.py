"""
Transformer-based Metric Learning Deinterleaver.

Encodes 6D PDW inputs into a high-dimensional embedding space optimized 
via Triplet Loss, preparing them for downstream clustering (HDBSCAN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDeinterleaver(nn.Module):
    """
    Transformer Encoder for PDW sequence deinterleaving.
    
    Transforms normalized PDWs into a clustered embedding space where pulses 
    from the same emitter are close, and pulses from different emitters are far.
    """
    def __init__(
        self, 
        pdw_dim: int = 6, 
        d_model: int = 128, 
        nhead: int = 8, 
        num_layers: int = 4, 
        dim_feedforward: int = 512, 
        dropout: float = 0.1, 
        embed_dim: int = 64
    ) -> None:
        """
        Initializes the Transformer Deinterleaver.
        
        Args:
            pdw_dim: Dimension of input PDWs (default 6).
            d_model: Internal model dimensionality.
            nhead: Number of multi-head attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of the feedforward network model.
            dropout: Dropout probability.
            embed_dim: Final output embedding dimension.
        """
        super().__init__()
        
        # Initial projection from PDW feature space to model dimension
        self.input_proj = nn.Linear(pdw_dim, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to target embedding space
        self.output_proj = nn.Linear(d_model, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the deinterleaver.
        
        Args:
            x: Tensor of shape (Batch, Seq_Len, pdw_dim).
            
        Returns:
            Tensor of L2-normalized embeddings of shape (Batch, Seq_Len, embed_dim).
        """
        # (Batch, Seq_Len, d_model)
        x = self.input_proj(x)
        
        # (Batch, Seq_Len, d_model)
        x = self.transformer(x)
        
        # (Batch, Seq_Len, embed_dim)
        x = self.output_proj(x)
        
        # L2 normalize embeddings across the feature dimension for metric learning (Triplet Loss)
        x = F.normalize(x, p=2, dim=-1)
        
        return x
