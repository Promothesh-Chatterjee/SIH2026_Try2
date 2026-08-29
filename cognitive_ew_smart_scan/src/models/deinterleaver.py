"""
Transformer-based Metric Learning Deinterleaver.

Encodes 6D PDWs into L2-normalised embeddings for HDBSCAN clustering.
Uses learnable ToA-based positional encoding for irregular pulse sequences.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import hdbscan  # type: ignore

    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False


class ToAPositionalEncoding(nn.Module):
    """Learnable positional encoding based on ToA (not index).

    Maps normalised ToA scalar to d_model via MLP, added to input projection.
    Critical for irregular PRI sequences where index ≠ time.

    Args:
        d_model: Model dimension.
    """

    def __init__(self, d_model: int = 128) -> None:
        """Initialise ToA positional encoding.

        Args:
            d_model: Output dimension.
        """
        super().__init__()
        self.toa_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, x_proj: torch.Tensor, toa_norm: torch.Tensor) -> torch.Tensor:
        """Add ToA encoding to projected inputs.

        Args:
            x_proj: (B, N, d_model) projected PDWs.
            toa_norm: (B, N) normalised ToA in [0,1] (first column of 6D).

        Returns:
            (B, N, d_model) with positional encoding added.
        """
        # (B, N, 1) → (B, N, d_model)
        pos = self.toa_proj(toa_norm.unsqueeze(-1))
        return x_proj + pos


class PDWTransformerEncoder(nn.Module):
    """Transformer encoder for PDW deinterleaving (also aliased as TransformerDeinterleaver).

    Input: (B, N, 6) normalised PDWs (ToA_norm, CF_norm, PW_norm, AoA_sin, AoA_cos, Amp_norm).
    Architecture: Linear(6→d_model) + ToA pos enc + TransformerEncoder(4 layers, 8 heads) + Linear(d_model→embed_dim) + L2 norm.

    Attributes:
        input_proj: Linear(6, d_model).
        pos_encoding: ToA-based learnable encoding.
        transformer: TransformerEncoder batch_first.
        output_proj: Linear(d_model, embed_dim).
    """

    def __init__(
        self,
        pdw_dim: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        embed_dim: int = 64,
    ) -> None:
        """Initialise encoder.

        Args:
            pdw_dim: Input feature dim (6).
            d_model: Transformer hidden dim.
            nhead: Attention heads.
            num_layers: Encoder layers.
            dim_feedforward: FFN dim.
            dropout: Dropout prob.
            embed_dim: Output embedding dim.
        """
        super().__init__()
        self.pdw_dim = pdw_dim
        self.d_model = d_model
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(pdw_dim, d_model)
        self.pos_encoding = ToAPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, embed_dim)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, 6) normalised PDWs.
            src_key_padding_mask: (B, N) bool mask where True = padding.

        Returns:
            (B, N, embed_dim) L2-normalised embeddings.
        """
        # (B, N, d_model)
        x_proj = self.input_proj(x)
        toa_norm = x[:, :, 0]
        x_proj = self.pos_encoding(x_proj, toa_norm)

        # Transformer with optional padding mask
        x = self.transformer(x_proj, src_key_padding_mask=src_key_padding_mask)

        # (B, N, embed_dim) + L2 norm (critical for triplet loss)
        x = self.output_proj(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

    @torch.inference_mode()
    def infer(self, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Efficient inference for variable-length sequences.

        Handles single pulse train (N,6) or batch (B,N,6). Falls back to CPU on OOM.

        Args:
            x: (N,6) or (B,N,6) float tensor/array.
            device: Target device.

        Returns:
            (N, embed_dim) or (B,N,embed_dim) embeddings.
        """
        try:
            target_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        except Exception:
            target_device = torch.device("cpu")

        was_training = self.training
        self.eval()
        try:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(target_device)
            self.to(target_device)
            # No padding mask needed if all sequences same length; caller may pass mask
            out = self.forward(x)
            if out.shape[0] == 1:
                return out.squeeze(0)
            return out
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning("CUDA OOM in infer — falling back to CPU")
                torch.cuda.empty_cache()
                self.to(torch.device("cpu"))
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                x = x.cpu()
                return self.forward(x).squeeze(0) if x.shape[0] == 1 else self.forward(x)
            raise
        finally:
            if was_training:
                self.train()


# Alias for backward compatibility with existing imports
TransformerDeinterleaver = PDWTransformerEncoder


def deinterleave(
    model: PDWTransformerEncoder,
    pdws_norm: np.ndarray,
    device: str = "cpu",
    min_cluster_size: int = 10,
    min_samples: int = 5,
) -> np.ndarray:
    """Run inference + HDBSCAN clustering to assign emitter labels.

    Args:
        model: Trained PDWTransformerEncoder.
        pdws_norm: (N,6) normalised PDWs.
        device: Device for model inference.
        min_cluster_size: HDBSCAN min_cluster_size.
        min_samples: HDBSCAN min_samples.

    Returns:
        (N,) int array of predicted labels (-1 = noise). Returns all -1 if clustering fails
        or all pulses are noise. Handles empty input.
    """
    if pdws_norm.size == 0:
        return np.array([], dtype=np.int32)
    if pdws_norm.shape[0] < min_cluster_size:
        logger.warning("Too few pulses (%d) for clustering — returning all noise", pdws_norm.shape[0])
        return np.full(pdws_norm.shape[0], -1, dtype=np.int32)

    # Inference
    try:
        embeddings = model.infer(pdws_norm, device=device)
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = np.asarray(embeddings)
        # If batch dim was added, squeeze handled in infer; ensure 2D
        if embeddings_np.ndim == 3:
            embeddings_np = embeddings_np[0]
    except Exception as exc:
        logger.error("Model inference failed: %s — returning all noise", exc)
        return np.full(pdws_norm.shape[0], -1, dtype=np.int32)

    # HDBSCAN clustering (euclidean, eom)
    if not _HDBSCAN_AVAILABLE:
        logger.warning("hdbscan not installed — returning all noise")
        return np.full(pdws_norm.shape[0], -1, dtype=np.int32)

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
        )
        labels = clusterer.fit_predict(embeddings_np)
        labels = labels.astype(np.int32)
        n_noise = int(np.sum(labels == -1))
        if n_noise == len(labels):
            logger.warning("HDBSCAN assigned all pulses to noise")
        else:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info("HDBSCAN found %d clusters (%d noise)", n_clusters, n_noise)
        return labels
    except Exception as exc:
        logger.error("HDBSCAN failed: %s — returning all noise", exc)
        return np.full(pdws_norm.shape[0], -1, dtype=np.int32)
