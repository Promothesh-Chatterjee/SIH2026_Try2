"""
Triplet Loss Training Loop for the Transformer Deinterleaver.

CRITICAL CONSTRAINT: Emitter labels in the TSRD dataset are locally unique
per .h5 file. Label '1' in file_A.h5 is entirely unrelated to label '1' in
file_B.h5. This script enforces file isolation in all batching logic.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from turing_deinterleaving_challenge import PulseTrain
except ImportError:
    PulseTrain = None

from ..models.deinterleaver import TransformerDeinterleaver
from ..preprocessing.normalise import normalise_pdws

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Helper
# ---------------------------------------------------------------------------

def load_file_for_training(
    file_path: Path,
    max_pulses: int,
    fit_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Loads a single .h5 file, normalises PDWs, and truncates/samples if needed.

    CRITICAL: Labels from this file are only valid within this call. Never 
    compare or mix labels across different .h5 files.

    Args:
        file_path: Absolute path to the .h5 PulseTrain file.
        max_pulses: Maximum number of pulses to use (sampled randomly if exceeded).
        fit_stats: Optional pre-computed normalisation statistics.

    Returns:
        Tuple of (normalised_pdws, labels, fit_stats_used).
    """
    if PulseTrain is None:
        raise ImportError("turing_deinterleaving_challenge is not installed.")

    pt = PulseTrain.load(file_path)
    raw = pt.data       # shape (N, 5)
    labels = pt.labels  # shape (N,) — file-local integers

    if len(raw) == 0:
        logger.warning(f"Empty pulse train: {file_path}")
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int64), fit_stats or {}

    # Subsample if too large
    if len(raw) > max_pulses:
        idx = np.random.choice(len(raw), max_pulses, replace=False)
        raw    = raw[idx]
        labels = labels[idx]

    pdws_norm, stats = normalise_pdws(raw, fit_stats)
    return pdws_norm, labels.astype(np.int64), stats


def mine_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch-hard triplet mining within a single file's embeddings.

    Selects the hardest negative (closest negative pair) and semi-hardest 
    positive for each anchor.

    CRITICAL: labels here are file-local. Never pass embeddings from multiple
    .h5 files into this function simultaneously.

    Args:
        embeddings: Tensor of shape (N, embed_dim), L2 normalised.
        labels: Tensor of shape (N,) — file-local integer labels.
        margin: Triplet loss margin.

    Returns:
        Tuple of (anchor, positive, negative) tensors each of shape (K, embed_dim).
    """
    n = embeddings.size(0)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # (N, N)

    anchors, positives, negatives = [], [], []

    for i in range(n):
        label_i = labels[i]
        pos_mask = (labels == label_i)
        neg_mask = ~pos_mask

        pos_mask[i] = False  # exclude self

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        # Hardest positive: farthest positive
        pos_dists = dist_matrix[i].clone()
        pos_dists[~pos_mask] = -1e9
        hardest_pos = torch.argmax(pos_dists)

        # Hardest negative: closest negative (within margin is fine)
        neg_dists = dist_matrix[i].clone()
        neg_dists[~neg_mask] = 1e9
        hardest_neg = torch.argmin(neg_dists)

        anchors.append(embeddings[i])
        positives.append(embeddings[hardest_pos])
        negatives.append(embeddings[hardest_neg])

    if len(anchors) == 0:
        return (
            torch.zeros(1, embeddings.size(1), device=embeddings.device),
            torch.zeros(1, embeddings.size(1), device=embeddings.device),
            torch.zeros(1, embeddings.size(1), device=embeddings.device),
        )

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def train_deinterleaver(model_cfg_path: str, train_cfg_path: str) -> None:
    """
    Full Triplet Loss training loop for the Transformer Deinterleaver.

    Args:
        model_cfg_path: Path to configs/model_config.yaml.
        train_cfg_path: Path to configs/training_config.yaml.
    """
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)["deinterleaver"]
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    data_root   = Path("data/stare")
    train_files = list((data_root / "train").glob("*.h5"))
    val_files   = list((data_root / "val").glob("*.h5"))

    if not train_files:
        raise FileNotFoundError(f"No training .h5 files found in {data_root / 'train'}")

    max_pulses = train_cfg["deinterleaver"]["max_pulses_per_train"]
    epochs     = train_cfg["deinterleaver"]["epochs"]
    lr         = train_cfg["deinterleaver"]["lr"]
    margin     = model_cfg["triplet_margin"]
    save_every = train_cfg["deinterleaver"]["save_every"]
    val_every  = train_cfg["deinterleaver"]["val_every"]

    model = TransformerDeinterleaver(
        pdw_dim        = model_cfg["pdw_dim"],
        d_model        = model_cfg["d_model"],
        nhead          = model_cfg["nhead"],
        num_layers     = model_cfg["num_layers"],
        dim_feedforward= model_cfg["dim_feedforward"],
        dropout        = model_cfg["dropout"],
        embed_dim      = model_cfg["embed_dim"],
    ).to(device)

    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=train_cfg["deinterleaver"]["weight_decay"])
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs)
    triplet_fn = nn.TripletMarginLoss(margin=margin, p=2)

    # Compute normalisation stats from training set only to prevent leakage
    logger.info("Computing normalisation statistics from training set …")
    fit_stats: dict = {}
    for fp in train_files[:min(50, len(train_files))]:
        _, _, fit_stats = load_file_for_training(fp, max_pulses, fit_stats if fit_stats else None)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_files)
        epoch_loss = 0.0
        n_batches  = 0

        for fp in train_files:
            # CRITICAL: each file is processed independently — labels never cross files
            pdws, labels, _ = load_file_for_training(fp, max_pulses, fit_stats)
            if len(pdws) < 4:
                continue

            pdws_t   = torch.tensor(pdws,   dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 6)
            labels_t = torch.tensor(labels, dtype=torch.long,    device=device)                # (N,)

            embeddings = model(pdws_t).squeeze(0)  # (N, embed_dim)

            anchors, positives, negatives = mine_triplets(embeddings, labels_t, margin)

            loss = triplet_fn(anchors, positives, negatives)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch}/{epochs} — Train TripletLoss: {avg_loss:.4f}")

        # Validation
        if epoch % val_every == 0:
            model.eval()
            val_loss = 0.0
            v_batches = 0
            with torch.inference_mode():
                for fp in val_files:
                    pdws, labels, _ = load_file_for_training(fp, max_pulses, fit_stats)
                    if len(pdws) < 4:
                        continue
                    pdws_t   = torch.tensor(pdws,   dtype=torch.float32, device=device).unsqueeze(0)
                    labels_t = torch.tensor(labels, dtype=torch.long,    device=device)
                    emb = model(pdws_t).squeeze(0)
                    a, p, n = mine_triplets(emb, labels_t, margin)
                    val_loss  += triplet_fn(a, p, n).item()
                    v_batches += 1
            logger.info(f"  Val TripletLoss: {val_loss / max(v_batches, 1):.4f}")

        # Checkpoint
        if epoch % save_every == 0:
            ckpt_path = f"checkpoints/deinterleaver_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  Checkpoint saved → {ckpt_path}")

    # Final save
    torch.save(model.state_dict(), "checkpoints/deinterleaver_final.pt")
    logger.info("Training complete. Final model saved to checkpoints/deinterleaver_final.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_deinterleaver("configs/model_config.yaml", "configs/training_config.yaml")
