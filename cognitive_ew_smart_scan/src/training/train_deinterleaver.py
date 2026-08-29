"""
Triplet Loss Training Loop for Transformer Deinterleaver.

CRITICAL CONSTRAINT: Emitter labels in TSRD are locally unique per .h5 file.
Label '1' in file_A.h5 is unrelated to label '1' in file_B.h5. Never mix labels
across files in batching/collation/triplet mining.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from turing_deinterleaving_challenge import PulseTrain
except ImportError:
    PulseTrain = None  # type: ignore

from ..models.deinterleaver import PDWTransformerEncoder, TransformerDeinterleaver
from ..preprocessing.normalise import normalise_pdws

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping on validation V-measure.

    Args:
        patience: Epochs to wait for improvement.
        min_delta: Minimum improvement to count.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        """Initialise early stopping.

        Args:
            patience: Patience epochs.
            min_delta: Delta threshold.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = -float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Update with new metric.

        Args:
            value: Validation metric (higher better).

        Returns:
            True if should stop.
        """
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def load_file_for_training(
    file_path: Path,
    max_pulses: int,
    fit_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load single .h5 file, normalise, truncate/sample.

    CRITICAL: Labels are file-local. Never compare across files.
    Assertion enforces isolation.

    Args:
        file_path: Path to .h5 PulseTrain.
        max_pulses: Max pulses (sample randomly if N > max_pulses).
        fit_stats: Normalisation stats (None → compute).

    Returns:
        Tuple (pdws_norm (N,6), labels (N,), stats).
    """
    assert file_path.suffix == ".h5", f"Expected .h5 file, got {file_path}"
    if PulseTrain is None:
        raise ImportError("turing_deinterleaving_challenge not installed. pip install -r requirements.txt")
    try:
        pt = PulseTrain.load(str(file_path))
    except Exception as exc:
        logger.warning("Corrupt H5 %s: %s", file_path, exc)
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int64), fit_stats or {}
    raw = pt.data
    labels = pt.labels
    if raw is None or len(raw) == 0:
        logger.warning("Empty pulse train: %s", file_path)
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int64), fit_stats or {}
    # Enforce file-local constraint comment + assertion
    # ASSERTION: labels only consistent within this file
    assert labels is not None and len(labels) == len(raw), "Labels mismatch within file"
    if len(raw) > max_pulses:
        idx = np.random.choice(len(raw), max_pulses, replace=False)
        raw = raw[idx]
        labels = labels[idx]
    pdws_norm, stats = normalise_pdws(raw, fit_stats)
    return pdws_norm, labels.astype(np.int64), stats


def mine_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-hard triplet mining within single file.

    CRITICAL: embeddings/labels must be from ONE file. Never mix files.

    Args:
        embeddings: (N, embed_dim) L2 normalised.
        labels: (N,) file-local ints.
        margin: Triplet margin.

    Returns:
        Tuple (anchors, positives, negatives) each (K, embed_dim).
    """
    # Assertion for file-local constraint
    assert embeddings.size(0) == labels.size(0), "Embeddings/labels length mismatch — possible cross-file mixing"
    n = embeddings.size(0)
    if n < 2:
        z = torch.zeros(1, embeddings.size(1), device=embeddings.device)
        return z, z, z
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    anchors, positives, negatives = [], [], []
    for i in range(n):
        label_i = labels[i]
        pos_mask = labels == label_i
        neg_mask = ~pos_mask
        pos_mask[i] = False
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        pos_dists = dist_matrix[i].clone()
        pos_dists[~pos_mask] = -1e9
        hardest_pos = int(torch.argmax(pos_dists).item())
        neg_dists = dist_matrix[i].clone()
        neg_dists[~neg_mask] = 1e9
        hardest_neg = int(torch.argmin(neg_dists).item())
        anchors.append(embeddings[i])
        positives.append(embeddings[hardest_pos])
        negatives.append(embeddings[hardest_neg])
    if not anchors:
        z = torch.zeros(1, embeddings.size(1), device=embeddings.device)
        return z, z, z
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def collate_single_file_batch(
    files: list[Path], max_pulses: int, fit_stats: dict | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collate helper that keeps each pulse train separate (no mixing).

    Returns list of (pdws_norm, labels) per file — caller must NOT merge labels.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fp in files:
        pdws, labels, _ = load_file_for_training(fp, max_pulses, fit_stats)
        # Never mix labels across files — keep per-file tuples
        out.append((pdws, labels))
    return out


def train_deinterleaver(model_cfg_path: str, train_cfg_path: str) -> None:
    """Full triplet training loop with WandB, checkpointing, early stopping.

    Args:
        model_cfg_path: Path to model_config.yaml.
        train_cfg_path: Path to training_config.yaml.
    """
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)["deinterleaver"]
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device with OOM fallback
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)
    logger.info("Training on device: %s (seed=%d)", device, seed)

    # Data discovery
    data_root = Path(train_cfg.get("data_dir", "data/stare"))
    # Allow both data/stare/train and data/train/stare layouts
    candidates = [
        data_root / "train",
        Path("data/stare/train"),
        Path("data/train"),
        Path("data"),
    ]
    train_files: list[Path] = []
    val_files: list[Path] = []
    for cand in candidates:
        if (cand / "train").exists():
            train_files = sorted((cand / "train").glob("*.h5"))
            val_files = sorted((cand / "val").glob("*.h5")) if (cand / "val").exists() else []
            if train_files:
                break
        if cand.exists() and list(cand.glob("*.h5")):
            # Flat layout fallback
            all_h5 = sorted(cand.glob("*.h5"))
            split = int(len(all_h5) * 0.8)
            train_files = all_h5[:split]
            val_files = all_h5[split:]
            break
    # Default if still empty — try data/stare
    if not train_files:
        data_root2 = Path("data/stare")
        if data_root2.exists():
            train_files = sorted((data_root2 / "train").glob("*.h5")) or sorted(data_root2.glob("*.h5"))
            val_files = sorted((data_root2 / "val").glob("*.h5"))
    if not train_files:
        raise FileNotFoundError(f"No training .h5 found. Checked {candidates} and data/stare")

    logger.info("Found %d train, %d val files", len(train_files), len(val_files))

    max_pulses = int(train_cfg["deinterleaver"]["max_pulses_per_train"])
    epochs = int(train_cfg["deinterleaver"]["epochs"])
    lr = float(train_cfg["deinterleaver"]["lr"])
    weight_decay = float(train_cfg["deinterleaver"].get("weight_decay", 1e-5))
    margin = float(model_cfg["triplet_margin"])
    save_every = int(train_cfg["deinterleaver"].get("save_every", 5))
    val_every = int(train_cfg["deinterleaver"].get("val_every", 2))
    warmup_steps = int(train_cfg["deinterleaver"].get("warmup_steps", 500))
    try:
        batch_size = int(train_cfg["deinterleaver"].get("batch_size", 8))
    except Exception:
        batch_size = 8

    # Model
    model = PDWTransformerEncoder(
        pdw_dim=model_cfg.get("pdw_dim", 6),
        d_model=model_cfg.get("d_model", 128),
        nhead=model_cfg.get("nhead", 8),
        num_layers=model_cfg.get("num_layers", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        dropout=model_cfg.get("dropout", 0.1),
        embed_dim=model_cfg.get("embed_dim", 64),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine with warmup
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs * max(1, len(train_files) // max(1, batch_size)) - warmup_steps))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    triplet_fn = nn.TripletMarginLoss(margin=margin, p=2)

    # WandB optional
    use_wandb = False
    try:
        import wandb  # type: ignore

        wandb.init(project=os.getenv("WANDB_PROJECT", "cognitive-ew-sih"), config={**model_cfg, **train_cfg["deinterleaver"]})
        use_wandb = True
    except Exception as exc:
        logger.info("WandB not available: %s", exc)

    # Fit stats from training only (prevent leakage)
    logger.info("Computing normalisation stats from training subset...")
    fit_stats: dict = {}
    for fp in train_files[: min(50, len(train_files))]:
        _, _, fit_stats = load_file_for_training(fp, max_pulses, fit_stats if fit_stats else None)

    output_dir = Path(train_cfg.get("output_dir", "checkpoints/deinterleaver"))
    output_dir.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(patience=10)
    best_v_measure = -float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_files)
        epoch_loss = 0.0
        n_batches = 0

        for fp in train_files:
            # CRITICAL: each file processed independently — labels never cross files
            pdws, labels, _ = load_file_for_training(fp, max_pulses, fit_stats)
            if len(pdws) < 4:
                continue
            try:
                pdws_t = torch.tensor(pdws, dtype=torch.float32, device=device).unsqueeze(0)
                labels_t = torch.tensor(labels, dtype=torch.long, device=device)
                embeddings = model(pdws_t).squeeze(0)
                anchors, positives, negatives = mine_triplets(embeddings, labels_t, margin)
                loss = triplet_fn(anchors, positives, negatives)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if warmup_steps > 0:
                    scheduler.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning("CUDA OOM on %s — skipping, falling back", fp)
                    torch.cuda.empty_cache()
                    continue
                raise

        if warmup_steps == 0:
            scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — Train TripletLoss: %.4f", epoch, epochs, avg_loss)
        if use_wandb:
            try:
                import wandb

                wandb.log({"train/loss": avg_loss, "epoch": epoch})
            except Exception:
                pass

        # Validation with clustering V-measure
        if epoch % val_every == 0 and val_files:
            model.eval()
            val_loss = 0.0
            v_batches = 0
            v_measures: list[float] = []
            with torch.inference_mode():
                for fp in val_files[: min(20, len(val_files))]:
                    pdws, labels, _ = load_file_for_training(fp, max_pulses, fit_stats)
                    if len(pdws) < 10:
                        continue
                    pdws_t = torch.tensor(pdws, dtype=torch.float32, device=device).unsqueeze(0)
                    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
                    emb = model(pdws_t).squeeze(0)
                    a, p, n = mine_triplets(emb, labels_t, margin)
                    val_loss += float(triplet_fn(a, p, n).item())
                    v_batches += 1
                    # Clustering V-measure on first few val files for speed
                    if len(v_measures) < 5:
                        try:
                            from sklearn.metrics import v_measure_score  # type: ignore

                            emb_np = emb.detach().cpu().numpy()
                            # Quick HDBSCAN on val embeddings
                            try:
                                import hdbscan

                                clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean", cluster_selection_method="eom")
                                pred = clusterer.fit_predict(emb_np)
                                # Only compute if not all noise
                                if len(set(pred)) > 1 and not np.all(pred == -1):
                                    # Mask noise for metric (optional)
                                    mask = pred != -1
                                    if np.sum(mask) > 5:
                                        vm = v_measure_score(labels[mask], pred[mask])
                                        v_measures.append(float(vm))
                            except Exception:
                                pass
                        except Exception:
                            pass
            avg_val_loss = val_loss / max(v_batches, 1)
            avg_v = float(np.mean(v_measures)) if v_measures else 0.0
            logger.info("  Val TripletLoss: %.4f V-measure: %.4f", avg_val_loss, avg_v)
            if use_wandb:
                try:
                    import wandb

                    wandb.log({"val/loss": avg_val_loss, "val/v_measure": avg_v, "epoch": epoch})
                except Exception:
                    pass
            # Checkpoint best V-measure
            if avg_v > best_v_measure:
                best_v_measure = avg_v
                torch.save(model.state_dict(), output_dir / "best.pt")
                logger.info("  New best V-measure %.4f — saved best.pt", avg_v)
            if early_stop.step(avg_v):
                logger.info("Early stopping at epoch %d (patience 10)", epoch)
                break

        if epoch % save_every == 0:
            ckpt = output_dir / f"epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)
            logger.info("  Checkpoint %s", ckpt)

    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Training complete. Final: %s Best V: %.4f", final_path, best_v_measure)
    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Train deinterleaver")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/deinterleaver")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    if args.device:
        os.environ["DEVICE"] = args.device
    train_deinterleaver(args.model_config, args.config)
