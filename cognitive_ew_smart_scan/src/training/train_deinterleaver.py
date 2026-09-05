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
from ..data.tsrd_manifest import build_manifest, dataset_fingerprint, resolve_split_dirs, validate_dataset, generate_dataset_report
from ..data.synthetic_dataset import ensure_local_fallback_dataset

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


def load_file_windows(
    file_path: Path,
    window_size: int = 1024,
    stride: int = 512,
    fit_stats: dict | None = None,
    max_windows_per_file: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load a single H5 pulse train, sort by ToA, slice into contiguous temporal windows.

    Preserves temporal PRI sequence structure (P0-3) without random sampling.
    """
    assert file_path.suffix in {".h5", ".hdf5"}, f"Expected .h5 file, got {file_path}"
    try:
        if PulseTrain is not None:
            try:
                pt = PulseTrain.load(str(file_path))
                raw = pt.data
                labels = pt.labels
            except Exception:
                raw = None
                labels = None
        else:
            raw = None
            labels = None

        if raw is None or labels is None:
            import h5py
            with h5py.File(str(file_path), "r") as handle:
                if "data" not in handle or "labels" not in handle:
                    return []
                raw = np.asarray(handle["data"])
                labels = np.asarray(handle["labels"]).reshape(-1)
    except Exception as exc:
        logger.warning("Corrupt H5 %s: %s", file_path, exc)
        return []

    if raw is None or len(raw) == 0:
        return []

    # Sort strictly by ToA to preserve temporal sequence and PRI causality
    sort_idx = np.argsort(raw[:, 0])
    raw = raw[sort_idx]
    labels = labels[sort_idx]

    # If shorter than window_size, return full sequence as a single window
    if len(raw) <= window_size:
        pdws_norm, _ = normalise_pdws(raw, fit_stats)
        return [(pdws_norm, labels.astype(np.int64))]

    # Generate contiguous windows with configurable stride
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    starts = list(range(0, len(raw) - window_size + 1, stride))
    if len(starts) > max_windows_per_file:
        indices = np.linspace(0, len(starts) - 1, max_windows_per_file, dtype=int)
        starts = [starts[i] for i in indices]

    for s in starts:
        w_raw = raw[s : s + window_size]
        w_labels = labels[s : s + window_size]
        pdws_norm, _ = normalise_pdws(w_raw, fit_stats)
        windows.append((pdws_norm, w_labels.astype(np.int64)))

    return windows


def load_file_for_training(
    file_path: Path,
    max_pulses: int,
    fit_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Backward-compatible loader returning first contiguous temporal slice."""
    windows = load_file_windows(file_path, window_size=max_pulses, stride=max_pulses, fit_stats=fit_stats, max_windows_per_file=1)
    if not windows:
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int64), fit_stats or {}
    pdws_norm, labels = windows[0]
    return pdws_norm, labels, fit_stats or {}


def mine_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Batch-hard triplet mining within a single file/window.

    CRITICAL: embeddings and labels must be from ONE file/window. Never mix across files.
    """
    assert embeddings.size(0) == labels.size(0), "Embeddings/labels length mismatch — possible cross-file mixing"
    if not torch.isfinite(embeddings).all():
        return None, None, None
    n = embeddings.size(0)
    if n < 4:
        return None, None, None

    # Skip if only 1 unique label exists in the window
    unique_labels, counts = torch.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        return None, None, None

    # Classes with >= 2 pulses can act as anchors
    valid_anchor_labels = set(unique_labels[counts >= 2].tolist())
    if not valid_anchor_labels:
        return None, None, None

    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    anchors, positives, negatives = [], [], []

    for i in range(n):
        label_i = int(labels[i].item())
        if label_i not in valid_anchor_labels:
            continue
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

        # Include semi-hard / hard violations
        if dist_matrix[i, hardest_pos] + margin > dist_matrix[i, hardest_neg]:
            anchors.append(embeddings[i])
            positives.append(embeddings[hardest_pos])
            negatives.append(embeddings[hardest_neg])

    if not anchors:
        return None, None, None
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def stitch_and_evaluate(
    model: nn.Module,
    file_path: Path,
    fit_stats: dict | None,
    window_size: int = 1024,
    stride: int = 512,
    device: str = "cpu",
    max_eval_pulses: int = 10000,
) -> dict[str, float]:
    """Evaluate deinterleaver on a full pulse train by stitching window embeddings (P0-3)."""
    import h5py
    from sklearn.metrics import (
        v_measure_score,
        adjusted_rand_score,
        adjusted_mutual_info_score,
        homogeneity_score,
        completeness_score,
        matthews_corrcoef,
        f1_score,
    )

    with h5py.File(str(file_path), "r") as handle:
        raw = np.asarray(handle["data"])
        labels = np.asarray(handle["labels"]).reshape(-1)

    if len(raw) == 0:
        return {}

    sort_idx = np.argsort(raw[:, 0])
    raw = raw[sort_idx][:max_eval_pulses]
    labels = labels[sort_idx][:max_eval_pulses]

    N = len(raw)
    embed_dim = getattr(model, "embed_dim", 64)
    stitched_embeddings = np.zeros((N, embed_dim), dtype=np.float32)
    counts = np.zeros(N, dtype=np.float32)

    model.eval()
    starts = list(range(0, max(1, N - window_size + 1), stride))
    if starts and starts[-1] + window_size < N:
        starts.append(N - window_size)

    with torch.no_grad():
        for s in starts:
            e = min(s + window_size, N)
            w_raw = raw[s:e]
            pdws_norm, _ = normalise_pdws(w_raw, fit_stats)
            inp = torch.from_numpy(pdws_norm).float().unsqueeze(0).to(device)
            emb = model(inp).squeeze(0).cpu().numpy()
            stitched_embeddings[s:e] += emb[: e - s]
            counts[s:e] += 1.0

    counts = np.maximum(counts, 1.0)[:, None]
    stitched_embeddings /= counts
    norms = np.linalg.norm(stitched_embeddings, axis=-1, keepdims=True)
    stitched_embeddings /= np.maximum(norms, 1e-8)

    # Cluster with HDBSCAN
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean", cluster_selection_method="eom")
        pred_labels = clusterer.fit_predict(stitched_embeddings)
    except Exception:
        pred_labels = np.full(N, -1, dtype=np.int32)

    # Metrics
    vm = float(v_measure_score(labels, pred_labels))
    ari = float(adjusted_rand_score(labels, pred_labels))
    ami = float(adjusted_mutual_info_score(labels, pred_labels))
    homo = float(homogeneity_score(labels, pred_labels))
    comp = float(completeness_score(labels, pred_labels))

    valid_mask = pred_labels != -1
    if np.any(valid_mask):
        mcc = float(matthews_corrcoef(labels[valid_mask], pred_labels[valid_mask]))
        f1 = float(f1_score(labels[valid_mask], pred_labels[valid_mask], average="macro"))
    else:
        mcc = 0.0
        f1 = 0.0

    n_pred = len(set(pred_labels) - {-1})
    return {
        "v_measure": vm,
        "ari": ari,
        "ami": ami,
        "homogeneity": homo,
        "completeness": comp,
        "mcc": mcc,
        "f1": f1,
        "n_clusters_predicted": n_pred,
        "n_emitters_true": len(np.unique(labels)),
        "num_pulses": N,
    }


def collate_single_file_batch(
    files: list[Path], max_pulses: int, fit_stats: dict | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collate helper that keeps each pulse train separate (no mixing)."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fp in files:
        pdws, labels, _ = load_file_for_training(fp, max_pulses, fit_stats)
        if len(pdws) > 0:
            out.append((pdws, labels))
    return out


def train_deinterleaver(
    model_cfg_path: str,
    train_cfg_path: str,
    max_files: int | None = None,
    epochs_override: int | None = None,
    quick_smoke: bool = False,
    data_dir_override: str | None = None,
    output_dir_override: str | None = None,
    training_mode_override: str | None = None,
) -> None:
    """Full triplet training loop with checkpointing and early stopping.

    Args:
        model_cfg_path: Path to model_config.yaml.
        train_cfg_path: Path to training_config.yaml.
        max_files: Optional cap on train files.
        epochs_override: Optional override for epochs count.
        quick_smoke: If True, runs 1 epoch on 2 files for fast integration testing.
        data_dir_override: CLI override for the dataset root (CLI > YAML > default).
        output_dir_override: CLI override for the checkpoint output directory
            (CLI > YAML > default).

    Raises:
        FileNotFoundError: If no training .h5 files are found.
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

    # Data discovery (CLI > env TSRD_DATA_ROOT > YAML data_dir > safe default).
    from ..data.tsrd_root import resolve_tsrd_root

    data_root = resolve_tsrd_root(cli_value=data_dir_override, config=train_cfg)

    # Check training mode - in real_tsrd mode, synthetic fallback is FORBIDDEN.
    training_mode = (
        training_mode_override
        or os.environ.get("TRAINING_MODE")
        or train_cfg.get("training_mode", "real_tsrd")
    )
    if training_mode not in {"real_tsrd", "synthetic"}:
        raise ValueError(f"Unsupported training_mode={training_mode!r}; expected real_tsrd or synthetic")
    allow_synthetic_fallback = training_mode != "real_tsrd"
    validation = validate_dataset(data_root) if training_mode == "real_tsrd" else {"valid": True, "errors": []}
    
    if training_mode == "real_tsrd":
        logger.info("Training mode: REAL TSRD - synthetic fallback DISABLED")
        # In real_tsrd mode, the specified data_root MUST exist and contain valid TSRD data
        if not data_root.exists():
            raise FileNotFoundError(f"TSRD data root does not exist: {data_root}")
        # Validate that the data_root actually has TSRD files
        if not validation["valid"]:
            raise FileNotFoundError(
                f"TSRD data root {data_root} does not contain valid TSRD data: {validation['errors']}"
            )
    else:
        logger.info("Training mode: SYNTHETIC - synthetic fallback ENABLED")
        if not validation["valid"]:
            logger.warning("Dataset validation reported issues: %s", validation["errors"])

    # Use SCAN mode for deinterleaver training (realistic observed data)
    train_mode = train_cfg.get("deinterleaver_mode", "scan")
    val_mode = train_cfg.get("deinterleaver_val_mode", "scan")

    candidate_roots = [data_root]
    train_files: list[Path] = []
    val_files: list[Path] = []
    if data_root.exists():
        split_dirs = resolve_split_dirs(data_root, train_mode)
        train_candidates = sorted(split_dirs["train"].glob("*.h5")) if split_dirs["train"].exists() else []
        val_split_dirs = resolve_split_dirs(data_root, val_mode)
        val_candidates = sorted(val_split_dirs["val"].glob("*.h5")) if val_split_dirs["val"].exists() else []
        if train_candidates:
            train_files = train_candidates
            val_files = val_candidates or train_candidates[: max(1, len(train_candidates) // 5)]
            logger.info("Using split discovery for %s/%s: %d train, %d val", data_root, train_mode, len(train_files), len(val_files))

    if not train_files:
        if allow_synthetic_fallback:
            logger.warning("No dataset found in local roots. Creating synthetic fallback dataset for safe local training.")
            ensure_local_fallback_dataset(data_root)
            split_dirs = resolve_split_dirs(data_root, train_mode)
            train_files = sorted(split_dirs["train"].glob("*.h5")) if split_dirs["train"].exists() else []
            val_files = sorted(split_dirs["val"].glob("*.h5")) if split_dirs["val"].exists() else []
        else:
            raise FileNotFoundError(
                f"No TSRD .h5 files found in real_tsrd mode. "
                f"Checked dataset roots: {candidate_roots}. "
                f"Set training_mode: synthetic in config to use synthetic data."
            )

    if not train_files:
        raise FileNotFoundError(f"No training .h5 found. Checked dataset roots: {candidate_roots}")

    if quick_smoke:
        max_files = 2
        epochs_override = 1

    if max_files is not None and max_files > 0:
        train_files = train_files[:max_files]
        val_files = val_files[: max(1, max_files // 5)]

    # Generate dataset report before training
    logger.info("Generating dataset report...")
    report = generate_dataset_report(train_files, val_files, train_mode)
    logger.info("Dataset report: %d train files, %d val files, %d train pulses, %d val pulses",
                report["train_files"], report["val_files"], report["train_pulses"], report["val_pulses"])

    logger.info("Found %d train, %d val files for training", len(train_files), len(val_files))

    max_pulses = 128 if quick_smoke else int(train_cfg["deinterleaver"]["max_pulses_per_train"])
    epochs = epochs_override if epochs_override is not None else int(train_cfg["deinterleaver"]["epochs"])
    lr = float(train_cfg["deinterleaver"]["lr"])
    weight_decay = float(train_cfg["deinterleaver"].get("weight_decay", 1e-5))
    margin = float(model_cfg["triplet_margin"])
    save_every = int(train_cfg["deinterleaver"].get("save_every", 5))
    val_every = 1 if quick_smoke else int(train_cfg["deinterleaver"].get("val_every", 2))
    warmup_steps = 0 if quick_smoke else int(train_cfg["deinterleaver"].get("warmup_steps", 500))

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
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(train_files) - warmup_steps))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    triplet_fn = nn.TripletMarginLoss(margin=margin, p=2)

    # Phase 17: canonical layout — never resolve to the ambiguous root
    # (config output_dir="checkpoints" is replaced by the canonical subdir).
    from ..utils.checkpoint_paths import DEINTERLEAVER_DIR, resolve_checkpoint_dir

    output_dir = resolve_checkpoint_dir(
        output_dir_override,
        train_cfg.get("output_dir"),
        DEINTERLEAVER_DIR,
        role="deinterleaver",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(data_root, output_path=output_dir / "dataset_manifest.json", mode=train_mode)
    data_fingerprint = dataset_fingerprint(train_files + val_files, data_root, train_mode)
    logger.info("Effective dataset root: %s; fingerprint: %s", data_root.resolve(), data_fingerprint)

    # Fit stats from training only (prevent leakage, P0-4)
    from ..preprocessing.normalise import fit_train_statistics, normalization_stats_hash, save_normalization_stats
    logger.info("Computing normalisation stats from training subset...")
    fit_stats = fit_train_statistics(train_files[: min(50, len(train_files))])
    save_normalization_stats(fit_stats, output_dir / "normalization_stats.json")

    early_stop = EarlyStopping(patience=10)
    best_v_measure = -float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_files)
        epoch_loss = 0.0
        n_batches = 0

        for fp in train_files:
            # Generate contiguous temporal windows (P0-3)
            windows = load_file_windows(
                fp,
                window_size=max_pulses,
                stride=max(1, max_pulses // 2),
                fit_stats=fit_stats,
                max_windows_per_file=2 if quick_smoke else 4,
            )
            for pdws, labels in windows:
                if len(pdws) < 4:
                    continue
                try:
                    pdws_t = torch.tensor(pdws, dtype=torch.float32, device=device).unsqueeze(0)
                    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
                    embeddings = model(pdws_t).squeeze(0)
                    anchors, positives, negatives = mine_triplets(embeddings, labels_t, margin)
                    if anchors is None or positives is None or negatives is None:
                        continue
                    if not torch.isfinite(anchors).all() or not torch.isfinite(positives).all() or not torch.isfinite(negatives).all():
                        continue
                    loss = triplet_fn(anchors, positives, negatives)
                    if not torch.isfinite(loss):
                        continue
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
                        logger.warning("CUDA OOM on %s — clearing cache", fp)
                        torch.cuda.empty_cache()
                        continue
                    raise

        if warmup_steps == 0:
            scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — Train TripletLoss: %.4f (%d windows)", epoch, epochs, avg_loss, n_batches)

        # Validation with full-train window stitching (P0-3)
        if epoch % val_every == 0 and val_files:
            model.eval()
            v_measures: list[float] = []
            val_subset = val_files[: min(5, len(val_files))]
            for fp in val_subset:
                try:
                    eval_res = stitch_and_evaluate(
                        model,
                        fp,
                        fit_stats=fit_stats,
                        window_size=max_pulses,
                        stride=max(1, max_pulses // 2),
                        device=str(device),
                        max_eval_pulses=5000 if not quick_smoke else 500,
                    )
                    if eval_res and "v_measure" in eval_res:
                        v_measures.append(eval_res["v_measure"])
                except Exception as exc:
                    logger.warning("Val evaluation failed on %s: %s", fp, exc)

            avg_v = float(np.mean(v_measures)) if v_measures else 0.0
            logger.info("  Val Stitched V-measure: %.4f (across %d files)", avg_v, len(v_measures))

            if avg_v > best_v_measure:
                best_v_measure = avg_v
                from ..utils.checkpoint_meta import build_train_metadata, save_state

                meta = build_train_metadata(
                    split="train",
                    n_bands=int(model_cfg.get("n_bands", 36)),
                    arch="PDWTransformerEncoder",
                    seed=seed,
                    metrics={"best_val_v_measure": float(avg_v)},
                    extra={
                        "mode": "deinterleaver",
                        "preproc_version": "v1",
                        "dataset_fingerprint": data_fingerprint,
                        "dataset_manifest": str(output_dir / "dataset_manifest.json"),
                        "normalization_stats_hash": normalization_stats_hash(fit_stats),
                        "normalization_stats_path": str(output_dir / "normalization_stats.json"),
                    },
                )
                save_state(model, output_dir / "best.pt", meta)
                logger.info("  New best V-measure %.4f — saved best.pt", avg_v)
            if early_stop.step(avg_v):
                logger.info("Early stopping at epoch %d", epoch)
                break

        if epoch % save_every == 0 or quick_smoke:
            ckpt = output_dir / f"epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)

    final_path = output_dir / "final.pt"
    from ..utils.checkpoint_meta import build_train_metadata, save_state

    final_meta = build_train_metadata(
        split="train",
        n_bands=int(model_cfg.get("n_bands", 36)),
        arch="PDWTransformerEncoder",
        seed=seed,
        metrics={"best_val_v_measure": float(best_v_measure)},
        extra={
            "mode": "deinterleaver",
            "dataset_fingerprint": data_fingerprint,
            "normalization_stats_hash": normalization_stats_hash(fit_stats),
            "normalization_stats_path": str(output_dir / "normalization_stats.json"),
        },
    )
    save_state(model, final_path, final_meta)
    # Phase 17: human-readable metadata.json sidecar (contract artifact).
    from ..utils.checkpoint_meta import write_checkpoint_metadata

    write_checkpoint_metadata(
        output_dir / "metadata.json",
        final_meta,
        artifacts=[
            "best.pt",
            "final.pt",
            "normalization_stats.json",
            "dataset_manifest.json",
        ],
    )
    logger.info("Training complete. Final: %s Best V: %.4f", final_path, best_v_measure)


def train_deinterleaver_safe() -> dict:
    """Safety wrapper used for local testing and CI."""
    seed = 42
    ensure_local_fallback_dataset(data_root="data", seed=seed)
    train_deinterleaver(
        "configs/model_config.yaml",
        "configs/training_config.yaml",
        max_files=2,
        epochs_override=1,
        quick_smoke=True,
        training_mode_override="synthetic",
    )
    return {"status": "ok", "dataset": "synthetic-fallback"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Train deinterleaver")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data-dir", type=str, default=None,
        help="Override dataset root (CLI > YAML data_dir).")
    parser.add_argument("--output-dir", type=str, default=None,
        help="Override checkpoint output dir (CLI > YAML output_dir).")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick-smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--training-mode", type=str, choices=["real_tsrd", "synthetic"], default=None,
        help="Training mode: real_tsrd (no synthetic fallback) or synthetic (allow fallback).")
    args = parser.parse_args()
    if args.device:
        os.environ["DEVICE"] = args.device
    if args.training_mode:
        os.environ["TRAINING_MODE"] = args.training_mode
    train_deinterleaver(
        args.model_config,
        args.config,
        max_files=args.max_files,
        epochs_override=args.epochs,
        quick_smoke=args.quick_smoke,
        data_dir_override=args.data_dir,
        output_dir_override=args.output_dir,
        training_mode_override=args.training_mode,
    )
