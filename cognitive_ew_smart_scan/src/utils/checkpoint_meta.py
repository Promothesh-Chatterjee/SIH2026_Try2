"""Checkpoint provenance metadata (canonical best.pt sidecar).

Adds git revision, dataset split, preproc version, feature order, model arch,
seed, training metrics and timestamp to saved model checkpoints so results are
reproducible and non-fabricated (P0-9). Metadata is stored alongside the state
dict as ``{"state_dict": ..., "metadata": {...}}`` — backward-compatible because
loaders already unwrap a nested ``state_dict``.
"""

from __future__ import annotations

import datetime
import os
import subprocess
from typing import Any

import torch

from src.contracts import FEATURE_ORDER as _FEATURE_ORDER

# Backward-compatible list export for checkpoint metadata callers.
FEATURE_ORDER = list(_FEATURE_ORDER)
PREPROC_VERSION = "v1"  # bump whenever normalise.py feature layout changes


def current_git_revision() -> str:
    """Return the short HEAD git revision, or "unknown" if not a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_train_metadata(
    *,
    split: str,
    n_bands: int,
    feature_order: list[str] | None = None,
    arch: str,
    seed: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a canonical checkpoint metadata blob.

    Args:
        split: Dataset split used for training (e.g. ``train``).
        n_bands: Number of frequency bands.
        feature_order: Per-band feature names (defaults to FEATURE_ORDER).
        arch: Model architecture identifier (e.g. ``PDWTransformerEncoder``).
        seed: Random seed used during training.
        metrics: Dict of measured training/validation metrics (e.g. best V-measure).
        extra: Any additional provenance keys (mode, data_root, etc.).

    Returns:
        Metadata dict.
    """
    order = list(feature_order) if feature_order else list(FEATURE_ORDER)
    meta: dict[str, Any] = {
        "git_revision": current_git_revision(),
        "split": split,
        "n_bands": int(n_bands),
        "preproc_version": PREPROC_VERSION,
        "feature_order_per_band": order,
        "arch": arch,
        "seed": int(seed),
        "metrics": dict(metrics),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


def save_state(model: torch.nn.Module, path: os.PathLike | str, metadata: dict[str, Any]) -> None:
    """Save a model state dict together with canonical metadata.

    Args:
        model: Module whose ``state_dict`` will be saved.
        path: Output checkpoint path (``*.pt``).
        metadata: Provenance metadata blob.
    """
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)