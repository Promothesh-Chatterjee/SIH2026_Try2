"""Canonical checkpoint artifact layout (Phase 17 contract).

Every producer (trainers, eval, export, API) agrees on one structure::

    checkpoints/
        deinterleaver/
            best.pt
            final.pt
            normalization_stats.json
            dataset_manifest.json
            metadata.json
        scheduler/
            best.pt
            final.pt
            metadata.json
        onnx/
            deinterleaver.onnx
            scheduler.onnx

The training config must NEVER point ``output_dir`` at the ambiguous root
(``checkpoints``) — a trainer must resolve to its canonical subdirectory so no
``checkpoints/best.pt``-style collisions are ever created.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_ROOT = Path("checkpoints")
DEINTERLEAVER_DIR = CHECKPOINT_ROOT / "deinterleaver"
SCHEDULER_DIR = CHECKPOINT_ROOT / "scheduler"
ONNX_DIR = CHECKPOINT_ROOT / "onnx"

# Canonical artifact file names per directory (source of truth for the contract).
DEINTERLEAVER_ARTIFACTS = (
    "best.pt",
    "final.pt",
    "normalization_stats.json",
    "dataset_manifest.json",
    "metadata.json",
)
SCHEDULER_ARTIFACTS = ("best.pt", "final.pt", "metadata.json")
ONNX_ARTIFACTS = ("deinterleaver.onnx", "scheduler.onnx")

# Directory basenames that are ambiguous artifact roots: writing model files
# directly into them would collide with / shadow the canonical subdirectories.
AMBIGUOUS_ARTIFACT_ROOTS = {
    "checkpoints",
    "weights",
    "models",
    "model",
    "output",
    "outputs",
    "results",
    "runs",
}


def resolve_checkpoint_dir(
    cli_override: str | Path | None,
    config_output_dir: str | Path | None,
    canonical_dir: str | Path,
    role: str = "model",
) -> Path:
    """Resolve the directory a trainer writes its artifacts into.

    Priority: CLI ``--output-dir`` > training-config ``output_dir`` > the
    canonical subdirectory. A config value whose basename is an ambiguous
    artifact root (e.g. ``checkpoints``) is IGNORED with a warning and replaced
    by the canonical subdirectory, so training can never litter
    ``checkpoints/best.pt`` alongside the canonical ``checkpoints/<sub>/*``.

    Args:
        cli_override: Explicit ``--output-dir`` (highest precedence).
        config_output_dir: ``output_dir`` from training_config.yaml (bad if root).
        canonical_dir: Canonical subdirectory (e.g. ``checkpoints/scheduler``).
        role: Human-readable role for the warning message.

    Returns:
        Path to use for this model's artifacts.
    """
    if cli_override:
        return Path(cli_override)
    if not config_output_dir:
        return Path(canonical_dir)
    p = Path(config_output_dir)
    if p.name in AMBIGUOUS_ARTIFACT_ROOTS or str(p) in (".", str(Path("."))):
        logger.warning(
            "training_config output_dir=%r is an ambiguous artifact root — "
            "using canonical %s/ for %s instead (no root-level checkpoints).",
            str(config_output_dir),
            canonical_dir,
            role,
        )
        return Path(canonical_dir)
    return p