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


# Phase 0 Canonical Production Baseline and Candidate Mirror Contracts
CANONICAL_PRODUCTION_BASELINE = Path("experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt")
REFERENCE_CANDIDATE_MIRROR = Path("experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt")
EXPECTED_FROZEN_SHA256 = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"


def verify_production_baseline_checkpoint(path: Path | str | None = None) -> Path:
    """Verify that the production baseline checkpoint exists and matches canonical SHA-256."""
    import hashlib

    p = Path(path) if path is not None else CANONICAL_PRODUCTION_BASELINE
    if not p.is_file():
        raise FileNotFoundError(f"Production baseline checkpoint not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != EXPECTED_FROZEN_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {p}:\n"
            f"  Expected: {EXPECTED_FROZEN_SHA256}\n"
            f"  Actual:   {actual_sha}"
        )
    return p


verify_production_checkpoint = verify_production_baseline_checkpoint