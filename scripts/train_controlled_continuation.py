"""Controlled continuation orchestrator for Gate-25k -> Gate-100k retraining.

Authoritative entrypoint enforcing:
1. Resumption from frozen Gate-25k checkpoint (SHA-256: 7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0).
2. Pure WEIGHTS-ONLY initialization: fresh optimizer, fresh replay buffer, fresh RNG, fresh exploration schedule.
3. Saving to isolated directory: experiments/checkpoints/scheduler_v2_continuation_100k.
4. Intermediate evaluation gates at 50,000, 75,000, and 100,000 steps.
5. Q-value stability clamping [target_q_min=-50.0, target_q_max=100.0] and Q^2 regularization.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("ew_core").resolve()))

from ew_core.training.train_scheduler import train_scheduler
from ew_core.utils.checkpoint_meta import CheckpointMode, validate_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_controlled_continuation")

CANONICAL_FROZEN_SHA = "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0"
DEFAULT_BASELINE = "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt"


def verify_baseline_checkpoint(ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {ckpt_path}")

    sha256 = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()

    if digest != CANONICAL_FROZEN_SHA:
        raise RuntimeError(
            f"FATAL: Baseline checkpoint SHA-256 mismatch!\n"
            f"Expected: {CANONICAL_FROZEN_SHA}\n"
            f"Actual:   {digest}\n"
            f"Path:     {ckpt_path}"
        )
    logger.info("Baseline checkpoint SHA-256 verified: %s", digest)

    is_valid, errors = validate_checkpoint(ckpt_path, mode=CheckpointMode.WEIGHTS_ONLY)
    if not is_valid:
        raise RuntimeError(f"FATAL: Baseline checkpoint validation failed: {errors}")
    logger.info("Baseline checkpoint validated successfully under WEIGHTS_ONLY contract.")


def main():
    parser = argparse.ArgumentParser(description="Controlled continuation training from Gate-25k to Gate-100k.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config_resume_100k.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default=DEFAULT_BASELINE,
        help="Path to frozen Gate-25k checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/checkpoints/scheduler_v2_continuation_100k",
        help="Directory to save continuation checkpoints.",
    )
    parser.add_argument(
        "--stop-step",
        type=int,
        default=100000,
        help="Step number to conclude training run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute short dry-run (e.g. stop after 100 steps past start_step) to verify pipeline.",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=50,
        help="Number of steps to run when --dry-run is set.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_ckpt)
    verify_baseline_checkpoint(baseline_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    staged_gates = [50000, 75000, 100000]
    stop_at_step = args.stop_step

    if args.dry_run:
        stop_at_step = 25000 + args.dry_run_steps
        staged_gates = [25000 + args.dry_run_steps]
        logger.info("DRY RUN ACTIVE: Training will execute from 25,000 to %d steps.", stop_at_step)
    else:
        logger.info("Starting controlled continuation from %s to step %d", baseline_path, stop_at_step)
        logger.info("Gates scheduled: %s", staged_gates)

    train_scheduler(
        model_cfg_path=args.model_config,
        train_cfg_path=args.config,
        output_dir_override=str(output_dir),
        resume_checkpoint=str(baseline_path),
        stop_at_step=stop_at_step,
        staged_gates=staged_gates,
        exploration_schedule="slower",
        target_q_max=100.0,
        target_q_min=-50.0,
        q_reg_coef=1e-4,
        targeted_exploration=True,
        band_discovery_quota=2,
    )
    logger.info("Controlled continuation concluded successfully.")


if __name__ == "__main__":
    main()
