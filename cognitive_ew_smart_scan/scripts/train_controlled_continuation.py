"""Controlled continuation orchestrator for 25k -> 40k training.

Enforces:
1. Resumption from frozen Gate-25k checkpoint without touching frozen weights.
2. Saving to separate directory: cognitive_ew_smart_scan/checkpoints/scheduler_v2_continuation.
3. Checkpoints at canonical 2.5k step gates: 27500, 30000, 32500, 35000, 37500, 40000.
4. Shadow validation every 500 steps on config_117 (dense) and config_29 (fast hopper).
5. Early-stopping sentinels: max |Q| > 50.0, NaN/Inf signals, action entropy < 1.0.
6. Machine-verifiable promotion evaluation under lexicographic promotion rules.
"""

import copy
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("cognitive_ew_smart_scan").resolve()))

from cognitive_ew_smart_scan.src.training.train_scheduler import train_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("controlled_continuation")

def main():
    frozen_ckpt = Path("cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt")
    assert frozen_ckpt.exists(), f"Frozen candidate not found: {frozen_ckpt}"

    output_dir = "cognitive_ew_smart_scan/checkpoints/scheduler_v2_continuation"
    os.makedirs(output_dir, exist_ok=True)

    config_path = "cognitive_ew_smart_scan/configs/training_config.yaml"
    model_config_path = "cognitive_ew_smart_scan/configs/model_config.yaml"

    staged_gates = [27500, 30000, 32500, 35000, 37500, 40000]

    logger.info("Starting controlled continuation from %s to step 40,000", frozen_ckpt)
    logger.info("Gates scheduled: %s", staged_gates)
    logger.info("Output directory: %s", output_dir)

    train_scheduler(
        model_cfg_path=model_config_path,
        train_cfg_path=config_path,
        output_dir_override=output_dir,
        resume_checkpoint=str(frozen_ckpt),
        stop_at_step=40000,
        staged_gates=staged_gates,
        exploration_schedule="c_rescue",
        target_q_max=50.0,
        target_q_min=-50.0,
        q_reg_coef=1e-4,
        targeted_exploration=True,
        band_discovery_quota=2,
    )
    logger.info("Controlled continuation completed successfully.")

if __name__ == "__main__":
    main()
