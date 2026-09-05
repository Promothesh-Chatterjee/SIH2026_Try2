"""Quick scheduler smoke test with reduced timesteps."""
import sys
sys.path.insert(0, "C:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/cognitive_ew_smart_scan")

import yaml
import torch
import numpy as np

from src.training.train_scheduler import train_scheduler

if __name__ == "__main__":
    # Override total_timesteps for quick smoke test
    import os
    os.environ["TOTAL_TIMESTEPS_OVERRIDE"] = "1000"

    from src.data.tsrd_root import resolve_config_data_dir

    train_scheduler(
        model_cfg_path="configs/model_config.yaml",
        train_cfg_path="configs/training_config.yaml",
        data_dir_override=str(resolve_config_data_dir("configs/training_config.yaml")),
        output_dir_override="checkpoints/scheduler_smoke",
    )