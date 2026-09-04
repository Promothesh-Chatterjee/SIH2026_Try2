#!/bin/bash
# Full DRQN scheduler training with Thompson warmup and MoE evaluation
set -e
python -m src.training.train_scheduler \
  --config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --output-dir checkpoints/scheduler/ \
  --device cuda
