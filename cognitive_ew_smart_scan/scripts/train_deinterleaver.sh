#!/bin/bash
# Full training run for deinterleaver
set -e
python -m src.training.train_deinterleaver \
  --config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --data-dir data/ \
  --output-dir checkpoints/deinterleaver/ \
  --device cuda
