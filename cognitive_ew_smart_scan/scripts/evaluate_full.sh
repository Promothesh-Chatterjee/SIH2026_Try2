#!/bin/bash
# Full evaluation over test set (scan mode)
set -e
python -m src.evaluation.evaluate_full \
  --deinterleaver-ckpt checkpoints/deinterleaver/best.pt \
  --scheduler-ckpt checkpoints/scheduler/best.pt \
  --config configs/model_config.yaml \
  --test-dir data/test/ \
  --output-dir results/ \
  --mode scan
