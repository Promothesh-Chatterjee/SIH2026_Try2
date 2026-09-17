#!/bin/bash
# Train both models sequentially
set -e
echo "=== Training deinterleaver ==="
bash scripts/train_deinterleaver.sh
echo "=== Training scheduler ==="
bash scripts/train_scheduler.sh
echo "=== All training complete ==="
