#!/usr/bin/env bash
# Run 4 SparseGT sweeps split across 2 GPUs, 50 runs each.
#
# Usage:
#   In tmux pane 1: bash scripts/run_all_sweeps.sh gpu0
#   In tmux pane 2: bash scripts/run_all_sweeps.sh gpu1
set -euo pipefail

SWEEP_SGT_RANDOM_PHILLY="d1h2w85g"
SWEEP_SGT_DATA_PHILLY="appuxvo1"
SWEEP_SGT_DATA_NORWAY="ikt0h80d"
SWEEP_SGT_RANDOM_NORWAY="ia3ydqf8"

ENTITY="alelab"
PROJECT="manifold-transformers-dev"
MAX_RUNS=50

cd /home/teresa/shortest-paths-nn

if [[ "${1:-}" == "gpu0" ]]; then
  export CUDA_VISIBLE_DEVICES=0
  echo "=== GPU 0: SparseGT-Random Philly, SparseGT-Data Norway ==="

  echo "--- SparseGT-Random Philly ($SWEEP_SGT_RANDOM_PHILLY) ---"
  wandb agent --count $MAX_RUNS "$ENTITY/$PROJECT/$SWEEP_SGT_RANDOM_PHILLY"

  echo "--- SparseGT-Data Norway ($SWEEP_SGT_DATA_NORWAY) ---"
  wandb agent --count $MAX_RUNS "$ENTITY/$PROJECT/$SWEEP_SGT_DATA_NORWAY"

elif [[ "${1:-}" == "gpu1" ]]; then
  export CUDA_VISIBLE_DEVICES=1
  echo "=== GPU 1: SparseGT-Data Philly, SparseGT-Random Norway ==="

  echo "--- SparseGT-Data Philly ($SWEEP_SGT_DATA_PHILLY) ---"
  wandb agent --count $MAX_RUNS "$ENTITY/$PROJECT/$SWEEP_SGT_DATA_PHILLY"

  echo "--- SparseGT-Random Norway ($SWEEP_SGT_RANDOM_NORWAY) ---"
  wandb agent --count $MAX_RUNS "$ENTITY/$PROJECT/$SWEEP_SGT_RANDOM_NORWAY"

else
  echo "Usage: bash scripts/run_all_sweeps.sh <gpu0|gpu1>"
  exit 1
fi
