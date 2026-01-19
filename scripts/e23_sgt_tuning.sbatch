#!/bin/bash
#SBATCH --job-name=e23-sgt-tuning
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/slurm-e23-sgt-tuning-%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-107%10

# e23_sgt_tuning: SparseGT hyperparameter grid search on res04
# Grid: num_hops=[3,5,7] x num_layers=[2,3,4,5] x hidden_dim=[64,128,256] x dropout=[0.0,0.01,0.05]
# Total: 3 x 4 x 3 x 3 = 108 combinations
# Usage: sbatch scripts/e23_sgt_tuning.sbatch

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/vast/home/j/jporras/sourcecode/shortest-paths-nn}
OUTPUT_DIR=${OUTPUT_DIR:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
WANDB_TAG=${WANDB_TAG:-e23_sgt_tuning}
TEST_DATA=${TEST_DATA:-generated2/full_test10ksrc-004.npz}
ENV_NAME=${ENV_NAME:-shortest-paths-nn}

# Fixed training parameters
RES="04"
EPOCHS=10
BATCH_SIZE=512
LR=0.0001

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"
export TERRAIN_OUTPUT_DIR="$OUTPUT_DIR"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# -----------------------------------------------------------------------------
# Hyperparameter Grid
# -----------------------------------------------------------------------------
NUM_HOPS_VALUES=(3 5 7)
NUM_LAYERS_VALUES=(2 3 4 5)
HIDDEN_DIM_VALUES=(64 128 256)
DROPOUT_VALUES=(0.0 0.01 0.05)

# Grid dimensions
N_HOPS=${#NUM_HOPS_VALUES[@]}      # 3
N_LAYERS=${#NUM_LAYERS_VALUES[@]}  # 4
N_HIDDEN=${#HIDDEN_DIM_VALUES[@]}  # 3
N_DROPOUT=${#DROPOUT_VALUES[@]}    # 3

# Map TASK_ID to hyperparameter indices
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Decompose TASK_ID into indices: TASK_ID = i_hops + N_HOPS * (i_layers + N_LAYERS * (i_hidden + N_HIDDEN * i_dropout))
i_dropout=$((TASK_ID / (N_HOPS * N_LAYERS * N_HIDDEN)))
remainder=$((TASK_ID % (N_HOPS * N_LAYERS * N_HIDDEN)))
i_hidden=$((remainder / (N_HOPS * N_LAYERS)))
remainder=$((remainder % (N_HOPS * N_LAYERS)))
i_layers=$((remainder / N_HOPS))
i_hops=$((remainder % N_HOPS))

# Get hyperparameter values
NUM_HOPS=${NUM_HOPS_VALUES[$i_hops]}
NUM_LAYERS=${NUM_LAYERS_VALUES[$i_layers]}
HIDDEN_DIM=${HIDDEN_DIM_VALUES[$i_hidden]}
DROPOUT=${DROPOUT_VALUES[$i_dropout]}
ATTN_DROPOUT=${DROPOUT}  # Same as dropout

log "========================================"
log "=== e23 SparseGT Tuning: Task ${TASK_ID}/107 ==="
log "========================================"
log "  num_hops: ${NUM_HOPS}"
log "  num_layers: ${NUM_LAYERS}"
log "  hidden_dim: ${HIDDEN_DIM}"
log "  dropout: ${DROPOUT}"
log "  attn_dropout: ${ATTN_DROPOUT}"
log "  resolution: res${RES}"
log "  epochs: ${EPOCHS}"
log "  batch_size: ${BATCH_SIZE}"
log "  lr: ${LR}"

# -----------------------------------------------------------------------------
# Generate Config File
# -----------------------------------------------------------------------------
CONFIG_FILE=$(mktemp --suffix=.yml)
trap "rm -f $CONFIG_FILE" EXIT

cat > "$CONFIG_FILE" << EOF
# Auto-generated config for e23_sgt_tuning task ${TASK_ID}
sparse-gt-rpearl:
  gnn:
    constr:
      input: 3
      hidden: ${HIDDEN_DIM}
      output: 64
      layers: 3
    layer_norm: false
    dropout: true
    activation: lrelu

    sparse_gt:
      hidden_dim: ${HIDDEN_DIM}
      num_layers: ${NUM_LAYERS}
      num_heads: 4
      num_hops: ${NUM_HOPS}
      rpearl_samples: 30
      rpearl_num_layers: 5
      dropout: ${DROPOUT}
      attn_dropout: ${ATTN_DROPOUT}

  mlp:
    constr:
      input: 64
      hidden: 128
      output: 1
      layers: 3
    layer_norm: false
    dropout: true
EOF

log "Generated config file: $CONFIG_FILE"
log "Config contents:"
cat "$CONFIG_FILE"

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------
TRAIN_ARGS=(
    --train-data "generated2/res${RES}_phase1.npz"
    --test-data "$TEST_DATA"
    --epochs "$EPOCHS"
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "norway/res${RES}"
    --config "$CONFIG_FILE"
    --siamese 1
    --vn 0
    --layer-type SparseGT
    --aggr 'sum+diff'
    --p 4
    --loss mse_loss
    --finetune 0
    --include-edge-attr 1
    --lr "$LR"
    --trial "hops${NUM_HOPS}_layers${NUM_LAYERS}_dim${HIDDEN_DIM}_drop${DROPOUT}"
    --new
    --wandb-tag "$WANDB_TAG" "hops-${NUM_HOPS}" "layers-${NUM_LAYERS}" "dim-${HIDDEN_DIM}" "drop-${DROPOUT}"
)

log "Command: python train_single_terrain_case.py ${TRAIN_ARGS[*]}"
python train_single_terrain_case.py "${TRAIN_ARGS[@]}"

log "========================================"
log "=== Task ${TASK_ID} completed ==="
log "========================================"
