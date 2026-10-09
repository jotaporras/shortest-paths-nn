#!/bin/bash
# Phase 2 MLP finetuning wrapper for Norway terrain graphs
# Usage: ./run_phase2_gnn.sh <resolution> [trial]
# Example: ./run_phase2_gnn.sh 01 1

set -e

RES=$1
TRIAL=${2:-1}

if [ -z "$RES" ]; then
    echo "Usage: ./run_phase2_gnn.sh <resolution> [trial]"
    echo "Example: ./run_phase2_gnn.sh 01 1"
    exit 1
fi

# Batch size: 32 for res01-05, 256 for res06-20
RES_NUM=$((10#$RES))
if [ $RES_NUM -le 5 ]; then
    BATCH_SIZE=32
else
    BATCH_SIZE=256
fi

# Compute Phase 1 model path (must match format_log_dir output structure)
PHASE1_MODEL="models/single_dataset/norway/res${RES}/TAGConv/no-vn/siamese/p-4/mse_loss/w-64-hw-128-dropout/${TRIAL}/new"

if [ ! -f "${PHASE1_MODEL}/final_model.pt" ]; then
    echo "ERROR: Phase 1 model not found at ${PHASE1_MODEL}/final_model.pt"
    echo "Run Phase 1 first: ./run_phase1_gnn.sh $RES $TRIAL"
    exit 1
fi

echo "=== Phase 2: Finetuning MLP on res${RES}_phase2.npz (batch_size=$BATCH_SIZE) ==="
echo "Loading GNN from: ${PHASE1_MODEL}"

python train-de-coupled.py \
    --finetune-from $PHASE1_MODEL \
    --train-data generated/res${RES}_phase2.npz \
    --epochs 1000 \
    --device cuda \
    --batch-size $BATCH_SIZE \
    --dataset-name res${RES}_phase2 \
    --config configs/w-64-hw-128-dropout.yml \
    --siamese 1 \
    --vn 0 \
    --layer-type TAGConv \
    --aggr 'sum+diff' \
    --p 4 \
    --loss mse_loss \
    --include-edge-attr 1 \
    --lr 0.001 \
    --trial $TRIAL \
    --new \
    --single-terrain-per-model \
    --wandb-tag e20TG_neurogf_terrain_graphs

