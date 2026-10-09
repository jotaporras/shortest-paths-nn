#!/bin/bash
# Phase 1 GNN training wrapper for Norway terrain graphs
# Usage: ./run_phase1_gnn.sh <resolution> [trial]
# Example: ./run_phase1_gnn.sh 01 1

set -e

RES=$1
TRIAL=${2:-1}

if [ -z "$RES" ]; then
    echo "Usage: ./run_phase1_gnn.sh <resolution> [trial]"
    echo "Example: ./run_phase1_gnn.sh 01 1"
    exit 1
fi

# Batch size: 32 for res01-05, 256 for res06-20
RES_NUM=$((10#$RES))
if [ $RES_NUM -le 5 ]; then
    BATCH_SIZE=32
else
    BATCH_SIZE=256
fi

echo "=== Phase 1: Training GNN on res${RES}_phase1.npz (batch_size=$BATCH_SIZE) ==="

python train_single_terrain_case.py \
    --train-data generated/res${RES}_phase1.npz \
    --test-data generated/full_test-001.npz \
    --epochs 500 \
    --device cuda \
    --batch-size $BATCH_SIZE \
    --dataset-name norway/res${RES} \
    --config configs/w-64-hw-128-dropout.yml \
    --siamese 1 \
    --vn 0 \
    --layer-type TAGConv \
    --aggr 'sum+diff' \
    --p 4 \
    --loss mse_loss \
    --finetune 0 \
    --include-edge-attr 1 \
    --lr 0.0001 \
    --trial $TRIAL \
    --new \
    --wandb-tag e20TG_neurogf_terrain_graphs

