#!/bin/bash
# This script generates the datasets for the smaller transferability analysis (res 21...)

#for RES in $(seq -w 21 40); do
for RES in $(seq -w 40 -1 21); do
    # Phase 1 dataset (for GNN training)
    python dataset/dataset.py \
        --name norway \
        --raw-data data/norway-smallest.txt \
        --filename data/generated2/res${RES}_phase1.npz \
        --graph-resolution $RES \
        --num-sources 500 \
        --dataset-size 50000 \
        --sampling-technique distance-based \
        --triangles

    # Phase 2 dataset (for MLP finetuning)
    python dataset/dataset.py \
        --name norway \
        --raw-data data/norway-smallest.txt \
        --filename data/generated2/res${RES}_phase2.npz \
        --graph-resolution $RES \
        --num-sources 500 \
        --dataset-size 100000 \
        --sampling-technique single-source-random \
        --triangles
done
