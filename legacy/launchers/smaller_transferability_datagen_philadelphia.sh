#!/bin/bash
# This script generates the datasets for the smaller transferability analysis (res 21...)


## TRAIN SETS
#for RES in $(seq -w 21 40); do
for RES in $(seq -w 40 -1 1); do
    # Phase 1 dataset (for GNN/SGT training)
    python dataset/dataset.py \
        --name phil \
        --raw-data data/flat-phil-data-1.txt \
        --filename data/generated_phil/res${RES}_phase1.npz \
        --graph-resolution $RES \
        --num-sources 500 \
        --dataset-size 50000 \
        --sampling-technique distance-based \
        --triangles
done

### TEST SETS
# We generate three datasets to have a feel for runtimes, but we'll probably only use res004.
python dataset/generate-test-dataset.py \
    --name phil \
    --raw-data data/flat-phil-data-1.txt \
    --filename data/generated_phil/full_test-003.npz \
    --graph-resolution 3 \
    --dataset-size 100000 \
    --triangles  # if you want diagonal edges

python dataset/generate-test-dataset.py \
    --name norway \
    --raw-data data/flat-phil-data-1.txt \
    --filename data/generated2/full_test-004.npz \
    --graph-resolution 4 \
    --dataset-size 100000 \
    --triangles  # if you want diagonal edges


python dataset/generate-test-dataset.py \
    --name phil \
    --raw-data data/flat-phil-data-1.txt \
    --filename data/generated_phil/full_test-008.npz \
    --graph-resolution 8 \
    --dataset-size 100000 \
    --triangles  # if you want diagonal edges