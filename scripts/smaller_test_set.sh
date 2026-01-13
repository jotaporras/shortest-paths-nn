# Jan 13 downsample to fit into memory.
# python dataset/generate-test-dataset.py \
#     --name norway \
#     --raw-data data/norway-smallest.txt \
#     --filename data/generated2/full_test-002.npz \
#     --graph-resolution 2 \
#     --dataset-size 100000 \
#     --triangles  # if you want diagonal edges

python dataset/generate-test-dataset.py \
    --name norway \
    --raw-data data/norway-smallest.txt \
    --filename data/generated2/full_test-003.npz \
    --graph-resolution 3 \
    --dataset-size 100000 \
    --triangles  # if you want diagonal edges

python dataset/generate-test-dataset.py \
    --name norway \
    --raw-data data/norway-smallest.txt \
    --filename data/generated2/full_test-004.npz \
    --graph-resolution 4 \
    --dataset-size 100000 \
    --triangles  # if you want diagonal edges