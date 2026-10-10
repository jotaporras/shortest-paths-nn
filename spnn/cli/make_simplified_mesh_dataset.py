"""
Dataset Generation Script

This script processes vertex and edge files into npz datasets
for training and testing graph models.

Acknowledgment:
Special thanks to Caris for her contributions and support in developing this script.
"""
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import argparse

def generate_dataset(vertex_file, edge_file, output_prefix, num_pairs=62500, test_fraction=0.2):
    """Save num_pairs distinct unordered (src, tar) pairs, then an 80/20 train/test split.

    Pairs are distinct as unordered pairs, so no pair (or its reverse) is in both splits.
    """
    raw_vertices = np.loadtxt(vertex_file)
    vertex_ids = raw_vertices[:, 0].astype(int)
    coordinates = raw_vertices[:, 1:4]

    id_to_index = {vid: idx for idx, vid in enumerate(vertex_ids)}

    node_features = coordinates

    raw_edges = np.loadtxt(edge_file, dtype=int)

    remapped_edges = []
    for v1, v2 in raw_edges:
        if v1 in id_to_index and v2 in id_to_index:
            remapped_edges.append([id_to_index[v1], id_to_index[v2]])

    remapped_edges = np.array(remapped_edges).T

    distances = []
    for src, tgt in remapped_edges.T:
        p1 = node_features[src]
        p2 = node_features[tgt]
        dist = np.linalg.norm(p1 - p2)
        distances.append(dist)
    distances = np.array(distances)

    G = nx.Graph()
    for (src, tgt, dist) in zip(remapped_edges[0], remapped_edges[1], distances):
        G.add_edge(int(src), int(tgt), weight=dist)

    srcs = []
    tars = []
    lengths = []
    node_indices = list(G.nodes)
    num_pairs = min(num_pairs, len(node_indices) * (len(node_indices) - 1) // 2)

    seen = set()
    attempts = 0
    while len(srcs) < num_pairs and attempts < 10 * num_pairs:
        attempts += 1
        src, tar = np.random.choice(node_indices, size=2, replace=False)
        key = (min(src, tar), max(src, tar))
        if key in seen:
            continue
        seen.add(key)
        try:
            length = nx.shortest_path_length(G, source=src, target=tar, weight='weight')
            srcs.append(src)
            tars.append(tar)
            lengths.append(length)
        except nx.NetworkXNoPath:
            continue

    srcs = np.array(srcs)
    tars = np.array(tars)
    lengths = np.array(lengths)

    np.savez(f"{output_prefix}.npz",
             edge_index=remapped_edges,
             distances=distances,
             node_features=node_features,
             srcs=srcs,
             tars=tars,
             lengths=lengths)

    print(f"Saved full dataset: {output_prefix}.npz")

    srcs_train, srcs_test, tars_train, tars_test, lengths_train, lengths_test = train_test_split(
        srcs, tars, lengths, test_size=test_fraction, random_state=42
    )

    np.savez(f"{output_prefix}_train.npz",
             edge_index=remapped_edges,
             distances=distances,
             node_features=node_features,
             srcs=srcs_train,
             tars=tars_train,
             lengths=lengths_train)

    np.savez(f"{output_prefix}_test.npz",
             edge_index=remapped_edges,
             distances=distances,
             node_features=node_features,
             srcs=srcs_test,
             tars=tars_test,
             lengths=lengths_test)

    print(f"Saved:\nTrain -> {output_prefix}_train.npz\nTest  -> {output_prefix}_test.npz")
    print(f"Train size: {len(srcs_train)}, Test size: {len(srcs_test)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertex_file", required=True, help="Path to simplified_vertices.txt")
    parser.add_argument("--edge_file", required=True, help="Path to simplified_edges.txt")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files")
    parser.add_argument("--num-pairs", type=int, default=62500, help="Distinct pairs before the split")
    parser.add_argument("--test-fraction", type=float, default=0.2)

    args = parser.parse_args()
    generate_dataset(args.vertex_file, args.edge_file, args.output_prefix,
                     num_pairs=args.num_pairs, test_fraction=args.test_fraction)


if __name__ == "__main__":
    main()
