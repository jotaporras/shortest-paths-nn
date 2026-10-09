"""Build a mesh graph from node and edge CSV files and save random training pairs.

Usage: python -m spnn.cli.make_mesh_dataset --filename out.npz --edge-input-data edges.csv \
           --node-feature-data nodes.csv --num-sources 128 --dataset-size 5000
"""
import argparse

from ..graphs.mesh import construct_mesh_graph, process_edges, process_node_features
from ..sampling.train_pairs import construct_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--edge-input-data', type=str)
    parser.add_argument('--node-feature-data', type=str)
    parser.add_argument('--num-sources', type=int)
    parser.add_argument('--dataset-size', type=int)

    args = parser.parse_args()
    node_idx_dict, node_features = process_node_features(args.node_feature_data)
    edges = process_edges(args.edge_input_data)

    G = construct_mesh_graph(edges, node_idx_dict)
    construct_dataset(G=G, 
                       node_features=node_features, 
                       filename=args.filename,
                       num_srcs=args.num_sources, 
                       samples_per_source = args.dataset_size//args.num_sources,
                       sampling_method='single-source-random')
    return 

if __name__ == '__main__':
    main()
