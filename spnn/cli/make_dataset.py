"""Build a DEM grid graph (or a mesh graph) and save training pairs.

Usage: python -m spnn.cli.make_dataset --name norway --raw-data ... --filename ... \
           --graph-resolution 20 --dataset-size 50000 --num-sources 128 --sampling-technique hybrid
"""
import argparse
import json
import os

import numpy as np

from ..graphs.grid import DATASET_INFO, construct_nx_graph, get_dem_xv_yv_, load_dem_data_
from ..graphs.mesh import mesh_to_graph
from ..graphs.schemes import SCHEMES, build_scheme_graph
from ..sampling.train_pairs import construct_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--dataset-size', type=int)
    parser.add_argument('--num-sources', type=int)
    parser.add_argument('--sampling-technique', type=str, default='random')
    parser.add_argument('--triangles', action='store_true')
    parser.add_argument('--edge-weight', action='store_true')
    parser.add_argument('--change-heights', action='store_true')
    parser.add_argument('--critical-point-threshhold', type=float, default=0.2)
    parser.add_argument('--mixture-param', type=float, default=0.2)
    parser.add_argument('--scheme', type=str, default=None, choices=SCHEMES,
                        help='Node-sampling scheme (spnn.graphs.schemes). N matches the grid at '
                             '--graph-resolution. Irregular schemes use kNN edges. Writes a '
                             '<filename>.json sidecar with the graph stats. Leave out for the old grid path.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the node sample and for the training pairs (with --scheme).')
    parser.add_argument('--knn-k', type=int, default=8)

    args = parser.parse_args()
    dem_res = DATASET_INFO[args.name][0]
    imperial = DATASET_INFO[args.name][1]
    if 'meshes' in args.raw_data:
        # Critical points are found on the DEM grid, so they need rows and cols.
        if 'critical-point' in args.sampling_technique:
            parser.error(f"--sampling-technique {args.sampling_technique} needs a DEM grid, not a mesh")
        edge_filename = os.path.join(args.raw_data, 'percent_edges')
        vertex_filename = os.path.join(args.raw_data, 'percent_vertices')
        G, node_features = mesh_to_graph(edge_filename, vertex_filename)
        construct_dataset(G=G,
                          node_features=node_features,
                          filename=args.filename,
                          num_srcs=args.num_sources,
                          samples_per_source=args.dataset_size // args.num_sources,
                          sampling_method=args.sampling_technique,
                          threshhold=args.critical_point_threshhold,
                          mixture=args.mixture_param)
    elif args.scheme is not None:
        make_scheme_dataset(args, dem_res, imperial)
    else:
        if args.name == 'la' or args.name == 'artificial':
            dem_array = np.load(args.raw_data)
        else:
            dem_array = load_dem_data_(args.raw_data, imperial)
        xv, yv, elevations = get_dem_xv_yv_(dem_array, dem_res)
        res = args.graph_resolution
        
        filename = args.filename
        
        xv_n = xv[::res, ::res]
        yv_n = yv[::res, ::res]
        elevations_n = elevations[::res, ::res]
        print(np.min(elevations_n))
        print('terrain shape:', elevations_n.shape)
        print('resolution:', res)
        if args.change_heights:
            for k in range(1, 60):
                #TODO this doesnt matter is a dead path
                filename = f'/data/terrain/datasets/norway/uncertainty/50/50k-{k}.npz'
                uncertainty = np.random.uniform(-0.050, 0.050, size=elevations_n.shape)
                elevations_n = uncertainty + elevations_n
                
                G, node_features = construct_nx_graph(xv_n, 
                                                      yv_n, 
                                                      elevations_n, 
                                                      triangles=args.triangles, 
                                                      scale=args.edge_weight)
                sampling_technique = args.sampling_technique
                
                construct_dataset(G = G, 
                                  node_features = node_features, 
                                  filename = filename, 
                                  num_srcs = args.num_sources,
                                  samples_per_source = args.dataset_size//args.num_sources,
                                  sampling_method=sampling_technique,
                                  rows = elevations_n.shape[0],
                                  cols = elevations_n.shape[1], 
                                  threshhold = args.critical_point_threshhold)
        else:
            G, node_features = construct_nx_graph(xv_n, 
                                                  yv_n, 
                                                  elevations_n, 
                                                  triangles=args.triangles, 
                                                  scale=args.edge_weight)
            filename = args.filename
            sampling_technique = args.sampling_technique
            
            construct_dataset(G = G, 
                                  node_features = node_features, 
                                  filename = filename, 
                                  num_srcs = args.num_sources,
                                  samples_per_source = args.dataset_size//args.num_sources,
                                  sampling_method=sampling_technique,
                                  rows = elevations_n.shape[0],
                                  cols = elevations_n.shape[1], 
                                  threshhold = args.critical_point_threshhold, 
                                  mixture=args.mixture_param)


# Pair samplers that only need the graph, not a DEM grid layout.
GRAPH_ONLY_SAMPLERS = ('single-source-random',)


def make_scheme_dataset(args, dem_res, imperial):
    """Training pairs on the graph of one node-sampling scheme (experiment e31)."""
    if args.scheme != 'grid' and args.sampling_technique not in GRAPH_ONLY_SAMPLERS:
        raise SystemExit(f"--scheme {args.scheme} needs --sampling-technique in {GRAPH_ONLY_SAMPLERS}")
    if args.change_heights or args.edge_weight or args.triangles:
        raise SystemExit("--scheme does not support --change-heights, --edge-weight or --triangles")
    dem_array = load_dem_data_(args.raw_data, imperial)
    xv, yv, elevations = get_dem_xv_yv_(dem_array, dem_res, visualize=False)
    G, node_features, dem_idx, info = build_scheme_graph(
        xv, yv, elevations, args.scheme, args.graph_resolution, seed=args.seed, knn_k=args.knn_k)
    print(json.dumps(info, indent=1))
    rows = cols = None
    if args.scheme == 'grid':
        rows, cols = elevations[::args.graph_resolution, ::args.graph_resolution].shape
    np.random.seed(args.seed)
    construct_dataset(G=G, node_features=node_features, filename=args.filename,
                      num_srcs=args.num_sources,
                      samples_per_source=args.dataset_size // args.num_sources,
                      sampling_method=args.sampling_technique, rows=rows, cols=cols,
                      threshhold=args.critical_point_threshhold, mixture=args.mixture_param)
    # Add the DEM cell of every node, for analysis; training ignores extra keys.
    path = args.filename if args.filename.endswith('.npz') else args.filename + '.npz'
    with np.load(path) as data:
        arrays = dict(data)
    np.savez(path, dem_idx=dem_idx, **arrays)
    info.update(raw_data=args.raw_data, sampling_technique=args.sampling_technique,
                num_sources=args.num_sources, dataset_size=args.dataset_size)
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump(info, f, indent=1)


if __name__ == '__main__':
    main()
