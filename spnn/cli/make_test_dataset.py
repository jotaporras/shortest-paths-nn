"""Build a DEM grid graph (4-neighbour, optional diagonals) and save test pairs.

Usage: python -m spnn.cli.make_test_dataset --name norway --raw-data ... --filename ... \
           --graph-resolution 20 --triangles
"""
import argparse

import numpy as np

from ..graphs.grid import DATASET_INFO, construct_nx_graph_4n, get_dem_xv_yv_, load_dem_data_
from ..sampling.test_pairs import construct_pyg_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--dataset-size', type=int)
    parser.add_argument('--distance-based-sampling', action='store_true')
    parser.add_argument('--triangles', action='store_true')

    args = parser.parse_args()
    dem_res = DATASET_INFO[args.name][0]
    imperial = DATASET_INFO[args.name][1]
    if args.name == 'la':
        dem_array = np.load(args.raw_data)
    else:
        dem_array = load_dem_data_(args.raw_data, imperial)
    xv, yv, elevations = get_dem_xv_yv_(dem_array, dem_res)
    # row,col = np.random.choice(elevations.shape[0], size=[2,])
    # print(row, col)
    # Norway
    # row = 122 
    # col = 1647
    ## Philadelphia
    # row = 181 
    # col = 613
    ## holland 
    # row = 439 
    # col = 471
    ## L A
    # 624 510
    # row = 624
    # col = 512
    # xv_n = xv[row:row+ 100, col:col+100]
    # yv_n = yv[row:row+100, col:col+100]
    # elevations_n = elevations[row:row+100, col:col+100]
    
    res = args.graph_resolution
    sz = args.dataset_size
    filename = args.filename
    xv_n = xv[::res, ::res]
    yv_n = yv[::res, ::res]
    elevations_n = elevations[::res, ::res]
    print('terrain shape:', elevations.shape)

    G, node_features = construct_nx_graph_4n(xv_n, yv_n, elevations_n, triangles=args.triangles)
    
    construct_pyg_dataset(G, 
                        node_features, 
                        filename, 
                        size=sz, 
                        distance_based=args.distance_based_sampling,
                        m = elevations_n.shape[1])


if __name__ == '__main__':
    main()
