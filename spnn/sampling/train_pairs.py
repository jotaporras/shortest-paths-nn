"""Training pairs: choose source nodes, sample targets, save an .npz dataset."""
import itertools
from functools import partial

import numpy as np
from tqdm import tqdm

from ..graphs.grid import to_pyg_graph
from .critical_points import find_critical_points, reshape_node_features_grid
from .targets import (compute_altitude_variance, distance_based, hybrid_sampling,
                      proximity_variance_sampling, random_sampling)

# Morse points for the mountain-ridges methods, made with
# https://github.com/wangjiayuan007/graph_recon_DM.
DEFAULT_MORSE_POINTS = '/data/sam/terrain/data/norway/{rows}-morse-points-t-{threshhold}.npy'


# Unused. Probability 1/hops^2 for every pair of a grid with m columns.
def generate_probabilities(N, m):
    all_pairs = list(itertools.combinations(range(N), 2))
    probabilities = []
    for src, tar in tqdm(all_pairs):
        hops = abs(src//m - tar//m) + abs(src % m - tar % m )
        probabilities.append(1/(hops**2) if hops > 0 else 1)
    return all_pairs, probabilities

def convert_to_node_idx(pairs, rows, cols):
    return pairs[:, 1] * rows + pairs[:, 0]

def construct_dataset(G,
                      node_features, 
                      filename, 
                      sampling_method, 
                      num_srcs, 
                      samples_per_source, 
                      rows=None,
                      cols=None,
                      threshhold=0.2, 
                      mixture=0.2,
                      morse_points_path=None):
    edges, distances = to_pyg_graph(G)
    
    if sampling_method == 'single-source-random':
        """
        Sample random source nodes

        For shortest paths, sample random target nodes from across the terrain
        """
        src_nodes = np.random.choice(len(node_features), size=num_srcs)
        src_sampling_fn_pairs = [(src_nodes, random_sampling)]
    elif sampling_method == 'critical-point-source':
        """
        Use only critical points as source nodes

        For shortest paths, sample random target nodes from across the terrain
        """
        node_features = np.array(node_features)
        terrain = reshape_node_features_grid(node_features, rows, cols)
        src_nodes = find_critical_points(terrain, threshhold)
        src_sampling_fn_pairs = [(src_nodes, random_sampling)]

    elif sampling_method == 'distance-based':
        """
        Sample random source nodes

        Sample more target points from around the source node
        """
        src_nodes = np.random.choice(len(node_features), size=num_srcs)
        src_sampling_fn_pairs = [(src_nodes, distance_based)]

    elif sampling_method == 'mixed-distance-based-critical-point':
        """
        Use a mixture of critical and random nodes as source nodes. 
        
        To sample shortest paths, we use distance based sampling function for target nodes.
        """
        node_features = np.array(node_features)
        terrain = reshape_node_features_grid(node_features, rows, cols)
        critical_points = find_critical_points(terrain, threshhold)
        num_cp = int(num_srcs*mixture)
        if num_cp > len(critical_points):
            num_random = num_srcs - len(critical_points)
            cp_srcs = critical_points.astype(int)
        else:
            cp_srcs = np.random.choice(critical_points, size=num_cp, replace=False).astype(int)
            num_random = num_srcs - num_cp
        random_srcs =  np.random.choice(len(node_features), size=num_random, replace=False).astype(int)
        print(len(random_srcs), len(cp_srcs))
        src_nodes = np.hstack((random_srcs, cp_srcs))
        src_sampling_fn_pairs = [(src_nodes, distance_based)]
    elif sampling_method == 'mixed-random-sampling-critical-point':
        """
        Use a mixture of critical and random nodes as source nodes.

        To sample shortest paths, we use random sampling function for target nodes
        """
        
        node_features = np.array(node_features)
        terrain = reshape_node_features_grid(node_features, rows, cols)
        critical_points = find_critical_points(terrain, threshhold)

        num_cp = int(num_srcs*mixture)
        if num_cp > len(critical_points):
            num_random = num_srcs - len(critical_points)
            cp_srcs = critical_points
        else:
            cp_srcs = np.random.choice(critical_points, size=num_cp, replace=False)
            num_random = num_srcs - num_cp
        random_srcs =  np.random.choice(len(node_features), size=num_random, replace=False)
        src_nodes = np.hstack((random_srcs, cp_srcs))
        src_sampling_fn_pairs = [(src_nodes, random_sampling)]

    elif sampling_method == 'mixed-db-critical-points-rs-random-points':
        """
        Use a mixture of critical and random nodes as source nodes

        Use distance based sampling for targets for critical points

        Use random sampling for targets for random points

        We construct the dataset here too and then exit. This is because we 
        want to use two different sampling functions. I should probably change
        this so we can do this in general. 
        """
        node_features = np.array(node_features)
        terrain = reshape_node_features_grid(node_features, rows, cols)
        critical_points = find_critical_points(terrain, threshhold)
        num_cp = int(num_srcs*mixture)
        
        if num_cp > len(critical_points):
            num_random = num_srcs - len(critical_points)
            print(num_random)
            cp_srcs = critical_points
        else:
            cp_srcs = np.random.choice(critical_points, size=num_cp, replace=False)
            num_random = num_srcs - num_cp
        random_points = np.random.choice(len(node_features), size=num_random, replace=False)
        print(len(np.unique(cp_srcs)), len(np.unique(random_points)))
        src_sampling_fn_pairs = [(cp_srcs, distance_based), (random_points, random_sampling)]

    elif sampling_method == 'mountain-ridges':
        """
        Sample source points only from mountain ridges

        Mountain ridges are found from discrete Morse reconstruction algorithm and generated using
        https://github.com/wangjiayuan007/graph_recon_DM (Jiayuan Wang). Download and follow the 
        directions in that repository to generate your own set of Morse points. 
        """
        
        morse_pts = np.load(morse_points_path or DEFAULT_MORSE_POINTS.format(rows=rows, threshhold=threshhold))
        morse_pts_idxs = convert_to_node_idx(morse_pts, rows=rows, cols=cols).astype(int)
        num_mp = int(num_srcs*mixture)
        num_random = num_srcs - num_mp
        print("Number of morse points", len(morse_pts_idxs))
        morse_pt_srcs = np.random.choice(morse_pts_idxs, size=num_mp, replace=False)
        random_srcs = np.random.choice(len(node_features), size=num_random, replace=False)
        print(len(morse_pt_srcs), len(random_srcs))

        src_nodes = np.hstack((morse_pt_srcs, 
                               random_srcs))
        print("Number of unique sources:", len(np.unique(src_nodes)))
        src_sampling_fn_pairs = [(src_nodes, random_sampling)]
    elif sampling_method == 'distance-based-mountain-ridges':
        """
        Sample source points only from mountain ridges

        Mountain ridges are found from discrete Morse reconstruction algorithm and generated using
        https://github.com/wangjiayuan007/graph_recon_DM (Jiayuan Wang). Download and follow the 
        directions in that repository to generate your own set of Morse points. 
        """
        
        morse_pts = np.load(morse_points_path or DEFAULT_MORSE_POINTS.format(rows=rows, threshhold=threshhold))
        morse_pts_idxs = convert_to_node_idx(morse_pts, rows=rows, cols=cols).astype(int)
        num_mp = int(num_srcs*mixture)
        num_random = num_srcs - num_mp
        print("Number of morse points", len(morse_pts_idxs))
        morse_pt_srcs = np.random.choice(morse_pts_idxs, size=num_mp, replace=False)
        random_srcs = np.random.choice(len(node_features), size=num_random, replace=False)
        print(len(morse_pt_srcs), len(random_srcs))

        src_nodes = np.hstack((morse_pt_srcs, 
                               random_srcs))
        print("Number of unique sources:", len(np.unique(src_nodes)))
        src_sampling_fn_pairs = [(src_nodes, distance_based)]
    elif sampling_method == 'proximity-variance':
        node_features = np.array(node_features)
        alt_var = compute_altitude_variance(node_features, rows, cols)
        src_probs = alt_var / (alt_var.sum() + 1e-12)
        src_nodes = np.random.choice(len(node_features), size=num_srcs, replace=False, p=src_probs)
        sampling_fn = partial(proximity_variance_sampling, altitude_variance=alt_var)
        src_sampling_fn_pairs = [(src_nodes, sampling_fn)]
    elif sampling_method == 'hybrid':
        node_features = np.array(node_features)
        alt_var = compute_altitude_variance(node_features, rows, cols)
        n_uniform_src = num_srcs // 2
        n_elev_src = num_srcs - n_uniform_src
        uniform_srcs = np.random.choice(len(node_features), size=n_uniform_src, replace=False)
        src_probs = alt_var / (alt_var.sum() + 1e-12)
        remaining = np.setdiff1d(np.arange(len(node_features)), uniform_srcs)
        remaining_probs = src_probs[remaining]
        remaining_probs = remaining_probs / remaining_probs.sum()
        elev_srcs = np.random.choice(remaining, size=n_elev_src, replace=False, p=remaining_probs)
        src_nodes = np.concatenate([uniform_srcs, elev_srcs])
        sampling_fn = partial(hybrid_sampling, altitude_variance=alt_var)
        src_sampling_fn_pairs = [(src_nodes, sampling_fn)]
    else:
        raise NotImplementedError("please choose between 'single-source-random', \
                                                         'critical-point-source', \
                                                         'distance-based', \
                                                         'mountain-ridges', \
                                                         'mixed-distance-based-critical-point', \
                                                         'mixed-random-sampling-critical-point'" )
    srcs = []
    tars = []
    lengths = []
    print("Number of source nodes:", num_srcs)
    print("Generating shortest path distances.....")
    for pair in src_sampling_fn_pairs:
        src_nodes = pair[0]
        sampling_fn = pair[1]
        for src in tqdm(src_nodes):
            # cols lets the grid samplers find a node's row and column.
            source, target, length = sampling_fn(G, samples_per_source, src=src, cols=cols)
            srcs += source
            tars += target
            lengths += length
    print("Number of lengths in dataset:", len(lengths))
    print("Saved dataset in:", filename)
    np.savez(filename, 
         edge_index = edges, 
         distances=distances, 
         srcs = srcs,
         tars = tars,
         lengths = lengths,
         node_features=node_features)
