"""Test pairs: every Dijkstra target from 10 random sources (the held-out test sets)."""
import networkx as nx
import numpy as np
from tqdm import tqdm, trange

from ..graphs.grid import to_pyg_graph


def get_neighbors(center, n=1):
    ret = []
    for dx in range(-n, n + 1):
        ydiff = n - abs(dx)
        for dy in range(-ydiff, ydiff + 1):
            ret.append((center[0] + dx, center[1] + dy))
    return ret

def get_ring(center, r):
    """Grid points at L1 (hop) distance exactly r from center."""
    if r == 0:
        return [tuple(center)]
    return [p for p in get_neighbors(center, n=r)
            if abs(p[0] - center[0]) + abs(p[1] - center[1]) == r]

# n = rows
# m = columns
def generate_src_tar_pairs(node_features, n, m, size=100, sampling_technique='random',
                           radii=(60, 100, 120, 140, 160, 200), num_per_radius=20):
    num_nodes = len(node_features)
    node_idxs = np.reshape(np.arange(n * m), (n, m))
    tars = []
    if sampling_technique == 'random':
        srcs = np.random.choice(np.arange(num_nodes), size = size)
        tars = np.random.choice(np.arange(num_nodes), size = size)
    elif sampling_technique == 'expanding-radius':
        # For each source, up to num_per_radius distinct targets on the ring at
        # each hop distance in radii. Rings that miss the grid give no targets.
        num_srcs = max(1, size // (len(radii) * num_per_radius))
        src_nodes = np.random.choice(np.arange(num_nodes), size = num_srcs)
        srcs = []
        for s in src_nodes:
            row, col = divmod(int(s), m)
            for r in radii:
                ring = [p for p in get_ring((row, col), r) if 0 <= p[0] < n and 0 <= p[1] < m]
                if not ring:
                    continue
                picks = np.random.choice(len(ring), size=min(num_per_radius, len(ring)), replace=False)
                for k in picks:
                    tars.append(node_idxs[ring[k]])
                    srcs.append(s)
    # sample sources from top 100 height points. 
    elif sampling_technique == 'height-sensitive-random':
        node_features = np.array(node_features)
        sorted_height_array = np.argsort(node_features[:, 2])
        num_srcs = 10
        num_per_src = int(size//20)
        srcs = []
        src_nodes = np.random.choice(sorted_height_array[-1000000:], size = num_srcs)
        for s in tqdm(src_nodes):
            tar_nodes = np.random.choice(len(node_features), size=num_per_src)
            for t in tar_nodes:
                tars.append(t)
                srcs.append(s)
    # check that all src, target nodes are in the graph
    for i in range(len(srcs)):
        assert srcs[i] >= 0 and srcs[i] < len(node_features)
        assert tars[i] >=0 and tars[i] < len(node_features)
    return srcs, tars

def single_src_dataset(G, node_features, filename, size=100):
    lengths = []
    srcs = []
    tars = []
    node_features = np.array(node_features)
    sorted_height_array = np.argsort(node_features[:, 2])
    num_srcs = 10
    src_nodes = np.random.choice(sorted_height_array[-1000000:], size = num_srcs)
    for i in trange(len(src_nodes)):
        #src = np.random.choice(num_nodes)
        src = src_nodes[i]
        all_pairs_shortest_paths = nx.single_source_dijkstra_path_length(G, src, weight='weight')
        for tar in all_pairs_shortest_paths:
            tars.append(tar)
            srcs.append(src)
            lengths.append(all_pairs_shortest_paths[tar])

    return srcs, tars, lengths

def construct_pyg_dataset(G, node_features, filename, size=100, distance_based=False, m=10):
    Nodes = np.sort(list(G.nodes()))

    edges, distances = to_pyg_graph(G)
    
    srcs = []
    tars = []
    lengths = []
    print("Generating shortest paths......")
    #jobs = []
    #pool = mp.Pool(processes=20)
    srcs, tars, lengths = single_src_dataset(G, node_features, filename, size=size)
    # samples = np.random.choice(len(srcs), size=100000, replace=False)
    srcs = np.array(srcs)
    tars = np.array(tars)
    lengths = np.array(lengths)
    # srcs, tars = generate_src_tar_pairs(node_features, m, m, size = size, sampling_technique = 'random')
    # for i in trange(len(srcs)):
    #     s = srcs[i]
    #     t = tars[i]
    #     length = nx.shortest_path_length(G, s, t, weight='weight')
    #     lengths.append(length)

    print("Saved dataset in:", filename)
    np.savez(filename, 
         edge_index = edges, 
         distances=distances, 
         nodes=Nodes,
         srcs = srcs,
         tars = tars,
         lengths = lengths,
         node_features=node_features)
