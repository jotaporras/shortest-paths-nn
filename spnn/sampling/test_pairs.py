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

# n = rows
# m = columns
def generate_src_tar_pairs(node_features, n, m, size=100, sampling_technique='random'):
    num_nodes = len(node_features)
    node_idxs = np.reshape(np.arange(n * m), (n, m))
    tars = []
    if sampling_technique == 'random':
        srcs = np.random.choice(np.arange(num_nodes), size = size)
        tars = np.random.choice(np.arange(num_nodes), size = size)
    elif sampling_technique == 'expanding-radius':
        radii = [60, 100, 120, 140, 160, 200]
        num_per_radius = 20
        num_srcs = size // (len(radii) * num_per_radius)
        srcs = np.random.choice(np.arange(num_nodes), size = num_srcs)
        for s in srcs:
            x_loc = s//n
            y_loc = s % m
            for r in radii: 
                # collect all nodes at radii 5
                nodes_at_radii = get_neighbors((x_loc, y_loc), n=r)
                for i in range(num_per_radius):
                    node = nodes_at_radii[np.random.choice(len(nodes_at_radii), replace = False)]
                    if node[0] >= n or node[0] < 0:
                        continue
                    if node[1] >= m or node[1] < 0:
                        continue
                    tar = node_idxs[node[0], node[1]]
                    tars.append(tar)
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
