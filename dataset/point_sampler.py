# I took this chunk of code from Haoyun Wang's final project
# from the Topological Data Analysis course from UCSD.

import numpy as np
import torch
import networkx as nx
from ripser import ripser, lower_star_img
from persim import plot_diagrams
from functools import partial

def random_sampling(graph: nx.Graph, samples_per_source, src=None):
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random
    target = np.random.choice(graph.number_of_nodes(), (samples_per_source, ), replace=False)
    distance = [distance[t.item()] for t in target]
    return [src] * samples_per_source, target.tolist(), distance


def distance_based(graph: nx.Graph, samples_per_source, src=None):
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random
    vertices = np.arange(graph.number_of_nodes())
    row_num = int(np.around(graph.number_of_nodes() ** 0.5))
    hops = abs(src // row_num - vertices // row_num) + abs(src % row_num - vertices % row_num) + 1
    probs = 1 / hops
    probs = probs / probs.sum()
    target = np.random.choice(vertices, (samples_per_source, ), p=probs, replace=False)
    distance = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), distance


def compute_altitude_variance(node_features, rows, cols):
    """Per-node elevation gradient magnitude on a (rows x cols) grid.

    Uses central-difference gradients in x and y, returning a flat array
    of shape (rows*cols,) with the L2 gradient magnitude at each node.
    """
    z = np.asarray(node_features)[:, 2].reshape(rows, cols)
    gy = np.empty_like(z)
    gx = np.empty_like(z)
    gy[1:-1, :] = (z[2:, :] - z[:-2, :]) / 2
    gy[0, :] = z[1, :] - z[0, :]
    gy[-1, :] = z[-1, :] - z[-2, :]
    gx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / 2
    gx[:, 0] = z[:, 1] - z[:, 0]
    gx[:, -1] = z[:, -1] - z[:, -2]
    return np.sqrt(gx**2 + gy**2).ravel()


def proximity_variance_sampling(graph: nx.Graph, samples_per_source, src=None,
                                altitude_variance=None,
                                proximity_decay=1.0,
                                variance_boost=2.0):
    """Sample targets biased toward nodes that are closer to the source
    (grid-Manhattan proximity) and located in regions of high elevation
    gradient.

    altitude_variance: 1-D array of length N (one per node), typically
    the output of compute_altitude_variance.
    """
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)

    N = graph.number_of_nodes()
    vertices = np.arange(N)
    row_num = int(np.around(N ** 0.5))

    hops = (abs(src // row_num - vertices // row_num)
            + abs(src % row_num - vertices % row_num) + 1)
    proximity_w = 1.0 / (hops ** proximity_decay)

    if altitude_variance is not None:
        var_w = altitude_variance / (altitude_variance.max() + 1e-12)
        probs = proximity_w + variance_boost * var_w
    else:
        probs = proximity_w

    probs[src] = 0.0
    probs = probs / probs.sum()

    target = np.random.choice(vertices, (samples_per_source,),
                              p=probs, replace=False)
    distance = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), distance


def hybrid_sampling(graph: nx.Graph, samples_per_source, src=None,
                    altitude_variance=None,
                    proximity_decay=1.0):
    """Sample targets: half uniformly, half biased toward nearby nodes."""
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)

    N = graph.number_of_nodes()
    vertices = np.arange(N)

    n_uniform = samples_per_source // 2
    n_proximity = samples_per_source - n_uniform

    all_others = np.delete(vertices, src)
    chosen_uniform = np.random.choice(all_others, size=n_uniform, replace=False)

    remaining = np.setdiff1d(all_others, chosen_uniform)
    row_num = int(np.around(N ** 0.5))
    hops = (abs(src // row_num - remaining // row_num)
            + abs(src % row_num - remaining % row_num) + 1)
    probs = 1.0 / (hops ** proximity_decay)
    probs = probs / probs.sum()
    chosen_proximity = np.random.choice(remaining, size=n_proximity,
                                        p=probs, replace=False)

    target = np.concatenate([chosen_uniform, chosen_proximity])
    dist_list = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), dist_list


def find_critical_points(terrain, threshold):
    # the original terrain has same-height points we must break the tie
    N = terrain.shape[0]
    terrain[:, :, 2] += torch.rand((N, N)) * 1e-5
    lower_dgm = lower_star_img(terrain[:, :, 2])
    upper_dgm = - lower_star_img(- terrain[:, :, 2])
    long_pers_lower_dgm = lower_dgm[lower_dgm[:, 1]- lower_dgm[:, 0] > threshold]
    long_pers_upper_dgm = upper_dgm[upper_dgm[:, 0]- upper_dgm[:, 1] > threshold]
    long_pers_dgm = np.concatenate([long_pers_lower_dgm, long_pers_upper_dgm])
    print(f"{long_pers_dgm.shape[0]} significant critical point pairs at threshhold {threshold}")

    flatten_terrain = terrain.flatten(0, 1)
    critical_idx_0 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_lower_dgm[i, 0])) for i in range(long_pers_lower_dgm.shape[0])]
    critical_idx_2 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_upper_dgm[i, 0])) for i in range(long_pers_upper_dgm.shape[0])]
    critical_idx_1 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_lower_dgm[i, 1])) for i in range(long_pers_lower_dgm.shape[0])] + \
                    [np.argmin(abs(flatten_terrain[:, 2] - long_pers_upper_dgm[i, 1])) for i in range(long_pers_upper_dgm.shape[0])]
    critical_idx_1 = list(set(critical_idx_1))

    critical_idx = torch.stack(critical_idx_0 + critical_idx_1 + critical_idx_2)
    # shuffle it
    critical_idx = critical_idx[torch.randperm(critical_idx.shape[0])]
    critical_idx = [src.item() for src in critical_idx]
    return critical_idx

def mesh_lower_star_filtration(mesh, threshhold):
    raise NotImplementedError("There should be a different critical point sampler for TINs")

def reshape_node_features_grid(node_features, rows, cols):
    c1 = node_features[:, 0].reshape(rows, cols)
    c2 = node_features[:, 1].reshape(rows, cols)
    c3 = node_features[:, 2].reshape(rows, cols)
    terrain = torch.tensor(np.stack([c1, c2, c3]), dtype=torch.float)
    terrain = np.transpose(terrain, (1, 2, 0))
    return terrain
