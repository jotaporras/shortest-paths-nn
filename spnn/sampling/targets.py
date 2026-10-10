"""Target samplers: given a source node, pick targets and their shortest-path lengths."""
import networkx as nx
import numpy as np


def random_sampling(graph: nx.Graph, samples_per_source, src=None):
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random, never the source itself
    candidates = np.delete(np.arange(graph.number_of_nodes()), src)
    target = np.random.choice(candidates, (samples_per_source, ), replace=False)
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
    probs[src] = 0  # never the source itself
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
