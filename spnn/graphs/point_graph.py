"""Graphs on irregular point sets (see ``point_sampling``).

Every builder returns ``(G, node_features)`` like ``grid.construct_nx_graph``:
node ids 0..N-1 in the row order of ``node_features`` and edge attribute
``weight`` = ||p_i - p_j||_p in 3D (km). So ``grid.to_pyg_graph`` works unchanged.
Neighbourhoods are computed in xy only.
"""
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .quadtree import build_quadtree, quadtree_simplification


def _weighted_graph(node_features, edges, p=2):
    node_features = np.asarray(node_features, dtype=float)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    w = np.linalg.norm(node_features[edges[:, 0]] - node_features[edges[:, 1]], ord=p, axis=1)
    G = nx.Graph()
    G.add_nodes_from(range(len(node_features)))
    G.add_weighted_edges_from((int(a), int(b), float(c)) for (a, b), c in zip(edges, w))
    return G, node_features


def build_knn_graph(node_features, k=8, p=2):
    """Symmetrized kNN graph on xy: edge (i, j) if j is among i's k nearest or vice versa."""
    xy = np.asarray(node_features)[:, :2]
    _, nbr = cKDTree(xy).query(xy, k=k + 1)
    src = np.repeat(np.arange(len(xy)), k)
    return _weighted_graph(node_features, np.stack([src, nbr[:, 1:].ravel()], axis=1), p=p)


def build_delaunay_graph(node_features, p=2):
    """Edges of the Delaunay triangulation of the xy points (planar)."""
    xy = np.asarray(node_features)[:, :2]
    tri = Delaunay(xy).simplices
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    return _weighted_graph(node_features, edges, p=p)


def mean_nn_spacing(node_features):
    """Mean xy distance from each point to its nearest neighbour (km)."""
    xy = np.asarray(node_features)[:, :2]
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(d[:, 1].mean())


def build_radius_graph(node_features, radius=None, p=2):
    """Epsilon-ball graph on xy. Default radius = 1.5 x ``mean_nn_spacing``."""
    xy = np.asarray(node_features)[:, :2]
    if radius is None:
        radius = 1.5 * mean_nn_spacing(node_features)
    pairs = cKDTree(xy).query_pairs(radius, output_type='ndarray')
    return _weighted_graph(node_features, pairs, p=p)


def build_quadtree_graph(xv, yv, elev, thresh, p=2):
    """Quadtree simplification of the full DEM (``quadtree.py``) as a weighted graph.

    A quadtree leaf is a block whose elevation range is <= ``thresh`` (km). Nodes are
    the leaf corners, edges are the leaf sides. Returns (G, node_features, dem_idx)
    with nodes relabelled to 0..N-1 in sorted ``dem_idx`` order.
    """
    rows, cols = elev.shape
    tree = build_quadtree(elev, thresh=thresh)
    Q = quadtree_simplification(tree, rows, cols)
    dem_idx = np.array(sorted(Q.nodes), dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(dem_idx)}
    feats = np.stack([xv.ravel()[dem_idx], yv.ravel()[dem_idx], elev.ravel()[dem_idx]], axis=1)
    edges = [(pos[int(a)], pos[int(b)]) for a, b in Q.edges]
    G, feats = _weighted_graph(feats, edges, p=p)
    return G, feats, dem_idx


def connect_components(G, node_features, max_frac=0.01, p=2):
    """Join every small component to the largest one by its shortest xy bridge.

    Each non-largest component gets one edge: the closest (in xy) pair between it
    and the largest component, weighted like every other edge. Raises if the
    small components hold more than ``max_frac`` of the nodes, because then the
    graph construction itself is wrong, not just unlucky. Returns the number of
    bridges added (0 for a connected graph). G is changed in place.
    """
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(comps) == 1:
        return 0
    n_small = sum(len(c) for c in comps[1:])
    if n_small > max_frac * G.number_of_nodes():
        raise ValueError(f"{len(comps) - 1} small components hold {n_small} of "
                         f"{G.number_of_nodes()} nodes (> {max_frac:.0%})")
    feats = np.asarray(node_features, dtype=float)
    main = np.fromiter(comps[0], dtype=np.int64)
    tree = cKDTree(feats[main, :2])
    for comp in comps[1:]:
        comp = np.fromiter(comp, dtype=np.int64)
        d, j = tree.query(feats[comp, :2])
        a, b = int(comp[np.argmin(d)]), int(main[j[np.argmin(d)]])
        G.add_edge(a, b, weight=float(np.linalg.norm(feats[a] - feats[b], ord=p)))
    return len(comps) - 1
