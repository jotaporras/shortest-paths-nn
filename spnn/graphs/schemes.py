"""Terrain graphs from one DEM under different node-sampling schemes (experiment e31).

``build_scheme_graph`` returns a graph whose node count matches the stride-``s``
grid of the same DEM, so schemes are compared at equal N:

- ``grid``: every s-th row and column, 8-neighbour edges (``grid.construct_nx_graph``).
- ``uniform``, ``poisson-disk``, ``slope-adaptive``: irregular DEM cells
  (``point_sampling``) with symmetrized kNN edges, k = 8 by default. kNN-8 on a
  grid gives the 8-neighbour grid, so only the node positions differ from ``grid``.
- ``quadtree``: leaf corners and leaf sides of the DEM quadtree, with the
  elevation threshold bisected to match N.

Edge weights are unscaled 3D Euclidean lengths (km) for every scheme. Small
components are bridged to the main one (``point_graph.connect_components``).
"""
import contextlib
import io

import networkx as nx
import numpy as np

from . import point_graph as pg
from . import point_sampling as ps
from .grid import construct_nx_graph

SCHEMES = ('grid', 'uniform', 'poisson-disk', 'slope-adaptive', 'quadtree')


def grid_node_count(shape, stride):
    rows, cols = shape
    return int(np.ceil(rows / stride) * np.ceil(cols / stride))


def _quadtree_for_count(xv, yv, elev, n_target, lo=1e-4, hi=0.5, iters=16):
    """Bisect log(thresh); the node count falls as thresh grows."""
    best = None
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            G, feats, idx = pg.build_quadtree_graph(xv, yv, elev, mid)
        n = G.number_of_nodes()
        if best is None or abs(n - n_target) < abs(best[0].number_of_nodes() - n_target):
            best = (G, feats, idx, mid)
        if n == n_target:
            break
        if n > n_target:
            lo = mid
        else:
            hi = mid
    return best


def build_scheme_graph(xv, yv, elev, scheme, stride, seed=0, knn_k=8, max_bridge_frac=0.01):
    """Return (G, node_features, dem_idx, info) for one scheme at the N of the stride grid."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {SCHEMES}")
    n = grid_node_count(elev.shape, stride)
    info = {'scheme': scheme, 'stride': stride, 'n_target': n, 'seed': seed}
    if scheme == 'grid':
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            G, feats = construct_nx_graph(xv[::stride, ::stride], yv[::stride, ::stride],
                                          elev[::stride, ::stride], p=2, scale=False)
        feats = np.asarray(feats, dtype=float)
        ref, dem_idx = ps.sample_regular_grid(xv, yv, elev, stride)
        if not np.allclose(ref, feats):
            raise RuntimeError("grid node order differs from sample_regular_grid")
        info['edge_rule'] = '8-neighbour grid'
    elif scheme == 'quadtree':
        G, feats, dem_idx, thresh = _quadtree_for_count(xv, yv, elev, n)
        info['edge_rule'] = 'quadtree leaf sides'
        info['quadtree_thresh_km'] = thresh
    else:
        if scheme == 'uniform':
            feats, dem_idx = ps.sample_uniform(xv, yv, elev, n, seed=seed)
        elif scheme == 'slope-adaptive':
            feats, dem_idx = ps.sample_slope_adaptive(xv, yv, elev, n, uniform_mix=0.5, seed=seed)
            info['uniform_mix'] = 0.5
        else:
            feats, dem_idx, d = ps.sample_poisson_disk(xv, yv, elev, n_target=n, seed=seed)
            info['poisson_min_dist_km'] = float(d)
        G, feats = pg.build_knn_graph(feats, k=knn_k)
        info['edge_rule'] = f'symmetrized kNN-{knn_k} (xy)'
    info['components_before_bridge'] = nx.number_connected_components(G)
    info['bridges'] = pg.connect_components(G, feats, max_frac=max_bridge_frac)
    w = np.array([d['weight'] for *_, d in G.edges(data=True)])
    info.update(n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges(),
                mean_degree=2 * G.number_of_edges() / G.number_of_nodes(),
                median_edge_km=float(np.median(w)), max_edge_km=float(w.max()))
    return G, np.asarray(feats, dtype=float), np.asarray(dem_idx, dtype=np.int64), info
