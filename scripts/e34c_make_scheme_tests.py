"""e34c: a full test set for each sampling scheme, on its stride-4 graph (250k nodes).

The grid test set full_test-004 is the stride-4 grid with 10 sources x all 250k targets. The e34 models
of the irregular schemes were tested on that grid, whose octile metric differs from their kNN-8 training
metric. This script builds the same kind of test set on the stride-4 graph of each irregular scheme, taken
from the e31/e34 training set norway_<scheme>_s04.npz (same nodes and edges, same 3D edge lengths).
Sources: for each of the 10 grid sources, the scheme node nearest in xy, so heatmaps stay comparable.

Usage: python scripts/e34c_make_scheme_tests.py [--schemes uniform poisson-disk ...] [--out DIR]
Output: <out>/norway_<scheme>_test-004.npz with the full_test-004 keys
        (edge_index, distances, nodes, srcs, tars, lengths, node_features) and a .json sidecar.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

GRID_TEST = Path('/home/jporras/data/spnn/full_test-004.npz')
TRAIN_DIR = Path('/home/jporras/data/spnn/e31')
SCHEMES = ['uniform', 'poisson-disk', 'slope-adaptive', 'quadtree']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schemes', nargs='+', default=SCHEMES)
    ap.add_argument('--out', default='/home/jporras/data/spnn/e34c')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = np.load(GRID_TEST)
    grid_srcs = np.unique(grid['srcs'])
    grid_xy = grid['node_features'][grid_srcs, :2]

    for scheme in args.schemes:
        t0 = time.time()
        train = np.load(TRAIN_DIR / f'norway_{scheme}_s04.npz')
        X = train['node_features']
        ei, w = train['edge_index'], train['distances']
        n = len(X)
        dist_xy, srcs = cKDTree(X[:, :2]).query(grid_xy)
        A = sp.csr_matrix((w, (ei[0], ei[1])), shape=(n, n))
        L = dijkstra(A, indices=srcs, directed=True)
        if np.isinf(L).any():
            raise SystemExit(f'{scheme}: {np.isinf(L).sum()} unreachable pairs')
        tars = np.tile(np.arange(n), len(srcs))
        src_col = np.repeat(srcs, n)
        np.savez(out / f'norway_{scheme}_test-004.npz', edge_index=ei, distances=w, nodes=np.arange(n),
                 srcs=src_col, tars=tars, lengths=L.ravel(), node_features=X)
        info = dict(scheme=scheme, source_train_set=str(TRAIN_DIR / f'norway_{scheme}_s04.npz'),
                    n_nodes=int(n), n_edges=int(ei.shape[1]), n_pairs=int(L.size),
                    grid_sources=grid_srcs.tolist(), sources=srcs.tolist(),
                    source_xy_offset_km=dist_xy.tolist(), mean_length_km=float(L.mean()),
                    seconds=round(time.time() - t0, 1))
        (out / f'norway_{scheme}_test-004.json').write_text(json.dumps(info, indent=1))
        print(f'{scheme}: {n} nodes, {L.size} pairs, mean length {L.mean():.2f} km, '
              f'max source offset {dist_xy.max() * 1000:.0f} m, {info["seconds"]} s')


if __name__ == '__main__':
    main()
