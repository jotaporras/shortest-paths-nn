"""Precompute and time Laplacian eigenvector encodings for a set of graphs (e35).

    python -m spnn.cli.make_lappe --data ~/data/spnn/e31/norway_*_s??.npz ~/data/spnn/full_test-004.npz \
        --k-max 64 --time-ks 2 4 8 16 32 64 --cache-dir ~/data/spnn/lappe

For every graph: the k_max lowest non-trivial eigenvectors go to the cache that training reads
(spnn.graphs.spectral.laplacian_pe). With --time-ks, each k is also computed on its own and its
wall time and peak memory are stored in the cache sidecar under `timings`, so the cost of the
encoding as a function of k is on record. Graphs that share an edge set share one cache entry.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..graphs.spectral import DEFAULT_CACHE_DIR, DEFAULT_K_MAX, compute_eigenvectors, graph_key, laplacian_pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help='npz files with edge_index and node_features')
    ap.add_argument('--k-max', type=int, default=DEFAULT_K_MAX)
    ap.add_argument('--time-ks', nargs='*', type=int, default=[], help='k values to time one by one')
    ap.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    ap.add_argument('--force-timings', action='store_true', help='Redo timings that are already stored')
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir).expanduser()

    for path in args.data:
        with np.load(path) as d:
            edge_index = d['edge_index']
            n = int(d['node_features'].shape[0])
        key = graph_key(n, edge_index)
        t0 = time.time()
        V, info = laplacian_pe(n, edge_index, args.k_max, cache_dir=cache_dir, k_max=args.k_max, source=path)
        print(f'{Path(path).name}: n={n} key={key} V{V.shape} k_max compute {info["seconds"]} s, '
              f'peak rss {info["peak_rss_gb"]} GB, total {time.time() - t0:.1f} s '
              f'(gap lambda_2-lambda_3 = {info["eigenvalues"][1] - info["eigenvalues"][2]:.2e})', flush=True)
        if not args.time_ks:
            continue
        json_path = cache_dir / f'lappe_{key}.json'
        sidecar = json.loads(json_path.read_text())
        timings = sidecar.setdefault('timings', {})
        for k in args.time_ks:
            if str(k) in timings and not args.force_timings:
                continue
            _, _, tinfo = compute_eigenvectors(n, edge_index, k)
            timings[str(k)] = dict(seconds=tinfo['seconds'], peak_rss_gb=tinfo['peak_rss_gb'], host=tinfo['host'],
                                   threads=tinfo['threads'])
            print(f'  k={k}: {tinfo["seconds"]} s, peak rss {tinfo["peak_rss_gb"]} GB', flush=True)
            json_path.write_text(json.dumps(sidecar, indent=1))


if __name__ == '__main__':
    main()
