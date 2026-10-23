"""Laplacian eigenvector positional encodings (e35).

The k lowest non-trivial eigenvectors of the symmetric normalised Laplacian
L = I - D^-1/2 A D^-1/2 of the unweighted, symmetrised graph without self loops (the same
operator TAGConv propagates with). They are computed with ARPACK on the normalised
adjacency (largest algebraic eigenvalues of D^-1/2 A D^-1/2 = smallest of L) and scaled
by sqrt(n), so entries are O(1) on every graph size and the only thing that changes
between graphs is the basis itself (signs, and rotations inside near-degenerate
eigenspaces). The cost of the computation is recorded next to the cache.

Cache: <cache_dir>/lappe_<key>.npz with V (n, k_max) float32 and the k_max + 1 lowest
eigenvalues, and lappe_<key>.json with the cost record. The key is a hash of the
undirected edge set, so a training set and a test set on the same graph share one
file and get identical encodings.
"""
import hashlib
import json
import logging
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.environ.get('SPNN_LAPPE_CACHE', '~/data/spnn/lappe')).expanduser()
DEFAULT_K_MAX = 64


def undirected_pairs(n, edge_index):
    """Sorted unique codes a * n + b of the undirected edges (a < b), self loops dropped."""
    ei = np.asarray(edge_index, dtype=np.int64)
    a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    keep = a != b
    return np.unique(a[keep] * n + b[keep])


def graph_key(n, edge_index):
    """16-hex-digit hash of (n, undirected edge set)."""
    pairs = undirected_pairs(n, edge_index)
    h = hashlib.sha1()
    h.update(np.int64(n).tobytes())
    h.update(pairs.tobytes())
    return h.hexdigest()[:16]


def normalized_adjacency(n, edge_index):
    """D^-1/2 A D^-1/2 of the binary symmetrised graph without self loops, as CSR."""
    pairs = undirected_pairs(n, edge_index)
    a, b = pairs // n, pairs % n
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    if np.any(deg == 0):
        raise ValueError(f'{int((deg == 0).sum())} isolated nodes; the normalised Laplacian needs degree > 0')
    dinv = sp.diags(1.0 / np.sqrt(deg))
    return (dinv @ A @ dinv).tocsr(), int(len(pairs))


def _peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # Linux: kB


def compute_eigenvectors(n, edge_index, k, tol=1e-6):
    """k lowest non-trivial eigenvectors of L, scaled by sqrt(n), and the cost record.

    Returns V (n, k) float32, eigenvalues of L (k + 1, ascending, the first is the trivial 0)
    and an info dict (n, edges, k, tol, seconds, peak_rss_gb, threads, host, scipy version).
    """
    An, n_edges = normalized_adjacency(n, edge_index)
    rss_before = _peak_rss_gb()
    t0 = time.time()
    lam, V = eigsh(An, k=k + 1, which='LA', tol=tol)
    seconds = time.time() - t0
    order = np.argsort(-lam)                     # largest of A_norm first = smallest of L first
    lam_L = 1.0 - lam[order]
    V = V[:, order]
    # Fix a sign convention per vector (largest-magnitude entry positive) so a cached file is
    # reproducible; the model still sees random sign flips in training.
    idx = np.argmax(np.abs(V), axis=0)
    V = V * np.sign(V[idx, np.arange(V.shape[1])])[None, :]
    V = (V[:, 1:] * np.sqrt(n)).astype(np.float32)
    info = dict(n=int(n), undirected_edges=n_edges, k=int(k), tol=tol, method='scipy eigsh LA on D^-1/2 A D^-1/2',
                seconds=round(seconds, 2), peak_rss_gb=round(_peak_rss_gb(), 3),
                peak_rss_before_gb=round(rss_before, 3), threads=os.cpu_count(), host=platform.node(),
                scipy_version=scipy.__version__, eigenvalues=[float(x) for x in lam_L])
    return V, lam_L, info


def laplacian_pe(n, edge_index, k, cache_dir=None, k_max=None, source=None):
    """V[:, :k] for the graph, from the cache or computed with k_max columns and cached.

    source: optional path or label of the dataset, appended to the cache sidecar's `sources`.
    Returns V (n, k) float32 and a dict with the cost record of the cached computation.
    """
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser()
    k_max = max(k, k_max or DEFAULT_K_MAX)
    key = graph_key(n, edge_index)
    npz_path, json_path = cache_dir / f'lappe_{key}.npz', cache_dir / f'lappe_{key}.json'
    info = None
    if npz_path.exists():
        with np.load(npz_path) as f:
            V = f['V']
        if V.shape[1] >= k:
            info = json.loads(json_path.read_text()) if json_path.exists() else {}
            logger.info('Laplacian PE from cache %s (%d of %d vectors)', npz_path, k, V.shape[1])
    if info is None:
        logger.info('Computing %d Laplacian eigenvectors of a %d-node graph (key %s)', k_max, n, key)
        V, lam, info = compute_eigenvectors(n, edge_index, k_max)
        info['key'] = key
        info['sources'] = []
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, V=V, eigenvalues=lam)
        json_path.write_text(json.dumps(info, indent=1))
    if source is not None and str(source) not in info.get('sources', []):
        info.setdefault('sources', []).append(str(source))
        json_path.write_text(json.dumps(info, indent=1))
    out = {kk: v for kk, v in info.items() if kk != 'sources'}
    out['k_used'] = int(k)
    out['cache'] = str(npz_path)
    return V[:, :k].copy(), out
