"""Spectral analysis of the shortest-path-distance signal for section 4 of e34_transfer_suite.ipynb.

Runs on plaza (dense eigh of the 28k-node stride-12 graphs, 20-30 GB each). The notebook reads the output.

    python notebooks/e34_spectral.py embed --scheme grid --stride 12          # test-graph embeddings, GPU
    python notebooks/e34_spectral.py eig --scheme uniform --stride 12 --device cuda --dtype float32   # eigh + spectra

Per scheme, `eig` writes $E34_RUNS/analysis/cache/spec_<scheme>_s<stride>.npz with
  lam         all eigenvalues of the normalized Laplacian (edge weights 1/w, w = 3D edge length)
  src_test    source node ids on the test grid; src: the nearest node of the scheme graph
  d           d_s on the scheme graph (Dijkstra, weights w), one row per source
  energy      squared GFT coefficients of d_s, one row per source, same order as lam
  energy_dw   the same for D^1/2 d_s (random-walk convention, see eig); use this one to compare schemes
and for the grid graph, whose nodes are test-grid nodes,
  err_<model>            signed relative error (d_hat - d) / d of the grid-s<stride> models, at the grid nodes
  err_energy_<model>     its squared GFT coefficients; err_energy_dw_<model>: the same for D^1/2 err
  V_low               the lowest --k-vectors eigenvectors (float32), for other signals in the notebook
"""
import argparse
import os
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

RUNS = Path(os.environ.get('E34_RUNS', '~/runs/e34')).expanduser()
DATA = Path(os.environ.get('E34_DATA', '~/data/spnn/e31')).expanduser()
TEST = Path(os.environ.get('E34_TEST', '~/data/spnn/full_test-004.npz')).expanduser()
REPO = Path(os.environ.get('SPNN_REPO', Path(__file__).resolve().parent.parent)).expanduser()
CACHE = RUNS / 'analysis' / 'cache'

MODELS = ['mlp', 'tagconv', 'sgt2']
MODEL_SPEC = {
    'mlp':     ('MLP',      'configs/e31-mlp.yml'),
    'tagconv': ('TAGConv',  'configs/e31-tagconv.yml'),
    'sgt2':    ('SparseGT', 'configs/e32-sparse-gt-k2.yml'),
}
BORDER_KM = 1.5


def load_graph(path):
    t = np.load(path)
    return t['node_features'], t['edge_index'], t['distances']


def normalized_laplacian(n, ei, w):
    # L = I - D^-1/2 A D^-1/2 with A_ij = 1 / w_ij, so that the 1-Lipschitz signal d_s has bounded Dirichlet energy.
    A = sp.csr_matrix((1.0 / w, (ei[0], ei[1])), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(deg))
    return (sp.eye(n) - Dm @ A @ Dm).tocsr(), deg


def pick_sources(X, n, seed):
    lo, hi = X[:, :2].min(0), X[:, :2].max(0)
    border = np.minimum(X[:, :2] - lo, hi - X[:, :2]).min(1)
    return np.random.default_rng(seed).choice(np.flatnonzero(border >= BORDER_KM), size=n, replace=False)


def embed(scheme, stride, models):
    import sys
    import torch
    import yaml
    sys.path.insert(0, str(REPO))
    from spnn.models.factory import configure_embedding_module

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, ei, w = load_graph(TEST)
    x = torch.tensor(X, dtype=torch.float, device=device)
    for model in models:
        path = CACHE / f'emb_{scheme}_s{stride:02d}_{model}.npy'
        if path.exists():
            print('exists', path)
            continue
        layer_type, cfg_path = MODEL_SPEC[model]
        cfg = next(iter(yaml.safe_load(open(REPO / cfg_path)).values()))
        if layer_type == 'SparseGT':
            cfg['gnn'].setdefault('sparse_gt', {})['embedding_mode'] = 'data'
        net = configure_embedding_module(cfg['gnn'], layer_type, edge_dim=1, new=True)
        weights = next((RUNS / f'norway_{scheme}_s{stride:02d}_{model}').glob('models/**/final_model.pt'))
        net.load_state_dict(torch.load(weights, map_location='cpu'))
        net = net.to(device).eval()
        with torch.no_grad():
            if layer_type == 'MLP':
                e = net(x)
            else:
                e = net(x, torch.tensor(ei, dtype=torch.long, device=device),
                        edge_attr=torch.tensor(w, dtype=torch.float, device=device).unsqueeze(-1))
        np.save(path, e.cpu().numpy().astype(np.float16))
        print('wrote', path, e.shape)
        del net, e
        if device == 'cuda':
            torch.cuda.empty_cache()


def dense_eigh(L, device, dtype):
    """All eigenpairs of the sparse symmetric L. Returns (lam as numpy, V, project) with project(S) = S @ V as numpy.

    CPU: float64 LAPACK, about 1 h per 28k-node graph with 9 threads on plaza.
    GPU: torch.linalg.eigh, float32 about 2 min and float64 about 6 min on an A6000 (timed at 8k nodes, scaled n^3).
    """
    if device == 'cpu':
        lam, V = scipy.linalg.eigh(L.toarray(), driver='evd', overwrite_a=True, check_finite=False)
        return lam, V, lambda S: S @ V
    import torch
    A = torch.from_numpy(L.toarray().astype(dtype)).to(device)
    lam, V = torch.linalg.eigh(A)
    del A
    torch.cuda.empty_cache()

    def project(S):
        return (torch.from_numpy(S.astype(dtype)).to(device) @ V).cpu().numpy()
    return lam.cpu().numpy().astype(np.float64), V, project


def eig(scheme, stride, n_sources, seed, k_vectors, err_strides, device='cpu', dtype='float64'):
    X, ei, w = load_graph(DATA / f'norway_{scheme}_s{stride:02d}.npz')
    n = len(X)
    L, deg = normalized_laplacian(n, ei, w)
    print(f'{scheme} s{stride}: {n} nodes, {len(w)} edges; eigh on {device} {dtype} ...', flush=True)
    lam, V, project = dense_eigh(L, device, dtype)
    print(f'eigh done, lambda in [{lam[0]:.2e}, {lam[-1]:.4f}]', flush=True)

    Xt = np.load(TEST)['node_features']
    src_test = pick_sources(Xt, n_sources, seed)
    src = cKDTree(X[:, :2]).query(Xt[src_test, :2])[1]
    G = sp.csr_matrix((w, (ei[0], ei[1])), shape=(n, n))
    D = dijkstra(G, indices=src)                              # n_sources x n
    # energy: squared coefficients <u_k, d_s>. energy_dw: squared coefficients <u_k, D^1/2 d_s>, the GFT of the
    # random-walk Laplacian (same eigenvalues). Without the D^1/2 scaling the degree pattern of an irregular graph
    # spreads a spatially smooth signal over the whole spectrum (Rayleigh quotient 0.014 on uniform vs 0.0013 on grid).
    sq = np.sqrt(deg)
    out = dict(lam=lam, deg=deg, src_test=src_test, src=src, src_xy=X[src, :2],
               d=D.astype(np.float32), energy=(project(D) ** 2).astype(np.float32),
               energy_dw=(project(D * sq) ** 2).astype(np.float32))

    if scheme == 'grid':
        dist, idx = cKDTree(Xt[:, :2]).query(X[:, :2])
        assert dist.max() < 1e-6, 'grid nodes must be test-grid nodes'
        _, eit, wt = load_graph(TEST)
        Gt = sp.csr_matrix((wt, (eit[0], eit[1])), shape=(len(Xt), len(Xt)))
        Dt = dijkstra(Gt, indices=src_test)                   # n_sources x N_test
        for est in err_strides:
            for model in MODELS:
                path = CACHE / f'emb_grid_s{est:02d}_{model}.npy'
                if not path.exists():
                    print('no embeddings', path)
                    continue
                E = np.load(path).astype(np.float32)
                err = np.empty((n_sources, n), dtype=np.float64)
                for r, s in enumerate(src_test):
                    pred = np.abs(E - E[s]).sum(1)
                    e = (pred - Dt[r]) / np.maximum(Dt[r], 1e-9)
                    e[s] = 0.0                                # the source itself: 0/0, set to 0 for the projection
                    err[r] = e[idx]
                tag = f'{model}' if est == stride else f'{model}_s{est}'
                out[f'err_{tag}'] = err.astype(np.float32)
                out[f'err_energy_{tag}'] = (project(err) ** 2).astype(np.float32)
                out[f'err_energy_dw_{tag}'] = (project(err * sq) ** 2).astype(np.float32)
                print(f'error spectrum {tag}: mean |err| {np.abs(err).mean():.4f}', flush=True)
        V_low = V[:, :k_vectors]
        out['V_low'] = (V_low if device == 'cpu' else V_low.cpu().numpy()).astype(np.float32)

    path = CACHE / f'spec_{scheme}_s{stride:02d}.npz'
    np.savez(path, **out)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('embed')
    p.add_argument('--scheme', default='grid')
    p.add_argument('--stride', type=int, default=12)
    p.add_argument('--models', nargs='+', default=MODELS)
    p = sub.add_parser('eig')
    p.add_argument('--scheme', required=True)
    p.add_argument('--stride', type=int, default=12)
    p.add_argument('--n-sources', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--k-vectors', type=int, default=2000)
    p.add_argument('--err-strides', type=int, nargs='+', default=[12])
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', default='float64', choices=['float32', 'float64'])
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'embed':
        embed(a.scheme, a.stride, a.models)
    else:
        eig(a.scheme, a.stride, a.n_sources, a.seed, a.k_vectors, a.err_strides, a.device, a.dtype)
