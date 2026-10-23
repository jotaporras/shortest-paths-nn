"""Laplacian eigenvector encodings (e35): spectral helpers, cache and the SparseGT lappe mode."""
import json

import numpy as np
import pytest
import torch

from spnn.graphs.spectral import compute_eigenvectors, graph_key, laplacian_pe, normalized_adjacency
from spnn.models.factory import configure_embedding_module


def grid_edges(rows, cols):
    """Directed 4-neighbour grid edge_index (both directions), plus a duplicate and a self loop."""
    idx = np.arange(rows * cols).reshape(rows, cols)
    src = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    dst = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    extra = np.array([[0, 5], [1, 5]])          # duplicate of (0,1) reversed and a self loop
    return np.concatenate([ei, extra], axis=1)


def test_graph_key_ignores_direction_duplicates_and_self_loops():
    ei = grid_edges(5, 6)
    n = 30
    key = graph_key(n, ei)
    assert len(key) == 16
    assert graph_key(n, ei[::-1]) == key                 # reversed directions
    assert graph_key(n, ei[:, :-2]) == key               # without the duplicate and the self loop
    assert graph_key(n, ei[:, np.random.default_rng(0).permutation(ei.shape[1])]) == key
    assert graph_key(n + 1, ei) != key


def test_normalized_adjacency_is_symmetric_binary_without_self_loops():
    ei = grid_edges(4, 4)
    An, n_edges = normalized_adjacency(16, ei)
    assert n_edges == 24
    assert abs(An - An.T).max() == 0
    assert An.diagonal().max() == 0
    A = (An != 0).astype(float)
    assert A.sum() == 48


def test_compute_eigenvectors_orthogonal_sorted_and_scaled():
    rows, cols = 12, 15
    n = rows * cols
    ei = grid_edges(rows, cols)
    k = 6
    V, lam, info = compute_eigenvectors(n, ei, k)
    assert V.shape == (n, k) and V.dtype == np.float32
    assert lam.shape == (k + 1,)
    assert abs(lam[0]) < 1e-6 and np.all(np.diff(lam) >= -1e-9)      # trivial first, ascending
    G = V.T @ V / n
    assert np.allclose(G, np.eye(k), atol=1e-3)                      # orthonormal after the sqrt(n) scaling
    # Rayleigh quotient of each column equals its eigenvalue of L = I - An.
    An, _ = normalized_adjacency(n, ei)
    Vd = V.astype(np.float64)
    rq = 1.0 - np.einsum('ik,ik->k', Vd, An @ Vd) / np.einsum('ik,ik->k', Vd, Vd)
    assert np.allclose(rq, lam[1:], atol=1e-5)
    assert info['n'] == n and info['k'] == k and info['seconds'] >= 0 and 'host' in info
    assert len(info['eigenvalues']) == k + 1


def test_laplacian_pe_cache_roundtrip_and_prefix(tmp_path):
    rows, cols = 10, 10
    n = rows * cols
    ei = grid_edges(rows, cols)
    V8, info = laplacian_pe(n, ei, 8, cache_dir=tmp_path, k_max=8, source='a.npz')
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [f'lappe_{info["key"]}.json', f'lappe_{info["key"]}.npz']
    V4, info4 = laplacian_pe(n, ei, 4, cache_dir=tmp_path, source='b.npz')
    assert np.array_equal(V4, V8[:, :4])                               # prefix of the cached file
    assert info4['cache'] == info['cache'] and info4['k_used'] == 4
    sidecar = json.loads((tmp_path / f'lappe_{info["key"]}.json').read_text())
    assert sidecar['sources'] == ['a.npz', 'b.npz']
    # A k above the cached width triggers a recomputation with the larger width.
    V12, _ = laplacian_pe(n, ei, 12, cache_dir=tmp_path, k_max=8)
    assert V12.shape == (n, 12)


def lappe_config(k):
    return {
        'constr': {'input': 3, 'hidden': 16, 'output': 16, 'layers': 2},
        'layer_norm': False, 'dropout': False, 'activation': 'lrelu',
        'sparse_gt': {'hidden_dim': 16, 'num_layers': 1, 'num_heads': 2, 'num_hops': 2,
                      'embedding_mode': 'lappe', 'lappe_k': k, 'dropout': 0.0, 'attn_dropout': 0.0},
    }


def test_sparse_gt_lappe_forward_and_sign_flip_invariance_in_eval():
    torch.manual_seed(0)
    rows, cols = 6, 7
    n = rows * cols
    ei = grid_edges(rows, cols)
    k = 4
    V, _, _ = compute_eigenvectors(n, ei, k)
    model = configure_embedding_module(lappe_config(k), 'SparseGT', edge_dim=1, new=True)
    xyz = torch.rand(n, 3)
    x = torch.cat([xyz, torch.from_numpy(V)], dim=1)
    edge_index = torch.from_numpy(ei).long()
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        assert out.shape == (n, 16)
        assert torch.isfinite(out).all()
        # Eval mode applies no sign flip: the output is deterministic.
        assert torch.allclose(model(x, edge_index), out)
    assert model.get_config_for_wandb()['sparse_gt_lappe_k'] == k
    with pytest.raises(ValueError, match='eigenvector'):
        model(xyz, edge_index)


def test_sparse_gt_lappe_training_sign_flip_is_random():
    torch.manual_seed(0)
    k = 8
    model = configure_embedding_module(lappe_config(k), 'SparseGT', edge_dim=1, new=True)
    pe = model.positional_encoding
    model.train()
    x = torch.ones(5, k)
    outs = {tuple(torch.sign(pe(x) @ torch.ones(16)).tolist()) for _ in range(20)}
    assert len(outs) > 1
