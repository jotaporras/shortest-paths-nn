"""e33 edge-length inputs: the TAGConv edge-weight kernel and the SparseGT euclidean attention bias."""
import torch

from spnn.models.gnn import GNNModel
from spnn.models.sparse_gt import SparseGTWithRPEARL, _csr_pair_distances


def _graph(n=30, e=90, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, generator=g)
    edge_index = torch.randint(0, n, (2, e), generator=g)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.linalg.vector_norm(x[edge_index[0]] - x[edge_index[1]], dim=1).unsqueeze(-1)
    return x, edge_index, edge_attr


def test_tagconv_kernel_changes_output_and_needs_edge_attr():
    x, edge_index, edge_attr = _graph()
    torch.manual_seed(0)
    weighted = GNNModel(input=3, output=8, hidden=16, layers=2, layer_type='TAGConv', K=2,
                        edge_weight_kernel='exp')
    plain = GNNModel(input=3, output=8, hidden=16, layers=2, layer_type='TAGConv', K=2)
    plain.load_state_dict(weighted.state_dict(), strict=False)
    assert not torch.allclose(weighted(x, edge_index, edge_attr=edge_attr),
                              plain(x, edge_index, edge_attr=edge_attr))
    w = weighted.edge_kernel(edge_attr)
    assert torch.all((w > 0) & (w <= 1))
    try:
        weighted(x, edge_index)
    except ValueError:
        pass
    else:
        raise AssertionError("missing edge_attr must raise")


def _sgt(num_hops, attn_bias):
    torch.manual_seed(0)
    return SparseGTWithRPEARL(input_dim=3, hidden_dim=16, output_dim=8, num_layers=2, num_heads=4,
                              num_hops=num_hops, rpearl_num_layers=2, dropout=0.0, attn_dropout=0.0,
                              embedding_mode='data', pe_k=2, attn_bias=attn_bias).eval()


def test_csr_pair_distances_match_edge_lengths():
    x, edge_index, edge_attr = _graph()
    n = x.shape[0]
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (n, n)).coalesce()
    d = _csr_pair_distances(A.to_sparse_csr(), x)
    i, j = A.indices()
    expected = torch.linalg.vector_norm(x[i] - x[j], dim=1)
    assert torch.allclose(d, expected)


def test_zero_slopes_reproduce_unbiased_sgt():
    x, edge_index, edge_attr = _graph()
    for k in (1, 2):
        biased, plain = _sgt(k, 'euclidean'), _sgt(k, 'none')
        plain.load_state_dict(biased.state_dict(), strict=False)
        with torch.no_grad():
            for layer in biased.backbone.attention_layers:
                layer.bias_slope.zero_()
            assert torch.allclose(biased(x, edge_index, edge_attr), plain(x, edge_index, edge_attr), atol=1e-6)
            for layer in biased.backbone.attention_layers:
                layer.bias_slope.fill_(-5.0)
            assert not torch.allclose(biased(x, edge_index, edge_attr), plain(x, edge_index, edge_attr), atol=1e-4)


def test_bias_slopes_get_gradients():
    x, edge_index, edge_attr = _graph()
    model = _sgt(2, 'euclidean').train()
    model(x, edge_index, edge_attr).sum().backward()
    for layer in model.backbone.attention_layers:
        assert layer.bias_slope.grad is not None and torch.any(layer.bias_slope.grad != 0)


def test_physical_scalars_get_no_weight_decay():
    from spnn.training import param_groups
    model = _sgt(2, 'euclidean')
    groups = param_groups(list(model.named_parameters()), 0.1)
    assert [g['weight_decay'] for g in groups] == [0.1, 0.0]
    assert len(groups[1]['params']) == len(model.backbone.attention_layers)
    gnn = GNNModel(input=3, output=8, hidden=16, layers=2, layer_type='TAGConv', K=2, edge_weight_kernel='exp')
    assert len(param_groups(list(gnn.named_parameters()), 0.1)[1]['params']) == 1
    # Models without physical scalars keep a single decayed group, so the e32 recipe is unchanged.
    assert len(param_groups(list(_sgt(2, 'none').named_parameters()), 0.1)) == 1
