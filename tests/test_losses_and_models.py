"""Loss functions, model construction and forward passes."""
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from conftest import requires_sparse_cpu
from spnn import losses
from spnn.data import SingleGraphShortestPathDataset
from spnn.metrics import compute_metrics
from spnn.models.factory import configure_embedding_module, normalize_activation
from spnn.models.gnn import LAYERS


def test_basic_losses():
    pred = torch.tensor([1.0, 2.0, 4.0])
    target = torch.tensor([1.0, 3.0, 2.0])
    assert losses.mse_loss(pred, target).item() == pytest.approx(5 / 3)
    assert losses.mae_loss(pred, target).item() == pytest.approx(1.0)
    assert losses.nmae_loss(pred, target).item() == pytest.approx((0 + 1 / 3 + 1) / 3)


def test_nmae_ignores_zero_targets():
    pred = torch.tensor([5.0, 2.0])
    target = torch.tensor([0.0, 4.0])
    assert losses.nmae_loss(pred, target).item() == pytest.approx(0.5)


def tiny_graph(num_nodes=12, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(num_nodes, 3, generator=gen)
    ring = torch.arange(num_nodes)
    edge_index = torch.stack([ring, (ring + 1) % num_nodes])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.rand(edge_index.shape[1], 1, generator=gen)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


GNN_CONFIG = {"constr": {"input": 3, "hidden": 8, "output": 6, "layers": 2},
              "layer_norm": False, "dropout": True, "activation": "lrelu"}


@pytest.mark.parametrize("layer_type", sorted(LAYERS))
def test_gnn_model_forward(layer_type):
    model = configure_embedding_module(GNN_CONFIG, layer_type, edge_dim=1)
    graph = tiny_graph()
    out = model(graph.x, graph.edge_index, graph.edge_attr)
    assert out.shape == (12, 6)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("activation", ["relu", "lrelu", "silu"])
def test_activation_names_resolve(activation):
    config = {**GNN_CONFIG, "activation": activation}
    model = configure_embedding_module(config, "TAGConv")
    assert type(model.activation).__name__ == normalize_activation(activation)


@pytest.mark.parametrize("k", [None, 2, 5])
def test_tagconv_k_from_config(k):
    # constr.K sets the TAGConv hops of every layer; without it, PyG's default K=3.
    constr = {**GNN_CONFIG["constr"], **({} if k is None else {"K": k})}
    model = configure_embedding_module({**GNN_CONFIG, "constr": constr}, "TAGConv")
    expected = 3 if k is None else k
    assert model.initial.K == expected
    assert all(layer.K == expected for layer in model.module_list)


def test_unknown_layer_type_fails():
    with pytest.raises(KeyError):
        configure_embedding_module(GNN_CONFIG, "NotALayer")


@requires_sparse_cpu
@pytest.mark.parametrize("mode", ["random", "data"])
def test_sparse_gt_forward(mode):
    config = {**GNN_CONFIG, "sparse_gt": {
        "hidden_dim": 8, "num_layers": 1, "num_heads": 2, "num_hops": 2,
        "rpearl_samples": 3, "rpearl_num_layers": 2, "pe_k": 2,
        "dropout": 0.1, "attn_dropout": 0.1, "embedding_mode": mode}}
    torch.manual_seed(0)
    model = configure_embedding_module(config, "SparseGT")
    model.eval()
    graph = tiny_graph()
    out = model(graph.x, graph.edge_index)
    assert out.shape == (12, 6)
    assert torch.isfinite(out).all()


@requires_sparse_cpu
@pytest.mark.parametrize("mode", ["random", "data"])
def test_sparse_gt_every_parameter_gets_a_gradient(mode):
    # Regression: a reentrant checkpoint(lambda: attn_out) call detached the
    # attention output, so Q, K and V never trained.
    config = {**GNN_CONFIG, "sparse_gt": {
        "hidden_dim": 8, "num_layers": 2, "num_heads": 2, "num_hops": 2,
        "rpearl_samples": 3, "rpearl_num_layers": 2, "pe_k": 2,
        "dropout": 0.1, "attn_dropout": 0.1, "embedding_mode": mode}}
    torch.manual_seed(0)
    model = configure_embedding_module(config, "SparseGT")
    model.train()
    graph = tiny_graph()
    out = model(graph.x, graph.edge_index)
    (out * torch.randn_like(out)).sum().backward()
    no_grad = [name for name, param in model.named_parameters()
               if param.requires_grad and param.grad is None]
    assert no_grad == []
    attn = [p for n, p in model.named_parameters() if ".Q.weight" in n or ".V.weight" in n]
    assert attn and all(p.grad.abs().sum() > 0 for p in attn)


def test_compute_metrics_with_single_pair():
    # A dataloader with exactly one pair gives a one-row predictions table.
    model = configure_embedding_module(GNN_CONFIG, "TAGConv")
    graph = tiny_graph()
    dataset = SingleGraphShortestPathDataset(
        torch.tensor([0], dtype=torch.int), torch.tensor([5], dtype=torch.int), torch.tensor([1.0]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    metrics = compute_metrics(model, None, loader, graph, "cpu", "TAGConv", True, 4, "mse_loss")
    df = metrics["all_preds_df"]
    assert df.shape == (1, 7)
    assert (df["srcs"].iloc[0], df["tars"].iloc[0], df["lengths"].iloc[0]) == (0, 5, 1.0)


def test_compute_metrics_skips_zero_length_pairs():
    # A src == tar pair (length 0) has no relative error: NaN in the table, and it
    # does not count in nmae or the acc_* denominators. A batch of only such pairs
    # must not turn nmae into NaN.
    model = configure_embedding_module(GNN_CONFIG, "TAGConv")
    graph = tiny_graph()
    srcs, tars = torch.tensor([0, 3, 1], dtype=torch.int), torch.tensor([0, 5, 7], dtype=torch.int)
    lengths = torch.tensor([0.0, 1.0, 2.0])
    loader = torch.utils.data.DataLoader(SingleGraphShortestPathDataset(srcs, tars, lengths), batch_size=1)
    metrics = compute_metrics(model, None, loader, graph, "cpu", "TAGConv", True, 4, "mse_loss")
    df = metrics["all_preds_df"]
    assert np.isnan(df["pred_nmaes"].iloc[0])
    expected = df["pred_nmaes"].iloc[1:].mean()
    assert metrics["nmae"] == pytest.approx(expected)
    acc = (df["pred_nmaes"].iloc[1:] < 0.05).mean()
    assert metrics["acc_5pct"] == pytest.approx(acc)
