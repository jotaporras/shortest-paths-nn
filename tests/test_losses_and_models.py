"""Loss functions, model construction and forward passes."""
import pytest
import torch
from torch_geometric.data import Data

import refactor_training as rt
from conftest import requires_sparse_cpu
from src import loss_funcs


def test_basic_losses():
    pred = torch.tensor([1.0, 2.0, 4.0])
    target = torch.tensor([1.0, 3.0, 2.0])
    assert loss_funcs.mse_loss(pred, target).item() == pytest.approx(5 / 3)
    assert loss_funcs.mae_loss(pred, target).item() == pytest.approx(1.0)
    assert loss_funcs.nmae_loss(pred, target).item() == pytest.approx((0 + 1 / 3 + 1) / 3)


def test_nmae_ignores_zero_targets():
    pred = torch.tensor([5.0, 2.0])
    target = torch.tensor([0.0, 4.0])
    assert loss_funcs.nmae_loss(pred, target).item() == pytest.approx(0.5)


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


@pytest.mark.parametrize("layer_type", [
    "TAGConv", "GCNConv", "GATConv", "TransformerConv", "GATv2Conv",
    pytest.param("GINConv", marks=pytest.mark.xfail(
        strict=True, raises=TypeError, reason="pass 2: create_graph_layer passes ints where GINConv needs an nn module")),
])
def test_gnn_model_forward(layer_type):
    model = rt.configure_embedding_module(GNN_CONFIG, layer_type, edge_dim=1)
    graph = tiny_graph()
    out = model(graph.x, graph.edge_index, graph.edge_attr)
    assert out.shape == (12, 6)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("activation", ["relu", "lrelu", "silu"])
def test_activation_names_resolve(activation):
    config = {**GNN_CONFIG, "activation": activation}
    model = rt.configure_embedding_module(config, "TAGConv")
    assert type(model.activation).__name__ == rt.normalize_activation(activation)


def test_tagconv_uses_pyg_default_k():
    # Characterizes the current behavior: K is never passed (baselines.py:43-48),
    # so every TAGConv has PyG's default K=3. Pass 2 makes K configurable.
    model = rt.configure_embedding_module(GNN_CONFIG, "TAGConv")
    assert model.initial.K == 3
    assert all(layer.K == 3 for layer in model.module_list)


def test_unknown_layer_type_fails():
    with pytest.raises(KeyError):
        rt.configure_embedding_module(GNN_CONFIG, "NotALayer")


@requires_sparse_cpu
@pytest.mark.parametrize("mode", ["random", "data"])
def test_sparse_gt_forward(mode):
    config = {**GNN_CONFIG, "sparse_gt": {
        "hidden_dim": 8, "num_layers": 1, "num_heads": 2, "num_hops": 2,
        "rpearl_samples": 3, "rpearl_num_layers": 2, "pe_k": 2,
        "dropout": 0.1, "attn_dropout": 0.1, "embedding_mode": mode}}
    torch.manual_seed(0)
    model = rt.configure_embedding_module(config, "SparseGT")
    model.eval()
    graph = tiny_graph()
    out = model(graph.x, graph.edge_index)
    assert out.shape == (12, 6)
    assert torch.isfinite(out).all()


def test_compute_metrics_with_single_pair_crashes():
    # pass 2: `.squeeze()` at refactor_training.py:155 collapses the (7, 1)
    # predictions array when a dataloader holds exactly one pair.
    model = rt.configure_embedding_module(GNN_CONFIG, "TAGConv")
    graph = tiny_graph()
    dataset = rt.SingleGraphShortestPathDataset(
        torch.tensor([0], dtype=torch.int), torch.tensor([5], dtype=torch.int), torch.tensor([1.0]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    with pytest.raises(ValueError, match="Shape of passed values"):
        rt.compute_metrics(model, None, loader, graph, "cpu", "TAGConv", True, 4, "mse_loss")
