"""DEM -> grid graph -> npz (dataset/dataset.py)."""
import networkx as nx
import numpy as np
import pytest

from conftest import build_terrain_npz, load_dataset_module, synthetic_dem


def grid_graph(size, dem_resolution=10.0, edge_weight=False):
    ds = load_dataset_module("dataset")
    xv, yv, elevation = ds.get_dem_xv_yv_(synthetic_dem(size, seed=0), dem_resolution, visualize=False)
    graph, node_features = ds.construct_nx_graph(xv, yv, elevation, scale=edge_weight)
    return graph, np.array(node_features)


@pytest.mark.parametrize("size", [2, 5, 9])
def test_grid_has_8_neighbour_edges(size):
    graph, node_features = grid_graph(size)
    assert graph.number_of_nodes() == size * size
    assert node_features.shape == (size * size, 3)
    # straight edges + diagonal edges of a king-move grid
    assert graph.number_of_edges() == 2 * size * (size - 1) + 2 * (size - 1) ** 2


def test_edge_weight_is_3d_euclidean_distance():
    graph, node_features = grid_graph(5)
    for u, v, weight in graph.edges(data="weight"):
        assert weight == pytest.approx(np.linalg.norm(node_features[u] - node_features[v]))


def test_edge_weight_flag_uses_current_formula():
    # Characterizes the current --edge-weight formula, (1 + |arctan(dz) / dxy|) * dist.
    # The intended formula is probably arctan(dz / dxy); pass 2 decides and updates this test.
    graph, node_features = grid_graph(5, edge_weight=True)
    for u, v, weight in graph.edges(data="weight"):
        p1, p2 = node_features[u], node_features[v]
        angle = np.abs(np.arctan(p1[2] - p2[2]) / np.linalg.norm(p2[:2] - p1[:2]))
        assert weight == pytest.approx((1 + angle) * np.linalg.norm(p1 - p2))


def test_npz_format_and_lengths_match_dijkstra(tmp_path):
    path = build_terrain_npz(tmp_path / "t.npz", size=7, dem_resolution=10.0, seed=3,
                             num_srcs=4, samples_per_source=10)
    data = np.load(path)
    assert set(data.files) == {"edge_index", "distances", "srcs", "tars", "lengths", "node_features"}
    num_nodes = len(data["node_features"])
    edge_index = data["edge_index"]
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == len(data["distances"])
    assert len(data["srcs"]) == len(data["tars"]) == len(data["lengths"]) == 40

    # every undirected edge is stored in both directions with the same weight
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for (u, v), w in zip(edge_index.T, data["distances"]):
        graph.add_edge(int(u), int(v), weight=float(w))
    assert graph.number_of_edges() * 2 == edge_index.shape[1]

    for src, tar, length in zip(data["srcs"], data["tars"], data["lengths"]):
        expected = nx.dijkstra_path_length(graph, int(src), int(tar))
        assert length == pytest.approx(expected)
