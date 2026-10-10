"""DEM -> grid graph -> npz (spnn.graphs.grid, spnn.sampling.train_pairs)."""
import networkx as nx
import numpy as np
import pytest

from conftest import build_terrain_npz, synthetic_dem
from spnn.graphs.grid import construct_nx_graph, construct_nx_graph_4n, get_dem_xv_yv_, slope_weighted_length


def grid_graph(size, dem_resolution=10.0, edge_weight=False):
    xv, yv, elevation = get_dem_xv_yv_(synthetic_dem(size, seed=0), dem_resolution, visualize=False)
    graph, node_features = construct_nx_graph(xv, yv, elevation, scale=edge_weight)
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


def test_edge_weight_flag_scales_by_slope_angle():
    # --edge-weight: (1 + arctan(|dz| / dxy)) * dist, the slope angle in radians.
    graph, node_features = grid_graph(5, edge_weight=True)
    for u, v, weight in graph.edges(data="weight"):
        p1, p2 = node_features[u], node_features[v]
        angle = np.arctan(abs(p1[2] - p2[2]) / np.linalg.norm(p2[:2] - p1[:2]))
        assert 0 <= angle < np.pi / 2
        assert weight == pytest.approx((1 + angle) * np.linalg.norm(p1 - p2))


def test_slope_weight_of_a_45_degree_edge():
    p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 5.0])
    assert slope_weighted_length(p1, p2) == pytest.approx((1 + np.pi / 4) * np.sqrt(50))


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


def test_non_square_dem_graph():
    rows, cols = 4, 7
    dem = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    xv, yv, elevation = get_dem_xv_yv_(dem, 10.0, visualize=False)
    assert xv.shape == yv.shape == (rows, cols)
    assert np.all(np.diff(xv, axis=1) > 0) and np.all(np.diff(yv, axis=0) > 0)
    graph, node_features = construct_nx_graph(xv, yv, elevation)
    assert graph.number_of_nodes() == rows * cols
    king_edges = rows * (cols - 1) + (rows - 1) * cols + 2 * (rows - 1) * (cols - 1)
    assert graph.number_of_edges() == king_edges
    # node index = row * cols + col
    assert node_features[1 * cols + 5][2] == dem[1, 5] / 1000


def test_non_square_4n_graph():
    rows, cols = 5, 3
    xv, yv, elevation = get_dem_xv_yv_(np.zeros((rows, cols)), 10.0, visualize=False)
    graph, _ = construct_nx_graph_4n(xv, yv, elevation)
    assert graph.number_of_edges() == rows * (cols - 1) + (rows - 1) * cols


def test_grid_hops_uses_column_count():
    from spnn.sampling.targets import grid_hops
    rows, cols = 4, 7
    src = 1 * cols + 2  # (1, 2)
    assert grid_hops(src, np.array([3 * cols + 6]), rows * cols, cols)[0] == 2 + 4 + 1
