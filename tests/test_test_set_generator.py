"""Characterization tests for the test-set generator (spnn.sampling.test_pairs).

This code builds the held-out test terrains. Pass 1 merged its grid code into
spnn.graphs.grid, so these tests pin its edge order and pairs.
"""
import numpy as np
import pytest

from conftest import synthetic_dem
from spnn.graphs.grid import construct_nx_graph, construct_nx_graph_4n, get_dem_xv_yv_, to_pyg_graph
from spnn.sampling import test_pairs
from golden_util import array_digest, check_golden

SIZE = 7
DEM_RESOLUTION = 10.0


@pytest.fixture(scope="module")
def grid():
    return get_dem_xv_yv_(synthetic_dem(SIZE, 3), DEM_RESOLUTION, visualize=False)


@pytest.mark.parametrize("triangles", [False, True])
def test_grid_graph_golden(grid, triangles, update_golden):
    """Pins the edge order of the 4-neighbour grid, with and without diagonals."""
    G, node_features = construct_nx_graph_4n(*grid, triangles=triangles)
    edges, distances = to_pyg_graph(G)
    observed = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "edge_index_digest": array_digest(edges),
        "distances_sum": float(np.sum(distances)),
        "node_features_sum": float(np.sum(node_features)),
    }
    check_golden(f"testgen_grid_triangles{int(triangles)}", observed, update_golden,
                 float_keys=("distances_sum", "node_features_sum"))


def test_triangles_match_eight_neighbour_edge_set(grid):
    """With --triangles the test graph has the same edges as dataset.py's 8-neighbour grid."""
    G_test, _ = construct_nx_graph_4n(*grid, triangles=True)
    G_train, _ = construct_nx_graph(*grid)
    as_set = lambda G: {frozenset(e) for e in G.edges()}
    assert as_set(G_test) == as_set(G_train)
    for u, v, w in G_test.edges(data="weight"):
        assert w == pytest.approx(G_train[u][v]["weight"], rel=1e-12)


def test_construct_pyg_dataset_golden(grid, tmp_path, update_golden):
    G, node_features = construct_nx_graph_4n(*grid, triangles=True)
    np.random.seed(0)
    path = tmp_path / "test.npz"
    test_pairs.construct_pyg_dataset(G, node_features, str(path), size=50, m=SIZE)
    data = np.load(path)
    assert set(data.files) == {"edge_index", "distances", "nodes", "srcs", "tars",
                               "lengths", "node_features"}
    observed = {
        "srcs": data["srcs"].tolist(),
        "tars_digest": array_digest(data["tars"]),
        "num_pairs": int(len(data["tars"])),
        "lengths_sum": float(np.sum(data["lengths"])),
        "nodes": data["nodes"].tolist(),
    }
    check_golden("testgen_construct_pyg_dataset", observed, update_golden,
                 float_keys=("lengths_sum",))


def test_single_src_dataset_keeps_source_as_target(grid):
    """Every source appears once as its own target with length 0.

    Pinned because the training samplers drop or should drop these pairs.
    """
    G, node_features = construct_nx_graph_4n(*grid, triangles=True)
    np.random.seed(0)
    srcs, tars, lengths = test_pairs.single_src_dataset(G, node_features, None)
    assert len(srcs) == 10 * SIZE * SIZE
    zero = [(s, t) for s, t, l in zip(srcs, tars, lengths) if l == 0]
    assert len(zero) == 10 and all(s == t for s, t in zero)


@pytest.mark.parametrize("technique", ["random", "height-sensitive-random"])
def test_generate_src_tar_pairs_golden(grid, technique, update_golden):
    _, node_features = construct_nx_graph_4n(*grid)
    np.random.seed(0)
    srcs, tars = test_pairs.generate_src_tar_pairs(node_features, SIZE, SIZE, size=40,
                                            sampling_technique=technique)
    observed = {"srcs": np.asarray(srcs).tolist(), "tars": np.asarray(tars).tolist()}
    check_golden(f"testgen_pairs_{technique}", observed, update_golden)


@pytest.mark.parametrize("rows,cols", [(SIZE, SIZE), (SIZE, SIZE + 3)])
def test_generate_src_tar_pairs_expanding_radius(rows, cols):
    """Each target lies on one of the rings around its source, inside the grid."""
    node_features = np.zeros((rows * cols, 3))
    radii = (1, 2, 3)
    np.random.seed(0)
    srcs, tars = test_pairs.generate_src_tar_pairs(node_features, rows, cols, size=24,
                                                   sampling_technique="expanding-radius",
                                                   radii=radii, num_per_radius=4)
    assert len(srcs) == len(tars) > 0
    for s, t in zip(srcs, tars):
        (sr, sc), (tr, tc) = divmod(int(s), cols), divmod(int(t), cols)
        assert abs(sr - tr) + abs(sc - tc) in radii


def test_get_ring():
    assert sorted(test_pairs.get_ring((0, 0), 1)) == [(-1, 0), (0, -1), (0, 1), (1, 0)]
    assert len(test_pairs.get_ring((5, 5), 3)) == 12


def test_get_neighbors_is_l1_ball():
    pts = test_pairs.get_neighbors((0, 0), n=2)
    assert len(pts) == 13
    assert all(abs(x) + abs(y) <= 2 for x, y in pts)
