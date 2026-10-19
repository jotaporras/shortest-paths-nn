"""Tests for spnn.graphs.schemes and make_dataset --scheme (experiment e31)."""
import json
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

from conftest import synthetic_dem
from spnn.graphs import point_graph as pg
from spnn.graphs import point_sampling as ps
from spnn.graphs.grid import construct_nx_graph, get_dem_xv_yv_
from spnn.graphs.schemes import SCHEMES, build_scheme_graph, grid_node_count

SIZE = 64


@pytest.fixture(scope="module")
def dem():
    return get_dem_xv_yv_(synthetic_dem(SIZE, 0), 10.0, visualize=False)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("stride", [4, 8])
def test_scheme_graph_is_connected_and_matches_grid_count(dem, scheme, stride):
    xv, yv, z = dem
    G, feats, dem_idx, info = build_scheme_graph(xv, yv, z, scheme, stride, seed=0)
    n = grid_node_count(z.shape, stride)
    assert nx.is_connected(G)
    assert G.number_of_nodes() == len(feats) == len(dem_idx)
    # Poisson-disk and quadtree are calibrated, so N is close but not exact.
    assert abs(G.number_of_nodes() - n) <= (0 if scheme in ('grid', 'uniform', 'slope-adaptive') else 0.1 * n)
    # Every node is a real DEM cell.
    np.testing.assert_allclose(feats[:, 2], z.ravel()[dem_idx])
    # Weights are 3D lengths.
    for a, b, d in list(G.edges(data=True))[:50]:
        assert d['weight'] == pytest.approx(np.linalg.norm(feats[a] - feats[b]))


def test_grid_scheme_is_the_make_dataset_grid(dem):
    xv, yv, z = dem
    G, feats, _, _ = build_scheme_graph(xv, yv, z, 'grid', 4)
    H, hf = construct_nx_graph(xv[::4, ::4], yv[::4, ::4], z[::4, ::4], p=2, scale=False)
    assert set(map(frozenset, G.edges())) == set(map(frozenset, H.edges()))


def test_knn8_equals_8_neighbour_grid_in_the_interior(dem):
    xv, yv, z = dem
    feats, _ = ps.sample_regular_grid(xv, yv, z, 4)
    K, _ = pg.build_knn_graph(feats, k=8)
    H, _ = construct_nx_graph(xv[::4, ::4], yv[::4, ::4], z[::4, ::4], p=2, scale=False)
    side = z[::4, ::4].shape[0]
    # Edge and corner nodes have fewer than 8 grid neighbours, so their kNN-8 reaches
    # inward (a corner reaches the cell 2 rows and 2 columns in). Symmetrization adds
    # those edges to the inner node too, so equality holds 3 or more cells from the edge.
    interior = [r * side + c for r in range(3, side - 3) for c in range(3, side - 3)]
    for u in interior:
        assert set(K[u]) == set(H[u])


def test_connect_components_bridges_to_the_nearest_node():
    feats = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0.0]])
    G, _ = pg._weighted_graph(feats, [(0, 1), (0, 2), (1, 2), (3, 4)])
    assert pg.connect_components(G, feats, max_frac=0.5) == 1
    assert nx.is_connected(G) and G.has_edge(1, 3)
    assert G[1][3]['weight'] == pytest.approx(9.0)


def test_connect_components_refuses_large_splits():
    feats = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0.0]])
    G, _ = pg._weighted_graph(feats, [(0, 1), (2, 3)])
    with pytest.raises(ValueError):
        pg.connect_components(G, feats, max_frac=0.01)


def test_make_dataset_scheme_end_to_end(tmp_path):
    raw = tmp_path / "dem.txt"
    arr = synthetic_dem(40, 1)
    raw.write_text("40 40\n" + "\n".join(" ".join(f"{v:.3f}" for v in row) for row in arr) + "\n")
    out = tmp_path / "uniform.npz"
    cmd = [sys.executable, "-m", "spnn.cli.make_dataset", "--name", "norway", "--raw-data", str(raw),
           "--filename", str(out), "--graph-resolution", "4", "--dataset-size", "60",
           "--num-sources", "6", "--sampling-technique", "single-source-random",
           "--scheme", "uniform", "--seed", "3"]
    subprocess.run(cmd, check=True, capture_output=True)
    data = np.load(out)
    info = json.loads((tmp_path / "uniform.json").read_text())
    assert info["n_nodes"] == len(data["node_features"]) == len(data["dem_idx"]) == 100
    assert len(data["srcs"]) == 60
    # Lengths are Dijkstra distances on the saved graph.
    G = nx.Graph()
    G.add_weighted_edges_from(zip(data["edge_index"][0], data["edge_index"][1], data["distances"]))
    s, t, length = int(data["srcs"][0]), int(data["tars"][0]), float(data["lengths"][0])
    assert nx.dijkstra_path_length(G, s, t) == pytest.approx(length)
