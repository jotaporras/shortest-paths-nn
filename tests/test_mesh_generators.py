"""Characterization tests for the triangle-mesh dataset generators.

Covers dataset/process_triangulations.py (CSV input) and
generate_simplified_mesh_dataset.py (whitespace text input).
"""
import importlib.util

import networkx as nx
import numpy as np
import pytest

from conftest import REPO_ROOT, load_dataset_module, synthetic_dem
from golden_util import array_digest, check_golden

SIZE = 5


def tiny_mesh(seed=0, contiguous=False):
    """A triangulated SIZE x SIZE grid with shuffled vertex ids.

    Ids are 0..N-1 when `contiguous`, else spread out with gaps.
    Returns (ids, coords in metres, edges as id pairs).
    """
    rng = np.random.default_rng(seed)
    z = synthetic_dem(SIZE, seed)
    ids = rng.permutation(SIZE * SIZE)
    if not contiguous:
        ids = ids * 3 + 7
    grid_ids = ids.reshape(SIZE, SIZE)
    coords = np.array([[i * 20.0, j * 20.0, z[i, j]] for i in range(SIZE) for j in range(SIZE)])
    edges = []
    for i in range(SIZE):
        for j in range(SIZE):
            if i + 1 < SIZE:
                edges.append((grid_ids[i, j], grid_ids[i + 1, j]))
            if j + 1 < SIZE:
                edges.append((grid_ids[i, j], grid_ids[i, j + 1]))
            if i + 1 < SIZE and j + 1 < SIZE:
                edges.append((grid_ids[i, j], grid_ids[i + 1, j + 1]))
    return ids, coords, edges


@pytest.fixture(scope="module")
def triang():
    return load_dataset_module("process_triangulations")


def write_mesh_csvs(tmp_path, contiguous=False):
    ids, coords, edges = tiny_mesh(contiguous=contiguous)
    nodes = tmp_path / "nodes.csv"
    nodes.write_text("id,x,y,z\n" + "".join(
        f"{i},{x},{y},{z}\n" for i, (x, y, z) in zip(ids, coords)))
    edge_file = tmp_path / "edges.csv"
    edge_file.write_text("a,b\n" + "".join(f"{a},{b}\n" for a, b in edges))
    return nodes, edge_file


@pytest.fixture
def mesh_csvs(tmp_path):
    return write_mesh_csvs(tmp_path)


def test_process_node_features_converts_to_km(triang, mesh_csvs):
    ids, coords, _ = tiny_mesh()
    node_idx_dict, node_feature_lst = triang.process_node_features(mesh_csvs[0])
    assert np.allclose(node_feature_lst, coords / 1000)
    assert node_idx_dict[int(ids[3])] == pytest.approx(list(coords[3] / 1000))


@pytest.mark.parametrize("scale", [False, True])
def test_mesh_graph_weights(triang, mesh_csvs, scale):
    node_idx_dict, _ = triang.process_node_features(mesh_csvs[0])
    edges = triang.process_edges(mesh_csvs[1])
    G = triang.construct_nx_graph(edges, node_idx_dict, scale=scale)
    assert G.number_of_edges() == len(edges)
    for u, v, w in G.edges(data="weight"):
        p1, p2 = np.array(node_idx_dict[u]), np.array(node_idx_dict[v])
        dist = np.linalg.norm(p1 - p2)
        if scale:
            # Same formula as dataset.py --edge-weight; pass 2 may change it.
            dist *= 1 + abs(np.arctan(p1[2] - p2[2]) / np.linalg.norm(p2[:2] - p1[:2]))
        assert w == pytest.approx(dist, rel=1e-12)


def test_process_triangulations_dataset_golden(triang, tmp_path, update_golden):
    """Pins the pairs for a mesh whose vertex ids are 0..N-1."""
    mesh_csvs = write_mesh_csvs(tmp_path, contiguous=True)
    node_idx_dict, node_features = triang.process_node_features(mesh_csvs[0])
    edges = triang.process_edges(mesh_csvs[1])
    G = triang.construct_nx_graph(edges, node_idx_dict)
    np.random.seed(0)
    path = tmp_path / "mesh.npz"
    triang.construct_dataset(G, node_features, str(path), num_srcs=4, samples_per_source=6)
    data = np.load(path)
    observed = {
        "srcs": data["srcs"].tolist(),
        "tars": data["tars"].tolist(),
        "lengths_sum": float(np.sum(data["lengths"])),
        "edge_index_digest": array_digest(data["edge_index"]),
        "distances_sum": float(np.sum(data["distances"])),
    }
    check_golden("mesh_process_triangulations", observed, update_golden,
                 float_keys=("lengths_sum", "distances_sum"))


def test_process_triangulations_noncontiguous_ids_crash(triang, mesh_csvs, tmp_path):
    """Bug for pass 2: sources are drawn as 0..N-1 indices, but graph nodes are raw ids.

    Also, edge_index keeps raw ids while node_features is in file order.
    """
    node_idx_dict, node_features = triang.process_node_features(mesh_csvs[0])
    G = triang.construct_nx_graph(triang.process_edges(mesh_csvs[1]), node_idx_dict)
    np.random.seed(0)
    with pytest.raises(nx.NodeNotFound):
        triang.construct_dataset(G, node_features, str(tmp_path / "x.npz"), 4, 6)


def test_process_triangulations_rejects_other_methods(triang, mesh_csvs, tmp_path):
    node_idx_dict, node_features = triang.process_node_features(mesh_csvs[0])
    G = triang.construct_nx_graph(triang.process_edges(mesh_csvs[1]), node_idx_dict)
    with pytest.raises(NotImplementedError):
        triang.construct_dataset(G, node_features, str(tmp_path / "x.npz"), 1, 1,
                                 sampling_method="hybrid")


@pytest.fixture(scope="module")
def simplified():
    spec = importlib.util.spec_from_file_location(
        "_generate_simplified_mesh_dataset", REPO_ROOT / "generate_simplified_mesh_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simplified_mesh_dataset_golden(simplified, tmp_path, update_golden):
    """generate_dataset draws a fixed 62,500 random pairs, then splits 80/20."""
    ids, coords, edges = tiny_mesh()
    vertex_file = tmp_path / "v.txt"
    np.savetxt(vertex_file, np.column_stack([ids, coords]))
    edge_file = tmp_path / "e.txt"
    np.savetxt(edge_file, np.array(edges, dtype=int), fmt="%d")
    np.random.seed(0)
    simplified.generate_dataset(vertex_file, edge_file, str(tmp_path / "m"))
    full, train, test = (np.load(tmp_path / f"m{s}.npz") for s in ("", "_train", "_test"))
    assert len(full["srcs"]) == 62500
    assert (len(train["srcs"]), len(test["srcs"])) == (50000, 12500)
    observed = {
        "srcs_digest": array_digest(full["srcs"]),
        "tars_digest": array_digest(full["tars"]),
        "train_srcs_digest": array_digest(train["srcs"]),
        "test_tars_digest": array_digest(test["tars"]),
        "lengths_sum": float(np.sum(full["lengths"])),
        "edge_index_digest": array_digest(full["edge_index"]),
        "distances_sum": float(np.sum(full["distances"])),
    }
    check_golden("mesh_simplified", observed, update_golden,
                 float_keys=("lengths_sum", "distances_sum"), rel=1e-9)
