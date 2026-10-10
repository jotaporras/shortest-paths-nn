"""Characterization tests for the triangle-mesh dataset generators.

Covers spnn.graphs.mesh with spnn.cli.make_mesh_dataset (CSV input) and
spnn.cli.make_simplified_mesh_dataset (whitespace text input).
"""
import networkx as nx
import numpy as np
import pytest

from conftest import synthetic_dem
from golden_util import array_digest, check_golden
from spnn.cli import make_simplified_mesh_dataset as simplified
from spnn.graphs.mesh import construct_mesh_graph, process_edges, process_node_features
from spnn.sampling.train_pairs import construct_dataset

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


def test_process_node_features_converts_to_km(mesh_csvs):
    ids, coords, _ = tiny_mesh()
    node_idx_dict, node_feature_lst = process_node_features(mesh_csvs[0])
    assert np.allclose(node_feature_lst, coords / 1000)
    assert node_idx_dict[int(ids[3])] == pytest.approx(list(coords[3] / 1000))


@pytest.mark.parametrize("scale", [False, True])
def test_mesh_graph_weights(mesh_csvs, scale):
    node_idx_dict, _ = process_node_features(mesh_csvs[0])
    edges = process_edges(mesh_csvs[1])
    G = construct_mesh_graph(edges, node_idx_dict, scale=scale)
    assert G.number_of_edges() == len(edges)
    coords = list(node_idx_dict.values())
    for u, v, w in G.edges(data="weight"):
        p1, p2 = np.array(coords[u]), np.array(coords[v])
        dist = np.linalg.norm(p1 - p2)
        if scale:
            # Same formula as make_dataset --edge-weight.
            dist *= 1 + np.arctan(abs(p1[2] - p2[2]) / np.linalg.norm(p2[:2] - p1[:2]))
        assert w == pytest.approx(dist, rel=1e-12)


def test_process_triangulations_dataset_golden(tmp_path, update_golden):
    """Pins the pairs for a mesh with shuffled, gapped vertex ids."""
    mesh_csvs = write_mesh_csvs(tmp_path)
    node_idx_dict, node_features = process_node_features(mesh_csvs[0])
    edges = process_edges(mesh_csvs[1])
    G = construct_mesh_graph(edges, node_idx_dict)
    np.random.seed(0)
    path = tmp_path / "mesh.npz"
    construct_dataset(G, node_features, str(path), sampling_method="single-source-random",
                      num_srcs=4, samples_per_source=6)
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


@pytest.mark.parametrize("contiguous", [False, True])
def test_mesh_nodes_match_node_features(tmp_path, contiguous):
    """Node i of the graph is row i of the node file, whatever the raw vertex ids are."""
    ids, coords, edges = tiny_mesh(contiguous=contiguous)
    node_csv, edge_csv = write_mesh_csvs(tmp_path, contiguous=contiguous)
    node_idx_dict, node_features = process_node_features(node_csv)
    G = construct_mesh_graph(process_edges(edge_csv), node_idx_dict)
    assert sorted(G.nodes) == list(range(len(ids)))
    row_of = {raw: i for i, raw in enumerate(ids)}
    for a, b in edges:
        u, v = row_of[a], row_of[b]
        assert G.has_edge(u, v)
        assert G[u][v]["weight"] == pytest.approx(np.linalg.norm(coords[u] - coords[v]) / 1000)
    np.random.seed(0)
    construct_dataset(G, node_features, str(tmp_path / "x.npz"), "single-source-random", 4, 6)
    data = np.load(tmp_path / "x.npz")
    for src, tar, length in zip(data["srcs"], data["tars"], data["lengths"]):
        assert length == pytest.approx(nx.dijkstra_path_length(G, int(src), int(tar)))


def test_simplified_mesh_dataset_golden(tmp_path, update_golden):
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


def test_make_dataset_cli_mesh_branch(tmp_path, monkeypatch):
    # A --raw-data path containing "meshes" reads percent_edges / percent_vertices.
    from spnn.cli import make_dataset

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    n = 4  # 4x4 grid of vertices, edges between horizontal and vertical neighbours
    (mesh_dir / "percent_vertices").write_text(
        "".join(f"{1000 * (i % n)} {1000 * (i // n)} {100 * (i % 3)}\n" for i in range(n * n)))
    edges = [(i, i + 1) for i in range(n * n) if i % n < n - 1] + [(i, i + n) for i in range(n * n - n)]
    (mesh_dir / "percent_edges").write_text("".join(f"{u} {v}\n" for u, v in edges))
    out = tmp_path / "mesh.npz"
    monkeypatch.setattr("sys.argv", [
        "make_dataset", "--name", "norway", "--raw-data", str(mesh_dir), "--filename", str(out),
        "--dataset-size", "12", "--num-sources", "3", "--sampling-technique", "single-source-random"])
    np.random.seed(0)
    make_dataset.main()
    data = np.load(out)
    assert len(data["node_features"]) == n * n
    assert len(data["srcs"]) == len(data["lengths"]) == 12
