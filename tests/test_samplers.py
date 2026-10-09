"""Training-pair samplers (spnn.sampling.targets, spnn.sampling.train_pairs)."""
import json
from functools import partial

import networkx as nx
import numpy as np
import pytest

from conftest import GOLDEN_DIR, build_terrain_npz, synthetic_dem
from spnn.graphs.grid import construct_nx_graph, get_dem_xv_yv_
from spnn.sampling import targets

SIZE = 9


@pytest.fixture(scope="module")
def grid():
    xv, yv, elevation = get_dem_xv_yv_(synthetic_dem(SIZE, seed=0), 10.0, visualize=False)
    graph, node_features = construct_nx_graph(xv, yv, elevation)
    return graph, np.array(node_features)


def samplers(node_features):
    alt_var = targets.compute_altitude_variance(node_features, SIZE, SIZE)
    return {
        "random_sampling": targets.random_sampling,
        "distance_based": targets.distance_based,
        "proximity_variance_sampling": partial(targets.proximity_variance_sampling, altitude_variance=alt_var),
        "hybrid_sampling": partial(targets.hybrid_sampling, altitude_variance=alt_var),
    }


SAMPLER_NAMES = ["random_sampling", "distance_based", "proximity_variance_sampling", "hybrid_sampling"]


@pytest.mark.parametrize("name", SAMPLER_NAMES)
def test_sampler_output_is_consistent(grid, name):
    graph, node_features = grid
    sampler = samplers(node_features)[name]
    np.random.seed(0)
    srcs, tars, lengths = sampler(graph, 12, src=5)
    assert srcs == [5] * 12
    assert len(tars) == len(lengths) == 12
    assert len(set(tars)) == 12, "targets are drawn without replacement"
    for tar, length in zip(tars, lengths):
        assert length == pytest.approx(nx.dijkstra_path_length(graph, 5, tar))


@pytest.mark.parametrize("name", SAMPLER_NAMES)
def test_sampler_is_deterministic_under_seed(grid, name):
    graph, node_features = grid
    sampler = samplers(node_features)[name]
    np.random.seed(123)
    first = sampler(graph, 10, src=7)
    np.random.seed(123)
    assert sampler(graph, 10, src=7) == first


@pytest.mark.parametrize("name", [
    pytest.param("random_sampling", marks=pytest.mark.xfail(strict=True, reason="pass 2: can pick src as target")),
    pytest.param("distance_based", marks=pytest.mark.xfail(strict=True, reason="pass 2: can pick src as target")),
    "proximity_variance_sampling",
    "hybrid_sampling",
])
def test_sampler_never_returns_source_as_target(grid, name):
    graph, node_features = grid
    sampler = samplers(node_features)[name]
    num_nodes = graph.number_of_nodes()
    # asking for all other nodes forces a self-pair if the sampler allows one
    for src in range(0, num_nodes, 10):
        np.random.seed(src)
        _, tars, _ = sampler(graph, num_nodes - 1, src=src)
        assert src not in tars


def test_altitude_variance_matches_numpy_gradient(grid):
    _, node_features = grid
    z = node_features[:, 2].reshape(SIZE, SIZE)
    gy, gx = np.gradient(z)
    expected = np.sqrt(gx ** 2 + gy ** 2).ravel()
    np.testing.assert_allclose(targets.compute_altitude_variance(node_features, SIZE, SIZE), expected)


@pytest.mark.parametrize("method", ["single-source-random", "distance-based", "proximity-variance", "hybrid"])
def test_construct_dataset_golden(tmp_path, method, update_golden):
    """Pins the exact pairs each method produces for a fixed seed.

    Pass 1 must keep these identical. A pass 2 change that alters them must
    say why and rerun with --update-golden.
    """
    path = build_terrain_npz(tmp_path / "d.npz", size=SIZE, dem_resolution=10.0, seed=0,
                             num_srcs=6, samples_per_source=8, sampling_method=method)
    data = np.load(path)
    observed = {
        "srcs": data["srcs"].tolist(),
        "tars": data["tars"].tolist(),
        "lengths_sum": float(np.sum(data["lengths"])),
        "distances_sum": float(np.sum(data["distances"])),
    }
    golden_path = GOLDEN_DIR / f"construct_dataset_{method}.json"
    if update_golden:
        golden_path.parent.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(observed, indent=1))
    golden = json.loads(golden_path.read_text())
    assert observed["srcs"] == golden["srcs"]
    assert observed["tars"] == golden["tars"]
    assert observed["lengths_sum"] == pytest.approx(golden["lengths_sum"], rel=1e-12)
    assert observed["distances_sum"] == pytest.approx(golden["distances_sum"], rel=1e-12)


def test_critical_point_source_crashes(tmp_path):
    # pass 2: construct_dataset uses an undefined `row` in this branch (dataset.py:210)
    with pytest.raises(NameError):
        build_terrain_npz(tmp_path / "c.npz", size=SIZE, dem_resolution=10.0, seed=0,
                          num_srcs=4, samples_per_source=4, sampling_method="critical-point-source")
