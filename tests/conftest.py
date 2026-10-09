"""Shared fixtures for the test suite.

wandb is disabled and TERRAIN_OUTPUT_DIR points at a temporary directory
before any test module imports the training code.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from spnn.graphs.grid import construct_nx_graph, get_dem_xv_yv_
from spnn.sampling.train_pairs import construct_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ.setdefault("TERRAIN_OUTPUT_DIR", tempfile.mkdtemp(prefix="spnn-tests-"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="Rewrite tests/golden/*.json from the current code.")


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


def synthetic_dem(size: int, seed: int) -> np.ndarray:
    """Smooth random terrain in metres: two Gaussian hills plus noise."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    hills = 300 * np.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2) / 0.2)
    hills += 150 * np.exp(-((x + 0.5) ** 2 + (y - 0.4) ** 2) / 0.1)
    return hills + rng.normal(0, 5, size=(size, size))


def build_terrain_npz(path: Path, size: int, dem_resolution: float, seed: int,
                      num_srcs: int, samples_per_source: int,
                      sampling_method: str = "hybrid", edge_weight: bool = False) -> Path:
    """Run the repo's own DEM -> graph -> pairs pipeline on a synthetic DEM."""
    xv, yv, elevation = get_dem_xv_yv_(synthetic_dem(size, seed), dem_resolution, visualize=False)
    graph, node_features = construct_nx_graph(xv, yv, elevation, scale=edge_weight)
    np.random.seed(seed)
    construct_dataset(G=graph, node_features=node_features, filename=str(path),
                         sampling_method=sampling_method, num_srcs=num_srcs,
                         samples_per_source=samples_per_source, rows=size, cols=size)
    return path


@pytest.fixture(scope="session")
def terrain_pair(tmp_path_factory):
    """A training terrain (10x10) and a different test terrain (8x8, other spacing).

    The test terrain has 480 pairs so that its 0.5% validation split has more
    than one pair (a single validation pair crashes compute_metrics today).
    """
    root = tmp_path_factory.mktemp("terrains")
    train = build_terrain_npz(root / "train.npz", size=10, dem_resolution=10.0, seed=0,
                              num_srcs=10, samples_per_source=20)
    test = build_terrain_npz(root / "test.npz", size=8, dem_resolution=3.0, seed=1,
                             num_srcs=16, samples_per_source=30)
    return train, test


requires_sparse_cpu = pytest.mark.skipif(
    not torch.backends.mkl.is_available(),
    reason="SparseGT uses torch.sparse.sampled_addmm, which needs MKL on CPU (absent on macOS arm64)",
)
