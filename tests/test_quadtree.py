"""Characterization tests for spnn.graphs.quadtree (quadtree terrain simplification).

No code imports this module today. It is kept as a graph-construction option.
"""
import numpy as np
import pytest

from conftest import synthetic_dem
from golden_util import array_digest, check_golden
from spnn.graphs import quadtree


def leaves(tree):
    return sorted(node.data for node in tree.all_nodes() if node.is_leaf())


def test_flat_array_is_single_root():
    tree = quadtree.build_quadtree(np.zeros((8, 8)), thresh=0.1)
    assert tree.size() == 1


def test_leaves_tile_the_array():
    arr = synthetic_dem(8, 0) / 1000
    tree = quadtree.build_quadtree(arr, thresh=0.05)
    covered = np.zeros(arr.shape, dtype=int)
    for lb, rb, ub, db in leaves(tree):
        covered[lb:rb, ub:db] += 1
    assert (covered == 1).all()


def test_leaf_bounds_index_the_transposed_array():
    """Current behaviour: leaf (lb, rb) is a column range and (ub, db) a row range."""
    arr = synthetic_dem(8, 0) / 1000
    thresh = 0.05
    for lb, rb, ub, db in leaves(quadtree.build_quadtree(arr, thresh=thresh)):
        block = arr.T[lb:rb, ub:db]
        assert block.size <= 1 or np.ptp(block) <= thresh


@pytest.mark.xfail(strict=True, reason="pass 2: build_dem_qt_ swaps the bounds of q2/q4, "
                                       "so (lb, rb) are columns, while quadtree_simplification "
                                       "reads them as rows")
def test_leaf_bounds_index_the_array_rows_first():
    arr = synthetic_dem(8, 0) / 1000
    thresh = 0.05
    for lb, rb, ub, db in leaves(quadtree.build_quadtree(arr, thresh=thresh)):
        block = arr[lb:rb, ub:db]
        assert block.size <= 1 or np.ptp(block) <= thresh


@pytest.mark.parametrize("size,thresh", [(8, 0.05), (9, 0.02)])
def test_quadtree_golden(size, thresh, update_golden):
    arr = synthetic_dem(size, 1) / 1000
    tree = quadtree.build_quadtree(arr, thresh=thresh)
    graph = quadtree.quadtree_simplification(tree, size, size)
    edges = sorted(tuple(sorted(map(int, e))) for e in graph.edges)
    observed = {
        "leaves": [list(map(int, leaf)) for leaf in leaves(tree)],
        "num_tree_nodes": tree.size(),
        "num_graph_nodes": graph.number_of_nodes(),
        "edges_digest": array_digest(edges),
        "num_edges": len(edges),
    }
    check_golden(f"quadtree_{size}_{thresh}", observed, update_golden)
