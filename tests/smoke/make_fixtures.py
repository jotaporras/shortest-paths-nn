"""Build the small npz files used by the plaza smoke test.

Each output keeps the full graph of its input file and a seeded subset of the
(src, tar, length) pairs, so a run exercises real graph sizes with few pairs.

Usage:
    python tests/smoke/make_fixtures.py --train <train.npz> --test <test.npz> --out <dir>
"""
import argparse
from pathlib import Path

import numpy as np

PAIR_KEYS = ("srcs", "tars", "lengths")


def subsample_pairs(path: Path, num_pairs: int, seed: int) -> dict:
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(arrays["srcs"]), size=num_pairs, replace=False))
    for key in PAIR_KEYS:
        arrays[key] = arrays[key][idx]
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, path in (("train", args.train), ("test", args.test)):
        arrays = subsample_pairs(path, args.num_pairs, args.seed)
        np.savez(args.out / f"smoke_{name}.npz", **arrays)
        num_nodes = len(arrays["node_features"])
        num_edges = arrays["edge_index"].shape[1]
        print(f"smoke_{name}.npz: {num_nodes} nodes, {num_edges} edges, "
              f"{len(arrays['srcs'])} pairs (from {path.name})")


if __name__ == "__main__":
    main()
