"""End-to-end single-stage training through the CLI entry point, on CPU.

Trains on one synthetic terrain and tests on another (cross-terrain, like
e30), then compares the final metrics with tests/golden/e2e_metrics.json.
Pass 1 must reproduce the golden values of each platform (darwin-arm64,
linux-x86_64) to rel 1e-6. SparseGT runs only where CPU sparse ops have MKL.
"""
import json
import math
import platform
import sys
from pathlib import Path

import pytest
import torch
import wandb
import yaml

import train_single_terrain_case
from conftest import GOLDEN_DIR, requires_sparse_cpu

GNN = {"constr": {"input": 3, "hidden": 8, "output": 8, "layers": 2},
       "layer_norm": False, "dropout": True, "activation": "lrelu"}
MLP = {"constr": {"input": 8, "hidden": 8, "output": 1, "layers": 2},
       "layer_norm": False, "dropout": True}
SPARSE_GT = {"hidden_dim": 8, "num_layers": 1, "num_heads": 2, "num_hops": 2,
             "rpearl_samples": 3, "rpearl_num_layers": 2, "pe_k": 2,
             "dropout": 0.1, "attn_dropout": 0.1}

CASES = {
    "tagconv": (["--layer-type", "TAGConv"], {"gnn": GNN, "mlp": MLP}),
    "sgt_random": (["--layer-type", "SparseGT", "--rpearl-embedding-mode", "random"],
                   {"gnn": {**GNN, "sparse_gt": SPARSE_GT}, "mlp": MLP}),
    "sgt_data": (["--layer-type", "SparseGT", "--rpearl-embedding-mode", "data"],
                 {"gnn": {**GNN, "sparse_gt": SPARSE_GT}, "mlp": MLP}),
}


def run_training(case, train_npz, test_npz, workdir: Path, monkeypatch):
    extra_args, config = CASES[case]
    config_path = workdir / "config.yml"
    config_path.write_text(yaml.safe_dump({f"tiny-{case}": config}))

    # wandb.init() rebinds wandb.log to the run's log method, so the recorder
    # is installed right after init.
    logged = []
    real_init = wandb.init

    def init_and_record(*args, **kwargs):
        run = real_init(*args, **kwargs)
        monkeypatch.setattr(wandb, "log", lambda data, *a, **kw: logged.append(dict(data)))
        return run

    monkeypatch.setattr(wandb, "init", init_and_record)
    monkeypatch.setattr(train_single_terrain_case, "output_dir", workdir)
    monkeypatch.setattr(sys, "argv", [
        "train_single_terrain_case.py",
        "--train-data", str(train_npz), "--test-data", str(test_npz),
        "--epochs", "3", "--device", "cpu", "--batch-size", "64",
        "--dataset-name", f"tests/{case}", "--config", str(config_path),
        "--siamese", "1", "--vn", "0", "--aggr", "sum+diff", "--p", "4",
        "--loss", "mse_loss", "--finetune", "0", "--include-edge-attr", "1",
        "--lr", "0.01", "--trial", "1", "--new", "--single-graph-full-batch",
        "--run-name", f"test-{case}", *extra_args,
    ])
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_single_terrain_case.main()

    final = [d for d in logged if "final_train_loss" in d]
    assert len(final) == 1, "expected one final-metrics wandb.log call"
    outputs = list(workdir.glob("models/**/runs/*"))
    return final[0], outputs


@pytest.mark.parametrize("case", [
    "tagconv",
    pytest.param("sgt_random", marks=requires_sparse_cpu),
    pytest.param("sgt_data", marks=requires_sparse_cpu),
])
def test_training_matches_golden(case, terrain_pair, tmp_path, monkeypatch, update_golden):
    train_npz, test_npz = terrain_pair
    metrics, outputs = run_training(case, train_npz, test_npz, tmp_path, monkeypatch)

    assert len(outputs) == 1
    assert (outputs[0] / "final_model.pt").exists()
    assert (outputs[0] / "preds.csv").exists()
    assert all(math.isfinite(v) for k, v in metrics.items() if "nmae" not in k)

    # CPU float kernels differ between platforms (Accelerate vs MKL), and a few
    # training steps amplify the difference, so each platform has its own file.
    platform_tag = f"{platform.system()}-{platform.machine()}".lower()
    golden_path = GOLDEN_DIR / f"e2e_{case}.{platform_tag}.json"
    if update_golden:
        golden_path.parent.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(metrics, indent=1, sort_keys=True))
    golden = json.loads(golden_path.read_text())
    assert metrics.keys() == golden.keys()
    for key, value in golden.items():
        assert metrics[key] == pytest.approx(value, rel=1e-6, nan_ok=True), key
