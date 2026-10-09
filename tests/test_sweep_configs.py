"""Characterization tests for the wandb sweep agent spnn.cli.sweep.

Pass 1 merged 6 scripts (now in legacy/sweeps/) into one CLI with presets.
These tests run each preset's train() with wandb, np.load and the training
call replaced by recorders, and pin every argument the agent passes on. The
goldens were recorded from the 6 original scripts.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from conftest import REPO_ROOT
from golden_util import check_golden
from spnn.cli import sweep

SWEEPS = ["gnn_norway", "gnn_philly",
          "sparse_gt_data_norway", "sparse_gt_data_philly",
          "sparse_gt_random_norway", "sparse_gt_random_philly"]

# One fixed value per sweep parameter, so the nested model config is deterministic.
SWEEP_VALUES = {"hidden_dim": 64, "num_layers": 3, "num_heads": 4, "num_hops": 2,
                "rpearl_num_layers": 2, "pe_k": 8, "dropout": 0.1, "attn_dropout": 0.05,
                "lr": 1e-3, "epochs": 7, "p": 4}


def rel(value):
    """Replace the repo path in strings and Paths, recursively, so goldens are portable."""
    if isinstance(value, dict):
        return {k: rel(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rel(v) for v in value]
    if isinstance(value, (str, Path)):
        return str(value).replace(str(REPO_ROOT), "<root>")
    return value


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_yaml_points_at_its_script(name):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / f"sweep-{name.replace('_', '-')}.yml")
                         .read_text())
    assert cfg["command"][-3:] == ["spnn.cli.sweep", "--preset", name]
    assert set(cfg["parameters"]) <= set(SWEEP_VALUES)


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_train_arguments_golden(name, terrain_pair, monkeypatch, update_golden):
    module = sweep
    calls = {}
    fake_wandb = SimpleNamespace(init=lambda: SimpleNamespace(id="RUNID"),
                                 config=dict(SWEEP_VALUES), finish=lambda: None)
    loaded = []

    def fake_load(path, allow_pickle=False):
        loaded.append(rel(path))
        train_path, test_path = terrain_pair
        return np.load(train_path if len(loaded) == 1 else test_path, allow_pickle=allow_pickle)

    def fake_format_log_dir(*args):
        calls["format_log_dir"] = rel(args)
        return "<log_dir>"

    def fake_train(**kwargs):
        calls["train"] = kwargs

    monkeypatch.setattr(module, "wandb", fake_wandb)
    monkeypatch.setattr(module, "np", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(module, "format_log_dir", fake_format_log_dir)
    monkeypatch.setattr(module, "train_few_cross_terrain_case", fake_train)
    module.train(name)

    kwargs = calls["train"]
    train_graph = kwargs.pop("train_dictionary")["graphs"][0]
    val = kwargs.pop("val_dictionary")
    test = kwargs.pop("test_dictionary")
    assert val["graphs"][0] is test["graphs"][0]
    assert train_graph.edge_attr.dtype.is_floating_point
    observed = {
        "constants": {
            "ENTITY": module.ENTITY,
            "PROJECT": module.PROJECT,
            "WANDB_TAG": module.PRESETS[name]["wandb_tag"],
            "TRAIN_FILE": rel(module.PROJECT_ROOT / module.PRESETS[name]["train_file"]),
            "TEST_FILE": rel(module.PROJECT_ROOT / module.PRESETS[name]["test_file"]),
            "DATASET_NAME": module.PRESETS[name]["dataset_name"],
            "RESOLUTION": module.PRESETS[name]["resolution"],
        },
        "loaded": loaded,
        "format_log_dir": calls["format_log_dir"],
        "train_kwargs": rel(kwargs),
        "val_size": len(val["dataloaders"][0].dataset),
        "test_size": len(test["dataloaders"][0].dataset),
        "batch_size": val["dataloaders"][0].batch_size,
    }
    check_golden(f"sweep_{name}", observed, update_golden)
