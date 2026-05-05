#!/usr/bin/env python
"""
Wandb sweep agent script for SparseGT with random embedding on Norway res=4.

Usage:
    # 1. Create the sweep (once):
    wandb sweep configs/sweep-sparse-gt-random-norway.yml

    # 2. Launch agent(s) — wandb agent calls this script directly:
    CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID>
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch_geometric.data import Data
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import refactor_training
from refactor_training import (
    npz_to_dataset,
    split_dataset_for_validation,
    format_log_dir,
)

ENTITY = "alelab"
PROJECT = "manifold-transformers-dev"
WANDB_TAG = "sweep_sgt_random_norway"

TRAIN_FILE = PROJECT_ROOT / "data" / "res10_hybrid.npz"
TEST_FILE = PROJECT_ROOT / "data" / "generated2" / "full_test-004.npz"
DATASET_NAME = "norway/res10"
RESOLUTION = 10


def build_model_config(cfg):
    """Construct the nested model config dict from flat wandb.config."""
    return {
        "sparse-gt-rpearl": {
            "gnn": {
                "constr": {
                    "input": 3,
                    "hidden": cfg["hidden_dim"],
                    "output": cfg["hidden_dim"],
                    "layers": 3,
                },
                "layer_norm": False,
                "dropout": True,
                "activation": "lrelu",
                "sparse_gt": {
                    "hidden_dim": cfg["hidden_dim"],
                    "num_layers": cfg["num_layers"],
                    "num_heads": cfg["num_heads"],
                    "num_hops": cfg["num_hops"],
                    "rpearl_num_layers": cfg["rpearl_num_layers"],
                    "dropout": cfg["dropout"],
                    "attn_dropout": cfg["attn_dropout"],
                    "embedding_mode": "random",
                    "pe_k": cfg["pe_k"],
                },
            },
            "mlp": {
                "constr": {
                    "input": cfg["hidden_dim"],
                    "hidden": 128,
                    "output": 1,
                    "layers": 3,
                },
                "layer_norm": False,
                "dropout": True,
            },
        }
    }


def train():
    run = wandb.init()
    cfg = dict(wandb.config)

    model_config_full = build_model_config(cfg)
    model_name = "sparse-gt-rpearl"
    model_config = model_config_full[model_name]

    train_data = np.load(str(TRAIN_FILE), allow_pickle=True)
    test_data = np.load(str(TEST_FILE), allow_pickle=True)

    train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)
    test_dataset_full, test_node_features, test_edge_index = npz_to_dataset(test_data)
    val_dataset, test_dataset = split_dataset_for_validation(
        test_dataset_full, val_fraction=0.005, seed=42
    )

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_edge_attr = train_data["distances"]
    test_edge_attr = test_data["distances"]
    edge_attr = torch.tensor(train_edge_attr).unsqueeze(-1)
    test_edge_attr_tensor = torch.tensor(test_edge_attr).unsqueeze(-1)

    graph_data = Data(
        x=train_node_features, edge_index=train_edge_index, edge_attr=edge_attr
    )
    test_graph_data = Data(
        x=test_node_features, edge_index=test_edge_index, edge_attr=test_edge_attr_tensor
    )

    train_dictionary = {"graphs": [graph_data], "dataloaders": [train_dataloader]}
    val_dictionary = {"graphs": [test_graph_data], "dataloaders": [val_dataloader]}
    test_dictionary = {"graphs": [test_graph_data], "dataloaders": [test_dataloader]}

    log_dir = format_log_dir(
        PROJECT_ROOT,
        DATASET_NAME,
        True,       # siamese
        model_name,
        False,      # vn
        "sum+diff",
        "mse_loss",
        "SparseGT",
        cfg.get("p", 4),
        run.id,
    )

    wandb_config = {
        "dataset_name": DATASET_NAME,
        "train_data": str(TRAIN_FILE),
        "test_data": str(TEST_FILE),
        "batch_size": batch_size,
        "resolution": RESOLUTION,
        "include_edge_attr": 1,
        "sweep": True,
    }

    refactor_training.train_few_cross_terrain_case(
        train_dictionary=train_dictionary,
        model_config=model_config,
        layer_type="SparseGT",
        device="cuda",
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        loss_func="mse_loss",
        aggr="sum+diff",
        base_log_dir=log_dir,
        p=cfg.get("p", 4),
        siamese=True,
        new=True,
        run_name=f"sweep-SparseGT-random-res{RESOLUTION:02d}",
        wandb_tag=[WANDB_TAG],
        wandb_config=wandb_config,
        single_graph_full_batch=True,
        test_dictionary=test_dictionary,
        val_dictionary=val_dictionary,
    )

    wandb.finish()


if __name__ == "__main__":
    train()
