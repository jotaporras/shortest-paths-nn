"""wandb sweep agent for one of the sweep presets below.

Usage:
    # 1. Create the sweep (once):
    wandb sweep configs/sweep-gnn-norway.yml

    # 2. Launch agent(s); the sweep YAML runs `python -m spnn.cli.sweep --preset <name>`:
    CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID>
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from ..data import npz_to_dataset, split_dataset_for_validation
from ..training import REPO_ROOT, format_log_dir, train_few_cross_terrain_case

PROJECT_ROOT = REPO_ROOT
ENTITY = "alelab"
PROJECT = "manifold-transformers-dev"

# Model families: the config name, layer type and run-name label for each.
MODELS = {
    "tagconv": {"model_name": "tagconv-k3", "layer_type": "TAGConv", "label": "TAGConv"},
    "sgt_data": {"model_name": "sparse-gt-rpearl", "layer_type": "SparseGT",
                 "label": "SparseGT-data", "embedding_mode": "data"},
    "sgt_random": {"model_name": "sparse-gt-rpearl", "layer_type": "SparseGT",
                   "label": "SparseGT-random", "embedding_mode": "random"},
}

DATA = Path("data")
PRESETS = {
    "gnn_norway": dict(model="tagconv", wandb_tag="sweep_gnn_norway",
                       train_file=DATA / "res03_hybrid.npz",
                       test_file=DATA / "generated2" / "full_test-004.npz",
                       dataset_name="norway/res03", resolution=3),
    "gnn_philly": dict(model="tagconv", wandb_tag="sweep_gnn_philly",
                       train_file=DATA / "philly_res04_hybrid.npz",
                       test_file=DATA / "generated2" / "philly_test_res04.npz",
                       dataset_name="philadelphia/res04", resolution=4),
    "sparse_gt_data_norway": dict(model="sgt_data", wandb_tag="sweep_sgt_data_norway",
                                  train_file=DATA / "res10_hybrid.npz",
                                  test_file=DATA / "generated2" / "full_test-004.npz",
                                  dataset_name="norway/res10", resolution=10),
    "sparse_gt_data_philly": dict(model="sgt_data", wandb_tag="sweep_sgt_data_philly",
                                  train_file=DATA / "philly_res10_hybrid.npz",
                                  test_file=DATA / "generated2" / "philly_test_res04.npz",
                                  dataset_name="philadelphia/res10", resolution=10,
                                  early_stopping_patience=30),
    "sparse_gt_random_norway": dict(model="sgt_random", wandb_tag="sweep_sgt_random_norway",
                                    train_file=DATA / "res10_hybrid.npz",
                                    test_file=DATA / "generated2" / "full_test-004.npz",
                                    dataset_name="norway/res10", resolution=10),
    "sparse_gt_random_philly": dict(model="sgt_random", wandb_tag="sweep_sgt_random_philly",
                                    train_file=DATA / "philly_res10_hybrid.npz",
                                    test_file=DATA / "generated2" / "philly_test_res04.npz",
                                    dataset_name="philadelphia/res10", resolution=10),
}


def build_model_config(cfg, model):
    """Construct the nested model config dict from flat wandb.config."""
    is_sgt = "embedding_mode" in MODELS[model]
    gnn = {
        "constr": {
            "input": 3,
            "hidden": cfg["hidden_dim"],
            "output": cfg["hidden_dim"],
            # SparseGT reads its depth from sparse_gt.num_layers; constr.layers is unused.
            "layers": 3 if is_sgt else cfg["num_layers"],
        },
        "layer_norm": False,
        "dropout": True,
        "activation": "lrelu",
    }
    if is_sgt:
        gnn["sparse_gt"] = {
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "num_heads": cfg["num_heads"],
            "num_hops": cfg["num_hops"],
            "rpearl_num_layers": cfg["rpearl_num_layers"],
            "dropout": cfg["dropout"],
            "attn_dropout": cfg["attn_dropout"],
            "embedding_mode": MODELS[model]["embedding_mode"],
            "pe_k": cfg["pe_k"],
        }
    mlp = {
        "constr": {
            "input": cfg["hidden_dim"],
            "hidden": 128,
            "output": 1,
            "layers": 3,
        },
        "layer_norm": False,
        "dropout": True,
    }
    return {MODELS[model]["model_name"]: {"gnn": gnn, "mlp": mlp}}


def train(preset_name):
    preset = PRESETS[preset_name]
    model = MODELS[preset["model"]]
    train_file = PROJECT_ROOT / preset["train_file"]
    test_file = PROJECT_ROOT / preset["test_file"]
    dataset_name = preset["dataset_name"]
    resolution = preset["resolution"]

    run = wandb.init()
    cfg = dict(wandb.config)

    model_name = model["model_name"]
    model_config = build_model_config(cfg, preset["model"])[model_name]

    train_data = np.load(str(train_file), allow_pickle=True)
    test_data = np.load(str(test_file), allow_pickle=True)

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
        dataset_name,
        True,       # siamese
        model_name,
        False,      # vn
        "sum+diff",
        "mse_loss",
        model["layer_type"],
        cfg.get("p", 4),
        run.id,
    )

    wandb_config = {
        "dataset_name": dataset_name,
        "train_data": str(train_file),
        "test_data": str(test_file),
        "batch_size": batch_size,
        "resolution": resolution,
        "include_edge_attr": 1,
        "sweep": True,
    }

    extra = {}
    if "early_stopping_patience" in preset:
        extra["early_stopping_patience"] = preset["early_stopping_patience"]

    train_few_cross_terrain_case(
        train_dictionary=train_dictionary,
        model_config=model_config,
        layer_type=model["layer_type"],
        device="cuda",
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        loss_func="mse_loss",
        aggr="sum+diff",
        base_log_dir=log_dir,
        p=cfg.get("p", 4),
        siamese=True,
        new=True,
        run_name=f"sweep-{model['label']}-res{resolution:02d}",
        wandb_tag=[preset["wandb_tag"]],
        wandb_config=wandb_config,
        single_graph_full_batch=True,
        test_dictionary=test_dictionary,
        val_dictionary=val_dictionary,
        **extra,
    )

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    args, _ = parser.parse_known_args()
    train(args.preset)


if __name__ == "__main__":
    main()
