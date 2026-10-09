"""Run train_single_terrain_case.py and write its logged metrics to a JSON file.

wandb (offline or disabled) does not leave a readable summary file, so this
wrapper records every wandb.log call and writes:
    final: the final train/test metrics dict
    val:   the per-epoch dicts that contain validation metrics

Set DETERMINISTIC=1 to use torch's deterministic kernels. This makes TAGConv
runs repeatable on GPU. It does not help SparseGT (its CSR ops have no
deterministic CUDA kernel) and slows it about 7x, so the smoke test sets it
for TAGConv only.

Usage:
    python tests/smoke/run_with_metrics.py <metrics.json> <train_single_terrain_case args...>
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    metrics_path = Path(sys.argv[1])
    logged = []
    real_init = wandb.init

    # wandb.init() rebinds wandb.log to the run's log method, so the recorder
    # is installed right after init and forwards to the real method.
    def init_and_record(*args, **kwargs):
        run = real_init(*args, **kwargs)
        run_log = wandb.log

        def record(data, *a, **kw):
            logged.append({k: v for k, v in data.items() if isinstance(v, (int, float))})
            return run_log(data, *a, **kw)

        wandb.log = record
        return run

    wandb.init = init_and_record
    if os.environ.get("DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
    sys.argv = ["train_single_terrain_case.py", *sys.argv[2:]]
    import train_single_terrain_case
    train_single_terrain_case.main()

    final = [d for d in logged if "final_train_loss" in d]
    val = [d for d in logged if any(k.startswith("val_") for k in d)]
    metrics_path.write_text(json.dumps({"final": final[-1] if final else None, "val": val}, indent=1))


if __name__ == "__main__":
    main()
