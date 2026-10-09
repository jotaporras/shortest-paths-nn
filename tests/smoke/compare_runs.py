"""Compare the metrics of two smoke-test output dirs (see smoke_test.sh).

Prints one row per (model, metric) and exits with status 1 if any metric or
prediction differs by more than that model's tolerance (TOLERANCES).

Tolerances come from two runs of the same commit on plaza (2026-10-09):
  - tagconv runs with deterministic kernels and repeats exactly.
  - SparseGT CSR ops are nondeterministic on GPU. Continuous metrics moved up
    to 2e-4 relative, and acc_* moved by a few pairs. The exact check for
    SparseGT is the CPU pytest suite (tests/test_training_e2e.py).

Usage:
    python tests/smoke/compare_runs.py <baseline_out> <candidate_out> [--exact]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

MODELS = ("tagconv", "sgt_random", "sgt_data")
METRIC_PREFIXES = ("final_train_", "test_", "best_val_nmae")
# model -> (rtol for continuous metrics and preds.csv, atol for acc_* fractions)
TOLERANCES = {"tagconv": (0.0, 0.0), "sgt_random": (2e-3, 2.5e-3), "sgt_data": (2e-3, 2.5e-3)}


def load_summary(out_dir: Path, model: str) -> dict:
    """Final metrics plus the best validation nmae, from run_with_metrics.py output."""
    logged = json.loads((out_dir / f"{model}.json").read_text())
    summary = {k: v for k, v in logged["final"].items() if k.startswith(METRIC_PREFIXES)}
    val_nmaes = [d["val_nmae"] for d in logged["val"] if "val_nmae" in d]
    if val_nmaes:
        summary["best_val_nmae"] = min(val_nmaes)
    return summary


def load_preds(model_dir: Path) -> pd.DataFrame:
    paths = sorted(model_dir.glob("models/**/preds.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected 1 preds.csv under {model_dir}, found {len(paths)}")
    return pd.read_csv(paths[0])


def rel_diff(a: float, b: float) -> float:
    if a == b or (math.isnan(a) and math.isnan(b)):
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--exact", action="store_true", help="Use zero tolerance for every model.")
    args = parser.parse_args()

    failed = False
    print(f"{'model':<11} {'metric':<22} {'baseline':>14} {'candidate':>14} {'rel diff':>10}")
    for model in MODELS:
        rtol, acc_atol = (0.0, 0.0) if args.exact else TOLERANCES[model]
        base = load_summary(args.baseline, model)
        cand = load_summary(args.candidate, model)
        for key in sorted(set(base) | set(cand)):
            if key not in base or key not in cand:
                print(f"{model:<11} {key:<22} only in {'baseline' if key in base else 'candidate'}")
                failed = True
                continue
            diff = rel_diff(float(base[key]), float(cand[key]))
            if "_acc_" in key:
                too_far = abs(float(base[key]) - float(cand[key])) > acc_atol
            else:
                too_far = diff > rtol
            flag = " FAIL" if too_far else ""
            failed |= bool(flag)
            print(f"{model:<11} {key:<22} {base[key]:>14.6g} {cand[key]:>14.6g} {diff:>10.2e}{flag}")

        base_preds = load_preds(args.baseline / model)
        cand_preds = load_preds(args.candidate / model)
        if base_preds.shape != cand_preds.shape:
            print(f"{model:<11} preds.csv shape {base_preds.shape} vs {cand_preds.shape} FAIL")
            failed = True
            continue
        # The evaluated pairs must always match. Per-row predictions are compared
        # only for models with a zero tolerance: on GPU, single SparseGT
        # predictions move more than the averaged metrics do.
        pair_cols = ["srcs", "tars", "lengths"]
        close = base_preds[pair_cols].equals(cand_preds[pair_cols])
        if rtol == 0.0:
            close &= base_preds.equals(cand_preds)
        label = "identical" if rtol == 0.0 else "same pairs"
        print(f"{model:<11} {'preds.csv':<22} {label if close else 'DIFFERENT':>14}{'' if close else ' FAIL'}")
        failed |= not close

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
