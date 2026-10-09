#!/usr/bin/env python3
"""
Backfill wandb run summaries with metrics computed from batch history.

This script computes non-batch metrics (e.g., test_mae from test_batch_mae)
and stores them in wandb run summaries so notebooks don't need to recompute them.

Usage:
    python scripts/backfill_wandb_metrics.py
    python scripts/backfill_wandb_metrics.py --dry-run
    python scripts/backfill_wandb_metrics.py --tag e23TG_neurogf_terrain_graph_2
"""

import argparse
import numpy as np
import wandb
from tqdm import tqdm


# Default configuration
DEFAULT_ENTITY = "alelab"
DEFAULT_PROJECT = "terrains"

# Mapping of target metric -> source batch metric
METRIC_MAPPINGS = {
    "test_mae": "test_batch_mae",
    "val_mae": "val_batch_mae",
}


def get_metric_history_mean(run: wandb.apis.public.Run, metric: str) -> float:
    """Compute the mean of a metric from run history."""
    history = run.history(keys=[metric], pandas=True)
    if history.empty or metric not in history.columns:
        return np.nan
    values = history[metric].dropna()
    return float(values.mean()) if len(values) > 0 else np.nan


def run_needs_backfill(run: wandb.apis.public.Run, target_metric: str, source_metric: str) -> bool:
    """Check if a run needs the metric to be backfilled."""
    # Skip if target metric already exists in summary
    if target_metric in run.summary and run.summary[target_metric] is not None:
        return False

    # Check if source metric exists in history (we'll verify during actual computation)
    return True


def backfill_run(run: wandb.apis.public.Run, metric_mappings: dict, dry_run: bool = False) -> dict:
    """
    Backfill metrics for a single run.

    Returns dict of metrics that were (or would be) updated.
    """
    updates = {}

    for target_metric, source_metric in metric_mappings.items():
        # Skip if already has the metric
        if target_metric in run.summary and run.summary[target_metric] is not None:
            continue

        # Compute from history
        mean_value = get_metric_history_mean(run, source_metric)

        if not np.isnan(mean_value):
            updates[target_metric] = mean_value

    # Apply updates to wandb
    if updates and not dry_run:
        run.summary.update(updates)

    return updates


def main():
    parser = argparse.ArgumentParser(
        description="Backfill wandb run summaries with computed metrics from batch history."
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"Wandb entity (default: {DEFAULT_ENTITY})"
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"Wandb project (default: {DEFAULT_PROJECT})"
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Only process runs with this tag (optional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Include running/crashed runs (default: only finished runs)"
    )

    args = parser.parse_args()

    api = wandb.Api()

    # Build filters
    filters = {}
    if not args.include_running:
        filters["state"] = "finished"
    if args.tag:
        filters["tags"] = args.tag

    # Fetch runs
    print(f"Fetching runs from {args.entity}/{args.project}...")
    if filters:
        print(f"  Filters: {filters}")

    runs = api.runs(f"{args.entity}/{args.project}", filters=filters)
    runs_list = list(runs)
    print(f"Found {len(runs_list)} runs")

    if args.dry_run:
        print("\n*** DRY RUN - No changes will be made ***\n")

    # Process runs
    updated_count = 0
    skipped_count = 0

    for run in tqdm(runs_list, desc="Processing runs"):
        updates = backfill_run(run, METRIC_MAPPINGS, dry_run=args.dry_run)

        if updates:
            updated_count += 1
            if args.dry_run:
                tqdm.write(f"  Would update {run.name} ({run.id}): {updates}")
            else:
                tqdm.write(f"  Updated {run.name} ({run.id}): {updates}")
        else:
            skipped_count += 1

    # Summary
    print(f"\nSummary:")
    print(f"  Runs processed: {len(runs_list)}")
    print(f"  Runs {'would be ' if args.dry_run else ''}updated: {updated_count}")
    print(f"  Runs skipped (already have metrics): {skipped_count}")

    if args.dry_run and updated_count > 0:
        print(f"\nRun without --dry-run to apply these changes.")


if __name__ == "__main__":
    main()
