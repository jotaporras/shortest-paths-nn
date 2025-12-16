#!/usr/bin/env python3
"""
Update wandb run configs with command-line arguments extracted from metadata.

This script finds all finished runs and updates their config with the full
set of parameters that were passed via command line but not logged to config.
"""

import wandb
import argparse
from typing import Optional


def parse_args_to_config(args_list: list) -> dict:
    """
    Parse command-line arguments list into a config dictionary.
    
    Args:
        args_list: List of command-line arguments, e.g., 
                   ['--train-data', 'file.npz', '--epochs', '500', '--new']
    
    Returns:
        Dictionary of parsed arguments with appropriate types.
    """
    config = {}
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        
        if arg.startswith('--'):
            key = arg[2:].replace('-', '_')  # --train-data -> train_data
            
            # Check if next item is a value or another flag
            if i + 1 < len(args_list) and not args_list[i + 1].startswith('--'):
                value = args_list[i + 1]
                # Try to convert to appropriate type
                value = convert_value(value)
                config[key] = value
                i += 2
            else:
                # Flag without value (e.g., --new)
                config[key] = True
                i += 1
        elif arg.startswith('-') and len(arg) == 2:
            # Short flag like -p
            key = arg[1:]
            if i + 1 < len(args_list) and not args_list[i + 1].startswith('-'):
                value = convert_value(args_list[i + 1])
                config[key] = value
                i += 2
            else:
                config[key] = True
                i += 1
        else:
            # Positional argument, skip
            i += 1
    
    return config


def convert_value(value: str):
    """Convert string value to appropriate Python type."""
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Check for boolean-like strings
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False
    
    # Return as string
    return value


def get_finished_runs(entity: str, project: str, tags: Optional[list] = None) -> list:
    """
    Get all finished runs from a wandb project.
    
    Args:
        entity: wandb entity (user or team)
        project: wandb project name
        tags: Optional list of tags to filter by
    
    Returns:
        List of finished Run objects.
    """
    api = wandb.Api()
    
    filters = {"state": "finished"}
    if tags:
        filters["tags"] = {"$in": tags}
    
    runs = api.runs(f"{entity}/{project}", filters=filters)
    
    return list(runs)


def update_run_config(run, dry_run: bool = True) -> dict:
    """
    Update a run's config with arguments extracted from metadata.
    
    Args:
        run: wandb Run object
        dry_run: If True, don't actually update, just return what would be updated
    
    Returns:
        Dictionary of new config entries that were/would be added.
    """
    # Get the command-line args from metadata
    metadata = run.metadata
    if not metadata or 'args' not in metadata:
        print(f"  ⚠ No args in metadata for run {run.id}")
        return {}
    
    args_list = metadata['args']
    parsed_config = parse_args_to_config(args_list)
    
    # Also add program info if available
    if 'program' in metadata:
        parsed_config['_program'] = metadata['program']
    if 'codePath' in metadata:
        parsed_config['_code_path'] = metadata['codePath']
    if 'git' in metadata:
        git_info = metadata['git']
        if 'commit' in git_info:
            parsed_config['_git_commit'] = git_info['commit']
        if 'remote' in git_info:
            parsed_config['_git_remote'] = git_info['remote']
    
    # Find new config entries (not already in run.config)
    existing_config = run.config or {}
    new_entries = {}
    for key, value in parsed_config.items():
        if key not in existing_config:
            new_entries[key] = value
    
    if not dry_run and new_entries:
        # Update the run's config
        run.config.update(new_entries)
        run.update()
        print(f"  ✓ Updated config with {len(new_entries)} new entries")
    
    return new_entries


def main():
    parser = argparse.ArgumentParser(
        description="Update wandb run configs with command-line arguments from metadata"
    )
    parser.add_argument(
        '--entity', 
        type=str, 
        default='alelab',
        help='wandb entity (default: alelab)'
    )
    parser.add_argument(
        '--project', 
        type=str, 
        default='terrains',
        help='wandb project name (default: terrains)'
    )
    parser.add_argument(
        '--tags', 
        type=str, 
        nargs='+',
        help='Filter runs by tags'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        help='Update a specific run by ID (instead of all finished runs)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Show what would be updated without making changes (default: True)'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply the updates (overrides --dry-run)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of runs to process'
    )
    
    args = parser.parse_args()
    
    dry_run = not args.apply
    
    api = wandb.Api()
    
    print(f"Entity: {args.entity}")
    print(f"Project: {args.project}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLYING CHANGES'}")
    print()
    
    if args.run_id:
        # Process single run
        run_path = f"{args.entity}/{args.project}/{args.run_id}"
        print(f"Fetching run: {run_path}")
        try:
            run = api.run(run_path)
            runs = [run]
        except Exception as e:
            print(f"Error fetching run: {e}")
            return
    else:
        # Get all finished runs
        print("Fetching finished runs...")
        runs = get_finished_runs(args.entity, args.project, args.tags)
        print(f"Found {len(runs)} finished runs")
    
    if args.limit:
        runs = runs[:args.limit]
        print(f"Processing first {args.limit} runs")
    
    print()
    
    total_updated = 0
    for i, run in enumerate(runs):
        print(f"[{i+1}/{len(runs)}] Run: {run.name} ({run.id})")
        print(f"  State: {run.state}")
        print(f"  Created: {run.created_at}")
        
        # Get current config
        current_config = run.config or {}
        print(f"  Current config keys: {list(current_config.keys())}")
        
        # Update config
        new_entries = update_run_config(run, dry_run=dry_run)
        
        if new_entries:
            print(f"  {'Would add' if dry_run else 'Added'} {len(new_entries)} new config entries:")
            for key, value in sorted(new_entries.items()):
                print(f"    {key}: {value}")
            total_updated += 1
        else:
            print("  No new entries to add")
        
        print()
    
    print("=" * 60)
    print(f"Summary: {'Would update' if dry_run else 'Updated'} {total_updated}/{len(runs)} runs")
    
    if dry_run:
        print("\nThis was a dry run. Use --apply to actually update the configs.")


if __name__ == "__main__":
    main()

