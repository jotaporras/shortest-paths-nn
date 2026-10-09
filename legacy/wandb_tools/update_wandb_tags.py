#!/usr/bin/env python3
"""
Script to update wandb run tags for jotaporras/terrains project.

Removes all existing tags and replaces them based on run name:
- If run name contains "stage1" -> ["e20TG_neurogf_terrain_graphs", "phase1"]
- If run name contains "stage2" -> ["e20TG_neurogf_terrain_graphs", "phase2"]
"""

import wandb


def update_run_tags():
    """Fetch all runs from jotaporras/terrains and update their tags."""
    api = wandb.Api()
    
    # Fetch all runs from the project
    project_path = "jotaporras/terrains"
    print(f"Fetching runs from {project_path}...")
    runs = api.runs(project_path)
    
    print(f"Found {len(runs)} runs")
    
    updated_count = 0
    skipped_count = 0
    
    for run in runs:
        run_name = run.name
        old_tags = run.tags
        
        # Determine new tags based on run name
        if "stage1" in run_name.lower():
            new_tags = ["e20TG_neurogf_terrain_graphs", "phase1"]
        elif "stage2" in run_name.lower():
            new_tags = ["e20TG_neurogf_terrain_graphs", "phase2"]
        else:
            print(f"  SKIP: '{run_name}' - no stage1/stage2 in name")
            skipped_count += 1
            continue
        
        # Update the run tags
        print(f"  UPDATE: '{run_name}'")
        print(f"    Old tags: {old_tags}")
        print(f"    New tags: {new_tags}")
        
        run.tags = new_tags
        run.update()
        updated_count += 1
    
    print(f"\nDone! Updated {updated_count} runs, skipped {skipped_count} runs.")


if __name__ == "__main__":
    update_run_tags()

