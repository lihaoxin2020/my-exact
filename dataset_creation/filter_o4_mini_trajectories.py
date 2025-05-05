#!/usr/bin/env python3
import os
import json

# Path to your filtered test set - these are the ones to EXCLUDE
FILTERED_TEST_PATH = '/Users/nikhilkhandekar/Documents/my-exact/configs/webarena/filtered_test_gitlab_v2.json'
# Path to the trajectories directory
TRAJECTORIES_DIR = '/Users/nikhilkhandekar/Documents/my-exact/trajectories_o4_mini'

# Load the filtered test set to get task IDs to exclude
with open(FILTERED_TEST_PATH, 'r') as f:
    filtered_tests = json.load(f)

# Extract task IDs from the filtered test set - these are the ones to exclude
excluded_task_ids = set(test['task_id'] for test in filtered_tests)
print(f"Found {len(excluded_task_ids)} task IDs to exclude in the filtered test set")

# Process each trajectory file
excluded_count = 0
kept_count = 0
total_files = 0

for filename in os.listdir(TRAJECTORIES_DIR):
    if not filename.endswith('.pkl.xz'):
        continue
    
    total_files += 1
    file_path = os.path.join(TRAJECTORIES_DIR, filename)
    
    # Extract task ID from filename (assuming format like "task_101.pkl.xz")
    try:
        task_id_str = filename.split('_')[1].split('.')[0]
        task_id = int(task_id_str)
    except (IndexError, ValueError):
        print(f"Warning: Could not parse task ID from {filename}, keeping file")
        kept_count += 1
        continue
    
    # Check if task ID is in the excluded set
    if task_id in excluded_task_ids:
        # Remove this file
        os.remove(file_path)
        excluded_count += 1
        print(f"Removed {filename}")
    else:
        # Keep this file
        kept_count += 1

print(f"Processed {total_files} total files")
print(f"Removed {excluded_count} files that were in the filtered test set")
print(f"Kept {kept_count} files that were not in the filtered test set")
print(f"Directory {TRAJECTORIES_DIR} has been filtered in-place") 