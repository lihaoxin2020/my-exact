#!/usr/bin/env python3
"""
Script to check all GitLab test JSONs and identify which ones have "require_reset" set to false.
This helps identify tasks that don't need to reset the environment between runs.
"""

import os
import json
import glob
from collections import defaultdict

def check_reset_flags(directory_path):
    """Check all JSON files in the directory for their require_reset flag."""
    
    # Results containers
    reset_false = []
    reset_true = []
    no_reset_flag = []
    
    # Stats tracking
    stats = defaultdict(int)
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    json_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    print(f"Found {len(json_files)} JSON files in {directory_path}")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Get file name for display
                filename = os.path.basename(file_path)
                task_id = data.get("task_id", "unknown")
                intent = data.get("intent", "")
                
                # Truncate intent if too long
                if len(intent) > 80:
                    intent = intent[:77] + "..."
                
                # Check reset flag
                if "require_reset" in data:
                    reset_value = data["require_reset"]
                    stats[f"require_reset={reset_value}"] += 1
                    
                    if reset_value is False:
                        reset_false.append((filename, task_id, intent))
                    else:
                        reset_true.append((filename, task_id, intent))
                else:
                    stats["no_reset_flag"] += 1
                    no_reset_flag.append((filename, task_id, intent))
                
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON file {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    return reset_false, reset_true, no_reset_flag, stats

def main():
    directory_path = "configs/webarena/test_gitlab_v2"
    
    reset_false, reset_true, no_reset_flag, stats = check_reset_flags(directory_path)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Print detailed information for files with require_reset=false
    print("\n=== Files with require_reset=false ===")
    if reset_false:
        print(f"Found {len(reset_false)} files with require_reset=false:")
        for filename, task_id, intent in reset_false:
            print(f"- {filename} (Task ID: {task_id}): {intent}")
    else:
        print("No files found with require_reset=false")
    
    # Print files missing the reset flag
    if no_reset_flag:
        print("\n=== Files missing require_reset flag ===")
        print(f"Found {len(no_reset_flag)} files without require_reset flag:")
        for filename, task_id, intent in no_reset_flag:
            print(f"- {filename} (Task ID: {task_id}): {intent}")

if __name__ == "__main__":
    main() 