#!/usr/bin/env python3
"""
Script to analyze GitLab test JSONs and generate parallel processing task configurations.
This helps identify tasks that don't need to reset the environment between runs
and creates optimized parallel processing configurations.
"""

import os
import json
import glob
from collections import defaultdict

def analyze_json_files(directory_path):
    """Analyze all JSON files in the directory for their properties."""
    
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
                file_id = os.path.splitext(filename)[0]
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
                        reset_false.append((file_id, task_id, intent))
                    else:
                        reset_true.append((file_id, task_id, intent))
                else:
                    stats["no_reset_flag"] += 1
                    no_reset_flag.append((file_id, task_id, intent))
                
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON file {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    return reset_false, reset_true, no_reset_flag, stats

def generate_parallel_script(task_ids, num_parallel=8, batch_size=5):
    """
    Generate a shell script for parallel processing based on task IDs.
    
    Args:
        task_ids: List of task IDs (file IDs) to process
        num_parallel: Number of parallel processes to use
        batch_size: Number of tasks per batch
    """
    # Make task_ids into a list of strings
    task_ids = [str(tid) for tid in task_ids]
    
    script_content = f"""#!/bin/bash
export PYTHONPATH=$(pwd)
python runners/eval/eval_vwa_parallel.py \\
    --env_name gitlab \\
    --save_dir data/webarena/eval_results/react_text/gitlab_no_reset \\
    --eval_script shells/gitlab/react_text_parallel.sh \\
    --run_mode greedy \\
    --test_indices {','.join(task_ids)} \\
    --num_parallel {num_parallel} \\
    --main_api_providers {','.join(['openai'] * num_parallel)} \\
    --num_task_per_script {batch_size} \\
    --num_task_per_reset {len(task_ids)}
"""
    return script_content

def write_to_file(content, file_path):
    """Write content to a file."""
    with open(file_path, 'w') as file:
        file.write(content)
    
    # Make the file executable
    os.chmod(file_path, 0o755)
    print(f"Script written to {file_path} and made executable")

def main():
    directory_path = "configs/webarena/test_gitlab_v2"
    
    reset_false, reset_true, no_reset_flag, stats = analyze_json_files(directory_path)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Print detailed information for files with require_reset=false
    print("\n=== Files with require_reset=false ===")
    if reset_false:
        print(f"Found {len(reset_false)} files with require_reset=false:")
        for file_id, task_id, intent in reset_false:
            print(f"- {file_id}.json (Task ID: {task_id}): {intent}")
            
        # Generate list of tasks that don't need reset
        no_reset_task_ids = [file_id for file_id, _, _ in reset_false]
        
        # Generate parallel script for these tasks
        script_content = generate_parallel_script(no_reset_task_ids)
        script_path = "run_parallel_no_reset.sh"
        write_to_file(script_content, script_path)
        
        # Generate task list file for future reference
        task_list_content = "\n".join(no_reset_task_ids)
        write_to_file(task_list_content, "gitlab_no_reset_tasks.txt")
    else:
        print("No files found with require_reset=false")
    
    # Print files missing the reset flag
    if no_reset_flag:
        print("\n=== Files missing require_reset flag ===")
        print(f"Found {len(no_reset_flag)} files without require_reset flag:")
        for file_id, task_id, intent in no_reset_flag:
            print(f"- {file_id}.json (Task ID: {task_id}): {intent}")

if __name__ == "__main__":
    main() 