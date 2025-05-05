#!/usr/bin/env python3
"""
Analyze trajectories from the o4_mini model to extract useful statistics
"""

import os
import json
import pickle
import lzma
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def load_trajectory(file_path):
    """Load a trajectory from a .pkl.xz file"""
    try:
        if file_path.endswith('.xz'):
            with lzma.open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
        # Handle different trajectory formats
        if isinstance(data, dict):
            if "trajectory" in data:
                return data["trajectory"], data.get("score", None)
            elif "task_data" in data and "trajectory" in data["task_data"]:
                return data["task_data"]["trajectory"], data.get("score", None)
        elif isinstance(data, list):
            return data, None
            
        return None, None
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None

def count_actions(trajectory):
    """Count the number of actions in a trajectory"""
    if not trajectory:
        return 0
        
    action_count = 0
    for item in trajectory:
        if not isinstance(item, dict):  # Actions are typically non-dict objects
            action_count += 1
    return action_count

def extract_action_types(trajectory):
    """Extract the types of actions taken in a trajectory"""
    action_types = []
    
    for item in trajectory:
        if not isinstance(item, dict):  # This is an action
            # Try to get action type
            if hasattr(item, 'action_type'):
                action_types.append(item.action_type)
            elif hasattr(item, 'type'):
                action_types.append(item.type)
            else:
                # Just use the class name as fallback
                action_types.append(item.__class__.__name__)
    
    return action_types

def load_performance_file(task_id, perf_dir):
    """Load a performance file for a given task ID"""
    perf_file = os.path.join(perf_dir, f"performance_{task_id}.json")
    if os.path.exists(perf_file):
        try:
            with open(perf_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {perf_file}: {str(e)}")
    return None

def analyze_trajectories(traj_dir, perf_dir=None, output_file=None):
    """Analyze trajectories and generate statistics"""
    results = {
        "total_trajectories": 0,
        "successful_tasks": 0,
        "failed_tasks": 0,
        "action_counts": [],
        "action_types": Counter(),
        "task_times": [],
        "success_rate": 0.0,
        "avg_num_actions": 0.0,
        "max_num_actions": 0,
        "min_num_actions": float('inf'),
        "tasks": {}
    }
    
    task_files = [f for f in os.listdir(traj_dir) if f.endswith('.pkl.xz') or f.endswith('.pkl')]
    
    for file_name in task_files:
        file_path = os.path.join(traj_dir, file_name)
        
        # Extract task ID from filename
        try:
            task_id = int(file_name.split('_')[1].split('.')[0])
        except (ValueError, IndexError):
            print(f"Could not extract task ID from {file_name}, skipping")
            continue
            
        # Load trajectory
        trajectory, score = load_trajectory(file_path)
        
        if trajectory is None:
            print(f"Could not load trajectory from {file_path}, skipping")
            continue
            
        # Load performance data if available
        perf_data = None
        if perf_dir:
            perf_data = load_performance_file(task_id, perf_dir)
            
        # Count actions
        action_count = count_actions(trajectory)
        results["action_counts"].append(action_count)
        
        # Extract action types
        action_types = extract_action_types(trajectory)
        for action_type in action_types:
            results["action_types"][action_type] += 1
            
        # Get task success/failure
        is_successful = False
        if score is not None:
            is_successful = score > 0
        elif perf_data and "scores" in perf_data:
            is_successful = perf_data["scores"] > 0
            
        if is_successful:
            results["successful_tasks"] += 1
        else:
            results["failed_tasks"] += 1
            
        # Get task time if available
        if perf_data and "times" in perf_data:
            results["task_times"].append(perf_data["times"])
            
        # Store per-task details
        results["tasks"][task_id] = {
            "num_actions": action_count,
            "successful": is_successful,
            "action_types": Counter(action_types),
        }
        if perf_data and "times" in perf_data:
            results["tasks"][task_id]["time"] = perf_data["times"]
        
        results["total_trajectories"] += 1
        
    # Calculate summary statistics
    if results["action_counts"]:
        results["avg_num_actions"] = np.mean(results["action_counts"])
        results["max_num_actions"] = max(results["action_counts"])
        results["min_num_actions"] = min(results["action_counts"])
        
    if results["total_trajectories"] > 0:
        results["success_rate"] = (results["successful_tasks"] / results["total_trajectories"]) * 100
        
    if results["task_times"]:
        results["avg_task_time"] = np.mean(results["task_times"])
        results["max_task_time"] = max(results["task_times"])
        results["min_task_time"] = min(results["task_times"])
    
    # Convert action_types Counter to a regular dict for JSON serialization
    results["action_types"] = dict(results["action_types"])
    
    # For each task, convert action_types Counter to dict
    for task_id in results["tasks"]:
        results["tasks"][task_id]["action_types"] = dict(results["tasks"][task_id]["action_types"])
    
    # Save results if output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    return results

def print_summary(results):
    """Print a summary of the trajectory analysis"""
    print(f"\n=== Trajectory Analysis Summary ===")
    print(f"Total trajectories analyzed: {results['total_trajectories']}")
    print(f"Success rate: {results['success_rate']:.2f}%")
    print(f"Successful tasks: {results['successful_tasks']}")
    print(f"Failed tasks: {results['failed_tasks']}")
    
    print(f"\nActions:")
    print(f"Average actions per task: {results['avg_num_actions']:.2f}")
    print(f"Max actions: {results['max_num_actions']}")
    print(f"Min actions: {results['min_num_actions']}")
    
    if "avg_task_time" in results:
        print(f"\nTask Times:")
        print(f"Average task time: {results['avg_task_time']:.2f} seconds")
        print(f"Max task time: {results['max_task_time']:.2f} seconds")
        print(f"Min task time: {results['min_task_time']:.2f} seconds")
    
    print(f"\nMost Common Action Types:")
    for action_type, count in sorted(results["action_types"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {action_type}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Analyze o4_mini trajectories")
    parser.add_argument("--traj_dir", default="/Users/nikhilkhandekar/Documents/my-exact/trajectories_o4_mini",
                        help="Directory containing trajectory files")
    parser.add_argument("--perf_dir", default="/Users/nikhilkhandekar/Documents/my-exact/performances_o4_mini",
                        help="Directory containing performance files")
    parser.add_argument("--output", default="o4_mini_analysis.json",
                        help="Output file for analysis results (JSON)")
    
    args = parser.parse_args()
    
    print(f"Analyzing trajectories in {args.traj_dir}")
    print(f"Using performance data from {args.perf_dir}")
    
    results = analyze_trajectories(args.traj_dir, args.perf_dir, args.output)
    print_summary(results)
    
    if args.output:
        print(f"\nDetailed analysis saved to {args.output}")

if __name__ == "__main__":
    main() 