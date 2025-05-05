#!/usr/bin/env python3
"""
Script to analyze MCTS trajectories and understand their structure,
particularly focusing on backtracking patterns.
"""

import os
import pickle
import lzma
import argparse
import copy
from tqdm import tqdm
import json

# Define constants for identifying backtracking
BACKTRACK_KWD = "We should take a step back"
GO_BACK_KWD = "go_back"

def load_trajectory(traj_path):
    """Load a trajectory from a compressed pickle file"""
    try:
        with lzma.open(traj_path, "rb") as fread:
            traj = pickle.load(fread)
        return traj
    except Exception as e:
        print(f"Error loading trajectory {traj_path}: {e}")
        return None

def get_direction_info(item):
    """Get the direction info (forward/backtrack) from an action"""
    if hasattr(item, '_direction'):
        return item._direction
    if hasattr(item, 'raw_prediction'):
        if BACKTRACK_KWD in item.raw_prediction or GO_BACK_KWD in item.raw_prediction.lower():
            return "backtrack"
    return "forward"

def count_backtracking_actions(trajectory):
    """Count the number of backtracking actions in a trajectory"""
    backtrack_count = 0
    total_actions = 0
    
    for item in trajectory:
        if hasattr(item, 'raw_prediction'):  # This is an action
            total_actions += 1
            if get_direction_info(item) == "backtrack":
                backtrack_count += 1
                
    return backtrack_count, total_actions

def analyze_trajectory(trajectory):
    """Analyze a single trajectory and return statistics"""
    if not trajectory:
        return None
    
    # Count states and actions
    state_count = 0
    action_count = 0
    backtrack_count = 0
    forward_count = 0
    
    # Check alternating pattern
    is_alternating = True
    prev_type = None
    
    for item in trajectory:
        if isinstance(item, dict):  # State
            state_count += 1
            current_type = "state"
        elif hasattr(item, 'raw_prediction'):  # Action
            action_count += 1
            current_type = "action"
            
            # Check direction
            if get_direction_info(item) == "backtrack":
                backtrack_count += 1
            else:
                forward_count += 1
        else:
            current_type = "unknown"
        
        # Check alternating pattern
        if prev_type is not None and prev_type == current_type:
            is_alternating = False
        prev_type = current_type
    
    # Check state-action-state pattern
    starts_with_state = len(trajectory) > 0 and isinstance(trajectory[0], dict)
    
    return {
        "total_length": len(trajectory),
        "state_count": state_count,
        "action_count": action_count,
        "backtrack_count": backtrack_count,
        "forward_count": forward_count,
        "is_alternating": is_alternating,
        "starts_with_state": starts_with_state
    }

def try_flattening(trajectory):
    """Attempt to flatten a trajectory by removing backtracking actions"""
    flattened_traj = []
    
    for item in trajectory:
        if hasattr(item, 'raw_prediction'):  # Action
            # Skip backtracking actions
            if get_direction_info(item) != "backtrack":
                flattened_traj.append(copy.deepcopy(item))
        else:  # State or other
            flattened_traj.append(copy.deepcopy(item))
    
    # Simple check if the flattened trajectory is valid
    has_state_action_pair = False
    for i in range(len(flattened_traj) - 1):
        if isinstance(flattened_traj[i], dict) and hasattr(flattened_traj[i+1], 'raw_prediction'):
            has_state_action_pair = True
            break
    
    return flattened_traj, has_state_action_pair

def print_trajectory_excerpt(trajectory, max_items=10):
    """Print a short excerpt of a trajectory for inspection"""
    print(f"\nTrajectory excerpt (up to {max_items} items):")
    
    for i, item in enumerate(trajectory[:max_items]):
        if isinstance(item, dict):  # State
            print(f"[{i}] State: URL = {item.get('url', 'N/A')}")
            if 'observation' in item and 'text' in item['observation']:
                text = item['observation']['text']
                print(f"    Text excerpt: {text[:100]}..." if len(text) > 100 else f"    Text: {text}")
        elif hasattr(item, 'raw_prediction'):  # Action
            direction = get_direction_info(item)
            action_type = getattr(item, 'action_type', 'unknown')
            print(f"[{i}] Action ({direction}): {action_type}")
            
            # Extract action from raw prediction
            raw = item.raw_prediction
            shortened = raw[:100] + "..." if len(raw) > 100 else raw
            print(f"    Raw: {shortened}")
            
            # Show element ID if present
            if hasattr(item, 'element_id') and item.element_id:
                print(f"    Element ID: {item.element_id}")
        else:
            print(f"[{i}] Unknown item type: {type(item)}")
    
    if len(trajectory) > max_items:
        print(f"... (plus {len(trajectory) - max_items} more items)")

def analyze_trajectories(trajectory_dir):
    """Analyze all trajectory files in a directory"""
    traj_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.pkl.xz')]
    print(f"Found {len(traj_files)} trajectory files")
    
    # Statistics collection
    all_stats = []
    tasks_with_all_backtracking = []
    tasks_with_no_backtracking = []
    
    for traj_file in tqdm(traj_files, desc="Analyzing trajectories"):
        try:
            # Extract task ID
            task_id = int(traj_file.split('.')[0].split('_')[1])
            
            # Load trajectory
            traj_path = os.path.join(trajectory_dir, traj_file)
            trajectory = load_trajectory(traj_path)
            
            if not trajectory:
                continue
            
            # Analyze
            stats = analyze_trajectory(trajectory)
            stats["task_id"] = task_id
            stats["filename"] = traj_file
            all_stats.append(stats)
            
            # Check if trajectory has all backtracking actions
            if stats["action_count"] > 0 and stats["backtrack_count"] == stats["action_count"]:
                tasks_with_all_backtracking.append(task_id)
            
            # Check if trajectory has no backtracking actions
            if stats["action_count"] > 0 and stats["backtrack_count"] == 0:
                tasks_with_no_backtracking.append(task_id)
                
            # Try flattening and check if it's valid
            flattened, is_valid = try_flattening(trajectory)
            stats["flattened_length"] = len(flattened)
            stats["flattened_valid"] = is_valid
            
            # If all actions are backtracking or this is our first file, print a sample
            if (stats["action_count"] > 0 and stats["backtrack_count"] == stats["action_count"]) or \
               traj_file == traj_files[0]:
                print(f"\n{'='*80}\nExamining trajectory {traj_file} (Task {task_id})")
                print(f"Stats: {stats}")
                print_trajectory_excerpt(trajectory)
                print("\nAfter flattening:")
                print(f"Length: {len(flattened)}, Valid: {is_valid}")
                print_trajectory_excerpt(flattened)
                
        except Exception as e:
            print(f"Error analyzing {traj_file}: {e}")
    
    # Print summary statistics
    total_trajectories = len(all_stats)
    
    if total_trajectories > 0:
        # Calculate averages
        avg_length = sum(s["total_length"] for s in all_stats) / total_trajectories
        avg_states = sum(s["state_count"] for s in all_stats) / total_trajectories
        avg_actions = sum(s["action_count"] for s in all_stats) / total_trajectories
        avg_backtrack = sum(s["backtrack_count"] for s in all_stats) / total_trajectories
        
        # Count trajectories that become invalid after flattening
        invalid_after_flattening = sum(1 for s in all_stats if not s["flattened_valid"])
        
        print("\n" + "="*80)
        print(f"SUMMARY STATISTICS ({total_trajectories} trajectories)")
        print(f"Average trajectory length: {avg_length:.2f}")
        print(f"Average states per trajectory: {avg_states:.2f}")
        print(f"Average actions per trajectory: {avg_actions:.2f}")
        print(f"Average backtracking actions per trajectory: {avg_backtrack:.2f}")
        print(f"Trajectories with all backtracking actions: {len(tasks_with_all_backtracking)} ({100*len(tasks_with_all_backtracking)/total_trajectories:.1f}%)")
        print(f"Trajectories with no backtracking actions: {len(tasks_with_no_backtracking)} ({100*len(tasks_with_no_backtracking)/total_trajectories:.1f}%)")
        print(f"Trajectories that become invalid after flattening: {invalid_after_flattening} ({100*invalid_after_flattening/total_trajectories:.1f}%)")
        
        # Save the results to a JSON file
        with open("trajectory_analysis.json", "w") as f:
            json.dump({
                "summary": {
                    "total_trajectories": total_trajectories,
                    "avg_length": avg_length,
                    "avg_states": avg_states,
                    "avg_actions": avg_actions,
                    "avg_backtrack": avg_backtrack,
                    "trajectories_all_backtracking": len(tasks_with_all_backtracking),
                    "trajectories_no_backtracking": len(tasks_with_no_backtracking),
                    "invalid_after_flattening": invalid_after_flattening
                },
                "tasks_with_all_backtracking": tasks_with_all_backtracking,
                "tasks_with_no_backtracking": tasks_with_no_backtracking,
                "trajectory_stats": all_stats
            }, f, indent=2)
        
        print(f"Results saved to trajectory_analysis.json")
    else:
        print("No valid trajectories found to analyze.")

def main():
    parser = argparse.ArgumentParser(description="Analyze MCTS trajectory structure and backtracking patterns")
    parser.add_argument("--trajectory_dir", type=str, required=True, 
                        help="Directory containing trajectory files (.pkl.xz)")
    
    args = parser.parse_args()
    analyze_trajectories(args.trajectory_dir)

if __name__ == "__main__":
    main() 