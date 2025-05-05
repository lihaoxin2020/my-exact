#!/usr/bin/env python3
"""
Script to analyze processed MCTS trajectories after they've been processed
by simplified_tree_to_data.py, focusing on understanding why flattening fails.
"""

import os
import pickle
import lzma
import json
import argparse
from collections import defaultdict

def inspect_processed_file(file_path):
    """Inspect a processed JSONL file to understand its structure."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Found {len(examples)} examples in {file_path}")
    
    # Analyze backtracking content
    backtracking_counts = []
    for example in examples:
        messages = example.get("messages", [])
        backtrack_count = 0
        
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "try a different approach" in content or "go_back" in content.lower():
                    backtrack_count += 1
        
        backtracking_counts.append(backtrack_count)
    
    # Calculate statistics
    if backtracking_counts:
        max_backtrack = max(backtracking_counts)
        avg_backtrack = sum(backtracking_counts) / len(backtracking_counts)
        
        # Create a histogram
        histogram = defaultdict(int)
        for count in backtracking_counts:
            histogram[count] += 1
        
        # Print results
        print(f"Max backtracking actions per example: {max_backtrack}")
        print(f"Average backtracking actions per example: {avg_backtrack:.2f}")
        print("\nBacktracking distribution:")
        for count in sorted(histogram.keys()):
            percentage = 100 * histogram[count] / len(backtracking_counts)
            print(f"  {count} backtracking actions: {histogram[count]} examples ({percentage:.1f}%)")
        
        # Inspect examples with most and least backtracking
        if max_backtrack > 0:
            max_idx = backtracking_counts.index(max_backtrack)
            print(f"\nExample with most backtracking ({max_backtrack} actions):")
            print_example_summary(examples[max_idx])
        
        min_idx = backtracking_counts.index(min(backtracking_counts))
        print(f"\nExample with least backtracking ({backtracking_counts[min_idx]} actions):")
        print_example_summary(examples[min_idx])
    else:
        print("No examples found with backtracking information.")

def print_example_summary(example):
    """Print a summary of an example from the processed data."""
    messages = example.get("messages", [])
    print(f"Task ID: {example.get('task_id')}")
    print(f"Number of messages: {len(messages)}")
    
    # Print system prompt
    for msg in messages:
        if msg.get("role") == "system":
            print(f"Intent: {msg.get('content')}")
            break
    
    # Print brief summary of each turn
    print("\nTurns summary:")
    for i, msg in enumerate(messages):
        role = msg.get("role")
        
        if role == "user":
            content = msg.get("content", "")
            # Extract URL if present
            url = "unknown"
            if "Current URL:" in content:
                url_line = [line for line in content.split("\n") if "Current URL:" in line]
                if url_line:
                    url = url_line[0].replace("Current URL:", "").strip()
            print(f"[{i}] User: URL = {url}")
            
        elif role == "assistant":
            content = msg.get("content", "")
            # Extract action
            action = "unknown"
            if "```" in content:
                import re
                match = re.search(r"```(.+?)```", content)
                if match:
                    action = match.group(1)
            
            # Check if it's a backtracking action
            is_backtrack = "try a different approach" in content or "go_back" in content.lower()
            backtrack_str = " (BACKTRACK)" if is_backtrack else ""
            
            print(f"[{i}] Assistant{backtrack_str}: {action}")
    
    # Print the last user turn and assistant response in full
    if len(messages) >= 2:
        last_user_idx = max([i for i, msg in enumerate(messages) if msg.get("role") == "user"])
        if last_user_idx < len(messages) - 1:
            last_assistant_idx = last_user_idx + 1
            
            print("\nLast user observation (excerpt):")
            user_content = messages[last_user_idx].get("content", "")
            print(user_content[:200] + "..." if len(user_content) > 200 else user_content)
            
            print("\nLast assistant response:")
            assistant_content = messages[last_assistant_idx].get("content", "")
            print(assistant_content)

def analyze_flattening_function(trajectory_dir):
    """Test the flattening function on actual trajectories to see why it fails."""
    from simplified_tree_to_data import load_trajectory, _remove_backtrack_to_normal_action, _filter_traj
    
    traj_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.pkl.xz')]
    if not traj_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return
    
    print(f"Testing flattening on {len(traj_files)} trajectories")
    
    # Track success/failure statistics
    success_count = 0
    failure_count = 0
    failure_reasons = defaultdict(int)
    
    # Analyze a subset for detailed inspection
    max_to_analyze = min(10, len(traj_files))
    
    for traj_file in traj_files[:max_to_analyze]:
        try:
            # Load the trajectory
            traj_path = os.path.join(trajectory_dir, traj_file)
            trajectory = load_trajectory(traj_path)
            
            if not trajectory:
                continue
            
            # Apply flattening
            print(f"\n{'='*80}\nAnalyzing {traj_file}")
            print(f"Original trajectory length: {len(trajectory)}")
            
            # Count states, actions, and backtracking before flattening
            state_count = sum(1 for item in trajectory if isinstance(item, dict))
            action_count = sum(1 for item in trajectory if hasattr(item, 'raw_prediction'))
            backtrack_count = sum(1 for item in trajectory if hasattr(item, 'raw_prediction') and 
                                 (hasattr(item, '_direction') and item._direction == 'backtrack'))
            
            print(f"States: {state_count}, Actions: {action_count}, Backtracking actions: {backtrack_count}")
            
            # Apply flattening
            flattened = _remove_backtrack_to_normal_action(trajectory)
            print(f"Flattened trajectory length: {len(flattened)}")
            
            # Count states and actions after flattening
            flat_state_count = sum(1 for item in flattened if isinstance(item, dict))
            flat_action_count = sum(1 for item in flattened if hasattr(item, 'raw_prediction'))
            
            print(f"After flattening - States: {flat_state_count}, Actions: {flat_action_count}")
            
            # Check why it might be failing
            if len(flattened) < 2:
                print("FAILURE: Flattened trajectory too short")
                failure_count += 1
                failure_reasons["too_short"] += 1
                continue
            
            # See if it passes the filter
            should_filter = _filter_traj(flattened)
            if should_filter:
                print("FAILURE: Flattened trajectory filtered out")
                failure_count += 1
                
                # Check why it was filtered
                if not hasattr(flattened[-1], 'raw_prediction'):
                    print("  Reason: Last item is not an action")
                    failure_reasons["last_not_action"] += 1
                elif len(flattened) < 2 or not isinstance(flattened[-2], dict):
                    print("  Reason: No state before last action")
                    failure_reasons["no_state_before_action"] += 1
                elif hasattr(flattened[-1], 'element_id') and flattened[-1].element_id:
                    element_id = flattened[-1].element_id
                    observation = flattened[-2].get('observation', {}).get('text', '')
                    if f"[{element_id}]" not in observation:
                        print(f"  Reason: Element ID {element_id} not found in observation")
                        failure_reasons["element_id_missing"] += 1
                else:
                    print("  Reason: Unknown")
                    failure_reasons["unknown"] += 1
            else:
                print("SUCCESS: Flattened trajectory valid")
                success_count += 1
                
            # Print sample of trajectory before and after
            print("\nBefore flattening (first 5 items):")
            for i, item in enumerate(trajectory[:5]):
                if isinstance(item, dict):
                    print(f"[{i}] State: URL={item.get('url', 'N/A')}")
                elif hasattr(item, 'raw_prediction'):
                    direction = getattr(item, '_direction', 'unknown')
                    print(f"[{i}] Action ({direction}): {item.raw_prediction[:50]}...")
            
            print("\nAfter flattening (first 5 items):")
            for i, item in enumerate(flattened[:5]):
                if isinstance(item, dict):
                    print(f"[{i}] State: URL={item.get('url', 'N/A')}")
                elif hasattr(item, 'raw_prediction'):
                    direction = getattr(item, '_direction', 'unknown')
                    print(f"[{i}] Action ({direction}): {item.raw_prediction[:50]}...")
            
        except Exception as e:
            print(f"Error analyzing {traj_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print(f"SUMMARY: Analyzed {max_to_analyze} trajectories")
    print(f"Success: {success_count} ({100*success_count/max_to_analyze:.1f}%)")
    print(f"Failure: {failure_count} ({100*failure_count/max_to_analyze:.1f}%)")
    
    if failure_reasons:
        print("\nFailure reasons:")
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count} ({100*count/failure_count:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze processed MCTS trajectories")
    parser.add_argument("--processed_file", type=str, help="Path to processed JSONL file")
    parser.add_argument("--trajectory_dir", type=str, help="Directory with original trajectory files")
    
    args = parser.parse_args()
    
    if args.processed_file:
        inspect_processed_file(args.processed_file)
    
    if args.trajectory_dir:
        analyze_flattening_function(args.trajectory_dir)
    
    if not args.processed_file and not args.trajectory_dir:
        parser.print_help()

if __name__ == "__main__":
    main() 