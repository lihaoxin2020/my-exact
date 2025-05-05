#!/usr/bin/env python3
"""
Analyze trajectory success rates from performance JSON files.
This script checks all performance_*.json files in a specified directory
and reports which trajectories were successful.
"""

import os
import json
import argparse
import glob
from pathlib import Path
import sys
from typing import Dict, List, Tuple

def analyze_performance_files(directory: str) -> Tuple[List[int], List[int], Dict]:
    """
    Analyze all performance_*.json files in the specified directory.
    
    Args:
        directory: Path to the directory containing performance files
    
    Returns:
        Tuple of (successful_indices, failed_indices, detailed_results)
    """
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
        
    # Find all performance JSON files
    pattern = os.path.join(directory, "performance_*.json")
    performance_files = glob.glob(pattern)
    
    if not performance_files:
        print(f"No performance_*.json files found in {directory}")
        sys.exit(1)
    
    successful_indices = []
    failed_indices = []
    detailed_results = {}
    
    for file_path in sorted(performance_files):
        filename = os.path.basename(file_path)
        try:
            # Extract the index from filename (performance_X.json)
            index = int(filename.split("_")[1].split(".")[0])
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if the task was successful
            # Different performance files may have different success indicators
            # Adjust these checks based on your actual JSON structure
            is_success = False
            reasons = []
            
            # Check potential success indicators
            if "success" in data:
                is_success = data["success"]
                if is_success:
                    reasons.append("success=True")
            
            if "is_success" in data:
                is_success = data["is_success"]
                if is_success:
                    reasons.append("is_success=True")
            
            if "completed" in data:
                is_success = data["completed"]
                if is_success:
                    reasons.append("completed=True")
            
            if "score" in data:
                score = data["score"]
                if isinstance(score, (int, float)) and score > 0:
                    is_success = True
                    reasons.append(f"score={score}")
            
            # Record the result
            if is_success:
                successful_indices.append(index)
                status = "SUCCESS"
            else:
                failed_indices.append(index)
                status = "FAILED"
            
            # Store detailed results
            detailed_results[index] = {
                "file": filename,
                "status": status,
                "reasons": reasons if reasons else ["No success indicators found"],
                "data": data
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error processing {filename}: {str(e)}")
            failed_indices.append(index)
            detailed_results[index] = {
                "file": filename,
                "status": "ERROR",
                "reasons": [str(e)],
                "data": None
            }
    
    return successful_indices, failed_indices, detailed_results

def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory success rates')
    parser.add_argument('--directory', '-d', default=None,
                        help='Directory containing performance_*.json files')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file for detailed results (JSON format)')
    parser.add_argument('--list-successful', '-s', action='store_true',
                        help='Only list successful trajectory indices')
    parser.add_argument('--list-failed', '-f', action='store_true',
                        help='Only list failed trajectory indices')
    
    args = parser.parse_args()
    
    # If no directory specified, try to find the latest results directory
    if args.directory is None:
        search_dir = "data/webarena/eval_results/search_agents"
        if os.path.exists(search_dir):
            dirs = [d for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d)) and d.startswith("gitlab_subset_")]
            if dirs:
                latest_dir = max(dirs)
                args.directory = os.path.join(search_dir, latest_dir, "performances")
                print(f"Using latest results directory: {args.directory}")
            else:
                print(f"No gitlab_subset_* directories found in {search_dir}")
                sys.exit(1)
        else:
            print(f"Directory {search_dir} not found")
            sys.exit(1)
    
    # Run the analysis
    successful_indices, failed_indices, detailed_results = analyze_performance_files(args.directory)
    
    # Generate report
    if args.list_successful:
        print("Successful trajectories:", ", ".join(map(str, sorted(successful_indices))))
    elif args.list_failed:
        print("Failed trajectories:", ", ".join(map(str, sorted(failed_indices))))
    else:
        print("\n=== Trajectory Success Analysis ===")
        print(f"Directory: {args.directory}")
        print(f"Total trajectories: {len(detailed_results)}")
        print(f"Successful trajectories: {len(successful_indices)} ({len(successful_indices) / len(detailed_results) * 100:.1f}%)")
        print(f"Failed trajectories: {len(failed_indices)} ({len(failed_indices) / len(detailed_results) * 100:.1f}%)")
        
        print("\nSuccessful indices:", ", ".join(map(str, sorted(successful_indices))))
        print("\nFailed indices:", ", ".join(map(str, sorted(failed_indices))))
    
    # Save detailed results if requested
    if args.output:
        output_path = args.output
        with open(output_path, 'w') as f:
            json.dump({
                "summary": {
                    "total": len(detailed_results),
                    "successful": len(successful_indices),
                    "failed": len(failed_indices),
                    "success_rate": len(successful_indices) / len(detailed_results) if detailed_results else 0
                },
                "successful_indices": sorted(successful_indices),
                "failed_indices": sorted(failed_indices),
                "detailed_results": detailed_results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main() 