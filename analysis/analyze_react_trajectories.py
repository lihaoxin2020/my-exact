import json
import os
import argparse
from collections import Counter
import matplotlib.pyplot as plt

def analyze_react_trajectories(trajectory_dir, output_file=None):
    """
    Analyzes ReACT agent trajectories to extract metrics and insights.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        output_file: Optional output file for results
    
    Returns:
        Dict containing analysis results
    """
    results = {
        'action_counts': Counter(),
        'success_rate': 0,
        'avg_trajectory_length': 0,
        'trajectories': []
    }
    
    # Count trajectories
    trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.pkl.xz') or f.endswith('.json')]
    
    if not trajectory_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return results
    
    successful = 0
    total_steps = 0
    
    for file in trajectory_files:
        filepath = os.path.join(trajectory_dir, file)
        
        try:
            # Process trajectory file
            if file.endswith('.json'):
                with open(filepath, 'r') as f:
                    trajectory = json.load(f)
            else:
                # For compressed pickle files, log that we'd need additional processing
                print(f"Skipping {file} - compressed format")
                continue
            
            # Extract success status if available
            if 'success' in trajectory:
                if trajectory['success']:
                    successful += 1
            
            # Count actions if available
            if 'actions' in trajectory:
                for action in trajectory['actions']:
                    if 'action_type' in action:
                        results['action_counts'][action['action_type']] += 1
                
                total_steps += len(trajectory['actions'])
            
            # Store basic trajectory info
            results['trajectories'].append({
                'file': file,
                'steps': len(trajectory.get('actions', [])),
                'success': trajectory.get('success', False)
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Calculate metrics
    if trajectory_files:
        results['success_rate'] = successful / len(trajectory_files)
        results['avg_trajectory_length'] = total_steps / len(trajectory_files)
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze ReACT agent trajectories')
    parser.add_argument('--trajectory_dir', type=str, required=True, help='Directory containing trajectory files')
    parser.add_argument('--output_file', type=str, help='Output file for analysis results')
    args = parser.parse_args()
    
    results = analyze_react_trajectories(args.trajectory_dir, args.output_file)
    
    # Print summary
    print(f"Analyzed {len(results['trajectories'])} trajectories")
    print(f"Success rate: {results['success_rate']:.2f}")
    print(f"Average trajectory length: {results['avg_trajectory_length']:.2f} steps")
    print("Top action types:")
    for action_type, count in results['action_counts'].most_common(5):
        print(f"  {action_type}: {count}")

if __name__ == "__main__":
    main() 