import os
import json
import shutil
import lzma
import pickle
from tqdm import tqdm

# Path configuration
base_dir = "/Users/nikhilkhandekar/Documents/my-exact/data/webarena/eval_results/react_text/gitlab_parallel_20250502_201103"
output_dir = "/Users/nikhilkhandekar/Documents/my-exact/data/training/react_text_successful"

# Create output directory structure
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)

def get_successful_tasks():
    """Identify tasks with a score of 1.0 in their performance file"""
    successful_tasks = []
    
    # Loop through all task directories
    for item in os.listdir(base_dir):
        if not item.startswith("task_"):
            continue
            
        task_id = item.split("_")[1]
        perf_file = os.path.join(base_dir, item, "performances", f"performance_{task_id}.json")
        
        # Check if performance file exists
        if not os.path.exists(perf_file):
            continue
            
        # Read performance file
        try:
            with open(perf_file, 'r') as f:
                perf_data = json.load(f)
                
            # Check if score is 1.0
            if perf_data.get("scores") == 1.0:
                successful_tasks.append(task_id)
                print(f"Found successful task: {task_id}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading performance file for task {task_id}: {e}")
    
    return successful_tasks

def extract_task_data(task_id):
    """Extract trajectory and config for a successful task"""
    task_dir = os.path.join(base_dir, f"task_{task_id}")
    
    # Copy trajectory file
    traj_file = os.path.join(task_dir, "trajectories", f"task_{task_id}.pkl.xz")
    if os.path.exists(traj_file):
        shutil.copy2(traj_file, os.path.join(output_dir, "trajectories", f"task_{task_id}.pkl.xz"))
    else:
        print(f"Trajectory file for task {task_id} not found")
        return False
    
    # Copy config file
    config_file = os.path.join(task_dir, "config.json")
    if os.path.exists(config_file):
        shutil.copy2(config_file, os.path.join(output_dir, f"config_{task_id}.json"))
    
    # Copy performance file
    perf_file = os.path.join(task_dir, "performances", f"performance_{task_id}.json")
    if os.path.exists(perf_file):
        shutil.copy2(perf_file, os.path.join(output_dir, f"performance_{task_id}.json"))
    
    return True

def extract_task_info(task_id):
    """Extract task information including intent and task details"""
    task_dir = os.path.join(base_dir, f"task_{task_id}")
    traj_file = os.path.join(task_dir, "trajectories", f"task_{task_id}.pkl.xz")
    
    try:
        with lzma.open(traj_file, "rb") as f:
            trajectory_data = pickle.load(f)
        
        # Extract task info
        if "task_info" in trajectory_data:
            task_info = trajectory_data["task_info"]
            intent = task_info.get("intent", "No intent available")
            
            # Save task info summary
            with open(os.path.join(output_dir, f"task_info_{task_id}.json"), 'w') as f:
                json.dump({
                    "task_id": task_id,
                    "intent": intent,
                    "score": 1.0
                }, f, indent=2)
                
            return intent
    except Exception as e:
        print(f"Error extracting info for task {task_id}: {e}")
    
    return None

def main():
    """Extract all successful trajectories and their associated data"""
    # Get successful tasks
    successful_tasks = get_successful_tasks()
    print(f"Found {len(successful_tasks)} successful tasks")
    
    # Save list of successful tasks
    with open(os.path.join(output_dir, "successful_tasks.json"), 'w') as f:
        json.dump(successful_tasks, f, indent=2)
    
    # Extract data for each successful task
    task_summaries = []
    for task_id in tqdm(successful_tasks, desc="Extracting successful trajectories"):
        if extract_task_data(task_id):
            intent = extract_task_info(task_id)
            if intent:
                task_summaries.append({
                    "task_id": task_id,
                    "intent": intent
                })
    
    # Save task summaries
    with open(os.path.join(output_dir, "task_summaries.json"), 'w') as f:
        json.dump(task_summaries, f, indent=2)
    
    print(f"Successfully extracted {len(task_summaries)} trajectories to {output_dir}")
    print("These trajectories can now be processed using process_react_trajectories.py")

if __name__ == "__main__":
    main() 