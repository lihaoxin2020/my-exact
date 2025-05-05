import os
import json
import shutil
import lzma
import pickle
from tqdm import tqdm

# Path configuration
base_dir = "/Users/nikhilkhandekar/Documents/my-exact/data/webarena/eval_results/react_text/gitlab_parallel_20250502_201103"
filtered_json_path = "/Users/nikhilkhandekar/Documents/my-exact/configs/webarena/filtered_test_gitlab_v2.json"
output_dir = "/Users/nikhilkhandekar/Documents/my-exact/data/training/react_filtered"

# Create output directory structure
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)

def get_filtered_task_ids_and_intents():
    """Get the task IDs and intents from the filtered_test_gitlab_v2.json file"""
    with open(filtered_json_path, 'r') as f:
        data = json.load(f)
    
    # Extract task IDs and intents
    task_data = {}
    for item in data:
        task_id = str(item["task_id"])
        intent = item.get("intent", "No intent available")
        task_data[task_id] = {"intent": intent}
    
    filtered_task_ids = list(task_data.keys())
    print(f"Found {len(filtered_task_ids)} task IDs in filtered_test_gitlab_v2.json")
    
    # Print a few samples
    print("Sample intents from filtered_test_gitlab_v2.json:")
    for i, task_id in enumerate(list(task_data.keys())[:3]):
        print(f"  Task {task_id}: {task_data[task_id]['intent'][:80]}{'...' if len(task_data[task_id]['intent']) > 80 else ''}")
    
    return filtered_task_ids, task_data

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
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading performance file for task {task_id}: {e}")
    
    print(f"Found {len(successful_tasks)} successful tasks")
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

def main():
    """Filter successful trajectories based on filtered_test_gitlab_v2.json"""
    # Get filtered task IDs and their intents
    filtered_task_ids, filtered_task_data = get_filtered_task_ids_and_intents()
    
    # Get successful tasks
    successful_tasks = get_successful_tasks()
    
    # Filter successful tasks to only include those in filtered_test_gitlab_v2.json
    filtered_successful_tasks = list(set(successful_tasks).intersection(set(filtered_task_ids)))
    
    print(f"Found {len(filtered_successful_tasks)} successful tasks that match the filtered task IDs")
    
    # Save list of filtered successful tasks
    with open(os.path.join(output_dir, "filtered_successful_tasks.json"), 'w') as f:
        json.dump(filtered_successful_tasks, f, indent=2)
    
    # Extract data for each filtered successful task
    task_summaries = []
    for task_id in tqdm(filtered_successful_tasks, desc="Extracting filtered trajectories"):
        if extract_task_data(task_id):
            # Get intent from filtered_task_data
            intent = filtered_task_data.get(task_id, {}).get("intent", "No intent available")
            print(f"Task {task_id} intent: {intent[:80]}{'...' if len(intent) > 80 else ''}")
            
            # Save task info summary
            with open(os.path.join(output_dir, f"task_info_{task_id}.json"), 'w') as f:
                json.dump({
                    "task_id": task_id,
                    "intent": intent,
                    "score": 1.0
                }, f, indent=2)
            
            task_summaries.append({
                "task_id": task_id,
                "intent": intent
            })
    
    # Save task summaries
    with open(os.path.join(output_dir, "task_summaries.json"), 'w') as f:
        json.dump(task_summaries, f, indent=2)
    
    print(f"Successfully extracted {len(task_summaries)} filtered trajectories to {output_dir}")
    
    # Print the list of task IDs for reference
    print("\nFiltered successful task IDs:")
    for task_id in sorted(filtered_successful_tasks, key=int):
        print(f"Task {task_id}")
    
    # Create an export file for process_react_trajectories.py
    export_data = {
        "trajectories": {},
        "task_info": {}
    }
    
    for task_id in task_summaries:
        task_id_str = task_id["task_id"]
        traj_path = os.path.join(output_dir, "trajectories", f"task_{task_id_str}.pkl.xz")
        if os.path.exists(traj_path):
            try:
                with lzma.open(traj_path, "rb") as f:
                    traj_data = pickle.load(f)
                    
                # Store trajectory data properly based on its type
                if isinstance(traj_data, dict) and "trajectory" in traj_data:
                    # If it's a dictionary with a trajectory key
                    export_data["trajectories"][task_id_str] = traj_data["trajectory"]
                elif isinstance(traj_data, list):
                    # If it's directly a list of trajectory steps
                    export_data["trajectories"][task_id_str] = traj_data
                else:
                    print(f"Warning: Unexpected trajectory data format for task {task_id_str}")
                    continue
                
                # Use intents from filtered_task_data
                export_data["task_info"][task_id_str] = {
                    "intent": filtered_task_data.get(task_id_str, {}).get("intent", "No intent available")
                }
                    
                print(f"Added trajectory for task {task_id_str} with {len(export_data['trajectories'][task_id_str])} steps")
            except Exception as e:
                print(f"Error processing trajectory for task {task_id_str}: {e}")
    
    # Save export data
    export_path = os.path.join(output_dir, "exported_data.pkl")
    with open(export_path, "wb") as f:
        pickle.dump(export_data, f)
    
    print(f"\nCreated export file at {export_path} with {len(export_data['trajectories'])} trajectories")

if __name__ == "__main__":
    main() 