import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add necessary environment variables
os.environ["VALUE_FUNC_PROVIDER"] = "openai"  # Replace with your actual provider
os.environ["VALUE_FUNC_API_BASE"] = "https://api.openai.com/v1"  # Replace with your actual API base

def process_gitlab_directories(base_result_dir, output_dir, env_name="gitlab", use_simplified=False, lenient_mode=False):
    """Process GitLab directories with either tree_to_data.py or simplified_tree_to_data.py"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all subdirectories that contain "gitlab" in their name
    gitlab_dirs = []
    for item in Path(base_result_dir).iterdir():
        if item.is_dir() and "gitlab" in item.name.lower():
            gitlab_dirs.append(item)
    
    if not gitlab_dirs:
        print(f"No gitlab directories found in {base_result_dir}")
        return False
    
    print(f"Found {len(gitlab_dirs)} gitlab directories to process:")
    for i, directory in enumerate(gitlab_dirs):
        print(f"{i+1}. {directory}")
    
    # Ask which directory to process
    print("\nOptions:")
    print("1. Process a specific directory")
    print("2. Process all directories")
    choice = input("Enter your choice (1/2): ")
    
    dirs_to_process = []
    if choice == "1":
        idx = int(input(f"Enter the number of the directory to process (1-{len(gitlab_dirs)}): ")) - 1
        if 0 <= idx < len(gitlab_dirs):
            dirs_to_process = [gitlab_dirs[idx]]
        else:
            print("Invalid selection.")
            return False
    elif choice == "2":
        dirs_to_process = gitlab_dirs
    else:
        print("Invalid choice.")
        return False
    
    # Choose processor script
    script_name = "simplified_tree_to_data.py" if use_simplified else "runners/train/tree_to_data.py"
    print(f"Using {'simplified (no reflection)' if use_simplified else 'regular R-MCTS'} processor: {script_name}")
    
    # Process each selected directory
    success_count = 0
    for result_dir in dirs_to_process:
        print(f"\nProcessing: {result_dir}")
        
        # Build command
        cmd = ["python", script_name, "--env_name", env_name, "--result_dir", str(result_dir), "--output_dir", output_dir]
        
        # Add additional flags based on options
        if use_simplified:
            cmd.append("--create_flattened")
            if lenient_mode:
                cmd.append("--lenient_mode")
        
        print(f"Running command: {' '.join(cmd)}")
        confirm = input("Proceed with processing? (y/n): ")
        if confirm.lower() == 'y':
            try:
                subprocess.run(cmd, check=True)
                print(f"✅ Successfully processed {result_dir}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"❌ Error processing {result_dir}: {e}")
        else:
            print("Skipped.")
    
    return success_count > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process trajectories for fine-tuning")
    parser.add_argument("--base_dir", type=str, default="/Users/nikhilkhandekar/Documents/my-exact/data/webarena/eval_results/react_text",
                       help="Base directory containing result subdirectories")
    parser.add_argument("--output_dir", type=str, default="./training_data_output",
                       help="Directory to save processed training data")
    parser.add_argument("--env_name", type=str, default="gitlab", 
                       help="Environment name (default: gitlab)")
    parser.add_argument("--simplified", action="store_true",
                       help="Use simplified_tree_to_data.py (no reflection) instead of tree_to_data.py")
    parser.add_argument("--lenient", action="store_true",
                       help="Use lenient mode to preserve more trajectories (only with --simplified)")
    
    args = parser.parse_args()
    
    if process_gitlab_directories(args.base_dir, args.output_dir, args.env_name, args.simplified, args.lenient):
        print("\nAll processing complete. Check the output directory for generated training data.")
    else:
        print("\nProcessing failed or was canceled.") 