import json
import os
import argparse
import pickle
import lzma

def convert_trajectories(input_dir, output_dir, format_type='sft'):
    """
    Converts trajectories to a format suitable for model training.
    
    Args:
        input_dir: Directory containing trajectory files
        output_dir: Directory to write converted data
        format_type: Type of conversion ('sft' or 'dpo')
    
    Returns:
        Number of trajectories successfully converted
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get trajectory files
    trajectory_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl.xz') or f.endswith('.json')]
    
    if not trajectory_files:
        print(f"No trajectory files found in {input_dir}")
        return 0
    
    converted_count = 0
    train_data = []
    
    for file in trajectory_files:
        filepath = os.path.join(input_dir, file)
        
        try:
            # Load trajectory
            if file.endswith('.json'):
                with open(filepath, 'r') as f:
                    trajectory = json.load(f)
            elif file.endswith('.pkl.xz'):
                with lzma.open(filepath, 'rb') as f:
                    trajectory = pickle.load(f)
            else:
                continue
            
            # Process trajectory based on format type
            if format_type == 'sft':
                # For SFT, we need to convert the trajectory to a conversation format
                conversation = convert_to_sft_format(trajectory)
                if conversation:
                    train_data.append(conversation)
                    converted_count += 1
            
            elif format_type == 'dpo':
                # For DPO, we would need both chosen and rejected examples
                # This is a placeholder - actual implementation would depend on project specifics
                print(f"DPO format conversion not fully implemented for {file}")
                continue
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Write output file
    if train_data:
        output_file = os.path.join(output_dir, f"{format_type}_data.jsonl")
        with open(output_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
    
    return converted_count

def convert_to_sft_format(trajectory):
    """
    Converts a trajectory to SFT training format.
    
    Args:
        trajectory: The trajectory data
        
    Returns:
        Dictionary in SFT training format
    """
    # This implementation will depend on the specific structure of your trajectories
    # and the desired format for your training data
    
    # Example conversion for a conversational format:
    if 'actions' not in trajectory or not trajectory.get('success', False):
        return None
    
    messages = []
    
    # Add system message
    messages.append({
        "role": "system",
        "content": "You are a web navigation assistant that helps users accomplish tasks online."
    })
    
    # Add task description as user message
    if 'task' in trajectory:
        messages.append({
            "role": "user",
            "content": f"Help me with this task: {trajectory['task']}"
        })
    
    # Convert actions to assistant messages
    for action in trajectory['actions']:
        if 'action_type' in action and 'action_input' in action:
            action_text = f"{action['action_type']}: {action['action_input']}"
            messages.append({
                "role": "assistant",
                "content": action_text
            })
            
            # Add observation as system message if available
            if 'observation' in action:
                messages.append({
                    "role": "system",
                    "content": f"Observation: {action['observation']}"
                })
    
    return {"messages": messages}

def main():
    parser = argparse.ArgumentParser(description='Convert trajectories to training format')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing trajectory files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write converted data')
    parser.add_argument('--format', type=str, default='sft', choices=['sft', 'dpo'], help='Conversion format')
    args = parser.parse_args()
    
    count = convert_trajectories(args.input_dir, args.output_dir, args.format)
    print(f"Successfully converted {count} trajectories to {args.format.upper()} format")

if __name__ == "__main__":
    main() 