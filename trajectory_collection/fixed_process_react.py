#!/usr/bin/env python
import os
import json
import pickle
import lzma
import argparse
from tqdm import tqdm
from PIL import Image
import sys
import importlib.util

# Add directory to Python path to find modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import required modules
from src.agentic.policy import CoTPolicyPConstructor
from src.helper_functions import get_action_description

# Global cache for image captioning
image_caption_cache = {}

def configure_captioning_fn():
    """Configure and return image captioning function"""
    try:
        from src.helper_functions import caption_image
        return caption_image
    except ImportError:
        print("Warning: caption_image not found, using placeholder")
        # Return a placeholder function
        return lambda img: "Image caption not available"

# Initialize captioning function
caption_image_fn = configure_captioning_fn()

def cached_caption_image_fn(images: list):
    """Cache image captions to avoid recalculating"""
    captions = []
    for img in images:
        img_hash = hash(img.tobytes())
        if img_hash in image_caption_cache:
            captions.append(image_caption_cache[img_hash])
        else:
            caption = caption_image_fn(img)
            image_caption_cache[img_hash] = caption
            captions.append(caption)
    return captions

def save_image_cache():
    """Save image caption cache to disk"""
    # Don't actually save, just for compatibility
    pass

def get_action_descs(trajectory, action_set_tag="id_accessibility_tree"):
    """Extract action descriptions from trajectory"""
    action_history = []
    for item in trajectory:
        if not isinstance(item, dict):  # This is an action
            action = item
            if hasattr(action, "action_type"):
                action_str = f"{action.action_type}"
                # Add element ID if available
                if hasattr(action, "element_id") and action.element_id:
                    action_str += f" on element [{action.element_id}]"
                action_history.append(action_str)
            else:
                # Fallback to string representation
                action_history.append(str(action))
    
    # Add a "None" action at the beginning if the list is empty
    if not action_history:
        action_history = ["None"]
    
    return action_history

def format_trajectory_to_chat(prompt_constructor, trajectory, task_info):
    """Format trajectory to chat format for training"""
    # Make sure the last element is a state
    assert isinstance(trajectory[-1], dict), "Last element of trajectory should be a state"

    images = task_info.get("images", [])  # intent images
    intent = task_info["intent"]
    meta_data = {}

    action_history_descs = get_action_descs(trajectory)
    meta_data["action_history"] = action_history_descs

    # Caption the input image, if provided
    if images is not None and len(images) > 0:
        image_input_caption = ""
        for image_i, image in enumerate(images):
            if image_i == 0:
                image_input_caption += f'Input image {image_i+1}: "{cached_caption_image_fn([image])[0]}"'
            else:
                image_input_caption += f'input image {image_i+1}: "{cached_caption_image_fn([image])[0]}"'
            if len(images) > 1:
                image_input_caption += ", "
        # Update intent to include captions of input images
        intent = f"{image_input_caption}\nIntent: {intent}"

    # Construct prompt (without llm_config which causes error)
    prompt = prompt_constructor.construct(
        trajectory, intent, meta_data  # empty images since we use caption for training data
    )
    
    # Add actions as assistant responses
    for i, item in enumerate(trajectory):
        if not isinstance(item, dict):  # This is an action
            action = item
            agent_resp = {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action.raw_prediction,
                    }
                ]
            }
            # Insert the response after the corresponding user message
            prompt.append(agent_resp)

    return prompt

def flatten_to_trainable_chat(chat: list, train_last_only=False):
    """Flatten chat data to trainable format"""
    train_sample = []
    for i, message in enumerate(chat):
        role = message["role"]
        content = message["content"]
        if isinstance(content, list):
            str_contents = []
            for c in content:
                if c["type"] == "text":
                    str_contents.append(c["text"])
            str_content = "\n\n".join(str_contents)
        else:
            assert isinstance(content, str)
            str_content = content
        
        if role in ["user", "system", "assistant"]:
            if not train_last_only or i == len(chat) - 1 or role != "assistant":
                train_sample.append({
                    "role": role,
                    "content": str_content
                })

    return train_sample

def process_react_trajectories(input_dir, output_dir, env_name, modality="text", lenient_mode=False):
    """
    Process ReACT agent trajectories for fine-tuning
    
    Args:
        input_dir: Directory containing ReACT agent results
        output_dir: Directory to save processed trajectories
        env_name: Environment name (e.g., "classifieds", "shopping")
        modality: Text or SoM modality
        lenient_mode: When True, use more lenient filtering to preserve more trajectories
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the policy constructor with fixed parameters
    try:
        # FIX: Remove llm_config parameter which was causing the error
        prompt_constructor = CoTPolicyPConstructor(
            instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json",
            action_set_tag="id_accessibility_tree",
            modality=modality
        )
    except TypeError as e:
        print(f"Error initializing prompt constructor: {e}")
        print("Trying alternative constructor initialization...")
        
        # Try another constructor format if the first one fails
        try:
            from src.agentic.policy import ExploratoryCoTPolicyPConstructor
            prompt_constructor = ExploratoryCoTPolicyPConstructor(
                instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
            )
        except Exception as e2:
            print(f"Error with alternative constructor: {e2}")
            raise RuntimeError("Could not initialize prompt constructor with any method")
    
    print(f"Successfully initialized prompt constructor of type: {type(prompt_constructor).__name__}")
    
    # Check if input is a directory with task directories or a directory with trajectory files
    task_dirs = []
    for item in os.listdir(input_dir):
        if item.startswith("task_") and os.path.isdir(os.path.join(input_dir, item)):
            task_dirs.append(item)
    
    # Scan input directory for trajectories
    success_trajs = []
    task_ids = []
    
    if task_dirs:
        # Process directory with task subdirectories
        print(f"Found {len(task_dirs)} task directories")
        
        for task_dir in tqdm(task_dirs, desc="Scanning task directories"):
            try:
                # Extract task ID from directory name
                task_id = int(task_dir.split("_")[1])
                
                # Look for trajectory file
                trajectory_path = os.path.join(input_dir, task_dir, "trajectories", f"{task_dir}.pkl.xz")
                if not os.path.exists(trajectory_path):
                    # Try without compression
                    trajectory_path = os.path.join(input_dir, task_dir, "trajectories", f"{task_dir}.pkl")
                    if not os.path.exists(trajectory_path):
                        continue
                
                # Try to load config for task info
                config_path = os.path.join(input_dir, task_dir, "config.json")
                task_info = None
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            task_info = {
                                "intent": config.get("intent", "No intent available"),
                                "images": []
                            }
                    except:
                        pass
                
                # Load trajectory file
                try:
                    if trajectory_path.endswith(".xz"):
                        with lzma.open(trajectory_path, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(trajectory_path, "rb") as f:
                            data = pickle.load(f)
                    
                    # Extract trajectory and task info
                    trajectory = None
                    if isinstance(data, dict):
                        if "trajectory" in data:
                            trajectory = data["trajectory"]
                            if not task_info and "task_info" in data:
                                task_info = data["task_info"]
                        elif "task_data" in data and "trajectory" in data["task_data"]:
                            trajectory = data["task_data"]["trajectory"]
                            if not task_info:
                                task_info = {"intent": data.get("intent", "No intent available"), "images": []}
                    
                    # Check if we have basic requirements
                    if trajectory and task_info:
                        success_trajs.append((trajectory, task_info))
                        task_ids.append(task_id)
                        print(f"Found trajectory for task {task_id}")
                except Exception as e:
                    print(f"Error loading {trajectory_path}: {e}")
            except Exception as e:
                print(f"Error processing directory {task_dir}: {e}")
    else:
        # Process directory with trajectory files
        print("Looking for individual trajectory files")
        
        for filename in tqdm(os.listdir(input_dir), desc="Scanning files"):
            if not filename.endswith(".pkl") and not filename.endswith(".xz"):
                continue
                
            filepath = os.path.join(input_dir, filename)
            
            # Extract task ID from filename
            if "_task_" in filename:
                try:
                    task_id = int(filename.split("_task_")[1].split("_")[0])
                except:
                    continue
            elif "task_" in filename:
                try: 
                    task_id = int(filename.split("task_")[1].split(".")[0])
                except:
                    continue
            else:
                continue
                
            # Load trajectory data
            try:
                if filename.endswith(".xz"):
                    with lzma.open(filepath, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(filepath, "rb") as f:
                        data = pickle.load(f)
                    
                # Extract trajectory and task info
                trajectory = None
                task_info = None
                
                if isinstance(data, dict):
                    if "trajectory" in data:
                        trajectory = data["trajectory"]
                        if "task_info" in data:
                            task_info = data["task_info"]
                    elif "task_data" in data and "trajectory" in data["task_data"]:
                        trajectory = data["task_data"]["trajectory"]
                        task_info = {"intent": data.get("intent", "No intent available"), "images": []}
                
                # When using lenient_mode, include all trajectories
                # Otherwise check for success
                if trajectory and task_info:
                    include_traj = lenient_mode
                    if not include_traj and "score" in data and data["score"] > 0:
                        include_traj = True
                    
                    if include_traj:
                        success_trajs.append((trajectory, task_info))
                        task_ids.append(task_id)
                        print(f"Found trajectory for task {task_id}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    print(f"Found {len(success_trajs)} trajectories to process")
    
    # Process trajectories
    all_train_samples = []
    
    for (trajectory, task_info), task_id in tqdm(zip(success_trajs, task_ids), desc="Processing trajectories"):
        try:
            # Format trajectory to chat
            chat_data = format_trajectory_to_chat(prompt_constructor, trajectory, task_info)
            
            # Convert to trainable format with filtering
            train_sample = flatten_to_trainable_chat(chat_data)
            
            # Apply additional filtering based on settings
            if train_sample:
                # Apply stricter filtering in non-lenient mode
                if not lenient_mode:
                    # Check if any assistant message contains backtracking
                    contains_backtrack = False
                    for msg in train_sample:
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "").lower()
                            if "go_back" in content or "try a different approach" in content:
                                contains_backtrack = True
                                break
                    
                    # Skip trajectories with backtracking in strict mode
                    if contains_backtrack:
                        print(f"Skipping trajectory {task_id} with backtracking (strict mode)")
                        continue
                
                sample_with_metadata = {
                    "messages": train_sample,
                    "metadata": {
                        "task_id": task_id,
                        "env": env_name
                    }
                }
                all_train_samples.append(sample_with_metadata)
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save processed data
    output_file = os.path.join(output_dir, f"{env_name}_react_training_data.json")
    with open(output_file, "w") as f:
        json.dump(all_train_samples, f, indent=2)
    
    # Also save in JSONL format for OpenAI fine-tuning
    jsonl_output_file = os.path.join(output_dir, f"{env_name}_react_training_data.jsonl")
    with open(jsonl_output_file, "w") as f:
        for sample in all_train_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Save OpenAI fine-tuning format (without metadata)
    openai_jsonl_file = os.path.join(output_dir, f"openai_fine_tuning_{env_name}.jsonl")
    with open(openai_jsonl_file, "w") as f:
        for sample in all_train_samples:
            f.write(json.dumps({"messages": sample["messages"]}) + "\n")
    
    print(f"Saved {len(all_train_samples)} training samples to:")
    print(f"- JSON: {output_file}")
    print(f"- JSONL: {jsonl_output_file}")
    print(f"- OpenAI format: {openai_jsonl_file}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ReACT agent trajectories for fine-tuning")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing ReACT agent results")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed trajectories")
    parser.add_argument("--env_name", type=str, required=True, choices=["classifields", "shopping", "reddit", "gitlab"], help="Environment name")
    parser.add_argument("--modality", type=str, default="text", choices=["text", "som"], help="Text or SoM modality")
    parser.add_argument("--lenient_mode", action="store_true", help="Use more lenient filtering to preserve more trajectories")
    
    args = parser.parse_args()
    
    process_react_trajectories(args.input_dir, args.output_dir, args.env_name, args.modality, args.lenient_mode) 