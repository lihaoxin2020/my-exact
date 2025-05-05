import os
import json
import pickle
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from cachetools import Cache
from browser_env.utils import pil_to_b64
from src.helper_functions import get_action_description
from src.agentic.policy import CoTPolicyPConstructor
from src.llms.lm_config import LMConfig
from src.evaluation import image_utils
import lzma

# Configure environment - replace these with your actual values
os.environ["DATASET"] = "webarena"  # or "webarena"
assert os.environ.get("OPENAI_API_KEY", None) is not None, "OPENAI_API_KEY not set"

# Setup LLM config
llm_config = LMConfig(
    provider="openai",
    model="gpt-4o",
    mode="chat"
)
llm_config.gen_config["temperature"] = 1.0
llm_config.gen_config["top_p"] = 0.95
llm_config.gen_config["max_tokens"] = 384
llm_config.gen_config["max_retry"] = 1

# Image captioning setup
IMAGE_CAPTION_CACHE = Cache(maxsize=1000)
cache_save_path = "react_ft_image_cache.pkl"

# Load cache if exists
if os.path.exists(cache_save_path):
    with open(cache_save_path, "rb") as fread:
        IMAGE_CAPTION_CACHE.update(pickle.load(fread))
    print(f"Loaded {len(IMAGE_CAPTION_CACHE)} cache entries")

def configure_captioning_fn():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    caption_image_fn = image_utils.get_captioning_fn(
        device, dtype, "Salesforce/blip2-flan-t5-xl"
    )
    return caption_image_fn

captioning_fn = configure_captioning_fn()

def cached_caption_image_fn(images: list):
    encoded_images_str = ""
    for image in images:
        encoded_images_str += pil_to_b64(image)
    if encoded_images_str in IMAGE_CAPTION_CACHE:
        return IMAGE_CAPTION_CACHE[encoded_images_str]
    
    captions = captioning_fn(images)
    IMAGE_CAPTION_CACHE[encoded_images_str] = captions
    return captions

def save_image_cache():
    with open(cache_save_path, "wb") as fwrite:
        pickle.dump(IMAGE_CAPTION_CACHE, fwrite)
    print(f"Saved {len(IMAGE_CAPTION_CACHE)} cache entries")
    return

def get_action_descs(trajectory, action_set_tag="id_accessibility_tree"):
    action_strs = ["None"]
    prev_state = None
    for data in trajectory:
        if isinstance(data, dict):
            prev_state = data
        else:
            action = data
            if 'obs_metadata' not in action.metadata:
                observation_metadata = prev_state['info']['observation_metadata']
            else:
                observation_metadata = action.metadata['obs_metadata']
            action_desc = get_action_description(
                action,
                observation_metadata=observation_metadata,
                action_set_tag=action_set_tag,
                prompt_constructor=None
            )
            action_strs.append(action_desc)
    return action_strs

def format_trajectory_to_chat(prompt_constructor, trajectory, task_info):
    # make sure the last one is state
    assert isinstance(trajectory[-1], dict), "Last element of trajectory should be a state"

    images = task_info.get("images", [])  # intent images
    intent = task_info["intent"]
    meta_data = {}

    action_history_descs = get_action_descs(trajectory)
    meta_data["action_history"] = action_history_descs

    # Caption the input image, if provided.
    if images is not None and len(images) > 0:
        image_input_caption = ""
        for image_i, image in enumerate(images):
            if image_i == 0:
                image_input_caption += f'Input image {image_i+1}: "{cached_caption_image_fn([image])[0]}"'
            else:
                image_input_caption += f'input image {image_i+1}: "{cached_caption_image_fn([image])[0]}"'
            if len(images) > 1:
                image_input_caption += ", "
        # Update intent to include captions of input images.
        intent = f"{image_input_caption}\nIntent: {intent}"

    # Construct prompt using the specified constructor
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
            # Since the prompt constructor should have already created user messages
            prompt.append(agent_resp)

    return prompt

def flatten_to_trainable_chat(chat: list, train_last_only=False):
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
    
    # Path to the instruction JSON for the prompt constructor
    instruction_path = "src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"

    # Initialise tokenizer compatible with the chosen LLM
    from src.llms.tokenizer import Tokenizer  # local import to avoid circular deps if any
    tokenizer = Tokenizer(llm_config.provider, llm_config.model)

    prompt_constructor = CoTPolicyPConstructor(
        instruction_path=instruction_path,
        lm_config=llm_config,
        tokenizer=tokenizer,
    )
    
    # Scan input directory for trajectories
    success_trajs = []
    task_ids = []
    
    # -----------------------------------------------------------
    # Case 1: input_dir directly contains trajectory files
    # -----------------------------------------------------------
    for filename in os.listdir(input_dir):
        if not filename.endswith((".pkl", ".xz")):
            continue
            
        has_mid_pattern = "_task_" in filename
        has_prefix_pattern = filename.startswith("task_")
        if not (has_mid_pattern or has_prefix_pattern):
            continue
        
            try:
            if has_mid_pattern:
                task_id = int(filename.split("_task_")[1].split("_")[0])
            else:
                task_id = int(filename.split("task_")[1].split(".")[0])
        except:
            task_id = None

        if task_id is None:
            continue

        def _load(fp):
            if fp.endswith(".xz"):
                with lzma.open(fp, "rb") as f:
                    return pickle.load(f)
            else:
                with open(fp, "rb") as f:
                    return pickle.load(f)

        try:
            data = _load(os.path.join(input_dir, filename))
            if lenient_mode or ("score" in data and data["score"] > 0):
                trajectory = data.get("trajectory", [])
                task_info = data.get("task_info", {})
                if trajectory and task_info:
                    success_trajs.append((trajectory, task_info))
                    task_ids.append(task_id)
        except Exception as e:
            print(f"Error loading {os.path.join(input_dir, filename)}: {e}")

    # -----------------------------------------------------------
    # Case 2: input_dir contains subdirectories task_<id>/trajectories/...
    # -----------------------------------------------------------
    for item in os.listdir(input_dir):
        task_dir = os.path.join(input_dir, item)
        if not os.path.isdir(task_dir) or not item.startswith("task_"):
            continue
        task_id_part = item.split("_")[1]
        try:
            task_id = int(task_id_part)
            except:
                continue

        traj_path = os.path.join(task_dir, "trajectories", f"task_{task_id}.pkl.xz")
        if not os.path.exists(traj_path):
            traj_path = os.path.join(task_dir, "trajectories", f"task_{task_id}.pkl")
        if not os.path.exists(traj_path):
            continue
            
        try:
            if traj_path.endswith(".xz"):
                with lzma.open(traj_path, "rb") as f:
                    data = pickle.load(f)
            else:
                with open(traj_path, "rb") as f:
                    data = pickle.load(f)
                
            if lenient_mode or ("score" in data and data["score"] > 0):
                trajectory = data.get("trajectory", [])
                task_info = data.get("task_info", {})
                if trajectory and task_info:
                    success_trajs.append((trajectory, task_info))
                    task_ids.append(task_id)
        except Exception as e:
            print(f"Error loading {traj_path}: {e}")
    
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
    
    # Save processed data
    output_file = os.path.join(output_dir, f"{env_name}_react_training_data.json")
    with open(output_file, "w") as f:
        json.dump(all_train_samples, f, indent=2)
    
    print(f"Saved {len(all_train_samples)} training samples to {output_file}")
    
    # Save image cache
    save_image_cache()
    
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