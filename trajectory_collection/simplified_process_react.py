import os
import json
import pickle
import lzma
import argparse
import re
import random
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# Try to import tiktoken for token counting (optional but recommended)
try:
    import tiktoken
    HAVE_TIKTOKEN = True
except ImportError:
    HAVE_TIKTOKEN = False
    print("Warning: tiktoken not found. Token counting will be approximated.")
    
# Set the tokenizer based on model
def get_tokenizer(model_name="gpt-4o"):
    """Get tokenizer for a specific model, similar to tree_to_data.py"""
    if HAVE_TIKTOKEN:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return encoding
        except:
            # Fallback to cl100k_base for newer models
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return encoding
            except:
                pass
    return None

def count_tokens(text, tokenizer=None):
    """Count tokens in text, similar to how tree_to_data.py estimates token counts"""
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    else:
        # Rough approximation when tiktoken is not available
        return len(text) // 4

def _find_all_links(text: str) -> list:
    """Find all links in the text, similar to tree_to_data.py"""
    pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(pattern, text)

def _replace_links(text: str) -> str:
    """Replace links with placeholders, similar to tree_to_data.py"""
    pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.sub(pattern, '[URL]', text)

def _filter_train_data(content: str) -> str:
    """
    Filter training data similar to _filter_train_data in tree_to_data.py
    - Truncate extremely long contents
    - Clean up formatting issues
    """
    # Truncate extremely long content
    if len(content) > 12000:
        content = content[:12000] + "... [truncated]"
    
    # Remove excessive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Replace HTML characters
    content = content.replace('&lt;', '<').replace('&gt;', '>')
    content = content.replace('&amp;', '&').replace('&quot;', '"')
    
    # Clean up whitespace
    content = re.sub(r' +', ' ', content)
    content = re.sub(r'\t+', ' ', content)
    
    # Replace links with placeholders for better generalization
    content = _replace_links(content)
    
    return content

def _clean_observation(obs_text):
    """Clean observation text similar to tree_to_data.py"""
    # Remove extremely long numbers/IDs
    obs_text = re.sub(r'\b\d{20,}\b', '[LONG_ID]', obs_text)
    
    # Remove email addresses
    obs_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', obs_text)
    
    # Shorten very long text sections by removing middle portions
    lines = obs_text.split('\n')
    if len(lines) > 500:
        # Keep first and last parts
        lines = lines[:250] + ['...'] + lines[-250:]
        obs_text = '\n'.join(lines)
    
    return obs_text

def _check_error_messages(content):
    """Check for common error messages that should be filtered, similar to tree_to_data.py"""
    error_keywords = [
        "no matching element found",
        "element not found",
        "could not find element",
        "element not visible",
        "no such element"
    ]
    content_lower = content.lower()
    for keyword in error_keywords:
        if keyword in content_lower:
            return True
    return False

def _trainable_chat_postprocessing(messages, tokenizer=None, max_tokens=16000):
    """Post-process the training samples similar to tree_to_data.py"""
    processed_messages = []
    
    # Calculate total tokens before processing
    total_tokens = 0
    for message in messages:
        if isinstance(message["content"], str):
            total_tokens += count_tokens(message["content"], tokenizer)
    
    # If total tokens exceeds max_tokens, apply more aggressive truncation
    aggressive_truncation = total_tokens > max_tokens
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        # Apply filtering to content
        if isinstance(content, str):
            content = _filter_train_data(content)
            
            # For user messages with observations, apply special cleaning
            if role == "user" and "Current webpage:" in content:
                parts = content.split("Current webpage:", 1)
                intro = parts[0] + "Current webpage:"
                obs_part = parts[1]
                
                # Clean the observation part
                obs_part = _clean_observation(obs_part)
                
                # Aggressively truncate if needed
                if aggressive_truncation and len(obs_part) > 2000:
                    obs_part = obs_part[:1000] + "...[truncated]..." + obs_part[-1000:]
                    
                content = intro + obs_part
            
            # For assistant messages (actions), ensure they're not too long
            # and check for error messages that should cause filtering
            if role == "assistant":
                if _check_error_messages(content):
                    # If we find error messages, return None to indicate this sample should be filtered
                    return None
                
                if len(content) > 2000:
                    content = content[:2000] + "...[truncated]"
        
        processed_messages.append({
            "role": role,
            "content": content
        })
    
    return processed_messages

def check_context_size(messages, tokenizer=None, max_tokens=16000):
    """Check if the dialogue fits within context limit and trim if needed"""
    if not messages:
        return messages, 0
    
    # Count tokens
    total_tokens = 0
    message_tokens = []
    
    for message in messages:
        if isinstance(message["content"], str):
            tokens = count_tokens(message["content"], tokenizer)
            message_tokens.append(tokens)
            total_tokens += tokens
        else:
            message_tokens.append(0)
    
    # If within limit, return as is
    if total_tokens <= max_tokens:
        return messages, total_tokens
    
    # Otherwise, we need to trim
    print(f"Warning: Context size ({total_tokens} tokens) exceeds limit ({max_tokens}), trimming...")
    
    # Keep system message and at least the last 6 messages (3 turns)
    if len(messages) <= 7:  # system + 6 messages
        return messages, total_tokens
    
    # Always keep system message and last 6 messages
    kept_messages = [messages[0]] + messages[-6:]
    kept_tokens = message_tokens[0] + sum(message_tokens[-6:])
    
    # Try to add as many middle messages as possible
    middle_messages = messages[1:-6]
    middle_tokens = message_tokens[1:-6]
    
    remaining_budget = max_tokens - kept_tokens
    
    for msg, tokens in zip(middle_messages, middle_tokens):
        if tokens <= remaining_budget:
            kept_messages.insert(1, msg)  # Insert after system message
            remaining_budget -= tokens
        else:
            break
    
    return kept_messages, max_tokens - remaining_budget

def _filter_chat_format(messages):
    """Check for format problems similar to tree_to_data.py's _filter_train_data"""
    # Skip if empty or too short
    if not messages or len(messages) < 3:  # Need at least system + user + assistant
        return True
    
    # Check the last assistant message for content
    last_message = messages[-1]
    if last_message["role"] != "assistant":
        return True
    
    if not last_message["content"] or last_message["content"].strip() == "":
        return True
    
    # Check for alternating user/assistant messages
    for i in range(1, len(messages) - 1):
        if i > 0 and messages[i]["role"] == messages[i+1]["role"]:
            return True
    
    return False

def _create_partial_trajectories(trajectory, action_history, observations, actions, max_traj_length=16, strict_mode=True):
    """
    Create partial trajectories for imitation learning similar to tree_to_data.py
    This creates (s,a), (s,a,s,a), (s,a,s,a,s,a) etc. patterns
    
    Args:
        trajectory: Full trajectory information
        action_history: List of action descriptions 
        observations: List of observation texts
        actions: List of action texts
        max_traj_length: Maximum trajectory length to consider
        strict_mode: When True, use stricter filtering to ensure high quality trajectories (always True)
    """
    partial_trajectories = []
    
    # Limit to avoid extremely long trajectories
    num_actions = min(len(actions), max_traj_length // 2)
    
    for end_idx in range(1, num_actions + 1):
        # Create truncated trajectory
        partial_actions = actions[:end_idx]
        partial_observations = observations[:end_idx]
        partial_action_history = action_history[:end_idx+1]  # +1 because action_history starts with "None"
        
        # Create messages for this partial trajectory
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": (
                "You are a helpful web assistant that completes tasks in a web browser. "
                "You will be given a task to complete and observations of the current webpage. "
                "Your goal is to complete the task by taking the most appropriate action."
            )
        })
        
        # Get intent
        intent = "No intent available"
        if isinstance(trajectory, dict) and "task_info" in trajectory:
            task_info = trajectory["task_info"]
            if isinstance(task_info, dict) and "intent" in task_info:
                intent = task_info["intent"]
        
        # Initial user message
        initial_prompt = f"Task: {intent}\n\n"
        if partial_observations and partial_observations[0]:
            initial_prompt += f"Current webpage:\n{partial_observations[0]}\n\n"
        initial_prompt += f"Action history: {', '.join(partial_action_history[:1])}"
        
        messages.append({
            "role": "user",
            "content": initial_prompt
        })
        
        # Add first assistant response
        if partial_actions:
            messages.append({
                "role": "assistant",
                "content": partial_actions[0]
            })
        
        # Add remaining conversation turns
        for i in range(1, end_idx):
            # Update action history for this turn
            current_history = partial_action_history[:i+1]
            
            # User message with updated observation
            user_prompt = f"Current webpage:\n{partial_observations[i]}\n\n"
            user_prompt += f"Action history: {', '.join(current_history)}"
            
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Assistant response
            messages.append({
                "role": "assistant",
                "content": partial_actions[i]
            })
        
        # Apply filtering based on strict_mode
        if not _filter_messages(messages, strict_mode):
            partial_trajectories.append(messages)
    
    return partial_trajectories

def _filter_messages(messages, strict_mode=True):
    """
    Filter message sequences based on quality criteria.
    Returns True if the messages should be filtered out.
    
    Always applies strict filtering to ensure high quality trajectories.
    """
    # Check if we have enough messages
    if not messages or len(messages) < 3:  # Need at least system + user + assistant
        return True
    
    # Check for alternating user/assistant roles
    for i in range(1, len(messages) - 1):
        if messages[i]["role"] == messages[i+1]["role"]:
            return True
    
    # Check the last message (should be from assistant)
    last_message = messages[-1]
    if last_message["role"] != "assistant":
        return True
    
    # Check for empty content
    content = last_message.get("content", "")
    if not content or content.strip() == "":
        return True
    
    # Check for error messages
    if _check_error_messages(content):
        return True
    
    # Always use strict mode checks
    if "go_back" in content.lower() or "take a step back" in content.lower():
        return True
    
    return False

def process_trajectories(input_dir, output_dir, apply_filters=True, max_samples=None, 
                        model="gpt-4o", max_tokens=16000, create_partials=True, strict_mode=True,
                        filter_file=None):
    """
    Process successful trajectories for fine-tuning without WebArena dependencies
    
    Args:
        input_dir: Directory containing filtered trajectories
        output_dir: Directory to save the processed training data
        apply_filters: Whether to apply filters and post-processing
        max_samples: Maximum number of samples to process (for testing)
        model: Target model for tokenization (default: gpt-4o)
        max_tokens: Maximum tokens allowed in a sample (default: 16000)
        create_partials: Whether to create partial trajectories (s,a), (s,a,s,a), etc.
        strict_mode: Always True to use strict filtering to ensure high quality trajectories
        filter_file: Path to JSON file containing task IDs to include
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load filter file if provided to get allowed task IDs
    allowed_task_ids = None
    if filter_file and os.path.exists(filter_file):
        try:
            with open(filter_file, 'r') as f:
                filter_data = json.load(f)
                
            # Extract task IDs from the filter file
            allowed_task_ids = set()
            for item in filter_data:
                if isinstance(item, dict) and "task_id" in item:
                    allowed_task_ids.add(item["task_id"])
            
            print(f"Loaded {len(allowed_task_ids)} allowed task IDs from filter file")
        except Exception as e:
            print(f"Error loading filter file: {e}")
            print("Proceeding without task ID filtering")
    
    # Setup tokenizer for context length management
    tokenizer = get_tokenizer(model) if HAVE_TIKTOKEN else None
    if tokenizer:
        print(f"Using {model} tokenizer for context length management")
    else:
        print(f"No tokenizer available, using approximate token counting")
    
    # Check if input is a directory or a specific trajectory file
    if os.path.isdir(input_dir):
        # Dictionary to store trajectories
        trajectories = {}
        task_info = {}
        
        # Find all task_X directories
        task_dirs = []
        for item in os.listdir(input_dir):
            if item.startswith("task_") and os.path.isdir(os.path.join(input_dir, item)):
                task_dirs.append(item)
        
        print(f"Found {len(task_dirs)} task directories")
        
        # Process each task directory
        loaded_count = 0
        for task_dir in tqdm(task_dirs, desc="Processing trajectories"):
            try:
                # Extract task ID from directory name
                task_id = int(task_dir.split("_")[1])
                
                # Skip if we have a filter and this task ID is not in the allowed list
                if allowed_task_ids is not None and task_id not in allowed_task_ids:
                    continue
                
                # Look for trajectory file
                trajectory_path = os.path.join(input_dir, task_dir, "trajectories", f"{task_dir}.pkl.xz")
                if not os.path.exists(trajectory_path):
                    # Try without compression
                    trajectory_path = os.path.join(input_dir, task_dir, "trajectories", f"{task_dir}.pkl")
                    if not os.path.exists(trajectory_path):
                        print(f"No trajectory file found for {task_dir}")
                        continue
                
                print(f"Loading trajectory from {trajectory_path}")
                
                # Load trajectory file
                try:
                    if trajectory_path.endswith(".xz"):
                        with lzma.open(trajectory_path, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(trajectory_path, "rb") as f:
                            data = pickle.load(f)
                    
                    # Check structure of loaded data and extract trajectory
                    if isinstance(data, dict):
                        # Extract trajectory based on structure
                        if "trajectory" in data:
                            trajectories[task_id] = data["trajectory"]
                            if "task_info" in data:
                                task_info[task_id] = data["task_info"]
                            loaded_count += 1
                            print(f"Loaded trajectory for task {task_id} (format 1)")
                        elif "task_data" in data and "trajectory" in data["task_data"]:
                            # Alternative structure
                            trajectories[task_id] = data["task_data"]["trajectory"]
                            task_info[task_id] = {"intent": data.get("intent", "No intent available")}
                            loaded_count += 1
                            print(f"Loaded trajectory for task {task_id} (format 2)")
                        # Additional format check - direct trajectory list
                        elif any(isinstance(item, list) for item in data.values()):
                            # Look for a list value that might be the trajectory
                            for key, value in data.items():
                                if isinstance(value, list) and len(value) > 0:
                                    trajectories[task_id] = value
                                    task_info[task_id] = {"intent": data.get("intent", "No intent available")}
                                    loaded_count += 1
                                    print(f"Loaded trajectory for task {task_id} (format 3)")
                                    break
                        else:
                            # Last resort - check if the data itself is the trajectory
                            trajectories[task_id] = data
                            task_info[task_id] = {"intent": "Task intent not available"}
                            loaded_count += 1
                            print(f"Loaded trajectory for task {task_id} (direct format)")
                    elif isinstance(data, list) and len(data) > 0:
                        # Direct trajectory list
                        trajectories[task_id] = data
                        task_info[task_id] = {"intent": "Task intent not available"}
                        loaded_count += 1
                        print(f"Loaded trajectory for task {task_id} (list format)")
                    else:
                        print(f"Unable to extract trajectory from data structure for task {task_id}")
                        print(f"Data type: {type(data)}")
                        if isinstance(data, dict):
                            print(f"Keys: {list(data.keys())}")
                except Exception as e:
                    print(f"Error loading trajectory file for task {task_id}: {e}")
            except Exception as e:
                print(f"Error processing {task_dir}: {e}")
        
        print(f"Successfully loaded {loaded_count} trajectories out of {len(task_dirs)} directories")
    
    else:
        # Assume input_dir is a direct path to an exported data file
        if not os.path.exists(input_dir):
            print(f"Error: Input path {input_dir} does not exist")
            return
        
        # Load the exported data
        if input_dir.endswith(".xz"):
            with lzma.open(input_dir, "rb") as f:
                data = pickle.load(f)
        else:
            with open(input_dir, "rb") as f:
                data = pickle.load(f)
        
        # If the data is already in the expected format
        if "trajectories" in data and "task_info" in data:
            trajectories = data["trajectories"]
            task_info = data["task_info"]
            
            # Filter by allowed task IDs if applicable
            if allowed_task_ids is not None:
                filtered_trajectories = {tid: traj for tid, traj in trajectories.items() if tid in allowed_task_ids}
                filtered_task_info = {tid: info for tid, info in task_info.items() if tid in allowed_task_ids}
                
                print(f"Filtered from {len(trajectories)} to {len(filtered_trajectories)} trajectories based on allowed task IDs")
                
                trajectories = filtered_trajectories
                task_info = filtered_task_info
        else:
            print("Error: Input file does not contain expected data format")
            return
    
    print(f"Processing {len(trajectories)} trajectories")
    print("Using strict filtering to ensure high quality trajectories")
    
    # Limit samples if specified
    if max_samples is not None and max_samples > 0:
        task_ids = list(trajectories.keys())[:max_samples]
        trajectories = {tid: trajectories[tid] for tid in task_ids}
        print(f"Limited to {len(trajectories)} samples for testing")
    
    # Print some sample information
    if trajectories:
        sample_task_id = list(trajectories.keys())[0]
        sample_traj = trajectories[sample_task_id]
        print(f"\nSample trajectory (Task {sample_task_id}):")
        print(f"- Number of steps: {len(sample_traj)}")
        
        # Print types of items in the trajectory
        item_types = set(type(item).__name__ for item in sample_traj)
        print(f"- Item types in trajectory: {', '.join(item_types)}")
        
        # Examine a few items
        for i, item in enumerate(sample_traj[:3]):
            if hasattr(item, 'action_type'):
                print(f"- Step {i}: Action with type '{item.action_type}'")
            elif isinstance(item, dict) and "observation" in item:
                obs_keys = item["observation"].keys() if isinstance(item["observation"], dict) else []
                print(f"- Step {i}: Observation with keys {list(obs_keys)}")
            else:
                print(f"- Step {i}: {type(item).__name__}")
    
    # Process each trajectory
    training_samples = []
    skipped_tasks = []
    token_stats = {"min": float('inf'), "max": 0, "avg": 0, "total": 0}
    
    for task_id, trajectory in tqdm(trajectories.items(), desc="Processing trajectories"):
        try:
            # Extract actions and observations
            actions = []
            observations = []
            action_history = ["None"]  # Start with None action as per tree_to_data.py
            
            # Get intent information
            intent = "No intent available"
            if task_id in task_info and isinstance(task_info[task_id], dict):
                if "intent" in task_info[task_id]:
                    intent = task_info[task_id]["intent"]
            
            # Find intent in config.json if not available in trajectory
            if intent == "No intent available":
                config_path = os.path.join(input_dir, f"task_{task_id}", "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                            if "intent" in config_data:
                                intent = config_data["intent"]
                                print(f"Found intent in config.json for task {task_id}")
                    except Exception as e:
                        print(f"Error reading config.json for task {task_id}: {e}")
            
            # Process trajectory
            print(f"Processing trajectory for task {task_id} with {len(trajectory)} steps")
            
            # Debug: print types of first few items
            for i, item in enumerate(trajectory[:3]):
                print(f"Item {i} type: {type(item).__name__}")
                if hasattr(item, "__dict__"):
                    print(f"  Attributes: {list(item.__dict__.keys())}")
                elif isinstance(item, dict):
                    print(f"  Keys: {list(item.keys())}")
            
            for item in trajectory:
                # Check if this is an observation
                if isinstance(item, dict) and "observation" in item:
                    # This is an observation
                    obs_text = None
                    if isinstance(item["observation"], dict) and "text" in item["observation"]:
                        obs_text = item["observation"]["text"]
                    elif isinstance(item["observation"], str):
                        obs_text = item["observation"]
                    
                    if obs_text:
                        observations.append(obs_text)
                    else:
                        print(f"Warning: Could not extract text from observation: {type(item['observation'])}")
                # Check if this is an action
                elif hasattr(item, "action_type") or hasattr(item, "raw_prediction") or hasattr(item, "answer"):
                    # This is likely an action
                    action_text = None
                    # Try different ways to get the action text
                    if hasattr(item, "raw_prediction") and item.raw_prediction:
                        action_text = item.raw_prediction
                    elif hasattr(item, "answer") and item.answer:
                        action_text = item.answer
                    elif hasattr(item, "__dict__"):
                        # Try to convert the entire object to a string representation
                        try:
                            action_text = str(item)
                        except:
                            pass
                    
                    if action_text:
                        actions.append(action_text)
                        # Add to action history
                        action_str = "Action"
                        if hasattr(item, "action_type"):
                            action_str = f"{item.action_type}"
                            if hasattr(item, "element_id") and item.element_id:
                                action_str += f" on element [{item.element_id}]"
                        action_history.append(action_str)
                    else:
                        print(f"Warning: Could not extract text for action: {type(item)}")
                else:
                    # This is an unknown item type
                    print(f"Warning: Unknown item type: {type(item).__name__}")
            
            # Balance actions and observations if needed
            if len(actions) > len(observations):
                print(f"Warning: More actions ({len(actions)}) than observations ({len(observations)})")
                # Truncate actions to match observations
                actions = actions[:len(observations)]
                action_history = action_history[:len(observations)]
            elif len(observations) > len(actions) + 1:  # +1 because we start with an observation
                print(f"Warning: More observations ({len(observations)}) than actions ({len(actions)})")
                # Truncate observations to match actions + 1
                observations = observations[:len(actions) + 1]
            
            # Skip if no actions found
            if not actions:
                print(f"Warning: No actions found for task {task_id}, skipping")
                skipped_tasks.append({"task_id": task_id, "reason": "no_actions"})
                continue
            
            # Skip if no observations found
            if not observations:
                print(f"Warning: No observations found for task {task_id}, skipping")
                skipped_tasks.append({"task_id": task_id, "reason": "no_observations"})
                continue
            
            print(f"Found {len(actions)} actions and {len(observations)} observations for task {task_id}")

            # Create either full trajectory or partial trajectories based on settings
            all_message_sets = []
            
            if create_partials:
                # Create multiple partial trajectories (s,a), (s,a,s,a), etc.
                trajectory_with_info = {"task_info": {"intent": intent}}
                all_message_sets = _create_partial_trajectories(
                    trajectory_with_info, action_history, observations, actions, 
                    strict_mode=strict_mode
                )
            else:
                # Create only the full trajectory - original behavior
                messages = []
                
                # System message
                messages.append({
                    "role": "system",
                    "content": (
                        "You are a helpful web assistant that completes tasks in a web browser. "
                        "You will be given a task to complete and observations of the current webpage. "
                        "Your goal is to complete the task by taking the most appropriate action."
                    )
                })
                
                # Initial user message
                initial_prompt = f"Task: {intent}\n\n"
                if observations and observations[0]:
                    initial_prompt += f"Current webpage:\n{observations[0]}\n\n"
                initial_prompt += f"Action history: {', '.join(action_history[:1])}"
                
                messages.append({
                    "role": "user",
                    "content": initial_prompt
                })
                
                # Add first assistant response
                if actions:
                    messages.append({
                        "role": "assistant",
                        "content": actions[0]
                    })
                
                # Add remaining conversation turns
                for i in range(1, min(len(actions), len(observations))):
                    # Update action history for this turn
                    current_history = action_history[:i+1]
                    
                    # User message with updated observation
                    user_prompt = f"Current webpage:\n{observations[i]}\n\n"
                    user_prompt += f"Action history: {', '.join(current_history)}"
                    
                    messages.append({
                        "role": "user",
                        "content": user_prompt
                    })
                    
                    # Assistant response
                    messages.append({
                        "role": "assistant",
                        "content": actions[i]
                    })
                
                # Apply filtering based on strict_mode
                if not _filter_messages(messages, strict_mode):
                    all_message_sets = [messages]
            
            # Process each message set (either one full or multiple partials)
            for messages in all_message_sets:
                # Apply post-processing to the messages if requested
                if apply_filters:
                    processed_messages = _trainable_chat_postprocessing(messages, tokenizer, max_tokens)
                    if processed_messages is None:
                        # This indicates the messages contained errors and should be skipped
                        continue
                    messages = processed_messages
                
                # Make sure context fits within token limit
                messages, total_tokens = check_context_size(messages, tokenizer, max_tokens)
                
                # Skip if no valid messages after processing
                if not messages or len(messages) < 3:
                    continue
                
                # Update token statistics
                token_stats["min"] = min(token_stats["min"], total_tokens)
                token_stats["max"] = max(token_stats["max"], total_tokens)
                token_stats["total"] += total_tokens
                
                # Add the training sample
                training_samples.append({
                    "messages": messages,
                    "metadata": {
                        "task_id": task_id,
                        "intent": intent,
                        "num_actions": len(messages) // 2,  # Approximation based on message count
                        "num_observations": len(messages) // 2,
                        "token_count": total_tokens
                    }
                })
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            skipped_tasks.append({"task_id": task_id, "reason": "error", "error": str(e)})
    
    if training_samples:
        token_stats["avg"] = token_stats["total"] / len(training_samples)
        if token_stats["min"] == float('inf'):
            token_stats["min"] = 0
    
    # Similar to tree_to_data.py, let's balance samples if we have too many
    MAX_SAMPLES = 500
    if len(training_samples) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES} samples for more balanced dataset")
        rng = random.Random(42)
        rng.shuffle(training_samples)
        training_samples = training_samples[:MAX_SAMPLES]
    
    # Save the training data
    str_date = datetime.now().strftime("%m%d")
    output_file = os.path.join(output_dir, f"react_gitlab_training_data_{str_date}.json")
    with open(output_file, "w") as f:
        json.dump(training_samples, f, indent=2)
    
    # Also save in JSONL format for OpenAI fine-tuning
    jsonl_output_file = os.path.join(output_dir, f"react_gitlab_training_data_{str_date}.jsonl")
    with open(jsonl_output_file, "w") as f:
        for sample in training_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Save OpenAI fine-tuning format (without metadata)
    openai_jsonl_file = os.path.join(output_dir, f"openai_fine_tuning_{str_date}.jsonl")
    with open(openai_jsonl_file, "w") as f:
        for sample in training_samples:
            f.write(json.dumps({"messages": sample["messages"]}) + "\n")
    
    # Save a summary of the data
    summary = {
        "total_trajectories": len(trajectories),
        "successful_conversions": len(training_samples),
        "skipped_tasks": skipped_tasks,
        "average_dialogue_turns": sum(len(sample["messages"]) // 2 for sample in training_samples) / max(1, len(training_samples)),
        "average_actions": sum(sample["metadata"]["num_actions"] for sample in training_samples) / max(1, len(training_samples)),
        "token_stats": token_stats
    }
    
    summary_file = os.path.join(output_dir, f"training_data_summary_{str_date}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSuccessfully created {len(training_samples)} training samples at {output_file}")
    print(f"Also saved in JSONL format at {jsonl_output_file}")
    print(f"OpenAI fine-tuning format saved at {openai_jsonl_file}")
    print(f"Skipped {len(skipped_tasks)} tasks")
    print(f"Average dialogue turns: {summary['average_dialogue_turns']:.1f}")
    print(f"Average actions per trajectory: {summary['average_actions']:.1f}")
    print(f"Token statistics: min={token_stats['min']}, max={token_stats['max']}, avg={token_stats['avg']:.1f}")
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process filtered trajectories for fine-tuning")
    parser.add_argument("--input_dir", type=str, default="/Users/nikhilkhandekar/Documents/my-exact/data/training/react_filtered",
                      help="Directory containing filtered trajectories")
    parser.add_argument("--output_dir", type=str, default="/Users/nikhilkhandekar/Documents/my-exact/data/training/react_fine_tuning",
                      help="Directory to save processed trajectories")
    parser.add_argument("--no_filters", action="store_true", 
                      help="Disable content filtering and post-processing")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum number of samples to process (for testing)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                      help="Target model for tokenization and context length (default: gpt-4o)")
    parser.add_argument("--max_tokens", type=int, default=16000,
                      help="Maximum tokens allowed in a sample (default: 16000)")
    parser.add_argument("--no_partials", action="store_true",
                      help="Disable creation of partial trajectories")
    parser.add_argument("--filter_file", type=str, default=None,
                      help="Path to JSON file containing task IDs to include")
    
    args = parser.parse_args()
    
    process_trajectories(
        args.input_dir, 
        args.output_dir, 
        apply_filters=not args.no_filters, 
        max_samples=args.max_samples,
        model=args.model,
        max_tokens=args.max_tokens,
        create_partials=not args.no_partials,
        strict_mode=True,
        filter_file=args.filter_file
    ) 