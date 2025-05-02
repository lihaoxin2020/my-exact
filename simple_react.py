#!/usr/bin/env python
"""
Simple ReACT agent using GPT-4o directly for GitLab tasks
"""
import os
import json
import time
import requests

# Create necessary directories
os.makedirs("data/training", exist_ok=True)

# Get API key from environment or prompt user
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

# Sample GitLab tasks
tasks = [
    "Create a new issue in the 'ProjectX' repository with the title 'Bug in login feature'",
    "Find the repository called 'Documentation' and star it",
    "Create a new branch called 'feature/login-fix' in the ProjectX repository",
    "Comment on issue #42 with 'This is being worked on'",
    "Merge the pull request titled 'Fix login bug'"
]

def call_openai(messages, model="gpt-4o"):
    """Call the OpenAI API with the given messages"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    return response.json()

# ReACT system prompt
system_prompt = """You are a web navigation agent for GitLab. Your goal is to complete tasks on the GitLab website.

You should respond in a ReACT format: reasoning about what to do, then deciding on an action.

The available actions are:
- CLICK [element_id]: Click on an element with the given ID
- TYPE [element_id] [text]: Type the text into an input field with the given ID
- SCROLL [direction]: Scroll the page up or down
- STOP [answer]: End the task with the given answer/result

First, reason about what you're seeing and the best action to take.
Then, end your response with one of the above actions.
"""

print("Generating ReACT trajectories with GPT-4o...")
trajectories = []

for i, task in enumerate(tasks):
    print(f"Task {i+1}/{len(tasks)}: {task}")
    
    # Initial observation for GitLab
    observation = "GitLab homepage showing a list of projects, navigation menu, and search bar."
    
    # Track the conversation
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task}\n\nObservation: {observation}"}
    ]
    
    # For tracking final data
    trajectory = [
        {"observation": {"text": observation}}
    ]
    
    # Simulate ReACT loop for a few steps
    for step in range(5):  # Maximum 5 steps per task
        # Call OpenAI
        response = call_openai(conversation)
        if not response:
            print(f"  Error in step {step+1}, skipping task")
            break
        
        # Extract reasoning and action
        assistant_response = response["choices"][0]["message"]["content"]
        
        # Split by newlines to separate reasoning from action
        parts = assistant_response.split("\n")
        # The last part should contain the action
        action_line = parts[-1].strip()
        reasoning = "\n".join(parts[:-1]).strip()
        
        # Extract action details
        if action_line.startswith("CLICK "):
            action_type = "CLICK"
            element_id = action_line[6:].strip()
            action_detail = {"action_type": action_type, "element_id": element_id, "raw_prediction": assistant_response}
            next_observation = f"Clicked on {element_id}. New page loaded showing related content."
        elif action_line.startswith("TYPE "):
            action_type = "TYPE"
            parts = action_line[5:].split(" ", 1)
            element_id = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            action_detail = {"action_type": action_type, "element_id": element_id, "value": value, "raw_prediction": assistant_response}
            next_observation = f"Typed '{value}' into {element_id} field."
        elif action_line.startswith("SCROLL "):
            action_type = "SCROLL"
            direction = action_line[7:].strip()
            action_detail = {"action_type": action_type, "direction": direction, "raw_prediction": assistant_response}
            next_observation = f"Scrolled {direction}. More content visible."
        elif action_line.startswith("STOP "):
            action_type = "STOP"
            answer = action_line[5:].strip()
            action_detail = {"action_type": action_type, "answer": answer, "raw_prediction": assistant_response}
            next_observation = "Task completed."
            
            # Add the action to trajectory
            trajectory.append({"action": action_detail})
            break
        else:
            # Could not parse action, use as is
            action_type = "UNKNOWN"
            action_detail = {"action_type": action_type, "raw_prediction": assistant_response}
            next_observation = "Unable to understand the action. Please try a valid action format."
        
        # Add to trajectory
        trajectory.append({"action": action_detail})
        trajectory.append({"observation": {"text": next_observation}})
        
        # Update conversation for next round
        conversation.append({"role": "assistant", "content": assistant_response})
        conversation.append({"role": "user", "content": f"Observation: {next_observation}"})
        
        # Break if STOP action
        if action_type == "STOP":
            break
            
        # Be nice to the API
        time.sleep(1)
    
    # Save the full trajectory
    action_history = []
    for item in trajectory:
        if "action" in item and "action_type" in item["action"]:
            action = item["action"]
            if action["action_type"] == "CLICK":
                action_history.append(f"CLICK {action['element_id']}")
            elif action["action_type"] == "TYPE":
                action_history.append(f"TYPE {action['element_id']} {action.get('value', '')}")
            elif action["action_type"] == "SCROLL":
                action_history.append(f"SCROLL {action.get('direction', '')}")
            elif action["action_type"] == "STOP":
                action_history.append(f"STOP {action.get('answer', '')}")
    
    trajectories.append({
        "task_id": i+1,
        "intent": task,
        "trajectory": trajectory,
        "score": 1.0,  # Assume all are successful for demonstration
        "action_history": action_history
    })
    
    print(f"  Generated trajectory with {len(trajectory)} steps")

# Save trajectories
output_dir = "data/webarena/eval_results/react/sample_data"
os.makedirs(output_dir, exist_ok=True)

for i, traj in enumerate(trajectories):
    with open(f"{output_dir}/test_task_{i+1}.json", "w") as f:
        json.dump(traj, f, indent=2)

print(f"\nSaved {len(trajectories)} trajectories to {output_dir}")

# Convert to finetuning format
def create_finetune_data():
    """Convert trajectories to finetuning format"""
    data_dir = "data/webarena/eval_results/react/sample_data"
    output_dir = "data/training"
    
    # Final training examples
    training_examples = []
    
    for task_file in [f"{data_dir}/test_task_{i+1}.json" for i in range(len(trajectories))]:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
            
        # Extract task information
        intent = task_data["intent"]
        trajectory = task_data["trajectory"]
        
        # Create SFT example
        prompt = f"Task: {intent}\n\n"
        completion = ""
        
        # Process observation-action pairs
        i = 0
        while i < len(trajectory):
            # Get observation
            if i < len(trajectory) and "observation" in trajectory[i]:
                obs = trajectory[i]
                if "text" in obs["observation"]:
                    prompt += f"Observation: {obs['observation']['text']}\n\n"
            
            # Get action (should be right after observation)
            if i+1 < len(trajectory) and "action" in trajectory[i+1]:
                act = trajectory[i+1]
                if "raw_prediction" in act["action"]:
                    # Use the raw prediction as completion
                    completion = act["action"]["raw_prediction"]
                    
                    # Create training example for this observation-action pair
                    training_examples.append({
                        "prompt": prompt.strip(),
                        "completion": completion.strip()
                    })
                    
                    # Reset prompt for next pair but keep task intent
                    if i+2 < len(trajectory) and "observation" in trajectory[i+2]:
                        prompt = f"Task: {intent}\n\n"
            
            i += 1
    
    # Save examples to a single JSONL file
    output_file = os.path.join(output_dir, "gitlab_finetune_data.jsonl")
    with open(output_file, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Created finetuning data with {len(training_examples)} examples")
    print(f"Output saved to: {output_file}")

# Convert to OpenAI chat format
def create_openai_format():
    """Convert to OpenAI finetuning format"""
    input_file = "data/training/gitlab_finetune_data.jsonl"
    output_file = "data/training/gitlab_openai_finetune.jsonl"
    
    training_examples = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            training_examples.append(data)
    
    with open(output_file, 'w') as f:
        for example in training_examples:
            # OpenAI format
            openai_format = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]}
                ]
            }
            f.write(json.dumps(openai_format) + "\n")
    
    print(f"Created OpenAI-compatible finetuning data: {output_file}")

# Run conversion
create_finetune_data()
create_openai_format()

print("\nFinetuning data preparation complete!")
print("You can now use these files for your model finetuning.")
print("data/training/gitlab_finetune_data.jsonl - Standard format")
print("data/training/gitlab_openai_finetune.jsonl - OpenAI chat format") 