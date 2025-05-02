#!/usr/bin/env python
"""
Simple script to create finetuning data from GitLab trajectories without heavy dependencies
"""
import os
import json
import glob
from datetime import datetime

# Create necessary directories
os.makedirs("data/training", exist_ok=True)
os.makedirs("data/webarena/eval_results/react/sample_data", exist_ok=True)

# Sample GitLab trajectories
sample_tasks = [
    {
        "task_id": 1,
        "intent": "Create a new issue in the 'ProjectX' repository with the title 'Bug in login feature' and description 'Users cannot login with correct credentials'",
        "trajectory": [
            {"observation": {"text": "GitLab homepage showing projects list"}},
            {"action": {"action_type": "CLICK", "element_id": "project-x", "raw_prediction": "I will click on the ProjectX repository to access it"}},
            {"observation": {"text": "ProjectX repository page with tabs for Issues, Merge Requests, etc."}},
            {"action": {"action_type": "CLICK", "element_id": "issues-tab", "raw_prediction": "I will click on the Issues tab to view existing issues"}},
            {"observation": {"text": "Issues page with 'New Issue' button"}},
            {"action": {"action_type": "CLICK", "element_id": "new-issue", "raw_prediction": "I will click the New Issue button to create a new issue"}},
            {"observation": {"text": "New Issue form with title and description fields"}},
            {"action": {"action_type": "TYPE", "element_id": "issue-title", "value": "Bug in login feature", "raw_prediction": "I will type the title 'Bug in login feature' in the title field"}},
            {"observation": {"text": "Title field filled"}},
            {"action": {"action_type": "TYPE", "element_id": "issue-description", "value": "Users cannot login with correct credentials", "raw_prediction": "I will type the description 'Users cannot login with correct credentials' in the description field"}},
            {"observation": {"text": "Form filled with title and description"}},
            {"action": {"action_type": "CLICK", "element_id": "submit-issue", "raw_prediction": "I will submit the new issue by clicking the Submit button"}},
            {"observation": {"text": "Issue created successfully"}},
            {"action": {"action_type": "STOP", "answer": "Successfully created a new issue in ProjectX titled 'Bug in login feature'", "raw_prediction": "The task is complete. I've created a new issue in the ProjectX repository with the title 'Bug in login feature' and description 'Users cannot login with correct credentials'"}}
        ],
        "score": 1.0,
        "action_history": [
            "CLICK project-x", 
            "CLICK issues-tab", 
            "CLICK new-issue", 
            "TYPE issue-title Bug in login feature",
            "TYPE issue-description Users cannot login with correct credentials",
            "CLICK submit-issue",
            "STOP Successfully created a new issue in ProjectX titled 'Bug in login feature'"
        ]
    },
    {
        "task_id": 2,
        "intent": "Find the repository called 'Documentation' and star it",
        "trajectory": [
            {"observation": {"text": "GitLab homepage showing projects list"}},
            {"action": {"action_type": "TYPE", "element_id": "search-input", "value": "Documentation", "raw_prediction": "I will search for the Documentation repository"}},
            {"observation": {"text": "Search results showing Documentation repository"}},
            {"action": {"action_type": "CLICK", "element_id": "documentation-repo", "raw_prediction": "I will click on the Documentation repository"}},
            {"observation": {"text": "Documentation repository page"}},
            {"action": {"action_type": "CLICK", "element_id": "star-button", "raw_prediction": "I will click on the star button to star the repository"}},
            {"observation": {"text": "Repository starred successfully"}},
            {"action": {"action_type": "STOP", "answer": "Successfully starred the Documentation repository", "raw_prediction": "The task is complete. I've found and starred the Documentation repository"}}
        ],
        "score": 1.0,
        "action_history": [
            "TYPE search-input Documentation", 
            "CLICK documentation-repo", 
            "CLICK star-button",
            "STOP Successfully starred the Documentation repository"
        ]
    }
]

# Save sample task data
for i, task in enumerate(sample_tasks):
    with open(f"data/webarena/eval_results/react/sample_data/test_task_{i+1}.json", "w") as f:
        json.dump(task, f, indent=2)

print("Created sample GitLab task data")

# Process trajectories for finetuning
def create_finetune_data():
    """Convert trajectories to finetuning format"""
    data_dir = "data/webarena/eval_results/react/sample_data"
    output_dir = "data/training"
    
    # Final training examples
    training_examples = []
    
    for task_file in glob.glob(f"{data_dir}/test_task_*.json"):
        with open(task_file, 'r') as f:
            task_data = json.load(f)
            
        # Extract task information
        intent = task_data["intent"]
        trajectory = task_data["trajectory"]
        
        # Create SFT example
        prompt = f"Task: {intent}\n\n"
        completion = ""
        
        # Process trajectory pairs (observation, action)
        for i in range(0, len(trajectory), 2):
            if i < len(trajectory):
                obs = trajectory[i]
                
                # Add observation to prompt
                if "observation" in obs and "text" in obs["observation"]:
                    prompt += f"Observation: {obs['observation']['text']}\n\n"
                
            if i+1 < len(trajectory):
                act = trajectory[i+1]
                
                # Add raw prediction as completion
                if "action" in act and "raw_prediction" in act["action"]:
                    reasoning = act["action"]["raw_prediction"]
                    
                    # Get action details
                    action_type = act["action"]["action_type"]
                    
                    if action_type == "CLICK":
                        element_id = act["action"]["element_id"]
                        action_str = f"CLICK {element_id}"
                    elif action_type == "TYPE":
                        element_id = act["action"]["element_id"]
                        value = act["action"]["value"]
                        action_str = f"TYPE {element_id} {value}"
                    elif action_type == "STOP":
                        answer = act["action"].get("answer", "")
                        action_str = f"STOP {answer}"
                    else:
                        # Handle other action types if needed
                        action_str = f"{action_type}"
                    
                    completion += f"I'll analyze the current state of the webpage.\n\n{reasoning}\n\nAction: {action_str}\n\n"
            
        # Add to training examples
        if prompt and completion:
            training_examples.append({
                "prompt": prompt.strip(),
                "completion": completion.strip()
            })
    
    # Save all examples to a single JSONL file
    output_file = os.path.join(output_dir, "gitlab_finetune_data.jsonl")
    with open(output_file, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Created finetuning data with {len(training_examples)} examples")
    print(f"Output saved to: {output_file}")

# Run the conversion
create_finetune_data()

# Also output OpenAI-compatible format
def create_openai_format():
    """Convert to OpenAI finetuning format"""
    input_file = "data/training/gitlab_finetune_data.jsonl"
    output_file = "data/training/gitlab_openai_finetune.jsonl"
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            # OpenAI format
            openai_format = {
                "messages": [
                    {"role": "system", "content": "You are a helpful web navigation assistant."},
                    {"role": "user", "content": data["prompt"]},
                    {"role": "assistant", "content": data["completion"]}
                ]
            }
            f_out.write(json.dumps(openai_format) + "\n")
    
    print(f"Created OpenAI-compatible finetuning data: {output_file}")

create_openai_format()

print("\nFinetuning data preparation complete!")
print("You can now use these files for your model finetuning.")
print("data/training/gitlab_finetune_data.jsonl - Standard format")
print("data/training/gitlab_openai_finetune.jsonl - OpenAI chat format") 