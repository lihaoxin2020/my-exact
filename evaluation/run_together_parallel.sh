#!/bin/bash
export PYTHONPATH=$(pwd)
export DATASET=webarena

# Setup environment variables
export HOST_NAME=http://ec2-52-14-25-77.us-east-2.compute.amazonaws.com
export GITLAB="$HOST_NAME:8023"
export REDDIT=example.com
export SHOPPING=example.com
export SHOPPING_ADMIN=example.com
export WIKIPEDIA=example.com
export MAP=example.com
export HOMEPAGE=example.com
export CLASSIFIEDS=example.com
export CLASSIFIEDS_RESET_TOKEN=dummy

# Set the TogetherAI API keys and configuration
export PROVIDER="together"
export TOGETHER_API_KEY="42da6722c7854fd01458df45459de6521298ec4b742b2bbb29b57173d90e3512"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# Dummy OpenAI API key needed for validation checks in the evaluation script
export OPENAI_API_KEY="dummy_key_for_validation_only"
# Make sure value function also uses Together
export VALUE_FUNC_PROVIDER="together"
export VALUE_FUNC_MODEL=$MODEL_NAME
# Make absolutely sure no fallback to OpenAI
export HF_TOKEN="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"
export BACKEND_PROVIDER="together"
export POLICY_PROVIDER="together"
export DEFAULT_PROVIDER="together"
export VALUE_FUNC_API_BASE="https://api.together.xyz/v1"

# Create URL mapping patch script
cat > patch_url_mappings.py << EOF
import os
import re

# Find the environment config file
env_config_paths = [
    "visualwebarena/browser_env/env_config.py",
    "browser_env/env_config.py"
]

env_config_path = None
for path in env_config_paths:
    if os.path.exists(path):
        env_config_path = path
        break

if not env_config_path:
    print("Error: Could not find environment config file")
    exit(1)

print(f"Found environment config at {env_config_path}")

# Read the current content
with open(env_config_path, "r") as f:
    content = f.read()

# Define our EC2 host
ec2_host = "http://ec2-52-14-25-77.us-east-2.compute.amazonaws.com:8023"
local_host = "http://gitlab.localhost:8080"

# Find all URL_MAPPINGS definitions
pattern = r'(URL_MAPPINGS\s*=\s*\{[^}]*\})'
matches = re.findall(pattern, content, re.DOTALL)

if matches:
    for match in matches:
        # Check if the GitLab URL is already mapped correctly
        if f'"{local_host}": "{ec2_host}"' in match:
            print(f"GitLab URL is already correctly mapped in {match[:50]}...")
            continue
            
        # If not, update the mapping
        if local_host in match:
            # Replace the existing mapping
            updated_mapping = re.sub(
                f'"{local_host}":\\s*"[^"]*"', 
                f'"{local_host}": "{ec2_host}"', 
                match
            )
        else:
            # Add a new mapping entry
            updated_mapping = match.rstrip("}") + f',\n    "{local_host}": "{ec2_host}"\n}}'
        
        # Replace in the content
        content = content.replace(match, updated_mapping)
        print(f"Updated URL mapping: {local_host} -> {ec2_host}")
    
    # Write changes back to the file
    with open(env_config_path, "w") as f:
        f.write(content)
    print("URL mappings updated successfully!")
else:
    print("Warning: Could not find URL_MAPPINGS in the file")
EOF

# Run the URL mapping patch
echo "Patching URL mappings to fix GitLab connectivity..."
python patch_url_mappings.py

# Verify TogetherAI API key is working
echo "Verifying TogetherAI API key..."
status_code=$(curl -s -o /dev/null -w "%{http_code}" https://api.together.xyz/v1/models -H "Authorization: Bearer $TOGETHER_API_KEY")
if [ "$status_code" != "200" ]; then
  echo "Error: TogetherAI API key verification failed (HTTP $status_code)"
  echo "Please update the script with a valid TogetherAI API key."
  exit 1
fi
echo "API key verified successfully!"

# Configuration
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
test_config_dir="configs/webarena/test_gitlab_v2_filtered"
agent="prompt"  # ReACT is implemented as a "prompt" agent type
max_steps=20
prompt_constructor_type=CoTPolicyPConstructor

# Generate timestamp for unique output directory
timestamp=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="data/webarena/eval_results/react_text/together_parallel_$timestamp"
mkdir -p $BATCH_DIR
mkdir -p $BATCH_DIR/logs

# Create trajectories directory for SFT data
TRAJECTORIES_DIR="$BATCH_DIR/trajectories"
mkdir -p $TRAJECTORIES_DIR

# Create a progress file
progress_file="$BATCH_DIR/progress.txt"
echo "0/164 tasks completed (0%)" > $progress_file

# Maximum number of concurrent tasks
MAX_CONCURRENT=10
completed=0
total_tasks=164

# Function to run a single task
run_task() {
  local task_idx=$1
  local output_dir="$BATCH_DIR/task_$task_idx"
  local log_file="$BATCH_DIR/logs/task_${task_idx}.log"
  
  mkdir -p "$output_dir"
  
  echo "Starting task $task_idx..." > "$log_file"
  
  # Set environment variables for GitLab URL redirection for this task
  export GITLAB="$HOST_NAME:8023"
  export GITLAB_URL="$HOST_NAME:8023"
  export URL_MAPPING_GITLAB="$HOST_NAME:8023"
  export OVERRIDE_GITLAB_URL="$HOST_NAME:8023"
  
  # Create a local URL mapping file for this task
  cat > "$output_dir/url_override.py" << EOF
import os
import builtins
import sys

# Add monkey patch to ensure gitlab.localhost is redirected correctly
original_import = builtins.__import__

def patched_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    
    # If this is the env_config module, patch URL_MAPPINGS
    if name == 'browser_env.env_config' and hasattr(module, 'URL_MAPPINGS'):
        ec2_host = "$HOST_NAME:8023"
        local_host = "http://gitlab.localhost:8080"
        
        # Ensure the mapping is correct
        if module.URL_MAPPINGS.get(local_host) != ec2_host:
            module.URL_MAPPINGS[local_host] = ec2_host
            print(f"Task {$task_idx}: Patched URL mapping at runtime: {local_host} -> {ec2_host}")
    
    return module

# Apply the monkey patch
builtins.__import__ = patched_import
EOF

  # Run with the URL mapping override
  PYTHONPATH="$output_dir:$PYTHONPATH" python -c "import url_override" >> "$log_file" 2>&1
  
  # Run the Python command with explicit parameters only
  python runners/eval/eval_vwa_agent.py \
    --instruction_path $instruction_path \
    --test_idx $task_idx \
    --model $MODEL_NAME \
    --provider $PROVIDER \
    --mode chat \
    --agent_type $agent \
    --prompt_constructor_type $prompt_constructor_type \
    --result_dir $output_dir \
    --test_config_base_dir $test_config_dir \
    --repeating_action_failure_th 10 \
    --parsing_failure_th 10 \
    --viewport_height 2048 \
    --max_obs_length 3840 \
    --action_set_tag id_accessibility_tree \
    --observation_type accessibility_tree \
    --temperature 0.0 \
    --top_p 0.9 \
    --eval_captioning_model_device cpu \
    --max_steps $max_steps \
    --current_viewport_only true \
    --render false >> "$log_file" 2>&1
    
  echo "Task $task_idx completed" >> "$log_file"
  
  # Copy trajectory to trajectories folder for easy SFT collection (as backup)
  if [ -f "$output_dir/trajectories/task_${task_idx}.pkl.xz" ]; then
    cp "$output_dir/trajectories/task_${task_idx}.pkl.xz" "$TRAJECTORIES_DIR/"
    echo "Saved trajectory for task $task_idx to trajectories folder" >> "$log_file"
  fi
  
  # Update progress counter and file
  completed=$((completed+1))
  percentage=$((completed*100/total_tasks))
  echo "$completed/$total_tasks tasks completed ($percentage%)" > $progress_file
  echo "Completed task $task_idx. Progress: $completed/$total_tasks ($percentage%)"
}

# Function to check current number of running tasks
count_running_tasks() {
  jobs -p | wc -l
}

echo "Running all GitLab tasks with TogetherAI model in parallel (max $MAX_CONCURRENT concurrent tasks)..."
echo "Using model: $MODEL_NAME"
echo "Results will be saved to $BATCH_DIR"
echo "Trajectories for SFT will be saved to $TRAJECTORIES_DIR"
echo "Progress will be tracked in $progress_file"

# Start tasks in batches to control concurrency
for i in {0..163}; do
  # Wait if we've reached the maximum number of concurrent tasks
  while [ $(count_running_tasks) -ge $MAX_CONCURRENT ]; do
    sleep 5
  done
  
  run_task $i &
  echo "Started task $i in background"
  # Short delay to prevent API rate limiting
  sleep 2
done

# Wait for all background processes to finish
echo "Waiting for all remaining tasks to complete..."
wait

echo "All tasks completed. Check results in $BATCH_DIR"
echo "SFT trajectories are available in $TRAJECTORIES_DIR"
echo "Final progress:"
cat $progress_file 