#!/bin/bash
export PYTHONPATH=$(pwd):$(pwd)/visualwebarena
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

# Set the API keys - ensure EVERY provider is set to "together"
export PROVIDER="together"
export TOGETHER_API_KEY="42da6722c7854fd01458df45459de6521298ec4b742b2bbb29b57173d90e3512"
export MODEL_NAME="nsk7153/Meta-Llama-3.1-8B-Instruct-Reference-react_gitlab-40401b2f"
# Need a valid format OpenAI key (even though it's not real) to pass validation
export OPENAI_API_KEY="sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890ab"
# Make sure value function also uses Together
export VALUE_FUNC_PROVIDER="together"
export VALUE_FUNC_MODEL=$MODEL_NAME
# Make absolutely sure no fallback to OpenAI
export HF_TOKEN="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"
export BACKEND_PROVIDER="together"
export POLICY_PROVIDER="together"
export DEFAULT_PROVIDER="together"

# Configuration
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
test_config_dir="configs/webarena/test_gitlab_lite"  # Using lite test config
agent="prompt"  # ReACT is implemented as a "prompt" agent type
max_steps=20
prompt_constructor_type=CoTPolicyPConstructor

# Context window management for Together AI
# The model has a 131,073 token context window
# We need to ensure input + output tokens stay under this limit
# Based on error logs, we need to reduce the observation length

# Generate timestamp for unique output directory
timestamp=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="data/webarena/eval_results/react_text/gitlab_parallel_${timestamp}"
mkdir -p $BATCH_DIR
mkdir -p $BATCH_DIR/logs

# Create trajectories directory for collecting all trajectory data
TRAJECTORIES_DIR="$BATCH_DIR/trajectories"
mkdir -p $TRAJECTORIES_DIR

# Create a progress file
progress_file="$BATCH_DIR/progress.txt"
echo "0/34 tasks completed (0%)" > $progress_file

# Maximum number of concurrent tasks
MAX_CONCURRENT=3
completed=0
total_tasks=34  # Using the lite test set (0-33)

# Function to run a single task
run_task() {
  local task_idx=$1
  local output_dir="$BATCH_DIR/task_$task_idx"
  local log_file="$BATCH_DIR/logs/task_${task_idx}.log"
  
  mkdir -p "$output_dir"
  mkdir -p "$output_dir/trajectories"
  
  echo "Starting task $task_idx..." > "$log_file"
  
  # Add environment variables for this specific task
  PROVIDER="together" \
  VALUE_FUNC_PROVIDER="together" \
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
    --max_obs_length 1000 \
    --max_tokens 512 \
    --action_set_tag id_accessibility_tree \
    --observation_type accessibility_tree \
    --temperature 0.0 \
    --top_p 0.9 \
    --max_steps $max_steps \
    --current_viewport_only true \
    --render false \
    --eval_captioning_model_device cpu >> "$log_file" 2>&1
    
  echo "Task $task_idx completed" >> "$log_file"
  
  # Copy any trajectory files found to central trajectories directory
  traj_files=$(find "$output_dir/trajectories" -name "*.pkl.xz" -type f 2>/dev/null)
  if [ -n "$traj_files" ]; then
    for traj_file in $traj_files; do
      # Extract the filename
      filename=$(basename "$traj_file")
      # Copy to central directory, rename with task ID if needed
      cp "$traj_file" "$TRAJECTORIES_DIR/${filename}"
      echo "Copied trajectory $filename for task $task_idx to central trajectories folder" >> "$log_file"
    done
  else
    echo "Warning: No trajectory files found for task $task_idx" >> "$log_file"
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

echo "Running GitLab tasks 0-33 with TogetherAI model using TEXT ONLY..."
echo "Using model: $MODEL_NAME"
echo "Maximum concurrent tasks: $MAX_CONCURRENT"
echo "Results will be saved to: $BATCH_DIR"
echo "All trajectories will be collected in: $TRAJECTORIES_DIR"
echo "Progress will be tracked in: $progress_file"

# Start tasks in batches to control concurrency
for i in $(seq 0 33); do
  # Wait if we've reached the maximum number of concurrent tasks
  while [ $(count_running_tasks) -ge $MAX_CONCURRENT ]; do
    sleep 5
  done
  
  run_task $i &
  echo "Started task $i in background"
  # Short delay to prevent overloading
  sleep 2
done

# Wait for all background processes to finish
echo "Waiting for all remaining tasks to complete..."
wait

# After everything is done, find any missed trajectories and collect them
echo "Searching for any additional trajectory files..."
find "$BATCH_DIR" -name "*.pkl.xz" -type f | while read traj_file; do
  filename=$(basename "$traj_file")
  if [ ! -f "$TRAJECTORIES_DIR/$filename" ]; then
    cp "$traj_file" "$TRAJECTORIES_DIR/"
    echo "Found and copied additional trajectory: $filename"
  fi
done

echo "All $total_tasks tasks completed. Check results in $BATCH_DIR"
echo "Trajectories are available in $TRAJECTORIES_DIR"
echo "Final progress:"
cat $progress_file

echo ""
echo "To convert pickle trajectories to JSON format, run:"
echo "python convert_trajectories.py --input_dir $TRAJECTORIES_DIR" 