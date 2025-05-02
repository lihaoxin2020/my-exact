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
export OPENAI_API_KEY="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"
export PROVIDER="openai"

# Verify API key is working
echo "Verifying OpenAI API key..."
status_code=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY")
if [ "$status_code" != "200" ]; then
  echo "Error: API key verification failed (HTTP $status_code)"
  echo "Please update the script with a valid OpenAI API key."
  exit 1
fi
echo "API key verified successfully!"

# Configuration
model="o4-mini"
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
test_config_dir="configs/webarena/test_gitlab_v2"
agent="prompt"
max_steps=20
prompt_constructor_type=CoTPolicyPConstructor

# Generate timestamp for unique output directory
timestamp=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="data/webarena/eval_results/react_text/gitlab_parallel_$timestamp"
mkdir -p $BATCH_DIR
mkdir -p $BATCH_DIR/logs

# Run the first 10 tasks in parallel
echo "Running the first 10 GitLab tasks in parallel..."
echo "Results will be saved to $BATCH_DIR"

# Function to run a single task
run_task() {
  local task_idx=$1
  local output_dir="$BATCH_DIR/task_$task_idx"
  local log_file="$BATCH_DIR/logs/task_${task_idx}.log"
  
  mkdir -p "$output_dir"
  
  echo "Starting task $task_idx..." > "$log_file"
  
  python runners/eval/eval_vwa_agent.py \
    --instruction_path $instruction_path \
    --test_idx $task_idx \
    --model $model \
    --provider $PROVIDER \
    --agent_type $agent \
    --prompt_constructor_type $prompt_constructor_type \
    --result_dir $output_dir \
    --test_config_base_dir $test_config_dir \
    --repeating_action_failure_th 5 \
    --viewport_height 2048 \
    --max_obs_length 3840 \
    --action_set_tag id_accessibility_tree \
    --observation_type accessibility_tree \
    --temperature 0.7 \
    --top_p 0.9 \
    --eval_captioning_model_device cpu \
    --max_steps $max_steps \
    --render false >> "$log_file" 2>&1
    
  echo "Task $task_idx completed" >> "$log_file"
}

# Start tasks in parallel
for i in {0..9}; do
  run_task $i &
  echo "Started task $i in background"
  # Short delay to prevent API rate limiting
  sleep 1
done

# Wait for all background processes to finish
echo "Waiting for all tasks to complete..."
wait

echo "All tasks completed. Check results in $BATCH_DIR" 