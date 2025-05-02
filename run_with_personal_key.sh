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
model="gpt-4o"
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
test_config_dir="configs/webarena/test_gitlab_v2"
agent="prompt"
max_steps=5
prompt_constructor_type=CoTPolicyPConstructor

# Task ID to run (change as needed)
test_idx=0

# Output directory
SAVE_DIR="data/webarena/eval_results/react_text/gitlab_single_task"
mkdir -p $SAVE_DIR

echo "Running GitLab task $test_idx..."

# Run the task directly
python runners/eval/eval_vwa_agent.py \
    --instruction_path $instruction_path \
    --test_idx $test_idx \
    --model $model \
    --provider $PROVIDER \
    --agent_type $agent \
    --prompt_constructor_type $prompt_constructor_type \
    --result_dir $SAVE_DIR \
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
    --render false

echo "Task $test_idx completed." 