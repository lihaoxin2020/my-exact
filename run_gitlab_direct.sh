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

# IMPORTANT: Update with your own OpenAI API key
# The project-scoped key may not work; replace with a personal key
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export PROVIDER="openai"

# Configuration
model="gpt-4o"
model_id="gpt-4o"
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
test_config_dir="configs/webarena/test_gitlab_v2"
agent="prompt"
max_steps=5
prompt_constructor_type=CoTPolicyPConstructor

# Verify OpenAI API key before proceeding
echo "Verifying OpenAI API key..."
if [[ "$OPENAI_API_KEY" == "YOUR_OPENAI_API_KEY" ]]; then
  echo "ERROR: Please replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key"
  exit 1
fi

# Test API connection
curl_output=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY")
if [[ $curl_output != "200" ]]; then
  echo "ERROR: Unable to connect to OpenAI API. HTTP status: $curl_output"
  echo "Please check your API key and internet connection."
  exit 1
fi
echo "OpenAI API connection verified successfully!"

# For example, run the first GitLab task (change test_idx as needed)
test_idx=0
SAVE_ROOT_DIR="data/webarena/eval_results/react_text/gitlab_single_task"
mkdir -p $SAVE_ROOT_DIR

echo "Running GitLab task $test_idx directly..."
# Using CPU mode for text-only interactions
python runners/eval/eval_vwa_agent.py \
--instruction_path $instruction_path \
--test_idx $test_idx \
--model $model \
--provider $PROVIDER \
--agent_type $agent \
--prompt_constructor_type $prompt_constructor_type \
--result_dir $SAVE_ROOT_DIR \
--test_config_base_dir $test_config_dir \
--repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
--action_set_tag id_accessibility_tree --observation_type accessibility_tree \
--temperature 0.7 --top_p 0.9 \
--eval_captioning_model_device cpu \
--max_steps $max_steps

echo "Task $test_idx completed." 