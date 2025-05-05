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
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
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
max_steps=5
prompt_constructor_type=CoTPolicyPConstructor
test_idx=0  # Just run task 0 for testing

# Context window management for Together AI
# The model has a 131,073 token context window
# We need to ensure input + output tokens stay under this limit
# Based on error logs, we need to reduce the observation length

# Generate timestamp for unique output directory
timestamp=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="data/webarena/eval_results/react_text/gitlab_single_test_${timestamp}"
mkdir -p $SAVE_DIR
mkdir -p $SAVE_DIR/trajectories

echo "==========================================================="
echo "Running SINGLE GitLab task $test_idx with TogetherAI model"
echo "Using model: $MODEL_NAME"
echo "Provider: $PROVIDER"
echo "Results will be saved to: $SAVE_DIR"
echo "==========================================================="

# Run the task directly (without background process)
python runners/eval/eval_vwa_agent.py \
  --instruction_path $instruction_path \
  --test_idx $test_idx \
  --model $MODEL_NAME \
  --provider $PROVIDER \
  --mode chat \
  --agent_type $agent \
  --prompt_constructor_type $prompt_constructor_type \
  --result_dir $SAVE_DIR \
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
  --eval_captioning_model_device cpu
  
echo "Task $test_idx completed. Results are in $SAVE_DIR"

# Check if trajectory file exists and print information
traj_files=$(find "$SAVE_DIR/trajectories" -name "*.pkl.xz" -type f 2>/dev/null)
if [ -n "$traj_files" ]; then
  echo "Trajectory files found:"
  ls -la $SAVE_DIR/trajectories/
  echo ""
  echo "To convert pickle trajectories to JSON format, run:"
  echo "python convert_trajectories.py --input_dir $SAVE_DIR/trajectories"
else
  echo "Warning: No trajectory files found in $SAVE_DIR/trajectories"
  echo "Checking for trajectories in unexpected locations..."
  find "$SAVE_DIR" -name "*.pkl.xz" -type f
fi 