#!/bin/bash
export PYTHONPATH=$(pwd)
export DATASET=webarena

## Define the model, result directory, and instruction path variables
export PROVIDER="openai"
export OPENAI_API_KEY="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"
[[[API_PROVIDER_ENV_VARS]]]  # replaced by runners/eval/eval_vwa_parallel.py

# Set up only GitLab URL
export HOST_NAME=http://ec2-52-14-25-77.us-east-2.compute.amazonaws.com
export GITLAB="$HOST_NAME:8023"

# Other variables (not used but referenced in code)
export REDDIT="example.com"
export SHOPPING="example.com"
export SHOPPING_ADMIN="example.com"
export WIKIPEDIA="example.com"
export MAP="example.com"
export HOMEPAGE="example.com"
export CLASSIFIEDS="example.com"
export CLASSIFIEDS_RESET_TOKEN="dummy"

EVAL_GPU_IDX=0

model="gpt-4o"
model_id="gpt-4o"
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"  # Using text-based prompt
test_config_dir="configs/webarena/test_gitlab_v2"

agent="prompt"  # ReACT is implemented as a "prompt" agent type
max_steps=5
prompt_constructor_type=CoTPolicyPConstructor  # Using a simpler prompt constructor for text modality

# Test GitLab task ID (will be replaced by parallel runner)
test_idx="[[[test_idx]]]"  # replaced by runners/eval/eval_vwa_parallel.py

RUN_FILE=runners/eval/eval_vwa_agent.py
# Output directory (will be replaced by parallel runner)
SAVE_ROOT_DIR="[[[SAVE_ROOT_DIR]]]"  # replaced by runners/eval/eval_vwa_parallel.py
echo "SAVEDIR=${SAVE_ROOT_DIR}"
mkdir -p $SAVE_ROOT_DIR
cp "$0" "${SAVE_ROOT_DIR}/run.sh"

####### start eval
# Note: This will connect to the GitLab service but not other services
echo "Running ReACT agent on GitLab only (text-only mode)..."
python $RUN_FILE \
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

##### cleanups
python runners/utils/repartition_log_files.py $SAVE_ROOT_DIR/log_files 