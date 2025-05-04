#!/bin/bash
export PYTHONPATH=$(pwd)
# source <path_to_api_key_envs>/.keys
# export DATASET=visualwebarena
export DATASET=webarena

export PROVIDER="openai"
export AGENT_LLM_API_BASE="https://api.openai.com/v1"
export AGENT_LLM_API_KEY="$(echo $OPENAI_API_KEY)"
export VALUE_FUNC_PROVIDER="openai"
export VALUE_FUNC_API_BASE="https://api.openai.com/v1"
# export RLM_PROVIDER="openai"  # not used as it will become PROVIDER
export OPENAI_API_BASE="https://api.openai.com/v1"
export AZURE_TOKEN_PROVIDER_BASE=""
export AZURE_OPENAI_API_VERSION=""

HOST_NAME=http://ec2-3-12-126-100.us-east-2.compute.amazonaws.com
export CLASSIFIEDS="$HOST_NAME:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
export SHOPPING="$HOST_NAME:7770"
export REDDIT="$HOST_NAME:9999"
export WIKIPEDIA="$HOST_NAME:8888"
export SHOPPING_ADMIN="$HOST_NAME:7780/admin"
export GITLAB="$HOST_NAME:8023"
export MAP="$HOST_NAME:3000"
export HOMEPAGE="$HOST_NAME:4399"


## Define the model, result directory, and instruction path variables
[[[API_PROVIDER_ENV_VARS]]]  # replaced by runners/eval/eval_vwa_parallel.py
EVAL_GPU_IDX=0,1

model="o4-mini"  # "gpt-4o-mini"
model_id="o4-mini"
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
test_config_dir="configs/webarena/test_gitlab_v2"

# change this to "prompt" to run the baseline without search
agent="mcts"

##### start of search config
max_depth=4  # max_depth=4 means 5 step lookahead
max_steps=20
branching_factor=5  # default 5
vf_budget=20        # default 20
time_budget=5.0     # 5.0 min per step (soft maximum), will override vf_budget if > 0.0

prompt_constructor_type=CoTPolicyPConstructor

# vfunc config
v_func_method=CoTwRubricValueFunction  # default DirectCoTValueFunction
##### end of search config

test_idx="[[[test_idx]]]"  # replaced by runners/eval/eval_vwa_parallel.py

RUN_FILE=runners/eval/eval_vwa_agent.py
SAVE_ROOT_DIR="[[[SAVE_ROOT_DIR]]]"  # replaced by runners/eval/eval_vwa_parallel.py
echo "SAVEDIR=${SAVE_ROOT_DIR}"
mkdir -p $SAVE_ROOT_DIR
cp "$0" "${SAVE_ROOT_DIR}/run.sh"

export DEBUG=True  # export DEBUG=''
####### start eval
# reset, reserving, and freeing is handled by an external script

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} \
python $RUN_FILE \
--instruction_path $instruction_path \
--test_idx $test_idx \
--model $model \
--provider $PROVIDER \
--agent_type $agent \
--prompt_constructor_type $prompt_constructor_type \
--branching_factor $branching_factor --vf_budget $vf_budget --time_budget $time_budget \
--value_function $model \
--value_function_method $v_func_method \
--result_dir $SAVE_ROOT_DIR \
--test_config_base_dir $test_config_dir \
--repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
--action_set_tag id_accessibility_tree  --observation_type accessibility_tree \
--top_p 0.95   --temperature 1.0  --max_steps $max_steps

####### end eval
python runners/utils/repartition_log_files.py $SAVE_ROOT_DIR/log_files