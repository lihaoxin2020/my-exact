#!/bin/bash
export PYTHONPATH=$(pwd)
export DATASET=webarena

## Define the model, result directory, and instruction path variables
export PROVIDER="openai"
export AGENT_LLM_API_BASE="https://api.openai.com/v1"
export AGENT_LLM_API_KEY="$(echo $OPENAI_API_KEY)"
export VALUE_FUNC_PROVIDER="openai"
export VALUE_FUNC_API_BASE="https://api.openai.com/v1"
export RLM_PROVIDER="openai"  # not used as it will become PROVIDER
export OPENAI_API_BASE="https://api.openai.com/v1"
export AZURE_TOKEN_PROVIDER_BASE=""
export AZURE_OPENAI_API_VERSION=""
EVAL_GPU_IDX=0

model="o4-mini"  # "gpt-4o-mini"
model_id="o4-mini"
# instruction_path="src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
# test_config_dir="configs/visualwebarena/test_classifieds_v2"
test_config_dir="configs/webarena/test_gitlab_v2"

# ##### start of search config
max_depth=4  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=5  # default 5
vf_budget=20        # default 20
time_budget=5.0     # 5.0 min per step (soft maximum), will override vf_budget if > 0.0

agent="mcts"

##### start of search config
max_depth=4  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=5  # default 5
vf_budget=20        # default 20
time_budget=5.0     # 5.0 min per step (soft maximum), will override vf_budget if > 0.0

# policy config
# prompt_constructor_type=MCoTPolicyPConstructor
prompt_constructor_type=CoTPolicyPConstructor

v_func_method=CoTwRubricValueFunction  # default CoTwRubricValueFunction
##### end of search config

# test_idx="10,11,70,71"  # task ids to run
test_idx="0,2"

RUN_FILE=runners/eval/eval_vwa_agent.py
SAVE_ROOT_DIR=data/${DATASET}/eval_results/$agent/example
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
--top_p 0.95 --temperature 1.0 --max_steps $max_steps

##### cleanups
python runners/utils/repartition_log_files.py $SAVE_ROOT_DIR/log_files