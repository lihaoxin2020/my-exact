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

# Configuration
model="gpt-4o"
model_id="gpt-4o"
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
test_config_dir="configs/webarena/test_gitlab_v2"
agent="prompt"
max_steps=5
prompt_constructor_type=CoTPolicyPConstructor

# Main output directory
MAIN_DIR="data/webarena/eval_results/react_text/gitlab_batch"
mkdir -p $MAIN_DIR

# How many tasks to run
start_idx=0
end_idx=195  # 0-indexed, so 196 tasks total would be 0-195

for test_idx in $(seq $start_idx $end_idx); do
    echo "==============================================="
    echo "Running GitLab task $test_idx..."
    echo "==============================================="
    
    # Create task-specific directory
    SAVE_ROOT_DIR="$MAIN_DIR/task_$test_idx"
    mkdir -p $SAVE_ROOT_DIR
    
    # Run the task
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
done

echo "All GitLab tasks completed." 