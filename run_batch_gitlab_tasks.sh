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
instruction_path="src/prompts/vwa/jsons/p_cot_id_actree_3s_final.json"
test_config_dir="configs/webarena/test_gitlab_v2"
agent="prompt"
max_steps=5
prompt_constructor_type=CoTPolicyPConstructor

# Run for which tasks?
START_IDX=0
END_IDX=9  # Just do the first 10 tasks to test

# Create a timestamp for this batch run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="data/webarena/eval_results/react_text/gitlab_batch_$TIMESTAMP"
mkdir -p $BATCH_DIR
mkdir -p $BATCH_DIR/logs

echo "Starting batch run for GitLab tasks $START_IDX to $END_IDX"
echo "Results will be saved to $BATCH_DIR"

# Run tasks one by one
for task_idx in $(seq $START_IDX $END_IDX); do
    echo "==================================================="
    echo "Running GitLab task $task_idx"
    echo "==================================================="
    
    # Create a task-specific directory
    TASK_DIR="$BATCH_DIR/task_$task_idx"
    mkdir -p $TASK_DIR
    
    # Run task and redirect output to log file
    python runners/eval/eval_vwa_agent.py \
        --instruction_path $instruction_path \
        --test_idx $task_idx \
        --model $model \
        --provider $PROVIDER \
        --agent_type $agent \
        --prompt_constructor_type $prompt_constructor_type \
        --result_dir $TASK_DIR \
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
        --render false \
        > $BATCH_DIR/logs/task_${task_idx}.log 2>&1
    
    echo "Task $task_idx completed"
    
    # Slight delay between tasks to prevent rate limiting
    sleep 2
done

echo "All tasks completed."
echo "Check $BATCH_DIR for results and $BATCH_DIR/logs for task logs." 