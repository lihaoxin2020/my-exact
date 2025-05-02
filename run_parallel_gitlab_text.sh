#!/bin/bash
export PYTHONPATH=$(pwd)

# Delete expired cookies to force new login
rm -f ./.auth/gitlab.shopping_state.json

# Set all required environment variables
export DATASET=webarena
export HOST_NAME=http://ec2-52-14-25-77.us-east-2.compute.amazonaws.com
export GITLAB=$HOST_NAME:8023
export REDDIT=example.com
export SHOPPING=example.com
export SHOPPING_ADMIN=example.com
export WIKIPEDIA=example.com
export MAP=example.com
export HOMEPAGE=example.com
export CLASSIFIEDS=example.com
export CLASSIFIEDS_RESET_TOKEN=dummy
export OPENAI_API_KEY=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

# Run the parallel evaluation script
python runners/eval/eval_vwa_parallel.py \
    --env_name gitlab \
    --save_dir data/webarena/eval_results/react_text/gitlab_all_tasks \
    --eval_script shells/gitlab/react_text_parallel.sh \
    --run_mode greedy \
    --start_idx 0 \
    --end_idx 196 \
    --num_parallel 16 \
    --main_api_providers openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai \
    --num_task_per_script 13 \
    --num_task_per_reset 197 