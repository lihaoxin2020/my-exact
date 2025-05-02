
#!/bin/bash
export PYTHONPATH=$(pwd)
python runners/eval/eval_vwa_parallel.py \
    --env_name gitlab \
    --save_dir data/webarena/eval_results/rmcts_mad/o4-mini_gitlab \
    --eval_script shells/gitlab/rmcts_mad.sh \
    --run_mode greedy \
    --start_idx 0 \
    --end_idx 196 \
    --num_parallel 16 \
    --main_api_providers openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai,openai \
    --num_task_per_script 13 \
    --num_task_per_reset 197
