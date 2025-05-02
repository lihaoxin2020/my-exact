#!/bin/bash

# experimental
export PYTHONPATH=$(pwd)
export VALUE_FUNC_PROVIDER=openai
export VALUE_FUNC_API_BASE=https://api.openai.com/v1
export HF_TOKEN=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
python runners/train/tree_to_data.py \
    --env_name gitlab \
    --result_dir data/webarena/eval_results/search_refactored/example \
    --output_dir data/training
