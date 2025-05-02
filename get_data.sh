#!/bin/bash

# experimental
python runners/train/tree_to_data.py \
    --env_name gitlab \
    --result_dir data/webarena/eval_results/rmcts_som/example \
    --output_dir data/training
