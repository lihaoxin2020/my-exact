#!/bin/bash

export HOST_NAME="YOUR_HOST_NAME"

# Set environment variables for website URLs
export REDDIT="http://example.com:9999"
export SHOPPING="http://example.com:7770"
export SHOPPING_ADMIN="http://example.com:7780/admin"
export GITLAB="http://ec2-xx-xx-xx-xx.us-east-2.compute.amazonaws.com:8023"
export WIKIPEDIA="http://example.com:8888"
export MAP="http://example.com:3000"
export HOMEPAGE="http://example.com:4399"
export CLASSIFIEDS="http://example.com:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

# Set API key if not already set
if [ -z "$OPENAI_API_KEY" ]; then
  export OPENAI_API_KEY="YOUR_API_KEY_HERE"
fi

# Set Python path to include current directory
export PYTHONPATH=$(pwd):$PYTHONPATH

# Define input and output directories
INPUT_DIR="/Users/nikhilkhandekar/Documents/my-exact/data/webarena/eval_results/react_text/gitlab_parallel_20250502_201103"
OUTPUT_DIR="/Users/nikhilkhandekar/Documents/my-exact/processed_datasets/react_gitlab"
ENV_NAME="gitlab"
MODALITY="text"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the script with required arguments
python process_react_trajectories.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --env_name "$ENV_NAME" \
  --modality "$MODALITY"

echo "Script completed." 