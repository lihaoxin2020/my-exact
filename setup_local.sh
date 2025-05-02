#!/bin/bash
# Local setup for testing and finetuning without remote connections

echo "=== Setting up local environment for ExACT ==="

# Set up environment variables for local testing
export DATASET=webarena
export HOST_NAME="localhost"
export REDDIT="$HOST_NAME:9999"
export SHOPPING="$HOST_NAME:7770"
export SHOPPING_ADMIN="$HOST_NAME:7780/admin"
export GITLAB="$HOST_NAME:8023"
export WIKIPEDIA="$HOST_NAME:8888"
export MAP="$HOST_NAME:3000"
export HOMEPAGE="$HOST_NAME:4399"
export CLASSIFIEDS="$HOST_NAME:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

# Create necessary directories
mkdir -p ./.auth
mkdir -p data/training
mkdir -p data/webarena/eval_results

echo "=== Local setup complete ==="
echo ""
echo "Since your EC2 instance connection failed, you have these options:"
echo ""
echo "1. Generate finetuning data from sample trajectories:"
echo "   $ chmod +x shells/example_local_finetune.sh"
echo "   $ ./shells/example_local_finetune.sh"
echo ""
echo "2. Use existing data if you have it:"
echo "   $ python runners/train/tree_to_data.py --env_name gitlab --result_dir path/to/your/data --output_dir data/training"
echo ""
echo "3. If you need to test connectivity to your EC2 instance:"
echo "   $ ping ec2-52-14-25-77.us-east-2.compute.amazonaws.com"
echo "   $ telnet ec2-52-14-25-77.us-east-2.compute.amazonaws.com 7770"
echo ""
echo "4. Check if you need to update security groups in AWS to allow connections" 