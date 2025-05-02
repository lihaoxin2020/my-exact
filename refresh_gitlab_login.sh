#!/bin/bash
export PYTHONPATH=$(pwd)
export DATASET=webarena

# Setup environment variables
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

# Make sure auth directory exists
mkdir -p ./.auth

# Delete existing GitLab cookies
rm -f ./.auth/gitlab*

# Run auto login script only for GitLab
echo "Creating new GitLab login..."
python visualwebarena/browser_env/auto_login.py --site_list gitlab --auth_folder ./.auth

echo "GitLab login refreshed. Now you can run the parallel script." 