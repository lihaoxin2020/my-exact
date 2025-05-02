#!/bin/bash
# re-validate login information with webarena dataset

export DATASET=webarena
export HOST_NAME=http://ec2-52-14-25-77.us-east-2.compute.amazonaws.com
export REDDIT=$HOST_NAME:9999
export SHOPPING=$HOST_NAME:7770
export SHOPPING_ADMIN=$HOST_NAME:7780/admin
export GITLAB=$HOST_NAME:8023
export WIKIPEDIA=$HOST_NAME:8888
export MAP=$HOST_NAME:3000
export HOMEPAGE=$HOST_NAME:4399

# These are not required for webarena but keeping them for compatibility
export CLASSIFIEDS=$HOST_NAME:9980
export CLASSIFIEDS_RESET_TOKEN=4b61655535e7ed388f0d40a93600254c

# Make auth directory if it doesn't exist
mkdir -p ./.auth

# Run auto login
python auto_login.py 