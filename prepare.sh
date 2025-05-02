#!/bin/bash
# re-validate login information - GitLab only

# Set minimal environment for GitLab only
export OPENAI_API_KEY=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
export DATASET=webarena
export HOST_NAME=http://ec2-52-14-25-77.us-east-2.compute.amazonaws.com
export GITLAB="$HOST_NAME:8023"

# Create auth directory
mkdir -p ./.auth

# Skip auto_login.py which tries to connect to all services
echo "Setting up minimal environment for GitLab only"
echo "Skipping auto_login.py to avoid connection issues with other services"

# If you want to run your GitLab agent now:
echo ""
echo "To run the GitLab agent with text modality:"
echo "$ chmod +x shells/example_react_text_gitlab.sh"
echo "$ ./shells/example_react_text_gitlab.sh"