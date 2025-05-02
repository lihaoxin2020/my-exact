##### VLM providers
export OPENAI_API_KEY=""
# export OPENAI_ORGANIZATION=org-xxx
export HF_TOKEN=hf_xxx
# optional keys (other providers)
# export AZURE_OPENAI_API_BASE=https://xxx
# export AZURE_TOKEN_PROVIDER_API_BASE=https://xxx
HOST_NAME=http://ec2-3-12-126-100.us-east-2.compute.amazonaws.com

##### (V)WA Web URLs
export DATASET=webarena
export CLASSIFIEDS="$HOST_NAME:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
export SHOPPING="$HOST_NAME:7770"
export REDDIT="$HOST_NAME:9999"
export WIKIPEDIA="$HOST_NAME:8888"
export SHOPPING_ADMIN="$HOST_NAME:7780/admin"
export GITLAB="$HOST_NAME:8023"
export MAP="$HOST_NAME:3000"
export HOMEPAGE="$HOST_NAME:4399"

# generate WA task configs
python runners/utils/generate_test_configs.py
