# Improving LLM Web Navigation with Direct Preference Optimization Over Test-Time Scaling Responses

This codebase is built on the ExACT project, which presents R-MCTS and Exploratory Learning for building o1-like models for agentic applications. For more details about the project, refer to [ORIGINAL_README.md](ORIGINAL_README.md).

See the project report [here](<CPSC 577 Report.pdf>) and checklist [here](<Reproducibility checklis - Google Docs.pdf>). 

## Installation and Setup

### 1. WebArena Environment Setup

First, set up the WebArena environment by following the official repository instructions:

1. Clone the WebArena repository:
   ```bash
   git clone https://github.com/web-arena-x/webarena.git
   cd webarena
   
   # create a python env with conda or venv where python=3.10
   pip install -r requirements.txt
   playwright install
   pip install -e .
   ```

2. Follow the official WebArena setup guide to set up the website environments using either AWS or Docker.
   - For Docker setup, follow the instructions in the WebArena repository's README to set up the local Docker environment.
   - Make sure all the websites (Classifieds, Shopping, Reddit, etc.) are running properly.

### 2. Project Setup

Once WebArena is set up, proceed with setting up the ExACT project:

1. Clone the ExACT repository:
   ```bash
   git clone https://github.com/microsoft/ExACT.git
   cd ExACT
   ```

2. Install the dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

   This will install all required packages specifically for this project. 

### 3. Configuration and Running

After installing the dependencies, follow the remaining setup instructions from the original documentation:

1. Export the necessary environment variables:
   ```bash
   ##### VLM providers
   export OPENAI_API_KEY=sk-xxx
   export OPENAI_ORGANIZATION=org-xxx
   export HF_TOKEN=hf_xxx
   # optional keys (other providers)
   export AZURE_OPENAI_API_BASE=https://xxx
   export AZURE_TOKEN_PROVIDER_API_BASE=https://xxx

   ##### (V)WA Web URLs
   export DATASET="<visualwebarena or webarena>"
   export CLASSIFIEDS="<your_classifieds_domain>:9980"
   export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
   export SHOPPING="<your_shopping_site_domain>:7770"
   export REDDIT="<your_reddit_domain>:9999"
   export WIKIPEDIA="<your_wikipedia_domain>:8888"
   export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
   export GITLAB="<your_gitlab_domain>:8023"
   export MAP="<your_map_domain>:3000"
   export HOMEPAGE="<your_homepage_domain>:4399"
   ```

2. Generate the task configurations:
   ```bash
   # generate VWA task configs
   export DATASET=visualwebarena
   python runners/utils/generate_test_configs.py
   # generate WA task configs
   export DATASET=webarena
   python runners/utils/generate_test_configs.py
   ```

3. Set up and launch the (V)WA Management server by following the instructions in the [ORIGINAL_README.md](ORIGINAL_README.md).

## Running Experiments

For instructions on how to run experiments, including quickstart examples and how to parallelize evaluations, please refer to the [ORIGINAL_README.md](ORIGINAL_README.md).
