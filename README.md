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
   git clone https://github.com/lihaoxin2020/my-exact.git
   cd my-exact
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
   # generate WA task configs
   export DATASET=webarena
   python runners/utils/generate_test_configs.py
   ```

3. Set up and launch the (V)WA Management server by following the instructions in the [ORIGINAL_README.md](ORIGINAL_README.md).

## Running Experiments

For instructions on how to run experiments, including quickstart examples and how to parallelize evaluations, please refer to the [ORIGINAL_README.md](ORIGINAL_README.md). 



### Collecting Trajectories
To collect trajectories from o4-mini, Llama 3.1-8B, and other models:

```bash
# For o4-mini trajectories
# For Llama 3.1-8B trajectories
python run_react_agent.py \
  --model="meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --site="shopping" \
  --task_id="02d5b793" \
  --max_steps=30 \
  --output_dir="./trajectories/llama31-8b/"
```

To extract and process WebArena trajectories:
```bash
python extract_webarena_trajectories.py \
  --input_dir="./trajectories/raw/" \
  --output_dir="./trajectories/processed/"
```

### Analyzing Trajectories
To visualize and analyze trajectories:
```bash
python visualize_trajectories.py \
  --trajectory_dir="./trajectories/" \
  --models="o4-mini,llama31-8b,ft-llama31" \
  --output_dir="./analysis/visualizations/"

python analyze_trajectories_comparison.py \
  --trajectory_dir="./trajectories/" \
  --models="o4-mini,llama31-8b,ft-llama31" \
  --output_file="./analysis/comparison_results.json"

python analyze_tokens.py \
  --trajectory_dir="./trajectories/" \
  --models="o4-mini,llama31-8b" \
  --output_file="./analysis/token_usage.json"
```

To collect and match performance data with trajectory files:
```bash
python collect_performance_trajectory_files.py \
  --performance_dir="./results/performance/" \
  --trajectory_dir="./trajectories/" \
  --output_dir="./analysis/matched/"
```

To filter trajectories based on specific criteria:
```bash
python filter_performances.py \
  --input_file="./results/performance/all_performances.json" \
  --output_file="./results/performance/filtered_performances.json" \
  --filter_criteria="success=True"

python filter_llama3_trajectories.py \
  --input_dir="./trajectories/llama31-8b/" \
  --output_dir="./trajectories/llama31-8b-filtered/" \
  --filter_criteria="max_steps=25"
```

### Creating Training Datasets
The SFT training dataset was created from 338 successful o4-mini trajectories (with 38 validation examples):

```bash
python ft_llama3.py create_sft_data \
  --input_dir="./trajectories/o4-mini/" \
  --output_dir="./training_data/sft/" \
  --train_ratio=0.9 \
  --filter_criteria="success=True"
```

This creates the following files:
- `./training_data/sft/train.json` - 338 examples
- `./training_data/sft/valid.json` - 38 examples

The preference data for DPO was created by pairing o4-mini outputs (preferred) with SFT model outputs (rejected):

```bash
python ft_llama3.py create_preference_data \
  --chosen_trajectory_dir="./trajectories/o4-mini/" \
  --rejected_trajectory_dir="./trajectories/sft-model/" \
  --output_dir="./training_data/dpo/" \
  --filter_criteria="paired=True"
```

### Training Models
SFT training was performed for 10 epochs with batch size 8 and learning rate 0.00002 using a linear scheduler:

```bash
python run_full_sft.py \
  --model_name="meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --train_file="./training_data/sft/train.json" \
  --valid_file="./training_data/sft/valid.json" \
  --output_dir="./models/sft-llama31-8b" \
  --epochs=10 \
  --batch_size=8 \
  --learning_rate=0.00002 \
  --scheduler="linear" \
  --together_api_key="your_together_api_key"
```

DPO training was performed with similar parameters:

```bash
python run_full_dpo.py \
  --model_name="./models/sft-llama31-8b" \
  --train_file="./training_data/dpo/train.json" \
  --valid_file="./training_data/dpo/valid.json" \
  --output_dir="./models/dpo-llama31-8b" \
  --epochs=10 \
  --batch_size=8 \
  --learning_rate=0.00002 \
  --scheduler="linear" \
  --together_api_key="your_together_api_key" \
  --beta=0.1
```

To fix DPO data format issues when needed:
```bash
python fix_dpo_format.py \
  --input_file="./training_data/dpo/train_raw.json" \
  --output_file="./training_data/dpo/train.json"
```

Running python run_full_dpo.py or python run_full_sft.py will set up TogetherAI to run the model for you. 

From this, you will need to set up your TogetherAI account and API key to run the training jobs. You can then deploy the model on TogetherAI and then create a script for training and evaluation. 


