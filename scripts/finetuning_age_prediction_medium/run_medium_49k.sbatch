#!/bin/bash -l
#SBATCH --job-name=methylgpt-medium-49k
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium/logs/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium/logs/%x_%j.err

set -euo pipefail

source /sci/labs/benjamin.yakir/netanel.azran/venv_torch22/bin/activate
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium
python finetuning_age_main.py
