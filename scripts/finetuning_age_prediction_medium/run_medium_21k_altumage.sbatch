#!/bin/bash -l
#SBATCH --job-name=methylgpt-medium-21k-altumage
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium/logs/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium/logs/%x_%j.err

set -euo pipefail

source /sci/labs/benjamin.yakir/netanel.azran/venv_torch22/bin/activate
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis
SCRIPT_DIR=$REPO/scripts/finetuning_age_prediction_medium
MAIN=$REPO/tutorials_local/MethylGPT_tutorials/finetuning_age_prediction/finetuning_age_main.py

cd $SCRIPT_DIR
python $MAIN \
    --args_json $SCRIPT_DIR/args.json \
    --train_yml $SCRIPT_DIR/train_methylgpt_21k_altumage.yml
