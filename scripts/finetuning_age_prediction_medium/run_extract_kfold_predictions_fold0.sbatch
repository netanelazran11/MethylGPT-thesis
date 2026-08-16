#!/bin/bash -l
#SBATCH --job-name=methylgpt-extract-predictions-fold0
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium/logs/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium/logs/%x_%j.err

set -euo pipefail

source /sci/labs/benjamin.yakir/netanel.azran/venv_torch22/bin/activate
export PYTHONUNBUFFERED=1
REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis
SCRIPT_DIR=$REPO/scripts/finetuning_age_prediction_medium

cd $SCRIPT_DIR
python -u extract_kfold_predictions.py --fold 0
