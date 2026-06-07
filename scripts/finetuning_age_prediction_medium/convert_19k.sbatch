#!/bin/bash -l
#SBATCH --job-name=convert-19k-parquet
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

REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis
python $REPO/scripts/finetuning_age_prediction_medium/convert_h5ad_to_parquet_19k.py
