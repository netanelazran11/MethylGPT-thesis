#!/bin/bash -l
#SBATCH --job-name=convert-21k-kfold-parquet
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

# NOTE: run this once, before submitting the 5 fold-specific training jobs.
# It builds train_fold{0..4}.parquet, valid_fold{0..4}.parquet, and the single
# shared test.parquet in one h5ad load, from GSM-ID lists synced from the
# methyl repo's outputs/kfold_splits/ (see --kfold_splits_dir default in the
# script). Those .npy files must already be present at that path before this
# job runs — this script does not sync/generate them.
python $REPO/scripts/finetuning_age_prediction_medium/convert_h5ad_to_parquet_21k_kfold.py
