#!/usr/bin/env python3
"""
Per-sample MethylGPT test predictions for one fold, for a paired subject-level
bootstrap comparison against MethylLlama V7b.

WHY THIS SCRIPT EXISTS
-----------------------
Official training/evaluation (finetuning_age_main.py) only ever accumulates
predictions in-memory (Age_Model.test_step -> regression_metric) and never
persists per-sample rows. This script reruns inference only (no training, no
gradient updates) against the already-selected best-valid_medae checkpoint
per fold, and writes per-sample (sample_id, true_age, predicted_age) rows.

CRITICAL: SAMPLE-ID RECOVERY
-----------------------------
test.parquet (built by convert_h5ad_to_parquet_21k_kfold.py) has ONLY "data"
and "age" columns -- no sample-ID column. Age_Dataset (finetuning_age_datasets.py)
indexes purely by row position; nothing in the training/eval pipeline has ever
used or needed a real GSM ID. This is exactly the same class of risk already
found on the MethylLlama V7b side (real GSM IDs silently replaced by a
positional index at one point in the pipeline).

To recover true GSM IDs here, this script independently reproduces the exact
boolean-mask row selection convert_h5ad_to_parquet_21k_kfold.py used against
the source h5ad (obs_names_arr = h5ad's own GSM-ID index; mask = isin(test_ids)),
which -- because neither methyl_aligned nor the mask reorders anything -- yields
sample IDs in the identical order as test.parquet's rows. This is verified,
not assumed: the h5ad-recovered ages are compared row-for-row against
test.parquet's actual age column, and the script aborts loudly (raises,
non-zero exit) rather than write output if they don't match exactly.

WHAT THIS SCRIPT DOES NOT DO
------------------------------
- Does not retrain, fine-tune, or modify any checkpoint.
- Does not modify finetuning_age_main.py / finetuning_age_models.py /
  finetuning_age_datasets.py / any existing training script or output.
- Does not touch the BMFM-RNA/methyl repo.

Usage (on cluster, one fold at a time -- see run_extract_kfold_predictions_fold{N}.sbatch):
    source /sci/labs/benjamin.yakir/netanel.azran/venv_torch22/bin/activate
    export PYTHONUNBUFFERED=1
    python extract_kfold_predictions.py --fold 0
"""

import methylgpt.modules.scGPT.scgpt as scgpt  # noqa: F401  (import order matches finetuning_age_main.py exactly)

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).parent.resolve()
FINETUNE_SRC_DIR = (
    SCRIPT_DIR.parent.parent
    / "tutorials_local" / "MethylGPT_tutorials" / "finetuning_age_prediction"
)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(FINETUNE_SRC_DIR))

from finetuning_age_datasets import CollatableVocab, Age_Dataset  # noqa: E402
from finetuning_age_models import methyGPT_Age_Model  # noqa: E402
from convert_h5ad_to_parquet_21k_kfold import load_gsm_ids  # noqa: E402  (reused, not reimplemented)


def log(msg: str) -> None:
    print(msg, flush=True)


def find_best_checkpoint(weights_save_path: Path):
    """
    Pick the checkpoint with the lowest valid_medae, parsed directly from the
    filename -- Lightning's ModelCheckpoint embeds it there via the
    weights_name template in finetuning_age_main.py:
        "..._{epoch:02d}-{step:02d}-{valid_medae:.4f}-{valid_mae:.4f}-{valid_s_r:.4f}.ckpt"
    Lightning auto-expands each "{name:.4f}" placeholder to "name=value" in the
    actual filename (confirmed against real cluster output), e.g.:
        ..._epoch=173-step=1174848-valid_medae=3.3403-valid_mae=5.4938-valid_s_r=0.9368.ckpt
    Prints every candidate found (auditable "why this checkpoint" evidence).
    """
    pattern = re.compile(r"valid_medae=(\d+\.\d{4})-valid_mae=(\d+\.\d{4})-valid_s_r=(\d+\.\d{4})\.ckpt$")
    candidates = []
    for ckpt in sorted(weights_save_path.glob("*.ckpt")):
        m = pattern.search(ckpt.name)
        if not m:
            log(f"  WARNING: could not parse valid_medae from filename, skipping: {ckpt.name}")
            continue
        valid_medae = float(m.group(1))
        candidates.append((valid_medae, ckpt))

    if not candidates:
        raise FileNotFoundError(
            f"No parseable .ckpt files found in {weights_save_path} -- "
            "cannot select a checkpoint."
        )

    candidates.sort(key=lambda t: t[0])
    log(f"  Found {len(candidates)} checkpoint(s) in {weights_save_path}:")
    for medae, ckpt in candidates:
        log(f"    valid_medae={medae:.4f}  {ckpt.name}")
    best_medae, best_ckpt = candidates[0]
    log(f"  -> selected (lowest valid_medae={best_medae:.4f}): {best_ckpt.name}")
    return best_ckpt, best_medae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--args_json", default=str(SCRIPT_DIR / "args.json"))
    parser.add_argument(
        "--train_yml", default=None,
        help="defaults to train_methylgpt_21k_altumage_fold{N}.yml next to this script",
    )
    parser.add_argument(
        "--h5ad",
        default="/sci/labs/benjamin.yakir/netanel.azran/data/"
                "data_methyl_21k_h5ad/altumage_21k_3way.h5ad",
    )
    parser.add_argument(
        "--kfold_splits_dir",
        default="/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/"
                "21k_altumage/kfold_splits",
    )
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR.parent.parent / "outputs" / "bootstrap_predictions" / "methylgpt"),
    )
    args = parser.parse_args()
    fold = args.fold
    train_yml = args.train_yml or str(SCRIPT_DIR / f"train_methylgpt_21k_altumage_fold{fold}.yml")

    log(f"=== MethylGPT fold {fold} -- per-sample prediction extraction ===")
    log(f"args_json  = {args.args_json}")
    log(f"train_yml  = {train_yml}")
    log(f"h5ad       = {args.h5ad}")
    log(f"splits_dir = {args.kfold_splits_dir}")

    with open(args.args_json) as f:
        pretrain_args = json.load(f)
    with open(train_yml) as f:
        add_args = yaml.safe_load(f)
    model_args = {**pretrain_args, **add_args}

    # Exact same CLI-override sequence finetuning_age_main.py applies
    # (there, from argparse defaults --mask_ratio=0, --mask_seed=42):
    model_args["mask_ratio"] = 0.0
    model_args["mask_seed"] = 42
    model_args["dropout"] = 0

    log(f"dataset tag       = {model_args.get('dataset')}")
    log(f"pretrained_file   = {model_args['pretrained_file']}")
    log(f"weights_save_path = {model_args['weights_save_path']}")
    log(f"train_file        = {model_args['train_file']}")
    log(f"test_file         = {model_args['test_file']}")

    # ------------------------------------------------------------
    # 1. Select the best-valid_medae checkpoint (auditable)
    # ------------------------------------------------------------
    log("Selecting checkpoint ...")
    weights_save_path = Path(model_args["weights_save_path"])
    ckpt_path, ckpt_valid_medae = find_best_checkpoint(weights_save_path)

    # ------------------------------------------------------------
    # 2. Rebuild vocab + scaler EXACTLY as training did. The scaler is
    #    part of the trained artifact -- it must be refit on THIS fold's
    #    own train split, not a shared/global one, or inverse_transform
    #    will silently produce the wrong ages.
    # ------------------------------------------------------------
    log("Building vocab ...")
    vocab = CollatableVocab(model_args)

    log(f"Loading train_file to refit age scaler: {model_args['train_file']}")
    train_df = pd.read_parquet(model_args["train_file"])
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df["age"].to_numpy().reshape(-1, 1))
    log(f"  scaler fit on {len(train_df)} training rows, "
        f"age range [{train_df['age'].min():.2f}, {train_df['age'].max():.2f}]")

    log(f"Loading test_file: {model_args['test_file']}")
    test_df = pd.read_parquet(model_args["test_file"])
    log(f"  test rows: {len(test_df)}")

    # ------------------------------------------------------------
    # 3. CRITICAL: recover real GSM sample IDs for test.parquet's row
    #    order (test.parquet itself has no ID column). Verified, not
    #    assumed -- see module docstring.
    # ------------------------------------------------------------
    log(f"Recovering true sample IDs from h5ad obs metadata: {args.h5ad}")
    import anndata
    # Full (non-backed) load, matching convert_h5ad_to_parquet_21k_kfold.py's
    # already-proven-working loading method exactly, rather than introducing
    # an untested backed="r" code path this script can't verify beforehand.
    adata = anndata.read_h5ad(args.h5ad)
    if not adata.obs_names.is_unique:
        raise RuntimeError("adata.obs_names is not unique -- cannot safely recover sample IDs.")
    obs_names_arr = np.asarray(adata.obs_names, dtype=str)
    h5ad_ages = adata.obs["age"].values

    test_ids = load_gsm_ids(Path(args.kfold_splits_dir) / "test_ids.npy")
    log(f"  test_ids.npy: {len(test_ids)} GSM IDs")

    mask = np.isin(obs_names_arr, test_ids)
    if mask.sum() != len(test_ids):
        raise RuntimeError(
            f"Expected {len(test_ids)} test IDs to match h5ad obs_names, matched {mask.sum()} -- "
            "aborting rather than guessing at sample order."
        )
    recovered_sample_ids = obs_names_arr[mask]
    recovered_ages = h5ad_ages[mask]

    if len(recovered_sample_ids) != len(test_df):
        raise RuntimeError(
            f"Row count mismatch: recovered {len(recovered_sample_ids)} IDs from h5ad but "
            f"test.parquet has {len(test_df)} rows -- ID recovery is NOT trustworthy, aborting."
        )

    parquet_ages = test_df["age"].to_numpy().astype(np.float64)
    ages_match = np.isclose(recovered_ages.astype(np.float64), parquet_ages, atol=1e-4, equal_nan=True)
    if not ages_match.all():
        n_mismatch = int((~ages_match).sum())
        raise RuntimeError(
            f"Age mismatch between h5ad-recovered row order and test.parquet's age column "
            f"({n_mismatch}/{len(parquet_ages)} rows differ) -- the ID-to-row mapping is WRONG. "
            "Aborting rather than writing untrustworthy sample IDs. This is exactly the silent "
            "ID-corruption risk already found on the MethylLlama side; do not bypass this check."
        )
    log(f"  VERIFIED: h5ad-derived ages match test.parquet's age column exactly for all "
        f"{len(parquet_ages)} rows (atol=1e-4). Sample-ID mapping is trustworthy.")

    if len(set(recovered_sample_ids)) != len(recovered_sample_ids):
        raise RuntimeError("Recovered sample IDs contain duplicates -- aborting.")

    # ------------------------------------------------------------
    # 4. Build model, load checkpoint, eval mode
    # ------------------------------------------------------------
    log("Building model ...")
    model = methyGPT_Age_Model(model_args=model_args, vocab=vocab, scaler=scaler)
    log(f"Loading checkpoint state_dict: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # 5. Deterministic inference: no masking, no shuffle, no dropout.
    #    mask_ratio==0 forces the target_value branch below, matching
    #    training_step/validation_step/test_step in finetuning_age_models.py
    #    exactly (not an approximation of it).
    # ------------------------------------------------------------
    test_dataset = Age_Dataset(vocab, test_df, scaler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collater,
        num_workers=0,
    )

    # Official training/eval ran under Lightning's precision="bf16-mixed"
    # (see finetuning_age_main.py's pl.Trainer(...) config). Match it here
    # via autocast so this is a faithful reproduction, not an fp32
    # approximation of what actually produced the official numbers.
    use_amp = device.type == "cuda"
    log(f"Autocast bf16-mixed: {use_amp} (matches official precision=\"bf16-mixed\")")

    log(f"Running inference over {len(test_dataset)} test samples ...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            gene_id, masked_value, target_value, ages_label, ages_label_norm = batch
            gene_id = gene_id.to(device)
            target_value = target_value.to(device)

            assert model_args["mask_ratio"] == 0.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred_age_norm = model(gene_id, target_value)
            pred_age_norm = pred_age_norm.view(-1)
            pred_age_np = scaler.inverse_transform(
                pred_age_norm.detach().to(torch.float32).cpu().numpy().reshape(-1, 1)
            )
            all_preds.append(float(pred_age_np[0, 0]))
            all_labels.append(float(ages_label.view(-1)[0].item()))

            if (i + 1) % 500 == 0 or (i + 1) == len(test_dataset):
                log(f"  ... {i + 1}/{len(test_dataset)} samples")

    all_preds = np.asarray(all_preds, dtype=np.float64)
    all_labels = np.asarray(all_labels, dtype=np.float64)

    if np.isnan(all_preds).any():
        n_nan = int(np.isnan(all_preds).sum())
        raise RuntimeError(f"{n_nan} NaN predictions produced -- aborting before writing output.")

    if not np.allclose(all_labels, parquet_ages, atol=1e-4, equal_nan=True):
        raise RuntimeError(
            "Age_Dataset-yielded ages don't match test.parquet's age column -- aborting."
        )

    # ------------------------------------------------------------
    # 6. Recompute metrics locally for the verification.json cross-check
    #    against WandB's official test_top1_* numbers for this fold
    #    (done by validate_kfold_predictions.py, using live WandB data,
    #    not a hardcoded/guessed tolerance).
    # ------------------------------------------------------------
    medae = float(median_absolute_error(all_labels, all_preds))
    mae = float(mean_absolute_error(all_labels, all_preds))
    r2 = float(r2_score(all_labels, all_preds))
    log(f"Recomputed test metrics: MedAE={medae:.4f} MAE={mae:.4f} R2={r2:.4f}")
    log("(cross-checked against this fold's official test_top1_* on WandB by validate_kfold_predictions.py)")

    # ------------------------------------------------------------
    # 7. Write outputs
    # ------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({
        "sample_id": recovered_sample_ids,
        "true_age": all_labels,
        "predicted_age": all_preds,
        "fold": fold,
        "model": "MethylGPT",
        "checkpoint": str(ckpt_path),
    })
    csv_path = output_dir / f"fold_{fold}_predictions.csv"
    out_df.to_csv(csv_path, index=False)
    log(f"Wrote {csv_path} ({len(out_df)} rows)")

    sorted_ids = sorted(recovered_sample_ids.tolist())
    id_hash = hashlib.sha256("\n".join(sorted_ids).encode("utf-8")).hexdigest()

    verification = {
        "fold": fold,
        "checkpoint": str(ckpt_path),
        "checkpoint_valid_medae": ckpt_valid_medae,
        "n_rows": int(len(out_df)),
        "n_unique_sample_ids": int(len(set(recovered_sample_ids))),
        "duplicate_id_count": int(len(recovered_sample_ids) - len(set(recovered_sample_ids))),
        "missing_prediction_count": int(np.isnan(all_preds).sum()),
        "medae": medae,
        "mae": mae,
        "r2": r2,
        "true_age_min": float(all_labels.min()),
        "true_age_max": float(all_labels.max()),
        "sorted_sample_id_sha256": id_hash,
    }
    json_path = output_dir / f"fold_{fold}_verification.json"
    with open(json_path, "w") as f:
        json.dump(verification, f, indent=2)
    log(f"Wrote {json_path}")
    log(f"=== fold {fold} done ===")


if __name__ == "__main__":
    main()
