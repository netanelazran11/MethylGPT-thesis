import os
from pathlib import Path
import methylgpt.modules.scGPT.scgpt as scgpt
current_directory = Path(__file__).parent.absolute()
from sklearn import preprocessing
import pandas as pd
import argparse
import json
import yaml
import torch
import lightning as pl

# Lightning import paths / strategy APIs can vary across 2.x installations.
try:
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.strategies import DDPStrategy
except Exception:  # pragma: no cover
    # Fallback for environments that still expose PyTorch Lightning directly
    from pytorch_lightning.loggers import WandbLogger  # type: ignore
    from pytorch_lightning import seed_everything  # type: ignore
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor  # type: ignore
    from pytorch_lightning.strategies import DDPStrategy  # type: ignore

from finetuning_age_datasets import CollatableVocab, Age_Dataset
from finetuning_age_models import methyGPT_Age_Model


def resolve_path(p: str, root: Path) -> str:
    """
    Resolve relative paths against repo root; keep absolute paths as-is.
    Returns a string path.
    """
    if p is None:
        return p
    p = str(p).strip()
    if p == "":
        return p
    pp = Path(p)
    return str(pp if pp.is_absolute() else (root / pp).resolve())

seed_everything(42, workers=True)


def train(args):

    # ------------------------------------------------------------
    # Resolve repo root robustly (walk up until we find a repo marker)
    # ------------------------------------------------------------
    def find_repo_root(start: Path) -> Path:
        for p in [start] + list(start.parents):
            if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "setup.cfg").exists():
                return p
        # Fallback: keep previous behavior if no marker exists
        return start.parents[4] if len(start.parents) >= 5 else start

    REPO_ROOT = find_repo_root(Path(__file__).resolve())

    # ------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------
    with open(resolve_path(args.args_json, REPO_ROOT), "r") as f:
        pretrain_args = json.load(f)

    with open(resolve_path(args.train_yml, REPO_ROOT), "r") as f:
        add_args = yaml.safe_load(f)

    model_args = {**pretrain_args, **add_args}

    # ------------------------------------------------------------
    # Resolve important paths to absolute (relative-safe)
    # ------------------------------------------------------------
    for k in [
        "pretrained_file",
        "weights_save_path",
        "train_file",
        "valid_file",
        "test_file",
        "valid_ckpt_path",
        "probe_id_dir",
        "parquet_dir",
    ]:
        if k in model_args:
            model_args[k] = resolve_path(model_args[k], REPO_ROOT)

    # Ensure weights directory exists
    if model_args.get("weights_save_path"):
        Path(model_args["weights_save_path"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Build run name (supports run_tag from YAML)
    # ------------------------------------------------------------
    dataset_name = model_args.get("dataset", "unknown_dataset")
    mask_ratio = model_args.get("mask_ratio", 0)

    run_tag = model_args.get("run_tag", "")
    tag_suffix = f"-{run_tag}" if run_tag else ""

    model_args["version"] = (
        f"Finetune-methylGPT-AltumAgeMLMPrediction"
        f"-mask{mask_ratio}"
        f"-dataset-{dataset_name}"
        f"{tag_suffix}"
    )

    model_args["weights_name"] = model_args["version"] + (
        "_{epoch:02d}-{step:02d}"
        "-{valid_medae:.4f}-{valid_mae:.4f}-{valid_s_r:.4f}"
        "-{test_medae:.4f}-{test_mae:.4f}-{test_s_r:.4f}"
    )

    # ------------------------------------------------------------
    # CLI overrides (kept minimal, like your original)
    # ------------------------------------------------------------
    model_args["mask_ratio"] = args.mask_ratio * 0.01
    model_args["mask_seed"] = args.mask_seed
    model_args["dropout"] = 0


    # Prepare data
    methyGPT_vocab = CollatableVocab(model_args)

    train_file = model_args["train_file"]
    valid_file = model_args["valid_file"]
    test_file = model_args["test_file"]
    train_df = pd.read_parquet(train_file)
    valid_df = pd.read_parquet(valid_file)
    test_df = pd.read_parquet(test_file)


    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df["age"].to_numpy().reshape(-1, 1))
    train_dataset = Age_Dataset(methyGPT_vocab, train_df, scaler)
    valid_dataset = Age_Dataset(methyGPT_vocab, valid_df, scaler)
    test_dataset = Age_Dataset(methyGPT_vocab, test_df, scaler)

    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_args["train_batch_size"],
        collate_fn=train_dataset.collater,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    valid_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collater,
        batch_size=model_args["valid_batch_size"],
        num_workers=4,
    )

    test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=test_dataset.collater,
        batch_size=model_args["valid_batch_size"],
        num_workers=4,
    )


    # Init model
    model = methyGPT_Age_Model(
                model_args=model_args,
                vocab=methyGPT_vocab,
                scaler=scaler,
                )


    if model_args["mode"] == "train":

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_args["weights_save_path"],
            filename=model_args["weights_name"],
            monitor="valid_medae",
            mode="min",
            save_top_k=1,
        )

        lr_logger = LearningRateMonitor()

        if model_args["wandb"]:
            wandb_save_path = os.path.join(str(current_directory) + "/wandb",  model_args["version"])
            os.makedirs(wandb_save_path, exist_ok=True)
            wandb_logger = WandbLogger(project=model_args["project"],
                                    name=model_args["version"],
                                    save_dir=wandb_save_path,
                                    )
        else:
            wandb_logger = None

        # train model
        trainer = pl.Trainer(
            default_root_dir=current_directory,
            logger=wandb_logger,
            devices=model_args["gpus"],
            accelerator="gpu",
            callbacks=[lr_logger, checkpoint_callback],
            gradient_clip_val=model_args["gradient_clip_val"],
            max_epochs=model_args["max_epochs"],
            strategy=DDPStrategy(find_unused_parameters=True),
            log_every_n_steps=model_args["log_every_n_steps"],
            precision="bf16-mixed",
        )

        trainer.fit(model, train_loader, [valid_loader, test_loader])

    elif model_args["mode"] == "valid":
        model.load_state_dict(torch.load(model_args["valid_ckpt_path"], map_location="cpu")['state_dict'], strict=True)
        model.eval()
        # validate model
        trainer = pl.Trainer(
            default_root_dir=current_directory,
            devices=1,
            accelerator="gpu",
            strategy="auto",
            precision="bf16-mixed",
        )

        trainer.validate(model, [valid_loader, test_loader])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required config inputs
    parser.add_argument(
        "--args_json",
        type=str,
        required=True,
        help="Path to the pretrain args JSON (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--train_yml",
        type=str,
        required=True,
        help="Path to the fine-tuning YAML (relative to repo root or absolute).",
    )

    # Optional overrides (kept as in your original)
    parser.add_argument("--mask_ratio", type=float, default=0, help="Mask ratio in percent (e.g., 15 for 15%).")
    parser.add_argument("--mask_seed", type=int, default=42, help="Seed for masking.")

    args = parser.parse_args()
    train(args)
