#!/usr/bin/env python3
"""
Convert 19k h5ad fine-tuning dataset to MethylGPT parquet format.

Aligns 19,608 CpG sites to the 49,156 MethylGPT reference.
CpGs not present in the 19k panel are filled with NaN (→ pad token at runtime).
Splits are taken from adata.obs["split"] (train / valid / test).

Usage (on cluster):
    python convert_h5ad_to_parquet_19k.py
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


def convert(h5ad_path: str, probe_id_csv: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load h5ad ───────────────────────────────────────────────────────────
    print(f"Loading {h5ad_path} ...")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}  splits: {adata.obs['split'].value_counts().to_dict()}")

    # ── Load 49k CpG reference ───────────────────────────────────────────────
    cpg_ref = pd.read_csv(probe_id_csv)["illumina_probe_id"].tolist()
    print(f"  Reference CpGs: {len(cpg_ref):,}")

    # ── Build methylation DataFrame (samples × 19k CpGs) ────────────────────
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)

    methyl_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # Align to 49k reference — missing CpGs become NaN
    methyl_aligned = methyl_df.reindex(columns=cpg_ref)   # shape [n_samples, 49156]
    n_present = methyl_df.columns.isin(cpg_ref).sum()
    print(f"  CpGs present in reference: {n_present:,} / {len(adata.var_names):,}")

    ages = adata.obs["age"].values

    # ── Save each split ──────────────────────────────────────────────────────
    for split in ["train", "valid", "test"]:
        mask = adata.obs["split"] == split
        n = mask.sum()
        if n == 0:
            print(f"  WARNING: no samples for split='{split}', skipping.")
            continue

        rows = methyl_aligned.loc[mask].values          # [n, 49156]
        split_ages = ages[mask.values]

        df = pd.DataFrame({
            "data": [row.tolist() for row in rows],
            "age":  split_ages.tolist(),
        })

        out_path = output_dir / f"{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {split}: {n:,} samples → {out_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad",
        default="/sci/labs/benjamin.yakir/netanel.azran/data/"
                "data_methyl_finetune_19k_h5ad/"
                "finetuning_19608_clean_stratified_no_outliers.h5ad",
    )
    parser.add_argument(
        "--probe_id_csv",
        default="/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT/"
                "cpg_ids/probe_ids_type3.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/"
                "19k_data/finetuning_data",
    )
    args = parser.parse_args()
    convert(args.h5ad, args.probe_id_csv, args.output_dir)
