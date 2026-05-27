#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W&B Group Analysis for MethylGPT fine-tuning runs.

Goal
----
Pull ALL runs for a given W&B group, extract configs + summary metrics,
optionally download history to compute "best validation" after warmup,
then:
  - produce clean tables (per-run + aggregated)
  - print the best run and its hyperparameters
  - export CSV/JSON artifacts
  - generate a few diagnostic plots

Usage (examples)
--------------
# Basic (uses env defaults):
python wandb_group_analysis_ft8k.py

# Override group + output directory:
WANDB_GROUP="finetune-methylGPT-8k-cpgs-Baseline" OUTDIR=./wandb_reports python wandb_group_analysis_ft8k.py

# Choose which metric defines "best":
BEST_METRIC=test_mae python wandb_group_analysis_ft8k.py

Notes
-----
- Requires `wandb login` already configured (WANDB_API_KEY).
- Designed to be robust to missing keys and older runs.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb


# ==============================
# CONFIG (env overrides)
# ==============================
ENTITY = os.getenv("WANDB_ENTITY", "netanelazran11-hebrew-university-of-jerusalem")
PROJECT = os.getenv("WANDB_PROJECT", "methylGPT_8K_ElasticNet_Features")
# User asked explicitly:
GROUP = os.getenv("WANDB_GROUP", "finetune-methylGPT-8k-cpgs-Baseline")

DEFAULT_OUTDIR = os.path.join("~", "Projects", "MethylGPT-thesis", "wandb_group_report")
OUTDIR = os.path.abspath(os.path.expanduser(os.getenv("OUTDIR", DEFAULT_OUTDIR)))

WANDB_TIMEOUT = int(os.getenv("WANDB_TIMEOUT", "600"))
# Which metric decides "best run" (prefer test metrics when available)
BEST_METRIC = os.getenv("BEST_METRIC", "test_mae")
# Tie-breaker (lower is better)
TIEBREAK_METRIC = os.getenv("TIEBREAK_METRIC", "valid_mae")

# If 0 => no cap; else cap rows per run history download
MAX_HISTORY_ROWS = int(os.getenv("MAX_HISTORY_ROWS", "0"))
# Drop first WARMUP_FRAC of epochs when searching "best validation"
WARMUP_FRAC = float(os.getenv("WARMUP_FRAC", "0.20"))
# Rolling window for smoothing "best validation" selection
BEST_WINDOW = int(os.getenv("BEST_WINDOW", "5"))
# Filter states: comma-separated, e.g. "finished,running"
ALLOWED_STATES = tuple(s.strip() for s in os.getenv("ALLOWED_STATES", "finished").split(",") if s.strip())

# ------------------------------
# Metric naming (matches typical keys)
# ------------------------------
VALID_MAE_KEY = "valid_mae"
VALID_MEDAE_KEY = "valid_medae"
VALID_RMSE_KEY = "valid_rmse"
VALID_R2_KEY = "valid_r2"
VALID_SR_KEY = "valid_s_r"

VALID_MAE_FALLBACK = "valid_mae_loss/dataloader_idx_0"

TRAIN_MAE_STEP_KEY = "train_mae_loss_step"
TRAIN_MAE_EPOCH_KEY = "train_mae_loss_epoch"

X_KEYS = ["epoch", "trainer/epoch", "trainer/global_step", "_step"]

TEST_METRICS = ["test_mae", "test_medae", "test_rmse", "test_r2", "test_s_r"]

# Parse seed from run.name like "__seed=40__" or "seed=40" or "_seed_40"
SEED_REGEX = re.compile(r"(?:^|.*?)(?:__|^|[^a-zA-Z])seed(?:=|_)(\d+)(?:__|[^0-9]|$)", re.IGNORECASE)

# Common hyperparam keys seen across your runs
HP_KEYS = [
    "pretrained_lr",
    "head_lr",
    "train_batch_size",
    "valid_batch_size",
    "eval_batch_size",
    "batch_size",  # may exist as a separate (sometimes misleading) config field
    "seed",
    "split_id",
    "mask_ratio",
    "mask_seed",
    "weight_decay",
    "dropout",
    "layer_size",
    "nlayers",
    "nhead",
    "d_model",
    "input_type",
]


# ==============================
# Helpers
# ==============================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)
        # string numbers
        if isinstance(x, str) and x.strip() != "":
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
    except Exception:
        return None
    return None


def first_present(d: Dict[str, Any], keys: List[str]) -> Tuple[Optional[str], Any]:
    for k in keys:
        if k in d:
            return k, d.get(k)
    return None, None


def parse_seed_from_name(name: str) -> Optional[int]:
    m = SEED_REGEX.match(name or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def pick_valid_mae_key(columns: Iterable[str]) -> Optional[str]:
    cols = set(columns)
    if VALID_MAE_KEY in cols:
        return VALID_MAE_KEY
    if VALID_MAE_FALLBACK in cols:
        return VALID_MAE_FALLBACK
    # sometimes logged with dataloader_idx_0 variations
    for c in cols:
        if "valid" in c and "mae" in c and "dataloader" in c:
            return c
    return None


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


@dataclass
class RunRecord:
    run_id: str
    name: str
    state: str
    url: str
    created_at: Optional[str]

    # hyperparams (best effort)
    config: Dict[str, Any]

    # summary metrics
    summary: Dict[str, Any]

    # derived
    seed: Optional[int]

    # validation-best (from history)
    best_valid_mae: Optional[float] = None
    best_valid_epoch: Optional[float] = None


# ==============================
# Core W&B pulling
# ==============================

def fetch_runs(api: wandb.Api, entity: str, project: str, group: str) -> List[wandb.apis.public.Run]:
    path = f"{entity}/{project}"
    # W&B filter uses Mongo-like operators
    filters = {"group": group}
    runs = api.runs(path=path, filters=filters)
    return list(runs)


def run_to_record(r: wandb.apis.public.Run) -> RunRecord:
    cfg = dict(r.config or {})
    summ = dict(r.summary or {})

    seed_cfg = cfg.get("seed")
    seed_name = parse_seed_from_name(r.name or "")
    seed = None
    if seed_cfg is not None:
        try:
            seed = int(seed_cfg)
        except Exception:
            seed = seed_name
    else:
        seed = seed_name

    return RunRecord(
        run_id=r.id,
        name=r.name or "",
        state=r.state or "",
        url=getattr(r, "url", "") or "",
        created_at=str(getattr(r, "created_at", "")) if getattr(r, "created_at", None) else None,
        config=cfg,
        summary=summ,
        seed=seed,
    )


def download_history_for_best_valid(
    r: wandb.apis.public.Run,
    warmup_frac: float,
    best_window: int,
    max_rows: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Returns (best_valid_mae, best_epoch).

    Strategy:
    - download history with at least epoch + a valid MAE column (if exists)
    - drop first warmup_frac of epochs
    - smooth with rolling mean (window=best_window)
    - pick min
    """

    # Download a decent set of keys; W&B will ignore missing ones
    keys = list(dict.fromkeys(["epoch", "trainer/epoch", VALID_MAE_KEY, VALID_MAE_FALLBACK]))

    try:
        hist = r.history(keys=keys, pandas=True)
    except Exception:
        # fallback: no keys filter
        hist = r.history(pandas=True)

    if hist is None or len(hist) == 0:
        return None, None

    if max_rows and len(hist) > max_rows:
        hist = hist.tail(max_rows).copy()

    # Identify x (epoch)
    x_key, _ = first_present(hist, ["epoch", "trainer/epoch"])
    if x_key is None:
        # last resort: use _step
        x_key, _ = first_present(hist, ["_step", "trainer/global_step"])
    if x_key is None:
        return None, None

    vkey = pick_valid_mae_key(hist.columns)
    if vkey is None:
        return None, None

    df = hist[[x_key, vkey]].dropna().copy()
    if len(df) == 0:
        return None, None

    # ensure numeric
    df[x_key] = pd.to_numeric(df[x_key], errors="coerce")
    df[vkey] = pd.to_numeric(df[vkey], errors="coerce")
    df = df.dropna()
    if len(df) == 0:
        return None, None

    # If epoch exists, warmup by fraction of max epoch
    if "epoch" in x_key:
        max_epoch = df[x_key].max()
        cutoff = warmup_frac * max_epoch
        df = df[df[x_key] >= cutoff]
        if len(df) == 0:
            return None, None

    # smooth + pick min
    df = df.sort_values(x_key)
    df["v_smooth"] = rolling_min(df[vkey], window=best_window)
    idx = df["v_smooth"].idxmin()
    best_val = float(df.loc[idx, "v_smooth"])
    best_ep = float(df.loc[idx, x_key])
    return best_val, best_ep


# ==============================
# Tabular outputs
# ==============================

def build_runs_table(records: List[RunRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for rec in records:
        c = rec.config
        s = rec.summary

        row: Dict[str, Any] = {
            "run_id": rec.run_id,
            "run": rec.name,
            "state": rec.state,
            "url": rec.url,
            "created_at": rec.created_at,
            "seed": rec.seed,
            "best_valid_mae": rec.best_valid_mae,
            "best_valid_epoch": rec.best_valid_epoch,
        }

        # hyperparams (best effort)
        for k in HP_KEYS:
            if k in c:
                row[k] = c.get(k)

        # test metrics (summary)
        for mk in TEST_METRICS:
            if mk in s:
                row[mk] = s.get(mk)

        # Also add commonly used val metrics from summary (if exist)
        for vk in [VALID_MAE_KEY, VALID_MEDAE_KEY, VALID_RMSE_KEY, VALID_R2_KEY, VALID_SR_KEY, VALID_MAE_FALLBACK]:
            if vk in s and vk not in row:
                row[vk] = s.get(vk)

        # Learning rate groups (if logged)
        for k in ["lr-Adam/pg1", "lr-Adam/pg2"]:
            if k in s:
                row[k] = s.get(k)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Coerce numeric where relevant
    numeric_cols = [
        "pretrained_lr",
        "head_lr",
        "train_batch_size",
        "valid_batch_size",
        "eval_batch_size",
        "batch_size",
        "seed",
        "split_id",
        "mask_ratio",
        "mask_seed",
        "weight_decay",
        "dropout",
        "best_valid_mae",
        "best_valid_epoch",
    ] + TEST_METRICS + [VALID_MAE_KEY, VALID_MAE_FALLBACK, "lr-Adam/pg1", "lr-Adam/pg2"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def best_run(df: pd.DataFrame, primary: str, tiebreak: str) -> pd.Series:
    """Return the best row (lower-is-better for mae/medae/rmse, higher-is-better for r2/spearman).

    Rules:
    - If primary metric exists, rank by it.
    - If not, fallback to tiebreak.
    - Determine direction by metric name.
    """

    def direction(metric: str) -> str:
        m = metric.lower()
        if any(x in m for x in ["mae", "medae", "rmse", "mse", "loss"]):
            return "min"
        if any(x in m for x in ["r2", "spearman", "pearson", "corr"]):
            return "max"
        # safe default
        return "min"

    if primary not in df.columns or df[primary].dropna().empty:
        metric = tiebreak
    else:
        metric = primary

    d = direction(metric)
    tmp = df.copy()
    tmp = tmp[tmp[metric].notna()].copy()
    if tmp.empty:
        # last resort
        return df.iloc[0]

    # Tie-break with tiebreak if available
    if tiebreak in tmp.columns:
        if direction(tiebreak) == "min":
            tmp = tmp.sort_values([metric, tiebreak], ascending=[d == "min", True])
        else:
            tmp = tmp.sort_values([metric, tiebreak], ascending=[d == "min", False])
    else:
        tmp = tmp.sort_values(metric, ascending=(d == "min"))

    return tmp.iloc[0]


def aggregate_by_hparams(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds/splits for each (pretrained_lr, head_lr) combination."""

    keys = [k for k in ["pretrained_lr", "head_lr"] if k in df.columns]
    if not keys:
        return pd.DataFrame()

    # choose the main score to aggregate
    score = BEST_METRIC if BEST_METRIC in df.columns else ("test_mae" if "test_mae" in df.columns else None)

    agg_cols = {
        "run_id": "count",
    }

    if score:
        agg_cols[score] = ["mean", "std", "min", "max"]

    # Add additional metrics if present
    for m in ["test_mae", "test_medae", "test_r2", "test_s_r", "best_valid_mae"]:
        if m in df.columns and m != score:
            agg_cols[m] = ["mean", "std"]

    g = df.groupby(keys, dropna=False).agg(agg_cols)

    # flatten columns
    g.columns = ["_".join([c for c in col if c]) for col in g.columns.to_flat_index()]
    g = g.rename(columns={"run_id_count": "n_runs"})
    g = g.reset_index()

    # sort by best score
    if score and f"{score}_mean" in g.columns:
        g = g.sort_values(f"{score}_mean", ascending=True)

    return g


# ==============================
# Plots
# ==============================

def plot_test_mae_sorted(df: pd.DataFrame, outdir: str) -> Optional[str]:
    if "test_mae" not in df.columns or df["test_mae"].dropna().empty:
        return None

    d = df.copy()
    d = d[d["test_mae"].notna()].copy()
    d = d.sort_values("test_mae", ascending=True)

    # label per run: show key hparams if present
    def label_row(r: pd.Series) -> str:
        plr = r.get("pretrained_lr")
        hlr = r.get("head_lr")
        seed = r.get("seed")
        sp = r.get("split_id")
        parts = []
        if not pd.isna(plr):
            parts.append(f"plr={plr:.0e}")
        if not pd.isna(hlr):
            parts.append(f"hlr={hlr:.0e}")
        if not pd.isna(seed):
            parts.append(f"seed={int(seed)}")
        if not pd.isna(sp):
            parts.append(f"split={int(sp)}")
        if parts:
            return " ".join(parts)
        return str(r.get("run"))[:60]

    labels = [label_row(r) for _, r in d.iterrows()]

    plt.figure()
    plt.barh(range(len(d)), d["test_mae"].values)
    plt.yticks(range(len(d)), labels)
    plt.xlabel("Test MAE")
    plt.title("Test MAE per run (sorted)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    path = os.path.join(outdir, "test_mae_sorted.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_scatter_lr(df: pd.DataFrame, outdir: str) -> Optional[str]:
    needed = ["pretrained_lr", "head_lr", "test_mae"]
    if not all(c in df.columns for c in needed):
        return None
    d = df.dropna(subset=needed).copy()
    if d.empty:
        return None

    plt.figure()
    plt.scatter(d["pretrained_lr"], d["head_lr"], s=40)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("pretrained_lr (log)")
    plt.ylabel("head_lr (log)")
    plt.title("LR grid (each point is a run)")
    plt.tight_layout()

    path = os.path.join(outdir, "lr_grid_scatter.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


# ==============================
# Main
# ==============================

def main() -> int:
    ensure_outdir(OUTDIR)

    print("=" * 80)
    print("W&B Group Analysis")
    print(f"ENTITY:  {ENTITY}")
    print(f"PROJECT: {PROJECT}")
    print(f"GROUP:   {GROUP}")
    print(f"OUTDIR:  {OUTDIR}")
    print(f"STATES:  {ALLOWED_STATES}")
    print(f"BEST_METRIC: {BEST_METRIC}")
    print("=" * 80)

    api = wandb.Api(timeout=WANDB_TIMEOUT)

    runs = fetch_runs(api, ENTITY, PROJECT, GROUP)
    print(f"Fetched {len(runs)} runs for group '{GROUP}'.")

    # Convert to records and optionally filter by state
    records: List[RunRecord] = []
    for r in runs:
        rec = run_to_record(r)
        if ALLOWED_STATES and rec.state not in ALLOWED_STATES:
            continue
        records.append(rec)

    print(f"Keeping {len(records)} runs after state filter.")

    # Compute best validation from history (optional but useful)
    # If you want to skip history downloading entirely, set ENV: SKIP_HISTORY=1
    skip_hist = os.getenv("SKIP_HISTORY", "0") == "1"
    if not skip_hist:
        print("Downloading history to compute best validation (may take time)...")
        for i, (r, rec) in enumerate(zip(runs, records), start=1):
            # map record->run by id (robust)
            # build dict for quick lookup
            pass

        run_by_id = {rr.id: rr for rr in runs}
        for i, rec in enumerate(records, start=1):
            rr = run_by_id.get(rec.run_id)
            if rr is None:
                continue
            try:
                bv, be = download_history_for_best_valid(
                    rr,
                    warmup_frac=WARMUP_FRAC,
                    best_window=BEST_WINDOW,
                    max_rows=MAX_HISTORY_ROWS,
                )
                rec.best_valid_mae = bv
                rec.best_valid_epoch = be
            except Exception as e:
                # keep going
                rec.best_valid_mae = None
                rec.best_valid_epoch = None
            if i % 10 == 0 or i == len(records):
                print(f"  progress: {i}/{len(records)}")

    # Build tables
    df = build_runs_table(records)

    # Save raw table
    raw_csv = os.path.join(OUTDIR, "runs_raw.csv")
    df.to_csv(raw_csv, index=False)
    print(f"Saved per-run table: {raw_csv}")

    # Best run
    br = best_run(df, primary=BEST_METRIC, tiebreak=TIEBREAK_METRIC)
    best_json = os.path.join(OUTDIR, "best_run.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(br.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "-" * 80)
    print("BEST RUN (by metric)")
    print(f"Primary metric: {BEST_METRIC}  (tiebreak: {TIEBREAK_METRIC})")
    print(f"Run:   {br.get('run')}")
    print(f"ID:    {br.get('run_id')}")
    print(f"State: {br.get('state')}")
    print(f"URL:   {br.get('url')}")
    print("--- Hyperparams ---")
    for k in ["pretrained_lr", "head_lr", "train_batch_size", "valid_batch_size", "seed", "split_id", "mask_ratio", "mask_seed"]:
        if k in br.index:
            print(f"{k:16s}: {br.get(k)}")
    print("--- Metrics ---")
    for k in ["test_mae", "test_medae", "test_rmse", "test_r2", "test_s_r", "best_valid_mae", "best_valid_epoch", VALID_MAE_KEY, VALID_MAE_FALLBACK]:
        if k in br.index:
            print(f"{k:16s}: {br.get(k)}")
    print(f"Saved best run json: {best_json}")
    print("-" * 80 + "\n")

    # Aggregation by hyperparams
    agg = aggregate_by_hparams(df)
    if not agg.empty:
        agg_csv = os.path.join(OUTDIR, "runs_agg_by_lr.csv")
        agg.to_csv(agg_csv, index=False)
        print(f"Saved aggregated table: {agg_csv}")
        print("\nTop aggregated rows:")
        print(agg.head(10).to_string(index=False))

    # Plots
    p1 = plot_test_mae_sorted(df, OUTDIR)
    if p1:
        print(f"Saved plot: {p1}")

    p2 = plot_scatter_lr(df, OUTDIR)
    if p2:
        print(f"Saved plot: {p2}")

    # Also save a compact markdown summary
    md_path = os.path.join(OUTDIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# W&B Group Analysis\n\n")
        f.write(f"- Entity: `{ENTITY}`\n")
        f.write(f"- Project: `{PROJECT}`\n")
        f.write(f"- Group: `{GROUP}`\n")
        f.write(f"- Runs analyzed: `{len(df)}`\n\n")
        f.write("## Best run\n")
        f.write(f"- run: `{br.get('run')}`\n")
        f.write(f"- run_id: `{br.get('run_id')}`\n")
        f.write(f"- url: {br.get('url')}\n")
        f.write("\n### Hyperparameters\n")
        for k in ["pretrained_lr", "head_lr", "train_batch_size", "valid_batch_size", "seed", "split_id", "mask_ratio", "mask_seed"]:
            if k in br.index:
                f.write(f"- {k}: `{br.get(k)}`\n")
        f.write("\n### Metrics\n")
        for k in ["test_mae", "test_medae", "test_rmse", "test_r2", "test_s_r", "best_valid_mae", "best_valid_epoch"]:
            if k in br.index:
                f.write(f"- {k}: `{br.get(k)}`\n")

        if not agg.empty:
            f.write("\n## Aggregation by (pretrained_lr, head_lr)\n")
            f.write("(see CSV for full table)\n\n")
            f.write(agg.head(10).to_markdown(index=False))
            f.write("\n")

    print(f"Saved markdown summary: {md_path}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
