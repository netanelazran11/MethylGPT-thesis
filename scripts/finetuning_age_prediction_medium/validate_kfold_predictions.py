#!/usr/bin/env python3
"""
Validate the 5 per-sample MethylGPT prediction files produced by
extract_kfold_predictions.py before they're trusted for the paired
subject-level bootstrap comparison against V7b.

Run on the cluster (needs the same venv as extraction; the determinism
re-check also needs GPU + h5ad + checkpoint access), after all 5
run_extract_kfold_predictions_fold{0..4}.sbatch jobs have finished:

    source /sci/labs/benjamin.yakir/netanel.azran/venv_torch22/bin/activate
    export WANDB_API_KEY=...   # same key already in the training sbatch scripts
    python validate_kfold_predictions.py

Exits non-zero if any check fails. Per-fold integrity checks run before any
metric recomputation and are wrapped so a corrupted fold is reported cleanly
rather than crashing validation of the other folds.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

EXPECTED_N = 2149
N_FOLDS = 5
REQUIRED_COLUMNS = ["sample_id", "true_age", "predicted_age", "fold", "model", "checkpoint"]

# Engineering tolerance for CSV-recomputed vs. official WandB test_top1_*
# metrics. Not zero: official evaluation ran inside Lightning's Trainer.test()
# (DDPStrategy wrapper, its own autocast entry point); this script's
# extraction reruns the same bf16-mixed autocast forward pass standalone.
# Both are the *real* model on the *real* checkpoint -- residual differences
# here are expected to be numerical-path noise, not a logic bug. The actual
# observed diff is always printed, never silently swallowed by the tolerance.
METRIC_TOL = 0.02


def report(ok: bool, msg: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {msg}", flush=True)
    return ok


class FoldResult:
    def __init__(self, fold):
        self.fold = fold
        self.ok = True
        self.df = None
        self.verification = None
        self.errors = []

    def fail(self, msg):
        self.ok = False
        self.errors.append(msg)
        report(False, f"fold {self.fold}: {msg}")


def load_and_check_integrity(fold: int, pred_dir: Path) -> FoldResult:
    """Structural/integrity checks only -- no metric math yet, so a corrupted
    fold (bad row count, NaNs, dup IDs) is caught and reported before it can
    blow up anything downstream."""
    res = FoldResult(fold)
    csv_path = pred_dir / f"fold_{fold}_predictions.csv"
    json_path = pred_dir / f"fold_{fold}_verification.json"

    if not csv_path.exists():
        res.fail(f"missing {csv_path}")
        return res
    if not json_path.exists():
        res.fail(f"missing {json_path}")
        return res

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        res.fail(f"could not read {csv_path}: {e}")
        return res

    try:
        verification = json.loads(json_path.read_text())
    except Exception as e:
        res.fail(f"could not parse {json_path}: {e}")
        return res

    res.df = df
    res.verification = verification

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        res.fail(f"missing columns: {missing_cols}")
        return res

    if len(df) != EXPECTED_N:
        res.fail(f"expected {EXPECTED_N} rows, got {len(df)}")

    n_unique = df["sample_id"].nunique()
    if n_unique != len(df):
        res.fail(f"{len(df) - n_unique} duplicate sample_id(s) in CSV")

    if n_unique != EXPECTED_N:
        res.fail(f"expected {EXPECTED_N} unique sample_id(s), got {n_unique}")

    n_nan_pred = df["predicted_age"].isna().sum()
    if n_nan_pred:
        res.fail(f"{n_nan_pred} NaN predicted_age value(s)")

    n_nan_true = df["true_age"].isna().sum()
    if n_nan_true:
        res.fail(f"{n_nan_true} NaN true_age value(s)")

    if (df["model"] != "MethylGPT").any():
        res.fail("model column contains a value other than 'MethylGPT'")

    if (df["fold"] != fold).any():
        res.fail(f"fold column contains a value other than {fold}")

    if res.ok:
        report(True, f"fold {fold}: integrity OK ({len(df)} rows, {n_unique} unique sample_ids)")

    return res


def check_metrics_vs_official(res: FoldResult, official: dict | None) -> bool:
    if not res.ok or res.df is None:
        return False
    df = res.df
    medae = median_absolute_error(df["true_age"], df["predicted_age"])
    mae = mean_absolute_error(df["true_age"], df["predicted_age"])
    r2 = r2_score(df["true_age"], df["predicted_age"])

    print(f"  fold {res.fold} recomputed from CSV: MedAE={medae:.4f} MAE={mae:.4f} R2={r2:.4f}", flush=True)

    if official is None:
        return report(False, f"fold {res.fold}: no official WandB test_top1 metrics available to compare against")

    ok = True
    for name, recomputed, off_key in [("MedAE", medae, "test_top1_medae"),
                                        ("MAE", mae, "test_top1_mae"),
                                        ("R2", r2, "test_top1_r2")]:
        off_val = official.get(off_key)
        if off_val is None:
            ok = report(False, f"fold {res.fold}: official {off_key} unavailable") and ok
            continue
        diff = abs(recomputed - off_val)
        this_ok = diff <= METRIC_TOL
        ok = report(
            this_ok,
            f"fold {res.fold}: {name} recomputed={recomputed:.4f} official={off_val:.4f} "
            f"diff={diff:.4f} (tol={METRIC_TOL})",
        ) and ok
    return ok


def fetch_official_metrics(entity: str) -> dict:
    """Live WandB pull, reusing fetch_kfold_test_results.py's own fetch_fold()
    rather than re-implementing the GraphQL query -- and rather than trusting
    a hardcoded/guessed number."""
    try:
        import fetch_kfold_test_results as fkr
    except ImportError as e:
        print(f"WARNING: could not import fetch_kfold_test_results.py ({e}); "
              "official-metric cross-check will be skipped.", flush=True)
        return {}

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("WARNING: WANDB_API_KEY not set; official-metric cross-check will be skipped.", flush=True)
        return {}

    official = {}
    for fold in range(N_FOLDS):
        row = fkr.fetch_fold(entity, fold, api_key)
        if row is not None:
            official[fold] = row
    return official


def check_cross_fold_consistency(results: list[FoldResult]) -> bool:
    ok_results = [r for r in results if r.ok]
    if len(ok_results) < N_FOLDS:
        report(False, f"cross-fold checks skipped: only {len(ok_results)}/{N_FOLDS} folds passed integrity checks")
        return False

    ok = True

    # 1. Identical sample-ID membership across all 5 folds (sha256 shortcut,
    #    then a real diff if the hashes ever disagree).
    hashes = {r.fold: r.verification.get("sorted_sample_id_sha256") for r in ok_results}
    unique_hashes = set(hashes.values())
    if len(unique_hashes) == 1:
        ok = report(True, f"all 5 folds share identical sample-ID membership (sha256={next(iter(unique_hashes))[:16]}...)") and ok
    else:
        ok = report(False, f"sample-ID membership differs across folds: {hashes}") and ok
        ids_by_fold = {r.fold: set(r.df["sample_id"]) for r in ok_results}
        base = ids_by_fold[ok_results[0].fold]
        for r in ok_results[1:]:
            diff = base.symmetric_difference(ids_by_fold[r.fold])
            if diff:
                print(f"    fold {ok_results[0].fold} vs fold {r.fold}: {len(diff)} differing IDs "
                      f"(e.g. {sorted(diff)[:5]})", flush=True)

    # 2. true_age identical for every sample_id across all 5 folds.
    merged = None
    for r in ok_results:
        sub = r.df[["sample_id", "true_age"]].rename(columns={"true_age": f"true_age_fold{r.fold}"})
        merged = sub if merged is None else merged.merge(sub, on="sample_id", how="outer")

    age_cols = [c for c in merged.columns if c.startswith("true_age_fold")]
    age_arr = merged[age_cols].to_numpy(dtype=np.float64)
    row_ranges = np.nanmax(age_arr, axis=1) - np.nanmin(age_arr, axis=1)
    n_inconsistent = int((row_ranges > 1e-6).sum())
    if n_inconsistent == 0:
        ok = report(True, f"true_age identical for all {len(merged)} sample_ids across all 5 folds") and ok
    else:
        ok = report(False, f"{n_inconsistent} sample_id(s) have inconsistent true_age across folds") and ok

    return ok


def check_determinism(fold: int) -> bool:
    """Rerun ONE fold's extraction a second time and diff predictions.
    Nothing fold-specific affects reproducibility, so proving it once is
    sufficient (halves runtime vs. checking all 5)."""
    pred_dir = Path(os.environ.get(
        "METHYLGPT_PRED_OUTPUT_DIR",
        str(SCRIPT_DIR.parent.parent / "outputs" / "bootstrap_predictions" / "methylgpt"),
    ))
    original_csv = pred_dir / f"fold_{fold}_predictions.csv"
    if not original_csv.exists():
        return report(False, f"determinism check: {original_csv} missing, cannot compare")

    rerun_dir = pred_dir / "_determinism_check"
    rerun_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rerunning fold {fold} extraction for determinism check -> {rerun_dir}", flush=True)

    cmd = [
        sys.executable, str(SCRIPT_DIR / "extract_kfold_predictions.py"),
        "--fold", str(fold),
        "--output_dir", str(rerun_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], flush=True)
        return report(False, f"determinism rerun of fold {fold} exited {proc.returncode}")

    rerun_csv = rerun_dir / f"fold_{fold}_predictions.csv"
    if not rerun_csv.exists():
        return report(False, f"determinism rerun did not produce {rerun_csv}")

    orig = pd.read_csv(original_csv).sort_values("sample_id").reset_index(drop=True)
    rerun = pd.read_csv(rerun_csv).sort_values("sample_id").reset_index(drop=True)

    if not orig["sample_id"].equals(rerun["sample_id"]):
        return report(False, "determinism check: sample_id sets differ between runs")

    diffs = (orig["predicted_age"] - rerun["predicted_age"]).abs()
    max_diff = float(diffs.max())
    ok = max_diff <= 1e-5
    return report(
        ok,
        f"determinism check fold {fold}: max |pred diff| between two runs = {max_diff:.8f} (tol=1e-5)",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        default=str(SCRIPT_DIR.parent.parent / "outputs" / "bootstrap_predictions" / "methylgpt"),
    )
    parser.add_argument("--entity", default="netanelazran11-hebrew-university-of-jerusalem")
    parser.add_argument("--skip_determinism", action="store_true",
                         help="Skip the rerun-and-diff determinism check (still runs by default).")
    parser.add_argument("--determinism_fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)

    print("=== 1. Per-fold integrity checks (before any metric math) ===", flush=True)
    results = [load_and_check_integrity(fold, pred_dir) for fold in range(N_FOLDS)]
    all_integrity_ok = all(r.ok for r in results)

    print("\n=== 2. Recomputed metrics vs. official WandB test_top1_* ===", flush=True)
    official = fetch_official_metrics(args.entity)
    metrics_ok = True
    for r in results:
        if r.ok:
            metrics_ok = check_metrics_vs_official(r, official.get(r.fold)) and metrics_ok
        else:
            print(f"  fold {r.fold}: skipped (failed integrity checks above)", flush=True)
            metrics_ok = False

    print("\n=== 3. Cross-fold consistency (identical sample-ID membership, identical true_age) ===", flush=True)
    cross_fold_ok = check_cross_fold_consistency(results)

    determinism_ok = True
    if not args.skip_determinism:
        print(f"\n=== 4. Determinism recheck (fold {args.determinism_fold}, rerun + diff) ===", flush=True)
        determinism_ok = check_determinism(args.determinism_fold)
    else:
        print("\n=== 4. Determinism recheck: SKIPPED (--skip_determinism) ===", flush=True)

    print("\n=== Summary ===", flush=True)
    overall_ok = all_integrity_ok and metrics_ok and cross_fold_ok and determinism_ok
    for label, ok in [
        ("Integrity (all 5 folds)", all_integrity_ok),
        ("Metrics vs. official WandB", metrics_ok),
        ("Cross-fold consistency", cross_fold_ok),
        ("Determinism recheck", determinism_ok),
    ]:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}", flush=True)

    if not overall_ok:
        print("\nVALIDATION FAILED -- see FAIL lines above.", flush=True)
        sys.exit(1)

    print("\nALL CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
