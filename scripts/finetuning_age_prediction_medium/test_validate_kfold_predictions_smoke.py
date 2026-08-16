#!/usr/bin/env python3
"""
Local smoke test for validate_kfold_predictions.py's integrity/cross-fold
logic, using synthetic fixtures -- no GPU, no cluster, no WandB, no real
checkpoints needed. Run on a laptop before ever trusting the validator
against real cluster output.

Covers:
  1. Five well-formed synthetic folds -> all integrity checks pass, and
     cross-fold consistency (identical sample-ID membership, identical
     true_age per sample_id) passes.
  2. The same five folds but with fold 2 deliberately corrupted three ways
     (wrong row count, a duplicate sample_id, a NaN prediction) -> fold 2
     is reported as FAIL without raising, and folds 0/1/3/4 are still
     correctly checked (one corrupted fold must not take down the rest).

Usage:
    python test_validate_kfold_predictions_smoke.py
"""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import validate_kfold_predictions as vkp  # noqa: E402

N = vkp.EXPECTED_N
N_FOLDS = vkp.N_FOLDS

FAILURES = []


def check(condition: bool, msg: str):
    status = "ok" if condition else "FAILED"
    print(f"  [{status}] {msg}")
    if not condition:
        FAILURES.append(msg)


def make_good_fold_files(out_dir: Path, fold: int, sample_ids, true_ages, rng):
    preds = true_ages + rng.normal(0, 1.0, size=len(true_ages))
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_age": true_ages,
        "predicted_age": preds,
        "fold": fold,
        "model": "MethylGPT",
        "checkpoint": f"/fake/checkpoints_fold{fold}/fake.ckpt",
    })
    df.to_csv(out_dir / f"fold_{fold}_predictions.csv", index=False)

    sorted_ids = sorted(sample_ids.tolist())
    id_hash = hashlib.sha256("\n".join(sorted_ids).encode("utf-8")).hexdigest()
    verification = {
        "fold": fold,
        "checkpoint": f"/fake/checkpoints_fold{fold}/fake.ckpt",
        "checkpoint_valid_medae": 3.3 + fold * 0.01,
        "n_rows": len(df),
        "n_unique_sample_ids": len(set(sample_ids)),
        "duplicate_id_count": 0,
        "missing_prediction_count": 0,
        "medae": float(np.median(np.abs(df["true_age"] - df["predicted_age"]))),
        "mae": float(np.mean(np.abs(df["true_age"] - df["predicted_age"]))),
        "r2": 0.85,
        "true_age_min": float(true_ages.min()),
        "true_age_max": float(true_ages.max()),
        "sorted_sample_id_sha256": id_hash,
    }
    (out_dir / f"fold_{fold}_verification.json").write_text(json.dumps(verification, indent=2))


def scenario_all_good():
    print("\n=== Scenario 1: five well-formed folds ===")
    rng = np.random.default_rng(0)
    sample_ids = np.array([f"GSM{100000 + i}" for i in range(N)])
    true_ages = rng.uniform(0, 100, size=N)

    tmp = Path(tempfile.mkdtemp(prefix="mgpt_smoke_good_"))
    try:
        for fold in range(N_FOLDS):
            make_good_fold_files(tmp, fold, sample_ids, true_ages, rng)

        results = [vkp.load_and_check_integrity(fold, tmp) for fold in range(N_FOLDS)]
        check(all(r.ok for r in results), "all 5 synthetic folds pass integrity checks")

        cross_ok = vkp.check_cross_fold_consistency(results)
        check(cross_ok is True, "cross-fold consistency passes for well-formed synthetic folds")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_corrupted_fold():
    print("\n=== Scenario 2: fold 2 deliberately corrupted (others must still be checked) ===")
    rng = np.random.default_rng(1)
    sample_ids = np.array([f"GSM{200000 + i}" for i in range(N)])
    true_ages = rng.uniform(0, 100, size=N)

    tmp = Path(tempfile.mkdtemp(prefix="mgpt_smoke_bad_"))
    try:
        for fold in range(N_FOLDS):
            make_good_fold_files(tmp, fold, sample_ids, true_ages, rng)

        # Corrupt fold 2: drop a row (wrong count), duplicate another
        # sample_id, and inject a NaN prediction.
        bad_csv = tmp / "fold_2_predictions.csv"
        df = pd.read_csv(bad_csv)
        df = df.iloc[:-1].copy()               # wrong row count
        df.iloc[0, df.columns.get_loc("sample_id")] = df.iloc[1]["sample_id"]  # duplicate ID
        df.iloc[5, df.columns.get_loc("predicted_age")] = float("nan")         # NaN prediction
        df.to_csv(bad_csv, index=False)

        results = [vkp.load_and_check_integrity(fold, tmp) for fold in range(N_FOLDS)]

        ok_folds = {r.fold for r in results if r.ok}
        check(ok_folds == {0, 1, 3, 4}, f"folds 0/1/3/4 pass, fold 2 fails (got ok={sorted(ok_folds)})")

        fold2 = results[2]
        check(not fold2.ok, "fold 2 correctly reported as FAILED, not raised as an exception")
        check(len(fold2.errors) >= 1, f"fold 2 has recorded error messages: {fold2.errors}")

        # Cross-fold check must not crash even though fold 2 is broken --
        # it should just report which folds it could actually check.
        try:
            cross_ok = vkp.check_cross_fold_consistency(results)
            check(cross_ok is False, "cross-fold consistency correctly reports failure (fold 2 missing/corrupted) without raising")
        except Exception as e:
            check(False, f"check_cross_fold_consistency raised instead of failing cleanly: {e!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    scenario_all_good()
    scenario_corrupted_fold()

    print("\n=== Smoke test summary ===")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All smoke-test checks passed.")


if __name__ == "__main__":
    main()
