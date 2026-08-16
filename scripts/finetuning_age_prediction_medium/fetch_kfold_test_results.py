#!/usr/bin/env python3
"""
Pull the current 5-fold CV results for MethylGPT 21k AltumAge directly from
WandB (no manual copy-paste) and print a summary table with mean +/- 95% CI.

Reruns live: reflects whatever is currently in WandB, so if a fold gets
resumed/rerun this always shows the latest state rather than a cached
snapshot.

Requires WANDB_API_KEY in the environment (same variable already exported
by run_medium_21k_altumage_fold*.sbatch):
    export WANDB_API_KEY=...
    python fetch_kfold_test_results.py

Usage:
    python fetch_kfold_test_results.py
    python fetch_kfold_test_results.py --entity <your-wandb-entity>
"""

import argparse
import json
import os
import sys

import requests
from scipy import stats

ENTITY_DEFAULT = "netanelazran11-hebrew-university-of-jerusalem"
PROJECT_TEMPLATE = "methylGPT_medium_21k_altumage_fold{fold}"
N_FOLDS = 5

RUNS_QUERY = """
query Runs($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runs(first: 10) {
      edges {
        node {
          name
          displayName
          state
          updatedAt
          summaryMetrics
        }
      }
    }
  }
}
"""


def fetch_fold(entity: str, fold: int, api_key: str) -> dict | None:
    project = PROJECT_TEMPLATE.format(fold=fold)
    resp = requests.post(
        "https://api.wandb.ai/graphql",
        auth=("api", api_key),
        json={"query": RUNS_QUERY, "variables": {"entity": entity, "project": project}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        print(f"  fold {fold}: GraphQL errors: {data['errors']}", file=sys.stderr)
        return None
    proj = data.get("data", {}).get("project")
    if proj is None:
        print(f"  fold {fold}: project '{project}' not found or no access", file=sys.stderr)
        return None
    edges = proj["runs"]["edges"]
    if not edges:
        print(f"  fold {fold}: no runs in project '{project}'", file=sys.stderr)
        return None
    # Most recently updated run for this fold, in case of reruns/resumes.
    node = max(edges, key=lambda e: e["node"]["updatedAt"])["node"]
    summary = json.loads(node["summaryMetrics"])
    return {
        "fold": fold,
        "run_id": node["name"],
        "state": node["state"],
        "updated_at": node["updatedAt"],
        "epoch": summary.get("epoch"),
        "valid_medae": summary.get("valid_medae"),
        "test_top1_medae": summary.get("test_top1_medae"),
        "test_top1_mae": summary.get("test_top1_mae"),
        "test_top1_r2": summary.get("test_top1_r2"),
    }


def mean_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n > 1:
        ci = stats.sem(vals) * stats.t.ppf(0.975, n - 1)
    else:
        ci = float("nan")
    return m, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=ENTITY_DEFAULT)
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("ERROR: WANDB_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)

    rows = []
    print(f"Fetching latest run per fold from entity '{args.entity}' ...")
    for fold in range(N_FOLDS):
        row = fetch_fold(args.entity, fold, api_key)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No fold results found.", file=sys.stderr)
        sys.exit(1)

    print()
    header = f"{'fold':>4} {'run_id':>10} {'state':>10} {'epoch':>6} {'test_medae':>11} {'test_mae':>9} {'test_r2':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        medae = r["test_top1_medae"]
        mae = r["test_top1_mae"]
        r2 = r["test_top1_r2"]
        print(f"{r['fold']:>4} {r['run_id']:>10} {r['state']:>10} {str(r['epoch']):>6} "
              f"{medae if medae is None else f'{medae:.4f}':>11} "
              f"{mae if mae is None else f'{mae:.4f}':>9} "
              f"{r2 if r2 is None else f'{r2:.4f}':>8}")

    finished = [r for r in rows if r["state"] == "finished" and r["test_top1_medae"] is not None]
    print(f"\n{len(finished)}/{N_FOLDS} folds finished with test results.")
    if len(finished) == N_FOLDS:
        for key, label in [("test_top1_medae", "MedAE"), ("test_top1_mae", "MAE"), ("test_top1_r2", "R2")]:
            m, ci = mean_ci([r[key] for r in finished])
            print(f"  {label}: {m:.4f} +/- {ci:.4f} (95% CI, n={len(finished)})")
    else:
        missing = [r["fold"] for r in rows if r["state"] != "finished" or r["test_top1_medae"] is None]
        print(f"  Not all folds done yet (missing/incomplete: {missing}) — mean/CI skipped.")


if __name__ == "__main__":
    main()
