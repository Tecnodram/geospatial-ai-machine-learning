#!/usr/bin/env python
# coding: utf-8
"""
rebuild_experiment_index.py
----------------------------
Scan experiments/ and reconstruct experiment_index.csv from available artifacts.

Priority per run directory:
  1. metadata.json (new format, written by immutable pipeline)
  2. config_snapshot.json + cv_report.json + feature_manifest_*.json (legacy)

Writes: experiments/experiment_index.csv
"""

import os
import json
import pandas as pd

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
INDEX_PATH = os.path.join(EXPERIMENTS_DIR, "experiment_index.csv")

COLS = [
    "experiment_id",
    "cv_mean",
    "drp_cv",
    "model_family",
    "y_mode_drp",
    "feature_count_drp",
    "submission_path",
]


def _load_manifest_count(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        mf = json.load(f)
    cols = mf if isinstance(mf, list) else mf.get("columns", mf.get("features", []))
    return len(cols)


def _submission_path_for(run_dir: str, run_id: str) -> str:
    """Find the canonical submission file for a run directory."""
    # New immutable path
    p = os.path.join(run_dir, "submission.csv")
    if os.path.exists(p):
        return p
    # Legacy fixed name in run dir
    p2 = os.path.join(run_dir, "submission_V5_2_OOFTE_fixkeys.csv")
    if os.path.exists(p2):
        return p2
    # Legacy as_is variant
    p3 = os.path.join(run_dir, "submission_V5_2_OOFTE_fixkeys_as_is.csv")
    if os.path.exists(p3):
        return p3
    return ""


def _row_from_metadata(run_id: str, run_dir: str) -> dict:
    """Build index row from new metadata.json."""
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as f:
        md = json.load(f)

    cv_vals = [v["mean"] for v in md.get("cv_metrics", {}).values()
               if isinstance(v, dict) and "mean" in v]
    cv_mean = round(sum(cv_vals) / len(cv_vals), 6) if cv_vals else ""
    drp_cv = md.get("cv_metrics", {}).get("Dissolved Reactive Phosphorus", {}).get("mean", "")
    if isinstance(drp_cv, float):
        drp_cv = round(drp_cv, 6)

    sub_path = md.get("submission_path", "") or _submission_path_for(run_dir, run_id)

    return {
        "experiment_id": run_id,
        "cv_mean": cv_mean,
        "drp_cv": drp_cv,
        "model_family": md.get("model_config", {}).get("drp_model", ""),
        "y_mode_drp": md.get("model_config", {}).get("drp_y_mode", ""),
        "feature_count_drp": md.get("feature_counts", {}).get("Dissolved Reactive Phosphorus", ""),
        "submission_path": sub_path,
    }


def _row_from_legacy(run_id: str, run_dir: str) -> dict:
    """Build index row from config_snapshot.json + cv_report.json (legacy)."""
    cfg_path = os.path.join(run_dir, "config_snapshot.json")
    cv_path  = os.path.join(run_dir, "cv_report.json")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    cv_report = {}
    if os.path.exists(cv_path):
        with open(cv_path, encoding="utf-8") as f:
            raw = json.load(f)
        # Guard: some legacy cv_reports are a list, not a dict
        cv_report = raw if isinstance(raw, dict) else {}

    drp_cfg = cfg.get("model", {}).get("cv_by_target", {}).get("Dissolved Reactive Phosphorus", {})
    # Old schema: targets is a list; model is flat (no cv_by_target)
    targets_field = cfg.get("targets", {})
    if isinstance(targets_field, list):
        ymode_map = {}
        drp_model = cfg.get("model", {}).get("type",
                    cfg.get("model", {}).get("name", "ExtraTreesRegressor"))
    else:
        ymode_map = targets_field.get("y_mode_by_target", {})
        drp_model = drp_cfg.get("name", "ExtraTreesRegressor")

    drp_ymode = ymode_map.get("Dissolved Reactive Phosphorus",
                              cfg.get("targets", {}).get("y_mode", "") if not isinstance(targets_field, list) else "")

    mf_path = os.path.join(run_dir, "feature_manifest_Dissolved_Reactive_Phosphorus.json")
    drp_feat_count = _load_manifest_count(mf_path)

    cv_vals = [v["mean"] for v in cv_report.values()
               if isinstance(v, dict) and "mean" in v]
    cv_mean = round(sum(cv_vals) / len(cv_vals), 6) if cv_vals else ""

    drp_cv_block = cv_report.get("Dissolved Reactive Phosphorus", {})
    drp_cv = ""
    if isinstance(drp_cv_block, dict) and "mean" in drp_cv_block:
        drp_cv = round(drp_cv_block["mean"], 6)

    sub_path = _submission_path_for(run_dir, run_id)

    return {
        "experiment_id": run_id,
        "cv_mean": cv_mean,
        "drp_cv": drp_cv,
        "model_family": drp_model,
        "y_mode_drp": drp_ymode,
        "feature_count_drp": drp_feat_count,
        "submission_path": sub_path,
    }


def rebuild():
    rows = []
    skipped = []

    for entry in sorted(os.listdir(EXPERIMENTS_DIR)):
        run_dir = os.path.join(EXPERIMENTS_DIR, entry)
        if not os.path.isdir(run_dir):
            continue
        # Only process exp_* timestamped directories
        if not entry.startswith("exp_"):
            continue

        run_id = entry
        meta_path = os.path.join(run_dir, "metadata.json")
        cfg_path  = os.path.join(run_dir, "config_snapshot.json")

        try:
            if os.path.exists(meta_path):
                row = _row_from_metadata(run_id, run_dir)
                source = "metadata.json"
            elif os.path.exists(cfg_path):
                row = _row_from_legacy(run_id, run_dir)
                source = "legacy"
            else:
                skipped.append(f"  SKIP {run_id}: no metadata.json or config_snapshot.json")
                continue
            rows.append(row)
            print(f"  OK   {run_id} [{source}]  drp_cv={row['drp_cv']}  model={row['model_family']}")
        except Exception as e:
            skipped.append(f"  ERR  {run_id}: {e}")

    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(INDEX_PATH, index=False)

    print(f"\nWrote {len(df)} rows -> {INDEX_PATH}")
    if skipped:
        print("\nSkipped / errors:")
        for s in skipped:
            print(s)

    return df


if __name__ == "__main__":
    print(f"Scanning {EXPERIMENTS_DIR} ...")
    df = rebuild()
    print("\nSample output:")
    print(df[["experiment_id", "cv_mean", "drp_cv", "model_family", "y_mode_drp",
              "feature_count_drp"]].to_string(index=False))
