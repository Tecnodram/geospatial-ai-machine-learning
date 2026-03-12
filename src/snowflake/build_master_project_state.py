#!/usr/bin/env python
"""Build EY_WQ.AUDIT.MASTER_PROJECT_STATE from local experiments and reference submissions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

sys.path.insert(0, str(Path(__file__).parent))
from _connect import get_connection

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
ANCHOR_PATH = ROOT / "submissions" / "submission_V4_4_DRP_tuned_ET_fixorder.csv"
BASELINE_PATH = ROOT / "submissions_batch" / "submission_V5_2_OOFTE_fixkeys_as_is.csv"
BLEND4_PATH = ROOT / "experiments" / "mini_sprint_blends" / "blend4_v44_80_134704_20_rank.csv"
BLEND85_PATH = ROOT / "experiments" / "exp_20260311_200705" / "mini_sprint_blends" / "blend_v44_85_134704_15_rank.csv"
ANCHOR_LB = 0.3039
BLEND4_LB = 0.3029
BLEND85_LB = 0.3029
SUBMISSION_NAME_CANDIDATES = [
    "submission_V5_2_OOFTE_fixkeys_as_is.csv",
    "submission_V5_2_OOFTE_fixkeys.csv",
]
SPECIAL_SUBMISSIONS = [
    ("anchor_v44", ANCHOR_PATH, "anchor_reference", ANCHOR_LB, "Historical leaderboard anchor"),
    ("blend4_v44_80_134704_20_rank", BLEND4_PATH, "validated_blend_reference", BLEND4_LB, "Validated leaderboard blend near anchor"),
    ("blend_v44_85_134704_15_rank", BLEND85_PATH, "validated_blend_reference", BLEND85_LB, "Validated leaderboard blend near anchor"),
    ("baseline_v52", BASELINE_PATH, "current_baseline", None, "Current technical baseline"),
    ("drp_basin_rain", ROOT / "experiments" / "drp_basin_rain" / "submission_V5_2_OOFTE_fixkeys_as_is.csv", "recent_drp_submission", None, "DRP basin rain submission-only variant"),
    ("drp_variant1", ROOT / "experiments" / "drp_variant1" / "submission_V5_2_OOFTE_fixkeys_as_is.csv", "recent_drp_submission", None, "DRP variant 1 submission-only variant"),
    ("drp_variant2", ROOT / "experiments" / "drp_variant2" / "submission_V5_2_OOFTE_fixkeys_as_is.csv", "recent_drp_submission", None, "DRP variant 2 submission-only variant"),
    ("drp_variant3", ROOT / "experiments" / "drp_variant3" / "submission_V5_2_OOFTE_fixkeys_as_is.csv", "recent_drp_submission", None, "DRP variant 3 submission-only variant"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_similarity(anchor_df: pd.DataFrame, candidate_path: Path) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    if not candidate_path.exists():
        return (None, None, None, None, None, None)
    cand = pd.read_csv(candidate_path)
    out = []
    for target in TARGETS:
        corr = cand[target].corr(anchor_df[target])
        mae = (cand[target] - anchor_df[target]).abs().mean()
        out.extend([float(corr), float(mae)])
    return tuple(out)


def latest_proxy_map() -> dict[str, dict]:
    log_path = ROOT / "leaderboard_log.csv"
    if not log_path.exists():
        return {}
    df = pd.read_csv(log_path)
    out = {}
    if "run_id" in df.columns:
        for run_id, sub in df.dropna(subset=["run_id"]).groupby("run_id", sort=False):
            row = sub.iloc[-1]
            out[str(run_id)] = row.to_dict()
    return out


def find_submission(exp_path: Path) -> Path | None:
    for name in SUBMISSION_NAME_CANDIDATES:
        candidate = exp_path / name
        if candidate.exists():
            return candidate
    return None


def manifest_feature_count(exp_path: Path) -> int | None:
    path = exp_path / "feature_manifest_Dissolved_Reactive_Phosphorus.json"
    if not path.exists():
        return None
    data = read_json(path)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        cols = data.get("columns") or data.get("features") or list(data.keys())
        return len(cols)
    return None


def build_experiment_rows(anchor_df: pd.DataFrame, registry_ids: set[str]) -> list[dict]:
    rows = []
    proxy_by_run = latest_proxy_map()
    exp_root = ROOT / "experiments"
    for exp_path in sorted([p for p in exp_root.iterdir() if p.is_dir()]):
        cfg_path = exp_path / "config_snapshot.json"
        cv_path = exp_path / "cv_report.json"
        if not cfg_path.exists() or not cv_path.exists():
            continue

        cfg = read_json(cfg_path)
        cv = read_json(cv_path)
        submission_path = find_submission(exp_path)
        ta = float(cv.get("Total Alkalinity", {}).get("mean", "nan"))
        ec = float(cv.get("Electrical Conductance", {}).get("mean", "nan"))
        drp = float(cv.get("Dissolved Reactive Phosphorus", {}).get("mean", "nan"))
        cv_mean = (ta + ec + drp) / 3.0
        drp_cfg = (
            cfg.get("model", {})
            .get("cv_by_target", {})
            .get("Dissolved Reactive Phosphorus", {})
        )
        drp_model = drp_cfg.get("name") if drp_cfg else "ExtraTreesRegressor"
        drp_y_mode = (
            cfg.get("targets", {})
            .get("y_mode_by_target", {})
            .get("Dissolved Reactive Phosphorus")
            or cfg.get("targets", {}).get("y_mode")
        )
        te_enabled = cfg.get("te", {}).get("enabled")
        proxies = proxy_by_run.get(exp_path.name, {})
        sim = safe_similarity(anchor_df, submission_path) if submission_path else (None, None, None, None, None, None)
        rows.append({
            "entry_id": exp_path.name,
            "entry_type": "experiment",
            "entry_group": "historical_experiment",
            "run_id": exp_path.name,
            "experiment_path": str(exp_path),
            "submission_name": submission_path.name if submission_path else None,
            "submission_path": str(submission_path) if submission_path else None,
            "registered_in_registry": exp_path.name in registry_ids,
            "ta_cv_mean": ta,
            "ec_cv_mean": ec,
            "drp_cv_mean": drp,
            "cv_mean": cv_mean,
            "leaderboard_score": None,
            "latest_proxy": proxies.get("proxy"),
            "recommended_name": proxies.get("recommended_name"),
            "drp_model": drp_model,
            "drp_y_mode": drp_y_mode,
            "te_enabled": te_enabled,
            "drp_feature_count": manifest_feature_count(exp_path),
            "ta_corr_to_anchor": sim[0],
            "ta_mae_to_anchor": sim[1],
            "ec_corr_to_anchor": sim[2],
            "ec_mae_to_anchor": sim[3],
            "drp_corr_to_anchor": sim[4],
            "drp_mae_to_anchor": sim[5],
            "what_changed": f"DRP model={drp_model}; DRP y_mode={drp_y_mode}; TE={te_enabled}",
            "notes": "experiment_with_config_and_cv",
        })
    return rows


def build_reference_rows(anchor_df: pd.DataFrame) -> list[dict]:
    rows = []
    for entry_id, path, group, lb, note in SPECIAL_SUBMISSIONS:
        if not path.exists():
            continue
        sim = safe_similarity(anchor_df, path)
        rows.append({
            "entry_id": entry_id,
            "entry_type": "reference_submission",
            "entry_group": group,
            "run_id": None,
            "experiment_path": str(path.parent),
            "submission_name": path.name,
            "submission_path": str(path),
            "registered_in_registry": False,
            "ta_cv_mean": None,
            "ec_cv_mean": None,
            "drp_cv_mean": None,
            "cv_mean": None,
            "leaderboard_score": lb,
            "latest_proxy": None,
            "recommended_name": None,
            "drp_model": None,
            "drp_y_mode": None,
            "te_enabled": None,
            "drp_feature_count": None,
            "ta_corr_to_anchor": sim[0],
            "ta_mae_to_anchor": sim[1],
            "ec_corr_to_anchor": sim[2],
            "ec_mae_to_anchor": sim[3],
            "drp_corr_to_anchor": sim[4],
            "drp_mae_to_anchor": sim[5],
            "what_changed": note,
            "notes": note,
        })
    return rows


def main() -> int:
    anchor_df = pd.read_csv(ANCHOR_PATH)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT RUN_ID FROM EY_WQ.AUDIT.EXPERIMENT_REGISTRY")
        registry_ids = {row[0] for row in cur.fetchall()}

    rows = build_experiment_rows(anchor_df, registry_ids) + build_reference_rows(anchor_df)
    df = pd.DataFrame(rows)
    df = df.sort_values(["entry_type", "cv_mean", "drp_cv_mean"], ascending=[True, False, False], na_position="last")

    out_csv = ROOT / "experiments" / "master_project_state.csv"
    df.to_csv(out_csv, index=False)

    success, nchunks, nrows, _ = write_pandas(
        conn=conn,
        df=df,
        table_name="MASTER_PROJECT_STATE",
        database="EY_WQ",
        schema="AUDIT",
        auto_create_table=True,
        overwrite=True,
    )
    conn.close()

    if not success:
        print("write_pandas reported failure")
        return 1

    print(f"[OK] wrote {nrows} rows to EY_WQ.AUDIT.MASTER_PROJECT_STATE in {nchunks} chunk(s)")
    print(f"[OK] wrote local CSV: {out_csv}")
    print(df[["entry_id", "entry_type", "cv_mean", "drp_cv_mean", "leaderboard_score", "drp_model", "drp_y_mode", "te_enabled", "drp_feature_count", "drp_corr_to_anchor", "drp_mae_to_anchor"]].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
