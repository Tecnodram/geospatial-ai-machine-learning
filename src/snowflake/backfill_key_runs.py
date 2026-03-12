#!/usr/bin/env python
"""Backfill key historical experiments into EY_WQ.AUDIT.EXPERIMENT_REGISTRY."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _connect import get_connection

ROOT = Path(__file__).resolve().parents[2]
KEY_RUNS = [
    ("exp_20260307_003919", "top_historical_cv_et_te_none"),
    ("exp_20260307_004254", "top_historical_cv_et_te_winsor"),
    ("exp_20260309_235254", "high_similarity_et_te_winsor"),
    ("exp_20260310_015917", "historical_best_drp_cv_et_te_winsor"),
    ("exp_20260310_001701", "best_drp_cv_et_te_winsor_recent"),
    ("exp_20260311_104847", "highest_mean_cv_et_sqrt_no_te"),
    ("exp_20260311_133241", "recent_hgb_no_te_none"),
    ("exp_20260311_134704", "recent_best_drp_et_te_none"),
    ("exp_20260311_135439", "recent_hgb_te_none"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_record(run_path: Path, notes: str) -> tuple:
    cfg = read_json(run_path / "config_snapshot.json")
    cv = read_json(run_path / "cv_report.json")

    ta = float(cv.get("Total Alkalinity", {}).get("mean", "nan"))
    ec = float(cv.get("Electrical Conductance", {}).get("mean", "nan"))
    drp = float(cv.get("Dissolved Reactive Phosphorus", {}).get("mean", "nan"))
    cv_mean = (ta + ec + drp) / 3.0

    drp_model = (
        cfg.get("model", {})
        .get("cv_by_target", {})
        .get("Dissolved Reactive Phosphorus", {})
        .get("name")
        or "ExtraTreesRegressor"
    )
    drp_mode = (
        cfg.get("targets", {})
        .get("y_mode_by_target", {})
        .get("Dissolved Reactive Phosphorus")
        or cfg.get("targets", {}).get("y_mode")
    )
    te_enabled = cfg.get("te", {}).get("enabled")

    return (
        run_path.name,
        str(run_path),
        str(run_path / "config_snapshot.json"),
        str(run_path / "cv_report.json"),
        ta,
        ec,
        drp,
        cv_mean,
        drp_model,
        drp_mode,
        te_enabled,
        notes,
    )


def main() -> int:
    conn = get_connection()
    inserted = 0
    skipped = 0
    missing = []

    with conn.cursor() as cur:
        cur.execute("SELECT RUN_ID FROM EY_WQ.AUDIT.EXPERIMENT_REGISTRY")
        existing = {row[0] for row in cur.fetchall()}

        insert_sql = """
        INSERT INTO EY_WQ.AUDIT.EXPERIMENT_REGISTRY (
            RUN_ID, RUN_PATH, CONFIG_PATH, CV_PATH,
            TARGET_TA_MEAN, TARGET_EC_MEAN, TARGET_DRP_MEAN, CV_MEAN,
            DRP_MODEL, DRP_Y_MODE, TE_ENABLED, NOTES
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for run_id, note in KEY_RUNS:
            run_path = ROOT / "experiments" / run_id
            if not (run_path / "config_snapshot.json").exists() or not (run_path / "cv_report.json").exists():
                missing.append(run_id)
                continue
            if run_id in existing:
                print(f"[SKIP] {run_id} already registered")
                skipped += 1
                continue
            record = build_record(run_path, note)
            cur.execute(insert_sql, record)
            print(f"[OK] inserted {run_id} | note={note}")
            inserted += 1

    conn.close()
    print()
    print(f"inserted={inserted} skipped={skipped} missing={len(missing)}")
    if missing:
        print("missing_runs:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
