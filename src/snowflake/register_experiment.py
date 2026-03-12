#!/usr/bin/env python
"""
Register a local experiment into Snowflake AUDIT.EXPERIMENT_REGISTRY.

Reads local run artifacts (config_snapshot.json and cv_report.json) and inserts
a compact audit record. This does not alter any training or submission path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _connect import get_connection


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-path", required=True, help="Path to experiments/exp_* folder")
    ap.add_argument("--database", default="EY_WQ")
    ap.add_argument("--warehouse", default="COMPUTE_WH")
    ap.add_argument("--role", default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    run_path = Path(args.run_path).resolve()
    run_id = run_path.name
    cfg_path = run_path / "config_snapshot.json"
    cv_path = run_path / "cv_report.json"

    if not cfg_path.exists() or not cv_path.exists():
        raise FileNotFoundError(f"Missing required run artifacts in {run_path}")

    cfg = read_json(cfg_path)
    cv = read_json(cv_path)

    ta = float(cv.get("Total Alkalinity", {}).get("mean", "nan"))
    ec = float(cv.get("Electrical Conductance", {}).get("mean", "nan"))
    drp = float(cv.get("Dissolved Reactive Phosphorus", {}).get("mean", "nan"))
    cv_mean = (ta + ec + drp) / 3.0

    drp_model = (
        cfg.get("model", {})
        .get("cv_by_target", {})
        .get("Dissolved Reactive Phosphorus", {})
        .get("name")
    )
    drp_mode = (
        cfg.get("targets", {})
        .get("y_mode_by_target", {})
        .get("Dissolved Reactive Phosphorus")
    )
    te_enabled = cfg.get("te", {}).get("enabled")

    conn = get_connection(warehouse=args.warehouse, database=args.database,
                          role=args.role)

    sql = """
    INSERT INTO EY_WQ.AUDIT.EXPERIMENT_REGISTRY (
        RUN_ID, RUN_PATH, CONFIG_PATH, CV_PATH,
        TARGET_TA_MEAN, TARGET_EC_MEAN, TARGET_DRP_MEAN, CV_MEAN,
        DRP_MODEL, DRP_Y_MODE, TE_ENABLED, NOTES
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS EY_WQ.AUDIT")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS EY_WQ.AUDIT.EXPERIMENT_REGISTRY (
                    REGISTERED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
                    RUN_ID STRING,
                    RUN_PATH STRING,
                    CONFIG_PATH STRING,
                    CV_PATH STRING,
                    TARGET_TA_MEAN FLOAT,
                    TARGET_EC_MEAN FLOAT,
                    TARGET_DRP_MEAN FLOAT,
                    CV_MEAN FLOAT,
                    DRP_MODEL STRING,
                    DRP_Y_MODE STRING,
                    TE_ENABLED BOOLEAN,
                    NOTES STRING
                )
                """
            )
            cur.execute(
                sql,
                (
                    run_id,
                    str(run_path),
                    str(cfg_path),
                    str(cv_path),
                    ta,
                    ec,
                    drp,
                    cv_mean,
                    drp_model,
                    drp_mode,
                    te_enabled,
                    args.notes,
                ),
            )
        print(f"[OK] Registered experiment: {run_id}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
