#!/usr/bin/env python
"""
Minimal Snowflake loader for EY Water Quality Challenge assets.

Loads core CSVs into Snowflake RAW schema with a stable naming convention,
without changing the existing local training/submission workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

sys.path.insert(0, str(Path(__file__).parent))
from _connect import get_connection


DATASETS = {
    "water_quality_training_dataset": "data/raw/water_quality_training_dataset.csv",
    "landsat_features_training": "data/raw/landsat_features_training.csv",
    "terraclimate_features_training": "data/raw/terraclimate_features_training.csv",
    "landsat_features_validation": "data/raw/landsat_features_validation.csv",
    "terraclimate_features_validation": "data/raw/terraclimate_features_validation.csv",
    "submission_template": "data/raw/submission_template.csv",
    "chirps_features_training": "data/external/chirps_features_training.csv",
    "chirps_features_validation": "data/external/chirps_features_validation.csv",
    "external_geofeatures_plus_hydro_v2": "data/external_geofeatures_plus_hydro_v2.csv",
    "external_geofeatures_hydro_v2": "data/external_geofeatures_hydro_v2.csv",
    "external_geofeatures": "data/external_geofeatures.csv",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep column names Snowflake-friendly and deterministic.
    out = df.copy()
    out.columns = [
        c.strip().upper().replace(" ", "_").replace("-", "_").replace("/", "_") for c in out.columns
    ]
    return out


def load_dataset(conn, root: Path, db: str, schema: str, table_name: str, rel_path: str) -> None:
    csv_path = root / rel_path
    if not csv_path.exists():
        print(f"[WARN] Missing file, skipping: {csv_path}")
        return

    print(f"[INFO] Loading {csv_path} -> {db}.{schema}.{table_name}")
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    success, nchunks, nrows, _ = write_pandas(
        conn=conn,
        df=df,
        table_name=table_name,
        database=db,
        schema=schema,
        auto_create_table=True,
        overwrite=True,
    )

    if success:
        print(f"[OK] {table_name}: rows={nrows}, chunks={nchunks}")
    else:
        print(f"[ERROR] write_pandas returned success=False for {table_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repository root path")
    ap.add_argument("--database", default="EY_WQ", help="Snowflake database")
    ap.add_argument("--warehouse", default="COMPUTE_WH", help="Snowflake warehouse")
    ap.add_argument("--role", default=None, help="Snowflake role")
    ap.add_argument("--raw-schema", default="RAW", help="Target raw schema")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    conn = get_connection(warehouse=args.warehouse, database=args.database,
                          role=args.role)

    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {args.database}")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {args.database}.{args.raw_schema}")

        for name, rel in DATASETS.items():
            load_dataset(
                conn=conn,
                root=root,
                db=args.database,
                schema=args.raw_schema,
                table_name=name.upper(),
                rel_path=rel,
            )

        print("\n[DONE] RAW loading completed.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
