#!/usr/bin/env python
"""Run feature-signal scaffold SQL for FEAT_SIG schema and versioned marts."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _connect import get_connection

ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = Path(__file__).parent / "build_feature_signal_scaffold.sql"


def split_statements(sql_text: str) -> list[str]:
    statements = []
    for raw in sql_text.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        lines = [ln.strip() for ln in stmt.splitlines() if ln.strip()]
        if lines and all(ln.startswith("--") for ln in lines):
            continue
        statements.append(stmt)
    return statements


def main() -> int:
    conn = get_connection(warehouse="COMPUTE_WH", database="EY_WQ")
    sql_text = SQL_PATH.read_text(encoding="utf-8")
    statements = split_statements(sql_text)
    sql_hash = hashlib.sha256(sql_text.encode("utf-8")).hexdigest()

    executed = 0
    errors = []
    with conn.cursor() as cur:
        for i, stmt in enumerate(statements, start=1):
            preview = stmt.replace("\n", " ")[:110]
            try:
                cur.execute(stmt)
                executed += 1
                print(f"[{i:02d}] OK  {preview}")
            except Exception as exc:
                print(f"[{i:02d}] ERR {preview}")
                print(f"      -> {exc}")
                errors.append((i, str(exc)))

        # Registry entry for scaffold run
        try:
            cur.execute(
                """
                INSERT INTO EY_WQ.AUDIT.FEATURE_BUILD_REGISTRY (
                    FEATURE_VERSION, BUILD_STEP, OUTPUT_TABLE, SOURCE_TABLES, SQL_HASH, ROW_COUNT, NOTES
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    "SIG_V1_0",
                    "scaffold_build",
                    "FEAT_SIG.*",
                    "RAW.*, MART.TRAIN_MODELREADY, MART.VALID_MODELREADY",
                    sql_hash,
                    executed,
                    "initial signal-family scaffold build",
                ),
            )
            print("[REGISTRY] Feature build registry updated")
        except Exception as exc:
            print(f"[REGISTRY] WARN could not write registry row: {exc}")

    conn.close()

    if errors:
        print(f"\nScaffold finished with {len(errors)} error(s).")
        return 1

    print("\nScaffold build completed successfully.")
    print(f"Executed statements: {executed}")
    print(f"SQL hash: {sql_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
