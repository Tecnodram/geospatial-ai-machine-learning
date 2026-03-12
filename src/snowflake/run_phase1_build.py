#!/usr/bin/env python
"""
Phase 1 — Execute build_feature_tables.sql against Snowflake.
Creates EY_WQ.MART.TRAIN_MODELREADY, EY_WQ.MART.VALID_MODELREADY,
and EY_WQ.AUDIT.EXPERIMENT_REGISTRY.
"""

import sys
from pathlib import Path

# Allow direct invocation from src/snowflake/
sys.path.insert(0, str(Path(__file__).parent))
from _connect import get_connection

SQL_PATH = Path(__file__).parent / "build_feature_tables.sql"


def main() -> int:
    conn = get_connection(warehouse="COMPUTE_WH", database="EY_WQ")

    raw_sql = SQL_PATH.read_text(encoding="utf-8")

    # Split on semicolons; skip empty or comment-only blocks
    statements = [
        s.strip()
        for s in raw_sql.split(";")
        if s.strip() and not all(
            line.startswith("--") or line == "" for line in s.strip().splitlines()
        )
    ]

    print(f"[Phase 1] {len(statements)} SQL statement(s) to execute\n")

    errors = []
    with conn.cursor() as cur:
        for i, stmt in enumerate(statements, 1):
            preview = stmt.replace("\n", " ")[:100]
            try:
                cur.execute(stmt)
                print(f"  [{i:02d}] OK  — {preview}")
            except Exception as exc:
                print(f"  [{i:02d}] ERR — {preview}")
                print(f"         -> {exc}")
                errors.append((i, str(exc)))

    conn.close()
    print()

    if errors:
        print(f"[Phase 1] FINISHED WITH {len(errors)} ERROR(S)")
        for idx, msg in errors:
            print(f"  Stmt {idx}: {msg}")
        return 1
    else:
        print("[Phase 1] ALL STATEMENTS EXECUTED SUCCESSFULLY — MART + AUDIT tables are ready.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
