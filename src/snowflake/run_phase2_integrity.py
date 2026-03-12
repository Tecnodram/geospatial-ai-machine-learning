#!/usr/bin/env python
"""
Phase 2 — Snowflake integrity checks for MART tables.

Runs row counts, duplicate key checks, and null key checks on
EY_WQ.MART.TRAIN_MODELREADY and EY_WQ.MART.VALID_MODELREADY.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct invocation from src/snowflake/
sys.path.insert(0, str(Path(__file__).parent))
from _connect import get_connection


CHECKS = {
    "train_rows":        "SELECT COUNT(*) FROM EY_WQ.MART.TRAIN_MODELREADY",
    "valid_rows":        "SELECT COUNT(*) FROM EY_WQ.MART.VALID_MODELREADY",
    "train_dup_keys":    """
        SELECT COUNT(*) FROM (
            SELECT LATITUDE, LONGITUDE, SAMPLE_DATE, COUNT(*) c
            FROM EY_WQ.MART.TRAIN_MODELREADY
            GROUP BY 1,2,3 HAVING c > 1
        )""",
    "valid_dup_keys":    """
        SELECT COUNT(*) FROM (
            SELECT LATITUDE, LONGITUDE, SAMPLE_DATE, COUNT(*) c
            FROM EY_WQ.MART.VALID_MODELREADY
            GROUP BY 1,2,3 HAVING c > 1
        )""",
    "train_null_keys":   """
        SELECT COUNT(*) FROM EY_WQ.MART.TRAIN_MODELREADY
        WHERE LATITUDE IS NULL OR LONGITUDE IS NULL OR SAMPLE_DATE IS NULL""",
    "valid_null_keys":   """
        SELECT COUNT(*) FROM EY_WQ.MART.VALID_MODELREADY
        WHERE LATITUDE IS NULL OR LONGITUDE IS NULL OR SAMPLE_DATE IS NULL""",
}

PLAUSIBLE_RANGES = {
    "train_rows":     (500, 50_000),
    "valid_rows":     (100, 10_000),
    "train_dup_keys": (0, 0),
    "valid_dup_keys": (0, 0),
    "train_null_keys":(0, 0),
    "valid_null_keys":(0, 0),
}


def interpret(name: str, value: int) -> str:
    lo, hi = PLAUSIBLE_RANGES[name]
    if lo <= value <= hi:
        return "OK"
    elif value == 0 and lo > 0:
        return "WARN — unexpectedly empty"
    elif value > hi:
        return "WARN — higher than expected"
    else:
        return f"WARN — expected [{lo}, {hi}]"


def main() -> int:
    conn = get_connection()
    results: dict[str, int] = {}
    errors: list[tuple[str, str]] = []

    print("[Phase 2] Running integrity checks...\n")
    with conn.cursor() as cur:
        for name, sql in CHECKS.items():
            try:
                cur.execute(sql)
                val = cur.fetchone()[0]
                results[name] = val
                status = interpret(name, val)
                print(f"  {name:<22} = {val:>8,}   [{status}]")
            except Exception as exc:
                print(f"  {name:<22} = ERROR: {exc}")
                errors.append((name, str(exc)))

    conn.close()
    print()

    # Interpretation
    print("── Interpretation ────────────────────────────────────────────")
    if errors:
        print(f"  {len(errors)} check(s) failed with errors — see above.")

    tr = results.get("train_rows", 0)
    vr = results.get("valid_rows", 0)
    td = results.get("train_dup_keys", -1)
    vd = results.get("valid_dup_keys", -1)
    tn = results.get("train_null_keys", -1)
    vn = results.get("valid_null_keys", -1)

    if tr > 0:
        print(f"  ✓ Train MART has {tr:,} rows — JOIN produced records.")
    else:
        print("  ✗ Train MART is EMPTY — INNER JOIN eliminated all rows. Check key column names and data types in RAW tables.")

    if vr > 0:
        print(f"  ✓ Valid MART has {vr:,} rows — submission template expanded.")
    else:
        print("  ✗ Valid MART is EMPTY — LEFT JOIN on submission template failed.")

    if td == 0:
        print("  ✓ No duplicate train keys.")
    elif td > 0:
        print(f"  ✗ {td:,} duplicate (LAT,LON,DATE) key group(s) in train MART — investigate RAW source.")

    if vd == 0:
        print("  ✓ No duplicate valid keys.")
    elif vd > 0:
        print(f"  ✗ {vd:,} duplicate (LAT,LON,DATE) key group(s) in valid MART.")

    if tn == 0:
        print("  ✓ No null train keys.")
    elif tn > 0:
        print(f"  ✗ {tn:,} row(s) with null key(s) in train MART.")

    if vn == 0:
        print("  ✓ No null valid keys.")
    elif vn > 0:
        print(f"  ✗ {vn:,} row(s) with null key(s) in valid MART.")

    print()
    if not errors and td == 0 and vd == 0 and tn == 0 and vn == 0 and tr > 0 and vr > 0:
        verdict = "PASS"
    elif errors or tr == 0 or vr == 0:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"

    print(f"[Phase 2] Integrity verdict: {verdict}")

    # Write results to file for audit
    import json, datetime
    out = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "results": results,
        "verdict": verdict,
        "errors": errors,
    }
    out_path = Path(__file__).parent.parent.parent / "experiments" / "snowflake_integrity_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Results saved to: {out_path}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
