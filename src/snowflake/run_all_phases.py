#!/usr/bin/env python
"""
Unified Snowflake integration runner — Phases 1-3.

Runs in order:
  Phase 1: Build MART + AUDIT tables  (build_feature_tables.sql)
  Phase 2: Integrity checks
  Phase 3: Register most-recent local experiment

Usage:
  cd C:/Projects/ey-water-quality-2026
  python src/snowflake/run_all_phases.py

Credentials: place in .snowflake.env at project root.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PYTHON = sys.executable
SNOWFLAKE_DIR = Path(__file__).parent


def run_phase(label: str, script: Path, extra_args: list[str] | None = None) -> int:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    cmd = [PYTHON, str(script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=str(SNOWFLAKE_DIR))
    return result.returncode


def find_most_recent_exp() -> Path | None:
    exp_dir = ROOT / "experiments"
    candidates = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and (d / "cv_report.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    print("EY Water Quality — Snowflake Integration Runner")
    print(f"Project root: {ROOT}\n")

    results: dict[str, str] = {}

    # Phase 1
    rc = run_phase("PHASE 1 — Build MART + AUDIT Tables", SNOWFLAKE_DIR / "run_phase1_build.py")
    results["phase1_build"] = "PASS" if rc == 0 else "FAIL"
    if rc != 0:
        print("\n[ABORT] Phase 1 failed. Fix SQL errors before continuing.")
        _print_summary(results)
        sys.exit(1)

    # Phase 2
    rc = run_phase("PHASE 2 — Integrity Checks", SNOWFLAKE_DIR / "run_phase2_integrity.py")
    results["phase2_integrity"] = "PASS" if rc == 0 else "PARTIAL"
    if rc != 0:
        print("\n[WARN] Phase 2 reported issues — review above before proceeding.")

    # Phase 3
    exp_path = find_most_recent_exp()
    if exp_path:
        rc = run_phase(
            f"PHASE 3 — Register Experiment: {exp_path.name}",
            SNOWFLAKE_DIR / "register_experiment.py",
            extra_args=["--run-path", str(exp_path), "--notes", "auto_registered_by_run_all_phases"],
        )
        results["phase3_register"] = "PASS" if rc == 0 else "FAIL"
    else:
        print("\n[SKIP] Phase 3: no experiment with cv_report.json found in experiments/")
        results["phase3_register"] = "SKIP"

    _print_summary(results)

    overall_ok = all(v in ("PASS", "SKIP", "PARTIAL") for v in results.values())
    sys.exit(0 if overall_ok else 1)


def _print_summary(results: dict[str, str]) -> None:
    print(f"\n{'='*60}")
    print("  EXECUTION SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:<30} {v}")
    print()


if __name__ == "__main__":
    main()
