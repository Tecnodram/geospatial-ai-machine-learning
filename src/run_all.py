#!/usr/bin/env python
# coding: utf-8

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # ✅ Asegura que todo se ejecute desde la RAÍZ del repo
    # (porque run_all.py está dentro de /src)
    repo_root = Path(__file__).resolve().parents[1]

    # 1) train pipeline
    subprocess.check_call(
        [sys.executable, str(repo_root / "src" / "train_pipeline.py"), "--config", args.config],
        cwd=str(repo_root),
    )

    # 2) batch blends + recommend
    subprocess.check_call(
        [sys.executable, str(repo_root / "src" / "batch_blends.py"), "--config", args.config],
        cwd=str(repo_root),
    )

if __name__ == "__main__":
    main()