#!/usr/bin/env python
# coding: utf-8

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import shutil

from tqdm import tqdm


def _derive_exp_id_from_config(config_path: str) -> str:
    """
    Intenta derivar exp_id desde el nombre del YAML:
      config_20260305_115009_06.yml -> 20260305_115009_06
    Si no se puede, usa el stem completo.
    """
    p = Path(config_path)
    stem = p.stem  # e.g. "config_20260305_115009_06"
    if stem.startswith("config_"):
        return stem.replace("config_", "", 1)
    return stem


def _load_out_dir_from_config(repo_root: Path, config_path: str) -> Path:
    """
    Lee cfg['io']['out_dir'] si existe, si no usa 'submissions_batch'.
    No rompe si falta yaml (pero en tu env sí existe).
    """
    try:
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        out_dir = cfg.get("io", {}).get("out_dir", "submissions_batch")
        return (repo_root / out_dir).resolve()
    except Exception:
        return (repo_root / "submissions_batch").resolve()


def _archive_submissions(out_dir: Path, archive_root: Path, run_tag: str) -> list[Path]:
    """
    Copia todos los submission_*.csv del out_dir a un subfolder archive,
    agregando sufijo run_tag para que no se sobreescriban.
    Deja intactos los archivos originales (importantísimo para batch_blends).
    """
    archive_root.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in sorted(out_dir.glob("submission_*.csv")):
        # e.g. submission_V5_2_OOFTE_fixkeys_as_is.csv
        dst_name = f"{src.stem}__{run_tag}{src.suffix}"
        dst = archive_root / dst_name
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # Asegura ejecución desde la RAÍZ del repo
    repo_root = Path(__file__).resolve().parents[1]

    # Identificadores únicos para no pisar submissions
    exp_id = _derive_exp_id_from_config(args.config)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{exp_id}__{ts}"

    out_dir = _load_out_dir_from_config(repo_root, args.config)

    steps = [
        ("Train pipeline", [sys.executable, str(repo_root / "src" / "train_pipeline.py"), "--config", args.config]),
        ("Batch blends",   [sys.executable, str(repo_root / "src" / "batch_blends.py"), "--config", args.config]),
    ]

    with tqdm(total=len(steps), desc="EY Pipeline", unit="step") as pbar:
        for name, cmd in steps:
            pbar.set_postfix_str(name)
            subprocess.check_call(cmd, cwd=str(repo_root))
            pbar.update(1)

    # === ARCHIVE: copia submissions con nombre único ===
    archive_dir = out_dir / "archive"
    copied = _archive_submissions(out_dir=out_dir, archive_root=archive_dir, run_tag=run_tag)

    print("\n================================================================================")
    print("ARCHIVE SUBMISSIONS (no sobrescribe):")
    print(f"Out dir:      {out_dir}")
    print(f"Archive dir:  {archive_dir}")
    print(f"Run tag:      {run_tag}")
    if copied:
        for p in copied:
            print(f"✅ Copied -> {p}")
    else:
        print("⚠️ No encontré archivos submission_*.csv para archivar.")
    print("================================================================================\n")


if __name__ == "__main__":
    main()