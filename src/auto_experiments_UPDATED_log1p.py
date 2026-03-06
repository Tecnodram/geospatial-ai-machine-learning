#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


TARGETS = [
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]

# Soporta tanto "±" como "+/-" (porque en train_pipeline.py ya imprimimos "+/-")
RE_CV = re.compile(
    r"(Total Alkalinity|Electrical Conductance|Dissolved Reactive Phosphorus)\s*\|\s*mean=([-\d\.]+)\s*(?:±|\+/-)\s*([-\d\.]+)\s*\|\s*folds=\[([^\]]+)\]"
)


def run(cmd, cwd: Path):
    """
    Ejecuta un comando y devuelve (returncode, stdout+stderr, duration_seconds).
    """
    import time

    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    dt = time.time() - t0
    return p.returncode, p.stdout, dt


def parse_cv(text: str) -> dict:
    """
    Extrae los CV means del bloque impreso por run_all/train_pipeline.
    Devuelve: {target: {"mean": float, "std": float, "folds": [..]}}
    """
    out = {}
    for m in RE_CV.finditer(text):
        target = m.group(1).strip()
        mean = float(m.group(2))
        std = float(m.group(3))
        folds_str = m.group(4).strip()
        folds = [float(x.strip()) for x in folds_str.split() if x.strip()]

        out[target] = {"mean": mean, "std": std, "folds": folds}
    return out


def overall_avg(cv: dict) -> float:
    """
    Promedio simple de means de los 3 targets si están presentes.
    Si falta alguno, devuelve nan.
    """
    if len(cv) != 3:
        return float("nan")
    return sum(cv[t]["mean"] for t in TARGETS) / 3.0


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def append_log(csv_path: Path, row: dict) -> None:
    """
    Escribe (append) una fila al CSV de experiments.
    IMPORTANTE: quoting=csv.QUOTE_MINIMAL para que campos con comas (e.g. te_grids="[0.05, 0.2]")
    se guarden bien y pandas pueda leer el archivo sin ParserError.
    """
    new_file = not csv_path.exists()

    columns = [
        "timestamp",
        "exp_id",
        "status",
        "grid_ta",
        "grid_ec",
        "grid_drp",
        "te_grids",
        "te_oof_grid",
        "y_drp",
        "cv_ta",
        "cv_ec",
        "cv_drp",
        "cv_avg",
        "config_path",
        "stdout_path",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in columns})


def make_variants(base_cfg: dict, n: int, seed: int):
    random.seed(seed)
    variants = []

    # Espacios de búsqueda (conservador, enfocado en DRP)
    te_grids_space = [
        [0.05, 0.2],
        [0.03, 0.15],
        [0.05, 0.15],
        [0.07, 0.2],
    ]
    te_oof_space = [0.15, 0.2, 0.25, 0.3]

    # Probamos grid DRP (es donde más inestabilidad hay)
    grid_drp_space = [0.05, 0.07, 0.09, 0.10, 0.12]

    for _ in range(n):
        cfg = json.loads(json.dumps(base_cfg))  # deep copy

        # Asegura cache (rápido)
        cfg["switches"]["build_dataset"] = False

        # grids TA/EC fijos (ya razonables); DRP explora
        cfg["cv"]["best_grid"]["Total Alkalinity"] = float(cfg["cv"]["best_grid"]["Total Alkalinity"])
        cfg["cv"]["best_grid"]["Electrical Conductance"] = float(cfg["cv"]["best_grid"]["Electrical Conductance"])
        cfg["cv"]["best_grid"]["Dissolved Reactive Phosphorus"] = float(random.choice(grid_drp_space))

        # target encoding (dos escalas)
        cfg["te"]["grids"] = random.choice(te_grids_space)
        cfg["te"]["oof_grid_for_groups"] = float(random.choice(te_oof_space))

        # y transform para DRP
        # - none o winsor (winsor suele ayudar por colas pesadas)
        cfg["targets"]["y_mode_by_target"]["Dissolved Reactive Phosphorus"] = random.choice(["none", "winsor", "log1p"])

        variants.append(cfg)

    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, default="config.yml", help="Config base para mutar (YAML).")
    ap.add_argument("--n", type=int, default=20, help="Número de experimentos.")
    ap.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad.")
    args = ap.parse_args()

    root = Path(".").resolve()
    base_cfg_path = (root / args.base_config).resolve()
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"No existe config base: {base_cfg_path}")

    base_cfg = load_yaml(base_cfg_path)

    exp_dir = root / "experiments"
    cfg_dir = exp_dir / "configs"
    out_dir = exp_dir / "out"
    exp_dir.mkdir(exist_ok=True)
    cfg_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    log_file = exp_dir / "experiments.csv"

    variants = make_variants(base_cfg, n=args.n, seed=args.seed)

    for i, cfg in enumerate(variants, start=1):
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i:02d}"
        cfg_path = cfg_dir / f"config_{exp_id}.yml"
        stdout_path = out_dir / f"{exp_id}_stdout.txt"

        save_yaml(cfg, cfg_path)

        cmd = [sys.executable, "src/run_all.py", "--config", str(cfg_path)]
        rc, combined_out, _ = run(cmd, root)

        stdout_path.write_text(combined_out, encoding="utf-8")

        cv = parse_cv(combined_out)
        avg = overall_avg(cv)

        # Status:
        # - Si parseamos los 3 targets, consideramos corrida "ok", aunque rc != 0 (dejamos rc visible).
        # - Si no parseamos 3, entonces sí es fail (rc y parsed ayudan a debug).
        status = (
            "ok"
            if (len(cv) == 3 and rc == 0)
            else (f"ok(rc={rc})" if len(cv) == 3 else f"fail(rc={rc}, parsed={len(cv)})")
        )

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exp_id": exp_id,
            "status": status,
            "grid_ta": cfg["cv"]["best_grid"]["Total Alkalinity"],
            "grid_ec": cfg["cv"]["best_grid"]["Electrical Conductance"],
            "grid_drp": cfg["cv"]["best_grid"]["Dissolved Reactive Phosphorus"],
            # te_grids tiene comas -> debe ir como string y el CSV debe quote-arlo
            "te_grids": json.dumps(cfg["te"]["grids"]),
            "te_oof_grid": cfg["te"]["oof_grid_for_groups"],
            "y_drp": cfg["targets"]["y_mode_by_target"]["Dissolved Reactive Phosphorus"],
            "cv_ta": cv.get("Total Alkalinity", {}).get("mean", ""),
            "cv_ec": cv.get("Electrical Conductance", {}).get("mean", ""),
            "cv_drp": cv.get("Dissolved Reactive Phosphorus", {}).get("mean", ""),
            "cv_avg": avg,
            "config_path": str(cfg_path),
            "stdout_path": str(stdout_path),
        }

        append_log(log_file, row)

        # imprime resumen como antes
        try:
            avg_print = float(avg)
        except Exception:
            avg_print = float("nan")

        print(
            f"[{i}/{args.n}] exp={exp_id} avg={avg_print:.4f} | "
            f"TA={row['cv_ta']} EC={row['cv_ec']} DRP={row['cv_drp']} | "
            f"y_drp={row['y_drp']} grid_drp={row['grid_drp']} te={cfg['te']['grids']} oof={row['te_oof_grid']}"
        )

    print("\nExperiments finished.")
    print("Results saved in experiments/experiments.csv")


if __name__ == "__main__":
    main()