#!/usr/bin/env python
# coding: utf-8

"""
DRP Stabilization Experiments

Tests different transforms and models for DRP.
"""

import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
import yaml

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yml"
SCRIPT_FILE = PROJECT_ROOT / "src" / "train_pipeline.py"
VENV_PATH = PROJECT_ROOT / ".venv" / "Scripts" / "Activate.ps1"

# Load base config
with open(CONFIG_FILE, 'r') as f:
    base_config = yaml.safe_load(f)

# Experiments
experiments = [
    # Baseline
    {"name": "baseline_sqrt_ET", "drp_transform": "sqrt", "drp_model": "ExtraTreesRegressor"},
    # Transforms
    {"name": "none_ET", "drp_transform": "none", "drp_model": "ExtraTreesRegressor"},
    {"name": "winsor_ET", "drp_transform": "winsor", "drp_model": "ExtraTreesRegressor"},
    {"name": "log1p_ET", "drp_transform": "log1p", "drp_model": "ExtraTreesRegressor"},
    # Models with sqrt
    {"name": "sqrt_RF", "drp_transform": "sqrt", "drp_model": "RandomForestRegressor"},
    {"name": "sqrt_HGB_poisson", "drp_transform": "sqrt", "drp_model": "HistGradientBoostingRegressor"},
]

print("=== DRP STABILIZATION EXPERIMENTS ===")
print(f"Total experiments: {len(experiments)}")

results = []

for i, exp in enumerate(experiments, 1):
    exp_name = exp["name"]
    drp_transform = exp["drp_transform"]
    drp_model = exp["drp_model"]

    print(f"\n[{i}/{len(experiments)}] {exp_name}")
    print(f"  DRP transform: {drp_transform}")
    print(f"  DRP model: {drp_model}")

    # Modify config
    config = base_config.copy()
    config["targets"]["y_mode_by_target"]["Dissolved Reactive Phosphorus"] = drp_transform
    config["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["name"] = drp_model
    config["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["name"] = drp_model

    # Set params
    if drp_model == "ExtraTreesRegressor":
        params = {"n_estimators": 1400, "min_samples_leaf": 3, "max_features": "sqrt", "n_jobs": -1, "random_state": 42}
    elif drp_model == "RandomForestRegressor":
        params = {"n_estimators": 1000, "min_samples_leaf": 3, "max_features": "sqrt", "n_jobs": -1, "random_state": 42}
    elif drp_model == "HistGradientBoostingRegressor":
        params = {"max_iter": 1000, "learning_rate": 0.1, "max_depth": 10, "min_samples_leaf": 20, "random_state": 42, "loss": "poisson"}

    config["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["params"] = params
    config["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["params"] = params

    # Save temp config
    temp_config_path = PROJECT_ROOT / f"temp_config_{exp_name}.yml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # Run
    python_exec = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    cmd = [python_exec, str(SCRIPT_FILE), "--config", str(temp_config_path), "--experiment_name", exp_name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        stdout = result.stdout

        if result.returncode != 0:
            print(f"ERROR: {exp_name} failed")
            results.append({"experiment": exp_name, "status": "FAILED"})
            continue

        # Parse DRP results
        drp_mean = None
        drp_std = None
        drp_folds = []

        for line in stdout.split("\n"):
            if "Dissolved Reactive Phosphorus" in line and "mean=" in line:
                parts = line.split("mean=")[1].split()
                drp_mean = float(parts[0])
                drp_std = float(parts[2])
            elif "Dissolved Reactive Phosphorus" in line and "folds=" in line:
                fold_str = line.split("folds=[")[1].split("]")[0]
                drp_folds = [float(x.strip()) for x in fold_str.split()]

        print(f"  DRP: {drp_mean:.4f} +/- {drp_std:.4f}")
        print(f"  Folds: {drp_folds}")

        results.append({
            "experiment": exp_name,
            "drp_transform": drp_transform,
            "drp_model": drp_model,
            "drp_mean": drp_mean,
            "drp_std": drp_std,
            "drp_folds": drp_folds,
            "status": "SUCCESS"
        })

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {exp_name}")
        results.append({"experiment": exp_name, "status": "TIMEOUT"})

    # Clean up
    if temp_config_path.exists():
        temp_config_path.unlink()

# Save results
results_df = pd.DataFrame(results)
results_csv = PROJECT_ROOT / "experiments" / "drp_stabilization_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"\nResults saved to: {results_csv}")

# Summary
print("\n=== DRP STABILIZATION SUMMARY ===")
for r in results:
    if r["status"] == "SUCCESS":
        print(f"{r['experiment']}: DRP {r['drp_mean']:.4f} +/- {r['drp_std']:.4f}")

print("\nComplete.")