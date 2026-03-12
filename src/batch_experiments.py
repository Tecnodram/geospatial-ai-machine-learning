#!/usr/bin/env python
# coding: utf-8

"""
Batch experiment runner for DRP model selection, transforms, and regularization.

Grid:
- Models: ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
- DRP Transforms: none, log1p, sqrt
- Regularization: default, stronger

Total: 3 × 3 × 2 = 18 experiments
"""

import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yml"
SCRIPT_FILE = PROJECT_ROOT / "src" / "train_pipeline.py"
RESULTS_CSV = PROJECT_ROOT / "experiments" / "batch_results.csv"
VENV_PATH = PROJECT_ROOT / ".venv" / "Scripts" / "Activate.ps1"

# Experiment grid
experiments = [
    {
        "name": "baseline_sqrt_ET",
        "drp_focused_enabled": False,
        "include_interactions": False,
        "include_proxies": False,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
    },
    {
        "name": "interactions_sqrt_ET",
        "drp_focused_enabled": True,
        "include_interactions": True,
        "include_proxies": False,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
    },
    {
        "name": "full_features_sqrt_ET",
        "drp_focused_enabled": True,
        "include_interactions": True,
        "include_proxies": True,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
    },
    # Mild tuning for DRP ET
    {
        "name": "tune_ET_minleaf2",
        "drp_focused_enabled": True,
        "include_interactions": True,
        "include_proxies": True,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "name": "tune_ET_minleaf4",
        "drp_focused_enabled": True,
        "include_interactions": True,
        "include_proxies": True,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
    },
    {
        "name": "tune_ET_minleaf8",
        "drp_focused_enabled": True,
        "include_interactions": True,
        "include_proxies": True,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
    },
    {
        "name": "tune_ET_maxfeat05",
        "drp_focused_enabled": True,
        "include_interactions": True,
        "include_proxies": True,
        "min_samples_leaf": 3,
        "max_features": 0.5,
    },
]

import yaml

# Load base config
with open(CONFIG_FILE, 'r') as f:
    base_config = yaml.safe_load(f)

print(f"=== BATCH EXPERIMENT RUNNER ===")
print(f"Total experiments: {len(experiments)}")
print(f"DRP-focused feature experiments with sqrt transform + ExtraTreesRegressor")
print()

# Track results
results = []

# Run each experiment
for i, exp in enumerate(experiments, 1):
    exp_name = exp["name"]
    drp_focused_enabled = exp["drp_focused_enabled"]
    include_interactions = exp["include_interactions"]
    include_proxies = exp["include_proxies"]
    min_samples_leaf = exp["min_samples_leaf"]
    max_features = exp["max_features"]
    
    print(f"\n{'='*80}")
    print(f"[{i}/{len(experiments)}] Running: {exp_name}")
    print(f"  DRP focused features: {drp_focused_enabled}")
    print(f"  Include interactions: {include_interactions}")
    print(f"  Include proxies: {include_proxies}")
    print(f"  ET min_samples_leaf: {min_samples_leaf}")
    print(f"  ET max_features: {max_features}")
    print(f"{'='*80}")
    
    # Modify config
    config = base_config.copy()
    config["features"]["drp_focused"]["enabled"] = drp_focused_enabled
    config["features"]["drp_focused"]["include_interactions"] = include_interactions
    config["features"]["drp_focused"]["include_proxies"] = include_proxies
    
    # Update DRP params in cv_by_target and final_by_target
    if "model" in config and "cv_by_target" in config["model"]:
        if "Dissolved Reactive Phosphorus" in config["model"]["cv_by_target"]:
            if "params" in config["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]:
                config["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["params"]["min_samples_leaf"] = min_samples_leaf
                config["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["params"]["max_features"] = max_features
    
    if "model" in config and "final_by_target" in config["model"]:
        if "Dissolved Reactive Phosphorus" in config["model"]["final_by_target"]:
            if "params" in config["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]:
                config["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["params"]["min_samples_leaf"] = min_samples_leaf
                config["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["params"]["max_features"] = max_features
    
    # Save temp config
    temp_config_path = PROJECT_ROOT / f"temp_config_{exp_name}.yml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Build command using venv python executable
    python_exec = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    cmd = [
        python_exec,
        str(SCRIPT_FILE),
        "--config", str(temp_config_path),
        "--experiment_name", exp_name,
    ]
    
    # Run pipeline
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f"ERROR: Experiment {exp_name} failed! Return code {result.returncode}")
            print("STDERR:", stderr[-500:] if stderr else "No error details")
            results.append({
                "experiment_name": exp_name,
                "drp_focused_enabled": drp_focused_enabled,
                "include_interactions": include_interactions,
                "include_proxies": include_proxies,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "status": "FAILED",
                "ta_mean": None,
                "ec_mean": None,
                "drp_mean": None,
                "submission_file": None,
            })
            continue
        
        # Parse CV results from output
        ta_mean, ec_mean, drp_mean = None, None, None
        submission_file = None
        
        # Extract CV scores from output
        for line in stdout.split("\n"):
            if "Total Alkalinity" in line and "mean=" in line:
                # Example: "Total Alkalinity        | mean=0.3994 +/- 0.0721"
                try:
                    parts = line.split("mean=")
                    score_part = parts[1].split()[0]
                    ta_mean = float(score_part)
                except:
                    pass
            elif "Electrical Conductance" in line and "mean=" in line:
                try:
                    parts = line.split("mean=")
                    score_part = parts[1].split()[0]
                    ec_mean = float(score_part)
                except:
                    pass
            elif "Dissolved Reactive Phosphorus" in line and "mean=" in line and "DRP CV pruning" not in line:
                try:
                    parts = line.split("mean=")
                    score_part = parts[1].split()[0]
                    drp_mean = float(score_part)
                except:
                    pass
            elif "Saved:" in line and ".csv" in line:
                # Extract submission filename
                try:
                    submission_file = line.split("/")[-1].strip()
                except:
                    pass
        
        results.append({
            "experiment_name": exp_name,
            "drp_focused_enabled": drp_focused_enabled,
            "include_interactions": include_interactions,
            "include_proxies": include_proxies,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "status": "OK",
            "ta_mean": ta_mean,
            "ec_mean": ec_mean,
            "drp_mean": drp_mean,
            "submission_file": submission_file,
        })
        
        print(f"✓ {exp_name} completed")
        print(f"  TA:  {ta_mean:.4f}" if ta_mean else "  TA:  PARSE_ERROR")
        print(f"  EC:  {ec_mean:.4f}" if ec_mean else "  EC:  PARSE_ERROR")
        print(f"  DRP: {drp_mean:.4f}" if drp_mean else "  DRP: PARSE_ERROR")
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Experiment {exp_name} timed out!")
        results.append({
            "experiment_name": exp_name,
            "drp_focused_enabled": drp_focused_enabled,
            "include_interactions": include_interactions,
            "include_proxies": include_proxies,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "status": "TIMEOUT",
            "ta_mean": None,
            "ec_mean": None,
            "drp_mean": None,
            "submission_file": None,
        })
    except Exception as e:
        print(f"ERROR: Experiment {exp_name} crashed: {e}")
        results.append({
            "experiment_name": exp_name,
            "drp_focused_enabled": drp_focused_enabled,
            "include_interactions": include_interactions,
            "include_proxies": include_proxies,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "status": "ERROR",
            "ta_mean": None,
            "ec_mean": None,
            "drp_mean": None,
            "submission_file": None,
        })

# Generate results summary
print(f"\n{'='*80}")
print("EXPERIMENT BATCH COMPLETE")
print(f"{'='*80}\n")

# Save results to CSV
df_results = pd.DataFrame(results)

# Ensure experiments dir exists
os.makedirs(PROJECT_ROOT / "experiments", exist_ok=True)

df_results.to_csv(RESULTS_CSV, index=False)
print(f"Results saved to: {RESULTS_CSV}")

# Print summary table (sorted by DRP score descending)
print("\n=== EXPERIMENTS RANKED BY DRP MEAN CV R²===\n")

# Filter successful runs
df_success = df_results[df_results["status"] == "OK"].copy()

if len(df_success) > 0:
    df_success = df_success.sort_values("drp_mean", ascending=False, na_position="last")
    
    # Print header
    print(f"{'Rank':<5} {'Experiment':<25} {'DRP Feat':<10} {'Inter':<6} {'Proxy':<6} {'MinLeaf':<8} {'MaxFeat':<8} {'TA':<8} {'EC':<8} {'DRP':<8} {'Submission':<40}")
    print("-" * 160)
    
    for rank, (idx, row) in enumerate(df_success.iterrows(), 1):
        ta_str = f"{row['ta_mean']:.4f}" if row['ta_mean'] is not None else "N/A"
        ec_str = f"{row['ec_mean']:.4f}" if row['ec_mean'] is not None else "N/A"
        drp_str = f"{row['drp_mean']:.4f}" if row['drp_mean'] is not None else "N/A"
        sub_str = row['submission_file'] if row['submission_file'] else "N/A"
        drp_feat = "Yes" if row['drp_focused_enabled'] else "No"
        inter = "Yes" if row['include_interactions'] else "No"
        prox = "Yes" if row['include_proxies'] else "No"
        minleaf = row['min_samples_leaf']
        maxfeat = row['max_features']
        
        print(f"{rank:<5} {row['experiment_name']:<25} {drp_feat:<10} {inter:<6} {prox:<6} {minleaf:<8} {maxfeat:<8} {ta_str:<8} {ec_str:<8} {drp_str:<8} {sub_str:<40}")
else:
    print("No successful experiments!")

# Print failure summary
df_failed = df_results[df_results["status"] != "OK"]
if len(df_failed) > 0:
    print(f"\n=== FAILED EXPERIMENTS ({len(df_failed)}) ===\n")
    for idx, row in df_failed.iterrows():
        print(f"  {row['experiment_name']:<25} {row['status']:<10}")

print(f"\n=== SUMMARY ===")
print(f"  Successful: {len(df_success)}/{len(experiments)}")
print(f"  Failed:     {len(df_failed)}/{len(experiments)}")

if len(df_success) > 0:
    best_drp = df_success.iloc[0]
    print(f"\n  BEST DRP: {best_drp['experiment_name']} (R² = {best_drp['drp_mean']:.4f})")
    print(f"    DRP Features: {'Yes' if best_drp['drp_focused_enabled'] else 'No'}")
    print(f"    Interactions: {'Yes' if best_drp['include_interactions'] else 'No'}")
    print(f"    Proxies: {'Yes' if best_drp['include_proxies'] else 'No'}")
    print(f"    ET min_samples_leaf: {best_drp['min_samples_leaf']}")
    print(f"    ET max_features: {best_drp['max_features']}")
    print(f"    TA: {best_drp['ta_mean']:.4f}, EC: {best_drp['ec_mean']:.4f}")
    print(f"    Submission: {best_drp['submission_file']}")
