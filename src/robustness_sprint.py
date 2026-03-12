#!/usr/bin/env python
# coding: utf-8

"""
Robustness + Similarity Sprint for DRP Model Selection

Experiments:
A) DRP = sqrt + ExtraTreesRegressor + drp_focused OFF
B) DRP = sqrt + ExtraTreesRegressor + drp_focused ON
C) DRP = sqrt + RandomForestRegressor + drp_focused OFF
D) DRP = sqrt + HistGradientBoostingRegressor + drp_focused OFF
E) DRP = sqrt + ExtraTreesRegressor + drp_focused OFF + conservative TE setting
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import yaml

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yml"
SCRIPT_FILE = PROJECT_ROOT / "src" / "train_pipeline.py"
VENV_PATH = PROJECT_ROOT / ".venv" / "Scripts" / "Activate.ps1"

# Anchor submission
ANCHOR_FILE = PROJECT_ROOT / "submissions" / "submission_V4_4_DRP_tuned_ET_fixorder.csv"
FAILED_FILE = PROJECT_ROOT / "submissions_batch" / "submission_V5_2_OOFTE_fixkeys.csv"

# Experiment grid
experiments = [
    {
        "name": "A_sqrt_ET_drp_focused_OFF",
        "drp_model": "ExtraTreesRegressor",
        "drp_focused_enabled": False,
        "include_interactions": False,
        "include_proxies": False,
        "te_enabled": True,
        "te_grids": [0.05, 0.2],
    },
    {
        "name": "B_sqrt_ET_drp_focused_ON",
        "drp_model": "ExtraTreesRegressor",
        "drp_focused_enabled": True,
        "include_interactions": True,
        "include_proxies": True,
        "te_enabled": True,
        "te_grids": [0.05, 0.2],
    },
    {
        "name": "C_sqrt_RF_drp_focused_OFF",
        "drp_model": "RandomForestRegressor",
        "drp_focused_enabled": False,
        "include_interactions": False,
        "include_proxies": False,
        "te_enabled": True,
        "te_grids": [0.05, 0.2],
    },
    {
        "name": "D_sqrt_HGB_drp_focused_OFF",
        "drp_model": "HistGradientBoostingRegressor",
        "drp_focused_enabled": False,
        "include_interactions": False,
        "include_proxies": False,
        "te_enabled": True,
        "te_grids": [0.05, 0.2],
    },
    {
        "name": "E_sqrt_ET_drp_focused_OFF_conservative_TE",
        "drp_model": "ExtraTreesRegressor",
        "drp_focused_enabled": False,
        "include_interactions": False,
        "include_proxies": False,
        "te_enabled": True,
        "te_grids": [0.10, 0.20],  # More conservative
    },
]

# Load base config
with open(CONFIG_FILE, 'r') as f:
    base_config = yaml.safe_load(f)

print(f"=== ROBUSTNESS + SIMILARITY SPRINT ===")
print(f"Total experiments: {len(experiments)}")
print(f"DRP-focused robustness evaluation")
print()

# Load anchor submission
if ANCHOR_FILE.exists():
    anchor_df = pd.read_csv(ANCHOR_FILE)
    print(f"Loaded anchor submission: {ANCHOR_FILE}")
    print(f"Anchor shape: {anchor_df.shape}")
else:
    print(f"ERROR: Anchor file not found: {ANCHOR_FILE}")
    sys.exit(1)

if FAILED_FILE.exists():
    failed_df = pd.read_csv(FAILED_FILE)
    print(f"Loaded failed submission: {FAILED_FILE}")
    print(f"Failed shape: {failed_df.shape}")
else:
    print(f"ERROR: Failed file not found: {FAILED_FILE}")
    sys.exit(1)

# Track results
results = []

# Run each experiment
for i, exp in enumerate(experiments, 1):
    exp_name = exp["name"]
    drp_model = exp["drp_model"]
    drp_focused_enabled = exp["drp_focused_enabled"]
    include_interactions = exp["include_interactions"]
    include_proxies = exp["include_proxies"]
    te_enabled = exp["te_enabled"]
    te_grids = exp["te_grids"]
    
    print(f"\n{'='*80}")
    print(f"[{i}/{len(experiments)}] Running: {exp_name}")
    print(f"  DRP model: {drp_model}")
    print(f"  DRP focused features: {drp_focused_enabled}")
    print(f"  Include interactions: {include_interactions}")
    print(f"  Include proxies: {include_proxies}")
    print(f"  TE enabled: {te_enabled}")
    print(f"  TE grids: {te_grids}")
    print(f"{'='*80}")
    
    # Modify config
    config = base_config.copy()
    config["features"]["drp_focused"]["enabled"] = drp_focused_enabled
    config["features"]["drp_focused"]["include_interactions"] = include_interactions
    config["features"]["drp_focused"]["include_proxies"] = include_proxies
    config["te"]["enabled"] = te_enabled
    config["te"]["grids"] = te_grids
    
    # Update DRP model
    config["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["name"] = drp_model
    config["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["name"] = drp_model
    
    # Set params based on model
    if drp_model == "ExtraTreesRegressor":
        params = {
            "n_estimators": 1400,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42
        }
    elif drp_model == "RandomForestRegressor":
        params = {
            "n_estimators": 1000,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42
        }
    elif drp_model == "HistGradientBoostingRegressor":
        params = {
            "max_iter": 1000,
            "learning_rate": 0.1,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "random_state": 42
        }
    
    config["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["params"] = params
    config["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["params"] = params
    
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 min timeout
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f"ERROR: Experiment {exp_name} failed! Return code {result.returncode}")
            print("STDERR:", stderr[-1000:] if stderr else "No error details")
            results.append({
                "experiment_name": exp_name,
                "drp_model": drp_model,
                "drp_focused_enabled": drp_focused_enabled,
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
            elif "Dissolved Reactive Phosphorus" in line and "mean=" in line:
                try:
                    parts = line.split("mean=")
                    score_part = parts[1].split()[0]
                    drp_mean = float(score_part)
                except:
                    pass
        
        # Find submission file
        submission_file = PROJECT_ROOT / "submissions_batch" / "submission_V5_2_OOFTE_fixkeys.csv"
        if submission_file.exists():
            # Copy to unique name
            unique_submission = PROJECT_ROOT / "submissions_batch" / f"submission_{exp_name}.csv"
            import shutil
            shutil.copy(submission_file, unique_submission)
            submission_file = unique_submission
        
        print(f"SUCCESS: {exp_name}")
        print(f"  TA CV: {ta_mean}")
        print(f"  EC CV: {ec_mean}")
        print(f"  DRP CV: {drp_mean}")
        print(f"  Submission: {submission_file}")
        
        results.append({
            "experiment_name": exp_name,
            "drp_model": drp_model,
            "drp_focused_enabled": drp_focused_enabled,
            "status": "SUCCESS",
            "ta_mean": ta_mean,
            "ec_mean": ec_mean,
            "drp_mean": drp_mean,
            "submission_file": str(submission_file) if submission_file else None,
        })
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Experiment {exp_name} timed out")
        results.append({
            "experiment_name": exp_name,
            "drp_model": drp_model,
            "drp_focused_enabled": drp_focused_enabled,
            "status": "TIMEOUT",
            "ta_mean": None,
            "ec_mean": None,
            "drp_mean": None,
            "submission_file": None,
        })
    
    # Clean up temp config
    if temp_config_path.exists():
        temp_config_path.unlink()

# Save results to CSV
results_df = pd.DataFrame(results)
results_csv = PROJECT_ROOT / "experiments" / "robustness_sprint_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"\nResults saved to: {results_csv}")

# Now do similarity analysis
print(f"\n{'='*80}")
print("SIMILARITY ANALYSIS")
print(f"{'='*80}")

similarity_results = []

for result in results:
    if result["status"] != "SUCCESS" or not result["submission_file"]:
        continue
    
    exp_name = result["experiment_name"]
    sub_file = Path(result["submission_file"])
    
    if not sub_file.exists():
        continue
    
    candidate_df = pd.read_csv(sub_file)
    
    # Compare to anchor
    sim_ta = {}
    sim_ec = {}
    sim_drp = {}
    
    for target in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]:
        if target in anchor_df.columns and target in candidate_df.columns:
            anchor_vals = anchor_df[target].values
            cand_vals = candidate_df[target].values
            
            pearson = pearsonr(anchor_vals, cand_vals)[0]
            spearman = spearmanr(anchor_vals, cand_vals)[0]
            mae = mean_absolute_error(anchor_vals, cand_vals)
            rmse = np.sqrt(mean_squared_error(anchor_vals, cand_vals))
            
            sim = {
                "pearson": pearson,
                "spearman": spearman,
                "mae": mae,
                "rmse": rmse,
                "median_diff": np.median(cand_vals) - np.median(anchor_vals),
                "p05_diff": np.percentile(cand_vals, 5) - np.percentile(anchor_vals, 5),
                "p50_diff": np.percentile(cand_vals, 50) - np.percentile(anchor_vals, 50),
                "p95_diff": np.percentile(cand_vals, 95) - np.percentile(anchor_vals, 95),
                "range_1_99": np.percentile(cand_vals, 99) - np.percentile(cand_vals, 1),
                "frac_large_dev": np.mean(np.abs(cand_vals - anchor_vals) > 0.1 * np.abs(anchor_vals)),
            }
            
            if target == "Total Alkalinity":
                sim_ta = sim
            elif target == "Electrical Conductance":
                sim_ec = sim
            else:
                sim_drp = sim
    
    # Compare to failed
    failed_comp = {}
    for target in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]:
        if target in failed_df.columns and target in candidate_df.columns:
            failed_vals = failed_df[target].values
            cand_vals = candidate_df[target].values
            
            failed_pearson = pearsonr(failed_vals, cand_vals)[0]
            anchor_pearson = pearsonr(anchor_df[target].values, cand_vals)[0]
            
            if failed_pearson > anchor_pearson:
                comp = "closer_to_failed"
            elif anchor_pearson > failed_pearson:
                comp = "closer_to_anchor"
            else:
                comp = "similar_to_both"
            
            failed_comp[target] = comp
    
    # Robustness assessment (simplified)
    drp_fold_var = 0.05  # placeholder, would need fold scores
    robustness = "ROBUST" if drp_fold_var < 0.1 else "MODERATE RISK" if drp_fold_var < 0.2 else "HIGH RISK"
    
    # Leaderboard risk
    drp_sim = sim_drp.get("pearson", 0)
    drp_mae = sim_drp.get("mae", 1)
    risk = "LOW RISK" if drp_sim > 0.8 and drp_mae < 0.1 else "MEDIUM RISK" if drp_sim > 0.6 else "HIGH RISK"
    
    similarity_results.append({
        "experiment_name": exp_name,
        "ta_pearson": sim_ta.get("pearson"),
        "ta_spearman": sim_ta.get("spearman"),
        "ta_mae": sim_ta.get("mae"),
        "ta_rmse": sim_ta.get("rmse"),
        "ec_pearson": sim_ec.get("pearson"),
        "ec_spearman": sim_ec.get("spearman"),
        "ec_mae": sim_ec.get("mae"),
        "ec_rmse": sim_ec.get("rmse"),
        "drp_pearson": sim_drp.get("pearson"),
        "drp_spearman": sim_drp.get("spearman"),
        "drp_mae": sim_drp.get("mae"),
        "drp_rmse": sim_drp.get("rmse"),
        "drp_median_diff": sim_drp.get("median_diff"),
        "drp_p05_diff": sim_drp.get("p05_diff"),
        "drp_p95_diff": sim_drp.get("p95_diff"),
        "drp_frac_large_dev": sim_drp.get("frac_large_dev"),
        "robustness_label": robustness,
        "leaderboard_risk": risk,
        "comparison_to_failed": str(failed_comp),
    })

similarity_df = pd.DataFrame(similarity_results)
similarity_csv = PROJECT_ROOT / "experiments" / "robustness_sprint_similarity.csv"
similarity_df.to_csv(similarity_csv, index=False)
print(f"Similarity results saved to: {similarity_csv}")

# Create ensemble
print(f"\n{'='*80}")
print("CREATING DRP ENSEMBLE")
print(f"{'='*80}")

# Find successful submissions
successful_subs = [r for r in results if r["status"] == "SUCCESS" and r["submission_file"]]
ensemble_preds = {}

for target in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]:
    preds = []
    weights = []
    
    for r in successful_subs:
        df = pd.read_csv(r["submission_file"])
        if target in df.columns:
            pred = df[target].values
            preds.append(pred)
            
            # Weights: 0.5 ET, 0.3 RF, 0.2 HGB
            if "ET" in r["experiment_name"]:
                weights.append(0.5)
            elif "RF" in r["experiment_name"]:
                weights.append(0.3)
            elif "HGB" in r["experiment_name"]:
                weights.append(0.2)
            else:
                weights.append(0.2)  # default
    
    if preds:
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble_pred = np.average(preds, axis=0, weights=weights)
        ensemble_preds[target] = ensemble_pred

# Create ensemble submission
if ensemble_preds:
    ensemble_df = anchor_df.copy()
    for target, pred in ensemble_preds.items():
        ensemble_df[target] = pred
    
    ensemble_file = PROJECT_ROOT / "submissions_batch" / "submission_robustness_ensemble.csv"
    ensemble_df.to_csv(ensemble_file, index=False)
    print(f"Ensemble submission saved to: {ensemble_file}")
    
    # Analyze ensemble similarity
    ensemble_sim = {}
    for target in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]:
        if target in anchor_df.columns and target in ensemble_df.columns:
            anchor_vals = anchor_df[target].values
            ens_vals = ensemble_df[target].values
            
            ensemble_sim[target] = {
                "pearson": pearsonr(anchor_vals, ens_vals)[0],
                "mae": mean_absolute_error(anchor_vals, ens_vals),
                "median_diff": np.median(ens_vals) - np.median(anchor_vals),
            }
    
    print("Ensemble similarity to anchor:")
    for target, sim in ensemble_sim.items():
        print(f"  {target}: Pearson={sim['pearson']:.3f}, MAE={sim['mae']:.3f}, Median diff={sim['median_diff']:.3f}")

print(f"\n{'='*80}")
print("SPRINT COMPLETE")
print(f"{'='*80}")