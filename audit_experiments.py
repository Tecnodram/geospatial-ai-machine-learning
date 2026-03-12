#!/usr/bin/env python
# coding: utf-8

import os
import json
import pandas as pd
from pathlib import Path

# Find all experiments
exp_dir = Path("experiments")
experiments = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")])

# Build audit report
audit_data = []

for exp_path in experiments[-15:]:  # Check last 15 experiments
    config_file = exp_path / "config_snapshot.json"
    cv_file = exp_path / "cv_report.json"
    
    if config_file.exists() and cv_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        with open(cv_file) as f:
            cv_data = json.load(f)
        
        drp_mode = config["targets"]["y_mode_by_target"].get("Dissolved Reactive Phosphorus", "unknown")
        drp_cv_model = config["model"]["cv_by_target"].get("Dissolved Reactive Phosphorus", {}).get("name", "ExtraTreesRegressor (default)")
        drp_final_model = config["model"]["final_by_target"].get("Dissolved Reactive Phosphorus", {}).get("name", "ExtraTreesRegressor (default)")
        
        drp_mean = cv_data["Dissolved Reactive Phosphorus"]["mean"]
        ta_mean = cv_data["Total Alkalinity"]["mean"]
        ec_mean = cv_data["Electrical Conductance"]["mean"]
        
        matches_intended = (drp_mode == "sqrt" and 
                          drp_cv_model == "ExtraTreesRegressor" and 
                          drp_final_model == "ExtraTreesRegressor")
        
        audit_data.append({
            "Experiment": exp_path.name,
            "DRP y_mode": drp_mode,
            "DRP CV Model": drp_cv_model,
            "DRP Final Model": drp_final_model,
            "TA R²": f"{ta_mean:.4f}",
            "EC R²": f"{ec_mean:.4f}",
            "DRP R²": f"{drp_mean:.4f}",
            "Matches Intended": "✓ YES" if matches_intended else "✗ NO"
        })

df = pd.DataFrame(audit_data)
print("\n" + "="*140)
print("RECENT EXPERIMENTS AUDIT (Last 15)")
print("="*140)
print(df.to_string(index=False))

# Find the best matching experiments
print("\n" + "="*140)
print("EXPERIMENTS MATCHING INTENDED MODEL (sqrt + ExtraTreesRegressor)")
print("="*140)
matching = df[df["Matches Intended"] == "✓ YES"].copy()
if len(matching) > 0:
    matching["DRP R² float"] = matching["DRP R²"].astype(float)
    matching_sorted = matching.sort_values("DRP R² float", ascending=False)
    print(matching_sorted[["Experiment", "DRP y_mode", "DRP CV Model", "DRP Final Model", "TA R²", "EC R²", "DRP R²"]].to_string(index=False))
else:
    print("NONE FOUND")
