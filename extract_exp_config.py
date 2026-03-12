#!/usr/bin/env python
# coding: utf-8

import os
import json
from pathlib import Path

# Find the latest experiment
exp_dir = Path("experiments")
latest_exp = max([d for d in exp_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_ctime)
print(f"Latest experiment: {latest_exp.name}")

# Read config snapshot
config_path = latest_exp / "config_snapshot.json"
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    print(f"\nDRP Features Config:")
    drp_feat = config.get('features', {}).get('drp_focused', {})
    print(f"  enabled: {drp_feat.get('enabled')}")
    print(f"  include_interactions: {drp_feat.get('include_interactions')}")
    print(f"  include_proxies: {drp_feat.get('include_proxies')}")
    
    print(f"\nDRP Model Config:")
    drp_model = config.get('model', {}).get('cv_by_target', {}).get('Dissolved Reactive Phosphorus', {})
    print(f"  name: {drp_model.get('name')}")
    print(f"  min_samples_leaf: {drp_model.get('params', {}).get('min_samples_leaf')}")
    print(f"  max_features: {drp_model.get('params', {}).get('max_features')}")
    
    print(f"\nTarget Transform:")
    print(f"  DRP y_mode: {config.get('targets', {}).get('y_mode_by_target', {}).get('Dissolved Reactive Phosphorus')}")

# Look for feature manifest to check feature count
manifest_path = latest_exp / "feature_manifest_Dissolved_Reactive_Phosphorus.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"\nDRP Feature Count: {len(manifest)}")
    print(f"Sample features (first 10): {manifest[:10]}")
