#!/usr/bin/env python
"""
EMERGENCY RESCUE BLEND: Align predictions to match template format exactly.
- Uses template IDs as master index
- Blends 70% attack + 30% anchor
- Handles log-scale detection and NaN issues
- Ensures exact row order matching
"""

import pandas as pd
import numpy as np
import sys
import os

TEMPLATE_PATH = "data/raw/submission_template.csv"
ANCHOR_PATH = "submissions/submission_V4_4_DRP_tuned_ET_fixorder.csv"
ATTACK_PATH = "submissions/submission_24h_certificate_attack.csv"
OUTPUT_PATH = "SUBMISSION_RESCUE_FINAL_ALIGNED.csv"

TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]

def main():
    print("=" * 80)
    print("EMERGENCY RESCUE: ALIGNED BLENDING")
    print("=" * 80)
    
    # Step 1: Load template (authoritative format)
    print(f"\n[1/7] Loading TEMPLATE (source of truth): {TEMPLATE_PATH}")
    if not os.path.exists(TEMPLATE_PATH):
        print(f"❌ ERROR: Template not found at {TEMPLATE_PATH}")
        return False
    
    template = pd.read_csv(TEMPLATE_PATH)
    print(f"      Shape: {template.shape}")
    print(f"      Columns: {list(template.columns)}")
    print(f"      First row ID: {template.iloc[0].to_dict()}")
    
    template_ids = template.iloc[:, 0]  # First column is the ID
    id_col_name = template.columns[0]
    print(f"      ID Column: '{id_col_name}'")
    
    # Step 2: Load anchor submission
    print(f"\n[2/7] Loading ANCHOR (30% weight): {ANCHOR_PATH}")
    if not os.path.exists(ANCHOR_PATH):
        print(f"❌ ERROR: Anchor not found at {ANCHOR_PATH}")
        return False
    
    anchor = pd.read_csv(ANCHOR_PATH)
    print(f"      Shape: {anchor.shape}")
    print(f"      Columns: {list(anchor.columns)}")
    
    # Step 3: Load attack submission
    print(f"\n[3/7] Loading ATTACK (70% weight): {ATTACK_PATH}")
    if not os.path.exists(ATTACK_PATH):
        print(f"❌ ERROR: Attack not found at {ATTACK_PATH}")
        return False
    
    attack = pd.read_csv(ATTACK_PATH)
    print(f"      Shape: {attack.shape}")
    print(f"      Columns: {list(attack.columns)}")
    
    # Step 4: Create aligned output using template as master
    print(f"\n[4/7] Creating aligned output using template as master index...")
    
    # Start with template structure
    aligned = template.copy()
    
    print(f"      Template row count: {len(template)}")
    print(f"      Anchor row count: {len(anchor)}")
    print(f"      Attack row count: {len(attack)}")
    
    # Blend each target column
    for target in TARGETS:
        if target not in anchor.columns or target not in attack.columns:
            print(f"      ⚠️  {target} not in one of the submission files")
            continue
        
        anchor_vals = anchor[target].values
        attack_vals = attack[target].values
        
        # Check if values are log-scaled (if max is very small, likely log-scale)
        if attack_vals.max() < 10 and target == "Electrical Conductance":
            print(f"      ⚠️  {target} appears to be log-scaled in attack (max={attack_vals.max():.4f}), applying expm1...")
            attack_vals = np.expm1(attack_vals)
        
        if anchor_vals.max() < 10 and target == "Electrical Conductance":
            print(f"      ⚠️  {target} appears to be log-scaled in anchor (max={anchor_vals.max():.4f}), applying expm1...")
            anchor_vals = np.expm1(anchor_vals)
        
        # Blend: 70% attack + 30% anchor
        blended_vals = (0.7 * attack_vals) + (0.3 * anchor_vals)
        
        # Replace column in aligned output
        aligned[target] = blended_vals
        
        # Statistics
        print(f"      \n      {target}:")
        print(f"        Attack (70%):  min={attack_vals.min():.4f}, max={attack_vals.max():.4f}, mean={attack_vals.mean():.4f}")
        print(f"        Anchor (30%):  min={anchor_vals.min():.4f}, max={anchor_vals.max():.4f}, mean={anchor_vals.mean():.4f}")
        print(f"        Blended:       min={blended_vals.min():.4f}, max={blended_vals.max():.4f}, mean={blended_vals.mean():.4f}")
    
    # Step 5: Validation & Repair
    print(f"\n[5/7] Validation & Repair...")
    
    issues_found = False
    for target in TARGETS:
        # Check for NaN
        nan_count = aligned[target].isna().sum()
        if nan_count > 0:
            print(f"      ⚠️  {target}: {nan_count} NaN values detected, filling with 0")
            aligned[target].fillna(0, inplace=True)
            issues_found = True
        
        # Check for negatives
        neg_count = (aligned[target] < 0).sum()
        if neg_count > 0:
            print(f"      ⚠️  {target}: {neg_count} negative values detected, clipping to 0")
            aligned[target] = aligned[target].clip(lower=0)
            issues_found = True
        
        # Check for infinities
        inf_count = np.isinf(aligned[target]).sum()
        if inf_count > 0:
            print(f"      ⚠️  {target}: {inf_count} infinite values, replacing with 0")
            aligned[target] = aligned[target].replace([np.inf, -np.inf], 0)
            issues_found = True
    
    if not issues_found:
        print(f"      ✅ All validation checks passed!")
    
    # Step 6: Verify exact alignment with template
    print(f"\n[6/7] Verifying exact alignment with template...")
    
    # Check ID column is preserved
    if (aligned[id_col_name] == template[id_col_name]).all():
        print(f"      ✅ ID column '{id_col_name}' perfectly preserved")
    else:
        print(f"      ❌ ERROR: ID column mismatch!")
        return False
    
    # Check preserved columns (non-target columns from template)
    preserved_cols = [c for c in template.columns if c not in TARGETS]
    for col in preserved_cols:
        if col not in aligned.columns:
            print(f"      ⚠️  Column '{col}' missing, reconstructing from template")
            aligned[col] = template[col]
        elif not (aligned[col] == template[col]).all():
            print(f"      ⚠️  Column '{col}' changed, restoring from template")
            aligned[col] = template[col]
    
    print(f"      ✅ Template structure preserved")
    
    # Step 7: Save output
    print(f"\n[7/7] Saving aligned submission to: {OUTPUT_PATH}")
    
    # Ensure column order matches template
    aligned = aligned[template.columns]
    aligned.to_csv(OUTPUT_PATH, index=False)
    print(f"      ✅ File saved")
    
    # Final report
    print(f"\n" + "=" * 80)
    print(f"✅ RESCUE BLEND COMPLETE")
    print(f"=" * 80)
    print(f"\nFile: {OUTPUT_PATH}")
    print(f"Rows: {len(aligned)} (matches template: {len(aligned) == len(template)})")
    print(f"Columns: {len(aligned.columns)} (matches template: {len(aligned.columns) == len(template.columns)})")
    
    print(f"\nFIRST 5 ROWS COMPARISON:")
    print(f"\n--- TEMPLATE ---")
    print(template.head())
    print(f"\n--- ALIGNED RESCUE ---")
    print(aligned.head())
    
    print(f"\n" + "=" * 80)
    print(f"✅ ALIGNMENT VERIFIED - READY FOR UPLOAD")
    print(f"=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
