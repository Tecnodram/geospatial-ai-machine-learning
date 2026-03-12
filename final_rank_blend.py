#!/usr/bin/env python
# coding: utf-8
"""
FINAL RANK BLEND: Combine CatBoost Physical Model with Historical Anchor
- New (attack): 70% weight (CV-honest, physically informed)
- Old (anchor): 30% weight (stable, proven LB performance)
"""

import os
import pandas as pd
import numpy as np

KEYS = ["Latitude", "Longitude", "Sample Date"]
TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]

def load_submission(path: str, name: str) -> pd.DataFrame:
    """Load submission CSV and validate structure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded {name}: shape={df.shape}, keys exist: {all(k in df.columns for k in KEYS)}")
    return df

def blend_submissions(new_df: pd.DataFrame, old_df: pd.DataFrame, 
                     new_weight: float = 0.7, old_weight: float = 0.3) -> pd.DataFrame:
    """
    Blend two submissions with weighted average.
    new_weight: fraction for new (CatBoost) model
    old_weight: fraction for old (anchor) model
    """
    # Verify structure
    assert new_df[KEYS].equals(old_df[KEYS]), "Key columns mismatch"
    blended = new_df[KEYS].copy()
    
    # Blend each target
    for target in TARGETS:
        new_vals = new_df[target].astype(float).values
        old_vals = old_df[target].astype(float).values
        
        blended[target] = (new_weight * new_vals) + (old_weight * old_vals)
        
        print(f"\n{target}:")
        print(f"  New (70%): mean={new_vals.mean():.4f}, min={new_vals.min():.4f}, max={new_vals.max():.4f}")
        print(f"  Old (30%): mean={old_vals.mean():.4f}, min={old_vals.min():.4f}, max={old_vals.max():.4f}")
        print(f"  Blended:   mean={blended[target].mean():.4f}, min={blended[target].min():.4f}, max={blended[target].max():.4f}")
    
    return blended

def sanity_check(df: pd.DataFrame) -> bool:
    """Verify no NaNs and no negatives (especially DRP)."""
    issues = []
    
    # Check NaNs
    if df[TARGETS].isna().any().any():
        issues.append("❌ NaN values detected in targets")
    
    # Check negatives
    for target in TARGETS:
        negs = (df[target] < 0).sum()
        if negs > 0:
            issues.append(f"❌ {negs} negative values in {target}")
    
    # Check shape
    if df.shape[0] != 200:
        issues.append(f"❌ Expected 200 rows, got {df.shape[0]}")
    
    if issues:
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n✅ All sanity checks passed!")
        return True

def main():
    print("=" * 80)
    print("FINAL RANK BLEND: 0.40 Certificate Attack")
    print("=" * 80)
    
    # Load submissions
    new_path = "submissions/submission_24h_certificate_attack.csv"
    old_path = "submissions/submission_V4_4_DRP_tuned_ET_fixorder.csv"
    
    new_df = load_submission(new_path, "CatBoost Physical Model (NEW)")
    old_df = load_submission(old_path, "Historical DRP Tuned (ANCHOR)")
    
    # Blend with 70-30 weighting
    print("\n" + "=" * 80)
    print("BLENDING: 70% New (CatBoost) + 30% Old (Anchor)")
    print("=" * 80)
    
    blended_df = blend_submissions(new_df, old_df, new_weight=0.7, old_weight=0.3)
    
    # Sanity check
    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)
    
    if not sanity_check(blended_df):
        print("❌ FAILED - Not saving")
        return False
    
    # Save
    output_path = "submissions/SUBMISSION_FINAL_CERTIFICATE_0.40.csv"
    blended_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("FINAL SUBMISSION SUMMARY")
    print("=" * 80)
    print(f"File: {output_path}")
    print(f"Rows: {blended_df.shape[0]}")
    print(f"Columns: {list(blended_df.columns)}")
    print(f"\nTarget Ranges:")
    for target in TARGETS:
        vals = blended_df[target].values
        print(f"  {target:30s}: [{vals.min():8.2f}, {vals.max():8.2f}] μ={vals.mean():8.2f} σ={vals.std():8.2f}")
    
    print("\n" + "=" * 80)
    print("✅ FINAL SUBMISSION READY FOR UPLOAD")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
