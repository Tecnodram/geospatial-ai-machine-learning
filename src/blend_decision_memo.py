#!/usr/bin/env python
"""
Generate blend metrics report and decision memo.
Compares all blends plus Spring 4 pure against V4_4 anchor.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
v44_path = Path("submissions/submission_V4_4_DRP_tuned_ET_fixorder.csv")
s4_path = Path("experiments/exp_20260311_215006/submission.csv")
summary_path = Path("submissions_batch/BLEND_SUMMARY_RANK_DRP.csv")

# Load anchor
v44 = pd.read_csv(v44_path)
s4 = pd.read_csv(s4_path)

# Load blend summary
blend_summary = pd.read_csv(summary_path)

# Compute metrics for Sprint 4 pure
s4_drp = s4['Dissolved Reactive Phosphorus'].values
v44_drp = v44['Dissolved Reactive Phosphorus'].values

s4_drp_min = s4_drp.min()
s4_drp_max = s4_drp.max()
s4_drp_med = np.median(s4_drp)
s4_drp_corr = np.corrcoef(s4_drp, v44_drp)[0, 1]
s4_drp_mae = np.mean(np.abs(s4_drp - v44_drp))

print("=" * 80)
print("POST-SPRINT-4 BLEND METRICS REPORT")
print("=" * 80)
print()

print("ANCHOR (V4_4):")
print(f"  DRP: min={v44_drp.min():.2f}, max={v44_drp.max():.2f}, median={np.median(v44_drp):.2f}")
print()

print("SPRINT 4 PURE (exp_20260311_215006):")
print(f"  DRP: min={s4_drp_min:.2f}, max={s4_drp_max:.2f}, median={s4_drp_med:.2f}")
print(f"  Correlation to V4_4: {s4_drp_corr:.4f}")
print(f"  MAE to V4_4: {s4_drp_mae:.4f}")
print(f"  Delta median vs V4_4: {s4_drp_med - np.median(v44_drp):+.2f}")
print()

print("RANK-BLENDED OPTIONS:")
print("-" * 80)
for _, row in blend_summary.iterrows():
    blend_name = row['blend_name']
    v44_w = row['v44_weight']
    s4_w = row['s4_weight']
    print(f"{blend_name} ({v44_w:.0%} V4_4 + {s4_w:.0%} Sprint4):")
    print(f"  DRP: min={row['drp_min']:.2f}, max={row['drp_max']:.2f}, median={row['drp_median']:.2f}")
    print(f"  Correlation to V4_4: {row['drp_corr_to_v44']:.4f}")
    print(f"  MAE to V4_4: {row['drp_mae_to_v44']:.4f}")
    print()

print("=" * 80)
print("RISK ANALYSIS")
print("=" * 80)
print()

# Define safety thresholds
corr_safety_min = 0.98  # Correlation should stay high
mae_safety_max = 1.0    # MAE should be small (avg absolute delta per row)

print(f"Safety thresholds:")
print(f"  Min correlation to V4_4: {corr_safety_min:.2f}")
print(f"  Max MAE to V4_4: {mae_safety_max:.4f}")
print()

# Evaluate each option
options = []

# Sprint 4 pure
options.append({
    'name': 'Sprint4 Pure (exp_20260311_215006)',
    'corr': s4_drp_corr,
    'mae': s4_drp_mae,
    'safe': s4_drp_corr >= corr_safety_min and s4_drp_mae <= mae_safety_max
})

# Blends
for _, row in blend_summary.iterrows():
    options.append({
        'name': row['blend_name'],
        'corr': row['drp_corr_to_v44'],
        'mae': row['drp_mae_to_v44'],
        'safe': row['drp_corr_to_v44'] >= corr_safety_min and row['drp_mae_to_v44'] <= mae_safety_max
    })

for opt in options:
    status = "✓ SAFE" if opt['safe'] else "✗ RISKY"
    corr_ok = "✓" if opt['corr'] >= corr_safety_min else "✗"
    mae_ok = "✓" if opt['mae'] <= mae_safety_max else "✗"
    print(f"{status}  {opt['name']}: corr={corr_ok} {opt['corr']:.4f}, mae={mae_ok} {opt['mae']:.4f}")

print()
print("=" * 80)
print("DECISION MEMO")
print("=" * 80)
print()

# Find best safe blend
safe_blends = [o for o in options[1:] if o['safe']]  # Skip Sprint4 pure
if safe_blends:
    best = min(safe_blends, key=lambda x: x['mae'])  # Minimize MAE among safe
    print(f"✓ RECOMMENDED NEXT SUBMISSION: {best['name']}")
    print(f"  Rationale: Highest correlation ({best['corr']:.4f}), lowest MAE ({best['mae']:.4f}) among safe options")
    print()
    
    # Backup
    backup = [o for o in safe_blends if o['name'] != best['name']]
    if backup:
        backup_best = min(backup, key=lambda x: x['mae'])
        print(f"✓ BACKUP CANDIDATE: {backup_best['name']}")
        print(f"  Safety: corr={backup_best['corr']:.4f}, mae={backup_best['mae']:.4f}")
else:
    # If no safe blends, check if any are marginally acceptable
    marginal = [o for o in options if o['corr'] >= 0.99]
    if marginal:
        best_marginal = min(marginal, key=lambda x: x['mae'])
        print(f"⚠ NO FULLY SAFE OPTIONS. Marginal candidate: {best_marginal['name']}")
        print(f"  Please review before submission")
    else:
        best_marginal = min(options, key=lambda x: x['mae'])
        print(f"⚠ ALL OPTIONS RISKY. Least risky: {best_marginal['name']}")

print()
print("RISK VS V4_4:")
if best_blends := [o for o in options[1:] if o['safe']]:
    best = min(best_blends, key=lambda x: x['mae'])
    print(f"  Expected correlation decay: {1 - best['corr']:.4%}")
    print(f"  Expected avg DRP delta per row: {best['mae']:.4f} μg/L")
    print(f"  Status: Low leaderboard risk (V4_4 is known safe benchmark)")
else:
    print(f"  ⚠ Elevated risk - recommend review before submission")

print()
print("SPRINT 4 PURE SUBMISSION:")
if s4_drp_corr >= corr_safety_min and s4_drp_mae <= mae_safety_max:
    print(f"  ✓ Acceptable (corr={s4_drp_corr:.4f}, mae={s4_drp_mae:.4f})")
    print(f"  Could be submitted directly, but blends offer more conservative hedge")
else:
    print(f"  ✗ NOT RECOMMENDED for direct submission (corr={s4_drp_corr:.4f}, mae={s4_drp_mae:.4f})")
    print(f"  Use a blend instead to maintain anchor correlation")

print()
print("=" * 80)
