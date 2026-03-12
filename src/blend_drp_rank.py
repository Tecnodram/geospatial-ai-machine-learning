#!/usr/bin/env python
"""
Rank-blend DRP only between V4_4 (anchor) and Sprint 4 (improvement).
Keeps TA and EC exactly from V4_4.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
v44_path = Path("submissions/submission_V4_4_DRP_tuned_ET_fixorder.csv")
s4_path = Path("experiments/exp_20260311_215006/submission.csv")
output_dir = Path("submissions_batch")
output_dir.mkdir(exist_ok=True)

# Load both submissions
v44 = pd.read_csv(v44_path)
s4 = pd.read_csv(s4_path)

# Ensure aligned on Latitude/Longitude/Sample Date
assert len(v44) == len(s4), f"Row mismatch: v44={len(v44)}, s4={len(s4)}"
assert (v44[['Latitude', 'Longitude', 'Sample Date']] == s4[['Latitude', 'Longitude', 'Sample Date']]).all().all(), \
    "Geometry/date mismatch"

# Rank DRP columns (lower is better in ranking)
v44_drp_rank = v44['Dissolved Reactive Phosphorus'].rank()
s4_drp_rank = s4['Dissolved Reactive Phosphorus'].rank()

# Define blend configs: (v44_weight, s4_weight, name)
blend_configs = [
    (0.90, 0.10, "blend_v44_90_s4_10_rank"),
    (0.85, 0.15, "blend_v44_85_s4_15_rank"),
    (0.80, 0.20, "blend_v44_80_s4_20_rank"),
]

results = []

for v44_w, s4_w, blend_name in blend_configs:
    # Rank blend: blend ranks, then invert back to values
    blended_rank = v44_w * v44_drp_rank + s4_w * s4_drp_rank
    
    # Sort original values by their ranks and map back to blended ranks
    v44_sorted = v44['Dissolved Reactive Phosphorus'].sort_values().reset_index(drop=True)
    s4_sorted = s4['Dissolved Reactive Phosphorus'].sort_values().reset_index(drop=True)
    
    # Map blended rank to values by interpolating position in the joint distribution
    # Simple approach: interpolate between v44 and s4 at each percentile
    percentiles = (blended_rank - 1) / (len(blended_rank) - 1)  # 0-1 scale
    blended_drp = np.zeros(len(blended_rank))
    for i, pct in enumerate(percentiles):
        idx_v44 = np.clip(pct * (len(v44_sorted) - 1), 0, len(v44_sorted) - 1)
        idx_s4 = np.clip(pct * (len(s4_sorted) - 1), 0, len(s4_sorted) - 1)
        
        # Interpolate within each sorted list
        v44_val = v44_sorted.iloc[int(idx_v44)] + (idx_v44 % 1) * (
            v44_sorted.iloc[int(idx_v44) + 1] - v44_sorted.iloc[int(idx_v44)]
            if int(idx_v44) < len(v44_sorted) - 1 else 0
        )
        s4_val = s4_sorted.iloc[int(idx_s4)] + (idx_s4 % 1) * (
            s4_sorted.iloc[int(idx_s4) + 1] - s4_sorted.iloc[int(idx_s4)]
            if int(idx_s4) < len(s4_sorted) - 1 else 0
        )
        
        # Blend the interpolated values
        blended_drp[i] = v44_w * v44_val + s4_w * s4_val
    
    # Create output DataFrame: TA and EC from V4_4, DRP blended
    output_df = v44[['Latitude', 'Longitude', 'Sample Date', 'Total Alkalinity', 'Electrical Conductance']].copy()
    output_df['Dissolved Reactive Phosphorus'] = blended_drp
    
    # Save to CSV
    output_path = output_dir / f"{blend_name}.csv"
    output_df.to_csv(output_path, index=False)
    
    # Compute metrics
    drp_min = blended_drp.min()
    drp_max = blended_drp.max()
    drp_med = np.median(blended_drp)
    drp_corr = np.corrcoef(blended_drp, v44['Dissolved Reactive Phosphorus'])[0, 1]
    drp_mae = np.mean(np.abs(blended_drp - v44['Dissolved Reactive Phosphorus']))
    
    results.append({
        'blend_name': blend_name,
        'file_path': str(output_path),
        'v44_weight': v44_w,
        's4_weight': s4_w,
        'drp_min': drp_min,
        'drp_max': drp_max,
        'drp_median': drp_med,
        'drp_corr_to_v44': drp_corr,
        'drp_mae_to_v44': drp_mae,
    })
    
    print(f"✓ {blend_name}")
    print(f"  Path: {output_path}")
    print(f"  DRP: min={drp_min:.2f}, max={drp_max:.2f}, median={drp_med:.2f}")
    print(f"  Correlation to V4_4: {drp_corr:.4f}")
    print(f"  MAE to V4_4: {drp_mae:.4f}")
    print()

# Save summary CSV
summary_df = pd.DataFrame(results)
summary_path = output_dir / "BLEND_SUMMARY_RANK_DRP.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary saved to {summary_path}")
print(summary_df.to_string(index=False))
