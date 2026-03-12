#!/usr/bin/env python
"""Verify basin-aware grouping prevents spatial leakage"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

train_df = pd.read_pickle('cache/train_modelready.pkl')

print('=== BASIN-AWARE GROUPING VERIFICATION ===\n')
print(f'Total stations in train: {len(train_df)}')
print(f'Unique basins in train: {train_df["basin_id"].nunique()}')
print(f'Basin ID completeness: {(1 - train_df["basin_id"].isna().sum()/len(train_df))*100:.1f}%')

# Verify basin separation across folds
groups = train_df['basin_id'].fillna('unknown').astype(str).values
gkf = GroupKFold(n_splits=3)
y = np.random.randn(len(train_df))

print(f'\nFold basin separation check:')
all_clean = True
for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups=groups), start=1):
    tr_basins = set(train_df.iloc[tr_idx]['basin_id'].dropna().unique())
    va_basins = set(train_df.iloc[va_idx]['basin_id'].dropna().unique())
    overlap = tr_basins & va_basins
    status = f'Overlap={len(overlap)}'
    if overlap:
        status += f' *** LEAKAGE: {str(list(overlap)[:2])}'
        all_clean = False
    else:
        status += ' CLEAN'
    print(f'  Fold {fold_i}: Train={len(tr_basins)}, Valid={len(va_basins)}, {status}')

if all_clean:
    print('\n✅ Basin-aware grouping prevents spatial leakage!')
else:
    print('\n⚠️  WARNING: Basin leakage detected!')
