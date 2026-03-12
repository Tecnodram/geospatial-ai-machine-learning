#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

# Check both submission files
files_to_check = [
    "submissions_batch/submission_V5_2_OOFTE_fixkeys.csv",
    "submissions_batch/submission_V5_2_OOFTE_fixkeys_as_is.csv",
    "experiments/exp_20260306_101703/submission_V5_2_OOFTE_fixkeys.csv"
]

for fpath in files_to_check:
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        print(f"\n{'='*70}")
        print(f"FILE: {fpath}")
        print(f"{'='*70}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Has NaN: {df.isnull().sum().sum()}")
        print(f"DRP min: {df['Dissolved Reactive Phosphorus'].min():.2f}")
        print(f"DRP max: {df['Dissolved Reactive Phosphorus'].max():.2f}")
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
    else:
        print(f"\nNOT FOUND: {fpath}")
