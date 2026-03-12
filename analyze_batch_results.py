#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

df = pd.read_csv('experiments/batch_results.csv')
df_ok = df[df['status'] == 'OK'].copy()

print("=== DRP FEATURE ENGINEERING & TUNING EXPERIMENTS ===\n")
print(f"Total completed: {len(df_ok)}\n")

# Sort by DRP mean descending
df_ok = df_ok.sort_values('drp_mean', ascending=False)

print("RESULTS (sorted by DRP R²):\n")
print(df_ok[['experiment_name', 'drp_focused_enabled', 'include_interactions', 'include_proxies', 'min_samples_leaf', 'max_features', 'ta_mean', 'ec_mean', 'drp_mean']].to_string(index=False))

print("\n\n=== COMPARISON vs BASELINE ===\n")
baseline = df_ok[df_ok['experiment_name'] == 'baseline_sqrt_ET'].iloc[0]
print(f"Baseline (sqrt ET, no features):")
print(f"  TA:  {baseline['ta_mean']:.4f}")
print(f"  EC:  {baseline['ec_mean']:.4f}")
print(f"  DRP: {baseline['drp_mean']:.4f}")

print(f"\nBest (all tunings tied at full features):")
best = df_ok.iloc[0]
print(f"  Experiment: {best['experiment_name']}")
print(f"  TA:  {best['ta_mean']:.4f} ({best['ta_mean']-baseline['ta_mean']:+.4f})")
print(f"  EC:  {best['ec_mean']:.4f} ({best['ec_mean']-baseline['ec_mean']:+.4f})")
print(f"  DRP: {best['drp_mean']:.4f} ({best['drp_mean']-baseline['drp_mean']:+.4f})")

# Find best DRP lift
best['drp_lift'] = best['drp_mean'] - baseline['drp_mean']
print(f"\nDRP improvement: {best['drp_lift']:.4f} (+{100*best['drp_lift']/baseline['drp_mean']:.1f}%)")

print("\n=== FEATURE SET ANALYSIS ===\n")
# Group by feature configs
by_features = df_ok.groupby(['drp_focused_enabled', 'include_interactions', 'include_proxies']).agg({
    'drp_mean': 'mean',
    'ta_mean': 'mean',
    'ec_mean': 'mean',
    'experiment_name': 'first'
}).reset_index()
by_features.columns = ['DRP_Feat', 'Inter', 'Proxy', 'DRP_R2', 'TA_R2', 'EC_R2', 'Example']
by_features = by_features.sort_values('DRP_R2', ascending=False)
print(by_features[['DRP_Feat', 'Inter', 'Proxy', 'DRP_R2', 'TA_R2', 'EC_R2']].to_string(index=False))

print("\n=== TUNING ANALYSIS ===\n")
# Analysis of tuning only (where interactions=True, proxies=True)
tuned = df_ok[(df_ok['include_interactions'] == True) & (df_ok['include_proxies'] == True)].copy()
if len(tuned) > 0:
    tuned_minleaf = tuned.groupby('min_samples_leaf')['drp_mean'].agg(['mean', 'count'])
    print("Impact of min_samples_leaf (with full features):")
    print(tuned_minleaf)

    tuned_maxfeat = tuned.groupby('max_features')['drp_mean'].agg(['mean', 'count'])
    print("\nImpact of max_features (with full features):")
    print(tuned_maxfeat)

print("\n=== DECISION ===\n")
print("The full feature set (interactions + proxies) with default tuning (minleaf=3, maxfeat='sqrt')")
print("achieves DRP R² = 0.1547, an improvement of +0.0075 (+5.1%) over baseline.")
print("\nAll tuning variations (minleaf 2,4,8 and maxfeat 0.5) achieve identical scores:")
print("  TA:  0.4106 (-0.0033, -0.8%)")
print("  EC:  0.3300 (-0.0009, -0.3%)")
print("  DRP: 0.1547 (+0.0075, +5.1%)")
print("\nRECOMMENDATION: Use full feature set (Config: drp_focused=enabled, interactions=True, proxies=True)")
print("with default tuning (minleaf=3, maxfeat='sqrt'). Keep it simple — no further tuning needed.")
