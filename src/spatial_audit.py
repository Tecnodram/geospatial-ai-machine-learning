#!/usr/bin/env python
# coding: utf-8

"""
Spatial Audit for EY Water Quality Pipeline

Analyzes current CV grouping and compares alternatives.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external_geofeatures_plus_hydro_v2.csv"

# Load data
train_df = pd.read_csv(RAW_DIR / "water_quality_training_dataset.csv")
external_df = pd.read_csv(EXTERNAL_DIR)

print(f"Training data shape: {train_df.shape}")
print(f"External features shape: {external_df.shape}")

# Merge external features (assuming by station lat/lon)
# For simplicity, assume external is per unique lat/lon
train_df = train_df.merge(external_df, on=['Latitude', 'Longitude'], how='left')

print(f"After merge shape: {train_df.shape}")
print(f"Basin_id exists: {'basin_id' in train_df.columns}")

# Current grouping function
def make_groups(df, grid):
    return (
        np.floor(df['Latitude'] / grid).astype(int).astype(str)
        + "_"
        + np.floor(df['Longitude'] / grid).astype(int).astype(str)
    )

# Current CV setup
GRID = 0.2  # from oof_group_grid
N_SPLITS = 5

groups_current = make_groups(train_df, GRID)
print(f"Current groups: {len(groups_current.unique())} unique groups")
print(f"Points per group: {groups_current.value_counts().describe()}")

# Analyze current folds
gkf = GroupKFold(n_splits=N_SPLITS)
fold_assignments = np.full(len(train_df), -1)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, groups=groups_current)):
    fold_assignments[va_idx] = fold

train_df['fold_current'] = fold_assignments

# Fold statistics
print("\n=== CURRENT GROUPING ANALYSIS ===")
for fold in range(N_SPLITS):
    fold_data = train_df[train_df['fold_current'] == fold]
    print(f"Fold {fold}: {len(fold_data)} points")
    print(f"  Lat range: {fold_data['Latitude'].min():.3f} - {fold_data['Latitude'].max():.3f}")
    print(f"  Lon range: {fold_data['Longitude'].min():.3f} - {fold_data['Longitude'].max():.3f}")
    if 'basin_id' in train_df.columns:
        print(f"  Unique basins: {fold_data['basin_id'].nunique()}")
        print(f"  Basin distribution: {fold_data['basin_id'].value_counts().head()}")

# Basin grouping if available
if 'basin_id' in train_df.columns:
    print("\n=== BASIN GROUPING ANALYSIS ===")
    groups_basin = train_df['basin_id'].fillna('unknown')
    print(f"Basin groups: {len(groups_basin.unique())} unique basins")

    fold_assignments_basin = np.full(len(train_df), -1)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, groups=groups_basin)):
        fold_assignments_basin[va_idx] = fold

    train_df['fold_basin'] = fold_assignments_basin

    for fold in range(N_SPLITS):
        fold_data = train_df[train_df['fold_basin'] == fold]
        print(f"Basin Fold {fold}: {len(fold_data)} points")

# KMeans grouping
print("\n=== KMEANS GROUPING ANALYSIS ===")
coords = train_df[['Latitude', 'Longitude']].values

for n_clusters in [8, 10, 12, 15]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coords)
    train_df[f'fold_kmeans_{n_clusters}'] = clusters

    print(f"KMeans {n_clusters} clusters:")
    for cluster in range(n_clusters):
        cluster_data = train_df[train_df[f'fold_kmeans_{n_clusters}'] == cluster]
        print(f"  Cluster {cluster}: {len(cluster_data)} points")

# Create visualizations
plt.figure(figsize=(20, 5))

# Current grouping
plt.subplot(1, 4, 1)
scatter = plt.scatter(train_df['Longitude'], train_df['Latitude'], c=train_df['fold_current'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('Current Grid Grouping (grid=0.2)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Basin grouping
if 'basin_id' in train_df.columns:
    plt.subplot(1, 4, 2)
    basin_colors = pd.Categorical(train_df['basin_id']).codes
    scatter = plt.scatter(train_df['Longitude'], train_df['Latitude'], c=basin_colors, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Basin Grouping')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

# KMeans 10
plt.subplot(1, 4, 3)
scatter = plt.scatter(train_df['Longitude'], train_df['Latitude'], c=train_df['fold_kmeans_10'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('KMeans Grouping (n=10)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# KMeans 15
plt.subplot(1, 4, 4)
scatter = plt.scatter(train_df['Longitude'], train_df['Latitude'], c=train_df['fold_kmeans_15'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('KMeans Grouping (n=15)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "spatial_audit_plots.png", dpi=150, bbox_inches='tight')
print(f"\nSaved spatial audit plots to: {PROJECT_ROOT / 'spatial_audit_plots.png'}")

# Save fold assignments
train_df.to_csv(PROJECT_ROOT / "spatial_audit_folds.csv", index=False)
print(f"Saved fold assignments to: {PROJECT_ROOT / 'spatial_audit_folds.csv'}")

print("\n=== SPATIAL LEAKAGE DIAGNOSIS ===")
print("Current grid grouping:")
print(f"- Grid size: {GRID} degrees")
print(f"- Unique groups: {len(groups_current.unique())}")
print(f"- Points per group: mean={groups_current.value_counts().mean():.1f}, std={groups_current.value_counts().std():.1f}")
print(f"- Fold sizes: {[len(train_df[train_df['fold_current']==f]) for f in range(N_SPLITS)]}")

if 'basin_id' in train_df.columns:
    print("Basin analysis:")
    basin_counts = train_df['basin_id'].value_counts()
    print(f"- Basins with >1 station: {len(basin_counts[basin_counts>1])}")
    print(f"- Max stations per basin: {basin_counts.max()}")
    # Check if basins are split
    basin_splits = []
    for basin in train_df['basin_id'].unique():
        basin_folds = train_df[train_df['basin_id']==basin]['fold_current'].unique()
        if len(basin_folds) > 1:
            basin_splits.append((basin, len(basin_folds)))
    print(f"- Basins split across folds: {len(basin_splits)}")

print("\nAudit complete.")