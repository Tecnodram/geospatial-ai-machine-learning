# Modeling Strategy

## High-Level Architecture

The project uses a target-aware multi-regressor architecture with shared preprocessing and per-target model control.

Pipeline stages:
1. Build model-ready matrix from merged geospatial data.
2. Construct spatially safe CV folds (basin-aware GroupKFold).
3. Train target-specific models with optional target transforms.
4. Produce out-of-fold diagnostics and final predictions.
5. Generate candidate blends and select stable recommendation.

## Why Group-Based CV

Random CV overestimates performance in spatial tasks due to local autocorrelation.

The strategy enforces group splits by basin_id (or grid fallback), so training and validation folds remain spatially separated. This is the core leakage-prevention mechanism.

## Target-Wise Modeling

### Total Alkalinity
- Strong baseline with ExtraTrees-style models.
- Benefits from broad geospatial and climate covariates.

### Electrical Conductance
- Similar ensemble approach with optional alternate regressor testing.
- Sensitive to hydro-climate context and spatial priors.

### Dissolved Reactive Phosphorus (DRP)
- Most difficult target with high fold variance.
- Evaluated with specialized transforms and model families.
- Often improved by hydro-interaction features and conservative regularization.

## Feature Regimes

Core feature regime includes:
- Landsat reflectance-derived indicators
- TerraClimate covariates
- CHIRPS precipitation
- HydroBASINS and HydroRIVERS descriptors

Enhanced DRP regime includes:
- moisture-hydrology interactions,
- logarithmic hydro scalings,
- proxy indices intended to represent transport and retention dynamics.

## Ensemble and Blend Layer

After model training, a blend layer creates conservative candidate outputs and score proxies. This supports robust selection under target instability, especially for DRP.

## Model Selection Criteria

Selection balances:
- cross-validated mean performance,
- DRP stability,
- feature reproducibility,
- leakage-safe behavior,
- operational maintainability.

## Current Closure Baseline

Best indexed run (by cv_mean in experiment_index.csv):
- exp_20260307_003919
- cv_mean: 0.312895
- DRP cv_mean: 0.198120
- model family: ExtraTreesRegressor
- DRP mode: none

This baseline is preserved as reference for portfolio reproducibility.
