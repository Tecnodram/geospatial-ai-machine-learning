# Methodology

## Objective

Build a reproducible geospatial machine learning framework to estimate water quality indicators from multisource Earth observation and hydrological data.

Targets:
- Total Alkalinity
- Electrical Conductance
- Dissolved Reactive Phosphorus (DRP)

## Data Integration Strategy

The modeling dataset is assembled by spatial-temporal key alignment using:
- Latitude
- Longitude
- Sample Date

Feature blocks:
- Remote sensing signatures from Landsat
- Climate covariates from TerraClimate
- Precipitation signals from CHIRPS
- Hydrological geometry and connectivity signals from HydroBASINS and HydroRIVERS
- Additional static geospatial descriptors (for example elevation, soil, land cover)

## Feature Engineering Principles

1. Hydro-ecological realism
- Encode proximity and connectedness to river networks and basin topology.

2. Temporal alignment discipline
- Keep date-level joins explicit and key-safe.

3. Scale-aware transformations
- Use bounded transforms (winsorization, clipping, sqrt) when target or feature distributions are heavy-tailed.

4. Conservative complexity
- Prefer interpretable engineered features and robust tree ensembles over opaque high-variance architectures.

## Validation Design

Spatial leakage risk is managed with group-based CV:
- Primary grouping: basin_id when available.
- Fallback grouping: coarse spatial grids.

This prevents neighboring samples in shared hydrologic structures from leaking between folds.

Fold-safe target encoding is applied inside each training fold only, then mapped to validation fold, preserving strict separation.

## Modeling Design

The pipeline supports per-target model specialization:
- Shared baseline family: tree ensembles (ExtraTrees / RandomForest).
- Optional target-specific overrides for DRP using HistGradientBoosting or ExtraTrees.

This enables robust handling of heterogeneous target behaviors:
- TA and EC: generally more stable broad-signal responses.
- DRP: noisier, sparse, and event-driven response, requiring stronger regularization and targeted transforms.

## Experiment Management

Each run stores immutable artifacts under experiments/exp_*:
- config_snapshot.json
- cv_report.json
- feature manifests
- generated submission file(s)

An experiment index tracks run_id, CV aggregates, model family, DRP mode, feature counts, and output paths.

## Operational Reproducibility

Reproducibility is based on:
- config-driven execution,
- deterministic random seeds,
- explicit artifact persistence,
- versioned scripts and reports,
- documented external credential requirements (Earth Engine, Snowflake).

## Limitations

- Site-level process dynamics are partially observed from remote proxies.
- DRP remains the hardest target due to episodic transport and chemistry nonlinearity.
- Transferability across hydro-climatic regimes requires local recalibration and validation.
