# EY Water Quality 2026

Geospatial Machine Learning Framework for Water Quality Monitoring

## Overview

This repository implements an end-to-end geospatial machine learning framework to estimate key water quality variables from Earth observation, climate, and hydrological data.

Predicted indicators:
- Total Alkalinity
- Electrical Conductance
- Dissolved Reactive Phosphorus (DRP)

The project combines remote sensing features, climate signals, and watershed structure to build reproducible predictive models suitable for environmental analytics and decision support.

## Motivation

Water quality monitoring remains sparse in many regions due to limited in-situ sampling coverage, operational costs, and delayed reporting cycles. A scalable geospatial ML approach can improve spatial coverage and enable earlier insight into watershed stress patterns.

This project demonstrates a practical pathway from raw geospatial data to validated predictions under leakage-aware spatial cross-validation.

## Data Sources

Core data families:
- Landsat-derived features
- TerraClimate covariates
- CHIRPS precipitation features
- HydroBASINS watershed attributes
- HydroRIVERS network descriptors
- Engineered geospatial terrain and soil context features

## Modeling Strategy

- Multi-target regression with target-aware model configuration.
- Basin-aware GroupKFold validation to reduce spatial leakage.
- Fold-safe target encoding for basin-level signal integration.
- Target-specific transform and model controls, especially for DRP.
- Conservative ensemble and blend workflow for robust deployment behavior.

Reference implementation files:
- src/train_pipeline.py
- src/run_all.py
- src/batch_blends.py

## Feature Engineering

Geospatial feature engineering scripts are organized in feature_engineering/:
- feature_engineering/extract_chirps_features.py
- feature_engineering/build_hydro_features.py
- feature_engineering/external_geofeatures.py

These scripts build precipitation and hydrology-aware covariates that are merged into model-ready datasets using Latitude, Longitude, and Sample Date keys.

## Experiment Methodology

- Configuration-driven experiments with immutable run folders under experiments/exp_*.
- Stored artifacts include config snapshots, CV reports, feature manifests, and generated submissions.
- Performance history is tracked in CSV registries for reproducibility and retrospective analysis.

For closure-level synthesis, see:
- docs/experiment_summary.md
- docs/project_audit_report.md

## Reproducibility

1. Create a Python 3.11+ environment.
2. Install dependencies from requirements.txt.
3. Place required raw and external data files in data/.
4. Run the pipeline with:

```bash
python src/run_all.py --config config.yml
```

Detailed guide:
- docs/reproduce_pipeline.md

## Current Best Indexed Baseline

From experiments/experiment_index.csv at project closure:
- Experiment: exp_20260307_003919
- CV mean: 0.312895
- DRP CV mean: 0.198120

This baseline is preserved with complete run artifacts in experiments/exp_20260307_003919/.

## Environmental Impact

A reproducible geospatial monitoring workflow can help:
- prioritize field sampling resources,
- identify watershed risk hotspots,
- support evidence-based nutrient management,
- improve resilience planning under climate variability.

By combining open geospatial data streams with hydrological context, this framework supports scalable, lower-latency water quality intelligence.

## Future Work

1. Uncertainty-aware prediction intervals for operational risk communication.
2. Regional transfer and domain adaptation across hydro-climatic regimes.
3. Graph-based river-network modeling.
4. Physics-informed constraints for nutrient transport realism.
5. Near-real-time ingestion for monitoring dashboards.

## Global Extension Potential

This framework is intentionally designed for adaptation beyond a single study region. It can be extended to:
- other countries and transboundary basins,
- watershed-scale monitoring programs,
- climate change impact analysis,
- environmental decision-support systems.

Global scalability is enabled by widely available geospatial inputs:
- satellite imagery,
- climate reanalysis and precipitation products,
- hydrological basin and river network datasets.

## Repository Documentation

- docs/project_audit_report.md
- docs/methodology.md
- docs/modeling_strategy.md
- docs/experiment_summary.md
- docs/reproduce_pipeline.md
- docs/project_closure.md
- docs/snowflake_experiment_log.csv
- data_structure.md

### Experiment Tracking

Experiments are logged in Snowflake to ensure reproducibility and traceability.

Example logged experiment:

| experiment_id | model | cv_score |
|---|---|---|
| exp_20260307_003919 | ExtraTreesRegressor | 0.3129 |

The exported experiment log is available in:

docs/snowflake_experiment_log.csv