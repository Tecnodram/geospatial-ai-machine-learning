# Project Audit Report

## Project
EY Water Quality 2026 - Geospatial Machine Learning Pipeline

Date: 2026-03-13
Audit scope: final archival and portfolio closure (documentation, structure, reproducibility, experiment lineage)

## 1. Pipeline Description

The project implements a configuration-driven geospatial ML pipeline for multi-target water quality regression:
- Targets: Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus (DRP)
- Core training entrypoint: src/train_pipeline.py
- End-to-end runner: src/run_all.py
- Post-training blend recommender: src/batch_blends.py

Primary execution flow:
1. Read raw inputs and engineered external feature tables.
2. Build model-ready train and validation matrices by keying on Latitude, Longitude, Sample Date.
3. Apply basin-aware/group-aware CV via GroupKFold and fold-safe encoders.
4. Train target-specific regressors and generate CV metrics.
5. Fit final models on full training set and generate submission artifacts.
6. Persist experiment artifacts (config snapshots, CV reports, manifests) under experiments/exp_*.

## 2. Modeling Strategy (Verified)

Observed implementation patterns:
- Group-aware validation:
  - Basin-first grouping when basin_id exists.
  - Grid fallback grouping when basin metadata is unavailable.
- Leakage control:
  - Fold-safe basin target encoding computed only from training folds.
  - Keys normalized before merging to reduce key drift (Latitude, Longitude, Sample Date).
- Multi-model support:
  - ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, CatBoostRegressor.
  - Target-specific model overrides for DRP are supported via config.
- Target transforms:
  - none, winsor, clip_p95, clip_p99, sqrt, and DRP-specific transform controls.

## 3. Feature Sources

The repository combines Earth observation, climate, and hydrography data:
- Landsat features (training/validation CSVs)
- TerraClimate features (training/validation CSVs)
- CHIRPS precipitation features
- HydroBASINS and HydroRIVERS engineered features
- Additional geospatial covariates (elevation, soil, land cover)

Data dependencies currently expected by the pipeline:
- data/raw/water_quality_training_dataset.csv
- data/raw/submission_template.csv
- data/raw/landsat_features_training.csv
- data/raw/landsat_features_validation.csv
- data/raw/terraclimate_features_training.csv
- data/raw/terraclimate_features_validation.csv
- data/external/chirps_features_training.csv
- data/external/chirps_features_validation.csv
- data/external_geofeatures_plus_hydro_v2.csv

## 4. Experiment History and Artifacts

Evidence of extensive iterative experimentation exists under experiments/:
- experiment_index.csv (master index of run-level metrics)
- experiments.csv and audit CSVs
- Numerous immutable run folders (exp_YYYYMMDD_HHMMSS)
- Batch result trackers and sprint-level reports

Each mature run folder includes most or all of:
- config_snapshot.json
- cv_report.json
- feature_manifest_*.json
- submission*.csv

This is sufficient for run-level reconstruction and retrospective analysis.

## 5. Current Best Configuration (By CV Mean in experiment_index.csv)

Top indexed run:
- experiment_id: exp_20260307_003919
- cv_mean: 0.312895
- DRP cv_mean: 0.198120
- model family: ExtraTreesRegressor
- DRP y_mode: none
- feature count (DRP): 88
- run artifacts: experiments/exp_20260307_003919/

Run-level CV report values (from cv_report.json):
- Total Alkalinity mean: 0.410605
- Electrical Conductance mean: 0.329959
- DRP mean: 0.198120

## 6. Reproducibility Assessment

Strengths:
- Config-first design with serialized config snapshots per run.
- Experiment outputs and CV reports versioned in run folders.
- Deterministic seeds are present in major model configs.
- Group-aware CV and fold-safe encoding reduce optimistic leakage.

Gaps / caveats:
- Dependency manifests exist but were split (requirements_full.txt and requirements_lock.txt); a cleaner requirements.txt was missing before closure.
- Some feature extraction scripts contain machine-specific defaults (for example absolute Windows paths in one geofeature script) and require environment adjustment.
- Earth Engine and Snowflake credentials are external prerequisites.
- Multiple historical configs in root can make entrypoint selection ambiguous without documentation.

## 7. Submission Artifacts and Traceability

Submission outputs are present in submissions/ and submissions_batch/.
The pipeline also archives run-local outputs in experiment folders and maintains leaderboard_log.csv metadata. This gives acceptable traceability from run config -> CV report -> generated submission.

## 8. Snowflake Integration Status

Snowflake integration code exists under src/snowflake/ with support for:
- loading raw datasets,
- building model-ready tables,
- registering experiments.

Credential loading supports .snowflake.env and environment variables. Integration is operationally ready for experiment metadata archival.

## 9. Audit Verdict

Project status: READY FOR ARCHIVAL AND PORTFOLIO PUBLICATION.

Conditions satisfied for closure:
- End-to-end geospatial ML pipeline present.
- Experiment lineage and artifacts persisted.
- Data and feature dependencies identifiable.
- Documentation and reproducibility hardening completed in this closure pass.

Recommended post-closure practice:
- Keep all future changes additive (new configs, new run folders, new docs) and avoid mutating historical run artifacts.
