# Sprint 1 Decision Report - Basin/Catchment Context (Controlled Single Run)

Date: 2026-03-11
Run ID: exp_20260311_212240
Run Path: experiments/exp_20260311_212240
Config: config_exp_sig_s1_basin.yml

## Scope check

- Exactly one controlled basin-only experiment was executed.
- No DWS or external chemical measurement datasets were used.
- Training remained local.
- Immutable artifacts were preserved under experiments/exp_20260311_212240/.
- Snowflake audit updates were executed where supported.

## A) Exact new basin features used

Feature pack switch: features.basin_context.enabled=true

Added feature columns:
- landcover_urban
- landcover_cropland
- basin_slope_loading
- basin_clay_pressure
- basin_ph_buffer
- discharge_contact_proxy
- connectivity_proxy
- landuse_pressure_proxy
- basin_mean_soil_clay_0_5
- basin_mean_soil_ph_0_5
- basin_mean_landcover_urban
- basin_mean_landcover_cropland
- basin_mean_slope
- basin_mean_elevation

Count impact:
- DRP feature count moved from 104 (reference exp_20260311_134704) to 109 (Sprint 1 run).

## B) Build location (Snowflake vs local Python)

- Built and consumed in this run: local Python in src/train_pipeline.py via add_basin_context_pack().
- Snowflake status: feature scaffold exists (FEAT_SIG schema/tables), but Sprint 1 run used local build path for controlled minimal integration.

## C) CV results by target

From experiments/exp_20260311_212240/cv_report.json:
- Total Alkalinity: mean=0.339618, std=0.131399
- Electrical Conductance: mean=0.340887, std=0.114973
- Dissolved Reactive Phosphorus: mean=0.164920, std=0.104937
- Mean CV: 0.281808

Reference comparison (exp_20260311_134704 from experiment_index.csv):
- Mean CV: 0.286501
- DRP CV: 0.164995

Delta (Sprint1 - Reference):
- Mean CV delta: -0.004693
- DRP CV delta: -0.000075

## D) Did DRP improve meaningfully?

No.

- DRP moved from 0.164995 to 0.164920 (slight decline, effectively flat).
- Change magnitude is not meaningful and does not justify claiming uplift.

## E) Is this family worth keeping for Sprint 4 combination?

Conditional keep (not as standalone winner).

- As a standalone family, basin-context pack did not improve performance.
- Keep as optional interaction support for Sprint 4 only if paired with a family that shows clear standalone gain (temporal memory or upstream pressure).
- If included later, prefer a trimmed subset (top-impact basin features) rather than full pack.

## F) Recommended next move

1. Proceed to Sprint 2 (temporal-only) with one controlled run, keeping the same model/hyperparameter backbone.
2. Keep basin_context disabled in default path; preserve the switch for future combinations.
3. For Sprint 4 eligibility, require each family to pass:
   - positive DRP CV delta vs reference
   - non-negative mean CV delta
   - no increase in fold instability.

## Audit and artifact status

Immutable outputs present:
- experiments/exp_20260311_212240/submission.csv
- experiments/exp_20260311_212240/config_snapshot.json
- experiments/exp_20260311_212240/cv_report.json
- experiments/exp_20260311_212240/metadata.json
- experiments/exp_20260311_212240/run_meta.json

Tracking updates:
- experiments/experiment_index.csv updated with exp_20260311_212240
- Snowflake AUDIT.EXPERIMENT_REGISTRY updated via register_experiment.py
- Snowflake AUDIT.MASTER_PROJECT_STATE refreshed via build_master_project_state.py
