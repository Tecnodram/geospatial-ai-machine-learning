# EY Water Quality - Feature Signal Upgrade Roadmap (Phase Next)

Date: 2026-03-11

## Scope and guardrails

- Objective: lift leaderboard performance from ~0.30 plateau toward ~0.40 by adding scientifically grounded signal families.
- Constraints:
  - No DWS or external chemical measurement datasets.
  - Local model training remains the source of truth.
  - Snowflake is used for feature engineering, orchestration, and audit/state management.
  - Immutable artifact rules remain enforced under experiments/exp_*/.

## A) Feature roadmap

### Family A: Basin/Catchment context features

Exact feature list:
- basin_area_km2
- upstream_area_km2
- catch_to_basin_ratio
- dist_to_river_m
- river_discharge_cms
- elevation
- slope
- soil_clay_0_5
- soil_ph_0_5
- soil_soc_0_5
- landcover
- hydro_region_k4
- hydro_region_k6
- hydro_cluster_k8

Derived context interactions:
- area_x_slope = upstream_area_km2 * slope
- ph_x_area = soil_ph_0_5 * upstream_area_km2
- clay_x_area = soil_clay_0_5 * upstream_area_km2
- discharge_contact_proxy = upstream_area_km2 / (river_discharge_cms + 1)

Source tables/files:
- Snowflake RAW:
  - EY_WQ.RAW.EXTERNAL_GEOFEATURES_PLUS_HYDRO_V2
  - EY_WQ.RAW.WATER_QUALITY_TRAINING_DATASET
  - EY_WQ.RAW.SUBMISSION_TEMPLATE
- Local fallback:
  - data/external_geofeatures_plus_hydro_v2.csv

Build location:
- Primary: Snowflake (deterministic, reusable, auditable joins).
- Secondary: local Python only for fold-safe cluster assignment if needed.

Scientific rationale:
- River chemistry is strongly controlled by basin scale, lithology/soil, and flow-path geometry.
- These features encode static spatial controls on buffering, transport residence time, and source loading potential.

### Family B: Temporal memory / lag features

Exact feature list:
- rain_lag_1
- rain_lag_3
- rain_lag_7
- rain_lag_14
- rain_roll_3
- rain_roll_7
- rain_roll_14
- rain_roll_30
- rain_roll_60
- rain_roll_90
- pet_roll_30
- rain_to_pet_30d
- rain_to_pet_60d
- dry_days_before_sample
- rain_anomaly_30d
- sin_doy
- cos_doy
- sin_month
- cos_month

Source tables/files:
- Snowflake RAW:
  - EY_WQ.RAW.CHIRPS_FEATURES_TRAINING
  - EY_WQ.RAW.CHIRPS_FEATURES_VALIDATION
  - EY_WQ.RAW.TERRACLIMATE_FEATURES_TRAINING
  - EY_WQ.RAW.TERRACLIMATE_FEATURES_VALIDATION
- Local fallback:
  - data/external/chirps_features_training.csv
  - data/external/chirps_features_validation.csv

Build location:
- Primary: Snowflake window functions for deterministic lag/rolling features.
- Secondary: local Python when fold-safe lag constraints require strict train-fold-only generation.

Scientific rationale:
- Nutrient concentration responds to antecedent wetness and runoff memory, not only same-day weather.
- Seasonal cycles capture phenology and hydrologic regime shifts influencing TA/EC/DRP.

### Family C: Upstream hydrologic pressure proxies

Exact feature list:
- upstream_cropland_pressure = upstream_area_km2 * cropland_fraction_5km
- upstream_urban_pressure = upstream_area_km2 * urban_fraction_5km
- upstream_soil_clay_pressure = upstream_area_km2 * soil_clay_0_5
- upstream_soc_pressure = upstream_area_km2 * soil_soc_0_5
- upstream_rain_30d_pressure = upstream_area_km2 * rain_roll_30
- upstream_rain_60d_pressure = upstream_area_km2 * rain_roll_60
- nutrient_transport_proxy = upstream_area_km2 * rain_roll_30 / (river_discharge_cms + 1)
- runoff_transfer_index = rain_roll_30 * slope * exp(-dist_to_river_m / 5000)
- connectivity_proxy = upstream_area_km2 / (dist_to_river_m + 1)
- stream_power_proxy = upstream_area_km2 * slope

Source tables/files:
- Snowflake FEAT_SIG intermediate outputs from Family A + Family B.

Build location:
- Snowflake first (materialized table-level proxies).
- Local Python only for fold-safe neighbor summaries if used.

Scientific rationale:
- DRP and EC spikes often reflect transport-conditioned source loading: land use x flow memory x network connectivity.

## B) Snowflake orchestration plan

### Schema and table plan

Use a dedicated schema for versioned feature engineering:
- EY_WQ.FEAT_SIG

Versioned tables:
- EY_WQ.FEAT_SIG.STATION_CONTEXT_V1
- EY_WQ.FEAT_SIG.TEMPORAL_MEMORY_V1_TRAIN
- EY_WQ.FEAT_SIG.TEMPORAL_MEMORY_V1_VALID
- EY_WQ.FEAT_SIG.UPSTREAM_PRESSURE_V1_TRAIN
- EY_WQ.FEAT_SIG.UPSTREAM_PRESSURE_V1_VALID
- EY_WQ.FEAT_SIG.TRAIN_MODELREADY_SIG_V1
- EY_WQ.FEAT_SIG.VALID_MODELREADY_SIG_V1

Audit tables:
- EY_WQ.AUDIT.FEATURE_BUILD_REGISTRY (feature_version, source_snapshot, sql_hash, build_ts, row_counts)
- EY_WQ.AUDIT.EXPERIMENT_REGISTRY (already existing; continue using)
- EY_WQ.AUDIT.MASTER_PROJECT_STATE (already existing; continue using)

### Naming convention

- Feature version: SIG_V{major}_{minor} (example: SIG_V1_0).
- Experiment config tags: feature_pack=sig_v1_0_familyA, sig_v1_0_familyB, sig_v1_0_familyC, sig_v1_0_combo.
- Submission artifact names stay immutable under experiments/exp_*/.

### Build execution pattern

1. Build/refresh FEAT_SIG versioned tables (SQL).
2. Validate row counts, key uniqueness, null-key constraints.
3. Snapshot build metadata in AUDIT.FEATURE_BUILD_REGISTRY.
4. Export to local cache as needed for fold-safe training.

## C) Staged experiment plan

### Sprint 1: Basin features only

- Enable Family A only.
- Keep model family anchor-consistent for DRP (ET + TE true + y_mode none where applicable).
- Run 3-5 tightly controlled configs (no broad sweeps).
- Success criteria:
  - DRP CV improves without major anchor divergence.
  - DRP corr_to_v44 >= 0.97 for conservative candidates.

### Sprint 2: Temporal features only

- Enable Family B only.
- Keep all non-temporal new features off.
- Focus on lag windows robustness and leakage safety.
- Success criteria:
  - DRP fold variance decreases.
  - Corr/MAE to V4_4 remains conservative for top candidates.

### Sprint 3: Upstream proxies only

- Enable Family C only (with required base columns from existing modelready).
- Prioritize physically plausible proxy scaling.
- Success criteria:
  - DRP tail behavior improves (high quantile alignment) without destabilizing TA/EC.

### Sprint 4: Combine only winning families

- Combine only families that beat their own sprint baseline on both CV and anchor-similarity risk controls.
- Candidate combinations:
  - A+B
  - A+C
  - B+C
  - A+B+C only if each family independently positive.
- Keep submission exploration conservative around validated anchors/blends.

## D) Residual-modeling roadmap (anchored on V4_4)

### Baseline prediction

- Baseline predictor: V4_4 anchor submission and/or its model-equivalent local reproduction.

### Residual target definition

For each target t:
- residual_t = y_true_t - y_pred_baseline_t

Priority target:
- DRP residual first (largest leaderboard sensitivity).

### Candidate residual models

- ExtraTreesRegressor (residual target)
- RandomForestRegressor (residual target)
- HistGradientBoostingRegressor (residual target; not replacing anchor family, only residual correction)
- Optional linear meta-corrector (Ridge/ElasticNet) for stability

### Residual correction strategy

- corrected_pred_t = baseline_pred_t + alpha_t * residual_pred_t
- Tune alpha_t conservatively (for DRP: 0.05 to 0.30 search grid)
- Clip DRP to non-negative and preserve submission shape/order.

### Validation strategy

- GroupKFold by basin_id only.
- Strict fold-safe residual training (no leakage from validation residuals).
- Decision gates:
  - mean CV uplift
  - DRP corr_to_v44 and MAE_to_v44 risk bounds
  - no catastrophic fold outliers

## E) Exact next implementation steps

1. Run Snowflake scaffold build:
- python src/snowflake/run_feature_signal_scaffold.py

2. Register feature build metadata row in AUDIT.FEATURE_BUILD_REGISTRY.

3. Create three config files for sprints:
- config_exp_sig_s1_basin.yml
- config_exp_sig_s2_temporal.yml
- config_exp_sig_s3_upstream.yml

4. Add feature-pack switch handling in local pipeline (read-only from FEAT_SIG exports; immutable outputs unchanged).

5. Execute Sprint 1-3 runs, each with conservative blend checks against:
- submission_V4_4_DRP_tuned_ET_fixorder.csv
- blend4_v44_80_134704_20_rank.csv
- blend_v44_85_134704_15_rank.csv

6. Auto-generate sprint audit summaries:
- DRP min/max/median
- corr_to_v44
- mae_to_v44
- leaderboard_log entry stubs

7. Launch Sprint 4 using only winning families.

8. Start residual stage on top Sprint 4 winner (and optionally V4_4 anchor), with alpha-constrained residual blending.
