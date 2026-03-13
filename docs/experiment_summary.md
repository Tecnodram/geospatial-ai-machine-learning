# Experiment Summary

## Scope

This document archives the major modeling phases, observed performance trends, and closure-level conclusions for the EY Water Quality geospatial ML project.

## Evolution Timeline

1. Baseline multi-target tree ensembles
- Established stable TA and EC performance.
- DRP remained comparatively low and high-variance.

2. Transform and robustness sweep for DRP
- Tested none, winsor, sqrt, clip variants.
- Performance improved when transforms and feature sets were aligned to DRP behavior.

3. Hydro and rain signal refinement
- Added basin, river, and precipitation interaction signals.
- Improved sensitivity to watershed transport dynamics.

4. Target-specific model specialization
- Compared ExtraTrees, RandomForest, HistGradientBoosting, and CatBoost variants.
- Preserved a conservative, reproducible tree-ensemble baseline at closure.

## Best Indexed Runs (From experiments/experiment_index.csv)

| Rank | Experiment ID | CV Mean | TA CV Mean | EC CV Mean | DRP CV Mean | DRP Model | DRP Mode |
|---|---|---:|---:|---:|---:|---|---|
| 1 | exp_20260307_003919 | 0.312895 | 0.410605 | 0.329959 | 0.198120 | ExtraTreesRegressor | none |
| 2 | exp_20260306_024632 | 0.312315 | - | - | 0.192261 | HistGradientBoostingRegressor | none |
| 3 | exp_20260307_004254 | 0.311980 | - | - | 0.195376 | ExtraTreesRegressor | winsor |
| 4 | exp_20260306_024032 | 0.311948 | - | - | 0.192677 | HistGradientBoostingRegressor | none |

Notes:
- TA/EC per-target means are explicitly available from cv_report.json for run exp_20260307_003919.
- Some index rows retain only aggregate metrics and DRP details.

## Representative Sprint-Level Tracker (Curated)

| Phase | Representative Experiment | TA R2 | EC R2 | DRP R2 | Mean R2 | Observation |
|---|---|---:|---:|---:|---:|---|
| Sprint 1 baseline | exp_sprint1_01_baseline_multi | 0.341 | 0.358 | 0.185 | 0.295 | Baseline multi-target setup |
| Sprint 1 calendar/rain | exp_sprint1_02_calendar_rain | 0.342 | 0.359 | 0.193 | 0.298 | Small DRP gain from temporal context |
| Sprint 3 hydro ablation | exp_sprint3_08_hydro_ablation | 0.315 | 0.357 | 0.188 | 0.287 | Removing hydro signals hurts TA/DRP |
| Sprint 3 rain ablation | exp_sprint3_09_rain_ablation | 0.337 | 0.356 | 0.203 | 0.299 | CHIRPS effect is target dependent |
| Sprint 4 hydro interactions | exp_sprint4_11_hydro_interactions | 0.337 | 0.356 | 0.203 | 0.299 | Hydro interactions help DRP robustness |

## DRP Modeling Challenges

Core challenges observed across runs:
- High fold-to-fold volatility due to episodic nutrient dynamics.
- Sensitivity to transform choice and upper-tail behavior.
- Strong dependence on hydro-connectivity and moisture interactions.
- Increased risk of overfit when model complexity is raised without stronger spatial constraints.

## Lessons Learned

1. Spatially safe CV is non-negotiable for realistic performance estimates.
2. Fold-safe target encoding is essential when adding basin-level signal.
3. DRP requires dedicated treatment; one-size-fits-all settings underperform.
4. Hydrography and precipitation interactions provide meaningful predictive signal.
5. Conservative, interpretable ensembles gave the best reproducible trade-off for closure.

## Closure Recommendation

For portfolio baseline and reproducibility demonstrations, use:
- experiment_id: exp_20260307_003919
- config snapshot: experiments/exp_20260307_003919/config_snapshot.json
- CV report: experiments/exp_20260307_003919/cv_report.json
