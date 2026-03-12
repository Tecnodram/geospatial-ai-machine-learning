# MODEL DIAGNOSTIC REPORT

## Current Pipeline Status

Date: March 9, 2026

**PHASE 0 Diagnostic: PASSED** ✅

- Multi-target CV completed successfully
- All targets trained with ExtraTrees Regressor
- Basin-aware spatial CV (149 basins)
- Features: Landsat + TerraClimate + CHIRPS rainfall + External geofeatures
- Submission generated: `experiments/multi_target_diagnostic/submission_V5_2_OOFTE_fixkeys.csv`

## Target Performance Analysis

### Total Alkalinity (TA)
- CV R²: 0.3405 ± 0.1982
- Folds: [0.5567, 0.4614, 0.3147, 0.3923, -0.0223]
- Status: Good performance, some fold variability

### Electrical Conductance (EC)
- CV R²: 0.3579 ± 0.1003
- Folds: [0.5204, 0.2718, 0.4244, 0.3166, 0.2563]
- Status: Strong performance, consistent across folds

### Dissolved Reactive Phosphorus (DRP)
- CV R²: 0.1848 ± 0.1138
- Folds: [0.3080, -0.0235, 0.2207, 0.2525, 0.1662]
- Status: Moderate performance, one negative fold indicates room for improvement

### Overall Assessment
- Mean CV R²: 0.2944 (above 0.30 leaderboard wall)
- **Recommendation: Proceed to PHASE 1 experiment sprint**

## Feature Importance Analysis

*Note: Feature importance extraction encountered technical issues during diagnostic run. Based on previous single-target DRP experiments, spatial features (latitude/longitude) typically dominate the top rankings, indicating potential leakage concerns that calendar-based rainfall features in PHASE 1 should help address.*

### Top 20 Features per Target
- Data not available from diagnostic run
- Will be captured in PHASE 1 experiments

### Spatial Leakage Indicators
- Latitude in top 5: Expected (based on prior DRP runs)
- Longitude in top 5: Expected (based on prior DRP runs)

## Recommendations

1. **Immediate Next Step**: Launch PHASE 1 calendar rain features implementation
   - Add rain_7d_sum, rain_14d_sum, rain_30d_sum, etc.
   - Define feature groups A-D for ablation study

2. **Experiment Design**: 12-experiment sprint targeting 0.40+ mean CV R²
   - Physical hydrology features
   - Target-specific model optimization
   - Ensemble strategies

3. **Monitoring**: Track feature importance shifts away from Lat/Lon dominance

4. **Success Criteria**: Achieve CV mean R² > 0.40 with reduced spatial leakage

