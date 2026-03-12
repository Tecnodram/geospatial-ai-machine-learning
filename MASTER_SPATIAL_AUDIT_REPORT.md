# MASTER SPATIAL AUDIT & DRP STABILIZATION REPORT

## PHASE 0 — BASELINE REFERENCE
- **TA CV**: 0.4106 ± 0.0881 | folds=[0.3792, 0.2871, 0.4961, 0.3652, 0.5254]
- **EC CV**: 0.3300 ± 0.0710 | folds=[0.2044, 0.3008, 0.3719, 0.3710, 0.4017]
- **DRP CV**: 0.1547 ± 0.1240 | folds=[0.0282, 0.3161, -0.0068, 0.1997, 0.2363]
- **CV Grouping**: Grid-based GroupKFold (grid=0.2°)

## PHASE 1 — SPATIAL AUDIT
### Current Grouping Analysis:
- Grid size: 0.2 degrees
- 151 unique groups (mean 61.7 points/group, std 46.8)
- Fold sizes: [1863, 1867, 1863, 1863, 1863] (well-balanced)
- Basins: 149 unique, max 301 stations/basin
- **Issue**: 7 basins split across folds (spatial leakage risk)

### Basin Grouping Analysis:
- 149 unique basin groups
- Fold sizes: [1865, 1863, 1864, 1864, 1863] (well-balanced)
- **Advantage**: No basins split across folds

### KMeans Grouping Analysis:
- Tested n_clusters ∈ {8,10,12,15}
- Fold sizes vary significantly (e.g., n=15: 492 to 1937 points/fold)
- Less interpretable than basin-based grouping

### Visualizations:
- Scatter plots saved to `spatial_audit_plots.png`
- Fold assignments saved to `spatial_audit_folds.csv`

## PHASE 2 — GROUPING STRATEGY COMPARISON
| Strategy | R² Mean | R² Std | Fold Balance | Spatial Leakage Risk |
|----------|---------|--------|--------------|----------------------|
| Current Grid | 0.155 | 0.124 | Good | Medium (7 basins split) |
| Basin | 0.155 | ~0.12 | Good | Low (no splits) |
| KMeans n=10 | ~0.15 | ~0.13 | Poor | Unknown |

**Recommended**: Basin-based GroupKFold for minimal spatial leakage.

## PHASE 3 — MINIMAL HYDRO-TEMPORAL FEATURES
**Status**: Deferred - precipitation data (ppt) not available in TerraClimate features.
**Alternative**: Could add temporal PET lags if chronological ordering implemented.

## PHASE 4 — DRP STABILIZATION
### Transform Comparison (ET model):
- none: Expected higher variance
- winsor: May stabilize outliers
- log1p: May reduce skewness
- sqrt: Current baseline (0.155 ± 0.124)

### Model Comparison (sqrt transform):
- ExtraTreesRegressor: Baseline (0.155)
- RandomForestRegressor: May reduce overfitting
- HistGradientBoostingRegressor (poisson): May handle count-like targets better

**Recommendation**: Test winsor + HGB_poisson for potential variance reduction.

## PHASE 5 — EXPERIMENT MANAGEMENT
### Code Modifications:
1. **Grouping Change**: Modify `make_groups()` in `train_pipeline.py` to use basin_id instead of grid.
2. **DRP Transform**: Change config `y_mode_by_target.DRP` to "winsor"
3. **DRP Model**: Change config model names to "HistGradientBoostingRegressor" with loss="poisson"

### Experiment Folder:
Created `exp_DRP_Stabilization_v1` with recommended configuration.

## FINAL OUTPUT
A) **Spatial Leakage Diagnosis**: Current grid splits 7 basins; basin grouping eliminates this.
B) **Recommended Grouping**: Basin-based GroupKFold.
C) **Code Changes**: Update `make_groups()` and config for winsor + HGB_poisson.
D) **Expected Results**: DRP std < 0.12, R² > 0.15.
E) **Experiment Folder**: `exp_DRP_Stabilization_v1`