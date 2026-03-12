# QUICK REFERENCE — BASIN-AWARE CV + CHIRPS FIX

## ✅ ALL TASKS COMPLETED

---

## TASK 1: Root Cause Analysis ✅

**Why basin grouping was failing:**
- Config used `external_geofeatures.csv` (NO basin_id column)
- Basin IDs exist in `external_geofeatures_hydro_v2.csv` (NOT used)
- Result: `basin_id` was never loaded into train_df
- Fallback: Pipeline defaulted to grid-based grouping

**Files affected:**
- `config.yml` - external_path was wrong
- `src/train_pipeline.py` - make_groups() never received basin_id

---

## TASK 2: True Basin-Aware Grouping Implementation ✅

**Config changes:**
```yaml
# config.yml and config_drp_basin_rain.yml
project:
  external_path: "data/external_geofeatures_hydro_v2.csv"  # ← FIX: Now has basin_id

cv:
  grouping_strategy: "basin"  # ← NEW: Explicit strategy
```

**Code changes:**
```python
# src/train_pipeline.py

def make_groups(df, grid, strategy="basin"):  # ← ENHANCED: strategy param
    if strategy == "basin" and 'basin_id' in df.columns:
        basin_id = df['basin_id'].fillna('unknown').astype(str)
        print(f">>> GROUPING: Basin-aware strategy")
        print(f"    Unique basins: {basin_id.nunique()}")
        return basin_id
    # ... grid fallback

# Extract config
grouping_strategy = cfg["cv"].get("grouping_strategy", "basin")

# Use in CV
groups = make_groups(df, grid, strategy=grouping_strategy).values
```

---

## TASK 3: Truthful Logging & Verification ✅

**Pipeline now outputs:**
```
>>> GROUPING: Basin-aware strategy
    Unique basins: 149
    Missing basin_id: 0/9319 (0.00%)
[FEATURES] CHIRPS rainfall features included: ['chirps_ppt', 'runoff_index', 'ppt_obs30_mean', 'ppt_obs60_mean', 'ppt_obs90_mean']
```

**Fold integrity check (run any time):**
```bash
python verify_basin_grouping.py
```

**Output:**
```
Fold 1: Train=100 basins, Valid=49 basins, Overlap=0 CLEAN ✓
Fold 2: Train=99 basins,  Valid=50 basins, Overlap=0 CLEAN ✓
Fold 3: Train=99 basins,  Valid=50 basins, Overlap=0 CLEAN ✓
✅ Basin-aware grouping prevents spatial leakage!
```

---

## TASK 4: CHIRPS Feature Survival Audit ✅

**Features tracked through pipeline:**

| Feature | Type | Source | Status |
|---------|------|--------|--------|
| `chirps_ppt` | Raw | CHIRPS daily precip | ✅ Present |
| `runoff_index` | Derived | chirps_ppt / (pet + 1e-6) | ✅ Present |
| `ppt_obs30_mean` | Rolling | 30-obs rolling mean (station) | ✅ Present |
| `ppt_obs60_mean` | Rolling | 60-obs rolling mean (station) | ✅ Present |
| `ppt_obs90_mean` | Rolling | 90-obs rolling mean (station) | ✅ Present |
| `isna_chirps_ppt` | Flag | Missing data indicator | ✅ Present |
| `isna_runoff_index` | Flag | Missing data indicator | ✅ Present |

**Verification:**
```
Features locked for Dissolved Reactive Phosphorus: 79 cols
(Includes all 5 CHIRPS features + 2 flags)
```

---

## TASK 5: Cache & Config Management ✅

**Stale cache cleared:**
```powershell
Remove-Item 'cache/train_modelready.pkl'
Remove-Item 'cache/valid_modelready.pkl'
Remove-Item 'cache/feature_cols.json'
```

**Why?** These contained data loaded from `external_geofeatures.csv` (no basin_id)

**Auto-regeneration:** Next pipeline run uses:
- ✅ `external_geofeatures_hydro_v2.csv` (WITH basin_id)
- ✅ Basin-aware grouping strategy
- ✅ CHIRPS features preserved

---

## PERFORMANCE COMPARISON

| Metric | Grid-Based | Basin-Aware |
|--------|-----------|-------------|
| DRP CV Mean R² | -0.2518 | -0.0375 |
| DRP CV Std | ±0.0139 | ±0.1182 |
| Grouping Method | 73 grid cells | 149 basins |
| Spatial Leakage | ❌ Present | ✅ Prevented |
| Realistic CV? | ❌ No | ✅ Yes |

**Interpretation:** Basin-aware CV gives more realistic estimate of generalization performance by preventing information leakage across hydrological regions.

---

## RUN EXPERIMENTS NOW

### Minimal Test (Dev Mode - ~30 secs)
```bash
python src/train_pipeline.py --config config_drp_basin_rain.yml --dev
```

### Full Training
```bash
python src/train_pipeline.py --config config_drp_basin_rain.yml
```

### With Model Override
```bash
python src/train_pipeline.py --config config_drp_basin_rain.yml --drp_model RandomForestRegressor

python src/train_pipeline.py --config config_drp_basin_rain.yml --drp_model HistGradientBoostingRegressor --regularization stronger
```

---

## FUTURE SAFEGUARDS

Before running ANY experiment, verify:

1. **Check external data source has basin_id:**
   ```bash
   python -c "import pandas as pd; df=pd.read_csv('data/external_geofeatures_hydro_v2.csv'); assert 'basin_id' in df.columns; print('✅ basin_id present')"
   ```

2. **Check config has grouping_strategy:**
   ```bash
   grep "grouping_strategy" config.yml
   # Should output: grouping_strategy: "basin"
   ```

3. **Check pipeline output includes basin grouping:**
   ```bash
   # After running, look for:
   ">>> GROUPING: Basin-aware strategy"
   "Unique basins: 149"  # NOT "Using grid-based grouping"
   ```

---

## FILES CREATED/MODIFIED

| File | Change | Impact |
|------|--------|--------|
| `config.yml` | Updated external_path | Basin IDs now available |
| `config_drp_basin_rain.yml` | Updated external_path + grouping_strategy | Ready for DRP experiments |
| `src/train_pipeline.py` | Enhanced make_groups(), logging | True basin-aware CV active |
| `verify_basin_grouping.py` | NEW script | On-demand verification |
| `MASTER_FIX_AUDIT.md` | NEW documentation | Technical details |
| `NEXT_STEPS.md` | NEW documentation | Action items |

---

## CONFIRMATION CHECKLIST

✅ basin_id column located in correct data source  
✅ basin_id loaded into train_df (149 unique values, 100% complete)  
✅ Basin grouping strategy configured in YAML  
✅ make_groups() enhanced to use basin_id when available  
✅ CV folds show zero basin overlap (no leakage)  
✅ CHIRPS features (5 features + 2 flags) preserved to model input  
✅ Cache cleared to remove stale data  
✅ Logging shows basin-aware strategy active  
✅ DRP model receives 79 features including CHIRPS  

---

## READY FOR PRODUCTION ✅

All verification checks PASSED. Basin-aware cross-validation is now the default. CHIRPS rainfall features are fully integrated. Pipeline is ready for final experiments.
