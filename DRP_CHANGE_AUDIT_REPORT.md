# DRP Change Audit Report

- Generated at: 2026-03-11T23:51:41
- Output directory: C:\Projects\ey-water-quality-2026\experiments\drp_change_audit_20260311_235139
- Compared files:
  - C:\Projects\ey-water-quality-2026\submissions\submission_V4_4_DRP_tuned_ET_fixorder.csv
  - C:\Projects\ey-water-quality-2026\experiments\exp_20260311_215006\submission.csv
  - C:\Projects\ey-water-quality-2026\submissions_batch\blend_v44_90_s4_10_rank.csv

## Phase 1: Load and Align

- All three files loaded with shape 200x6.
- Key-order aligned on ['Latitude', 'Longitude', 'Sample Date']: PASS.
- Comparison columns built: drp_v44, drp_s4, drp_blend9010, delta_s4_vs_v44, delta_blend_vs_v44, delta_blend_vs_s4.

## Phase 2: DRP Change Analysis

- DRP correlation to V4_4: Sprint4=0.8809, Blend9010=0.9985
- DRP MAE to V4_4: Sprint4=2.9372, Blend9010=0.3374
- Largest |delta_s4_vs_v44| row: lat=-33.329167, lon=26.077500, date=2014-05-28, delta=-9.5049
- Largest |delta_blend_vs_v44| row: lat=-33.329167, lon=26.077500, date=2011-05-06, delta=-1.9961

## Phase 3: Geospatial and Temporal Patterns

- Highest-change latitude band (by mean |delta_s4_vs_v44|): (-32.992, -32.086]
- Highest-change longitude band (by mean |delta_s4_vs_v44|): (25.43, 27.367]
- Month with highest mean |delta_s4_vs_v44|: 12
- basin_id coverage: 100.0%
- PET coverage: 100.0%
- CHIRPS coverage: 100.0%

## Phase 4: Decision Interpretation

1. Sprint 4 changes DRP most in specific geo-temporal pockets rather than uniformly across all 200 rows.
2. The 90/10 blend dampens magnitude strongly while preserving directional shifts for many high-delta rows.
3. Change concentration by month and lat/lon quantile suggests temporal-memory and basin-context interactions rather than random noise.
4. Pattern is physically plausible when high deltas co-occur with rainfall/PET stress proxies and basin attributes.
5. Hypothesis: residual gains may come from selective weighting by hydro-climate regime (wet vs dry stress windows) instead of global DRP shift.

## Artifacts

- DRP_CHANGE_SUMMARY.csv
- drp_change_detail.csv
- top25_abs_delta_s4_vs_v44.csv
- top25_abs_delta_blend_vs_v44.csv
- lat_band_change_patterns.csv
- lon_band_change_patterns.csv
- month_change_patterns.csv
- doy_bin_change_patterns.csv
- basin_change_patterns_top25.csv (if basin_id join succeeded)
- proxy_delta_correlations.csv (if enough non-null context)
- hist_delta_s4_vs_v44.png
- hist_delta_blend_vs_v44.png
- scatter_drp_v44_vs_s4.png
- scatter_drp_v44_vs_blend9010.png

## Safe Progress Update (2026-03-12)

- New immutable candidate: C:\Projects\ey-water-quality-2026\submissions_batch\blend_v44_regime_m12_s4_15_else05_rank.csv
- Rule: DRP rank blend only, with Sprint 4 weight=0.15 when month==12 and 0.05 otherwise.
- TA/EC policy: copied exactly from V4_4 anchor (`submission_V4_4_DRP_tuned_ET_fixorder.csv`).
- Month coverage: month 12 rows=13; other months rows=187; effective S4 weight=0.0565.

### Safety Metrics (candidate)

- DRP mean=30.005676
- DRP median=28.410180
- DRP min=16.730570
- DRP max=41.124877
- Correlation to V4_4=0.999439
- MAE to V4_4=0.111462
- Correlation to blend_v44_90_s4_10_rank=0.999436
- MAE to blend_v44_90_s4_10_rank=0.269000

### Risk Read

- Relative to 90/10 blend, this candidate is **safer** by anchor-distance criteria:
  - higher corr-to-V4_4 (0.999439 vs 0.998488)
  - lower MAE-to-V4_4 (0.111462 vs 0.337393)