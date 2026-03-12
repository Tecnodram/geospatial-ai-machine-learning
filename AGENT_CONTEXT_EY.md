# EY Water Quality 2026 — Agent Context

## Objective
Improve model performance for the EY AI and Data Challenge 2026 water quality competition.

Targets:
- Total Alkalinity
- Electrical Conductance
- Dissolved Reactive Phosphorus

Primary optimization goal:
Increase mean CV and likely leaderboard performance while preserving spatial validity and avoiding leakage.

## Current status
Current pipeline already runs end-to-end with:
- src/train_pipeline.py
- src/run_all.py
- src/batch_blends.py
- src/auto_experiments.py

External features currently used:
- data/external_geofeatures_plus_hydro_v2.csv

Current reality:
- TA is relatively strong and stable
- EC is reasonably stable
- DRP is the main bottleneck
- Avoid degrading TA and EC while improving DRP

## Validation constraints
- Geographic GroupKFold must remain intact
- No leakage allowed
- All feature engineering must be reproducible
- Submission keys/order must remain identical to submission_template.csv

## Important project facts
- Training data has repeated monitoring stations over time
- Many observations share the same coordinates
- Site-aware and temporal-aware features may help, especially for DRP
- Hydrology and climate-event features are scientifically plausible for phosphorus transport

## Allowed files to modify first
- src/train_pipeline.py
- config.yml
- src/auto_experiments.py

## Files to modify only when justified
- src/05_external_geofeatures.py
- src/build_hydro_features.py
- src/batch_blends.py
- src/run_all.py

## Agent behavior rules
1. Make small, testable changes only
2. Explain hypothesis before editing
3. State leakage risk explicitly
4. Preserve spatial CV
5. Do not rewrite the entire pipeline
6. Prefer DRP-focused experiments
7. Keep TA and EC stable
8. After every edit, provide a changelog-style summary
9. Do not change submission key alignment logic unless necessary
10. Think like a competition participant, not a generic coder

## First priority experiments
1. DRP station-aware features without leakage
2. DRP hydrology V3 features
3. DRP climate-event aggregation features
4. Conservative target-aware feature pruning