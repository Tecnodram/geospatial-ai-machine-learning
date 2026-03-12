# Sprint 2 Temporal-Only Decision Report

Run ID: exp_20260311_213420
Date: 2026-03-11
Config: config_exp_sig_s2_temporal.yml

A. Exact new temporal features used
- sin_doy
- cos_doy
- sin_month
- cos_month
- rain_lag_1
- rain_lag_3
- rain_lag_7
- rain_lag_14
- rain_roll_3
- rain_roll_7
- rain_roll_14
- rain_roll_30
- dry_spell_obs
- rain_anom_30
- pet_roll_30
- rain_to_pet_30

B. Build location (Snowflake vs local)
- Built in local Python during dataset build in train pipeline.
- No DWS or external chemical measurement datasets were used.

C. CV results by target (5-fold)
- Total Alkalinity: mean=0.354243, std=0.140341
- Electrical Conductance: mean=0.365717, std=0.094857
- Dissolved Reactive Phosphorus: mean=0.175838, std=0.108779
- Overall CV mean: 0.298599

D. DRP improvement assessment
- Sprint 1 basin-only DRP CV: 0.164920 (exp_20260311_212240)
- Sprint 2 temporal-only DRP CV: 0.175838 (exp_20260311_213420)
- Delta vs Sprint 1: +0.010918
- Reference DRP CV (exp_20260311_134704): 0.164995
- Delta vs reference: +0.010843

Conclusion: No meaningful DRP improvement. Temporal-only run underperformed both Sprint 1 and the reference on DRP CV.

E. Keep/drop recommendation for Sprint 4 combination
- Keep conditionally for combination testing only.
- Do not promote temporal-only as a standalone family.

F. Recommended next move
- Proceed to Sprint 3 upstream-only with the same single-run discipline.
- If Sprint 3 shows standalone DRP gain, test only one targeted Sprint 4 combination pairing that winner with the best conditional family.
