# Sprint 4 Temporal + Basin Subset Decision Report

Run ID: exp_20260311_215006
Date: 2026-03-11
Config: config_exp_sig_s4_temporal_basin.yml

A. Exact new features used
Temporal (full)
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

Basin subset (only)
- basin_mean_soil_clay_0_5
- basin_mean_soil_ph_0_5
- basin_mean_elevation
- basin_mean_slope

B. Build location (Snowflake vs local)
- Features were built in local Python during dataset build in the training pipeline.
- Snowflake was used for audit registry refresh only.

C. CV results by target (5-fold)
- Total Alkalinity: mean=0.344776, std=0.131033
- Electrical Conductance: mean=0.359289, std=0.111810
- Dissolved Reactive Phosphorus: mean=0.181467, std=0.085374
- Overall CV mean: 0.295177

D. DRP improvement assessment
Reference values:
- Sprint 1 DRP CV (exp_20260311_212240): 0.164920
- Sprint 2 DRP CV (exp_20260311_213420): 0.175838
- Sprint 3 DRP CV (exp_20260311_214447): 0.151357
- Reference DRP CV (exp_20260311_134704): 0.164995
- Sprint 4 DRP CV (exp_20260311_215006): 0.181467

Deltas for Sprint 4:
- vs Sprint 1: +0.016547
- vs Sprint 2: +0.005629
- vs Sprint 3: +0.030110
- vs reference exp_20260311_134704: +0.016472

Conclusion: Sprint 4 combination improves DRP CV versus all comparison runs, including Sprint 2 temporal-only.

E. Keep/drop recommendation
- Keep this family combination as the current strongest signal-upgrade candidate.
- Preserve the same backbone and feature subset constraints when validating on leaderboard-safe blending steps.

F. Recommended next move
- Stop feature-family expansion for now.
- Move to one conservative leaderboard-safe submission strategy centered on Sprint 4 output, with anchor-preserving DRP blend safety checks before any submission decision.
