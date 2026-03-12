# Sprint 3 Upstream-Only Decision Report

Run ID: exp_20260311_214447
Date: 2026-03-11
Config: config_exp_sig_s3_upstream.yml

A. Exact new upstream features used
- upstream_stream_power
- upstream_discharge_contact
- upstream_connectivity
- upstream_river_decay
- upstream_soil_clay_pressure
- upstream_soc_pressure
- upstream_chirps_ppt_pressure
- upstream_chirps_ppt_transport
- upstream_ppt_obs30_mean_pressure
- upstream_ppt_obs30_mean_transport
- upstream_ppt_obs60_mean_pressure
- upstream_ppt_obs60_mean_transport
- upstream_ppt_obs90_mean_pressure
- upstream_ppt_obs90_mean_transport

B. Build location (Snowflake vs local)
- Built in local Python during dataset build in the training pipeline.
- No DWS or external chemical measurement datasets were used.

C. CV results by target (5-fold)
- Total Alkalinity: mean=0.353798, std=0.138350
- Electrical Conductance: mean=0.346326, std=0.100401
- Dissolved Reactive Phosphorus: mean=0.151357, std=0.108372
- Overall CV mean: 0.283827

D. DRP improvement assessment versus Sprint 1, Sprint 2, and reference
- Sprint 1 basin-only DRP CV: 0.164920 (exp_20260311_212240)
- Sprint 2 temporal-only DRP CV: 0.175838 (exp_20260311_213420)
- Reference DRP CV: 0.164995 (exp_20260311_134704)
- Sprint 3 upstream-only DRP CV: 0.151357 (exp_20260311_214447)

Deltas:
- vs Sprint 1: -0.013563
- vs Sprint 2: -0.024481
- vs reference: -0.013638

Conclusion: upstream-only did not improve DRP and is materially weaker than Sprint 1, Sprint 2, and the reference run.

E. Keep/drop recommendation for Sprint 4 combination
- Keep only as a low-priority conditional option.
- Do not use upstream-only as a standalone candidate.

F. Recommended next move
- Do not launch broad searches.
- For Sprint 4, test exactly one conservative combination run that starts from Sprint 2 temporal-only and adds only a small, high-confidence subset of upstream features.
- Keep model backbone unchanged and compare directly against Sprint 2 and the reference.
