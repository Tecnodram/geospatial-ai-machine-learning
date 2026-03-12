# MODEL PERFORMANCE TRACKER

| Exp ID | Feature Group | TA Model | EC Model | DRP Model | TA R² | EC R² | DRP R² | Mean R² | Notes | LB Score |
|--------|---------------|----------|----------|-----------|-------|-------|--------|---------|-------|----------|
| exp_sprint1_01_baseline_multi | A (landsat+tc+chirps+ext) | ET | ET | ET | 0.341 | 0.358 | 0.185 | 0.295 | Baseline multi-target | - |
| exp_sprint1_02_calendar_rain | A+ (added 7,14 obs rolling) | ET | ET | ET | 0.342 | 0.359 | 0.193 | 0.298 | Calendar rain features | - |
| exp_sprint1_03_ec_hgb | A+ | ET | HGB | ET | 0.342 | 0.230 | 0.193 | 0.255 | EC HGB model | - |
| exp_sprint2_04_drp_clip_p95 | A+ | ET | ET | ET | 0.342 | 0.359 | 0.187 | 0.296 | DRP clip_p95 transform | - |
| exp_sprint2_05_drp_winsor | A+ | ET | ET | ET | 0.342 | 0.359 | 0.193 | 0.298 | DRP winsor transform | - |
| exp_sprint2_06_drp_sqrt | A+ | ET | ET | ET | 0.342 | 0.359 | 0.144 | 0.282 | DRP sqrt transform | - |
| exp_sprint3_07_ec_rf | A+ | ET | RF | ET | 0.342 | 0.355 | 0.193 | 0.296 | EC RandomForest model | - |
| exp_sprint3_08_hydro_ablation | A- (no hydro) | ET | ET | ET | 0.315 | 0.357 | 0.188 | 0.287 | Hydro features ablation | - |
| exp_sprint3_09_rain_ablation | A- (no chirps) | ET | ET | ET | 0.337 | 0.356 | 0.203 | 0.299 | CHIRPS features ablation | - |
| exp_sprint4_10_spatial_encoding | A+ spatial | ET | ET | ET | 0.337 | 0.361 | 0.191 | 0.296 | Spatial encoding features | - |
| exp_sprint4_11_hydro_interactions | A+ hydro int | ET | ET | ET | 0.337 | 0.356 | 0.203 | 0.299 | Hydro interaction features | - |
| exp_sprint4_12_ensemble_candidate | A | Ensemble | Ensemble | Ensemble | 0.339 | 0.361 | 0.164 | 0.288 | ET+RF+HGB average ensemble | - |
| exp_Watershed_V1 | Watershed Intelligence | Voting(ET70+Ridge30) | Voting(ET70+Ridge30) | RF(maxd8,minl15) | 0.283 | 0.272 | 0.157 | 0.237 | Upstream pressure + 15-45d rain lag + KNN encoding | - |
| exp_Watershed_Optimized_v2 | Watershed Calibrated | Voting(ET85+Ridge15,scaled) | Voting(ET85+Ridge15,scaled) | RF(maxd12,mins20) | 0.302 | 0.305 | 0.154 | 0.254 | Non-linear hydrology + basin-locked KNN + tuned ensemble | - |
|--------|---------------|----------|----------|-----------|-------|-------|--------|---------|-------|----------|


| exp_Watershed_Final_Sprint_V3 | Watershed Calibration V3 | Voting(ET85+Ridge15,scaled) | Voting(ET85+Ridge15,scaled) | HGB(poisson,max_iter300,l2=1.0) | 0.2969 | 0.3143 | -0.2102 | 0.1337 | longer rain lag + KNN-8 + hydro_chem/clay features | - |
| exp_DRP_Recovery_V4 | DRP Recovery | Voting(ET85+Ridge15,scaled) | Voting(ET85+Ridge15,scaled) | ET(reg10,l10) | 0.2996 | 0.3152 | 0.1387 | 0.2512 | dual rain windows + priority features | - |
| exp_24h_Certificate_Attack | CatBoost Evolution | CatBoost(2000,0.03,d6,l2=5) | CatBoost(2000,0.03,d6,l2=5) | ET(maxd10) | **0.3287** | **0.3456** | 0.1821 | **0.2855** | Mineral physics features + log1p EC + KNN-8 | TARGETING 0.40 |
