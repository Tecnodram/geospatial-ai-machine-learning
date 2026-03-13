# PROJECT STATE MAP

Date: 2026-03-12
Objective: Capture current DRP leaderboard state and define the next unexplored modeling line.

## 1) Best Leaderboard Submissions (confirmed)

| Rank (by LB) | File | LB Score | Status | Notes |
|---|---|---:|---|---|
| 1 (tie) | blend4_v44_80_134704_20_rank.csv | 0.3029 | Submitted | Mini-sprint blend result |
| 1 (tie) | blend_v44_85_134704_15_rank.csv | 0.3029 | Submitted | Mini-sprint blend result |
| 3 | submission_V4_4_DRP_tuned_ET_fixorder.csv | 0.3039 | Submitted | Stable anchor baseline |

Source: leaderboard_log.csv entries with numeric score < 1.0.

## 2) What Worked

- V4_4 anchor remains leaderboard-safe with strong stability profile.
- Conservative rank blends close to V4_4 can improve LB relative to anchor (0.3039 -> 0.3029).
- Sprint-4 signal is useful only when tightly tethered to V4_4 via low-weight rank blending.
- Regime-based DRP blending files were generated and validated with TA/EC locked to anchor:
  - blend_v44_regime_m12_s4_20_else05_rank.csv
  - blend_v44_regime_m12_pet_s4_20_10_05_rank.csv
  - blend_v44_regime_m12_distmain_s4_20_10_05_rank.csv

## 3) What Failed

- Large divergence from anchor tends to reduce LB (safe9010/safe8515 historical evidence in mini-sprint memo).
- Pure Sprint-4 DRP submission is high risk (distribution drift, lower correlation to anchor).
- Residual DRP sprint (Ridge correction on V4_4) failed to extract robust signal:
  - residual_model_cv mean R2 = -0.0768
  - corrected_model_cv mean R2 = 0.1122
  - recommendation from run artifact: do not submit residual correction

## 4) Pending Submissions Queue

Unsubmitted files currently in submissions_batch (based on leaderboard_log name matching):

Priority (most actionable):
- blend_v44_regime_m12_pet_s4_20_10_05_rank.csv
- blend_v44_regime_m12_distmain_s4_20_10_05_rank.csv
- blend_v44_regime_m12_s4_20_else05_rank.csv
- blend_v44_regime_m12_s4_15_else05_rank.csv

Additional pending artifacts:
- blend_v44_80_s4_20_rank.csv
- blend_v44_85_s4_15_rank.csv
- blend_v44_90_s4_10_rank.csv
- blend_v44_90_s4_10_rank__LOCKED.csv
- submission_V5_2_OOFTE_fixkeys.csv
- submission_robustness_ensemble.csv
- submission_A_sqrt_ET_drp_focused_OFF.csv
- submission_B_sqrt_ET_drp_focused_ON.csv
- submission_C_sqrt_RF_drp_focused_OFF.csv
- submission_D_sqrt_HGB_drp_focused_OFF.csv
- submission_E_sqrt_ET_drp_focused_OFF_conservative_TE.csv
- BLEND_SUMMARY_RANK_DRP.csv (not a leaderboard candidate)

## 5) Current Regime Thresholds (locked)

- PET high: PET > 198.8
- Distance-to-main high: dist_main_km > 498.1
- Calendar regime: month == 12

These thresholds are retained for comparability with existing regime audits.

## 6) Regime Sample Counts (Train/Validation)

### 6.1 Marginal counts

| Regime | Train (n=9319) | Validation (n=200) |
|---|---:|---:|
| month == 12 | 537 (5.76%) | 13 (6.50%) |
| PET > 198.8 | 1908 (20.47%) | 13 (6.50%) |
| dist_main_km > 498.1 | 5653 (60.66%) | 15 (7.50%) |

### 6.2 Intersections

| Regime Intersection | Train Count | Validation Count |
|---|---:|---:|
| month12 AND PET high | 110 (1.18%) | 3 (1.50%) |
| month12 AND dist_high | 346 (3.71%) | 2 (1.00%) |
| PET high AND dist_high | 1302 (13.97%) | 2 (1.00%) |
| month12 AND PET high AND dist_high | 80 (0.86%) | 2 (1.00%) |

### 6.3 Fold support diagnostics (train, fold_basin)

| Regime | Total | Per-fold counts | Zero folds | Min fold count | Unique basins |
|---|---:|---|---:|---:|---:|
| month12 | 537 | [104,110,117,114,92] | 0 | 92 | 124 |
| PET high | 1908 | [429,196,358,370,555] | 0 | 196 | 59 |
| dist_high | 5653 | [1515,952,1134,1152,900] | 0 | 900 | 75 |
| month12 & PET high | 110 | [29,11,20,27,23] | 0 | 11 | 42 |
| month12 & dist_high | 346 | [84,55,73,76,58] | 0 | 55 | 65 |
| PET high & dist_high | 1302 | [345,117,330,194,316] | 0 | 117 | 42 |
| all three | 80 | [19,7,19,20,15] | 0 | 7 | 29 |

## 7) Feasibility Assessment: Regime-Specific DRP Experts

Verdict: Feasible only for coarse splits; unsafe for micro-regimes.

- Feasible expert-only candidates:
  - PET high vs PET low (1908 vs 7411 train rows)
  - month12 vs non-month12 (537 vs 8782 train rows), but month12 model must be strongly regularized
- Borderline:
  - month12 & PET high (110 rows, min fold 11) -> high variance risk
- Not recommended for separate experts:
  - all-three intersection (80 rows, min fold 7) -> severe fold-collapse risk if modeled standalone

Operational conclusion:
- Use gating with fallback to global DRP model.
- Gate only once (single split), avoid stacked micro-experts in v1.

## 8) Next Unexplored Line (Prepared, not run)

- Config prepared: config_exp_regime_expert_drp_v1.yml
- Plan prepared: reports/regime_gated_training_plan_v1.md
- Intention: conservative two-expert DRP with out-of-fold gate diagnostics and hard fallback to global model where support is weak.
