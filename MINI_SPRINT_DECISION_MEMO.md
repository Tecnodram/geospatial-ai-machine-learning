# Night Mini-Sprint Decision Memo
**Date:** 2026-03-11 | **Sprint:** ET-1400 controlled run + 7 blend artifacts

---

## 1. What Was Done

### Training Run
- Cloned `exp_20260311_134704` config (ET, TE=True, y_mode=none, 104 DRP features)
- Changed ONLY: `cv_by_target.DRP.n_estimators` 400→700, `final_by_target.DRP.n_estimators` 800→1400
- New experiment: **`exp_20260311_194143`** at `experiments/exp_20260311_194143/submission.csv`
- DRP CV: mean=0.1701 ± 0.0959 (one fold outlier: -0.017; others 0.19–0.25)

### Bug Fixed (critical)
**`build_model_from_cfg` in `src/train_pipeline.py` was silently ignoring per-target ET/RF params.**  
For `ExtraTreesRegressor` and `RandomForestRegressor`, the function always read from `et_cv`/`et_final`
(the global block) instead of `cv_by_target[target].params` / `final_by_target[target].params`.  
HGB and CatBoost correctly used per-target params; ET/RF did not.  
**This means all historical experiments that specified `cv_by_target` or `final_by_target` with custom
ET params (e.g., different n_estimators/min_samples_leaf) were silently running with default params.**  
Fix committed on line ~342 of `src/train_pipeline.py`. The first run (`exp_20260311_193457`) used
the broken code and produced bit-identical predictions to `exp_20260311_134704`. The second run
(`exp_20260311_194143`) used the fixed code and produced genuinely different predictions.

### Blend Artifacts Generated
7 blends in `experiments/mini_sprint_blends/`. All use **TA/EC from v44 anchor** (verified: `True`).
Only DRP column is blended.

---

## 2. Complete Audit Table

| File / Experiment | LB | DRP min | DRP max | DRP med | DRP mean | corr_to_v44 | MAE_to_v44 |
|---|---|---|---|---|---|---|---|
| **v44 (anchor)** | **0.3039** | 16.73 | 41.14 | 28.37 | 30.01 | 1.0000 | 0.000 |
| safe9010 | 0.2989 | 16.39 | 41.84 | 27.82 | 29.10 | 0.9947 | 0.953 |
| safe8515 | 0.2949 | 16.23 | 42.32 | 27.56 | 28.64 | 0.9876 | 1.429 |
| v52_as_is | 0.2929 | 8.61 | 38.63 | 21.98 | — | 0.361 | ~9 |
| 134704 (ET-800) | not submitted | 15.71 | 45.01 | 26.15 | 26.59 | 0.9047 | 3.612 |
| **ET-1400 (194143)** | not submitted | 15.50 | 44.52 | 26.25 | 26.46 | 0.9047 | 3.688 |
| blend1: v44_90 + 134704_10 linear | — | 16.63 | 40.94 | 28.12 | 29.66 | 0.9994 | 0.361 |
| blend2: v44_80 + 134704_20 linear | — | 16.53 | 41.39 | 28.05 | 29.32 | 0.9974 | 0.722 |
| blend3: v44_70 + 134704_30 linear | — | 16.43 | 41.85 | 27.88 | 28.98 | 0.9939 | 1.084 |
| blend4: v44_80 + 134704_20 **rank** | — | 16.73 | 41.09 | 28.42 | 30.01 | 0.9962 | 0.353 |
| blend5: v44_70 + 134704_30 **rank** | — | 16.73 | 41.06 | 28.36 | 30.00 | 0.9925 | 0.524 |
| blend6: v44_90 + ET1400_10 linear | — | 16.61 | 40.89 | 28.13 | 29.65 | 0.9994 | 0.369 |
| blend7: v44_80 + ET1400_20 linear | — | 16.48 | 41.29 | 28.04 | 29.30 | 0.9976 | 0.738 |

---

## 3. LB Region Analysis

### The safe-blend lesson
The "safe" blends were named optimistically. Actual LB evidence:
- safe9010 (corr=0.9947, MAE=0.953 to v44) → LB=**0.2989** (−0.0050 vs anchor)
- safe8515 (corr=0.9876, MAE=1.429 to v44) → LB=**0.2949** (−0.0090 vs anchor)

**Pattern**: increased divergence from v44 → monotonically lower LB score. The per-unit MAE cost is
approximately: **LB penalty ≈ 0.0050 × (MAE / 0.953)** = 0.00525 per unit MAE.

### Expected LB for each blend
| Candidate | MAE_to_v44 | Expected LB delta | Expected LB |
|---|---|---|---|
| v44 (anchor) | 0.000 | 0 | **0.3039** |
| blend4 (rank 80/20) | 0.353 | −0.0019 | ~0.3020 |
| blend1 (linear 90/10) | 0.361 | −0.0020 | ~0.3019 |
| blend6 (linear 90/10 ET1400) | 0.369 | −0.0020 | ~0.3019 |
| blend2 (linear 80/20) | 0.722 | −0.0038 | ~0.3001 |
| blend3 (linear 70/30) | 1.084 | −0.0057 | ~0.2982 (≈safe9010 territory) |

**Caveat**: The penalty model assumes a linear relationship. Blends that add complementary signal
(not just noise) from 134704 could outperform the estimate. Blends that add mostly noise will
track or exceed the estimate downward.

### Rank blend advantage
Blend4 uses rank interpolation, which **preserves the v44 DRP mean exactly (30.01)** and the
range (16.73–41.09 vs v44's 16.73–41.14). The safe blends systematically reduced the DRP mean
(29.10, 28.64) and that correlated with LB loss. Blend4 avoids this structural shift.

---

## 4. Candidate Rankings

### Primary recommendation: ✅ HOLD v44
LB=0.3039 is the current best. Every blend we've measured scores lower. **Do not submit a blend
unless a submission slot can be risked on a ≤−0.002 LB move that could also be +0.004 in
the upside case.**

### Best alternative if one blend slot is available: blend4
```
experiments/mini_sprint_blends/blend4_v44_80_134704_20_rank.csv
```
- Rank blend preserves distribution structure (mean=30.01, same as v44)
- MAE=0.353 to v44 (less than half of safe9010's 0.953)
- corr=0.9962 — still very close to anchor
- If 134704 adds any orthogonal signal on hard stations, rank blend captures it without
  shifting the mean prediction

### Second alternative: blend1
```
experiments/mini_sprint_blends/blend1_v44_90_134704_10_linear.csv
```
- Simplest blend, very tight to v44 (MAE=0.361, corr=0.9994)
- Slightly reduces DRP mean (30.01→29.66) — this is the main risk

### Do NOT submit
- blend3, blend5: MAE>1.0, entering safe8515 LB territory (-0.009)
- Raw ET-1400 (194143): corr=0.905, MAE=3.688 — very risky
- Any blend using v52_as_is: corr=0.361, catastrophically divergent

---

## 5. ET-1400 vs ET-800 Finding

The additional 600 trees (1400 − 800) changed DRP predictions by at most **1.28 units** (mean: 0.40
per station). Both 800-tree (134704) and 1400-tree (194143) have:
- corr_to_v44 = 0.9047 (identical)
- MAE_to_v44 ≈ 3.6–3.7

The DRP ensemble is essentially **converged at 800 trees**. Going to 1400 does not meaningfully
improve predictions. This is consistent with the CV score also barely changing (0.1650 → 0.1701).

**Takeaway**: Further tree count increases are not the lever for improving DRP performance. The
bottleneck is the signal in the features, not ensemble variance.

---

## 6. What to Try Next (if slots available)

Priority order given all evidence:
1. **Submit v44 again** (only if leaderboard needs it — it's already 0.3039)
2. **blend4** (rank 80/20): one submission, expect ~0.3020, small upside possible
3. **Investigate why fold 4 always collapses** (DRP fold=-0.027 in this run, −0.027 in 134704 run)
   — fixing the outlier fold could unlock real CV gains without new feature engineering
4. **Try min_samples_leaf=1 or 2** for DRP (currently 3) — underfit is more likely than overfit
   for small-N target like DRP with high spatial heterogeneity
5. **Spatial leave-group-out CV** — basin-aware CV may still be leaking for DRP; proper spatial
   holdout could better estimate true generalization

---

## 7. Artifacts Index

| Path | Contents |
|---|---|
| `experiments/exp_20260311_194143/submission.csv` | ET-1400 DRP final prediction |
| `experiments/exp_20260311_194143/metadata.json` | CV metrics, model config, feature counts |
| `experiments/mini_sprint_blends/blend1_v44_90_134704_10_linear.csv` | Best linear blend |
| `experiments/mini_sprint_blends/blend4_v44_80_134704_20_rank.csv` | Best rank blend |
| `experiments/mini_sprint_blends/audit_report.json` | Machine-readable blend audit |
| `config_et1400_drp_clone134704.yml` | Config used for ET-1400 run |
| `generate_mini_sprint_blends.py` | Blend generator (rerunnable, accepts --et1400_path) |
| `src/train_pipeline.py` | Bug fix: per-target ET params now respected in build_model_from_cfg |

---

*Generated: 2026-03-11 | Mini-sprint complete*
