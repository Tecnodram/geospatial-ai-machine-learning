# Regime-Gated Training Plan V1 (Prepared, Not Run)

## Goal
Add a conservative DRP expert-model layer while preserving leaderboard safety.

## Guardrails
- Keep TA and EC exactly from V4_4 anchor.
- Keep basin-aware fold integrity (fold_basin / GroupKFold discipline).
- Use only one gate in v1.
- Require support thresholds before activating an expert model.

## V1 Split Choice
Primary split: PET high gate (pet > 198.8)

Reason:
- Best support among meaningful regimes for a binary split.
- Train support: 1908 PET-high samples (59 basins), 7411 PET-low.
- All folds have PET-high observations, minimizing fold-collapse risk.

## Training Procedure
1. Build global DRP model on full training set.
2. Build gate labels from threshold (pet > 198.8).
3. Train Expert-High on PET-high subset if support thresholds pass.
4. Train Expert-Low on PET-low subset if support thresholds pass.
5. During OOF CV, route each row through gate to matching expert.
6. If gate subset in any fold violates support constraints, fallback to global model for that branch.
7. Produce OOF metrics for:
   - Global-only DRP
   - Gated-expert DRP
   - Gated-expert blended with V4_4 (alpha=0.10)

## Inference Procedure (200-row validation)
1. Build regime flags from validation PET and dist_main.
2. Route by PET gate:
   - PET-high -> Expert-High
   - PET-low -> Expert-Low
3. If expert unavailable (support failure), use global model output.
4. Final DRP = V4_4_DRP + alpha * (gated_pred - global_pred)
5. Clip DRP at zero.

## Safety Checks
- Submission shape is exactly 200 x 6.
- No NaN values.
- No negative DRP values.
- TA/EC columns byte-identical to V4_4.
- Report corr/MAE to V4_4 before any submission decision.

## Explicitly Deferred (Not in V1)
- Triple-intersection experts (month12 & pet_high & dist_high): too sparse.
- Multi-gate trees (month then PET then distance): high overfit risk.
- Per-regime custom feature spaces: defer until gate proves value.

## Exit Criteria
Proceed to leaderboard only if:
- Gated-expert DRP CV improves over global DRP in at least 3/5 folds, and
- Mean DRP CV gain is positive with non-catastrophic worst fold, and
- Corr to V4_4 remains >= 0.995 after alpha=0.10 blend.
