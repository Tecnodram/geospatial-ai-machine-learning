# Regime-Expert DRP Feasibility Memo

Date: 2026-03-12

## Decision
Regime-expert DRP modeling is feasible in a conservative form, but only for coarse regime splits.

## Evidence Snapshot
- Train support is adequate for broad splits:
  - PET-high: 1908 rows, all 5 folds populated, 59 basins
  - month12: 537 rows, all 5 folds populated, 124 basins
- Micro-regime intersections are sparse:
  - month12 & PET-high: 110 rows, min fold count 11
  - all three (month12 & PET-high & dist-high): 80 rows, min fold count 7
- Validation has very small intersection support (2-3 rows), increasing overfit risk for specialized micro-experts.

## Feasibility Verdict
- Yes, feasible for one binary gate with fallback.
- No, not safe yet for multi-gate or triple-intersection experts.

## Safest First Split
PET-high gate (PET > 198.8).

Why this first:
- Better sample support than month12-only micro-splits.
- Better hydrology/seasonality relevance for DRP transport behavior.
- Enough positive-branch examples per fold to avoid immediate fold collapse.

## Overfitting Risks
- Branch overfitting in sparse regimes (especially all-three intersection).
- Instability from branch-level parameter tuning with tiny fold support.
- High variance in CV if branch routing is too granular.
- Leaderboard drift if expert predictions are not tethered to anchor behavior.

## Risk Controls (v1)
- Single gate only (PET-high vs PET-low).
- Hard fallback to global model when support thresholds fail.
- Conservative blending with anchor (alpha=0.10).
- Require fold-level wins (>=3/5 folds) before any submission candidate is promoted.

## Recommendation
Proceed with prepared v1 gate plan only; do not expand to multi-regime expert trees until single-gate evidence is positive and stable.
