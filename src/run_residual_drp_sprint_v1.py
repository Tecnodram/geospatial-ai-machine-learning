#!/usr/bin/env python
# coding: utf-8
"""
Controlled Residual Modeling Sprint V1 for DRP.

Architecture:
  1. Load training data: spatial_audit_folds.csv + TerraClimate PET + CHIRPS PPT
  2. Run V4_4-equivalent base model (ET n=700) in 5-fold GroupKFold(basin) -> OOF DRP predictions
  3. Compute DRP residuals = y_true - y_oof
  4. Build regime+temporal features for train and validation sets
  5. Train RidgeCV residual model on those features vs DRP residuals
  6. Report residual model CV R2 using same folds
  7. Apply correction: pred_drp_final = v44_drp + alpha * resid_pred_valid
  8. Build submission: TA/EC from V4_4 anchor, DRP corrected
  9. Save immutable artifacts in timestamped experiment folder
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"
KEYS = [LAT_COL, LON_COL, DATE_COL]
DRP_COL = "Dissolved Reactive Phosphorus"
TA_COL  = "Total Alkalinity"
EC_COL  = "Electrical Conductance"
SUBMISSION_COLS = [LAT_COL, LON_COL, DATE_COL, TA_COL, EC_COL, DRP_COL]

ROOT = Path(__file__).resolve().parents[1]


# ── Utilities ──────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_keys(df: pd.DataFrame, dayfirst: bool = True) -> pd.DataFrame:
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    if DATE_COL in out.columns:
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=dayfirst)
    return out


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_training_data(cfg: dict) -> pd.DataFrame:
    """
    Load training dataset:
    - Base: spatial_audit_folds.csv  (9319 rows with targets, hydro, fold_basin)
    - Join: TerraClimate PET + CHIRPS PPT on KEYS
    """
    data_cfg = cfg["data"]
    base_path = ROOT / data_cfg["spatial_folds_path"]
    tc_path   = ROOT / data_cfg["tc_train_path"]
    ch_path   = ROOT / data_cfg["chirps_train_path"]

    # Base dataset (has DRP target, hydro features, fold_basin)
    base = normalize_keys(pd.read_csv(base_path))

    # TC uses DD-MM-YYYY (dayfirst=True); CHIRPS uses ISO YYYY-MM-DD (dayfirst=False)
    tc = normalize_keys(pd.read_csv(tc_path), dayfirst=True)    # cols: Lat, Lon, Date, pet
    ch = normalize_keys(pd.read_csv(ch_path), dayfirst=False)   # cols: Lat, Lon, Date, chirps_ppt

    # Join climate on exact key match (all rows should match wq)
    df = base.merge(tc[KEYS + ["pet"]], on=KEYS, how="left")
    df = df.merge(ch[KEYS + ["chirps_ppt"]], on=KEYS, how="left")

    print(f"[DATA] Training: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"[DATA] PET null: {df['pet'].isna().sum()}")
    print(f"[DATA] CHIRPS null: {df['chirps_ppt'].isna().sum()}")
    print(f"[DATA] DRP null: {df[DRP_COL].isna().sum()}")
    return df


def load_validation_data(cfg: dict) -> pd.DataFrame:
    """
    Load 200-row validation set:
    - Template (keys order anchor)
    - Merge: TerraClimate PET + CHIRPS PPT + geo features (for dist_main_km etc.)
    """
    data_cfg = cfg["data"]
    template_path = ROOT / data_cfg["template_path"]
    tc_path       = ROOT / data_cfg["tc_valid_path"]
    ch_path       = ROOT / data_cfg["chirps_valid_path"]
    geo_path      = ROOT / data_cfg["geo_features_path"]

    template = pd.read_csv(template_path)          # raw, do NOT normalize (preserve output format)
    template_norm = normalize_keys(template)

    tc = normalize_keys(pd.read_csv(tc_path), dayfirst=True)   # TC: DD-MM-YYYY
    ch = normalize_keys(pd.read_csv(ch_path), dayfirst=False)  # CHIRPS: ISO YYYY-MM-DD

    geo = pd.read_csv(geo_path)
    geo[LAT_COL] = pd.to_numeric(geo[LAT_COL], errors="coerce")
    geo[LON_COL] = pd.to_numeric(geo[LON_COL], errors="coerce")

    # Build valid: start from template normalized keys
    valid = template_norm[[LAT_COL, LON_COL, DATE_COL]].copy()
    valid = valid.merge(tc[KEYS + ["pet"]], on=KEYS, how="left")
    valid = valid.merge(ch[KEYS + ["chirps_ppt"]], on=KEYS, how="left")
    valid = valid.merge(geo, on=[LAT_COL, LON_COL], how="left")

    print(f"[DATA] Validation: {valid.shape[0]} rows, {valid.shape[1]} cols")
    print(f"[DATA] Valid PET null: {valid['pet'].isna().sum()}")
    print(f"[DATA] Valid dist_main_km null: {valid['dist_main_km'].isna().sum()}")
    assert len(valid) == 200, f"Expected 200 validation rows, got {len(valid)}"
    return template, valid


# ── Feature Engineering ────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, sin_doy, cos_doy, year from Sample Date."""
    out = df.copy()
    if DATE_COL not in out.columns or not pd.api.types.is_datetime64_any_dtype(out[DATE_COL]):
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    out["month"]   = out[DATE_COL].dt.month
    out["year"]    = out[DATE_COL].dt.year
    doy            = out[DATE_COL].dt.dayofyear
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.0)
    return out


def add_regime_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add binary regime flags based on validated thresholds."""
    out = df.copy()
    rt = cfg["regime_thresholds"]
    m_flag   = int(rt["month_flag"])
    pet_thr  = float(rt["pet_high_threshold"])
    dist_thr = float(rt["dist_main_high_threshold"])

    if "month" not in out.columns:
        out = add_temporal_features(out)

    out["is_m12"]        = (out["month"] == m_flag).astype(int)
    out["is_pet_high"]   = (out["pet"].fillna(0) > pet_thr).astype(int)
    out["is_dist_high"]  = (out["dist_main_km"].fillna(0) > dist_thr).astype(int)
    # interaction flags
    out["is_pet_high_x_m12"]  = out["is_pet_high"]  * out["is_m12"]
    out["is_dist_high_x_m12"] = out["is_dist_high"] * out["is_m12"]
    return out


def get_residual_feature_cols(cfg: dict) -> list[str]:
    """Return ordered list of regime+temporal+hydro features for residual model."""
    spec = cfg["residual_model"]["features"]
    return (
        spec["temporal"]
        + spec["climate"]
        + spec["hydro_subset"]
        + spec["regime_flags"]
    )


def get_base_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    All numeric non-target, non-fold, non-key columns for the base model.
    Exclude fold columns and identifier columns.
    """
    drop_cols = {
        DRP_COL, TA_COL, EC_COL, DATE_COL, LAT_COL, LON_COL,
        "basin_id", "pfaf_id",
        "fold_current", "fold_basin", "fold_kmeans_8",
        "fold_kmeans_10", "fold_kmeans_12", "fold_kmeans_15",
    }
    return [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]


# ── Base Model OOF ─────────────────────────────────────────────────────────────

def run_base_oof(train_df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Run the base ET model in 5-fold OOF using pre-computed fold_basin assignments.
    Returns oof_preds array (shape = len(train_df),).
    """
    bm_cfg = cfg["base_model"]
    fold_col = bm_cfg["fold_column"]
    folds = sorted(train_df[fold_col].dropna().unique())
    n_folds = len(folds)

    model = ExtraTreesRegressor(
        n_estimators=int(bm_cfg["n_estimators"]),
        min_samples_leaf=int(bm_cfg["min_samples_leaf"]),
        max_features=bm_cfg["max_features"],
        random_state=int(bm_cfg["random_state"]),
        n_jobs=int(bm_cfg["n_jobs"]),
    )
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m",   model),
    ])

    y_all       = train_df[DRP_COL].astype(float).values
    oof_preds   = np.full(len(train_df), np.nan)
    base_scores = []
    feat_cols   = get_base_feature_cols(train_df)

    print(f"\n[BASE MODEL] Running {n_folds}-fold OOF | {len(feat_cols)} features")
    for fold_val in folds:
        tr_mask = train_df[fold_col] != fold_val
        va_mask = train_df[fold_col] == fold_val

        X_tr = train_df.loc[tr_mask, feat_cols].copy()
        y_tr = y_all[tr_mask.values]
        X_va = train_df.loc[va_mask, feat_cols].copy()
        y_va = y_all[va_mask.values]

        pipe_copy = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m",   ExtraTreesRegressor(
                n_estimators=int(bm_cfg["n_estimators"]),
                min_samples_leaf=int(bm_cfg["min_samples_leaf"]),
                max_features=bm_cfg["max_features"],
                random_state=int(bm_cfg["random_state"]),
                n_jobs=int(bm_cfg["n_jobs"]),
            )),
        ])
        pipe_copy.fit(X_tr, y_tr)
        pred_va = np.maximum(pipe_copy.predict(X_va), 0.0)
        oof_preds[va_mask.values] = pred_va
        sc = float(r2_score(y_va, pred_va))
        base_scores.append(sc)
        print(f"  Fold {int(fold_val)}: R2={sc:.4f}  (n_va={va_mask.sum()})")

    valid_mask = ~np.isnan(oof_preds)
    mean_r2 = float(np.mean(base_scores))
    print(f"[BASE MODEL] OOF R2 mean: {mean_r2:.4f} ± {float(np.std(base_scores)):.4f}")
    return oof_preds, base_scores, feat_cols


# ── Residual Model CV ──────────────────────────────────────────────────────────

def run_residual_cv(train_df: pd.DataFrame, oof_preds: np.ndarray, cfg: dict) -> dict:
    """
    Evaluate the RidgeCV residual model in the same 5-fold scheme.
    Residual target = y_true_drp - y_oof_base_drp.
    Features: temporal + climate + hydro_subset + regime_flags.
    """
    fold_col    = cfg["base_model"]["fold_column"]
    folds       = sorted(train_df[fold_col].dropna().unique())
    resid_feats = get_residual_feature_cols(cfg)
    alphas      = cfg["residual_model"]["alphas"]

    y_true      = train_df[DRP_COL].astype(float).values
    residuals   = y_true - oof_preds   # residual target

    resid_scores  = []
    corrected_scores = []
    oof_resid     = np.full(len(train_df), np.nan)
    oof_corrected = np.full(len(train_df), np.nan)

    print(f"\n[RESIDUAL MODEL] Running {len(folds)}-fold CV | {len(resid_feats)} features")
    print(f"  Features: {resid_feats}")

    for fold_val in folds:
        tr_mask = train_df[fold_col] != fold_val
        va_mask = train_df[fold_col] == fold_val

        X_tr_r = train_df.loc[tr_mask, resid_feats].copy()
        y_tr_r = residuals[tr_mask.values]
        X_va_r = train_df.loc[va_mask, resid_feats].copy()

        # True residuals in this fold's validation set
        resid_va_true = residuals[va_mask.values]
        y_va_true     = y_true[va_mask.values]
        oof_base_va   = oof_preds[va_mask.values]

        ridge = Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("ridge", RidgeCV(alphas=alphas, cv=3)),
        ])
        ridge.fit(X_tr_r, y_tr_r)
        resid_pred_va    = ridge.predict(X_va_r)
        corrected_pred   = np.maximum(oof_base_va + resid_pred_va, 0.0)

        sc_resid     = float(r2_score(resid_va_true, resid_pred_va))
        sc_corrected = float(r2_score(y_va_true, corrected_pred))
        resid_scores.append(sc_resid)
        corrected_scores.append(sc_corrected)
        oof_resid[va_mask.values]     = resid_pred_va
        oof_corrected[va_mask.values] = corrected_pred

        chosen_alpha = float(ridge.named_steps["ridge"].alpha_)
        print(f"  Fold {int(fold_val)}: resid_R2={sc_resid:.4f} | corrected_R2={sc_corrected:.4f}"
              f" | ridge_alpha={chosen_alpha:.1f}")

    mean_resid = float(np.mean(resid_scores))
    mean_corr  = float(np.mean(corrected_scores))
    print(f"[RESIDUAL MODEL] Residual CV R2: {mean_resid:.4f} ± {float(np.std(resid_scores)):.4f}")
    print(f"[RESIDUAL MODEL] Corrected CV R2: {mean_corr:.4f} ± {float(np.std(corrected_scores)):.4f}")

    return {
        "resid_scores":     [float(s) for s in resid_scores],
        "corrected_scores": [float(s) for s in corrected_scores],
        "mean_resid_r2":    mean_resid,
        "std_resid_r2":     float(np.std(resid_scores)),
        "mean_corrected_r2": mean_corr,
        "std_corrected_r2":  float(np.std(corrected_scores)),
        "oof_resid":        oof_resid,
        "oof_corrected":    oof_corrected,
    }


# ── Final Training and Prediction ─────────────────────────────────────────────

def train_final_residual_model(train_df: pd.DataFrame, residuals: np.ndarray, cfg: dict) -> Pipeline:
    """Train residual model on full training set."""
    resid_feats = get_residual_feature_cols(cfg)
    alphas = cfg["residual_model"]["alphas"]

    ridge = Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("ridge", RidgeCV(alphas=alphas, cv=5)),
    ])
    ridge.fit(train_df[resid_feats], residuals)
    chosen = float(ridge.named_steps["ridge"].alpha_)
    print(f"[FINAL RESIDUAL] Best alpha={chosen:.1f} | coef sum: {np.abs(ridge.named_steps['ridge'].coef_).sum():.4f}")
    return ridge, resid_feats


def build_corrected_submission(
    template_raw: pd.DataFrame,
    valid_df: pd.DataFrame,
    resid_model: Pipeline,
    resid_feats: list[str],
    anchor_path: str,
    alpha: float,
) -> pd.DataFrame:
    """
    Build 200-row submission:
    - TA/EC: from V4_4 anchor (exact)
    - DRP: V4_4_drp + alpha * resid_model_pred
    """
    anchor = pd.read_csv(anchor_path)
    assert len(anchor) == 200, f"Anchor rows={len(anchor)}"

    resid_pred_valid = resid_model.predict(valid_df[resid_feats].copy())
    v44_drp = anchor[DRP_COL].values.astype(float)
    pred_drp_corrected = np.maximum(v44_drp + alpha * resid_pred_valid, 0.0)

    print(f"[SUBMISSION] DRP correction stats: min={resid_pred_valid.min():.4f} "
          f"max={resid_pred_valid.max():.4f} mean={resid_pred_valid.mean():.4f}")
    print(f"[SUBMISSION] V4_4 DRP: mean={v44_drp.mean():.4f} | Corrected: mean={pred_drp_corrected.mean():.4f}")

    out = template_raw.copy(deep=True)
    # Set TA and EC from anchor exactly
    out[TA_COL] = anchor[TA_COL].values
    out[EC_COL] = anchor[EC_COL].values
    out[DRP_COL] = pred_drp_corrected

    assert out.shape == (200, 6), f"Unexpected shape: {out.shape}"
    assert out.isnull().sum().sum() == 0, "NaN in submission!"
    assert (out[DRP_COL] >= 0).all(), "Negative DRP detected!"

    # Correlation and MAE to anchor
    corr = float(np.corrcoef(v44_drp, pred_drp_corrected)[0, 1])
    mae  = float(np.mean(np.abs(v44_drp - pred_drp_corrected)))
    print(f"[SUBMISSION] corr_to_V4_4={corr:.6f} | MAE_to_V4_4={mae:.6f}")

    return out, resid_pred_valid, corr, mae


# ── Artifact Saving ────────────────────────────────────────────────────────────

def save_artifacts(
    exp_dir: Path,
    submission: pd.DataFrame,
    sub_name: str,
    base_scores: list,
    base_feat_cols: list,
    cv_results: dict,
    resid_coef: np.ndarray,
    resid_feats: list,
    chosen_alpha: float,
    v44_drp: np.ndarray,
    resid_pred: np.ndarray,
    corr_to_v44: float,
    mae_to_v44: float,
    cfg: dict,
) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Submission CSV
    sub_path = exp_dir / sub_name
    submission.to_csv(sub_path, index=False)
    print(f"[ARTIFACTS] Submission: {sub_path}")

    # CV report
    cv_report = {
        "generated_at": ts(),
        "config": {
            "base_model": cfg["base_model"],
            "residual_features": resid_feats,
            "alpha": cfg["correction"]["alpha"],
            "regime_thresholds": cfg["regime_thresholds"],
        },
        "base_model_cv": {
            "mean": float(np.mean(base_scores)),
            "std": float(np.std(base_scores)),
            "folds": base_scores,
        },
        "residual_model_cv": {
            "mean": cv_results["mean_resid_r2"],
            "std":  cv_results["std_resid_r2"],
            "folds": cv_results["resid_scores"],
        },
        "corrected_model_cv": {
            "mean": cv_results["mean_corrected_r2"],
            "std":  cv_results["std_corrected_r2"],
            "folds": cv_results["corrected_scores"],
        },
    }
    with open(exp_dir / "cv_report.json", "w", encoding="utf-8") as f:
        json.dump(cv_report, f, indent=2)

    # Metadata
    metadata = {
        "generated_at":         ts(),
        "exp_name":             exp_dir.name,
        "anchor_lb":            0.3039,
        "anchor_submission":    cfg["anchor"]["submission_path"],
        "base_model_type":      cfg["base_model"]["type"],
        "residual_model_type":  cfg["residual_model"]["type"],
        "ridge_alpha":          float(chosen_alpha),
        "residual_features":    resid_feats,
        "n_base_features":      len(base_feat_cols),
        "correction_alpha":     float(cfg["correction"]["alpha"]),
        "regime_thresholds":    cfg["regime_thresholds"],
        "base_model_cv_mean":   float(np.mean(base_scores)),
        "residual_cv_mean":     cv_results["mean_resid_r2"],
        "corrected_cv_mean":    cv_results["mean_corrected_r2"],
        "submission_drp_mean":  float(submission[DRP_COL].mean()),
        "submission_drp_std":   float(submission[DRP_COL].std()),
        "submission_drp_min":   float(submission[DRP_COL].min()),
        "submission_drp_max":   float(submission[DRP_COL].max()),
        "corr_to_v44":          corr_to_v44,
        "mae_to_v44":           mae_to_v44,
        "submission_file":      sub_name,
    }
    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Ridge coefficients for interpretability
    coef_df = pd.DataFrame({
        "feature": resid_feats,
        "coefficient": list(resid_coef),
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    coef_df.to_csv(exp_dir / "residual_model_coef.csv", index=False)

    print(f"[ARTIFACTS] cv_report.json, metadata.json, residual_model_coef.csv saved to {exp_dir}")
    print("\n[COEFFICIENTS] Top residual model coefficients:")
    for _, row in coef_df.head(10).iterrows():
        print(f"  {row['feature']:30s}  {row['coefficient']:+.6f}")


def update_experiment_index(exp_dir: Path, metadata: dict) -> None:
    """Append a row to experiments/experiment_index.csv."""
    index_path = ROOT / "experiments" / "experiment_index.csv"
    cols = [
        "experiment_id", "generated_at", "base_model_cv_mean",
        "residual_cv_mean", "corrected_cv_mean",
        "correction_alpha", "corr_to_v44", "mae_to_v44",
        "submission_file", "notes",
    ]
    row = {
        "experiment_id":       exp_dir.name,
        "generated_at":        metadata["generated_at"],
        "base_model_cv_mean":  metadata["base_model_cv_mean"],
        "residual_cv_mean":    metadata["residual_cv_mean"],
        "corrected_cv_mean":   metadata["corrected_cv_mean"],
        "correction_alpha":    metadata["correction_alpha"],
        "corr_to_v44":         metadata["corr_to_v44"],
        "mae_to_v44":          metadata["mae_to_v44"],
        "submission_file":     metadata["submission_file"],
        "notes":               "residual_drp_v1: temporal+regime RidgeCV correction on V4_4",
    }
    if index_path.exists():
        df = pd.read_csv(index_path)
    else:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(index_path, index=False)
    print(f"[INDEX] experiment_index.csv updated ({len(df)} rows)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_exp_residual_drp_v1.yml")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        cfg_path = Path(args.config)
    cfg = load_cfg(str(cfg_path))

    print("=" * 70)
    print(f"RESIDUAL DRP SPRINT V1  |  started: {ts()}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_df   = load_training_data(cfg)
    template_raw, valid_df = load_validation_data(cfg)

    # ── 2. Add temporal + regime features to TRAIN ────────────────────────────
    train_df = add_temporal_features(train_df)
    train_df = add_regime_flags(train_df, cfg)

    # ── 3. Add temporal + regime features to VALID ────────────────────────────
    valid_df = add_temporal_features(valid_df)
    valid_df = add_regime_flags(valid_df, cfg)

    # Verify regime flag coverage in validation
    print(f"\n[REGIME] Training flag counts:")
    for flag in ["is_m12", "is_pet_high", "is_dist_high"]:
        n = int(train_df[flag].sum())
        print(f"  {flag}: {n}/{len(train_df)} ({100*n/len(train_df):.1f}%)")
    print(f"[REGIME] Validation flag counts:")
    for flag in ["is_m12", "is_pet_high", "is_dist_high"]:
        n = int(valid_df[flag].sum())
        print(f"  {flag}: {n}/{len(valid_df)} ({100*n/len(valid_df):.1f}%)")

    # ── 4. Base model OOF (V4_4-equivalent) ──────────────────────────────────
    oof_preds, base_scores, base_feat_cols = run_base_oof(train_df, cfg)

    # ── 5. Residuals on training ──────────────────────────────────────────────
    y_true    = train_df[DRP_COL].astype(float).values
    residuals = y_true - oof_preds
    print(f"\n[RESIDUALS] mean={residuals.mean():.4f} std={residuals.std():.4f} "
          f"min={residuals.min():.4f} max={residuals.max():.4f}")

    # ── 6. Residual model CV ──────────────────────────────────────────────────
    cv_results = run_residual_cv(train_df, oof_preds, cfg)

    # ── 7. Train final residual model on full training set ────────────────────
    resid_model, resid_feats = train_final_residual_model(train_df, residuals, cfg)
    ridge_step  = resid_model.named_steps["ridge"]
    chosen_alpha = float(ridge_step.alpha_)
    resid_coef   = ridge_step.coef_

    # ── 8. Build corrected submission ─────────────────────────────────────────
    alpha = float(cfg["correction"]["alpha"])
    anchor_path = str(ROOT / cfg["anchor"]["submission_path"])

    submission, resid_pred_valid, corr_to_v44, mae_to_v44 = build_corrected_submission(
        template_raw, valid_df, resid_model, resid_feats, anchor_path, alpha
    )

    # ── 9. Save artifacts ──────────────────────────────────────────────────────
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name    = f"residual_drp_v1_{run_ts}"
    exp_dir     = ROOT / "experiments" / exp_name
    sub_name    = f"submission_{exp_name}.csv"

    anchor   = pd.read_csv(anchor_path)
    v44_drp  = anchor[DRP_COL].values.astype(float)

    # Rebuild metadata for index update
    metadata = {
        "generated_at":        ts(),
        "exp_name":            exp_name,
        "base_model_cv_mean":  float(np.mean(base_scores)),
        "residual_cv_mean":    cv_results["mean_resid_r2"],
        "corrected_cv_mean":   cv_results["mean_corrected_r2"],
        "correction_alpha":    alpha,
        "corr_to_v44":         corr_to_v44,
        "mae_to_v44":          mae_to_v44,
        "submission_file":     sub_name,
    }

    save_artifacts(
        exp_dir       = exp_dir,
        submission    = submission,
        sub_name      = sub_name,
        base_scores   = base_scores,
        base_feat_cols= base_feat_cols,
        cv_results    = cv_results,
        resid_coef    = resid_coef,
        resid_feats   = resid_feats,
        chosen_alpha  = chosen_alpha,
        v44_drp       = v44_drp,
        resid_pred    = resid_pred_valid,
        corr_to_v44   = corr_to_v44,
        mae_to_v44    = mae_to_v44,
        cfg           = cfg,
    )

    update_experiment_index(exp_dir, metadata)

    # ── 10. Final decision guidance ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPRINT SUMMARY")
    print("=" * 70)
    print(f"  Base model OOF R2 (DRP):     {np.mean(base_scores):.4f}")
    print(f"  Residual model CV R2:        {cv_results['mean_resid_r2']:.4f}")
    print(f"  Corrected DRP CV R2:         {cv_results['mean_corrected_r2']:.4f}")
    print(f"  Alpha applied:               {alpha}")
    print(f"  Corr to V4_4 (DRP):          {corr_to_v44:.6f}")
    print(f"  MAE to V4_4 (DRP):           {mae_to_v44:.4f}")
    print(f"  Submission DRP mean:         {submission[DRP_COL].mean():.4f}")
    print(f"  Experiment folder:           {exp_dir}")

    if cv_results["mean_resid_r2"] > 0.01:
        print("\n  ✓ RESIDUAL SIGNAL DETECTED — submission may improve on V4_4 LB=0.3039")
        print("    Recommendation: SUBMIT")
    elif cv_results["mean_resid_r2"] > -0.01:
        print("\n  ~ NEGLIGIBLE SIGNAL — correction is near-neutral")
        print("    Recommendation: LOW CONFIDENCE SUBMIT (conservative alpha protects)")
    else:
        print("\n  ✗ NEGATIVE RESIDUAL SIGNAL — residual model hurts on CV")
        print("    Recommendation: DO NOT SUBMIT — residual model adds noise")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
