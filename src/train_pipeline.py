#!/usr/bin/env python
# coding: utf-8

import os, json, argparse, datetime, shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor

# Custom ensemble regressor
from sklearn.base import BaseEstimator, RegressorMixin

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        self.estimators = estimators
    
    def fit(self, X, y):
        self.estimators_ = []
        for name, est in self.estimators:
            fitted_est = clone(est).fit(X, y)
            self.estimators_.append((name, fitted_est))
        return self
    
    def predict(self, X):
        preds = [est.predict(X) for name, est in self.estimators_]
        return np.mean(preds, axis=0)

# ----------------------------
# YAML loader
# ----------------------------
def load_yaml(path: str) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Falta PyYAML. Instala con: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"
KEYS = [LAT_COL, LON_COL, DATE_COL]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def now_run_id(prefix="exp"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"

def safe_write_text(path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def safe_write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def start_experiment_run(cfg: dict, cache_dir: str):
    exp_dir = cfg.get("project", {}).get("exp_dir", "experiments")
    ensure_dir(exp_dir)
    run_id = now_run_id("exp")
    run_path = os.path.join(exp_dir, run_id)
    ensure_dir(run_path)

    # snapshot config (json)
    safe_write_json(os.path.join(run_path, "config_snapshot.json"), cfg)

    # pointer for other scripts (batch_blends)
    ensure_dir(cache_dir)
    safe_write_text(os.path.join(cache_dir, "last_run_path.txt"), run_path + "\n")

    return run_id, run_path

def _load_manifest_count(path: str):
    """Return feature count from a manifest JSON (list or dict with 'columns'/'features' key)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        mf = json.load(f)
    cols = mf if isinstance(mf, list) else mf.get("columns", mf.get("features", []))
    return len(cols)


def _update_experiment_index(run_path: str, run_id: str, metadata: dict):
    """Upsert a row for run_id into experiments/experiment_index.csv."""
    index_path = os.path.join(os.path.dirname(run_path), "experiment_index.csv")
    cols = [
        "experiment_id", "cv_mean", "drp_cv", "model_family",
        "y_mode_drp", "feature_count_drp", "submission_path",
    ]
    cv_vals = [v["mean"] for v in metadata.get("cv_metrics", {}).values()
               if isinstance(v, dict) and "mean" in v]
    cv_mean = round(sum(cv_vals) / len(cv_vals), 6) if cv_vals else ""
    drp_cv = metadata.get("cv_metrics", {}).get("Dissolved Reactive Phosphorus", {}).get("mean", "")
    if isinstance(drp_cv, float):
        drp_cv = round(drp_cv, 6)
    row = {
        "experiment_id": run_id,
        "cv_mean": cv_mean,
        "drp_cv": drp_cv,
        "model_family": metadata.get("model_config", {}).get("drp_model", ""),
        "y_mode_drp": metadata.get("model_config", {}).get("drp_y_mode", ""),
        "feature_count_drp": metadata.get("feature_counts", {}).get("Dissolved Reactive Phosphorus", ""),
        "submission_path": metadata.get("submission_path", ""),
    }
    if os.path.exists(index_path):
        df = pd.read_csv(index_path)
    else:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[df["experiment_id"] != run_id]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(index_path, index=False)
    print(f"experiment_index.csv updated ({len(df)} rows) -> {index_path}")


def normalize_keys_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    """Solo para matching interno (NO para template crudo)."""
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    return out

def make_groups(df, grid, strategy="basin"):
    """
    Spatial grouping for fold generation.
    strategy: "basin" (preferred) or "grid" (fallback)
    Returns grouping array for GroupKFold.
    """
    if strategy == "basin" and 'basin_id' in df.columns:
        # Basin-aware grouping (eliminates spatial leakage across basins)
        basin_id = df['basin_id'].fillna('unknown').astype(str)
        n_unique = basin_id.nunique()
        n_missing = (df['basin_id'].isna()).sum()
        print(f"\n>>> GROUPING: Basin-aware strategy")
        print(f"    Unique basins: {n_unique}")
        print(f"    Missing basin_id: {n_missing}/{len(df)} ({100*n_missing/len(df):.2f}%)")
        return basin_id
    elif strategy == "basin" and 'basin_id' not in df.columns:
        # Fallback warning
        print(f"\n>>> WARNING: Basin grouping requested but basin_id NOT found in data.")
        print(f"    Falling back to grid-based grouping.")
    
    # Grid-based fallback
    groups = (
        np.floor(df[LAT_COL] / grid).astype(int).astype(str)
        + "_"
        + np.floor(df[LON_COL] / grid).astype(int).astype(str)
    )
    n_unique = groups.nunique()
    print(f"\n>>> GROUPING: Grid-based strategy")
    print(f"    Grid size: {grid}")
    print(f"    Unique groups: {n_unique}")
    return groups

def y_transform_fit(y_tr: np.ndarray, mode: str):
    y_tr = np.asarray(y_tr, dtype=float)
    if mode == "none":
        return (lambda x: np.asarray(x, dtype=float),
                lambda x: np.asarray(x, dtype=float))
    if mode == "winsor":
        lo = np.nanquantile(y_tr, 0.01)
        hi = np.nanquantile(y_tr, 0.99)
        fwd = lambda x: np.clip(np.asarray(x, dtype=float), lo, hi)
        inv = lambda x: np.asarray(x, dtype=float)
        return fwd, inv
    if mode == "clip_p95":
        # Clip at 95th percentile (no inverse transform needed for Poisson)
        p95 = np.nanquantile(y_tr, 0.95)
        fwd = lambda x: np.minimum(np.asarray(x, dtype=float), p95)
        inv = lambda x: np.asarray(x, dtype=float)
        return fwd, inv
    if mode == "clip_p99":
        # Clip at 99th percentile (no inverse transform needed for Poisson)
        p99 = np.nanquantile(y_tr, 0.99)
        fwd = lambda x: np.minimum(np.asarray(x, dtype=float), p99)
        inv = lambda x: np.asarray(x, dtype=float)
        return fwd, inv
    if mode == "sqrt":
        fwd = lambda x: np.sqrt(np.maximum(np.asarray(x, dtype=float), 0.0))
        inv = lambda x: np.square(np.asarray(x, dtype=float))
        return fwd, inv
    raise ValueError(f"Unknown y_mode: {mode}")

# ----------------------------
# DRP target transform (invertible)
# ----------------------------
def drp_transform_fwd_inv(y_tr: np.ndarray, transform_type: str):
    """
    Returns (fwd, inv) pair for DRP target transformation.
    transform_type: "none", "log1p", or "sqrt"
    """
    y_tr = np.asarray(y_tr, dtype=float)
    
    if transform_type == "none":
        return (lambda x: np.asarray(x, dtype=float),
                lambda x: np.asarray(x, dtype=float))
    
    if transform_type == "log1p":
        # y_tr stays same, but we train on log1p(y_tr) and invert with expm1
        fwd = lambda x: np.log1p(np.asarray(x, dtype=float))
        inv = lambda x: np.expm1(np.asarray(x, dtype=float))
        return fwd, inv
    
    if transform_type == "sqrt":
        # y_tr stays same, but we train on sqrt(y_tr) and invert with square
        fwd = lambda x: np.sqrt(np.maximum(np.asarray(x, dtype=float), 0.0))
        inv = lambda x: np.square(np.asarray(x, dtype=float))
        return fwd, inv
    
    raise ValueError(f"Unknown DRP transform: {transform_type}")

# ----------------------------
# Regularization parameter mapper
# ----------------------------
def apply_regularization_to_cfg(cfg: dict, reg_level: str, model_name: str):
    """
    Returns modified cfg for regularization adjustments.
    reg_level: "default" or "stronger"
    model_name: "ExtraTreesRegressor", "RandomForestRegressor", "HistGradientBoostingRegressor"
    """
    import copy
    cfg = copy.deepcopy(cfg)  # Don't modify original
    
    if reg_level == "default":
        # No changes - use defaults
        return cfg
    
    if reg_level != "stronger":
        raise ValueError(f"Unknown regularization level: {reg_level}")
    
    # Stronger regularization
    if model_name in ["ExtraTreesRegressor", "RandomForestRegressor"]:
        # Shallower trees, higher min_samples_leaf (ET/RF trees)
        cfg["model"]["et_cv"]["min_samples_leaf"] = 5
        cfg["model"]["et_final"]["min_samples_leaf"] = 5
    
    elif model_name == "HistGradientBoostingRegressor":
        # Lower learning rate, higher regularization, higher min_samples_leaf
        if "cv_by_target" in cfg["model"] and "Dissolved Reactive Phosphorus" in cfg["model"]["cv_by_target"]:
            cfg["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["params"]["learning_rate"] = 0.015
            cfg["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["params"]["l2_regularization"] = 1.0
            cfg["model"]["cv_by_target"]["Dissolved Reactive Phosphorus"]["params"]["min_samples_leaf"] = 50
        
        if "final_by_target" in cfg["model"] and "Dissolved Reactive Phosphorus" in cfg["model"]["final_by_target"]:
            cfg["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["params"]["learning_rate"] = 0.015
            cfg["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["params"]["l2_regularization"] = 1.0
            cfg["model"]["final_by_target"]["Dissolved Reactive Phosphorus"]["params"]["min_samples_leaf"] = 50
    
    return cfg

# ----------------------------
# Model factory override for experiments
# ----------------------------
def build_model_from_cfg(cfg: dict, which: str, target: str, exp_model_override: str = None):
    """
    which: "cv" or "final"
    exp_model_override: optional experiment override for DRP (e.g., "RandomForestRegressor")
    Supports:
      - ExtraTreesRegressor (default)
      - RandomForestRegressor
      - HistGradientBoostingRegressor (useful for DRP with loss='poisson')
      - "ensemble" : averages ET, RF, HGB
    """
    # stacking option (only for DRP) takes precedence
    if target == "Dissolved Reactive Phosphorus" and cfg.get("model", {}).get("stacking", {}).get("enabled", False):
        # build stacking regressor
        # base models: ET (max_depth=10), HGB (params from cfg), Ridge
        et_params = cfg.get("model", {}).get(f"et_{which}", {}).copy()
        et_params["max_depth"] = 10
        base_et = ExtraTreesRegressor(**et_params)
        hgb_key = "hgb_cv_drp" if which == "cv" else "hgb_final_drp"
        hgb_params = cfg.get("model", {}).get(hgb_key, {}).copy()
        base_hgb = HistGradientBoostingRegressor(**hgb_params)
        base_ridge = Ridge()
        meta_choice = cfg.get("model", {}).get("stacking", {}).get("meta", "ridgecv")
        meta = RidgeCV() if meta_choice == "ridgecv" else LassoCV()
        return StackingRegressor(
            estimators=[("et", base_et), ("hgb", base_hgb), ("ridge", base_ridge)],
            final_estimator=meta,
            passthrough=True,
            cv=5,
        )

    # Use override if provided and target is DRP
    if exp_model_override and target == "Dissolved Reactive Phosphorus":
        model_name = exp_model_override
    else:
        model_cfg = cfg.get("model", {})
        # optional per-target override
        by_t = model_cfg.get(f"{which}_by_target", {}) or {}
        spec = by_t.get(target)

        if spec is None:
            # fallback: original behavior
            params = model_cfg.get(f"et_{which}", {})
            return ExtraTreesRegressor(**params)

        model_name = str(spec.get("name", "")).strip()
    
    # Special case for VotingRegressor
    if model_name == "VotingRegressor":
        by_t = cfg.get("model", {}).get(f"{which}_by_target", {}) or {}
        spec = by_t.get(target) or {}
        estimators_spec = spec.get("estimators", [])
        estimators = []
        weights = []
        for est_spec in estimators_spec:
            est_name = est_spec["name"]
            est_params = est_spec.get("params", {})
            if est_name == "ExtraTreesRegressor":
                est = ExtraTreesRegressor(**est_params)
            elif est_name == "Ridge":
                # Wrap Ridge with StandardScaler
                from sklearn.preprocessing import StandardScaler
                est = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(**est_params))
                ])
            else:
                raise ValueError(f"Unknown estimator in VotingRegressor: {est_name}")
            estimators.append((est_name.lower(), est))
            weights.append(est_spec.get("weight", 1.0))
        return VotingRegressor(estimators=estimators, weights=weights)
    
    # Get params based on which (cv or final)
    # For ET/RF: prefer per-target params from {which}_by_target if explicit params are present,
    # otherwise fall back to the global et_{which} block.
    if model_name in ["ExtraTreesRegressor", "RandomForestRegressor"]:
        _by_t = cfg.get("model", {}).get(f"{which}_by_target", {}) or {}
        _spec = _by_t.get(target) or {}
        _per_target_params = _spec.get("params") or {}
        if _per_target_params:
            params = _per_target_params
        else:
            params = cfg.get("model", {}).get(f"et_{which}", {})
    elif model_name == "HistGradientBoostingRegressor":
        by_t = cfg.get("model", {}).get(f"{which}_by_target", {}) or {}
        spec = by_t.get(target) or {}
        params = spec.get("params", {}) or {}
    elif model_name == "CatBoostRegressor":
        by_t = cfg.get("model", {}).get(f"{which}_by_target", {}) or {}
        spec = by_t.get(target) or {}
        params = spec.get("params", {}) or {}
    elif model_name == "VotingRegressor":
        # Already handled above
        pass
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if model_name == "ExtraTreesRegressor":
        return ExtraTreesRegressor(**params)
    
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(**params)

    if model_name == "HistGradientBoostingRegressor":
        # NOTE: for loss='poisson', y must be >= 0
        return HistGradientBoostingRegressor(**params)

    if model_name == "CatBoostRegressor":
        return CatBoostRegressor(**params)

    raise ValueError(f"Unknown model: {model_name}")

# ----------------------------
# Feature Engineering (V4)
# ----------------------------

# Utility: one-hot encode top-N landcover categories (applied during dataset build)
def add_landcover_ohe(train_df: pd.DataFrame, valid_df: pd.DataFrame, top_n: int = 5):
    if "landcover" not in train_df.columns:
        return train_df, valid_df
    top_categories = train_df["landcover"].value_counts().nlargest(top_n).index.tolist()
    for cat in top_categories:
        col = f"landcover_{cat}"
        train_df[col] = (train_df["landcover"] == cat).astype(int)
        valid_df[col] = (valid_df["landcover"] == cat).astype(int)
    return train_df, valid_df

# Utility: compute distance to basin centroid
from math import radians, cos, sin, sqrt, atan2

def _haversine(lat1, lon1, lat2, lon2):
    # returns distance in degrees approximation (small distances)
    # we use simple Euclidean on lat/lon because projection distortions are small at scale
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def add_dist_to_centroid(df: pd.DataFrame, centroid_map: dict):
    df = df.copy()
    if "basin_id" not in df.columns:
        df["dist_to_centroid"] = np.nan
        return df
    def compute(row):
        bid = row.get("basin_id")
        if pd.isna(bid):
            return np.nan
        c = centroid_map.get(bid)
        if c is None:
            return np.nan
        return _haversine(row[LAT_COL], row[LON_COL], c[0], c[1])
    df["dist_to_centroid"] = df.apply(compute, axis=1)
    return df


def add_basin_context_pack(train_df: pd.DataFrame, valid_df: pd.DataFrame, cfg: dict):
    """
    Optional Sprint-1 basin/catchment feature family.
    Train-valid safe: basin-level maps are fitted on train only and applied to valid.
    """
    features_cfg = cfg.get("features", {}) if cfg else {}
    basin_cfg = features_cfg.get("basin_context", {})
    if not basin_cfg.get("enabled", False):
        return train_df, valid_df, []

    tr = train_df.copy()
    va = valid_df.copy()
    created = []
    subset_only = bool(basin_cfg.get("subset_only", False))
    include_features = basin_cfg.get("include_features", []) or []

    # Row-level, static basin/catchment proxies
    for df in (tr, va):
        if "landcover" in df.columns:
            if "landcover_urban" not in df.columns:
                df["landcover_urban"] = (df["landcover"] == 50).astype(int)
            if "landcover_cropland" not in df.columns:
                df["landcover_cropland"] = (df["landcover"] == 40).astype(int)

        if "basin_area_km2" in df.columns:
            df["log_basin_area_km2"] = np.log1p(df["basin_area_km2"].clip(lower=0))
        if "upstream_area_km2" in df.columns:
            df["log_upstream_area_km2"] = np.log1p(df["upstream_area_km2"].clip(lower=0))
        if "dist_to_river_m" in df.columns:
            df["log_dist_to_river_m"] = np.log1p(df["dist_to_river_m"].clip(lower=0))

        if "upstream_area_km2" in df.columns and "slope" in df.columns:
            df["basin_slope_loading"] = df["upstream_area_km2"] * df["slope"]
        if "upstream_area_km2" in df.columns and "soil_clay_0_5" in df.columns:
            df["basin_clay_pressure"] = df["upstream_area_km2"] * df["soil_clay_0_5"]
        if "upstream_area_km2" in df.columns and "soil_ph_0_5" in df.columns:
            df["basin_ph_buffer"] = df["upstream_area_km2"] * df["soil_ph_0_5"]
        if "upstream_area_km2" in df.columns and "river_discharge_cms" in df.columns:
            df["discharge_contact_proxy"] = df["upstream_area_km2"] / (df["river_discharge_cms"] + 1.0)
        if "upstream_area_km2" in df.columns and "dist_to_river_m" in df.columns:
            df["connectivity_proxy"] = df["upstream_area_km2"] / (df["dist_to_river_m"] + 1.0)
        if "upstream_area_km2" in df.columns and "landcover_urban" in df.columns and "landcover_cropland" in df.columns:
            df["landuse_pressure_proxy"] = df["upstream_area_km2"] * (df["landcover_urban"] + df["landcover_cropland"])

    # Basin-level aggregates (fit on train only)
    if "basin_id" in tr.columns:
        agg_cols = [
            c for c in [
                "soil_clay_0_5",
                "soil_ph_0_5",
                "landcover_urban",
                "landcover_cropland",
                "slope",
                "elevation",
            ] if c in tr.columns
        ]
        for c in agg_cols:
            basin_col = f"basin_mean_{c}"
            m = tr.groupby("basin_id")[c].mean()
            global_mean = float(tr[c].mean())
            tr[basin_col] = tr["basin_id"].map(m).fillna(global_mean)
            if "basin_id" in va.columns:
                va[basin_col] = va["basin_id"].map(m).fillna(global_mean)
            else:
                va[basin_col] = global_mean

    # Optional strict subset mode for controlled family-combination experiments.
    if subset_only and include_features:
        base_cols_local = set(train_df.columns)
        added_cols = [c for c in tr.columns if c not in base_cols_local]
        drop_cols = [c for c in added_cols if c not in include_features]
        if drop_cols:
            tr = tr.drop(columns=drop_cols, errors="ignore")
            va = va.drop(columns=drop_cols, errors="ignore")

    # Track created columns for audit
    base_cols = set(train_df.columns)
    created = [c for c in tr.columns if c not in base_cols]
    return tr, va, sorted(created)


def add_temporal_context_pack(df: pd.DataFrame, cfg: dict):
    """
    Optional Sprint-2 temporal memory/lag feature family.
    Adds only temporal features derived from existing time-series covariates.
    """
    features_cfg = cfg.get("features", {}) if cfg else {}
    temp_cfg = features_cfg.get("temporal_context", {})
    if not temp_cfg.get("enabled", False):
        return df.copy(), []

    out = df.copy()
    base_cols = set(out.columns)

    # Temporal cyclic features from sample date
    if DATE_COL in out.columns:
        if "dayofyear" not in out.columns:
            out["dayofyear"] = out[DATE_COL].dt.dayofyear
        if "month" not in out.columns:
            out["month"] = out[DATE_COL].dt.month
        out["sin_doy"] = np.sin(2.0 * np.pi * out["dayofyear"] / 365.0)
        out["cos_doy"] = np.cos(2.0 * np.pi * out["dayofyear"] / 365.0)
        out["sin_month"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2.0 * np.pi * out["month"] / 12.0)

    # Station-wise lag/rolling memory features
    if "chirps_ppt" in out.columns:
        out = out.sort_values([LAT_COL, LON_COL, DATE_COL]).copy()
        g_rain = out.groupby([LAT_COL, LON_COL])["chirps_ppt"]

        out["rain_lag_1"] = g_rain.shift(1)
        out["rain_lag_3"] = g_rain.shift(3)
        out["rain_lag_7"] = g_rain.shift(7)
        out["rain_lag_14"] = g_rain.shift(14)

        out["rain_roll_3"] = g_rain.shift(1).rolling(window=3, min_periods=1).sum()
        out["rain_roll_7"] = g_rain.shift(1).rolling(window=7, min_periods=1).sum()
        out["rain_roll_14"] = g_rain.shift(1).rolling(window=14, min_periods=1).sum()
        out["rain_roll_30"] = g_rain.shift(1).rolling(window=30, min_periods=1).sum()

        # Observation-based dry-spell proxy
        out["dry_spell_obs"] = g_rain.transform(
            lambda x: x.shift(1).rolling(window=14, min_periods=1).apply(lambda y: float(np.sum(np.nan_to_num(y) <= 0.1)), raw=True)
        )

        # Short anomaly proxy relative to rolling baseline
        baseline_30 = g_rain.shift(1).rolling(window=30, min_periods=5).mean()
        out["rain_anom_30"] = out["chirps_ppt"] - baseline_30

    if "pet" in out.columns:
        out = out.sort_values([LAT_COL, LON_COL, DATE_COL]).copy()
        g_pet = out.groupby([LAT_COL, LON_COL])["pet"]
        out["pet_roll_30"] = g_pet.shift(1).rolling(window=30, min_periods=1).sum()
        if "rain_roll_30" in out.columns:
            out["rain_to_pet_30"] = out["rain_roll_30"] / (out["pet_roll_30"] + 1.0)

    created = [c for c in out.columns if c not in base_cols]
    return out, sorted(created)


def add_upstream_context_pack(df: pd.DataFrame, cfg: dict):
    """
    Optional Sprint-3 upstream hydrologic pressure feature family.
    Adds upstream-pressure proxies from already-available local covariates.
    """
    features_cfg = cfg.get("features", {}) if cfg else {}
    upstream_cfg = features_cfg.get("upstream_context", {})
    if not upstream_cfg.get("enabled", False):
        return df.copy(), []

    out = df.copy()
    base_cols = set(out.columns)

    up = pd.to_numeric(out.get("upstream_area_km2", np.nan), errors="coerce")
    slope = pd.to_numeric(out.get("slope", np.nan), errors="coerce")
    discharge = pd.to_numeric(out.get("river_discharge_cms", np.nan), errors="coerce")
    dist = pd.to_numeric(out.get("dist_to_river_m", np.nan), errors="coerce")
    clay = pd.to_numeric(out.get("soil_clay_0_5", np.nan), errors="coerce")
    soc = pd.to_numeric(out.get("soil_soc_0_5", np.nan), errors="coerce")

    # Core upstream transport/pressure proxies
    if "upstream_area_km2" in out.columns and "slope" in out.columns:
        out["upstream_stream_power"] = up * slope
    if "upstream_area_km2" in out.columns and "river_discharge_cms" in out.columns:
        out["upstream_discharge_contact"] = up / (discharge + 1.0)
    if "upstream_area_km2" in out.columns and "dist_to_river_m" in out.columns:
        out["upstream_connectivity"] = up / (dist + 1.0)
        out["upstream_river_decay"] = up * np.exp(-dist / 5000.0)

    if "upstream_area_km2" in out.columns and "soil_clay_0_5" in out.columns:
        out["upstream_soil_clay_pressure"] = up * clay
    if "upstream_area_km2" in out.columns and "soil_soc_0_5" in out.columns:
        out["upstream_soc_pressure"] = up * soc

    # Rainfall-weighted pressure (uses existing CHIRPS-derived columns when present)
    rain_cols = [
        c for c in ["chirps_ppt", "ppt_obs30_mean", "ppt_obs60_mean", "ppt_obs90_mean"] if c in out.columns
    ]
    for c in rain_cols:
        rain = pd.to_numeric(out[c], errors="coerce")
        if "upstream_area_km2" in out.columns:
            out[f"upstream_{c}_pressure"] = up * rain
        if "upstream_area_km2" in out.columns and "river_discharge_cms" in out.columns:
            out[f"upstream_{c}_transport"] = (up * rain) / (discharge + 1.0)

    created = [c for c in out.columns if c not in base_cols]
    return out, sorted(created)

# Utility: basin-level target encoding (fold-safe when called inside CV)
def add_basin_target_mean(train_df: pd.DataFrame,
                          valid_df: pd.DataFrame,
                          target: str,
                          n_splits: int = 5):
    tr = train_df.copy()
    va = valid_df.copy()

    if "basin_id" not in tr.columns:
        # nothing to do
        tr["basin_te"] = np.nan
        va["basin_te"] = np.nan
        return tr, va

    groups = tr["basin_id"].fillna("unknown").astype(str).values
    gkf = GroupKFold(n_splits=n_splits)
    y = tr[target].astype(float).values
    col = f"basin_te_{target[:3].lower()}"
    tr[col] = np.nan

    for tr_idx, va_idx in gkf.split(tr, y, groups=groups):
        mean_map = pd.Series(y[tr_idx], index=tr.index[tr_idx]).groupby(tr.loc[tr.index[tr_idx], "basin_id"]).mean()
        tr.loc[tr.index[va_idx], col] = tr.loc[tr.index[va_idx], "basin_id"].map(mean_map)
    # fill remaining with overall mean
    overall = float(np.nanmean(y))
    tr[col] = tr[col].fillna(overall)

    # full-map for valid
    full_map = pd.Series(y, index=tr.index).groupby(tr["basin_id"]).mean()
    va[col] = va["basin_id"].map(full_map).fillna(overall)
    return tr, va

def enrich_features(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    df = df.copy()

    if "NDMI" in df.columns and "pet" in df.columns:
        df["NDMI_x_pet"] = df["NDMI"] * df["pet"]
    if "MNDWI" in df.columns and "pet" in df.columns:
        df["MNDWI_x_pet"] = df["MNDWI"] * df["pet"]

    if "swir16" in df.columns and "swir22" in df.columns:
        df["swir_ratio_16_22"] = df["swir16"] / (df["swir22"] + 1e-6)
    if "nir" in df.columns and "swir22" in df.columns:
        df["nir_ratio_swir22"] = df["nir"] / (df["swir22"] + 1e-6)
    if "green" in df.columns and "swir16" in df.columns:
        df["green_ratio_swir16"] = df["green"] / (df["swir16"] + 1e-6)

    if "nir" in df.columns and "swir22" in df.columns:
        df["nir_minus_swir22"] = df["nir"] - df["swir22"]
    if "green" in df.columns and "swir16" in df.columns:
        df["green_minus_swir16"] = df["green"] - df["swir16"]

    for col in ["pet", "nir", "green", "swir16", "swir22"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(np.abs(df[col]))

    df["lat2"] = df[LAT_COL] ** 2
    df["lon2"] = df[LON_COL] ** 2
    df["lat_lon"] = df[LAT_COL] * df[LON_COL]

    # Rainfall-derived features
    if "chirps_ppt" in df.columns and "pet" in df.columns:
        df["runoff_index"] = df["chirps_ppt"] / (df["pet"] + 1e-6)
        # new wetness proxy (identical formula but semantically distinct)
        df["wetness_index"] = df["chirps_ppt"] / (df["pet"] + 1e-6)

    # Observation-based rolling rainfall features (approximations, not true daily windows)
    if "chirps_ppt" in df.columns:
        # Group by station and sort by date
        df = df.sort_values([LAT_COL, LON_COL, DATE_COL])
        # Rolling means over observations (not calendar days)
        rolling_windows = cfg.get("features", {}).get("rolling_windows", [30, 60, 90]) if cfg else [30, 60, 90]
        for w in rolling_windows:
            df[f"ppt_obs{w}_mean"] = df.groupby([LAT_COL, LON_COL])["chirps_ppt"].transform(lambda x: x.rolling(window=w, min_periods=1).mean())

    # DRP-focused features
    if cfg is not None:
        drp_focused = cfg.get("features", {}).get("drp_focused", {})
        if drp_focused.get("enabled", False):
            # A) Interaction terms between hydrology and moisture/water state
            if drp_focused.get("include_interactions", False):
                if "NDMI" in df.columns and "dist_to_river_m" in df.columns:
                    df["NDMI_x_dist_to_river_m"] = df["NDMI"] * df["dist_to_river_m"]
                if "MNDWI" in df.columns and "dist_to_river_m" in df.columns:
                    df["MNDWI_x_dist_to_river_m"] = df["MNDWI"] * df["dist_to_river_m"]
                if "NDMI" in df.columns and "upstream_area_km2" in df.columns:
                    df["NDMI_x_upstream_area_km2"] = df["NDMI"] * df["upstream_area_km2"]
                if "MNDWI" in df.columns and "upstream_area_km2" in df.columns:
                    df["MNDWI_x_upstream_area_km2"] = df["MNDWI"] * df["upstream_area_km2"]

                # B) Interaction terms between climate dryness and hydrology
                if "pet" in df.columns and "dist_to_river_m" in df.columns:
                    df["pet_x_dist_to_river_m"] = df["pet"] * df["dist_to_river_m"]
                if "pet" in df.columns and "basin_area_km2" in df.columns:
                    df["pet_x_basin_area_km2"] = df["pet"] * df["basin_area_km2"]
                if "pet" in df.columns and "upstream_area_km2" in df.columns:
                    df["pet_x_upstream_area_km2"] = df["pet"] * df["upstream_area_km2"]

            # C) Log-scaled hydrology variants
            if "dist_to_river_m" in df.columns:
                df["log_dist_to_river_m"] = np.log1p(df["dist_to_river_m"])
            if "basin_area_km2" in df.columns:
                df["log_basin_area_km2"] = np.log1p(df["basin_area_km2"])
            if "upstream_area_km2" in df.columns:
                df["log_upstream_area_km2"] = np.log1p(df["upstream_area_km2"])

            # D) Simple phosphorus-risk proxy features
            if drp_focused.get("include_proxies", False):
                if "MNDWI" in df.columns and "dist_to_river_m" in df.columns:
                    df["wetness_hydro_proxy"] = df["MNDWI"] / (np.log1p(df["dist_to_river_m"]) + 1)
                if "NDMI" in df.columns and "dist_to_river_m" in df.columns:
                    df["moisture_hydro_proxy"] = df["NDMI"] / (np.log1p(df["dist_to_river_m"]) + 1)

    # Spatial encoding features
    if cfg is not None and cfg.get("features", {}).get("add_spatial_encoding", False):
        import math
        df["sin_lat"] = np.sin(df[LAT_COL] * math.pi / 180)
        df["cos_lat"] = np.cos(df[LAT_COL] * math.pi / 180)
        df["sin_lon"] = np.sin(df[LON_COL] * math.pi / 180)
        df["cos_lon"] = np.cos(df[LON_COL] * math.pi / 180)
        
        # Calculate distance to centroid
        centroid_lat = df[LAT_COL].mean()
        centroid_lon = df[LON_COL].mean()
        df["dist_center"] = np.sqrt((df[LAT_COL] - centroid_lat)**2 + (df[LON_COL] - centroid_lon)**2)

    # Hydro interaction features
    if cfg is not None and cfg.get("features", {}).get("add_hydro_interactions", False):
        if "soil_clay_0_5" in df.columns and "slope" in df.columns:
            df["soil_clay_slope"] = df["soil_clay_0_5"] * df["slope"]
        if "soil_clay_0_5" in df.columns and "elevation" in df.columns:
            df["soil_clay_elevation"] = df["soil_clay_0_5"] * df["elevation"]
        if "slope" in df.columns and any(col.startswith("landcover_") for col in df.columns):
            # slope * landcover (using first landcover column as proxy)
            landcover_cols = [col for col in df.columns if col.startswith("landcover_")]
            if landcover_cols:
                df["slope_landcover"] = df["slope"] * df[landcover_cols[0]]
        if "soil_ph_0_5" in df.columns and any(col.startswith("landcover_") for col in df.columns):
            landcover_cols = [col for col in df.columns if col.startswith("landcover_")]
            if landcover_cols:
                df["soil_ph_landcover"] = df["soil_ph_0_5"] * df[landcover_cols[0]]

    # Upstream pressure features (basin aggregation)
    if cfg is not None and cfg.get("features", {}).get("upstream_pressure", False):
        if "basin_id" in df.columns:
            # Create landcover_urban if not exists (ESA WorldCover: 50 = urban)
            if "landcover_urban" not in df.columns and "landcover" in df.columns:
                df["landcover_urban"] = (df["landcover"] == 50).astype(int)
            
            # Calculate basin averages for soil and landcover features
            basin_avg_soil_clay = df.groupby("basin_id")["soil_clay_0_5"].transform("mean")
            basin_avg_soil_ph = df.groupby("basin_id")["soil_ph_0_5"].transform("mean")
            basin_avg_landcover_urban = df.groupby("basin_id")["landcover_urban"].transform("mean")
            
            df["basin_avg_soil_clay"] = basin_avg_soil_clay
            df["basin_avg_soil_ph"] = basin_avg_soil_ph
            df["basin_avg_landcover_urban"] = basin_avg_landcover_urban

    # Precip sum 15-45 days prior
    if cfg is not None and cfg.get("features", {}).get("precip_15_45_days", False):
        if "chirps_ppt" in df.columns:
            # Sort by station and date
            df = df.sort_values([LAT_COL, LON_COL, DATE_COL])
            # Rolling sum of precipitation from 15 to 45 days prior (approximately 30-60 observations)
            # This is a rough approximation since we don't have daily data
            df["precip_sum_15_to_45_days_prior"] = df.groupby([LAT_COL, LON_COL])["chirps_ppt"].transform(
                lambda x: x.shift(30).rolling(window=30, min_periods=1).sum()
            )

    # Short-window rainfall (surface flush)
    if cfg is not None and cfg.get("features", {}).get("precip_7_14_days", False):
        if "chirps_ppt" in df.columns:
            df = df.sort_values([LAT_COL, LON_COL, DATE_COL])
            df["precip_sum_7_to_14_days_prior"] = df.groupby([LAT_COL, LON_COL])["chirps_ppt"].transform(
                lambda x: x.shift(14).rolling(window=7, min_periods=1).sum()
            )

    # Precip sum 25-55 days prior (longer lag window)
    if cfg is not None and cfg.get("features", {}).get("precip_25_55_days", False):
        if "chirps_ppt" in df.columns:
            df = df.sort_values([LAT_COL, LON_COL, DATE_COL])
            # shift ~50 obs (~25 days) then roll 60 obs (~30 days) to approximate 25–55 day window
            df["precip_sum_25_to_55_days_prior"] = df.groupby([LAT_COL, LON_COL])["chirps_ppt"].transform(
                lambda x: x.shift(50).rolling(window=60, min_periods=1).sum()
            )

    # Non-linear hydrology features (flushing effect)
    if cfg is not None and cfg.get("features", {}).get("nonlinear_hydrology", False):
        if "precip_sum_15_to_45_days_prior" in df.columns:
            # Squared rainfall (captures acceleration of flushing)
            df["rain_lag_squared"] = df["precip_sum_15_to_45_days_prior"] ** 2
            
            # Dilution proxy (rain intensity relative to soil properties)
            if "basin_avg_soil_clay" in df.columns:
                df["rain_ratio"] = df["precip_sum_15_to_45_days_prior"] / (df["basin_avg_soil_clay"] + 1e-6)

            # new hydro-chem interaction: mobility of nutrients
            if "rain_ratio" in df.columns and "slope" in df.columns and "soil_ph_0_5" in df.columns:
                df["hydro_chem_mobility"] = df["rain_ratio"] * df["slope"] * df["soil_ph_0_5"]
            # clay saturation proxy
            if "rain_lag_squared" in df.columns and "soil_clay_0_5" in df.columns:
                df["clay_saturation"] = df["rain_lag_squared"] / (df["soil_clay_0_5"] + 1e-6)

            # dual-window rain interaction
            if "precip_sum_7_to_14_days_prior" in df.columns and "precip_sum_25_to_55_days_prior" in df.columns:
                df["rain_interaction"] = (
                    df["precip_sum_7_to_14_days_prior"] * df["precip_sum_25_to_55_days_prior"]
                )

    # Mineral loading physics features
    if "basin_id" in df.columns and "soil_ph_0_5" in df.columns:
        basin_avg_soil_ph = df.groupby("basin_id")["soil_ph_0_5"].transform("mean")
        df["basin_avg_soil_ph"] = basin_avg_soil_ph

    if "basin_avg_soil_ph" in df.columns and "elevation" in df.columns and "pet" in df.columns:
        df["mineral_concentration_index"] = (df["basin_avg_soil_ph"] * df["elevation"]) / (df["pet"] + 1e-6)

    if "upstream_area_km2" in df.columns and "slope" in df.columns:
        df["contact_time_proxy"] = df["upstream_area_km2"] / (df["slope"] + 1e-6)

    return df

def add_valid_lag1_and_time(df: pd.DataFrame, lag_cols=("pet", "NDMI", "MNDWI")) -> pd.DataFrame:
    df = df.copy()
    df["_orig_order"] = np.arange(len(df))
    df = df.sort_values([LAT_COL, LON_COL, DATE_COL])

    prev_date = df.groupby([LAT_COL, LON_COL])[DATE_COL].shift(1)
    month_diff = (
        (df[DATE_COL].dt.year - prev_date.dt.year) * 12
        + (df[DATE_COL].dt.month - prev_date.dt.month)
    )
    has_lag1 = (month_diff == 1)

    for c in lag_cols:
        if c in df.columns:
            lag = df.groupby([LAT_COL, LON_COL])[c].shift(1)
            df[f"{c}_lag1"] = lag
            df.loc[~has_lag1, f"{c}_lag1"] = np.nan

    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["dayofyear"] = df[DATE_COL].dt.dayofyear

    df = df.sort_values("_orig_order").drop(columns=["_orig_order"])
    return df

def add_spatial_basis_v4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lat = df[LAT_COL].astype(float)
    lon = df[LON_COL].astype(float)

    df["lat3"] = lat ** 3
    df["lon3"] = lon ** 3

    rad = np.pi / 180.0
    df["sin_lat"] = np.sin(lat * rad)
    df["cos_lat"] = np.cos(lat * rad)
    df["sin_lon"] = np.sin(lon * rad)
    df["cos_lon"] = np.cos(lon * rad)

    lat0 = float(lat.median())
    lon0 = float(lon.median())
    df["dist_center"] = np.sqrt((lat - lat0) ** 2 + (lon - lon0) ** 2)
    return df

# ----------------------------
# Station-aware features (fold-safe, no leakage)
# ----------------------------
def add_station_features(df: pd.DataFrame, train_df: pd.DataFrame = None, 
                        enabled: bool = True) -> pd.DataFrame:
    """
    Add station-based features using training set statistics.
    Always fold-safe: station stats computed ONLY from training indices.
    
    Args:
        df: DataFrame to add features to (validation set in CV, validation set in final)
        train_df: DataFrame to compute statistics from (fold's train in CV, full train in final)
        enabled: If False, return df unchanged
    
    Returns:
        df with numeric feature station_obs_count (station_id is internal helper only)
    """
    if not enabled or train_df is None:
        return df
    
    df = df.copy()
    
    # Station ID: rounded coordinates (internal helper only, NOT a feature)
    station_id = (
        np.floor(df[LAT_COL] * 1e4).astype(int).astype(str) + "_" +
        np.floor(df[LON_COL] * 1e4).astype(int).astype(str)
    )
    
    # Station observation count (from training set only - no leakage)
    station_counts = train_df.groupby([LAT_COL, LON_COL]).size()
    station_count_map = {}
    for (lat, lon), count in station_counts.items():
        key = (float(lat), float(lon))
        station_count_map[key] = int(count)
    
    def get_station_count(row):
        key = (float(row[LAT_COL]), float(row[LON_COL]))
        return station_count_map.get(key, 0)
    
    df["station_obs_count"] = df.apply(get_station_count, axis=1)
    
    return df

# ----------------------------
# Missingness flags (idempotente)
# ----------------------------
def add_missing_flags(train_df, valid_df):
    cols = ["NDMI","MNDWI","nir","green","swir16","swir22","pet","chirps_ppt","runoff_index","NDMI_lag1","MNDWI_lag1","pet_lag1"]
    for c in cols:
        name = f"isna_{c}"
        if c in train_df.columns and name not in train_df.columns:
            train_df[name] = train_df[c].isna().astype(int)
        if c in valid_df.columns and name not in valid_df.columns:
            valid_df[name] = valid_df[c].isna().astype(int)
    return train_df, valid_df

# ----------------------------
# OOF Target Encoding (TRAIN) + Full-map (VALID)
# ----------------------------
def add_oof_te(train_df: pd.DataFrame,
               valid_df: pd.DataFrame,
               target: str,
               te_grids=(0.05, 0.2),
               oof_group_grid=0.2,
               n_splits=5,
               prefix="te"):

    tr = train_df.copy()
    va = valid_df.copy()

    groups = make_groups(tr, oof_group_grid).values
    gkf = GroupKFold(n_splits=n_splits)

    y = tr[target].astype(float).values
    global_mean = float(np.nanmean(y))

    for g in te_grids:
        cell_col = f"__cell_{str(g).replace('.','p')}"
        tr[cell_col] = (
            np.floor(tr[LAT_COL]/g).astype(int).astype(str) + "_" +
            np.floor(tr[LON_COL]/g).astype(int).astype(str)
        )
        va[cell_col] = (
            np.floor(va[LAT_COL]/g).astype(int).astype(str) + "_" +
            np.floor(va[LON_COL]/g).astype(int).astype(str)
        )

        out_col = f"{prefix}_{target[:3].lower()}_g{str(g).replace('.','p')}"
        tr[out_col] = np.nan

        # OOF fill for TRAIN
        for tr_idx, va_idx in gkf.split(tr, y, groups=groups):
            y_tr = y[tr_idx]
            mean_tr = float(np.nanmean(y_tr))
            m = pd.Series(y_tr, index=tr.index[tr_idx]).groupby(tr.loc[tr.index[tr_idx], cell_col]).mean()
            tr.loc[tr.index[va_idx], out_col] = (
                tr.loc[tr.index[va_idx], cell_col]
                .map(m)
                .fillna(mean_tr)
                .astype(float)
            )

        # FULL-MAP for VALID
        full_map = pd.Series(y, index=tr.index).groupby(tr[cell_col]).mean()
        va[out_col] = va[cell_col].map(full_map).fillna(global_mean).astype(float)

    tmp_cols = [c for c in tr.columns if c.startswith("__cell_")]
    tr = tr.drop(columns=tmp_cols, errors="ignore")
    va = va.drop(columns=tmp_cols, errors="ignore")

    return tr, va

def add_knn_target_encoding(train_df: pd.DataFrame,
                           valid_df: pd.DataFrame,
                           target: str,
                           n_neighbors: int = 5,
                           basin_locked: bool = False) -> tuple:
    """
    KNN target encoding: average target value of k nearest neighbors.
    Uses spatial distance (lat/lon) for neighbor finding.
    If basin_locked=True, only considers neighbors within the same basin_id.
    Fallback to basin_avg if basin has <3 stations.
    """
    tr = train_df.copy()
    va = valid_df.copy()
    
    coords_cols = [LAT_COL, LON_COL]
    encoding_col = f"knn_{target[:3].lower()}_encoding"
    
    if basin_locked and "basin_id" in tr.columns:
        # Basin-locked KNN: process each basin separately
        tr[encoding_col] = np.nan
        va[encoding_col] = np.nan
        
        for basin in tr["basin_id"].unique():
            if pd.isna(basin):
                continue
            
            # Filter training data for this basin
            basin_mask_tr = tr["basin_id"] == basin
            basin_data_tr = tr.loc[basin_mask_tr]
            
            # Check if basin has enough stations
            if len(basin_data_tr) < 3:
                # Fallback to basin average
                basin_avg = basin_data_tr[target].mean()
                tr.loc[basin_mask_tr, encoding_col] = basin_avg
                
                # For validation, use basin average too
                basin_mask_va = va["basin_id"] == basin
                va.loc[basin_mask_va, encoding_col] = basin_avg
            else:
                # Fit KNN on basin data
                knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(basin_data_tr)), metric='euclidean')
                knn.fit(basin_data_tr[coords_cols], basin_data_tr[target])
                
                # Encode training data within basin
                tr.loc[basin_mask_tr, encoding_col] = knn.predict(basin_data_tr[coords_cols])
                
                # Encode validation data within basin
                basin_mask_va = va["basin_id"] == basin
                if basin_mask_va.sum() > 0:
                    va.loc[basin_mask_va, encoding_col] = knn.predict(va.loc[basin_mask_va, coords_cols])
    else:
        # Global KNN (original behavior)
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(tr[coords_cols], tr[target])
        
        tr[encoding_col] = knn.predict(tr[coords_cols])
        va[encoding_col] = knn.predict(va[coords_cols])
    
    return tr, va

# ----------------------------
# Pipeline builder
# ----------------------------
def build_pipeline(model, feature_cols, cfg=None, target=None):
    # Standard scaling after median imputation helps with stacking and linear models
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", __import__("sklearn").preprocessing.StandardScaler()),
                ]),
                feature_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("model", clone(model))])

def numeric_feature_cols(df, targets, cfg=None):
    # Exclude all known water quality targets, not just the ones being trained
    known_targets = {"Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"}
    exclude = known_targets | {DATE_COL, LAT_COL, LON_COL}  # never feed raw coords to tree models
    feat_cols = [c for c in df.columns if (c not in exclude) and pd.api.types.is_numeric_dtype(df[c])]

    # drop lat/lon if requested (physical-only test)
    if cfg is not None and cfg.get("features", {}).get("drop_coords", False):
        feat_cols = [c for c in feat_cols if c not in [LAT_COL, LON_COL]]

    # ensure priority features remain
    if cfg is not None:
        priority = cfg.get("features", {}).get("priority_features", [])
        for p in priority:
            if p in df.columns and p not in feat_cols:
                feat_cols.append(p)
    
    # Filter features based on config flags
    if cfg is not None:
        features_cfg = cfg.get("features", {})
        
        # Define feature groups
        landsat_features = [
            'NDMI', 'MNDWI', 'nir', 'green', 'swir16', 'swir22',
            'NDMI_x_pet', 'MNDWI_x_pet', 'swir_ratio_16_22', 'nir_ratio_swir22', 'green_ratio_swir16',
            'nir_minus_swir22', 'green_minus_swir16', 'log_pet', 'log_nir', 'log_green', 'log_swir16', 'log_swir22'
        ]
        terraclimate_features = ['pet', 'log_pet']  # pet and its log
        chirps_features = [
            'chirps_ppt', 'runoff_index', 'wetness_index',
            'ppt_obs7_mean', 'ppt_obs14_mean', 'ppt_obs30_mean', 'ppt_obs60_mean', 'ppt_obs90_mean'
        ]
        external_features = [
            'dist_to_river_m', 'basin_area_km2', 'upstream_area_km2', 'slope', 'elevation',
            'log_dist_to_river_m', 'log_basin_area_km2', 'log_upstream_area_km2',
            'NDMI_x_dist_to_river_m', 'MNDWI_x_dist_to_river_m', 'NDMI_x_upstream_area_km2', 'MNDWI_x_upstream_area_km2',
            'pet_x_dist_to_river_m', 'pet_x_basin_area_km2', 'pet_x_upstream_area_km2', 'wetness_hydro_proxy'
        ] + [col for col in df.columns if col.startswith('landcover_')]  # landcover one-hot features
        
        # Filter based on flags
        if not features_cfg.get("use_landsat", True):
            feat_cols = [c for c in feat_cols if c not in landsat_features]
        if not features_cfg.get("use_terraclimate", True):
            feat_cols = [c for c in feat_cols if c not in terraclimate_features]
        if not features_cfg.get("use_chirps", True):
            feat_cols = [c for c in feat_cols if c not in chirps_features]
        if not features_cfg.get("use_external_geofeatures", True):
            feat_cols = [c for c in feat_cols if c not in external_features]
    
    # Verify CHIRPS features are present (if enabled)
    chirps_features = ['chirps_ppt', 'runoff_index', 'ppt_obs30_mean', 'ppt_obs60_mean', 'ppt_obs90_mean',
                        'precip_sum_15_to_45_days_prior', 'precip_sum_25_to_55_days_prior', 'precip_sum_7_to_14_days_prior', 'rain_interaction']
    found_chirps = [f for f in chirps_features if f in feat_cols]
    if found_chirps:
        print(f"    [FEATURES] CHIRPS rainfall features included: {found_chirps}")
    
    return feat_cols

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dev", action="store_true", help="force dev_mode true")
    # Experiment parameters
    ap.add_argument("--experiment_name", default="baseline", help="experiment identifier")
    ap.add_argument("--drp_model", default=None, help="DRP model override: ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor")
    ap.add_argument("--regularization", default="default", help="regularization level: default, stronger")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    cfg = load_yaml(args.config)
    
    # Apply regularization adjustments if needed (use model if override specified, else default ET)
    if args.regularization == "stronger":
        model_for_reg = args.drp_model if args.drp_model else "ExtraTreesRegressor"
        cfg = apply_regularization_to_cfg(cfg, args.regularization, model_for_reg)
    
    # Validate experiment parameters
    valid_models = [None, "ExtraTreesRegressor", "RandomForestRegressor", "HistGradientBoostingRegressor"]
    if args.drp_model not in valid_models:
        raise ValueError(f"Invalid drp_model: {args.drp_model}")
    
    valid_regs = ["default", "stronger"]
    if args.regularization not in valid_regs:
        raise ValueError(f"Invalid regularization: {args.regularization}")

    raw_dir = cfg["project"]["raw_dir"]
    external_path = cfg["project"]["external_path"]
    cache_dir = cfg["project"]["cache_dir"]
    ensure_dir(cache_dir)

    out_dir = cfg["project"]["out_dir"]
    ensure_dir(out_dir)

    # Start experiment run (auto-logging)
    run_id, run_path = start_experiment_run(cfg, cache_dir)
    print(f"EXP RUN: {run_id} -> {run_path}")

    switches = cfg["switches"]
    if args.dev:
        switches["dev_mode"] = True

    targets = cfg["targets"]["list"]

    # y_mode global + override por target
    y_mode_default = cfg["targets"]["y_mode"]
    y_mode_by_target = cfg.get("targets", {}).get("y_mode_by_target", {})

    cv_folds = cfg["cv"]["folds_dev"] if switches["dev_mode"] else cfg["cv"]["folds"]
    best_grid = cfg["cv"]["best_grid"]
    grouping_strategy = cfg["cv"].get("grouping_strategy", "basin")

    te_enabled = cfg["te"]["enabled"]
    te_grids = tuple(cfg["te"]["grids"])
    te_oof_grid = float(cfg["te"]["oof_grid_for_groups"])

    # -------- template RAW (intacto) --------
    template_path = os.path.join(raw_dir, cfg["io"]["template_name"])
    template_raw = pd.read_csv(template_path)  # NO normalizar
    assert template_raw.shape[0] == 200, "Template no esperado"
    template_norm = normalize_keys_for_matching(template_raw)

    # -------- cache paths --------
    cache_train = os.path.join(cache_dir, "train_modelready.pkl")
    cache_valid = os.path.join(cache_dir, "valid_modelready.pkl")
    cache_feat  = os.path.join(cache_dir, "feature_cols.json")
    basin_pack_cols = []
    temporal_pack_cols = []
    upstream_pack_cols = []

    # -------- build dataset --------
    if (not switches["build_dataset"]) and os.path.exists(cache_train) and os.path.exists(cache_valid) and os.path.exists(cache_feat):
        train_df = pd.read_pickle(cache_train)
        valid_df = pd.read_pickle(cache_valid)
        with open(cache_feat, "r", encoding="utf-8") as f:
            _base_feature_cols = json.load(f)
        print("Using CACHE dataset.")
    else:
        # load raw
        wq = pd.read_csv(os.path.join(raw_dir, cfg["io"]["train_wq_name"]))
        ls_tr = pd.read_csv(os.path.join(raw_dir, cfg["io"]["train_ls_name"]))
        tc_tr = pd.read_csv(os.path.join(raw_dir, cfg["io"]["train_tc_name"]))
        ls_va = pd.read_csv(os.path.join(raw_dir, cfg["io"]["valid_ls_name"]))
        tc_va = pd.read_csv(os.path.join(raw_dir, cfg["io"]["valid_tc_name"]))

        # CHIRPS rainfall data
        chirps_tr = pd.read_csv(cfg["project"]["chirps_train_path"])
        chirps_va = pd.read_csv(cfg["project"]["chirps_valid_path"])

        # normalize for merges
        wq = normalize_keys_for_matching(wq)
        ls_tr = normalize_keys_for_matching(ls_tr)
        tc_tr = normalize_keys_for_matching(tc_tr)
        ls_va = normalize_keys_for_matching(ls_va)
        tc_va = normalize_keys_for_matching(tc_va)
        chirps_tr = normalize_keys_for_matching(chirps_tr)
        chirps_va = normalize_keys_for_matching(chirps_va)

        # merge train
        wq_date = wq.dropna(subset=[DATE_COL]).copy()
        train_df = (
            wq_date
            .merge(ls_tr, on=KEYS, how="inner", suffixes=("", "_ls"))
            .merge(tc_tr, on=KEYS, how="inner", suffixes=("", "_tc"))
            .merge(chirps_tr, on=KEYS, how="left", suffixes=("", "_chirps"))
        )
        train_df = train_df.loc[:, ~train_df.columns.duplicated()].copy()

        # merge valid
        valid_df = (
            ls_va
            .merge(tc_va, on=KEYS, how="inner")
            .merge(chirps_va, on=KEYS, how="left")
        )
        # align to template ORDER (by normalized keys)
        if not template_norm[KEYS].equals(valid_df[KEYS]):
            valid_df = template_norm[KEYS].merge(valid_df, on=KEYS, how="left")

        # externals
        ext = pd.read_csv(external_path)
        ext[LAT_COL] = pd.to_numeric(ext[LAT_COL], errors="coerce")
        ext[LON_COL] = pd.to_numeric(ext[LON_COL], errors="coerce")

        train_df = train_df.merge(ext, on=[LAT_COL, LON_COL], how="left")
        valid_df = valid_df.merge(ext, on=[LAT_COL, LON_COL], how="left")

        # --- LANDCOVER ONE-HOT TOP5 ---
        train_df, valid_df = add_landcover_ohe(train_df, valid_df, top_n=5)

        # --- DISTANCE TO BASIN CENTROID ---
        if "basin_id" in train_df.columns:
            centroid_map = (
                train_df.groupby("basin_id")[ [LAT_COL, LON_COL] ]
                .mean()
                .to_dict(orient="index")
            )
            # convert to mapping of id->(lat,lon)
            centroid_map = {k: (v[LAT_COL], v[LON_COL]) for k, v in centroid_map.items()}
            train_df = add_dist_to_centroid(train_df, centroid_map)
            valid_df = add_dist_to_centroid(valid_df, centroid_map)
        else:
            train_df["dist_to_centroid"] = np.nan
            valid_df["dist_to_centroid"] = np.nan

        # sanity locks (norm keys)
        assert train_df.duplicated(subset=KEYS).sum() == 0, "Duplicated keys in train"
        assert len(valid_df) == 200, "Valid must be 200"
        assert template_norm[KEYS].equals(valid_df[KEYS]), "Valid not aligned to template keys (normalized)"

        # FE V4
        train_df, valid_df, basin_pack_cols = add_basin_context_pack(train_df, valid_df, cfg)
        train_df = enrich_features(train_df, cfg)
        valid_df = enrich_features(valid_df, cfg)

        train_df = add_valid_lag1_and_time(train_df)
        valid_df = add_valid_lag1_and_time(valid_df)

        train_df, temporal_pack_cols = add_temporal_context_pack(train_df, cfg)
        valid_df, _ = add_temporal_context_pack(valid_df, cfg)

        train_df, upstream_pack_cols = add_upstream_context_pack(train_df, cfg)
        valid_df, _ = add_upstream_context_pack(valid_df, cfg)

        train_df = add_spatial_basis_v4(train_df)
        valid_df = add_spatial_basis_v4(valid_df)

        # Ensure valid_df order matches template after all processing
        valid_df = template_norm[KEYS].merge(valid_df, on=KEYS, how="left")

        # post-FE alignment lock
        assert template_norm[KEYS].equals(valid_df[KEYS]), "POST-FE valid order scrambled"

        # missing flags
        train_df, valid_df = add_missing_flags(train_df, valid_df)

        # cache save
        pd.to_pickle(train_df, cache_train)
        pd.to_pickle(valid_df, cache_valid)
        with open(cache_feat, "w", encoding="utf-8") as f:
            json.dump(numeric_feature_cols(train_df, targets, cfg), f, ensure_ascii=False, indent=2)

        print("Cache saved.")

    # Write minimal dataset/meta info into run folder
    meta = {
        "run_id": run_id,
        "cache_train": cache_train,
        "cache_valid": cache_valid,
        "cache_feat": cache_feat,
        "out_dir": out_dir,
        "raw_dir": raw_dir,
        "te_enabled": bool(te_enabled),
        "te_grids": list(te_grids),
        "cv_folds": int(cv_folds),
        "dev_mode": bool(switches["dev_mode"]),
        "experiment_name": str(args.experiment_name),
        "drp_model": str(args.drp_model) if args.drp_model else "default",
        "regularization": str(args.regularization),
        "basin_context_enabled": bool(cfg.get("features", {}).get("basin_context", {}).get("enabled", False)),
        "basin_context_features_added": basin_pack_cols,
        "temporal_context_enabled": bool(cfg.get("features", {}).get("temporal_context", {}).get("enabled", False)),
        "temporal_context_features_added": temporal_pack_cols,
        "upstream_context_enabled": bool(cfg.get("features", {}).get("upstream_context", {}).get("enabled", False)),
        "upstream_context_features_added": upstream_pack_cols,
    }
    safe_write_json(os.path.join(run_path, "run_meta.json"), meta)

    # ----------------------------
    # CV (fold-safe TE dentro del fold)
    # ----------------------------
    et_cv_params = cfg["model"]["et_cv"]
    model_cv_default = ExtraTreesRegressor(**et_cv_params)
    def cv_with_fold_te(df, target):
        grid = float(best_grid[target])
        y = df[target].astype(float).values
        groups = make_groups(df, grid, strategy=grouping_strategy).values
        gkf = GroupKFold(n_splits=cv_folds)

        scores = []

        fold_iter = gkf.split(df, y, groups=groups)
        fold_iter = tqdm(
            fold_iter,
            total=cv_folds,
            desc=f"CV folds | {target}",
            leave=False
        )

        for fold_i, (tr_idx, va_idx) in enumerate(fold_iter, start=1):
            X_tr = df.iloc[tr_idx].copy()
            X_va = df.iloc[va_idx].copy()
            y_tr = y[tr_idx]
            y_va = y[va_idx]
            
            # Verify grouping integrity: if basin_id exists, check for leakage
            if 'basin_id' in df.columns and grouping_strategy == "basin":
                tr_basins = set(X_tr['basin_id'].dropna().unique())
                va_basins = set(X_va['basin_id'].dropna().unique())
                overlap = tr_basins & va_basins
                if overlap:
                    print(f"  [FOLD {fold_i}] WARNING: Basin overlap detected! Basins {overlap} appear in both train and valid.")
                else:
                    pass  # Clean separation - no output to avoid noise

            

            # If using Poisson loss, enforce non-negative targets
            _m = cfg.get("model", {}).get("cv_by_target", {}).get(target)
            if _m and _m.get("name") == "HistGradientBoostingRegressor" and str(_m.get("params", {}).get("loss", "")) == "poisson":
                y_tr = np.maximum(y_tr, 0.0)
                y_va = np.maximum(y_va, 0.0)
            mode_t = y_mode_by_target.get(target, y_mode_default)
            fwd, inv = y_transform_fit(y_tr, mode_t)
            y_tr_t = fwd(y_tr)

            # TE fold-safe (con y_tr ORIGINAL, no winsor)
            if te_enabled:
                X_tr, X_va = add_oof_te(
                    X_tr, X_va,
                    target=target,
                    te_grids=te_grids,
                    oof_group_grid=te_oof_grid,
                    n_splits=3,  # interno rápido dentro del fold
                    prefix="tefold"
                )

            # basin-level target encoding
            if target == "Dissolved Reactive Phosphorus":
                X_tr, X_va = add_basin_target_mean(X_tr, X_va, target=target, n_splits=3)

            # Station-aware features (fold-safe: stats only from X_tr)
            station_aware_enabled = cfg.get("features", {}).get("station_aware", {}).get("enabled", False)
            X_tr = add_station_features(X_tr, train_df=X_tr, enabled=station_aware_enabled)
            X_va = add_station_features(X_va, train_df=X_tr, enabled=station_aware_enabled)

            # KNN target encoding (fold-safe)
            knn_encoding_enabled = cfg.get("features", {}).get("knn_encoding", False)
            if knn_encoding_enabled:
                basin_locked = cfg.get("features", {}).get("basin_locked_knn", False)
                knn_n = int(cfg.get("features", {}).get("knn_n_neighbors", 5))
                X_tr, X_va = add_knn_target_encoding(X_tr, X_va, target=target, n_neighbors=knn_n, basin_locked=basin_locked)

            feat = numeric_feature_cols(X_tr, targets, cfg)

            # DRP feature pruning: reduce noise (only for Dissolved Reactive Phosphorus)
            if target == "Dissolved Reactive Phosphorus":
                drp_prune_enabled = cfg.get("features", {}).get("drp_prune", {}).get("enabled", False)
                if drp_prune_enabled:
                    max_missing = float(cfg.get("features", {}).get("drp_prune", {}).get("max_missing_pct", 0.40))
                    min_var = float(cfg.get("features", {}).get("drp_prune", {}).get("min_variance", 1e-6))
                    
                    pruned_feat = []
                    for col in feat:
                        missing_pct = X_tr[col].isna().mean()
                        variance = X_tr[col].var() or 0.0
                        if missing_pct <= max_missing and variance > min_var:
                            pruned_feat.append(col)
                    
                    n_before = len(feat)
                    feat = pruned_feat
                    print(f"  DRP CV pruning: {n_before} -> {len(feat)} features")

            # LOCK CV: mismas columnas train vs val en el fold
            missing_in_va = [c for c in feat if c not in X_va.columns]
            assert len(missing_in_va) == 0, (
                f"ERROR CV: fold-valid missing {len(missing_in_va)} cols for target={target}. "
                f"Ej: {missing_in_va[:10]}"
            )

            X_tr_X = X_tr[feat].copy()
            X_va_X = X_va[feat].copy()
            assert list(X_tr_X.columns) == list(X_va_X.columns), "ERROR CV: column order mismatch"

            model_cv = build_model_from_cfg(cfg, "cv", target, exp_model_override=args.drp_model)
            pipe = build_pipeline(model_cv, feat, cfg, t)
            pipe.fit(X_tr_X, y_tr_t)
            pred = inv(pipe.predict(X_va_X))

            if target == "Dissolved Reactive Phosphorus":
                pred = np.maximum(pred, 0.0)

            sc = r2_score(y_va, pred)
            scores.append(sc)

            fold_iter.set_postfix(fold=fold_i, r2=f"{sc:.4f}")

        return float(np.mean(scores)), float(np.std(scores)), scores

    cv_report = {}
    if switches["run_cv"]:
        print("\n" + "="*80)
        print(f"CV REPORT | folds={cv_folds} | dev_mode={switches['dev_mode']} | TE={te_enabled} (fold-safe)")

        for t in tqdm(targets, desc="CV targets", unit="target"):
            m, s, folds = cv_with_fold_te(train_df, t)
            cv_report[t] = {"mean": m, "std": s, "folds": [float(x) for x in folds]}
            print(f"{t:>28} | mean={m:.4f} +/- {s:.4f} | folds={np.round(folds,4)}")

        safe_write_json(os.path.join(run_path, "cv_report.json"), cv_report)

    # ----------------------------
    # FINAL TRAIN + SUBMISSION (OOF TE TRAIN + full-map VALID)
    # ----------------------------
    if switches["train_final"]:
        et_final_params = cfg["model"]["et_final"]
        model_final_default = ExtraTreesRegressor(**et_final_params)
# start from a CLEAN submission frame
        submission_out = template_raw.copy(deep=True)

        # train/valid working copies (normalized for modeling)
        tr0 = train_df.copy()
        va0 = valid_df.copy()

        for t in tqdm(targets, desc="FINAL train (per target)", unit="target"):
            tr = tr0.copy()
            va = va0.copy()

            if te_enabled:
                tr, va = add_oof_te(
                    tr, va,
                    target=t,
                    te_grids=te_grids,
                    oof_group_grid=te_oof_grid,
                    n_splits=cv_folds,     # OOF serio
                    prefix="te"
                )

            # basin-level target encoding in final stage
            if t == "Dissolved Reactive Phosphorus":
                tr, va = add_basin_target_mean(tr, va, target=t, n_splits=cv_folds)

            # Station-aware features (final: stats from full training set)
            station_aware_enabled = cfg.get("features", {}).get("station_aware", {}).get("enabled", False)
            tr = add_station_features(tr, train_df=tr, enabled=station_aware_enabled)
            va = add_station_features(va, train_df=tr, enabled=station_aware_enabled)

            # KNN target encoding (final stage)
            if cfg.get("features", {}).get("knn_encoding", False):
                basin_locked = cfg.get("features", {}).get("basin_locked_knn", False)
                knn_n = int(cfg.get("features", {}).get("knn_n_neighbors", 5))
                tr, va = add_knn_target_encoding(tr, va, target=t, n_neighbors=knn_n, basin_locked=basin_locked)

            y = tr[t].astype(float).values
            

            # If using Poisson loss, enforce non-negative targets
            _mf = cfg.get("model", {}).get("final_by_target", {}).get(t)
            if _mf and _mf.get("name") == "HistGradientBoostingRegressor" and str(_mf.get("params", {}).get("loss", "")) == "poisson":
                y = np.maximum(y, 0.0)
            mode_t = y_mode_by_target.get(t, y_mode_default)
            fwd, inv = y_transform_fit(y, mode_t)
            y_t = fwd(y)

            feat_cols = numeric_feature_cols(tr, targets, cfg)

            # DRP feature pruning: reduce noise (only for Dissolved Reactive Phosphorus)
            if t == "Dissolved Reactive Phosphorus":
                drp_prune_enabled = cfg.get("features", {}).get("drp_prune", {}).get("enabled", False)
                if drp_prune_enabled:
                    max_missing = float(cfg.get("features", {}).get("drp_prune", {}).get("max_missing_pct", 0.40))
                    min_var = float(cfg.get("features", {}).get("drp_prune", {}).get("min_variance", 1e-6))
                    
                    pruned_feat_cols = []
                    for col in feat_cols:
                        missing_pct = tr[col].isna().mean()
                        variance = tr[col].var() or 0.0
                        if missing_pct <= max_missing and variance > min_var:
                            pruned_feat_cols.append(col)
                    
                    n_before = len(feat_cols)
                    feat_cols = pruned_feat_cols
                    print(f"Features locked for {t}: {len(feat_cols)} cols (pruned from {n_before})")
            else:
                print(f"Features locked for {t}: {len(feat_cols)} cols")

            manifest_name = f"feature_manifest_{t.replace(' ','_')}.json"
            manifest_path_cache = os.path.join(cache_dir, manifest_name)
            with open(manifest_path_cache, "w", encoding="utf-8") as f:
                json.dump(feat_cols, f, ensure_ascii=False, indent=2)

            shutil.copy2(manifest_path_cache, os.path.join(run_path, manifest_name))

            missing_in_valid = [c for c in feat_cols if c not in va.columns]
            assert len(missing_in_valid) == 0, (
                f"ERROR: valid is missing {len(missing_in_valid)} features for target={t}. "
                f"Ej: {missing_in_valid[:10]}"
            )

            tr_X = tr[feat_cols].copy()
            va_X = va[feat_cols].copy()
            assert list(tr_X.columns) == list(va_X.columns), "ERROR: column order mismatch train vs valid"

            print(f"Features locked for {t}: {len(feat_cols)} cols")

            model_final = build_model_from_cfg(cfg, "final", t, exp_model_override=args.drp_model)
            pipe = build_pipeline(model_final, feat_cols, cfg, t)
            pipe.fit(tr_X, y_t)

            # Save feature importances if available
            if hasattr(pipe.named_steps['model'], 'feature_importances_'):
                importances = pipe.named_steps['model'].feature_importances_
                feat_importance = dict(zip(feat_cols, importances))
                with open(os.path.join(run_path, f"feature_importance_{t.replace(' ','_')}.json"), "w", encoding="utf-8") as f:
                    json.dump(feat_importance, f, indent=2, ensure_ascii=False)

            pred = inv(pipe.predict(va_X))

            if t == "Dissolved Reactive Phosphorus":
                pred = np.maximum(pred, 0.0)

            submission_out[t] = pred

        # Fill missing targets from reference submission
        all_targets = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
        missing_targets = [t for t in all_targets if t not in targets]
        if missing_targets:
            ref_path = os.path.join(project_root, "submissions", "submission_V4_4_DRP_tuned_ET_fixorder.csv")
            ref_df = pd.read_csv(ref_path)
            for t in missing_targets:
                submission_out[t] = ref_df[t]

        # Submission is written directly into the run directory — immutable, never overwritten.
        out_name = "submission.csv"
        out_path = os.path.join(run_path, out_name)
        submission_out.to_csv(out_path, index=False)

        chk = pd.read_csv(out_path)
        assert chk.shape == (200, 6)
        assert not chk[targets].isna().any().any()
        assert chk[KEYS].equals(template_raw[KEYS]), "KEYS no idénticas al template crudo"
        assert list(chk.columns) == list(template_raw.columns), "ERROR: submission columns != template columns"

        print("\nSaved:", out_path)
        print("DRP min/max:", float(chk["Dissolved Reactive Phosphorus"].min()), float(chk["Dissolved Reactive Phosphorus"].max()))
        safe_write_json(os.path.join(run_path, "artifacts.json"), {"main_submission": out_path})

        # Write metadata.json — single-file summary of config + features + CV + submission path
        _drp_cfg_block = cfg.get("model", {}).get("cv_by_target", {}).get("Dissolved Reactive Phosphorus", {})
        _ymode_map = cfg.get("targets", {}).get("y_mode_by_target", {})
        metadata = {
            "experiment_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "model_config": {
                "drp_model": _drp_cfg_block.get("name", "ExtraTreesRegressor"),
                "drp_params": _drp_cfg_block.get("params", cfg.get("model", {}).get("et_final", {})),
                "drp_y_mode": _ymode_map.get(
                    "Dissolved Reactive Phosphorus",
                    cfg.get("targets", {}).get("y_mode", "none"),
                ),
                "te_enabled": bool(te_enabled),
                "te_grids": list(te_grids),
            },
            "feature_counts": {
                t: _load_manifest_count(
                    os.path.join(run_path, f"feature_manifest_{t.replace(' ', '_')}.json")
                )
                for t in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
            },
            "cv_metrics": {
                t: {"mean": round(v["mean"], 6), "std": round(v["std"], 6)}
                for t, v in cv_report.items()
            } if cv_report else {},
            "submission_path": out_path,
            "feature_pack": {
                "basin_context_enabled": bool(cfg.get("features", {}).get("basin_context", {}).get("enabled", False)),
                "basin_context_features_added": basin_pack_cols,
                "temporal_context_enabled": bool(cfg.get("features", {}).get("temporal_context", {}).get("enabled", False)),
                "temporal_context_features_added": temporal_pack_cols,
                "upstream_context_enabled": bool(cfg.get("features", {}).get("upstream_context", {}).get("enabled", False)),
                "upstream_context_features_added": upstream_pack_cols,
            },
        }
        safe_write_json(os.path.join(run_path, "metadata.json"), metadata)
        print(f"metadata.json -> {os.path.join(run_path, 'metadata.json')}")

        # Append to global experiment index
        _update_experiment_index(run_path, run_id, metadata)

if __name__ == "__main__":
    main()
