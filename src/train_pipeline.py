#!/usr/bin/env python
# coding: utf-8

import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor

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

def normalize_keys_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    """Solo para matching interno (NO para template crudo)."""
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    return out

def make_groups(df, grid):
    return (
        np.floor(df[LAT_COL] / grid).astype(int).astype(str)
        + "_"
        + np.floor(df[LON_COL] / grid).astype(int).astype(str)
    )

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
    raise ValueError(f"Unknown y_mode: {mode}")

# ----------------------------
# Feature Engineering (V4)
# ----------------------------
def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
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
# Missingness flags (idempotente)
# ----------------------------
def add_missing_flags(train_df, valid_df):
    cols = ["NDMI","MNDWI","nir","green","swir16","swir22","pet","NDMI_lag1","MNDWI_lag1","pet_lag1"]
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

# ----------------------------
# Pipeline builder
# ----------------------------
def build_pipeline(model, feature_cols):
    pre = ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("model", clone(model))])

def numeric_feature_cols(df, targets):
    exclude = set(targets) | {DATE_COL}
    return [c for c in df.columns if (c not in exclude) and pd.api.types.is_numeric_dtype(df[c])]

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dev", action="store_true", help="force dev_mode true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    raw_dir = cfg["project"]["raw_dir"]
    external_path = cfg["project"]["external_path"]
    cache_dir = cfg["project"]["cache_dir"]
    ensure_dir(cache_dir)

    out_dir = cfg["project"]["out_dir"]
    ensure_dir(out_dir)

    switches = cfg["switches"]
    if args.dev:
        switches["dev_mode"] = True

    targets = cfg["targets"]["list"]

    # ✅ y_mode global + override por target
    y_mode_default = cfg["targets"]["y_mode"]
    y_mode_by_target = cfg.get("targets", {}).get("y_mode_by_target", {})

    cv_folds = cfg["cv"]["folds_dev"] if switches["dev_mode"] else cfg["cv"]["folds"]
    best_grid = cfg["cv"]["best_grid"]

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

    # -------- build dataset --------
    if (not switches["build_dataset"]) and os.path.exists(cache_train) and os.path.exists(cache_valid) and os.path.exists(cache_feat):
        train_df = pd.read_pickle(cache_train)
        valid_df = pd.read_pickle(cache_valid)
        with open(cache_feat, "r", encoding="utf-8") as f:
            _base_feature_cols = json.load(f)
        print("⚡ Using CACHE dataset.")
    else:
        # load raw
        wq = pd.read_csv(os.path.join(raw_dir, cfg["io"]["train_wq_name"]))
        ls_tr = pd.read_csv(os.path.join(raw_dir, cfg["io"]["train_ls_name"]))
        tc_tr = pd.read_csv(os.path.join(raw_dir, cfg["io"]["train_tc_name"]))
        ls_va = pd.read_csv(os.path.join(raw_dir, cfg["io"]["valid_ls_name"]))
        tc_va = pd.read_csv(os.path.join(raw_dir, cfg["io"]["valid_tc_name"]))

        # normalize for merges
        wq = normalize_keys_for_matching(wq)
        ls_tr = normalize_keys_for_matching(ls_tr)
        tc_tr = normalize_keys_for_matching(tc_tr)
        ls_va = normalize_keys_for_matching(ls_va)
        tc_va = normalize_keys_for_matching(tc_va)

        # merge train
        wq_date = wq.dropna(subset=[DATE_COL]).copy()
        train_df = (
            wq_date
            .merge(ls_tr, on=KEYS, how="inner", suffixes=("", "_ls"))
            .merge(tc_tr, on=KEYS, how="inner", suffixes=("", "_tc"))
        )
        train_df = train_df.loc[:, ~train_df.columns.duplicated()].copy()

        # merge valid
        valid_df = ls_va.merge(tc_va, on=KEYS, how="inner")
        # align to template ORDER (by normalized keys)
        if not template_norm[KEYS].equals(valid_df[KEYS]):
            valid_df = template_norm[KEYS].merge(valid_df, on=KEYS, how="left")

        # externals
        ext = pd.read_csv(external_path)
        ext[LAT_COL] = pd.to_numeric(ext[LAT_COL], errors="coerce")
        ext[LON_COL] = pd.to_numeric(ext[LON_COL], errors="coerce")

        train_df = train_df.merge(ext, on=[LAT_COL, LON_COL], how="left")
        valid_df = valid_df.merge(ext, on=[LAT_COL, LON_COL], how="left")

        # sanity locks (norm keys)
        assert train_df.duplicated(subset=KEYS).sum() == 0, "Duplicated keys in train"
        assert len(valid_df) == 200, "Valid must be 200"
        assert template_norm[KEYS].equals(valid_df[KEYS]), "Valid not aligned to template keys (normalized)"

        # FE V4
        train_df = enrich_features(train_df)
        valid_df = enrich_features(valid_df)

        train_df = add_valid_lag1_and_time(train_df)
        valid_df = add_valid_lag1_and_time(valid_df)

        train_df = add_spatial_basis_v4(train_df)
        valid_df = add_spatial_basis_v4(valid_df)

        # post-FE alignment lock
        assert template_norm[KEYS].equals(valid_df[KEYS]), "POST-FE valid order scrambled"

        # missing flags
        train_df, valid_df = add_missing_flags(train_df, valid_df)

        # cache save
        pd.to_pickle(train_df, cache_train)
        pd.to_pickle(valid_df, cache_valid)
        with open(cache_feat, "w", encoding="utf-8") as f:
            json.dump(numeric_feature_cols(train_df, targets), f, ensure_ascii=False, indent=2)

        print("✅ Cache saved.")

    # ----------------------------
    # CV (fold-safe TE dentro del fold)
    # ----------------------------
    et_cv_params = cfg["model"]["et_cv"]
    model_cv = ExtraTreesRegressor(**et_cv_params)

    def cv_with_fold_te(df, target):
        grid = float(best_grid[target])
        y = df[target].astype(float).values
        groups = make_groups(df, grid).values
        gkf = GroupKFold(n_splits=cv_folds)

        scores = []
        for tr_idx, va_idx in gkf.split(df, y, groups=groups):
            X_tr = df.iloc[tr_idx].copy()
            X_va = df.iloc[va_idx].copy()
            y_tr = y[tr_idx]
            y_va = y[va_idx]

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

            feat = numeric_feature_cols(X_tr, targets)

            # ===== LOCK CV: mismas columnas train vs val en el fold =====
            missing_in_va = [c for c in feat if c not in X_va.columns]
            assert len(missing_in_va) == 0, (
                f"ERROR CV: fold-valid missing {len(missing_in_va)} cols for target={target}. "
                f"Ej: {missing_in_va[:10]}"
            )

            X_tr_X = X_tr[feat].copy()
            X_va_X = X_va[feat].copy()
            assert list(X_tr_X.columns) == list(X_va_X.columns), "ERROR CV: column order mismatch"

            pipe = build_pipeline(model_cv, feat)
            pipe.fit(X_tr_X, y_tr_t)
            pred = inv(pipe.predict(X_va_X))

            if target == "Dissolved Reactive Phosphorus":
                pred = np.maximum(pred, 0.0)

            scores.append(r2_score(y_va, pred))

        return float(np.mean(scores)), float(np.std(scores)), scores

    if switches["run_cv"]:
        print("\n" + "="*80)
        print(f"CV REPORT | folds={cv_folds} | dev_mode={switches['dev_mode']} | TE={te_enabled} (fold-safe)")
        for t in targets:
            m, s, folds = cv_with_fold_te(train_df, t)
            print(f"{t:>28} | mean={m:.4f} ± {s:.4f} | folds={np.round(folds,4)}")

    # ----------------------------
    # FINAL TRAIN + SUBMISSION (OOF TE TRAIN + full-map VALID)
    # ----------------------------
    if switches["train_final"]:
        et_final_params = cfg["model"]["et_final"]
        model_final = ExtraTreesRegressor(**et_final_params)

        # start from a CLEAN submission frame
        submission_out = template_raw.copy(deep=True)

        # train/valid working copies (normalized for modeling)
        tr0 = train_df.copy()
        va0 = valid_df.copy()

        for t in targets:
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

            y = tr[t].astype(float).values
            mode_t = y_mode_by_target.get(t, y_mode_default)
            fwd, inv = y_transform_fit(y, mode_t)
            y_t = fwd(y)

            feat_cols = numeric_feature_cols(tr, targets)

            # Guardar manifest por target (auditable/reproducible)
            manifest_path = os.path.join(cache_dir, f"feature_manifest_{t.replace(' ','_')}.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(feat_cols, f, ensure_ascii=False, indent=2)

            # ===== LOCK FINAL: mismas features y mismo orden train vs valid =====
            missing_in_valid = [c for c in feat_cols if c not in va.columns]
            assert len(missing_in_valid) == 0, (
                f"ERROR: valid is missing {len(missing_in_valid)} features for target={t}. "
                f"Ej: {missing_in_valid[:10]}"
            )

            tr_X = tr[feat_cols].copy()
            va_X = va[feat_cols].copy()
            assert list(tr_X.columns) == list(va_X.columns), "ERROR: column order mismatch train vs valid"

            print(f"✅ Features locked for {t}: {len(feat_cols)} cols")

            pipe = build_pipeline(model_final, feat_cols)
            pipe.fit(tr_X, y_t)
            pred = inv(pipe.predict(va_X))

            if t == "Dissolved Reactive Phosphorus":
                pred = np.maximum(pred, 0.0)

            submission_out[t] = pred

        # CRITICAL: KEYS EXACTAS ya vienen del template_raw (intacto)
        out_path = os.path.join(out_dir, "submission_V5_2_OOFTE_fixkeys.csv")
        submission_out.to_csv(out_path, index=False)

        # final sanity
        chk = pd.read_csv(out_path)
        assert chk.shape == (200, 6)
        assert not chk[targets].isna().any().any()
        assert chk[KEYS].equals(template_raw[KEYS]), "KEYS no idénticas al template crudo"
        assert list(chk.columns) == list(template_raw.columns), "ERROR: submission columns != template columns"

        print("\n✅ Saved:", out_path)
        print("DRP min/max:", float(chk["Dissolved Reactive Phosphorus"].min()), float(chk["Dissolved Reactive Phosphorus"].max()))

if __name__ == "__main__":
    main()