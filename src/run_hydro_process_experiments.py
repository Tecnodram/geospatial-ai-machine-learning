#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"
KEYS = [LAT_COL, LON_COL, DATE_COL]
TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    return out


def winsor_fit(y: np.ndarray, q=(0.01, 0.99)):
    lo = float(np.nanquantile(y, q[0]))
    hi = float(np.nanquantile(y, q[1]))
    return lo, hi


def winsor_apply(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(y, dtype=float), lo, hi)


def build_dataset(root: Path):
    raw = root / "data" / "raw"
    ext_hydro = root / "data" / "external_geofeatures_hydro_v2.csv"
    ext_geo = root / "data" / "external_geofeatures.csv"
    chirps_tr = root / "data" / "external" / "chirps_features_training.csv"
    chirps_va = root / "data" / "external" / "chirps_features_validation.csv"

    wq = normalize_keys(pd.read_csv(raw / "water_quality_training_dataset.csv"))
    ls_tr = normalize_keys(pd.read_csv(raw / "landsat_features_training.csv"))
    tc_tr = normalize_keys(pd.read_csv(raw / "terraclimate_features_training.csv"))
    ls_va = normalize_keys(pd.read_csv(raw / "landsat_features_validation.csv"))
    tc_va = normalize_keys(pd.read_csv(raw / "terraclimate_features_validation.csv"))
    c_tr = normalize_keys(pd.read_csv(chirps_tr))
    c_va = normalize_keys(pd.read_csv(chirps_va))

    ext_h = pd.read_csv(ext_hydro)
    ext_g = pd.read_csv(ext_geo)
    for d in (ext_h, ext_g):
        d[LAT_COL] = pd.to_numeric(d[LAT_COL], errors="coerce")
        d[LON_COL] = pd.to_numeric(d[LON_COL], errors="coerce")

    train = (
        wq.merge(ls_tr, on=KEYS, how="inner")
        .merge(tc_tr, on=KEYS, how="inner")
        .merge(c_tr, on=KEYS, how="left")
        .merge(ext_h, on=[LAT_COL, LON_COL], how="left")
        .merge(ext_g, on=[LAT_COL, LON_COL], how="left")
    )
    valid = (
        ls_va.merge(tc_va, on=KEYS, how="inner")
        .merge(c_va, on=KEYS, how="left")
        .merge(ext_h, on=[LAT_COL, LON_COL], how="left")
        .merge(ext_g, on=[LAT_COL, LON_COL], how="left")
    )

    template_raw = pd.read_csv(raw / "submission_template.csv")
    template_norm = normalize_keys(template_raw)
    valid = template_norm[KEYS].merge(valid, on=KEYS, how="left")

    return train, valid, template_raw, template_norm


def add_hydro_process_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    if "landcover_urban" not in x.columns:
        if "landcover" in x.columns:
            x["landcover_urban"] = (x["landcover"].astype(float) == 50.0).astype(float)
        else:
            x["landcover_urban"] = 0.0

    up = x.get("upstream_area_km2", pd.Series(0.0, index=x.index)).astype(float)
    clay = x.get("soil_clay_0_5", pd.Series(0.0, index=x.index)).astype(float)
    ph = x.get("soil_ph_0_5", pd.Series(0.0, index=x.index)).astype(float)
    soc = x.get("soil_soc_0_5", pd.Series(0.0, index=x.index)).astype(float)
    slope = x.get("slope", pd.Series(0.0, index=x.index)).astype(float)
    elev = x.get("elevation", pd.Series(0.0, index=x.index)).astype(float)
    discharge = x.get("river_discharge_cms", pd.Series(0.0, index=x.index)).astype(float)
    dist = x.get("dist_to_river_m", pd.Series(0.0, index=x.index)).astype(float)
    ndmi = x.get("NDMI", pd.Series(0.0, index=x.index)).astype(float)
    precip = x.get("chirps_ppt", pd.Series(0.0, index=x.index)).astype(float)

    # Phase 1: flow accumulation features
    x["soil_clay_flow"] = clay * up
    x["urban_flow"] = x["landcover_urban"].astype(float) * up
    x["soil_ph_buffer"] = ph * np.log1p(np.maximum(up, 0.0))
    x["flow_contact_time"] = up / (np.maximum(discharge, 0.0) + 1.0)
    x["chemical_accumulation_index"] = np.log1p(np.maximum(up, 0.0)) * soc / (np.maximum(discharge, 0.0) + 1.0)

    # Phase 2: river proximity decay
    x["river_decay_5km"] = np.exp(-np.maximum(dist, 0.0) / 5000.0)
    x["river_decay_10km"] = np.exp(-np.maximum(dist, 0.0) / 10000.0)
    x["river_influence"] = x["river_decay_5km"] * np.log1p(np.maximum(up, 0.0))

    # Phase 3: hydro-climate interactions
    x["NDMI_upstream"] = ndmi * up
    x["slope_runoff"] = slope * precip
    x["soil_precip_interaction"] = clay * precip
    x["ph_runoff_proxy"] = ph * slope

    # Phase 4: geological accumulation model
    x["mineral_contact_time"] = np.log1p(np.maximum(up, 0.0)) * elev / (np.maximum(discharge, 0.0) + 1.0)
    x["mineral_gradient"] = elev * x["river_decay_5km"]

    return x


def add_hydro_cluster(train_df: pd.DataFrame, valid_df: pd.DataFrame, k: int = 6):
    cluster_cols = [
        LAT_COL,
        LON_COL,
        "elevation",
        "slope",
        "basin_area_km2",
        "upstream_area_km2",
        "dist_to_river_m",
        "river_discharge_cms",
        "catch_to_basin_ratio",
        "soil_clay_0_5",
        "landcover",
    ]
    present = [c for c in cluster_cols if c in train_df.columns]
    if len(present) < 4:
        train_df = train_df.copy()
        valid_df = valid_df.copy()
        train_df["hydro_cluster"] = 0
        valid_df["hydro_cluster"] = 0
        return train_df, valid_df

    imp = SimpleImputer(strategy="median")
    trX = imp.fit_transform(train_df[present])
    vaX = imp.transform(valid_df[present])
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["hydro_cluster"] = km.fit_predict(trX)
    valid_df["hydro_cluster"] = km.predict(vaX)
    return train_df, valid_df


def feature_cols(df: pd.DataFrame, include_hydro_cluster: bool) -> list:
    drop_cols = set(TARGETS + [DATE_COL, LAT_COL, LON_COL, "basin_id", "station_id"])
    cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not include_hydro_cluster:
        cols = [c for c in cols if c != "hydro_cluster"]
    return cols


def make_reg_pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def make_meta_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def cv_extra_trees(train_df: pd.DataFrame, use_cluster: bool):
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    gkf = GroupKFold(n_splits=5)

    fold_scores = {t: [] for t in TARGETS}
    oof_preds = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    for tr_idx, va_idx in gkf.split(train_df, train_df[TARGETS[0]].values, groups=groups):
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()
        feats = feature_cols(tr, include_hydro_cluster=use_cluster)

        model = ExtraTreesRegressor(
            n_estimators=2200,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        pipe = make_reg_pipe(model)

        for t in TARGETS:
            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)
            pipe.fit(tr[feats], y_tr)
            pred = pipe.predict(va[feats])
            pred = np.maximum(pred, 0.0)
            oof_preds[t][va_idx] = pred
            fold_scores[t].append(r2_score(va[t].values, pred))

    report = {
        t: {
            "mean": float(np.mean(fold_scores[t])),
            "std": float(np.std(fold_scores[t])),
            "var": float(np.var(fold_scores[t])),
            "folds": [float(v) for v in fold_scores[t]],
        }
        for t in TARGETS
    }
    report["mean_cv"] = float(np.mean([report[t]["mean"] for t in TARGETS]))
    return report, oof_preds


def cv_stacking_with_drp_two_stage(train_df: pd.DataFrame):
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    gkf = GroupKFold(n_splits=5)

    fold_scores = {t: [] for t in TARGETS}
    oof_preds = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    for tr_idx, va_idx in gkf.split(train_df, train_df[TARGETS[0]].values, groups=groups):
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()
        feats = feature_cols(tr, include_hydro_cluster=True)

        # TA and EC: OOF stacking (ET, RF, HGB -> Ridge)
        for t in ["Total Alkalinity", "Electrical Conductance"]:
            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)
            y_va = va[t].values

            et = make_reg_pipe(ExtraTreesRegressor(n_estimators=1800, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
            rf = make_reg_pipe(RandomForestRegressor(n_estimators=1200, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
            hgb = make_reg_pipe(HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=6, min_samples_leaf=20, random_state=42))

            et.fit(tr[feats], y_tr)
            rf.fit(tr[feats], y_tr)
            hgb.fit(tr[feats], y_tr)

            m_tr = pd.DataFrame({
                "pred_ET": et.predict(tr[feats]),
                "pred_RF": rf.predict(tr[feats]),
                "pred_HGB": hgb.predict(tr[feats]),
                "hydro_cluster": tr["hydro_cluster"].values,
            })
            m_va = pd.DataFrame({
                "pred_ET": et.predict(va[feats]),
                "pred_RF": rf.predict(va[feats]),
                "pred_HGB": hgb.predict(va[feats]),
                "hydro_cluster": va["hydro_cluster"].values,
            })
            meta = make_meta_pipe()
            meta.fit(m_tr, y_tr)
            pred = np.maximum(meta.predict(m_va), 0.0)
            oof_preds[t][va_idx] = pred
            fold_scores[t].append(r2_score(y_va, pred))

        # DRP: two-stage model
        t = "Dissolved Reactive Phosphorus"
        y_tr_drp = tr[t].values.astype(float)
        y_va_drp = va[t].values.astype(float)
        q75 = float(np.nanquantile(y_tr_drp, 0.75))
        y_high = (y_tr_drp >= q75).astype(int)

        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=5, min_samples_leaf=25, random_state=42)),
        ])
        clf.fit(tr[feats], y_high)
        p_high_tr = clf.predict_proba(tr[feats])[:, 1]
        p_high_va = clf.predict_proba(va[feats])[:, 1]

        tr2 = tr[feats].copy()
        va2 = va[feats].copy()
        tr2["P_high"] = p_high_tr
        va2["P_high"] = p_high_va

        lo, hi = winsor_fit(y_tr_drp)
        y_drp_w = winsor_apply(y_tr_drp, lo, hi)
        reg = make_reg_pipe(ExtraTreesRegressor(n_estimators=2600, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        reg.fit(tr2, y_drp_w)
        reg_pred = np.maximum(reg.predict(va2), 0.0)
        drp_pred = 0.6 * reg_pred + 0.4 * (p_high_va * reg_pred)
        drp_pred = np.maximum(drp_pred, 0.0)

        oof_preds[t][va_idx] = drp_pred
        fold_scores[t].append(r2_score(y_va_drp, drp_pred))

    report = {
        t: {
            "mean": float(np.mean(fold_scores[t])),
            "std": float(np.std(fold_scores[t])),
            "var": float(np.var(fold_scores[t])),
            "folds": [float(v) for v in fold_scores[t]],
        }
        for t in TARGETS
    }
    report["mean_cv"] = float(np.mean([report[t]["mean"] for t in TARGETS]))
    return report, oof_preds


def fit_predict_valid_et(train_df: pd.DataFrame, valid_df: pd.DataFrame, use_cluster: bool):
    feats = feature_cols(train_df, include_hydro_cluster=use_cluster)
    out = {}
    for t in TARGETS:
        lo, hi = winsor_fit(train_df[t].values)
        y = winsor_apply(train_df[t].values, lo, hi)
        model = make_reg_pipe(ExtraTreesRegressor(n_estimators=2200, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        model.fit(train_df[feats], y)
        out[t] = np.maximum(model.predict(valid_df[feats]), 0.0)
    return out


def fit_predict_valid_stacking_drp2(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    feats = feature_cols(train_df, include_hydro_cluster=True)
    out = {}

    for t in ["Total Alkalinity", "Electrical Conductance"]:
        lo, hi = winsor_fit(train_df[t].values)
        y = winsor_apply(train_df[t].values, lo, hi)

        et = make_reg_pipe(ExtraTreesRegressor(n_estimators=1800, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        rf = make_reg_pipe(RandomForestRegressor(n_estimators=1200, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        hgb = make_reg_pipe(HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=6, min_samples_leaf=20, random_state=42))
        et.fit(train_df[feats], y)
        rf.fit(train_df[feats], y)
        hgb.fit(train_df[feats], y)

        m_tr = pd.DataFrame({
            "pred_ET": et.predict(train_df[feats]),
            "pred_RF": rf.predict(train_df[feats]),
            "pred_HGB": hgb.predict(train_df[feats]),
            "hydro_cluster": train_df["hydro_cluster"].values,
        })
        m_va = pd.DataFrame({
            "pred_ET": et.predict(valid_df[feats]),
            "pred_RF": rf.predict(valid_df[feats]),
            "pred_HGB": hgb.predict(valid_df[feats]),
            "hydro_cluster": valid_df["hydro_cluster"].values,
        })
        meta = make_meta_pipe()
        meta.fit(m_tr, y)
        out[t] = np.maximum(meta.predict(m_va), 0.0)

    t = "Dissolved Reactive Phosphorus"
    y = train_df[t].values.astype(float)
    q75 = float(np.nanquantile(y, 0.75))
    y_high = (y >= q75).astype(int)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=5, min_samples_leaf=25, random_state=42)),
    ])
    clf.fit(train_df[feats], y_high)
    p_high_tr = clf.predict_proba(train_df[feats])[:, 1]
    p_high_va = clf.predict_proba(valid_df[feats])[:, 1]

    tr2 = train_df[feats].copy()
    va2 = valid_df[feats].copy()
    tr2["P_high"] = p_high_tr
    va2["P_high"] = p_high_va

    lo, hi = winsor_fit(y)
    y_w = winsor_apply(y, lo, hi)
    reg = make_reg_pipe(ExtraTreesRegressor(n_estimators=2600, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
    reg.fit(tr2, y_w)
    reg_pred = np.maximum(reg.predict(va2), 0.0)
    out[t] = np.maximum(0.6 * reg_pred + 0.4 * (p_high_va * reg_pred), 0.0)

    return out


def write_submission(template: pd.DataFrame, preds: dict, out_path: Path):
    sub = template.copy()
    for t in TARGETS:
        sub[t] = np.nan_to_num(preds[t], nan=0.0)
        sub[t] = np.maximum(sub[t], 0.0)
    sub.to_csv(out_path, index=False)


def print_report(name: str, report: dict):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    for t in TARGETS:
        r = report[t]
        print(f"{t:>28} | mean={r['mean']:.4f} std={r['std']:.4f} var={r['var']:.4f} folds={np.round(r['folds'],4)}")
    print(f"{'MEAN CV':>28} | {report['mean_cv']:.4f}")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "experiments" / "hydro_process_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    train, valid, template_raw, _template_norm = build_dataset(root)
    train = add_hydro_process_features(train)
    valid = add_hydro_process_features(valid)
    train, valid = add_hydro_cluster(train, valid, k=6)

    # EXP 1: Flow features + ET (with hydro_cluster feature)
    rep1, _ = cv_extra_trees(train, use_cluster=True)
    pred1 = fit_predict_valid_et(train, valid, use_cluster=True)
    sub1_path = out_dir / "submission_EXP_FLOW_FEATURES_V1.csv"
    write_submission(template_raw, pred1, sub1_path)
    print_report("EXP_FLOW_FEATURES_V1", rep1)

    # EXP 2: OOF stacking + DRP two-stage
    rep2, _ = cv_stacking_with_drp_two_stage(train)
    pred2 = fit_predict_valid_stacking_drp2(train, valid)
    sub2_path = out_dir / "submission_EXP_FLOW_FEATURES_STACKING_V1.csv"
    write_submission(template_raw, pred2, sub2_path)
    print_report("EXP_FLOW_FEATURES_STACKING_V1", rep2)

    # Pick best by mean CV
    best_name = "EXP_FLOW_FEATURES_V1" if rep1["mean_cv"] >= rep2["mean_cv"] else "EXP_FLOW_FEATURES_STACKING_V1"
    best_sub = sub1_path if best_name == "EXP_FLOW_FEATURES_V1" else sub2_path

    summary = {
        "EXP_FLOW_FEATURES_V1": rep1,
        "EXP_FLOW_FEATURES_STACKING_V1": rep2,
        "best_experiment": best_name,
        "best_submission": str(best_sub),
        "stop_condition_mean_cv_ge_0_38": bool(max(rep1["mean_cv"], rep2["mean_cv"]) >= 0.38),
    }
    with open(out_dir / "hydro_process_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBest experiment:", best_name)
    print("Best submission:", best_sub)
    print("Stop condition reached (mean CV >= 0.38):", summary["stop_condition_mean_cv_ge_0_38"])


if __name__ == "__main__":
    main()
