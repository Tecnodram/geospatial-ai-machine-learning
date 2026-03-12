#!/usr/bin/env python
# coding: utf-8

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"
KEYS = [LAT_COL, LON_COL, DATE_COL]
TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]


@dataclass
class ExpResult:
    name: str
    ta: float
    ec: float
    drp: float
    mean_cv: float
    details: dict


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    return out


def winsor_fit(y: np.ndarray, q=(0.01, 0.99)):
    return float(np.nanquantile(y, q[0])), float(np.nanquantile(y, q[1]))


def winsor_apply(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(y, dtype=float), lo, hi)


def load_data(root: Path):
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

    # Minimal existing interactions only (no new large family)
    for df in (train, valid):
        if "NDMI" in df.columns and "upstream_area_km2" in df.columns:
            df["NDMI_upstream"] = df["NDMI"] * df["upstream_area_km2"]
        if "slope" in df.columns and "chirps_ppt" in df.columns:
            df["slope_runoff"] = df["slope"] * df["chirps_ppt"]

    return train, valid, template_raw


def add_hydro_cluster(train_df: pd.DataFrame, valid_df: pd.DataFrame, k: int):
    cols = [
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
    present = [c for c in cols if c in train_df.columns]
    tr = train_df.copy()
    va = valid_df.copy()
    if len(present) < 4:
        tr["hydro_cluster"] = 0
        va["hydro_cluster"] = 0
        return tr, va

    imp = SimpleImputer(strategy="median")
    trX = imp.fit_transform(tr[present])
    vaX = imp.transform(va[present])
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    tr["hydro_cluster"] = km.fit_predict(trX)
    va["hydro_cluster"] = km.predict(vaX)
    return tr, va


def feature_cols(df: pd.DataFrame):
    drop_cols = set(TARGETS + [DATE_COL, LAT_COL, LON_COL, "basin_id", "station_id"])
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def mk_pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", model)])


def mk_meta(meta_name: str):
    if meta_name == "ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    if meta_name == "elasticnet":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=10000, random_state=42))])
    if meta_name == "hgb_small":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", HistGradientBoostingRegressor(max_iter=220, learning_rate=0.05, max_depth=3, min_samples_leaf=20, random_state=42))])
    raise ValueError(f"Unknown meta model: {meta_name}")


def target_model_bank(target: str):
    # Controlled per-target tuning of base model hyperparameters
    if target == "Total Alkalinity":
        et = ExtraTreesRegressor(n_estimators=1800, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
        rf = RandomForestRegressor(n_estimators=1100, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
        hgb = HistGradientBoostingRegressor(max_iter=900, learning_rate=0.05, max_depth=6, min_samples_leaf=18, random_state=42)
    elif target == "Electrical Conductance":
        et = ExtraTreesRegressor(n_estimators=2200, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
        rf = RandomForestRegressor(n_estimators=1300, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
        hgb = HistGradientBoostingRegressor(max_iter=1100, learning_rate=0.04, max_depth=6, min_samples_leaf=20, random_state=42)
    else:  # DRP stacked branch
        et = ExtraTreesRegressor(n_estimators=2600, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
        rf = RandomForestRegressor(n_estimators=1500, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
        hgb = HistGradientBoostingRegressor(max_iter=1200, learning_rate=0.04, max_depth=5, min_samples_leaf=22, random_state=42)
    return {"ET": et, "RF": rf, "HGB": hgb}


def search_blend_weights(y, pred_et, pred_rf, pred_hgb, groups):
    gkf = GroupKFold(n_splits=5)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best = (-9e9, (1 / 3, 1 / 3, 1 / 3))

    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0.0:
                continue
            pred = w1 * pred_et + w2 * pred_rf + w3 * pred_hgb
            scores = []
            for _, va_idx in gkf.split(pred, y, groups):
                scores.append(r2_score(y[va_idx], pred[va_idx]))
            m = float(np.mean(scores))
            if m > best[0]:
                best = (m, (float(w1), float(w2), float(w3)))
    return best[1], best[0]


def stacked_oof_for_target(df: pd.DataFrame, target: str, meta_name: str):
    groups = df["basin_id"].fillna("unknown").astype(str).values
    y = df[target].astype(float).values
    feats = feature_cols(df)
    gkf = GroupKFold(n_splits=5)

    oof_et = np.zeros(len(df), dtype=float)
    oof_rf = np.zeros(len(df), dtype=float)
    oof_hgb = np.zeros(len(df), dtype=float)

    models = target_model_bank(target)

    for tr_idx, va_idx in gkf.split(df, y, groups):
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]
        lo, hi = winsor_fit(tr[target].values)
        y_tr = winsor_apply(tr[target].values, lo, hi)

        p_et = mk_pipe(models["ET"])
        p_rf = mk_pipe(models["RF"])
        p_hgb = mk_pipe(models["HGB"])
        p_et.fit(tr[feats], y_tr)
        p_rf.fit(tr[feats], y_tr)
        p_hgb.fit(tr[feats], y_tr)

        oof_et[va_idx] = np.maximum(p_et.predict(va[feats]), 0.0)
        oof_rf[va_idx] = np.maximum(p_rf.predict(va[feats]), 0.0)
        oof_hgb[va_idx] = np.maximum(p_hgb.predict(va[feats]), 0.0)

    # OOF weight search
    best_w, best_weighted_cv = search_blend_weights(y, oof_et, oof_rf, oof_hgb, groups)
    oof_weighted = best_w[0] * oof_et + best_w[1] * oof_rf + best_w[2] * oof_hgb

    # OOF meta model evaluation
    meta_X = pd.DataFrame({
        "pred_ET": oof_et,
        "pred_RF": oof_rf,
        "pred_HGB": oof_hgb,
        "hydro_cluster": df["hydro_cluster"].astype(float).values,
    })
    gkf2 = GroupKFold(n_splits=5)
    oof_meta = np.zeros(len(df), dtype=float)
    for tr_idx, va_idx in gkf2.split(meta_X, y, groups):
        m = mk_meta(meta_name)
        m.fit(meta_X.iloc[tr_idx], y[tr_idx])
        oof_meta[va_idx] = m.predict(meta_X.iloc[va_idx])
    oof_meta = np.maximum(oof_meta, 0.0)
    meta_cv = float(r2_score(y, oof_meta))

    return {
        "y": y,
        "oof_et": oof_et,
        "oof_rf": oof_rf,
        "oof_hgb": oof_hgb,
        "oof_weighted": oof_weighted,
        "oof_meta": oof_meta,
        "best_weights": best_w,
        "best_weighted_cv": float(best_weighted_cv),
        "meta_cv": meta_cv,
    }


def drp_two_stage_oof(df: pd.DataFrame, threshold_q: float):
    t = "Dissolved Reactive Phosphorus"
    groups = df["basin_id"].fillna("unknown").astype(str).values
    y = df[t].astype(float).values
    feats = feature_cols(df)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(df), dtype=float)

    for tr_idx, va_idx in gkf.split(df, y, groups):
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]

        y_tr = tr[t].values.astype(float)
        qv = float(np.nanquantile(y_tr, threshold_q))
        y_bin = (y_tr >= qv).astype(int)

        clf = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=5, min_samples_leaf=25, random_state=42)),
        ])
        clf.fit(tr[feats], y_bin)
        p_tr = clf.predict_proba(tr[feats])[:, 1]
        p_va = clf.predict_proba(va[feats])[:, 1]

        tr2 = tr[feats].copy()
        va2 = va[feats].copy()
        tr2["P_high"] = p_tr
        va2["P_high"] = p_va

        lo, hi = winsor_fit(y_tr)
        y_tr_w = winsor_apply(y_tr, lo, hi)
        reg = mk_pipe(ExtraTreesRegressor(n_estimators=2600, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        reg.fit(tr2, y_tr_w)
        reg_pred = np.maximum(reg.predict(va2), 0.0)
        oof[va_idx] = np.maximum(0.6 * reg_pred + 0.4 * (p_va * reg_pred), 0.0)

    return oof


def fit_predict_valid(df_tr: pd.DataFrame, df_va: pd.DataFrame, target: str, meta_name: str, best_weights):
    feats = feature_cols(df_tr)
    y = df_tr[target].values.astype(float)
    lo, hi = winsor_fit(y)
    y_w = winsor_apply(y, lo, hi)

    models = target_model_bank(target)
    p_et = mk_pipe(models["ET"])
    p_rf = mk_pipe(models["RF"])
    p_hgb = mk_pipe(models["HGB"])
    p_et.fit(df_tr[feats], y_w)
    p_rf.fit(df_tr[feats], y_w)
    p_hgb.fit(df_tr[feats], y_w)

    pred_et_tr = np.maximum(p_et.predict(df_tr[feats]), 0.0)
    pred_rf_tr = np.maximum(p_rf.predict(df_tr[feats]), 0.0)
    pred_hgb_tr = np.maximum(p_hgb.predict(df_tr[feats]), 0.0)
    pred_et_va = np.maximum(p_et.predict(df_va[feats]), 0.0)
    pred_rf_va = np.maximum(p_rf.predict(df_va[feats]), 0.0)
    pred_hgb_va = np.maximum(p_hgb.predict(df_va[feats]), 0.0)

    meta_tr = pd.DataFrame({
        "pred_ET": pred_et_tr,
        "pred_RF": pred_rf_tr,
        "pred_HGB": pred_hgb_tr,
        "hydro_cluster": df_tr["hydro_cluster"].astype(float).values,
    })
    meta_va = pd.DataFrame({
        "pred_ET": pred_et_va,
        "pred_RF": pred_rf_va,
        "pred_HGB": pred_hgb_va,
        "hydro_cluster": df_va["hydro_cluster"].astype(float).values,
    })
    meta = mk_meta(meta_name)
    meta.fit(meta_tr, y)
    pred_meta = np.maximum(meta.predict(meta_va), 0.0)

    pred_weighted = np.maximum(
        best_weights[0] * pred_et_va + best_weights[1] * pred_rf_va + best_weights[2] * pred_hgb_va,
        0.0,
    )
    return pred_meta, pred_weighted


def write_submission(template_raw: pd.DataFrame, preds: dict, out_path: Path):
    sub = template_raw.copy()
    for t in TARGETS:
        sub[t] = np.maximum(np.nan_to_num(preds[t], nan=0.0), 0.0)
    sub.to_csv(out_path, index=False)


def run_experiment(df: pd.DataFrame, exp_name: str, meta_name: str, drp_threshold: float, drp_two_stage_alpha: float):
    ta_pack = stacked_oof_for_target(df, "Total Alkalinity", meta_name)
    ec_pack = stacked_oof_for_target(df, "Electrical Conductance", meta_name)
    drp_stack_pack = stacked_oof_for_target(df, "Dissolved Reactive Phosphorus", meta_name)
    drp_two = drp_two_stage_oof(df, threshold_q=drp_threshold)

    y_ta = ta_pack["y"]
    y_ec = ec_pack["y"]
    y_drp = drp_stack_pack["y"]

    # Blend stacked DRP with two-stage DRP
    drp_final = np.maximum(
        drp_two_stage_alpha * drp_two + (1.0 - drp_two_stage_alpha) * drp_stack_pack["oof_meta"],
        0.0,
    )

    r_ta = float(r2_score(y_ta, ta_pack["oof_meta"]))
    r_ec = float(r2_score(y_ec, ec_pack["oof_meta"]))
    r_drp = float(r2_score(y_drp, drp_final))
    mean_cv = float(np.mean([r_ta, r_ec, r_drp]))

    return ExpResult(
        name=exp_name,
        ta=r_ta,
        ec=r_ec,
        drp=r_drp,
        mean_cv=mean_cv,
        details={
            "meta_model": meta_name,
            "drp_threshold": drp_threshold,
            "drp_two_stage_alpha": drp_two_stage_alpha,
            "weights_ta": ta_pack["best_weights"],
            "weights_ec": ec_pack["best_weights"],
            "weights_drp_stack": drp_stack_pack["best_weights"],
            "weighted_cv_ta": ta_pack["best_weighted_cv"],
            "weighted_cv_ec": ec_pack["best_weighted_cv"],
            "weighted_cv_drp_stack": drp_stack_pack["best_weighted_cv"],
        },
    )


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "experiments" / "controlled_stacking_sprint"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_base, valid_base, template_raw = load_data(root)

    results = []

    # Experiments 1-3: hydro_cluster k sweep with ridge
    k_runs = []
    for i, k in enumerate([4, 6, 8], start=1):
        tr, _ = add_hydro_cluster(train_base, valid_base, k=k)
        r = run_experiment(
            tr,
            exp_name=f"EXP_{i}_K{k}_RIDGE_P75_A0.60",
            meta_name="ridge",
            drp_threshold=0.75,
            drp_two_stage_alpha=0.60,
        )
        results.append(r)
        k_runs.append((k, r.mean_cv))

    best_k = sorted(k_runs, key=lambda x: x[1], reverse=True)[0][0]

    # Experiment 4: elasticnet
    tr_best, va_best = add_hydro_cluster(train_base, valid_base, k=best_k)
    r4 = run_experiment(
        tr_best,
        exp_name=f"EXP_4_K{best_k}_ELASTICNET_P75_A0.60",
        meta_name="elasticnet",
        drp_threshold=0.75,
        drp_two_stage_alpha=0.60,
    )
    results.append(r4)

    # Experiment 5: hgb_small meta
    r5 = run_experiment(
        tr_best,
        exp_name=f"EXP_5_K{best_k}_HGBSMALL_P75_A0.60",
        meta_name="hgb_small",
        drp_threshold=0.75,
        drp_two_stage_alpha=0.60,
    )
    results.append(r5)

    # Experiment 6: DRP threshold/blend sweep using best meta from exps 1-5
    best_prev = sorted(results, key=lambda x: x.mean_cv, reverse=True)[0]
    best_meta = best_prev.details["meta_model"]
    best_cfg = None
    for q in [0.70, 0.75, 0.80, 0.85]:
        for a in [0.40, 0.50, 0.60, 0.70, 0.80]:
            rr = run_experiment(
                tr_best,
                exp_name=f"EXP_6_K{best_k}_{best_meta.upper()}_P{int(q*100)}_A{a:.2f}",
                meta_name=best_meta,
                drp_threshold=q,
                drp_two_stage_alpha=a,
            )
            if best_cfg is None or rr.mean_cv > best_cfg.mean_cv:
                best_cfg = rr
    results.append(best_cfg)

    results_sorted = sorted(results, key=lambda x: x.mean_cv, reverse=True)
    best = results_sorted[0]

    # Build final submission with best experiment settings
    tr_final, va_final = add_hydro_cluster(train_base, valid_base, k=best_k)

    ta_pack = stacked_oof_for_target(tr_final, "Total Alkalinity", best.details["meta_model"])
    ec_pack = stacked_oof_for_target(tr_final, "Electrical Conductance", best.details["meta_model"])
    drp_stack_pack = stacked_oof_for_target(tr_final, "Dissolved Reactive Phosphorus", best.details["meta_model"])

    ta_meta_pred, _ = fit_predict_valid(tr_final, va_final, "Total Alkalinity", best.details["meta_model"], ta_pack["best_weights"])
    ec_meta_pred, _ = fit_predict_valid(tr_final, va_final, "Electrical Conductance", best.details["meta_model"], ec_pack["best_weights"])
    drp_meta_pred, _ = fit_predict_valid(tr_final, va_final, "Dissolved Reactive Phosphorus", best.details["meta_model"], drp_stack_pack["best_weights"])

    # DRP two-stage valid prediction with best threshold
    feats = feature_cols(tr_final)
    y_drp = tr_final["Dissolved Reactive Phosphorus"].values.astype(float)
    qv = float(np.nanquantile(y_drp, best.details["drp_threshold"]))
    y_bin = (y_drp >= qv).astype(int)
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=5, min_samples_leaf=25, random_state=42)),
    ])
    clf.fit(tr_final[feats], y_bin)
    p_tr = clf.predict_proba(tr_final[feats])[:, 1]
    p_va = clf.predict_proba(va_final[feats])[:, 1]

    tr2 = tr_final[feats].copy()
    va2 = va_final[feats].copy()
    tr2["P_high"] = p_tr
    va2["P_high"] = p_va
    lo, hi = winsor_fit(y_drp)
    y_w = winsor_apply(y_drp, lo, hi)
    reg = mk_pipe(ExtraTreesRegressor(n_estimators=2600, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
    reg.fit(tr2, y_w)
    drp_two_va = np.maximum(reg.predict(va2), 0.0)
    drp_two_va = np.maximum(0.6 * drp_two_va + 0.4 * (p_va * drp_two_va), 0.0)

    drp_final_va = np.maximum(
        best.details["drp_two_stage_alpha"] * drp_two_va + (1.0 - best.details["drp_two_stage_alpha"]) * drp_meta_pred,
        0.0,
    )

    final_preds = {
        "Total Alkalinity": ta_meta_pred,
        "Electrical Conductance": ec_meta_pred,
        "Dissolved Reactive Phosphorus": drp_final_va,
    }

    sub_name = f"submission_CONTROLLED_STACKING_SPRINT_{best.name}.csv".replace(" ", "_")
    sub_path = out_dir / sub_name
    write_submission(template_raw, final_preds, sub_path)

    report = {
        "experiments": [
            {
                "name": r.name,
                "TA": r.ta,
                "EC": r.ec,
                "DRP": r.drp,
                "MeanCV": r.mean_cv,
                "details": r.details,
            }
            for r in results
        ],
        "best_experiment": {
            "name": best.name,
            "TA": best.ta,
            "EC": best.ec,
            "DRP": best.drp,
            "MeanCV": best.mean_cv,
            "details": best.details,
        },
        "best_configuration_per_target": {
            "TA": "Stacked meta-model (best meta in sprint)",
            "EC": "Stacked meta-model (best meta in sprint)",
            "DRP": "Two-stage + stacked blend (best threshold/alpha in sprint)",
        },
        "recommended_submission_file": str(sub_path),
    }
    with open(out_dir / "controlled_stacking_sprint_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 90)
    print("CONTROLLED OPTIMIZATION SPRINT (6 EXPERIMENTS)")
    print("=" * 90)
    for r in results:
        print(f"{r.name:45s} | TA={r.ta:.4f} EC={r.ec:.4f} DRP={r.drp:.4f} Mean={r.mean_cv:.4f}")
    print("-" * 90)
    print(f"BEST: {best.name} | TA={best.ta:.4f} EC={best.ec:.4f} DRP={best.drp:.4f} Mean={best.mean_cv:.4f}")
    print("Recommended submission:", sub_path)


if __name__ == "__main__":
    main()
