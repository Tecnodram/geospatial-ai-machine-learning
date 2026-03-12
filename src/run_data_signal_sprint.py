#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import json
import traceback
from datetime import datetime
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

from features.data_signal_features import (
    add_landuse_pressure,
    add_moisture_dryness_package,
    add_rainfall_antecedent_package,
    add_upstream_pressure,
)

LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"
KEYS = [LAT_COL, LON_COL, DATE_COL]
TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]


def ts():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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


class AuditLogger:
    def __init__(self, root: Path, total_experiments: int, benchmark_mean: float = 0.320):
        self.root = root
        self.total_experiments = total_experiments
        self.progress_path = root / "progress.json"
        self.running_results_path = root / "running_results.csv"
        self.errors_path = root / "errors.log"
        self.started_at = ts()
        self.completed_experiments = 0
        self.best_mean = benchmark_mean
        self.best_name = "EXP_STACKING_V1"
        if not self.running_results_path.exists():
            pd.DataFrame(columns=["experiment", "target", "fold", "r2", "timestamp"]).to_csv(self.running_results_path, index=False)
        self._write_progress(
            status="running",
            current_experiment="",
            experiment_index=0,
            target="",
            fold=0,
        )

    def _write_progress(self, status: str, current_experiment: str, experiment_index: int, target: str, fold: int):
        payload = {
            "status": status,
            "current_experiment": current_experiment,
            "experiment_index": int(experiment_index),
            "total_experiments": int(self.total_experiments),
            "target": target,
            "fold": int(fold),
            "started_at": self.started_at,
            "updated_at": ts(),
            "completed_experiments": int(self.completed_experiments),
            "best_mean_cv_so_far": float(self.best_mean),
            "best_experiment_so_far": self.best_name,
        }
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def start_experiment(self, idx: int, name: str):
        print(f"[EXPERIMENT] {idx}/{self.total_experiments}")
        print(f"[EXPERIMENT NAME] {name}")
        self._write_progress("running", name, idx, "", 0)

    def start_target_fold(self, idx: int, exp_name: str, target: str, fold: int, total_folds: int):
        print(f"[TARGET] {target} | [FOLD] {fold}/{total_folds} | [STATUS] running")
        self._write_progress("running", exp_name, idx, target, fold)

    def log_fold_result(self, experiment: str, target: str, fold: int, r2: float):
        row = pd.DataFrame([{
            "experiment": experiment,
            "target": target,
            "fold": int(fold),
            "r2": float(r2),
            "timestamp": ts(),
        }])
        row.to_csv(self.running_results_path, mode="a", index=False, header=False)

    def complete_experiment(self, idx: int, exp_name: str, mean_cv: float):
        self.completed_experiments += 1
        if mean_cv > self.best_mean:
            self.best_mean = float(mean_cv)
            self.best_name = exp_name
        print(f"[STATUS] completed | {exp_name} | mean_cv={mean_cv:.4f}")
        self._write_progress("running", exp_name, idx, "", 0)

    def fail_experiment(self, idx: int, exp_name: str, err: Exception):
        print(f"[STATUS] failed | {exp_name} | {err}")
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts()}] {exp_name}\n{traceback.format_exc()}\n")
        self._write_progress("running", exp_name, idx, "", 0)

    def finalize(self):
        self._write_progress("completed", self.best_name, self.total_experiments, "", 0)


def mk_pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", model)])


def mk_meta(name: str):
    if name == "ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=10000, random_state=42))])


def add_hydro_cluster(train_df: pd.DataFrame, valid_df: pd.DataFrame):
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
    km = KMeans(n_clusters=6, random_state=42, n_init=20)
    tr["hydro_cluster"] = km.fit_predict(trX)
    va["hydro_cluster"] = km.predict(vaX)
    return tr, va


def feature_cols(df: pd.DataFrame):
    drop_cols = set(TARGETS + [DATE_COL, LAT_COL, LON_COL, "basin_id", "station_id"])
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def load_base_data(root: Path):
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

    train, valid = add_hydro_cluster(train, valid)
    return train, valid, template_raw


def apply_feature_blocks(df: pd.DataFrame, blocks: list[str], skipped: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "rain" in blocks:
        out = add_rainfall_antecedent_package(out, skipped)
    if "moisture" in blocks:
        out = add_moisture_dryness_package(out, skipped)
    if "landuse" in blocks:
        out = add_landuse_pressure(out, skipped)
    if "upstream" in blocks:
        out = add_upstream_pressure(out, skipped)
    return out


def model_bank(target: str):
    if target == "Total Alkalinity":
        return ExtraTreesRegressor(n_estimators=1800, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
    if target == "Electrical Conductance":
        return ExtraTreesRegressor(n_estimators=2200, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
    return ExtraTreesRegressor(n_estimators=2600, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)


def run_cv_et(train_df: pd.DataFrame, exp_name: str, exp_idx: int, logger: AuditLogger):
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    folds = GroupKFold(n_splits=5)
    feats = feature_cols(train_df)

    out = {t: [] for t in TARGETS}
    oof = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    for t in TARGETS:
        y_all = train_df[t].values.astype(float)
        for fold_i, (tr_idx, va_idx) in enumerate(folds.split(train_df, y_all, groups), start=1):
            logger.start_target_fold(exp_idx, exp_name, t, fold_i, 5)
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)
            model = mk_pipe(model_bank(t))
            model.fit(tr[feats], y_tr)
            pred = np.maximum(model.predict(va[feats]), 0.0)
            sc = float(r2_score(va[t].values, pred))
            out[t].append(sc)
            oof[t][va_idx] = pred
            logger.log_fold_result(exp_name, t, fold_i, sc)
            print(f"[TARGET] {t} | [FOLD] {fold_i}/5 | [STATUS] completed | r2={sc:.4f}")

    rep = {
        t: {
            "mean": float(np.mean(out[t])),
            "std": float(np.std(out[t])),
            "folds": [float(x) for x in out[t]],
        }
        for t in TARGETS
    }
    rep["mean_cv"] = float(np.mean([rep[t]["mean"] for t in TARGETS]))
    return rep, oof, feats


def run_cv_stack_drp(train_df: pd.DataFrame, exp_name: str, exp_idx: int, logger: AuditLogger):
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    folds = GroupKFold(n_splits=5)
    feats = feature_cols(train_df)

    fold_scores = {t: [] for t in TARGETS}
    oof = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    for t in ["Total Alkalinity", "Electrical Conductance"]:
        y_all = train_df[t].values.astype(float)
        for fold_i, (tr_idx, va_idx) in enumerate(folds.split(train_df, y_all, groups), start=1):
            logger.start_target_fold(exp_idx, exp_name, t, fold_i, 5)
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)

            et = mk_pipe(ExtraTreesRegressor(n_estimators=1800, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
            rf = mk_pipe(RandomForestRegressor(n_estimators=1200, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
            hgb = mk_pipe(HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=6, min_samples_leaf=20, random_state=42))
            et.fit(tr[feats], y_tr)
            rf.fit(tr[feats], y_tr)
            hgb.fit(tr[feats], y_tr)

            tr_meta = pd.DataFrame({
                "pred_ET": et.predict(tr[feats]),
                "pred_RF": rf.predict(tr[feats]),
                "pred_HGB": hgb.predict(tr[feats]),
                "hydro_cluster": tr["hydro_cluster"].values,
            })
            va_meta = pd.DataFrame({
                "pred_ET": et.predict(va[feats]),
                "pred_RF": rf.predict(va[feats]),
                "pred_HGB": hgb.predict(va[feats]),
                "hydro_cluster": va["hydro_cluster"].values,
            })
            meta = mk_meta("elasticnet")
            meta.fit(tr_meta, y_tr)
            pred = np.maximum(meta.predict(va_meta), 0.0)
            sc = float(r2_score(va[t].values, pred))
            fold_scores[t].append(sc)
            oof[t][va_idx] = pred
            logger.log_fold_result(exp_name, t, fold_i, sc)
            print(f"[TARGET] {t} | [FOLD] {fold_i}/5 | [STATUS] completed | r2={sc:.4f}")

    # DRP two-stage
    t = "Dissolved Reactive Phosphorus"
    y_all = train_df[t].values.astype(float)
    for fold_i, (tr_idx, va_idx) in enumerate(folds.split(train_df, y_all, groups), start=1):
        logger.start_target_fold(exp_idx, exp_name, t, fold_i, 5)
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        y_tr = tr[t].values.astype(float)

        q = float(np.nanquantile(y_tr, 0.75))
        y_bin = (y_tr >= q).astype(int)
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
        pred = np.maximum(0.6 * reg_pred + 0.4 * (p_va * reg_pred), 0.0)
        sc = float(r2_score(va[t].values, pred))
        fold_scores[t].append(sc)
        oof[t][va_idx] = pred
        logger.log_fold_result(exp_name, t, fold_i, sc)
        print(f"[TARGET] {t} | [FOLD] {fold_i}/5 | [STATUS] completed | r2={sc:.4f}")

    rep = {
        tt: {
            "mean": float(np.mean(fold_scores[tt])),
            "std": float(np.std(fold_scores[tt])),
            "folds": [float(x) for x in fold_scores[tt]],
        }
        for tt in TARGETS
    }
    rep["mean_cv"] = float(np.mean([rep[tt]["mean"] for tt in TARGETS]))
    return rep, oof, feats


def fit_predict_valid_et(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    feats = feature_cols(train_df)
    out = {}
    for t in TARGETS:
        lo, hi = winsor_fit(train_df[t].values)
        y = winsor_apply(train_df[t].values, lo, hi)
        m = mk_pipe(model_bank(t))
        m.fit(train_df[feats], y)
        out[t] = np.maximum(m.predict(valid_df[feats]), 0.0)
    return out


def fit_predict_valid_stack_drp(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    feats = feature_cols(train_df)
    out = {}

    for t in ["Total Alkalinity", "Electrical Conductance"]:
        lo, hi = winsor_fit(train_df[t].values)
        y = winsor_apply(train_df[t].values, lo, hi)

        et = mk_pipe(ExtraTreesRegressor(n_estimators=1800, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        rf = mk_pipe(RandomForestRegressor(n_estimators=1200, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        hgb = mk_pipe(HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=6, min_samples_leaf=20, random_state=42))
        et.fit(train_df[feats], y)
        rf.fit(train_df[feats], y)
        hgb.fit(train_df[feats], y)

        tr_meta = pd.DataFrame({
            "pred_ET": et.predict(train_df[feats]),
            "pred_RF": rf.predict(train_df[feats]),
            "pred_HGB": hgb.predict(train_df[feats]),
            "hydro_cluster": train_df["hydro_cluster"].values,
        })
        va_meta = pd.DataFrame({
            "pred_ET": et.predict(valid_df[feats]),
            "pred_RF": rf.predict(valid_df[feats]),
            "pred_HGB": hgb.predict(valid_df[feats]),
            "hydro_cluster": valid_df["hydro_cluster"].values,
        })
        meta = mk_meta("elasticnet")
        meta.fit(tr_meta, y)
        out[t] = np.maximum(meta.predict(va_meta), 0.0)

    t = "Dissolved Reactive Phosphorus"
    y = train_df[t].values.astype(float)
    q = float(np.nanquantile(y, 0.75))
    y_bin = (y >= q).astype(int)
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=5, min_samples_leaf=25, random_state=42)),
    ])
    clf.fit(train_df[feats], y_bin)
    p_tr = clf.predict_proba(train_df[feats])[:, 1]
    p_va = clf.predict_proba(valid_df[feats])[:, 1]

    tr2 = train_df[feats].copy()
    va2 = valid_df[feats].copy()
    tr2["P_high"] = p_tr
    va2["P_high"] = p_va

    lo, hi = winsor_fit(y)
    y_w = winsor_apply(y, lo, hi)
    reg = mk_pipe(ExtraTreesRegressor(n_estimators=2600, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
    reg.fit(tr2, y_w)
    reg_pred = np.maximum(reg.predict(va2), 0.0)
    out[t] = np.maximum(0.6 * reg_pred + 0.4 * (p_va * reg_pred), 0.0)

    return out


def write_feature_importance(train_df: pd.DataFrame, feats: list[str], exp_dir: Path):
    rows = []
    for t in TARGETS:
        y = train_df[t].values.astype(float)
        lo, hi = winsor_fit(y)
        y_w = winsor_apply(y, lo, hi)
        model = ExtraTreesRegressor(n_estimators=1200, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
        pipe = mk_pipe(model)
        pipe.fit(train_df[feats], y_w)
        imp = pipe.named_steps["m"].feature_importances_
        rank_idx = np.argsort(imp)[::-1][:20]
        for r, idx in enumerate(rank_idx, start=1):
            rows.append({
                "target": t,
                "feature": feats[idx],
                "importance": float(imp[idx]),
                "rank": r,
            })
    pd.DataFrame(rows).to_csv(exp_dir / "feature_importance.csv", index=False)


def write_submission(template_raw: pd.DataFrame, preds: dict, out_path: Path):
    sub = template_raw.copy()
    for t in TARGETS:
        sub[t] = np.maximum(np.nan_to_num(preds[t], nan=0.0), 0.0)
    sub.to_csv(out_path, index=False)


def run_single_experiment(
    exp_name: str,
    exp_idx: int,
    total_experiments: int,
    train_base: pd.DataFrame,
    valid_base: pd.DataFrame,
    template_raw: pd.DataFrame,
    blocks: list[str],
    stack_mode: bool,
    logger: AuditLogger,
    sprint_dir: Path,
):
    skipped = []
    exp_dir = sprint_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.start_experiment(exp_idx, exp_name)

    tr = apply_feature_blocks(train_base, blocks, skipped)
    va = apply_feature_blocks(valid_base, blocks, skipped)

    config = {
        "experiment": exp_name,
        "index": exp_idx,
        "total_experiments": total_experiments,
        "blocks": blocks,
        "stack_mode": bool(stack_mode),
        "rules": {
            "grouping_only_basin_id": True,
            "no_basin_as_feature": True,
            "no_station_as_feature": True,
            "no_target_encoding": True,
            "no_knn_encoding": True,
            "meta_models_allowed": ["Ridge", "ElasticNet"],
        },
        "started_at": ts(),
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    feats = feature_cols(tr)
    with open(exp_dir / "features_used.json", "w", encoding="utf-8") as f:
        json.dump({"n_features": len(feats), "features": feats, "skipped_features": skipped}, f, indent=2)

    if stack_mode:
        cv_rep, _, used_feats = run_cv_stack_drp(tr, exp_name, exp_idx, logger)
        preds = fit_predict_valid_stack_drp(tr, va)
        model_summary = {
            "family": ["ExtraTrees", "RandomForest", "HistGradientBoosting"],
            "meta_model": "ElasticNet",
            "drp_two_stage": True,
        }
    else:
        cv_rep, _, used_feats = run_cv_et(tr, exp_name, exp_idx, logger)
        preds = fit_predict_valid_et(tr, va)
        model_summary = {
            "family": ["ExtraTrees", "RandomForest", "HistGradientBoosting"],
            "meta_model": None,
            "drp_two_stage": False,
        }

    with open(exp_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_rep, f, indent=2)
    with open(exp_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2)

    write_feature_importance(tr, used_feats, exp_dir)

    sub_path = exp_dir / f"submission_{exp_name}.csv"
    write_submission(template_raw, preds, sub_path)

    logger.complete_experiment(exp_idx, exp_name, cv_rep["mean_cv"])
    return {
        "experiment": exp_name,
        "TA_mean_cv": cv_rep["Total Alkalinity"]["mean"],
        "EC_mean_cv": cv_rep["Electrical Conductance"]["mean"],
        "DRP_mean_cv": cv_rep["Dissolved Reactive Phosphorus"]["mean"],
        "TA_std": cv_rep["Total Alkalinity"]["std"],
        "EC_std": cv_rep["Electrical Conductance"]["std"],
        "DRP_std": cv_rep["Dissolved Reactive Phosphorus"]["std"],
        "mean_cv": cv_rep["mean_cv"],
        "blocks": ",".join(blocks),
        "submission_file": str(sub_path),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    sprint_dir = root / "experiments" / "data_signal_sprint"
    sprint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[START] {ts()}")

    # Methodology note (traceability)
    methodology = sprint_dir / "landuse_feature_methodology.md"
    methodology.write_text(
        "# Land Use Feature Methodology\n"
        "- Landcover classes are converted to masks: cropland=40, urban=50, natural=all non-cropland/non-urban valid classes.\n"
        "- Fractions are computed with BallTree (haversine) neighborhood proportions at 1km/5km/10km radii.\n"
        "- Features are unsupervised and use only static covariates, not target values.\n",
        encoding="utf-8",
    )

    train_base, valid_base, template_raw = load_base_data(root)

    logger = AuditLogger(sprint_dir, total_experiments=6, benchmark_mean=0.320)

    # Internal baseline probe for selecting best combo blocks (not counted as one of 6)
    baseline_probe = run_cv_et(train_base, "BASELINE_PROBE", 0, logger)[0]["mean_cv"]

    experiments = [
        ("EXP_1_BASE_PLUS_RAIN", ["rain"], False),
        ("EXP_2_BASE_PLUS_RAIN_MOISTURE", ["rain", "moisture"], False),
        ("EXP_3_BASE_PLUS_LANDUSE", ["landuse"], False),
        ("EXP_4_BASE_PLUS_UPSTREAM", ["upstream"], False),
    ]

    summary_rows = []
    individual_scores = {}

    for idx, (name, blocks, stack_mode) in enumerate(experiments, start=1):
        try:
            row = run_single_experiment(
                exp_name=name,
                exp_idx=idx,
                total_experiments=6,
                train_base=train_base,
                valid_base=valid_base,
                template_raw=template_raw,
                blocks=blocks,
                stack_mode=stack_mode,
                logger=logger,
                sprint_dir=sprint_dir,
            )
            summary_rows.append(row)
            individual_scores[name] = row["mean_cv"]
        except Exception as e:
            logger.fail_experiment(idx, name, e)
            continue

    improved_blocks = []
    for name, blocks, _ in experiments:
        if name in individual_scores and individual_scores[name] > baseline_probe:
            improved_blocks.extend(blocks)
    improved_blocks = sorted(set(improved_blocks))
    if len(improved_blocks) == 0:
        improved_blocks = ["rain"]

    # EXP 5
    try:
        row5 = run_single_experiment(
            exp_name="EXP_5_BASE_PLUS_BEST_COMBO",
            exp_idx=5,
            total_experiments=6,
            train_base=train_base,
            valid_base=valid_base,
            template_raw=template_raw,
            blocks=improved_blocks,
            stack_mode=False,
            logger=logger,
            sprint_dir=sprint_dir,
        )
        summary_rows.append(row5)
    except Exception as e:
        logger.fail_experiment(5, "EXP_5_BASE_PLUS_BEST_COMBO", e)

    # EXP 6
    try:
        row6 = run_single_experiment(
            exp_name="EXP_6_FINAL_STACK_DRP",
            exp_idx=6,
            total_experiments=6,
            train_base=train_base,
            valid_base=valid_base,
            template_raw=template_raw,
            blocks=improved_blocks,
            stack_mode=True,
            logger=logger,
            sprint_dir=sprint_dir,
        )
        summary_rows.append(row6)
    except Exception as e:
        logger.fail_experiment(6, "EXP_6_FINAL_STACK_DRP", e)

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) == 0:
        logger.finalize()
        raise RuntimeError("All experiments failed. Check errors.log")

    summary_df = summary_df.sort_values("mean_cv", ascending=False).reset_index(drop=True)
    summary_path = sprint_dir / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best = summary_df.iloc[0]
    best_sub = Path(best["submission_file"])
    final_sub = root / "submissions" / "submission_DATA_SIGNAL_SPRINT_BEST.csv"
    best_df = pd.read_csv(best_sub)
    best_df.to_csv(final_sub, index=False)

    # final leaderboard print
    print("\nFINAL LEADERBOARD")
    print("EXPERIMENT | TA | EC | DRP | MEAN")
    for _, r in summary_df.iterrows():
        print(f"{r['experiment']} | {r['TA_mean_cv']:.4f} | {r['EC_mean_cv']:.4f} | {r['DRP_mean_cv']:.4f} | {r['mean_cv']:.4f}")

    # top global feature importances from best experiment
    fi_path = sprint_dir / best["experiment"] / "feature_importance.csv"
    top_imp = []
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        agg = fi.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(20)
        top_imp = agg.to_dict(orient="records")

    benchmark = 0.320
    report = {
        "benchmark_mean_cv": benchmark,
        "best_experiment": best["experiment"],
        "best_mean_cv": float(best["mean_cv"]),
        "recommended_submission": str(final_sub),
        "feature_blocks_improved": [
            {
                "experiment": r["experiment"],
                "blocks": r["blocks"],
                "mean_cv": float(r["mean_cv"]),
            }
            for _, r in summary_df.iterrows()
            if float(r["mean_cv"]) > benchmark
        ],
        "feature_blocks_degraded": [
            {
                "experiment": r["experiment"],
                "blocks": r["blocks"],
                "mean_cv": float(r["mean_cv"]),
            }
            for _, r in summary_df.iterrows()
            if float(r["mean_cv"]) <= benchmark
        ],
        "top_global_feature_importances": top_imp,
        "next_recommendation": "Focus on data quality and temporal rainfall fidelity (true daily antecedent windows) before introducing new model families.",
    }
    with open(sprint_dir / "data_signal_sprint_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.finalize()


if __name__ == "__main__":
    main()
