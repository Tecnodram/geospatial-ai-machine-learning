#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features.data_signal_features import (
    add_landuse_pressure,
    add_rainfall_antecedent_package,
    add_upstream_pressure,
)

LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"
KEYS = [LAT_COL, LON_COL, DATE_COL]
TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
TARGET_SHORT = {
    "Total Alkalinity": "TA",
    "Electrical Conductance": "EC",
    "Dissolved Reactive Phosphorus": "DRP",
}


def ts() -> str:
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
        self.best_score = float(benchmark_mean)
        self.best_experiment = "EXP_STACKING_V1"

        if not self.running_results_path.exists():
            pd.DataFrame(columns=["experiment", "target", "fold", "r2", "timestamp"]).to_csv(
                self.running_results_path, index=False
            )

        self.write_progress(
            status="running",
            current_experiment="",
            experiment_index=0,
            target="",
            fold=0,
        )

    def write_progress(
        self,
        status: str,
        current_experiment: str,
        experiment_index: int,
        target: str,
        fold: int,
    ):
        payload = {
            "status": status,
            "current_experiment": current_experiment,
            "experiment_index": int(experiment_index),
            "total_experiments": int(self.total_experiments),
            "target": target,
            "fold": int(fold),
            "started_at": self.started_at,
            "updated_at": ts(),
            "best_score_so_far": float(self.best_score),
            "best_experiment": self.best_experiment,
        }
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def start_experiment(self, exp_name: str, exp_idx: int):
        print("[EXPERIMENT START]")
        print(ts())
        print(exp_name)
        self.write_progress(
            status="running",
            current_experiment=exp_name,
            experiment_index=exp_idx,
            target="",
            fold=0,
        )

    def start_fold(self, exp_name: str, exp_idx: int, target: str, fold: int, total_folds: int):
        print("[TARGET]")
        print(TARGET_SHORT[target])
        print("[FOLD]")
        print(f"{fold}/{total_folds}")
        self.write_progress(
            status="running",
            current_experiment=exp_name,
            experiment_index=exp_idx,
            target=TARGET_SHORT[target],
            fold=fold,
        )

    def log_fold(self, experiment: str, target: str, fold: int, score: float):
        print("[SCORE PARTIAL]")
        print(f"r2={score:.6f}")
        pd.DataFrame(
            [
                {
                    "experiment": experiment,
                    "target": TARGET_SHORT[target],
                    "fold": int(fold),
                    "r2": float(score),
                    "timestamp": ts(),
                }
            ]
        ).to_csv(self.running_results_path, mode="a", index=False, header=False)

    def complete_experiment(self, exp_name: str, exp_idx: int, mean_cv: float):
        if float(mean_cv) > self.best_score:
            self.best_score = float(mean_cv)
            self.best_experiment = exp_name
        self.write_progress(
            status="running",
            current_experiment=exp_name,
            experiment_index=exp_idx,
            target="",
            fold=0,
        )

    def fail_experiment(self, exp_name: str, exp_idx: int):
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts()}] {exp_name}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        self.write_progress(
            status="running",
            current_experiment=exp_name,
            experiment_index=exp_idx,
            target="",
            fold=0,
        )

    def finalize(self):
        self.write_progress(
            status="completed",
            current_experiment=self.best_experiment,
            experiment_index=self.total_experiments,
            target="",
            fold=0,
        )


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
    return train, valid, template_raw


def add_temporal_memory_features(df: pd.DataFrame, skipped: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "chirps_ppt" not in out.columns:
        skipped.append("temporal rainfall features skipped: chirps_ppt missing")
        return out

    out = out.sort_values([LAT_COL, LON_COL, DATE_COL]).copy()
    g_rain = out.groupby([LAT_COL, LON_COL])["chirps_ppt"]

    out["rain_lag_1"] = g_rain.transform(lambda x: x.shift(1))
    out["rain_lag_3"] = g_rain.transform(lambda x: x.shift(3))
    out["rain_lag_7"] = g_rain.transform(lambda x: x.shift(7))
    out["rain_lag_14"] = g_rain.transform(lambda x: x.shift(14))

    out["rain_roll_7"] = g_rain.transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).sum())
    out["rain_roll_30"] = g_rain.transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).sum())

    if "pet" in out.columns:
        g_pet = out.groupby([LAT_COL, LON_COL])["pet"]
        out["pet_roll_30"] = g_pet.transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).sum())
    else:
        skipped.append("pet_roll_30 skipped: pet missing")

    return out


def apply_feature_mode(df: pd.DataFrame, mode: str, skipped: list[str]) -> pd.DataFrame:
    out = df.copy()
    if mode in {"full", "full_temporal", "hybrid", "stack"}:
        out = add_rainfall_antecedent_package(out, skipped)
        out = add_landuse_pressure(out, skipped)
        out = add_upstream_pressure(out, skipped)
    if mode in {"temporal", "full_temporal", "stack"}:
        out = add_temporal_memory_features(out, skipped)
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop_cols = set(TARGETS + [DATE_COL, LAT_COL, LON_COL, "basin_id", "station_id", "station", "station_name", "site_id"])
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def model_for_name(name: str):
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=900,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=900,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=1400, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
    if name == "randomforest":
        return RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown model name: {name}")


def train_predict_single_fold(tr_x: pd.DataFrame, tr_y: np.ndarray, va_x: pd.DataFrame, model_name: str) -> tuple[np.ndarray, object]:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", model_for_name(model_name)),
    ])
    pipe.fit(tr_x, tr_y)
    pred = np.maximum(pipe.predict(va_x), 0.0)
    return pred, pipe


def run_cv_single(
    train_df: pd.DataFrame,
    features: list[str],
    exp_name: str,
    exp_idx: int,
    logger: AuditLogger,
    model_name: str,
):
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    gkf = GroupKFold(n_splits=5)

    scores = {t: [] for t in TARGETS}
    oof = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    for t in TARGETS:
        y_all = train_df[t].astype(float).values
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_all, groups), start=1):
            logger.start_fold(exp_name, exp_idx, t, fold_i, 5)
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)
            pred, _ = train_predict_single_fold(tr[features], y_tr, va[features], model_name)
            sc = float(r2_score(va[t].values, pred))
            scores[t].append(sc)
            oof[t][va_idx] = pred
            logger.log_fold(exp_name, t, fold_i, sc)

    rep = {
        t: {
            "mean": float(np.mean(scores[t])),
            "std": float(np.std(scores[t])),
            "folds": [float(x) for x in scores[t]],
        }
        for t in TARGETS
    }
    rep["mean_cv"] = float(np.mean([rep[t]["mean"] for t in TARGETS]))
    return rep, oof


def run_cv_hybrid(
    train_df: pd.DataFrame,
    features: list[str],
    exp_name: str,
    exp_idx: int,
    logger: AuditLogger,
):
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    gkf = GroupKFold(n_splits=5)
    base_names = ["extratrees", "randomforest", "lightgbm", "xgboost"]

    scores = {t: [] for t in TARGETS}
    oof = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    for t in TARGETS:
        y_all = train_df[t].astype(float).values
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_all, groups), start=1):
            logger.start_fold(exp_name, exp_idx, t, fold_i, 5)
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)

            preds = []
            for base_name in base_names:
                p, _ = train_predict_single_fold(tr[features], y_tr, va[features], base_name)
                preds.append(p)
            pred = np.maximum(np.mean(np.column_stack(preds), axis=1), 0.0)

            sc = float(r2_score(va[t].values, pred))
            scores[t].append(sc)
            oof[t][va_idx] = pred
            logger.log_fold(exp_name, t, fold_i, sc)

    rep = {
        t: {
            "mean": float(np.mean(scores[t])),
            "std": float(np.std(scores[t])),
            "folds": [float(x) for x in scores[t]],
        }
        for t in TARGETS
    }
    rep["mean_cv"] = float(np.mean([rep[t]["mean"] for t in TARGETS]))
    return rep, oof


def run_cv_stack(
    train_df: pd.DataFrame,
    features: list[str],
    exp_name: str,
    exp_idx: int,
    logger: AuditLogger,
):
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    gkf = GroupKFold(n_splits=5)
    base_names = ["extratrees", "randomforest", "lightgbm", "xgboost"]

    scores = {t: [] for t in TARGETS}
    oof = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    for t in TARGETS:
        y_all = train_df[t].astype(float).values
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_all, groups), start=1):
            logger.start_fold(exp_name, exp_idx, t, fold_i, 5)
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)

            pred_tr_list = []
            pred_va_list = []
            for base_name in base_names:
                base_pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("m", model_for_name(base_name)),
                ])
                base_pipe.fit(tr[features], y_tr)
                pred_tr_list.append(np.maximum(base_pipe.predict(tr[features]), 0.0))
                pred_va_list.append(np.maximum(base_pipe.predict(va[features]), 0.0))

            meta_tr = pd.DataFrame(
                {
                    "pred_et": pred_tr_list[0],
                    "pred_rf": pred_tr_list[1],
                    "pred_lgbm": pred_tr_list[2],
                    "pred_xgb": pred_tr_list[3],
                }
            )
            meta_va = pd.DataFrame(
                {
                    "pred_et": pred_va_list[0],
                    "pred_rf": pred_va_list[1],
                    "pred_lgbm": pred_va_list[2],
                    "pred_xgb": pred_va_list[3],
                }
            )

            meta = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("m", ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=10000, random_state=42)),
                ]
            )
            meta.fit(meta_tr, y_tr)
            pred = np.maximum(meta.predict(meta_va), 0.0)
            sc = float(r2_score(va[t].values, pred))
            scores[t].append(sc)
            oof[t][va_idx] = pred
            logger.log_fold(exp_name, t, fold_i, sc)

    rep = {
        t: {
            "mean": float(np.mean(scores[t])),
            "std": float(np.std(scores[t])),
            "folds": [float(x) for x in scores[t]],
        }
        for t in TARGETS
    }
    rep["mean_cv"] = float(np.mean([rep[t]["mean"] for t in TARGETS]))
    return rep, oof


def fit_predict_single(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], model_name: str):
    preds = {}
    for t in TARGETS:
        lo, hi = winsor_fit(train_df[t].values)
        y = winsor_apply(train_df[t].values, lo, hi)
        p, _ = train_predict_single_fold(train_df[features], y, valid_df[features], model_name)
        preds[t] = p
    return preds


def fit_predict_hybrid(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str]):
    preds = {}
    base_names = ["extratrees", "randomforest", "lightgbm", "xgboost"]
    for t in TARGETS:
        lo, hi = winsor_fit(train_df[t].values)
        y = winsor_apply(train_df[t].values, lo, hi)
        out = []
        for base_name in base_names:
            p, _ = train_predict_single_fold(train_df[features], y, valid_df[features], base_name)
            out.append(p)
        preds[t] = np.maximum(np.mean(np.column_stack(out), axis=1), 0.0)
    return preds


def fit_predict_stack(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str]):
    preds = {}
    base_names = ["extratrees", "randomforest", "lightgbm", "xgboost"]
    for t in TARGETS:
        lo, hi = winsor_fit(train_df[t].values)
        y = winsor_apply(train_df[t].values, lo, hi)

        pred_tr_list = []
        pred_va_list = []
        for base_name in base_names:
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", model_for_name(base_name)),
            ])
            pipe.fit(train_df[features], y)
            pred_tr_list.append(np.maximum(pipe.predict(train_df[features]), 0.0))
            pred_va_list.append(np.maximum(pipe.predict(valid_df[features]), 0.0))

        meta_tr = pd.DataFrame(
            {
                "pred_et": pred_tr_list[0],
                "pred_rf": pred_tr_list[1],
                "pred_lgbm": pred_tr_list[2],
                "pred_xgb": pred_tr_list[3],
            }
        )
        meta_va = pd.DataFrame(
            {
                "pred_et": pred_va_list[0],
                "pred_rf": pred_va_list[1],
                "pred_lgbm": pred_va_list[2],
                "pred_xgb": pred_va_list[3],
            }
        )
        meta = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("m", ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=10000, random_state=42)),
            ]
        )
        meta.fit(meta_tr, y)
        preds[t] = np.maximum(meta.predict(meta_va), 0.0)
    return preds


def write_feature_importance(train_df: pd.DataFrame, features: list[str], exp_dir: Path):
    rows = []
    for t in TARGETS:
        y = train_df[t].astype(float).values
        lo, hi = winsor_fit(y)
        y_w = winsor_apply(y, lo, hi)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            (
                "m",
                ExtraTreesRegressor(
                    n_estimators=1200,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ])
        pipe.fit(train_df[features], y_w)
        imp = pipe.named_steps["m"].feature_importances_
        top_idx = np.argsort(imp)[::-1][:50]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "target": TARGET_SHORT[t],
                    "feature": features[idx],
                    "importance": float(imp[idx]),
                    "rank": rank,
                    "method": "ExtraTrees proxy",
                }
            )
    pd.DataFrame(rows).to_csv(exp_dir / "feature_importance.csv", index=False)


def save_submission(template_raw: pd.DataFrame, preds: dict, out_path: Path):
    sub = template_raw.copy()
    for t in TARGETS:
        sub[t] = np.maximum(np.nan_to_num(preds[t], nan=0.0), 0.0)
    sub.to_csv(out_path, index=False)


def run_experiment(
    train_base: pd.DataFrame,
    valid_base: pd.DataFrame,
    template_raw: pd.DataFrame,
    exp_name: str,
    exp_idx: int,
    total_experiments: int,
    feature_mode: str,
    runner: str,
    logger: AuditLogger,
    sprint_dir: Path,
):
    exp_dir = sprint_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    logger.start_experiment(exp_name, exp_idx)

    tr = apply_feature_mode(train_base, feature_mode, skipped)
    va = apply_feature_mode(valid_base, feature_mode, skipped)
    feats = feature_cols(tr)

    config = {
        "experiment": exp_name,
        "index": exp_idx,
        "total_experiments": total_experiments,
        "feature_mode": feature_mode,
        "runner": runner,
        "rules": {
            "basin_id_grouping_only": True,
            "basin_id_not_feature": True,
            "station_identifiers_forbidden": True,
            "target_encoding": False,
            "fold_leakage": False,
            "dataset_structure_changes": "none_except_temporal_memory_features",
        },
        "started_at": ts(),
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(exp_dir / "features_used.json", "w", encoding="utf-8") as f:
        json.dump({"n_features": len(feats), "features": feats, "skipped_notes": skipped}, f, indent=2)

    if runner == "single_lightgbm":
        cv_rep, _ = run_cv_single(tr, feats, exp_name, exp_idx, logger, model_name="lightgbm")
        preds = fit_predict_single(tr, va, feats, model_name="lightgbm")
        model_summary = {"models": ["LightGBM"], "ensemble": False, "stacking": False}
    elif runner == "single_xgboost":
        cv_rep, _ = run_cv_single(tr, feats, exp_name, exp_idx, logger, model_name="xgboost")
        preds = fit_predict_single(tr, va, feats, model_name="xgboost")
        model_summary = {"models": ["XGBoost"], "ensemble": False, "stacking": False}
    elif runner == "hybrid_avg":
        cv_rep, _ = run_cv_hybrid(tr, feats, exp_name, exp_idx, logger)
        preds = fit_predict_hybrid(tr, va, feats)
        model_summary = {
            "models": ["ExtraTrees", "RandomForest", "LightGBM", "XGBoost"],
            "ensemble": "simple_average",
            "stacking": False,
        }
    elif runner == "final_stack":
        cv_rep, _ = run_cv_stack(tr, feats, exp_name, exp_idx, logger)
        preds = fit_predict_stack(tr, va, feats)
        model_summary = {
            "models": ["ExtraTrees", "RandomForest", "LightGBM", "XGBoost"],
            "ensemble": "ElasticNet stack",
            "stacking": True,
            "meta_model": "ElasticNet(alpha=0.01,l1_ratio=0.3)",
        }
    else:
        raise ValueError(f"Unknown runner mode: {runner}")

    with open(exp_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_rep, f, indent=2)
    with open(exp_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2)

    write_feature_importance(tr, feats, exp_dir)

    sub_path = exp_dir / f"submission_{exp_name}.csv"
    save_submission(template_raw, preds, sub_path)

    logger.complete_experiment(exp_name, exp_idx, cv_rep["mean_cv"])

    return {
        "experiment": exp_name,
        "TA": float(cv_rep["Total Alkalinity"]["mean"]),
        "EC": float(cv_rep["Electrical Conductance"]["mean"]),
        "DRP": float(cv_rep["Dissolved Reactive Phosphorus"]["mean"]),
        "MEAN": float(cv_rep["mean_cv"]),
        "submission_file": str(sub_path),
        "feature_mode": feature_mode,
        "runner": runner,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    sprint_dir = root / "experiments" / "final_model_sprint"
    sprint_dir.mkdir(parents=True, exist_ok=True)

    train_base, valid_base, template_raw = load_base_data(root)

    experiments = [
        ("EXP1_LIGHTGBM_BASE", "baseline", "single_lightgbm"),
        ("EXP2_XGBOOST_BASE", "baseline", "single_xgboost"),
        ("EXP3_LIGHTGBM_FULL_FEATURES", "full", "single_lightgbm"),
        ("EXP4_XGBOOST_TEMPORAL", "full_temporal", "single_xgboost"),
        ("EXP5_HYBRID_ENSEMBLE", "hybrid", "hybrid_avg"),
        ("EXP6_FINAL_STACK", "stack", "final_stack"),
    ]

    logger = AuditLogger(sprint_dir, total_experiments=len(experiments), benchmark_mean=0.320)

    summary_rows = []
    for idx, (name, feature_mode, runner) in enumerate(experiments, start=1):
        try:
            row = run_experiment(
                train_base=train_base,
                valid_base=valid_base,
                template_raw=template_raw,
                exp_name=name,
                exp_idx=idx,
                total_experiments=len(experiments),
                feature_mode=feature_mode,
                runner=runner,
                logger=logger,
                sprint_dir=sprint_dir,
            )
            summary_rows.append(row)
        except Exception:
            logger.fail_experiment(name, idx)
            continue

    if len(summary_rows) == 0:
        logger.finalize()
        raise RuntimeError("All experiments failed. Check experiments/final_model_sprint/errors.log")

    summary = pd.DataFrame(summary_rows).sort_values("MEAN", ascending=False).reset_index(drop=True)
    summary.to_csv(sprint_dir / "experiment_summary.csv", index=False)

    print("\nEXPERIMENT | TA | EC | DRP | MEAN")
    for _, r in summary.iterrows():
        print(f"{r['experiment']} | {r['TA']:.4f} | {r['EC']:.4f} | {r['DRP']:.4f} | {r['MEAN']:.4f}")

    best = summary.iloc[0]
    benchmark = 0.320
    final_submission_path = root / "submissions" / "submission_FINAL_MODEL_SPRINT_BEST.csv"
    final_saved = False
    if float(best["MEAN"]) > benchmark:
        pd.read_csv(best["submission_file"]).to_csv(final_submission_path, index=False)
        final_saved = True

    report = {
        "sprint": "FINAL_MODEL_EXPLORATION_SPRINT",
        "generated_at": ts(),
        "benchmark_mean_cv": benchmark,
        "best_experiment": str(best["experiment"]),
        "best_mean_cv": float(best["MEAN"]),
        "best_targets": {
            "TA": str(summary.loc[summary["TA"].idxmax(), "experiment"]),
            "EC": str(summary.loc[summary["EC"].idxmax(), "experiment"]),
            "DRP": str(summary.loc[summary["DRP"].idxmax(), "experiment"]),
        },
        "submission_saved": final_saved,
        "submission_path": str(final_submission_path) if final_saved else None,
        "leaderboard": summary.to_dict(orient="records"),
        "audit_files": {
            "progress": str(sprint_dir / "progress.json"),
            "running_results": str(sprint_dir / "running_results.csv"),
            "errors": str(sprint_dir / "errors.log"),
            "summary": str(sprint_dir / "experiment_summary.csv"),
        },
    }
    with open(sprint_dir / "final_model_sprint_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.finalize()


if __name__ == "__main__":
    main()
