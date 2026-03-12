#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

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
    def __init__(self, root: Path, total_experiments: int):
        self.root = root
        self.total_experiments = int(total_experiments)
        self.progress_path = root / "progress.json"
        self.running_results_path = root / "running_results.csv"
        self.errors_path = root / "errors.log"
        self.started_at = ts()

        if not self.running_results_path.exists():
            pd.DataFrame(columns=["experiment", "target", "fold", "r2", "timestamp"]).to_csv(
                self.running_results_path, index=False
            )

        self.write_progress("running", "", 0, "", 0)

    def write_progress(self, status: str, current_experiment: str, experiment_index: int, target: str, fold: int):
        payload = {
            "status": status,
            "current_experiment": current_experiment,
            "experiment_index": int(experiment_index),
            "total_experiments": self.total_experiments,
            "target": target,
            "fold": int(fold),
            "started_at": self.started_at,
            "updated_at": ts(),
        }
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def start_experiment(self, exp_name: str, exp_idx: int):
        print("[EXPERIMENT]")
        print(exp_name)
        self.write_progress("running", exp_name, exp_idx, "", 0)

    def start_fold(self, exp_name: str, exp_idx: int, target: str, fold: int, total_folds: int):
        print("[TARGET]")
        print(TARGET_SHORT[target])
        print("[FOLD]")
        print(f"{fold}/{total_folds}")
        self.write_progress("running", exp_name, exp_idx, TARGET_SHORT[target], fold)

    def log_result(self, exp_name: str, target: str, fold: int, score: float):
        print("[R2 PARTIAL]")
        print(f"{score:.6f}")
        pd.DataFrame([
            {
                "experiment": exp_name,
                "target": TARGET_SHORT[target],
                "fold": int(fold),
                "r2": float(score),
                "timestamp": ts(),
            }
        ]).to_csv(self.running_results_path, mode="a", index=False, header=False)

    def fail(self, exp_name: str, exp_idx: int):
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts()}] {exp_name}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        self.write_progress("running", exp_name, exp_idx, "", 0)

    def finalize(self):
        self.write_progress("completed", "EXP3_CORRECTED_MODEL", self.total_experiments, "", 0)


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


def add_temporal_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "chirps_ppt" not in out.columns:
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
    return out


def add_validated_feature_stack(df: pd.DataFrame) -> pd.DataFrame:
    out = add_rainfall_antecedent_package(df.copy(), skipped=[])
    out = add_landuse_pressure(out, skipped=[])
    out = add_upstream_pressure(out, skipped=[])
    out = add_temporal_memory_features(out)
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop_cols = set(
        TARGETS + [DATE_COL, LAT_COL, LON_COL, "basin_id", "station_id", "station", "station_name", "site_id"]
    )
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def make_base_model() -> Pipeline:
    model = ExtraTreesRegressor(
        n_estimators=1600,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", model)])


def make_residual_model() -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", model)])


def build_spatial_features(df: pd.DataFrame, center_lat: float, center_lon: float) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    lat = pd.to_numeric(df[LAT_COL], errors="coerce")
    lon = pd.to_numeric(df[LON_COL], errors="coerce")
    x["lat"] = lat
    x["lon"] = lon
    x["lat2"] = lat * lat
    x["lon2"] = lon * lon
    x["lat_lon"] = lat * lon
    x["distance_to_center"] = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
    return x


def cross_validate_spatial_residual(train_df: pd.DataFrame, logger: AuditLogger):
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    gkf = GroupKFold(n_splits=5)
    tab_feats = feature_cols(train_df)

    oof_base = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}
    oof_resid = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}
    oof_final = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    base_scores = {t: [] for t in TARGETS}
    residual_scores = {t: [] for t in TARGETS}
    corrected_scores = {t: [] for t in TARGETS}

    # EXP1 + EXP2 + EXP3 in one leakage-safe loop
    for t in TARGETS:
        y_all = train_df[t].astype(float).values
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_all, groups), start=1):
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]

            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)
            y_va = va[t].values.astype(float)

            # EXP1_BASE_MODEL
            logger.start_fold("EXP1_BASE_MODEL", 1, t, fold_i, 5)
            base_model = make_base_model()
            base_model.fit(tr[tab_feats], y_tr)
            pred_base_va = np.maximum(base_model.predict(va[tab_feats]), 0.0)
            sc_base = float(r2_score(y_va, pred_base_va))
            logger.log_result("EXP1_BASE_MODEL", t, fold_i, sc_base)

            # EXP2_RESIDUAL_MODEL
            pred_base_tr = np.maximum(base_model.predict(tr[tab_feats]), 0.0)
            resid_tr = tr[t].values.astype(float) - pred_base_tr

            center_lat = float(pd.to_numeric(tr[LAT_COL], errors="coerce").mean())
            center_lon = float(pd.to_numeric(tr[LON_COL], errors="coerce").mean())
            x_sp_tr = build_spatial_features(tr, center_lat, center_lon)
            x_sp_va = build_spatial_features(va, center_lat, center_lon)

            logger.start_fold("EXP2_RESIDUAL_MODEL", 2, t, fold_i, 5)
            resid_model = make_residual_model()
            resid_model.fit(x_sp_tr, resid_tr)
            pred_resid_va = resid_model.predict(x_sp_va)
            resid_true_va = y_va - pred_base_va
            sc_resid = float(r2_score(resid_true_va, pred_resid_va))
            logger.log_result("EXP2_RESIDUAL_MODEL", t, fold_i, sc_resid)

            # EXP3_CORRECTED_MODEL
            logger.start_fold("EXP3_CORRECTED_MODEL", 3, t, fold_i, 5)
            pred_final_va = np.maximum(pred_base_va + pred_resid_va, 0.0)
            sc_final = float(r2_score(y_va, pred_final_va))
            logger.log_result("EXP3_CORRECTED_MODEL", t, fold_i, sc_final)

            oof_base[t][va_idx] = pred_base_va
            oof_resid[t][va_idx] = pred_resid_va
            oof_final[t][va_idx] = pred_final_va
            base_scores[t].append(sc_base)
            residual_scores[t].append(sc_resid)
            corrected_scores[t].append(sc_final)

    def pack(scores: dict[str, list[float]]):
        out = {
            t: {
                "mean": float(np.mean(scores[t])),
                "std": float(np.std(scores[t])),
                "folds": [float(x) for x in scores[t]],
            }
            for t in TARGETS
        }
        out["mean_cv"] = float(np.mean([out[t]["mean"] for t in TARGETS]))
        return out

    return {
        "base": pack(base_scores),
        "residual": pack(residual_scores),
        "corrected": pack(corrected_scores),
        "oof_base": oof_base,
        "oof_residual": oof_resid,
        "oof_final": oof_final,
        "tabular_features": tab_feats,
    }


def fit_predict_valid(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    tab_feats = feature_cols(train_df)
    preds_base = {}
    preds_res = {}
    preds_final = {}

    center_lat = float(pd.to_numeric(train_df[LAT_COL], errors="coerce").mean())
    center_lon = float(pd.to_numeric(train_df[LON_COL], errors="coerce").mean())

    for t in TARGETS:
        lo, hi = winsor_fit(train_df[t].values)
        y_tr = winsor_apply(train_df[t].values, lo, hi)

        base_model = make_base_model()
        base_model.fit(train_df[tab_feats], y_tr)
        pred_base_tr = np.maximum(base_model.predict(train_df[tab_feats]), 0.0)
        pred_base_va = np.maximum(base_model.predict(valid_df[tab_feats]), 0.0)

        resid_target = train_df[t].values.astype(float) - pred_base_tr
        x_sp_tr = build_spatial_features(train_df, center_lat, center_lon)
        x_sp_va = build_spatial_features(valid_df, center_lat, center_lon)

        resid_model = make_residual_model()
        resid_model.fit(x_sp_tr, resid_target)
        pred_res_va = resid_model.predict(x_sp_va)

        preds_base[t] = np.maximum(pred_base_va, 0.0)
        preds_res[t] = pred_res_va
        preds_final[t] = np.maximum(pred_base_va + pred_res_va, 0.0)

    return preds_base, preds_res, preds_final


def main():
    root = Path(__file__).resolve().parents[1]
    sprint_dir = root / "experiments" / "spatial_residual_sprint"
    sprint_dir.mkdir(parents=True, exist_ok=True)

    logger = AuditLogger(sprint_dir, total_experiments=3)

    try:
        train_raw, valid_raw, template_raw = load_base_data(root)
        train = add_validated_feature_stack(train_raw)
        valid = add_validated_feature_stack(valid_raw)

        logger.start_experiment("EXP1_BASE_MODEL", 1)
        logger.start_experiment("EXP2_RESIDUAL_MODEL", 2)
        logger.start_experiment("EXP3_CORRECTED_MODEL", 3)

        cv = cross_validate_spatial_residual(train, logger)

        # Save fold-level OOF predictions
        oof_rows = []
        for t in TARGETS:
            oof_rows.append(
                pd.DataFrame(
                    {
                        "target": TARGET_SHORT[t],
                        "pred_base": cv["oof_base"][t],
                        "pred_residual": cv["oof_residual"][t],
                        "pred_final": cv["oof_final"][t],
                        "y_true": train[t].values.astype(float),
                    }
                )
            )
        pd.concat(oof_rows, ignore_index=True).to_csv(sprint_dir / "oof_predictions.csv", index=False)

        base_mean = float(cv["base"]["mean_cv"])
        corrected_mean = float(cv["corrected"]["mean_cv"])

        leaderboard = pd.DataFrame(
            [
                {
                    "model": "BASE_MODEL",
                    "TA": cv["base"]["Total Alkalinity"]["mean"],
                    "EC": cv["base"]["Electrical Conductance"]["mean"],
                    "DRP": cv["base"]["Dissolved Reactive Phosphorus"]["mean"],
                    "MEAN": base_mean,
                },
                {
                    "model": "SPATIAL_CORRECTED",
                    "TA": cv["corrected"]["Total Alkalinity"]["mean"],
                    "EC": cv["corrected"]["Electrical Conductance"]["mean"],
                    "DRP": cv["corrected"]["Dissolved Reactive Phosphorus"]["mean"],
                    "MEAN": corrected_mean,
                },
            ]
        )
        leaderboard.to_csv(sprint_dir / "leaderboard.csv", index=False)

        preds_base, preds_res, preds_final = fit_predict_valid(train, valid)

        # Save base and corrected submissions in sprint folder for audit
        sub_base = template_raw.copy()
        sub_final = template_raw.copy()
        for t in TARGETS:
            sub_base[t] = np.maximum(np.nan_to_num(preds_base[t], nan=0.0), 0.0)
            sub_final[t] = np.maximum(np.nan_to_num(preds_final[t], nan=0.0), 0.0)
        sub_base.to_csv(sprint_dir / "submission_base_model.csv", index=False)
        sub_final.to_csv(sprint_dir / "submission_spatial_corrected.csv", index=False)

        saved_submission = False
        final_sub_path = root / "submissions" / "submission_SPATIAL_RESIDUAL_MODEL.csv"
        if corrected_mean > base_mean:
            sub_final.to_csv(final_sub_path, index=False)
            saved_submission = True

        report = {
            "generated_at": ts(),
            "base_model_cv": cv["base"],
            "residual_model_cv": cv["residual"],
            "spatial_corrected_cv": cv["corrected"],
            "base_mean_cv": base_mean,
            "corrected_mean_cv": corrected_mean,
            "improved": bool(corrected_mean > base_mean),
            "submission_saved": saved_submission,
            "saved_submission_path": str(final_sub_path) if saved_submission else None,
            "rules": {
                "grouping_by_basin_only": True,
                "basin_excluded_from_features": True,
                "station_identifiers_excluded": True,
                "target_encoding_used": False,
            },
        }
        with open(sprint_dir / "spatial_residual_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("\nLEADERBOARD")
        print("BASE MODEL CV")
        print(f"TA={cv['base']['Total Alkalinity']['mean']:.4f} EC={cv['base']['Electrical Conductance']['mean']:.4f} DRP={cv['base']['Dissolved Reactive Phosphorus']['mean']:.4f} MEAN={base_mean:.4f}")
        print("SPATIAL CORRECTED CV")
        print(f"TA={cv['corrected']['Total Alkalinity']['mean']:.4f} EC={cv['corrected']['Electrical Conductance']['mean']:.4f} DRP={cv['corrected']['Dissolved Reactive Phosphorus']['mean']:.4f} MEAN={corrected_mean:.4f}")

    except Exception:
        logger.fail("SPATIAL_RESIDUAL_SPRINT", 3)
        raise
    finally:
        if not logger.errors_path.exists():
            logger.errors_path.write_text("", encoding="utf-8")
        logger.finalize()


if __name__ == "__main__":
    main()
