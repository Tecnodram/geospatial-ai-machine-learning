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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

BENCHMARK_MEAN_CV = 0.320


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
    def __init__(self, root: Path, total_experiments: int, benchmark_mean: float):
        self.root = root
        self.total_experiments = int(total_experiments)
        self.progress_path = root / "progress.json"
        self.running_results_path = root / "running_results.csv"
        self.errors_path = root / "errors.log"
        self.started_at = ts()
        self.completed_experiments = 0
        self.best_mean = float(benchmark_mean)
        self.best_name = "EXP_STACKING_V1"

        if not self.running_results_path.exists():
            pd.DataFrame(columns=["experiment", "target", "fold", "r2", "timestamp"]).to_csv(
                self.running_results_path, index=False
            )

        if not self.errors_path.exists():
            self.errors_path.write_text("", encoding="utf-8")

        self.write_progress("running", "", 0, "", 0)

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
            "completed_experiments": int(self.completed_experiments),
            "best_mean_cv_so_far": float(self.best_mean),
            "best_experiment_so_far": self.best_name,
        }
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def start_experiment(self, exp_name: str, exp_idx: int):
        print(f"[START] {ts()}")
        print(f"[EXPERIMENT] {exp_idx}/{self.total_experiments}")
        print(f"[EXPERIMENT NAME] {exp_name}")
        self.write_progress("running", exp_name, exp_idx, "", 0)

    def start_fold(self, exp_name: str, exp_idx: int, target: str, fold: int, total_folds: int):
        print(f"[TARGET] {TARGET_SHORT[target]}")
        print(f"[FOLD] {fold}/{total_folds}")
        print("[STATUS] running")
        self.write_progress("running", exp_name, exp_idx, TARGET_SHORT[target], fold)

    def complete_fold(self, exp_name: str, exp_idx: int, target: str, fold: int, score: float):
        print("[STATUS] completed")
        print(f"[R2 PARTIAL] {score:.6f}")
        pd.DataFrame(
            [
                {
                    "experiment": exp_name,
                    "target": TARGET_SHORT[target],
                    "fold": int(fold),
                    "r2": float(score),
                    "timestamp": ts(),
                }
            ]
        ).to_csv(self.running_results_path, mode="a", index=False, header=False)
        self.write_progress("running", exp_name, exp_idx, TARGET_SHORT[target], fold)

    def complete_experiment(self, exp_name: str, exp_idx: int, mean_cv: float):
        self.completed_experiments += 1
        if float(mean_cv) > self.best_mean:
            self.best_mean = float(mean_cv)
            self.best_name = exp_name
        self.write_progress("running", exp_name, exp_idx, "", 0)

    def fail_experiment(self, exp_name: str, exp_idx: int):
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts()}] {exp_name}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        self.write_progress("running", exp_name, exp_idx, "", 0)

    def finalize(self):
        self.write_progress("completed", self.best_name, self.total_experiments, "", 0)


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
        skipped.append("temporal rainfall lags skipped: chirps_ppt missing")
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


def add_validated_base_features(df: pd.DataFrame, skipped: list[str]) -> pd.DataFrame:
    out = add_rainfall_antecedent_package(df.copy(), skipped)
    out = add_landuse_pressure(out, skipped)
    out = add_upstream_pressure(out, skipped)
    out = add_temporal_memory_features(out, skipped)
    return out


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df.get(name, pd.Series(np.nan, index=df.index)), errors="coerce")


def add_connectivity_proxies(df: pd.DataFrame, skipped: list[str]) -> pd.DataFrame:
    out = df.copy()

    dist = _num_col(out, "dist_to_river_m").clip(lower=0)
    up = _num_col(out, "upstream_area_km2").clip(lower=0)
    slope = _num_col(out, "slope").clip(lower=0)
    discharge = _num_col(out, "river_discharge_cms").clip(lower=0)
    rain30 = _num_col(out, "rain_30d_sum").clip(lower=0)
    c2b = _num_col(out, "catch_to_basin_ratio")
    crop5 = _num_col(out, "cropland_fraction_5km").clip(lower=0)
    urban5 = _num_col(out, "urban_fraction_5km").clip(lower=0)

    if dist.notna().any():
        out["log_dist_to_river"] = np.log1p(dist)
        out["river_influence_decay_1"] = np.exp(-dist / 1000.0)
        out["river_influence_decay_5"] = np.exp(-dist / 5000.0)
        out["river_influence_decay_10"] = np.exp(-dist / 10000.0)
    else:
        skipped.append("connectivity distance decays skipped: dist_to_river_m missing")

    if up.notna().any() and slope.notna().any():
        out["stream_power_proxy"] = slope * up
    else:
        skipped.append("stream_power_proxy skipped: slope or upstream_area_km2 missing")

    if up.notna().any() and dist.notna().any():
        out["connectivity_proxy"] = up / (dist + 1.0)
    else:
        skipped.append("connectivity_proxy skipped: dist_to_river_m or upstream_area_km2 missing")

    if up.notna().any() and discharge.notna().any():
        out["discharge_contact_proxy"] = up / (discharge + 1.0)
    else:
        skipped.append("discharge_contact_proxy skipped: river_discharge_cms or upstream_area_km2 missing")

    if c2b.notna().any():
        out["basin_compaction_proxy"] = c2b
    else:
        skipped.append("basin_compaction_proxy skipped: catch_to_basin_ratio missing")

    if up.notna().any() and (crop5.notna().any() or urban5.notna().any()):
        out["upstream_pressure_index"] = up * (crop5.fillna(0.0) + urban5.fillna(0.0))
    else:
        skipped.append("upstream_pressure_index skipped: upstream area or landuse fractions missing")

    if rain30.notna().any() and slope.notna().any() and "river_influence_decay_5" in out.columns:
        out["runoff_transfer_index"] = rain30 * slope * out["river_influence_decay_5"]
    else:
        skipped.append("runoff_transfer_index skipped: rain_30d_sum/slope/river decay missing")

    if up.notna().any() and rain30.notna().any() and discharge.notna().any():
        out["nutrient_transport_proxy"] = up * rain30 / (discharge + 1.0)
    else:
        skipped.append("nutrient_transport_proxy skipped: upstream area/rain/discharge missing")

    return out


def find_hydro_neighbor_source_cols(df: pd.DataFrame) -> dict[str, str]:
    preferred = {
        "elevation": ["elevation"],
        "slope": ["slope"],
        "upstream_area": ["upstream_area_km2"],
        "dist_to_river": ["dist_to_river_m"],
        "rain_30d": ["rain_30d_sum", "rain_roll_30"],
        "cropland_fraction": ["cropland_fraction_5km", "cropland_fraction_10km", "cropland_fraction_1km"],
        "urban_fraction": ["urban_fraction_5km", "urban_fraction_10km", "urban_fraction_1km"],
    }
    out: dict[str, str] = {}
    for feat_name, candidates in preferred.items():
        for c in candidates:
            if c in df.columns:
                out[feat_name] = c
                break
    return out


def build_hydro_space(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    candidates = [
        LAT_COL,
        LON_COL,
        "elevation",
        "slope",
        "upstream_area_km2",
        "dist_to_river_m",
        "river_discharge_cms",
        "cropland_fraction_5km",
        "urban_fraction_5km",
        "landcover",
    ]
    present = [c for c in candidates if c in df.columns]
    if len(present) < 2:
        x = pd.DataFrame(index=df.index)
        x["dummy_hydro_space"] = 0.0
        return x, ["dummy_hydro_space"]

    x = df[present].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    return x, present


def build_fold_safe_hydro_neighbor_features(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    n_neighbors: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr_space, _ = build_hydro_space(tr_df)
    va_space, _ = build_hydro_space(va_df)

    source_cols = find_hydro_neighbor_source_cols(tr_df)
    if len(source_cols) == 0:
        empty_tr = pd.DataFrame(index=tr_df.index)
        empty_va = pd.DataFrame(index=va_df.index)
        return empty_tr, empty_va, []

    tr_values = tr_df[[source_cols[k] for k in source_cols]].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    k_tr = min(max(2, n_neighbors + 1), len(tr_df))
    k_va = min(max(1, n_neighbors), len(tr_df))

    nbrs = NearestNeighbors(n_neighbors=k_tr, metric="euclidean")
    nbrs.fit(tr_space.values)

    d_tr, idx_tr = nbrs.kneighbors(tr_space.values, n_neighbors=k_tr, return_distance=True)
    feat_tr = pd.DataFrame(index=tr_df.index)
    for short_name, src_col in source_cols.items():
        vals = tr_values[src_col].values.astype(float)
        neigh_idx = idx_tr[:, 1:] if idx_tr.shape[1] > 1 else idx_tr
        feat_tr[f"hydro_neighbor_mean_{short_name}"] = np.nanmean(vals[neigh_idx], axis=1)

    nbrs_va = NearestNeighbors(n_neighbors=k_va, metric="euclidean")
    nbrs_va.fit(tr_space.values)
    _, idx_va = nbrs_va.kneighbors(va_space.values, n_neighbors=k_va, return_distance=True)
    feat_va = pd.DataFrame(index=va_df.index)
    for short_name, src_col in source_cols.items():
        vals = tr_values[src_col].values.astype(float)
        feat_va[f"hydro_neighbor_mean_{short_name}"] = np.nanmean(vals[idx_va], axis=1)

    created_cols = feat_va.columns.tolist()
    return feat_tr, feat_va, created_cols


def add_hydro_region_features(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr = tr_df.copy()
    va = va_df.copy()

    tr_space, used_cols = build_hydro_space(tr)
    va_space, _ = build_hydro_space(va)

    kk = min(max(2, int(k)), max(2, len(tr_space) // 2))
    km = KMeans(n_clusters=kk, random_state=42, n_init=20)
    tr[f"hydro_region_k{k}"] = km.fit_predict(tr_space.values)
    va[f"hydro_region_k{k}"] = km.predict(va_space.values)

    return tr, va, used_cols


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop_cols = set(
        TARGETS
        + [
            DATE_COL,
            LAT_COL,
            LON_COL,
            "basin_id",
            "station_id",
            "station",
            "station_name",
            "site_id",
            "pfaf_id",
        ]
    )
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def mk_pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", model)])


def model_bank(target: str):
    if target == "Total Alkalinity":
        return ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
    if target == "Electrical Conductance":
        return ExtraTreesRegressor(n_estimators=900, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
    return ExtraTreesRegressor(n_estimators=1100, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)


def run_cv_et(
    train_df: pd.DataFrame,
    exp_name: str,
    exp_idx: int,
    logger: AuditLogger,
    use_connectivity: bool,
    use_neighbors: bool,
    region_mode: str,
) -> tuple[dict, dict, list[str], dict]:
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    folds = GroupKFold(n_splits=5)

    fold_scores = {t: [] for t in TARGETS}
    oof = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    region_k4_scores = {t: [] for t in TARGETS}
    region_k6_scores = {t: [] for t in TARGETS}

    used_feature_union: set[str] = set()

    for t in TARGETS:
        y_all = train_df[t].values.astype(float)
        for fold_i, (tr_idx, va_idx) in enumerate(folds.split(train_df, y_all, groups), start=1):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()

            if use_connectivity:
                tr = add_connectivity_proxies(tr, skipped=[])
                va = add_connectivity_proxies(va, skipped=[])

            if use_neighbors:
                ntr, nva, _ = build_fold_safe_hydro_neighbor_features(tr, va)
                tr = pd.concat([tr, ntr], axis=1)
                va = pd.concat([va, nva], axis=1)

            if region_mode == "k4":
                tr, va, _ = add_hydro_region_features(tr, va, k=4)
            elif region_mode == "k6":
                tr, va, _ = add_hydro_region_features(tr, va, k=6)
            elif region_mode == "auto_best":
                tr4, va4, _ = add_hydro_region_features(tr, va, k=4)
                tr6, va6, _ = add_hydro_region_features(tr, va, k=6)

                feats4 = feature_cols(tr4)
                feats6 = feature_cols(tr6)

                lo, hi = winsor_fit(tr4[t].values)
                y_tr = winsor_apply(tr4[t].values, lo, hi)

                m4 = mk_pipe(model_bank(t))
                m6 = mk_pipe(model_bank(t))
                m4.fit(tr4[feats4], y_tr)
                m6.fit(tr6[feats6], y_tr)

                pred4 = np.maximum(m4.predict(va4[feats4]), 0.0)
                pred6 = np.maximum(m6.predict(va6[feats6]), 0.0)

                sc4 = float(r2_score(va4[t].values.astype(float), pred4))
                sc6 = float(r2_score(va6[t].values.astype(float), pred6))
                region_k4_scores[t].append(sc4)
                region_k6_scores[t].append(sc6)

                if sc6 > sc4:
                    tr = tr6
                    va = va6
                else:
                    tr = tr4
                    va = va4

            feats = feature_cols(tr)
            used_feature_union.update(feats)

            logger.start_fold(exp_name, exp_idx, t, fold_i, 5)

            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)
            model = mk_pipe(model_bank(t))
            model.fit(tr[feats], y_tr)
            pred = np.maximum(model.predict(va[feats]), 0.0)
            sc = float(r2_score(va[t].values.astype(float), pred))

            fold_scores[t].append(sc)
            oof[t][va_idx] = pred
            logger.complete_fold(exp_name, exp_idx, t, fold_i, sc)

    rep = {
        tt: {
            "mean": float(np.mean(fold_scores[tt])),
            "std": float(np.std(fold_scores[tt])),
            "folds": [float(x) for x in fold_scores[tt]],
        }
        for tt in TARGETS
    }
    rep["mean_cv"] = float(np.mean([rep[tt]["mean"] for tt in TARGETS]))

    extra = {}
    if region_mode == "auto_best":
        extra = {
            "k4": {
                tt: {
                    "mean": float(np.mean(region_k4_scores[tt])) if len(region_k4_scores[tt]) else np.nan,
                    "std": float(np.std(region_k4_scores[tt])) if len(region_k4_scores[tt]) else np.nan,
                    "folds": [float(x) for x in region_k4_scores[tt]],
                }
                for tt in TARGETS
            },
            "k6": {
                tt: {
                    "mean": float(np.mean(region_k6_scores[tt])) if len(region_k6_scores[tt]) else np.nan,
                    "std": float(np.std(region_k6_scores[tt])) if len(region_k6_scores[tt]) else np.nan,
                    "folds": [float(x) for x in region_k6_scores[tt]],
                }
                for tt in TARGETS
            },
        }
        extra["k4"]["mean_cv"] = float(np.mean([extra["k4"][tt]["mean"] for tt in TARGETS]))
        extra["k6"]["mean_cv"] = float(np.mean([extra["k6"][tt]["mean"] for tt in TARGETS]))

    return rep, oof, sorted(used_feature_union), extra


def fit_predict_valid_et(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    use_connectivity: bool,
    use_neighbors: bool,
    region_mode: str,
) -> dict[str, np.ndarray]:
    tr = train_df.copy()
    va = valid_df.copy()

    if use_connectivity:
        tr = add_connectivity_proxies(tr, skipped=[])
        va = add_connectivity_proxies(va, skipped=[])

    if use_neighbors:
        ntr, nva, _ = build_fold_safe_hydro_neighbor_features(tr, va)
        tr = pd.concat([tr, ntr], axis=1)
        va = pd.concat([va, nva], axis=1)

    if region_mode == "k4":
        tr, va, _ = add_hydro_region_features(tr, va, k=4)
    elif region_mode == "k6":
        tr, va, _ = add_hydro_region_features(tr, va, k=6)
    elif region_mode == "auto_best":
        tr4, va4, _ = add_hydro_region_features(tr, va, k=4)
        tr6, va6, _ = add_hydro_region_features(tr, va, k=6)

        score4 = 0.0
        score6 = 0.0
        for t in TARGETS:
            f4 = feature_cols(tr4)
            f6 = feature_cols(tr6)
            lo, hi = winsor_fit(tr4[t].values)
            y = winsor_apply(tr4[t].values, lo, hi)
            m4 = mk_pipe(model_bank(t))
            m6 = mk_pipe(model_bank(t))
            m4.fit(tr4[f4], y)
            m6.fit(tr6[f6], y)
            p4 = np.maximum(m4.predict(tr4[f4]), 0.0)
            p6 = np.maximum(m6.predict(tr6[f6]), 0.0)
            score4 += float(r2_score(tr4[t].values.astype(float), p4))
            score6 += float(r2_score(tr6[t].values.astype(float), p6))

        if score6 > score4:
            tr, va = tr6, va6
        else:
            tr, va = tr4, va4

    feats = feature_cols(tr)
    out = {}
    for t in TARGETS:
        lo, hi = winsor_fit(tr[t].values)
        y = winsor_apply(tr[t].values, lo, hi)
        m = mk_pipe(model_bank(t))
        m.fit(tr[feats], y)
        out[t] = np.maximum(m.predict(va[feats]), 0.0)
    return out


def mk_meta() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=10000, random_state=42)),
        ]
    )


def run_cv_stack_drp(
    train_df: pd.DataFrame,
    exp_name: str,
    exp_idx: int,
    logger: AuditLogger,
    use_connectivity: bool,
    use_neighbors: bool,
    region_mode: str,
) -> tuple[dict, dict, list[str], dict]:
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    folds = GroupKFold(n_splits=5)

    fold_scores = {t: [] for t in TARGETS}
    oof = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}

    used_feature_union: set[str] = set()

    for t in ["Total Alkalinity", "Electrical Conductance"]:
        y_all = train_df[t].values.astype(float)
        for fold_i, (tr_idx, va_idx) in enumerate(folds.split(train_df, y_all, groups), start=1):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()

            if use_connectivity:
                tr = add_connectivity_proxies(tr, skipped=[])
                va = add_connectivity_proxies(va, skipped=[])
            if use_neighbors:
                ntr, nva, _ = build_fold_safe_hydro_neighbor_features(tr, va)
                tr = pd.concat([tr, ntr], axis=1)
                va = pd.concat([va, nva], axis=1)
            if region_mode == "k4":
                tr, va, _ = add_hydro_region_features(tr, va, k=4)
            elif region_mode == "k6":
                tr, va, _ = add_hydro_region_features(tr, va, k=6)

            feats = feature_cols(tr)
            used_feature_union.update(feats)

            logger.start_fold(exp_name, exp_idx, t, fold_i, 5)

            lo, hi = winsor_fit(tr[t].values)
            y_tr = winsor_apply(tr[t].values, lo, hi)

            et = mk_pipe(ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
            rf = mk_pipe(RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
            hgb = mk_pipe(HistGradientBoostingRegressor(max_iter=450, learning_rate=0.05, max_depth=6, min_samples_leaf=20, random_state=42))

            et.fit(tr[feats], y_tr)
            rf.fit(tr[feats], y_tr)
            hgb.fit(tr[feats], y_tr)

            tr_meta = pd.DataFrame(
                {
                    "pred_ET": et.predict(tr[feats]),
                    "pred_RF": rf.predict(tr[feats]),
                    "pred_HGB": hgb.predict(tr[feats]),
                }
            )
            va_meta = pd.DataFrame(
                {
                    "pred_ET": et.predict(va[feats]),
                    "pred_RF": rf.predict(va[feats]),
                    "pred_HGB": hgb.predict(va[feats]),
                }
            )

            meta = mk_meta()
            meta.fit(tr_meta, y_tr)
            pred = np.maximum(meta.predict(va_meta), 0.0)
            sc = float(r2_score(va[t].values.astype(float), pred))

            fold_scores[t].append(sc)
            oof[t][va_idx] = pred
            logger.complete_fold(exp_name, exp_idx, t, fold_i, sc)

    t = "Dissolved Reactive Phosphorus"
    y_all = train_df[t].values.astype(float)
    for fold_i, (tr_idx, va_idx) in enumerate(folds.split(train_df, y_all, groups), start=1):
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()

        if use_connectivity:
            tr = add_connectivity_proxies(tr, skipped=[])
            va = add_connectivity_proxies(va, skipped=[])
        if use_neighbors:
            ntr, nva, _ = build_fold_safe_hydro_neighbor_features(tr, va)
            tr = pd.concat([tr, ntr], axis=1)
            va = pd.concat([va, nva], axis=1)
        if region_mode == "k4":
            tr, va, _ = add_hydro_region_features(tr, va, k=4)
        elif region_mode == "k6":
            tr, va, _ = add_hydro_region_features(tr, va, k=6)

        feats = feature_cols(tr)
        used_feature_union.update(feats)

        logger.start_fold(exp_name, exp_idx, t, fold_i, 5)

        y_tr = tr[t].values.astype(float)
        q = float(np.nanquantile(y_tr, 0.75))
        y_bin = (y_tr >= q).astype(int)

        clf = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        learning_rate=0.05,
                        max_depth=5,
                        min_samples_leaf=25,
                        random_state=42,
                    ),
                ),
            ]
        )
        clf.fit(tr[feats], y_bin)
        p_tr = clf.predict_proba(tr[feats])[:, 1]
        p_va = clf.predict_proba(va[feats])[:, 1]

        tr2 = tr[feats].copy()
        va2 = va[feats].copy()
        tr2["P_high"] = p_tr
        va2["P_high"] = p_va

        lo, hi = winsor_fit(y_tr)
        y_tr_w = winsor_apply(y_tr, lo, hi)
        reg = mk_pipe(ExtraTreesRegressor(n_estimators=1100, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        reg.fit(tr2, y_tr_w)
        reg_pred = np.maximum(reg.predict(va2), 0.0)
        pred = np.maximum(0.6 * reg_pred + 0.4 * (p_va * reg_pred), 0.0)

        sc = float(r2_score(va[t].values.astype(float), pred))
        fold_scores[t].append(sc)
        oof[t][va_idx] = pred
        logger.complete_fold(exp_name, exp_idx, t, fold_i, sc)

    rep = {
        tt: {
            "mean": float(np.mean(fold_scores[tt])),
            "std": float(np.std(fold_scores[tt])),
            "folds": [float(x) for x in fold_scores[tt]],
        }
        for tt in TARGETS
    }
    rep["mean_cv"] = float(np.mean([rep[tt]["mean"] for tt in TARGETS]))
    return rep, oof, sorted(used_feature_union), {}


def fit_predict_valid_stack_drp(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    use_connectivity: bool,
    use_neighbors: bool,
    region_mode: str,
) -> dict[str, np.ndarray]:
    tr = train_df.copy()
    va = valid_df.copy()

    if use_connectivity:
        tr = add_connectivity_proxies(tr, skipped=[])
        va = add_connectivity_proxies(va, skipped=[])
    if use_neighbors:
        ntr, nva, _ = build_fold_safe_hydro_neighbor_features(tr, va)
        tr = pd.concat([tr, ntr], axis=1)
        va = pd.concat([va, nva], axis=1)
    if region_mode == "k4":
        tr, va, _ = add_hydro_region_features(tr, va, k=4)
    elif region_mode == "k6":
        tr, va, _ = add_hydro_region_features(tr, va, k=6)

    feats = feature_cols(tr)
    out: dict[str, np.ndarray] = {}

    for t in ["Total Alkalinity", "Electrical Conductance"]:
        lo, hi = winsor_fit(tr[t].values)
        y = winsor_apply(tr[t].values, lo, hi)

        et = mk_pipe(ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        rf = mk_pipe(RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
        hgb = mk_pipe(HistGradientBoostingRegressor(max_iter=450, learning_rate=0.05, max_depth=6, min_samples_leaf=20, random_state=42))

        et.fit(tr[feats], y)
        rf.fit(tr[feats], y)
        hgb.fit(tr[feats], y)

        tr_meta = pd.DataFrame(
            {
                "pred_ET": et.predict(tr[feats]),
                "pred_RF": rf.predict(tr[feats]),
                "pred_HGB": hgb.predict(tr[feats]),
            }
        )
        va_meta = pd.DataFrame(
            {
                "pred_ET": et.predict(va[feats]),
                "pred_RF": rf.predict(va[feats]),
                "pred_HGB": hgb.predict(va[feats]),
            }
        )
        meta = mk_meta()
        meta.fit(tr_meta, y)
        out[t] = np.maximum(meta.predict(va_meta), 0.0)

    t = "Dissolved Reactive Phosphorus"
    y = tr[t].values.astype(float)
    q = float(np.nanquantile(y, 0.75))
    y_bin = (y >= q).astype(int)

    clf = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=500,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_leaf=25,
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(tr[feats], y_bin)
    p_tr = clf.predict_proba(tr[feats])[:, 1]
    p_va = clf.predict_proba(va[feats])[:, 1]

    tr2 = tr[feats].copy()
    va2 = va[feats].copy()
    tr2["P_high"] = p_tr
    va2["P_high"] = p_va

    lo, hi = winsor_fit(y)
    y_w = winsor_apply(y, lo, hi)
    reg = mk_pipe(ExtraTreesRegressor(n_estimators=1100, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1))
    reg.fit(tr2, y_w)
    reg_pred = np.maximum(reg.predict(va2), 0.0)
    out[t] = np.maximum(0.6 * reg_pred + 0.4 * (p_va * reg_pred), 0.0)

    return out


def write_feature_importance(train_df: pd.DataFrame, feats: list[str], exp_dir: Path):
    rows = []
    if len(feats) == 0:
        pd.DataFrame(columns=["target", "feature", "importance", "rank"]).to_csv(exp_dir / "feature_importance.csv", index=False)
        return

    for t in TARGETS:
        y = train_df[t].values.astype(float)
        lo, hi = winsor_fit(y)
        y_w = winsor_apply(y, lo, hi)
        model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)
        pipe = mk_pipe(model)
        pipe.fit(train_df[feats], y_w)
        imp = pipe.named_steps["m"].feature_importances_
        rank_idx = np.argsort(imp)[::-1][:15]
        for r, idx in enumerate(rank_idx, start=1):
            rows.append(
                {
                    "target": TARGET_SHORT[t],
                    "feature": feats[idx],
                    "importance": float(imp[idx]),
                    "rank": r,
                }
            )
    pd.DataFrame(rows).to_csv(exp_dir / "feature_importance.csv", index=False)


def write_submission(template_raw: pd.DataFrame, preds: dict, out_path: Path):
    sub = template_raw.copy()
    for t in TARGETS:
        sub[t] = np.maximum(np.nan_to_num(preds[t], nan=0.0), 0.0)
    sub.to_csv(out_path, index=False)


def build_methodology_text(
    exp_name: str,
    blocks_text: str,
    region_mode: str,
    region_probe: dict,
    skipped: list[str],
) -> str:
    lines = [
        f"# Methodology - {exp_name}",
        "",
        "## Included Blocks",
        blocks_text,
        "",
        "## Leakage Controls",
        "- GroupKFold with group key = basin_id.",
        "- basin_id excluded from model features.",
        "- station/site identifiers excluded from model features.",
        "- No target encoding, no KNN target encoding, no station memorization.",
        "- Hydro-neighbor summaries are computed from feature-space covariates only (no targets).",
        "- For each fold, validation neighbor aggregates use training-fold rows only.",
        "",
        "## Connectivity Formulas",
        "- log_dist_to_river = log1p(dist_to_river_m)",
        "- river_influence_decay_1 = exp(-dist_to_river_m / 1000)",
        "- river_influence_decay_5 = exp(-dist_to_river_m / 5000)",
        "- river_influence_decay_10 = exp(-dist_to_river_m / 10000)",
        "- stream_power_proxy = slope * upstream_area_km2",
        "- connectivity_proxy = upstream_area_km2 / (dist_to_river_m + 1)",
        "- discharge_contact_proxy = upstream_area_km2 / (river_discharge_cms + 1)",
        "- basin_compaction_proxy = catch_to_basin_ratio",
        "- upstream_pressure_index = upstream_area_km2 * (cropland_fraction_5km + urban_fraction_5km)",
        "- runoff_transfer_index = rain_30d_sum * slope * river_influence_decay_5",
        "- nutrient_transport_proxy = upstream_area_km2 * rain_30d_sum / (river_discharge_cms + 1)",
    ]

    if region_mode in {"k4", "k6", "auto_best"}:
        lines.extend(["", "## Hydro Regions", "- Features are created from fold-train-only KMeans over hydro-spatial covariates."])
    if region_mode == "auto_best" and len(region_probe):
        lines.append(f"- Region probe k4 mean_cv: {region_probe.get('k4_mean_cv', np.nan):.6f}")
        lines.append(f"- Region probe k6 mean_cv: {region_probe.get('k6_mean_cv', np.nan):.6f}")

    if skipped:
        lines.extend(["", "## Skipped/Unavailable", *[f"- {s}" for s in skipped]])

    return "\n".join(lines) + "\n"


def run_single_experiment(
    exp_name: str,
    exp_idx: int,
    total_experiments: int,
    train_base: pd.DataFrame,
    valid_base: pd.DataFrame,
    template_raw: pd.DataFrame,
    use_connectivity: bool,
    use_neighbors: bool,
    region_mode: str,
    stack_mode: bool,
    logger: AuditLogger,
    sprint_dir: Path,
) -> dict:
    skipped: list[str] = []
    exp_dir = sprint_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.start_experiment(exp_name, exp_idx)

    if stack_mode:
        cv_rep, _, used_feats, extra = run_cv_stack_drp(
            train_df=train_base,
            exp_name=exp_name,
            exp_idx=exp_idx,
            logger=logger,
            use_connectivity=use_connectivity,
            use_neighbors=use_neighbors,
            region_mode=region_mode,
        )
        preds = fit_predict_valid_stack_drp(
            train_df=train_base,
            valid_df=valid_base,
            use_connectivity=use_connectivity,
            use_neighbors=use_neighbors,
            region_mode=region_mode,
        )
        model_summary = {
            "model_families": ["ExtraTrees", "RandomForest", "HistGradientBoosting"],
            "meta_model": "ElasticNet",
            "drp_specialist": "HGB classifier + ET regressor two-stage",
        }
    else:
        cv_rep, _, used_feats, extra = run_cv_et(
            train_df=train_base,
            exp_name=exp_name,
            exp_idx=exp_idx,
            logger=logger,
            use_connectivity=use_connectivity,
            use_neighbors=use_neighbors,
            region_mode=region_mode,
        )
        preds = fit_predict_valid_et(
            train_df=train_base,
            valid_df=valid_base,
            use_connectivity=use_connectivity,
            use_neighbors=use_neighbors,
            region_mode=region_mode,
        )
        model_summary = {
            "model_families": ["ExtraTrees"],
            "meta_model": None,
            "drp_specialist": False,
        }

    config = {
        "experiment": exp_name,
        "index": exp_idx,
        "total_experiments": total_experiments,
        "feature_blocks": {
            "connectivity_proxies": bool(use_connectivity),
            "hydro_neighbor_features": bool(use_neighbors),
            "hydro_regions": region_mode,
        },
        "stack_mode": bool(stack_mode),
        "rules": {
            "basin_id_group_only": True,
            "basin_id_excluded_from_features": True,
            "station_id_excluded_from_features": True,
            "target_encoding_used": False,
            "knn_target_encoding_used": False,
        },
        "started_at": ts(),
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    artifact_train = train_base.copy()
    if use_connectivity:
        artifact_train = add_connectivity_proxies(artifact_train, skipped=[])
    if use_neighbors:
        ntr, _, _ = build_fold_safe_hydro_neighbor_features(artifact_train, artifact_train)
        artifact_train = pd.concat([artifact_train, ntr], axis=1)
    if region_mode == "k4":
        artifact_train, _, _ = add_hydro_region_features(artifact_train, artifact_train, k=4)
    elif region_mode == "k6":
        artifact_train, _, _ = add_hydro_region_features(artifact_train, artifact_train, k=6)
    elif region_mode == "auto_best":
        tr4, _, _ = add_hydro_region_features(artifact_train, artifact_train, k=4)
        tr6, _, _ = add_hydro_region_features(artifact_train, artifact_train, k=6)
        artifact_train = artifact_train.copy()
        artifact_train["hydro_region_k4"] = tr4["hydro_region_k4"].values
        artifact_train["hydro_region_k6"] = tr6["hydro_region_k6"].values

    artifact_feats = [c for c in used_feats if c in artifact_train.columns]
    missing_feats = [c for c in used_feats if c not in artifact_train.columns]

    with open(exp_dir / "features_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_features": len(artifact_feats),
                "features": artifact_feats,
                "missing_from_artifact_frame": missing_feats,
                "skipped_features": skipped,
            },
            f,
            indent=2,
        )

    if region_mode == "auto_best" and len(extra):
        cv_rep["hydro_region_probe"] = extra

    with open(exp_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_rep, f, indent=2)

    with open(exp_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2)

    write_feature_importance(artifact_train, artifact_feats, exp_dir)

    blocks_text = (
        f"- Connectivity proxies: {use_connectivity}\n"
        f"- Hydro-neighbor feature summaries (fold-safe): {use_neighbors}\n"
        f"- Hydro-region mode: {region_mode}\n"
        f"- Stacking mode: {stack_mode}"
    )
    region_probe = {}
    if region_mode == "auto_best" and len(extra):
        region_probe = {
            "k4_mean_cv": float(extra["k4"]["mean_cv"]),
            "k6_mean_cv": float(extra["k6"]["mean_cv"]),
        }
    methodology = build_methodology_text(exp_name, blocks_text, region_mode, region_probe, skipped)
    (exp_dir / "methodology_notes.md").write_text(methodology, encoding="utf-8")

    sub_path = exp_dir / f"submission_{exp_name}.csv"
    write_submission(template_raw, preds, sub_path)

    logger.complete_experiment(exp_name, exp_idx, cv_rep["mean_cv"])

    return {
        "experiment": exp_name,
        "TA_mean_cv": cv_rep["Total Alkalinity"]["mean"],
        "EC_mean_cv": cv_rep["Electrical Conductance"]["mean"],
        "DRP_mean_cv": cv_rep["Dissolved Reactive Phosphorus"]["mean"],
        "TA_std": cv_rep["Total Alkalinity"]["std"],
        "EC_std": cv_rep["Electrical Conductance"]["std"],
        "DRP_std": cv_rep["Dissolved Reactive Phosphorus"]["std"],
        "mean_cv": cv_rep["mean_cv"],
        "blocks": blocks_text.replace("\n", " | "),
        "submission_file": str(sub_path),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    sprint_dir = root / "experiments" / "river_network_sprint"
    sprint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[START] {ts()}")

    skipped_global: list[str] = []
    train_raw, valid_raw, template_raw = load_base_data(root)
    train_base = add_validated_base_features(train_raw, skipped_global)
    valid_base = add_validated_base_features(valid_raw, skipped_global)

    logger = AuditLogger(sprint_dir, total_experiments=6, benchmark_mean=BENCHMARK_MEAN_CV)

    summary_rows = []

    experiments = [
        {
            "name": "EXP_1_BASELINE_RECHECK",
            "use_connectivity": False,
            "use_neighbors": False,
            "region_mode": "none",
            "stack_mode": False,
        },
        {
            "name": "EXP_2_CONNECTIVITY_PROXIES",
            "use_connectivity": True,
            "use_neighbors": False,
            "region_mode": "none",
            "stack_mode": False,
        },
        {
            "name": "EXP_3_HYDRO_NEIGHBOR_FEATURES",
            "use_connectivity": False,
            "use_neighbors": True,
            "region_mode": "none",
            "stack_mode": False,
        },
        {
            "name": "EXP_4_HYDRO_REGIONS",
            "use_connectivity": False,
            "use_neighbors": False,
            "region_mode": "auto_best",
            "stack_mode": False,
        },
    ]

    for idx, exp in enumerate(experiments, start=1):
        try:
            row = run_single_experiment(
                exp_name=exp["name"],
                exp_idx=idx,
                total_experiments=6,
                train_base=train_base,
                valid_base=valid_base,
                template_raw=template_raw,
                use_connectivity=exp["use_connectivity"],
                use_neighbors=exp["use_neighbors"],
                region_mode=exp["region_mode"],
                stack_mode=exp["stack_mode"],
                logger=logger,
                sprint_dir=sprint_dir,
            )
            summary_rows.append(row)
        except Exception:
            logger.fail_experiment(exp["name"], idx)
            continue

    if len(summary_rows) == 0:
        logger.finalize()
        raise RuntimeError("All experiments failed in EXP1-EXP4. Check errors.log")

    baseline_row = next((r for r in summary_rows if r["experiment"] == "EXP_1_BASELINE_RECHECK"), None)
    baseline_mean = float(baseline_row["mean_cv"]) if baseline_row else -1e9

    isolated_map = {
        "connectivity": next((r for r in summary_rows if r["experiment"] == "EXP_2_CONNECTIVITY_PROXIES"), None),
        "neighbors": next((r for r in summary_rows if r["experiment"] == "EXP_3_HYDRO_NEIGHBOR_FEATURES"), None),
        "regions": next((r for r in summary_rows if r["experiment"] == "EXP_4_HYDRO_REGIONS"), None),
    }

    improved_blocks = []
    for block_name, row in isolated_map.items():
        if row is not None and float(row["mean_cv"]) > baseline_mean:
            improved_blocks.append(block_name)

    combo_use_connectivity = "connectivity" in improved_blocks
    combo_use_neighbors = "neighbors" in improved_blocks
    combo_region_mode = "auto_best" if "regions" in improved_blocks else "none"

    if len(improved_blocks) == 0:
        # least harmful single block
        candidates = [
            ("connectivity", isolated_map["connectivity"]),
            ("neighbors", isolated_map["neighbors"]),
            ("regions", isolated_map["regions"]),
        ]
        candidates = [(n, r) for n, r in candidates if r is not None]
        if len(candidates):
            least_harmful = max(candidates, key=lambda x: float(x[1]["mean_cv"]))[0]
            improved_blocks = [least_harmful]
            combo_use_connectivity = least_harmful == "connectivity"
            combo_use_neighbors = least_harmful == "neighbors"
            combo_region_mode = "auto_best" if least_harmful == "regions" else "none"

    try:
        row5 = run_single_experiment(
            exp_name="EXP_5_NETWORK_PROXY_COMBO",
            exp_idx=5,
            total_experiments=6,
            train_base=train_base,
            valid_base=valid_base,
            template_raw=template_raw,
            use_connectivity=combo_use_connectivity,
            use_neighbors=combo_use_neighbors,
            region_mode=combo_region_mode,
            stack_mode=False,
            logger=logger,
            sprint_dir=sprint_dir,
        )
        row5["selected_blocks_from_exp2_4"] = ",".join(improved_blocks)
        summary_rows.append(row5)
    except Exception:
        logger.fail_experiment("EXP_5_NETWORK_PROXY_COMBO", 5)

    # EXP6 uses best set from EXP1-EXP5
    if len(summary_rows):
        temp_df = pd.DataFrame(summary_rows)
        best_pre6 = temp_df.sort_values("mean_cv", ascending=False).iloc[0]["experiment"]
    else:
        best_pre6 = "EXP_1_BASELINE_RECHECK"

    if best_pre6 == "EXP_2_CONNECTIVITY_PROXIES":
        exp6_connectivity, exp6_neighbors, exp6_region = True, False, "none"
    elif best_pre6 == "EXP_3_HYDRO_NEIGHBOR_FEATURES":
        exp6_connectivity, exp6_neighbors, exp6_region = False, True, "none"
    elif best_pre6 == "EXP_4_HYDRO_REGIONS":
        exp6_connectivity, exp6_neighbors, exp6_region = False, False, "auto_best"
    elif best_pre6 == "EXP_5_NETWORK_PROXY_COMBO":
        exp6_connectivity, exp6_neighbors, exp6_region = combo_use_connectivity, combo_use_neighbors, combo_region_mode
    else:
        exp6_connectivity, exp6_neighbors, exp6_region = False, False, "none"

    try:
        row6 = run_single_experiment(
            exp_name="EXP_6_FINAL_NETWORK_STACK",
            exp_idx=6,
            total_experiments=6,
            train_base=train_base,
            valid_base=valid_base,
            template_raw=template_raw,
            use_connectivity=exp6_connectivity,
            use_neighbors=exp6_neighbors,
            region_mode=exp6_region,
            stack_mode=True,
            logger=logger,
            sprint_dir=sprint_dir,
        )
        summary_rows.append(row6)
    except Exception:
        logger.fail_experiment("EXP_6_FINAL_NETWORK_STACK", 6)

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) == 0:
        logger.finalize()
        raise RuntimeError("All experiments failed. Check errors.log")

    summary_df = summary_df.sort_values("mean_cv", ascending=False).reset_index(drop=True)
    summary_df.to_csv(sprint_dir / "experiment_summary.csv", index=False)

    best = summary_df.iloc[0]
    best_mean = float(best["mean_cv"])

    final_submission_path = root / "submissions" / "submission_RIVER_NETWORK_SPRINT_BEST.csv"
    submission_saved = False
    recommended_submission = None

    if best_mean > BENCHMARK_MEAN_CV:
        best_sub = Path(best["submission_file"])
        pd.read_csv(best_sub).to_csv(final_submission_path, index=False)
        submission_saved = True
        recommended_submission = str(final_submission_path)

    # final concise leaderboard output
    print("\nFINAL LEADERBOARD")
    for _, r in summary_df.iterrows():
        print(f"{r['experiment']} | TA={r['TA_mean_cv']:.4f} EC={r['EC_mean_cv']:.4f} DRP={r['DRP_mean_cv']:.4f} Mean={r['mean_cv']:.4f}")

    print("\nBEST experiment:", best["experiment"])
    print("Beat current project best (~0.320):", "YES" if best_mean > BENCHMARK_MEAN_CV else "NO")
    if submission_saved:
        print("Recommended submission file:", final_submission_path)
    else:
        print("Recommended submission file: None (benchmark not beaten)")

    # concise interpretation
    exp_scores = {row["experiment"]: float(row["mean_cv"]) for row in summary_rows}
    isolated = [
        ("connectivity proxies", exp_scores.get("EXP_2_CONNECTIVITY_PROXIES", np.nan)),
        ("hydro-neighbor summaries", exp_scores.get("EXP_3_HYDRO_NEIGHBOR_FEATURES", np.nan)),
        ("hydro regions", exp_scores.get("EXP_4_HYDRO_REGIONS", np.nan)),
    ]
    isolated = [(n, s) for n, s in isolated if not pd.isna(s)]
    helpful = max(isolated, key=lambda x: x[1]) if len(isolated) else ("none", np.nan)
    weak = min(isolated, key=lambda x: x[1]) if len(isolated) else ("none", np.nan)

    if len(isolated):
        print(f"Most helpful hydrological connectivity signal: {helpful[0]} (mean CV={helpful[1]:.4f})")
        print(f"Least helpful signal in isolation: {weak[0]} (mean CV={weak[1]:.4f})")
    else:
        print("Most helpful hydrological connectivity signal: not available")
        print("Least helpful signal in isolation: not available")

    report = {
        "generated_at": ts(),
        "benchmark_mean_cv": BENCHMARK_MEAN_CV,
        "best_experiment": best["experiment"],
        "best_mean_cv": best_mean,
        "beat_project_benchmark": bool(best_mean > BENCHMARK_MEAN_CV),
        "benchmark_statement": (
            "River-network-aware features did not surpass the current project benchmark."
            if best_mean <= BENCHMARK_MEAN_CV
            else "River-network-aware features surpassed the current project benchmark."
        ),
        "submission_saved": submission_saved,
        "recommended_submission": recommended_submission,
        "improved_blocks_over_exp1": improved_blocks,
        "experiments": summary_df.to_dict(orient="records"),
        "interpretation": {
            "most_helpful_signal": helpful[0],
            "least_helpful_signal": weak[0],
            "note": "Interpretations are based on isolated block experiments (EXP2-EXP4).",
        },
        "global_skipped_notes": skipped_global,
    }
    with open(sprint_dir / "river_network_sprint_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.finalize()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
