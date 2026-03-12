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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
SUBMISSION_NAMES = {
    "EXP1_BASELINE_ML": "submission_BASELINE_ML.csv",
    "EXP2_RESIDUAL_KRIGING": "submission_KRIGING_RESIDUAL.csv",
    "EXP3_RESIDUAL_KRIGING_HYDRO_REGION": "submission_KRIGING_HYDRO_REGION.csv",
    "EXP4_ENSEMBLE_KRIGING": "submission_KRIGING_ENSEMBLE.csv",
}
BENCHMARK_MEAN = 0.320
RANDOM_STATE = 42


def ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    return out


def winsor_fit(y: np.ndarray, q: tuple[float, float] = (0.01, 0.99)) -> tuple[float, float]:
    return float(np.nanquantile(y, q[0])), float(np.nanquantile(y, q[1]))


def winsor_apply(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(y, dtype=float), lo, hi)


class AuditLogger:
    def __init__(self, root: Path, total_experiments: int, benchmark_mean: float = BENCHMARK_MEAN):
        self.root = root
        self.total_experiments = total_experiments
        self.progress_path = root / "progress.json"
        self.running_results_path = root / "running_results.csv"
        self.errors_path = root / "errors.log"
        self.started_at = ts()
        self.best_score = float(benchmark_mean)
        self.best_experiment = "benchmark_reference"

        pd.DataFrame(columns=["experiment", "target", "fold", "r2", "timestamp"]).to_csv(
            self.running_results_path, index=False
        )
        self.errors_path.write_text("", encoding="utf-8")
        self.write_progress("running", "", 0, "", 0)

    def write_progress(
        self,
        status: str,
        current_experiment: str,
        experiment_index: int,
        target: str,
        fold: int,
    ) -> None:
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

    def start_experiment(self, exp_name: str, exp_idx: int) -> None:
        print("[EXPERIMENT START]")
        print(ts())
        print(exp_name)
        self.write_progress("running", exp_name, exp_idx, "", 0)

    def start_fold(self, exp_name: str, exp_idx: int, target: str, fold: int, total_folds: int) -> None:
        print("[TARGET]")
        print(TARGET_SHORT[target])
        print("[FOLD]")
        print(f"{fold}/{total_folds}")
        self.write_progress("running", exp_name, exp_idx, TARGET_SHORT[target], fold)

    def log_fold(self, experiment: str, target: str, fold: int, score: float) -> None:
        print("[R2 PARTIAL]")
        print(f"r2={score:.6f}")
        pd.DataFrame(
            [{
                "experiment": experiment,
                "target": TARGET_SHORT[target],
                "fold": int(fold),
                "r2": float(score),
                "timestamp": ts(),
            }]
        ).to_csv(self.running_results_path, mode="a", index=False, header=False)

    def complete_experiment(self, exp_name: str, exp_idx: int, mean_cv: float) -> None:
        if float(mean_cv) > self.best_score:
            self.best_score = float(mean_cv)
            self.best_experiment = exp_name
        self.write_progress("running", exp_name, exp_idx, "", 0)

    def fail_experiment(self, exp_name: str, exp_idx: int) -> None:
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts()}] {exp_name}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        self.write_progress("running", exp_name, exp_idx, "", 0)

    def finalize(self) -> None:
        self.write_progress("completed", self.best_experiment, self.total_experiments, "", 0)


def load_base_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    train = wq.merge(ls_tr, on=KEYS, how="inner").merge(tc_tr, on=KEYS, how="inner")
    valid = ls_va.merge(tc_va, on=KEYS, how="inner")

    if chirps_tr.exists() and chirps_va.exists():
        train = train.merge(normalize_keys(pd.read_csv(chirps_tr)), on=KEYS, how="left")
        valid = valid.merge(normalize_keys(pd.read_csv(chirps_va)), on=KEYS, how="left")

    for extra_path in (ext_hydro, ext_geo):
        if extra_path.exists():
            extra_df = pd.read_csv(extra_path)
            extra_df[LAT_COL] = pd.to_numeric(extra_df[LAT_COL], errors="coerce")
            extra_df[LON_COL] = pd.to_numeric(extra_df[LON_COL], errors="coerce")
            train = train.merge(extra_df, on=[LAT_COL, LON_COL], how="left")
            valid = valid.merge(extra_df, on=[LAT_COL, LON_COL], how="left")

    template_raw = pd.read_csv(raw / "submission_template.csv")
    template_norm = normalize_keys(template_raw)
    valid = template_norm[KEYS].merge(valid, on=KEYS, how="left")
    return train, valid, template_raw


def get_hydro_region_base_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        LAT_COL,
        LON_COL,
        "elevation",
        "slope",
        "upstream_area_km2",
        "dist_to_river_m",
        "river_discharge_cms",
    ]
    available = [col for col in preferred if col in df.columns]

    landcover_candidates = [
        col for col in df.columns
        if any(token in col.lower() for token in ["cropland", "urban", "forest", "grass", "wetland", "shrub", "landcover"])
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    soil_candidates = [
        col for col in df.columns
        if "soil" in col.lower() and pd.api.types.is_numeric_dtype(df[col])
    ]

    cols = available + sorted(landcover_candidates)[:6] + sorted(soil_candidates)[:4]
    deduped: list[str] = []
    for col in cols:
        if col not in deduped:
            deduped.append(col)
    return deduped


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop_cols = set(
        TARGETS + [DATE_COL, LAT_COL, LON_COL, "basin_id", "station_id", "station", "station_name", "site_id"]
    )
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def make_et_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "m",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def fit_ml_model(tr_x: pd.DataFrame, tr_y: np.ndarray) -> Pipeline:
    pipe = make_et_pipeline()
    pipe.fit(tr_x, tr_y)
    return pipe


def normalize_with_bounds(coords: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return (coords - mins) / denom


def fit_spatial_residual_model(coords_tr: np.ndarray, residuals_tr: np.ndarray) -> tuple[GaussianProcessRegressor, tuple[np.ndarray, np.ndarray]]:
    mins = coords_tr.min(axis=0)
    maxs = coords_tr.max(axis=0)
    coords_norm = normalize_with_bounds(coords_tr, mins, maxs)

    max_points = 1200
    if len(coords_norm) > max_points:
        rng = np.random.default_rng(RANDOM_STATE)
        take = np.sort(rng.choice(len(coords_norm), size=max_points, replace=False))
        coords_norm = coords_norm[take]
        residuals_tr = residuals_tr[take]

    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.2, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=RANDOM_STATE,
    )
    gpr.fit(coords_norm, residuals_tr)
    return gpr, (mins, maxs)


def predict_spatial_residuals(model: GaussianProcessRegressor, bounds: tuple[np.ndarray, np.ndarray], coords_va: np.ndarray) -> np.ndarray:
    mins, maxs = bounds
    coords_norm = normalize_with_bounds(coords_va, mins, maxs)
    return model.predict(coords_norm)


def add_fold_safe_hydro_regions(
    train_fold: pd.DataFrame,
    valid_fold: pd.DataFrame,
    train_full: pd.DataFrame,
    valid_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cols = get_hydro_region_base_columns(train_full)
    if len(cols) < 2:
        return train_fold.copy(), valid_fold.copy(), []

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_region = scaler.fit_transform(imp.fit_transform(train_fold[cols]))
    valid_region = scaler.transform(imp.transform(valid_fold[cols]))

    tr_out = train_fold.copy()
    va_out = valid_fold.copy()
    added: list[str] = []
    for k in (4, 6):
        if len(train_fold) < k:
            continue
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        tr_col = f"hydro_region_k{k}"
        tr_out[tr_col] = km.fit_predict(train_region)
        va_out[tr_col] = km.predict(valid_region)
        added.append(tr_col)
    return tr_out, va_out, added


def evaluate_experiment(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    base_features: list[str],
    logger: AuditLogger,
    exp_name: str,
    exp_idx: int,
    use_kriging: bool,
    use_hydro_region: bool,
    ensemble_average: bool,
) -> tuple[dict[str, np.ndarray], dict, dict[str, list[float]]]:
    logger.start_experiment(exp_name, exp_idx)
    groups = train_df["basin_id"].fillna("unknown").astype(str).values
    gkf = GroupKFold(n_splits=5)

    scores = {t: [] for t in TARGETS}
    oof = {t: np.zeros(len(train_df), dtype=float) for t in TARGETS}
    fold_residual_stats = {t: [] for t in TARGETS}

    for target in TARGETS:
        y_all = train_df[target].astype(float).values
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_all, groups), start=1):
            logger.start_fold(exp_name, exp_idx, target, fold_i, 5)

            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()
            if use_hydro_region:
                tr, va, added_cols = add_fold_safe_hydro_regions(tr, va, train_df, valid_df)
            else:
                added_cols = []

            features = list(dict.fromkeys(base_features + added_cols))
            lo, hi = winsor_fit(tr[target].values)
            y_tr = winsor_apply(tr[target].values, lo, hi)
            y_va = va[target].astype(float).values

            pipe = fit_ml_model(tr[features], y_tr)
            pred_ml_tr = np.maximum(pipe.predict(tr[features]), 0.0)
            pred_ml_va = np.maximum(pipe.predict(va[features]), 0.0)
            pred_final = pred_ml_va.copy()

            residual_summary = 0.0
            if use_kriging:
                residuals_tr = y_tr - pred_ml_tr
                residual_summary = float(np.std(residuals_tr))
                coords_tr = tr[[LAT_COL, LON_COL]].to_numpy(dtype=float)
                coords_va = va[[LAT_COL, LON_COL]].to_numpy(dtype=float)
                try:
                    gpr, bounds = fit_spatial_residual_model(coords_tr, residuals_tr)
                    pred_residuals_va = predict_spatial_residuals(gpr, bounds, coords_va)
                except Exception as err:
                    pred_residuals_va = np.zeros(len(va), dtype=float)
                    with open(logger.errors_path, "a", encoding="utf-8") as f:
                        f.write(f"[{ts()}] {exp_name} {TARGET_SHORT[target]} fold {fold_i} kriging fallback\n{err}\n")

                pred_corrected = np.maximum(pred_ml_va + pred_residuals_va, 0.0)
                pred_final = (pred_ml_va + pred_corrected) / 2.0 if ensemble_average else pred_corrected

            score = float(r2_score(y_va, pred_final))
            scores[target].append(score)
            fold_residual_stats[target].append(residual_summary)
            oof[target][va_idx] = pred_final
            logger.log_fold(exp_name, target, fold_i, score)

    report = {
        TARGET_SHORT[target]: {
            "mean": float(np.mean(scores[target])),
            "std": float(np.std(scores[target])),
            "folds": [float(x) for x in scores[target]],
            "mean_training_residual_std": float(np.mean(fold_residual_stats[target])),
        }
        for target in TARGETS
    }
    report["mean_cv"] = float(np.mean([report[TARGET_SHORT[target]]["mean"] for target in TARGETS]))
    logger.complete_experiment(exp_name, exp_idx, report["mean_cv"])
    return oof, report, fold_residual_stats


def build_submission_predictions(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    base_features: list[str],
    use_kriging: bool,
    use_hydro_region: bool,
    ensemble_average: bool,
) -> pd.DataFrame:
    train_fit = train_df.copy()
    valid_fit = valid_df.copy()
    added_cols: list[str] = []
    if use_hydro_region:
        train_fit, valid_fit, added_cols = add_fold_safe_hydro_regions(train_fit, valid_fit, train_df, valid_df)

    features = list(dict.fromkeys(base_features + added_cols))
    pred_map: dict[str, np.ndarray] = {}

    for target in TARGETS:
        y_tr = train_df[target].astype(float).values
        lo, hi = winsor_fit(y_tr)
        y_tr = winsor_apply(y_tr, lo, hi)

        pipe = fit_ml_model(train_fit[features], y_tr)
        pred_ml_tr = np.maximum(pipe.predict(train_fit[features]), 0.0)
        pred_ml_va = np.maximum(pipe.predict(valid_fit[features]), 0.0)
        pred_final = pred_ml_va.copy()

        if use_kriging:
            residuals_tr = y_tr - pred_ml_tr
            coords_tr = train_df[[LAT_COL, LON_COL]].to_numpy(dtype=float)
            coords_va = valid_df[[LAT_COL, LON_COL]].to_numpy(dtype=float)
            try:
                gpr, bounds = fit_spatial_residual_model(coords_tr, residuals_tr)
                pred_residuals_va = predict_spatial_residuals(gpr, bounds, coords_va)
            except Exception:
                pred_residuals_va = np.zeros(len(valid_df), dtype=float)

            pred_corrected = np.maximum(pred_ml_va + pred_residuals_va, 0.0)
            pred_final = (pred_ml_va + pred_corrected) / 2.0 if ensemble_average else pred_corrected

        pred_map[target] = np.maximum(pred_final, 0.0)

    submission = valid_df[KEYS].copy()
    for target in TARGETS:
        submission[target] = pred_map[target]
    return submission


def align_submission_to_template(pred_df: pd.DataFrame, template_raw: pd.DataFrame) -> pd.DataFrame:
    template_norm = normalize_keys(template_raw)
    merged = template_norm[KEYS].merge(pred_df, on=KEYS, how="left")
    output = template_raw.copy()
    for target in TARGETS:
        output[target] = np.maximum(merged[target].astype(float).fillna(0.0).to_numpy(), 0.0)
    return output


def print_leaderboard(summary_df: pd.DataFrame) -> None:
    print("EXPERIMENT | TA | EC | DRP | MEAN")
    for _, row in summary_df.iterrows():
        print(f"{row['EXPERIMENT']} | {row['TA']:.4f} | {row['EC']:.4f} | {row['DRP']:.4f} | {row['MEAN']:.4f}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sprint_root = root / "experiments" / "kriging_residual_sprint"
    submissions_root = root / "submissions"
    sprint_root.mkdir(parents=True, exist_ok=True)
    submissions_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RESIDUAL GEOSTATISTICAL SPRINT (KRIGING)")
    print("=" * 70)

    train_df, valid_df, template_raw = load_base_data(root)
    base_features = feature_cols(train_df)
    logger = AuditLogger(sprint_root, total_experiments=4)

    experiments = [
        ("EXP1_BASELINE_ML", 1, False, False, False),
        ("EXP2_RESIDUAL_KRIGING", 2, True, False, False),
        ("EXP3_RESIDUAL_KRIGING_HYDRO_REGION", 3, True, True, False),
        ("EXP4_ENSEMBLE_KRIGING", 4, True, False, True),
    ]

    experiment_reports: dict[str, dict] = {}
    saved_submissions: dict[str, str] = {}

    for exp_name, exp_idx, use_kriging, use_hydro_region, ensemble_average in experiments:
        try:
            _, report, residual_stats = evaluate_experiment(
                train_df=train_df,
                valid_df=valid_df,
                base_features=base_features,
                logger=logger,
                exp_name=exp_name,
                exp_idx=exp_idx,
                use_kriging=use_kriging,
                use_hydro_region=use_hydro_region,
                ensemble_average=ensemble_average,
            )
            report["residual_std_by_target"] = {
                TARGET_SHORT[target]: [float(v) for v in residual_stats[target]] for target in TARGETS
            }
            experiment_reports[exp_name] = report

            pred_df = build_submission_predictions(
                train_df=train_df,
                valid_df=valid_df,
                base_features=base_features,
                use_kriging=use_kriging,
                use_hydro_region=use_hydro_region,
                ensemble_average=ensemble_average,
            )
            submission = align_submission_to_template(pred_df, template_raw)
            submission_name = SUBMISSION_NAMES[exp_name]
            submission_path = submissions_root / submission_name
            submission.to_csv(submission_path, index=False)
            saved_submissions[exp_name] = str(submission_path)
            print(f"Saved submission: {submission_path}")
        except Exception:
            logger.fail_experiment(exp_name, exp_idx)

    leaderboard_rows = []
    for exp_name, report in experiment_reports.items():
        leaderboard_rows.append(
            {
                "EXPERIMENT": exp_name,
                "TA": report["TA"]["mean"],
                "EC": report["EC"]["mean"],
                "DRP": report["DRP"]["mean"],
                "MEAN": report["mean_cv"],
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("MEAN", ascending=False).reset_index(drop=True)
    leaderboard_df.to_csv(sprint_root / "leaderboard.csv", index=False)
    print_leaderboard(leaderboard_df)

    if leaderboard_df.empty:
        best_experiment = None
        best_submission = None
    else:
        best_experiment = str(leaderboard_df.iloc[0]["EXPERIMENT"])
        best_submission = saved_submissions.get(best_experiment)

    interpretation = {
        "baseline_vs_kriging_mean_delta": None,
        "hydro_region_increment_over_kriging": None,
        "ensemble_increment_over_kriging": None,
        "spatial_residual_patterns": {},
    }
    if "EXP1_BASELINE_ML" in experiment_reports and "EXP2_RESIDUAL_KRIGING" in experiment_reports:
        interpretation["baseline_vs_kriging_mean_delta"] = float(
            experiment_reports["EXP2_RESIDUAL_KRIGING"]["mean_cv"] - experiment_reports["EXP1_BASELINE_ML"]["mean_cv"]
        )
    if "EXP2_RESIDUAL_KRIGING" in experiment_reports and "EXP3_RESIDUAL_KRIGING_HYDRO_REGION" in experiment_reports:
        interpretation["hydro_region_increment_over_kriging"] = float(
            experiment_reports["EXP3_RESIDUAL_KRIGING_HYDRO_REGION"]["mean_cv"]
            - experiment_reports["EXP2_RESIDUAL_KRIGING"]["mean_cv"]
        )
    if "EXP2_RESIDUAL_KRIGING" in experiment_reports and "EXP4_ENSEMBLE_KRIGING" in experiment_reports:
        interpretation["ensemble_increment_over_kriging"] = float(
            experiment_reports["EXP4_ENSEMBLE_KRIGING"]["mean_cv"] - experiment_reports["EXP2_RESIDUAL_KRIGING"]["mean_cv"]
        )

    for target in TARGETS:
        short = TARGET_SHORT[target]
        baseline = experiment_reports.get("EXP1_BASELINE_ML", {}).get(short, {}).get("mean")
        kriging = experiment_reports.get("EXP2_RESIDUAL_KRIGING", {}).get(short, {}).get("mean")
        hydro = experiment_reports.get("EXP3_RESIDUAL_KRIGING_HYDRO_REGION", {}).get(short, {}).get("mean")
        ensemble = experiment_reports.get("EXP4_ENSEMBLE_KRIGING", {}).get(short, {}).get("mean")
        interpretation["spatial_residual_patterns"][short] = {
            "baseline": baseline,
            "kriging": kriging,
            "hydro_region": hydro,
            "ensemble": ensemble,
            "best_variant": max(
                [("baseline", baseline), ("kriging", kriging), ("hydro_region", hydro), ("ensemble", ensemble)],
                key=lambda x: -1e18 if x[1] is None else x[1],
            )[0],
        }

    report_payload = {
        "timestamp": ts(),
        "benchmark_mean_cv": BENCHMARK_MEAN,
        "beat_project_benchmark": bool(not leaderboard_df.empty and leaderboard_df.iloc[0]["MEAN"] > BENCHMARK_MEAN),
        "best_experiment": best_experiment,
        "best_submission_file": best_submission,
        "experiments": experiment_reports,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "interpretation": interpretation,
    }
    with open(sprint_root / "kriging_residual_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.finalize()


if __name__ == "__main__":
    main()