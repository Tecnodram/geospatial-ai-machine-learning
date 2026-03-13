#!/usr/bin/env python
# coding: utf-8
"""
Hydrologically informed feature engineering + regime-aware modeling research runner.

Deliverables:
1) Feature ranking report
2) Controlled experiment config
3) CV performance comparison
4) Recommendation on meaningful signal gains
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"
DRP_COL = "Dissolved Reactive Phosphorus"
KEYS = [LAT_COL, LON_COL, DATE_COL]


@dataclass
class BranchSupport:
    total_count: int
    unique_basins: int
    active_expert: bool
    fallback_reason: str


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    return out


def merge_training_frame(cfg: dict, root: Path) -> pd.DataFrame:
    data_cfg = cfg["data"]

    base = normalize_keys(pd.read_csv(root / data_cfg["fold_base_path"]))
    tc = normalize_keys(pd.read_csv(root / data_cfg["terraclimate_train_path"]))
    ls = normalize_keys(pd.read_csv(root / data_cfg["landsat_train_path"]))
    chirps = normalize_keys(pd.read_csv(root / data_cfg["chirps_train_path"]))

    out = (
        base
        .merge(tc[[LAT_COL, LON_COL, DATE_COL, "pet"]], on=KEYS, how="left")
        .merge(ls, on=KEYS, how="left", suffixes=("", "_ls"))
        .merge(chirps[[LAT_COL, LON_COL, DATE_COL, "chirps_ppt"]], on=KEYS, how="left")
    )
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def add_research_features(df: pd.DataFrame, rolling_window: int) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    # Normalize key numeric inputs.
    for c in ["pet", "chirps_ppt", "soil_clay_0_5", "slope"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values([LAT_COL, LON_COL, DATE_COL]).copy()
    if "chirps_ppt" in out.columns:
        out["chirps_ppt_roll"] = (
            out.groupby([LAT_COL, LON_COL])["chirps_ppt"]
            .transform(lambda s: s.rolling(window=rolling_window, min_periods=1).mean())
        )

    # NDVI is optional; compute from red band if present, otherwise mark unavailable.
    ndvi_available = False
    if "NDVI" in out.columns:
        out["NDVI_proxy"] = pd.to_numeric(out["NDVI"], errors="coerce")
        ndvi_available = True
    elif "nir" in out.columns and "red" in out.columns:
        nir = pd.to_numeric(out["nir"], errors="coerce")
        red = pd.to_numeric(out["red"], errors="coerce")
        out["NDVI_proxy"] = (nir - red) / (nir + red + 1e-6)
        ndvi_available = True

    available = set(out.columns)

    def safe_ratio(a: str, b: str) -> pd.Series:
        return out[a] / (out[b] + 1e-6)

    # Line A candidates.
    if {"chirps_ppt", "pet"}.issubset(available):
        out["A_precip_over_pet"] = safe_ratio("chirps_ppt", "pet")
    if {"chirps_ppt_roll", "pet"}.issubset(available):
        out["A_roll_precip_over_pet"] = safe_ratio("chirps_ppt_roll", "pet")
    if ndvi_available and "chirps_ppt" in available:
        out["A_ndvi_x_precip"] = out["NDVI_proxy"] * out["chirps_ppt"]
    if {"soil_clay_0_5", "chirps_ppt"}.issubset(available):
        out["A_soil_clay_x_precip"] = out["soil_clay_0_5"] * out["chirps_ppt"]
    if {"slope", "chirps_ppt"}.issubset(available):
        out["A_slope_x_precip"] = out["slope"] * out["chirps_ppt"]
    if ndvi_available and "soil_clay_0_5" in available:
        out["A_ndvi_x_soil_clay"] = out["NDVI_proxy"] * out["soil_clay_0_5"]

    # Line C proxies (explicit names requested).
    if {"chirps_ppt", "slope"}.issubset(available):
        out["C_runoff_proxy"] = out["chirps_ppt"] * out["slope"]
    if {"chirps_ppt", "soil_clay_0_5"}.issubset(available):
        out["C_nutrient_transport"] = out["chirps_ppt"] * out["soil_clay_0_5"]
    if {"chirps_ppt", "pet"}.issubset(available):
        out["C_wetness_index"] = safe_ratio("chirps_ppt", "pet")
    if ndvi_available and "chirps_ppt" in available:
        out["C_vegetation_runoff"] = out["NDVI_proxy"] * out["chirps_ppt"]
    if ndvi_available and "soil_clay_0_5" in available:
        out["C_soil_retention"] = out["soil_clay_0_5"] * out["NDVI_proxy"]

    candidate_cols = [c for c in out.columns if c.startswith("A_") or c.startswith("C_")]

    metadata = {
        "ndvi_available": ndvi_available,
        "candidate_count": len(candidate_cols),
        "candidate_columns": candidate_cols,
    }
    return out, metadata


def get_base_feature_columns(df: pd.DataFrame, fold_col: str) -> list[str]:
    exclude = {
        DATE_COL,
        DRP_COL,
        "Total Alkalinity",
        "Electrical Conductance",
        LAT_COL,
        LON_COL,
        fold_col,
        "fold_current",
        "fold_kmeans_8",
        "fold_kmeans_10",
        "fold_kmeans_12",
        "fold_kmeans_15",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c == "basin_id":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_model(model_cfg: dict) -> ExtraTreesRegressor:
    params = model_cfg["params"]
    return ExtraTreesRegressor(
        n_estimators=int(params["n_estimators"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        random_state=int(params["random_state"]),
        n_jobs=int(params["n_jobs"]),
    )


def cv_score_fixed_folds(df: pd.DataFrame, fold_col: str, features: list[str], model_cfg: dict) -> dict:
    folds = sorted(df[fold_col].dropna().unique().tolist())
    y_all = pd.to_numeric(df[DRP_COL], errors="coerce")

    scores = []
    for fold in folds:
        tr_mask = df[fold_col] != fold
        va_mask = df[fold_col] == fold

        tr = df.loc[tr_mask, features]
        va = df.loc[va_mask, features]
        y_tr = y_all.loc[tr_mask].values
        y_va = y_all.loc[va_mask].values

        imp = SimpleImputer(strategy="median")
        x_tr = imp.fit_transform(tr)
        x_va = imp.transform(va)

        model = make_model(model_cfg)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_va)
        scores.append(float(r2_score(y_va, pred)))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": [float(s) for s in scores],
    }


def correlation_stats(df: pd.DataFrame, col: str) -> dict:
    s = pd.to_numeric(df[col], errors="coerce")
    y = pd.to_numeric(df[DRP_COL], errors="coerce")
    mask = s.notna() & y.notna()
    n = int(mask.sum())
    if n < 20:
        return {"pearson": np.nan, "spearman": np.nan, "n": n}

    pearson = float(s[mask].corr(y[mask], method="pearson"))
    spearman = float(s[mask].corr(y[mask], method="spearman"))
    return {"pearson": pearson, "spearman": spearman, "n": n}


def evaluate_branch_support(train_df: pd.DataFrame, branch_mask: pd.Series, min_count: int, min_basins: int) -> BranchSupport:
    sub = train_df.loc[branch_mask].copy()
    total_count = int(len(sub))
    unique_basins = int(sub["basin_id"].nunique()) if "basin_id" in sub.columns else 0

    active = (total_count >= min_count) and (unique_basins >= min_basins)
    reasons = []
    if total_count < min_count:
        reasons.append(f"count<{min_count}")
    if unique_basins < min_basins:
        reasons.append(f"basins<{min_basins}")

    return BranchSupport(
        total_count=total_count,
        unique_basins=unique_basins,
        active_expert=active,
        fallback_reason="" if active else ";".join(reasons),
    )


def regime_expert_cv(
    df: pd.DataFrame,
    fold_col: str,
    features: list[str],
    model_cfg: dict,
    pet_threshold: float,
    min_count: int,
    min_basins: int,
) -> tuple[dict, list[dict]]:
    folds = sorted(df[fold_col].dropna().unique().tolist())
    y_all = pd.to_numeric(df[DRP_COL], errors="coerce")

    scores = []
    diagnostics = []

    for fold in folds:
        tr_mask = df[fold_col] != fold
        va_mask = df[fold_col] == fold

        tr_df = df.loc[tr_mask].copy()
        va_df = df.loc[va_mask].copy()

        tr_df["pet_high"] = pd.to_numeric(tr_df["pet"], errors="coerce") > pet_threshold
        va_df["pet_high"] = pd.to_numeric(va_df["pet"], errors="coerce") > pet_threshold
        tr_df["pet_low"] = ~tr_df["pet_high"]
        va_df["pet_low"] = ~va_df["pet_high"]

        y_tr = y_all.loc[tr_mask].values
        y_va = y_all.loc[va_mask].values

        imp = SimpleImputer(strategy="median")
        x_tr = imp.fit_transform(tr_df[features])
        x_va = imp.transform(va_df[features])

        # Always-fit global fallback model.
        global_model = make_model(model_cfg)
        global_model.fit(x_tr, y_tr)
        pred = global_model.predict(x_va)

        hi_support = evaluate_branch_support(tr_df, tr_df["pet_high"], min_count, min_basins)
        lo_support = evaluate_branch_support(tr_df, tr_df["pet_low"], min_count, min_basins)

        hi_mask_va = va_df["pet_high"].values
        lo_mask_va = va_df["pet_low"].values

        if hi_support.active_expert and hi_mask_va.any():
            tr_hi = tr_df.loc[tr_df["pet_high"], features]
            y_hi = tr_df.loc[tr_df["pet_high"], DRP_COL].astype(float).values
            imp_hi = SimpleImputer(strategy="median")
            x_hi = imp_hi.fit_transform(tr_hi)
            x_hi_va = imp_hi.transform(va_df.loc[hi_mask_va, features])
            m_hi = make_model(model_cfg)
            m_hi.fit(x_hi, y_hi)
            pred[hi_mask_va] = m_hi.predict(x_hi_va)

        if lo_support.active_expert and lo_mask_va.any():
            tr_lo = tr_df.loc[tr_df["pet_low"], features]
            y_lo = tr_df.loc[tr_df["pet_low"], DRP_COL].astype(float).values
            imp_lo = SimpleImputer(strategy="median")
            x_lo = imp_lo.fit_transform(tr_lo)
            x_lo_va = imp_lo.transform(va_df.loc[lo_mask_va, features])
            m_lo = make_model(model_cfg)
            m_lo.fit(x_lo, y_lo)
            pred[lo_mask_va] = m_lo.predict(x_lo_va)

        sc = float(r2_score(y_va, pred))
        scores.append(sc)

        diagnostics.append(
            {
                "fold": int(fold),
                "r2": sc,
                "pet_high_active": bool(hi_support.active_expert),
                "pet_low_active": bool(lo_support.active_expert),
                "pet_high_count": hi_support.total_count,
                "pet_low_count": lo_support.total_count,
                "pet_high_basins": hi_support.unique_basins,
                "pet_low_basins": lo_support.unique_basins,
                "pet_high_fallback_reason": hi_support.fallback_reason,
                "pet_low_fallback_reason": lo_support.fallback_reason,
            }
        )

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": [float(s) for s in scores],
    }, diagnostics


def write_markdown_summary(path: Path, run_id: str, cfg_path: str, ranking_df: pd.DataFrame, cv_df: pd.DataFrame, notes: list[str], recommendation: str) -> None:
    lines = [
        "# Hydrologically Informed Signal Research Report",
        "",
        f"Run ID: {run_id}",
        f"Config: {cfg_path}",
        "",
        "## Top Interaction Ranking",
        "",
        "| feature | family | delta_cv_r2 | abs_spearman | abs_pearson | rank_score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in ranking_df.head(12).iterrows():
        lines.append(
            f"| {r['feature']} | {r['family']} | {r['delta_cv_r2']:.6f} | {r['abs_spearman']:.4f} | {r['abs_pearson']:.4f} | {r['rank_score']:.2f} |"
        )

    lines += [
        "",
        "## CV Performance Comparison",
        "",
        "| experiment | mean_r2 | std_r2 |",
        "|---|---:|---:|",
    ]
    for _, r in cv_df.iterrows():
        lines.append(f"| {r['experiment']} | {r['mean_r2']:.6f} | {r['std_r2']:.6f} |")

    lines += [
        "",
        "## Notes",
    ]
    for n in notes:
        lines.append(f"- {n}")

    lines += [
        "",
        "## Recommendation",
        recommendation,
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_exp_hydro_regime_signal_v1.yml")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / args.config
    if not cfg_path.exists():
        cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    run_id = f"{cfg['project']['name']}_{ts()}"
    out_dir = root / cfg["project"]["out_dir"] / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    df = merge_training_frame(cfg, root)
    df, feat_meta = add_research_features(df, rolling_window=int(cfg["features"]["rolling_window_obs"]))

    fold_col = cfg["cv"]["fold_column"]
    required = [fold_col, DRP_COL, "pet", "chirps_ppt", "soil_clay_0_5", "slope"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise RuntimeError(f"Missing required columns: {missing_required}")

    # Keep rows with known fold and target.
    df = df.loc[df[fold_col].notna() & df[DRP_COL].notna()].copy()

    base_features = get_base_feature_columns(df, fold_col)
    baseline_cv = cv_score_fixed_folds(df, fold_col, base_features, cfg["model"])

    candidate_cols = feat_meta["candidate_columns"]
    rows = []

    for col in candidate_cols:
        corr = correlation_stats(df, col)
        feat_set = base_features + [col]
        cv_one = cv_score_fixed_folds(df, fold_col, feat_set, cfg["model"])
        delta = cv_one["mean"] - baseline_cv["mean"]
        rows.append(
            {
                "feature": col,
                "family": "A" if col.startswith("A_") else "C",
                "pearson": corr["pearson"],
                "spearman": corr["spearman"],
                "abs_pearson": abs(corr["pearson"]) if pd.notna(corr["pearson"]) else np.nan,
                "abs_spearman": abs(corr["spearman"]) if pd.notna(corr["spearman"]) else np.nan,
                "n": corr["n"],
                "cv_mean_with_feature": cv_one["mean"],
                "delta_cv_r2": delta,
            }
        )

    ranking_df = pd.DataFrame(rows)
    if ranking_df.empty:
        raise RuntimeError("No candidate features were generated.")

    # Predictive potential rank: prioritize CV gain, then monotonic and linear association.
    ranking_df["rk_delta"] = ranking_df["delta_cv_r2"].rank(ascending=False, method="min")
    ranking_df["rk_spearman"] = ranking_df["abs_spearman"].rank(ascending=False, method="min")
    ranking_df["rk_pearson"] = ranking_df["abs_pearson"].rank(ascending=False, method="min")
    ranking_df["rank_score"] = 0.6 * ranking_df["rk_delta"] + 0.25 * ranking_df["rk_spearman"] + 0.15 * ranking_df["rk_pearson"]
    ranking_df = ranking_df.sort_values(["rank_score", "delta_cv_r2"], ascending=[True, False]).reset_index(drop=True)

    configured_top_k = int(cfg["features"].get("top_k_interactions", 8))
    top_k = min(max(5, configured_top_k), 10, len(ranking_df))
    selected_top = ranking_df.head(top_k)["feature"].tolist()

    # Controlled experiments.
    line_a_cols = [c for c in candidate_cols if c.startswith("A_")]
    line_c_cols = [c for c in candidate_cols if c.startswith("C_")]

    cv_rows = []

    cv_rows.append({
        "experiment": "baseline_global_et",
        "mean_r2": baseline_cv["mean"],
        "std_r2": baseline_cv["std"],
        "folds": baseline_cv["folds"],
    })

    line_a_top = [c for c in selected_top if c in line_a_cols]
    if line_a_top:
        cv_a = cv_score_fixed_folds(df, fold_col, base_features + line_a_top, cfg["model"])
        cv_rows.append({"experiment": "line_a_top_interactions", "mean_r2": cv_a["mean"], "std_r2": cv_a["std"], "folds": cv_a["folds"]})

    if line_c_cols:
        cv_c = cv_score_fixed_folds(df, fold_col, base_features + line_c_cols, cfg["model"])
        cv_rows.append({"experiment": "line_c_transport_proxies", "mean_r2": cv_c["mean"], "std_r2": cv_c["std"], "folds": cv_c["folds"]})

    cv_ac = cv_score_fixed_folds(df, fold_col, base_features + selected_top, cfg["model"])
    cv_rows.append({"experiment": "line_a_c_selected_top", "mean_r2": cv_ac["mean"], "std_r2": cv_ac["std"], "folds": cv_ac["folds"]})

    regime_cv, regime_diag = regime_expert_cv(
        df=df,
        fold_col=fold_col,
        features=base_features + selected_top,
        model_cfg=cfg["model"],
        pet_threshold=float(cfg["regime"]["pet_high_threshold"]),
        min_count=int(cfg["regime"]["min_branch_train_count"]),
        min_basins=int(cfg["regime"]["min_branch_unique_basins"]),
    )
    cv_rows.append({
        "experiment": "line_b_regime_expert_pet_gate",
        "mean_r2": regime_cv["mean"],
        "std_r2": regime_cv["std"],
        "folds": regime_cv["folds"],
    })

    cv_df = pd.DataFrame(cv_rows).sort_values("mean_r2", ascending=False).reset_index(drop=True)

    best_row = cv_df.iloc[0]
    base_mean = float(baseline_cv["mean"])
    gain_vs_base = float(best_row["mean_r2"] - base_mean)

    notes = []
    if not feat_meta["ndvi_available"]:
        notes.append("NDVI was not available in allowed datasets; NDVI-based candidates were skipped.")
    notes.append(f"Candidate interactions evaluated: {len(candidate_cols)}")
    notes.append(f"Top features selected for controlled run: {selected_top}")
    notes.append(f"PET threshold for regime split: {cfg['regime']['pet_high_threshold']}")

    meaningful = gain_vs_base >= 0.002
    if meaningful:
        recommendation = (
            f"Meaningful new signal detected. Best family/experiment: {best_row['experiment']} with "
            f"CV mean {best_row['mean_r2']:.6f} (gain vs baseline {gain_vs_base:+.6f}). "
            "Proceed to a submission-safe confirmation run with this family and fixed parameters."
        )
    else:
        recommendation = (
            f"No meaningful new signal in this controlled pass. Best observed gain vs baseline is {gain_vs_base:+.6f}, "
            "below the practical threshold (+0.002 CV R2). Keep the current backbone and defer this family."
        )

    # Deliverables.
    ranking_out = ranking_df[[
        "feature", "family", "delta_cv_r2", "cv_mean_with_feature", "abs_spearman", "abs_pearson", "pearson", "spearman", "n", "rank_score"
    ]].copy()
    ranking_out.to_csv(out_dir / "feature_ranking_report.csv", index=False)

    cv_out = cv_df[["experiment", "mean_r2", "std_r2", "folds"]].copy()
    cv_out.to_csv(out_dir / "cv_performance_comparison.csv", index=False)

    controlled_cfg = {
        "run_id": run_id,
        "source_config": str(cfg_path),
        "selected_top_features": selected_top,
        "line_a_features_available": line_a_cols,
        "line_c_features_available": line_c_cols,
        "pet_high_threshold": float(cfg["regime"]["pet_high_threshold"]),
        "model": cfg["model"],
        "cv": cfg["cv"],
        "constraints": {
            "external_datasets_used": False,
            "large_hyperparameter_sweeps": False,
            "spatial_cv_preserved": True,
        },
    }
    with open(out_dir / "controlled_experiment_config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(controlled_cfg, f, sort_keys=False)

    payload = {
        "run_id": run_id,
        "config": str(cfg_path),
        "baseline_cv": baseline_cv,
        "selected_top_features": selected_top,
        "cv_comparison": cv_rows,
        "regime_diagnostics": regime_diag,
        "feature_metadata": feat_meta,
        "recommendation": recommendation,
    }
    with open(out_dir / "cv_performance_comparison.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(out_dir / "regime_expert_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "diagnostics": regime_diag}, f, indent=2)

    write_markdown_summary(
        path=out_dir / "feature_ranking_report.md",
        run_id=run_id,
        cfg_path=str(cfg_path),
        ranking_df=ranking_out,
        cv_df=cv_out,
        notes=notes,
        recommendation=recommendation,
    )

    if cfg.get("execution", {}).get("save_feature_matrix_sample", False):
        sample_cols = [LAT_COL, LON_COL, DATE_COL, DRP_COL] + selected_top
        df[sample_cols].head(300).to_csv(out_dir / "feature_matrix_sample.csv", index=False)

    print("=" * 72)
    print("HYDRO REGIME SIGNAL RESEARCH RUN COMPLETE")
    print("=" * 72)
    print(f"run_dir: {out_dir}")
    print(f"baseline_cv_mean: {baseline_cv['mean']:.6f}")
    print(f"best_experiment: {best_row['experiment']} ({best_row['mean_r2']:.6f})")
    print(f"selected_top_features ({len(selected_top)}): {selected_top}")
    print("deliverables:")
    print(f"- {out_dir / 'feature_ranking_report.csv'}")
    print(f"- {out_dir / 'controlled_experiment_config.yml'}")
    print(f"- {out_dir / 'cv_performance_comparison.csv'}")
    print(f"- {out_dir / 'feature_ranking_report.md'}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
