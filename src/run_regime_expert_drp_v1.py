#!/usr/bin/env python
# coding: utf-8
"""
PET-gated DRP expert pipeline (guard-focused, single controlled run).

Modes:
- Default / --dry-run: readiness diagnostics only (no training).
- --allow-train: single controlled training/evaluation run if and only if safety guards pass.

Safety properties:
- Uses immutable timestamped experiment folders.
- Never writes to existing validated submissions.
- Explicit branch fallback to global model when support guards fail.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"
DRP_COL = "Dissolved Reactive Phosphorus"
TA_COL = "Total Alkalinity"
EC_COL = "Electrical Conductance"
KEYS = [LAT_COL, LON_COL, DATE_COL]
SUBMISSION_COLS = [LAT_COL, LON_COL, DATE_COL, TA_COL, EC_COL, DRP_COL]


@dataclass
class BranchSupport:
    name: str
    total_count: int
    unique_basins: int
    per_fold_counts: list[int]
    all_folds_present: bool
    min_fold_count: int
    zero_folds: int
    count_ok: bool
    basin_ok: bool
    sparse_ok: bool
    active_expert: bool
    fallback_reason: str


@dataclass
class GuardConfig:
    min_total_count: int
    min_unique_basins: int
    min_fold_count: int


def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_keys(df: pd.DataFrame, dayfirst: bool) -> pd.DataFrame:
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    if DATE_COL in out.columns:
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=dayfirst)
    return out


def make_model(model_cfg: dict) -> Pipeline:
    if model_cfg.get("type") != "ExtraTreesRegressor":
        raise ValueError("Only ExtraTreesRegressor is supported in v1 controlled run.")
    params = model_cfg["params"]
    model = ExtraTreesRegressor(
        n_estimators=int(params["n_estimators"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        random_state=int(params["random_state"]),
        n_jobs=int(params["n_jobs"]),
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", model),
    ])


def add_feature_pack(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[DATE_COL]):
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)

    # Temporal
    out["month"] = out[DATE_COL].dt.month
    out["year"] = out[DATE_COL].dt.year
    doy = out[DATE_COL].dt.dayofyear
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.0)

    # Regime flags
    m12 = int(cfg["thresholds"]["month_flag"])
    pet_thr = float(cfg["thresholds"]["pet_high_threshold"])
    dist_thr = float(cfg["thresholds"]["dist_main_high_threshold"])

    out["is_m12"] = (out["month"] == m12).astype(int)
    out["is_pet_high"] = (out["pet"] > pet_thr).astype(int)
    out["is_dist_high"] = (out["dist_main_km"] > dist_thr).astype(int)
    out["is_pet_high_x_m12"] = out["is_pet_high"] * out["is_m12"]
    out["is_dist_high_x_m12"] = out["is_dist_high"] * out["is_m12"]

    # Gate columns
    out["pet_high"] = out["pet"] > pet_thr
    out["pet_low"] = ~out["pet_high"]

    return out


def get_feature_cols(cfg: dict) -> list[str]:
    cols = []
    if cfg["features"].get("include_temporal", True):
        cols += ["month", "sin_doy", "cos_doy", "year"]
    if cfg["features"].get("include_climate", True):
        cols += ["pet"]
    if cfg["features"].get("include_hydro_subset", True):
        cols += ["dist_main_km", "basin_area_km2", "log_upstream_area_km2"]
    if cfg["features"].get("include_regime_flags", True):
        cols += ["is_m12", "is_pet_high", "is_dist_high", "is_pet_high_x_m12", "is_dist_high_x_m12"]
    return cols


def evaluate_branch_support(
    df: pd.DataFrame,
    fold_col: str,
    branch_col: str,
    guard: GuardConfig,
) -> BranchSupport:
    folds = sorted(df[fold_col].dropna().unique().tolist())
    sub = df.loc[df[branch_col]].copy()

    total_count = int(len(sub))
    unique_basins = int(sub["basin_id"].nunique())

    fold_counts_series = sub[fold_col].value_counts()
    per_fold_counts = [int(fold_counts_series.get(f, 0)) for f in folds]

    all_folds_present = all(c > 0 for c in per_fold_counts)
    min_fold_count = min(per_fold_counts) if per_fold_counts else 0
    zero_folds = sum(1 for c in per_fold_counts if c == 0)

    count_ok = total_count >= guard.min_total_count
    basin_ok = unique_basins >= guard.min_unique_basins
    sparse_ok = min_fold_count >= guard.min_fold_count

    active_expert = all_folds_present and count_ok and basin_ok and sparse_ok

    reasons = []
    if not all_folds_present:
        reasons.append(f"missing fold support (zero_folds={zero_folds})")
    if not count_ok:
        reasons.append(f"count<{guard.min_total_count}")
    if not basin_ok:
        reasons.append(f"unique_basins<{guard.min_unique_basins}")
    if not sparse_ok:
        reasons.append(f"min_fold_count<{guard.min_fold_count}")

    fallback_reason = "" if active_expert else "; ".join(reasons)

    return BranchSupport(
        name=branch_col,
        total_count=total_count,
        unique_basins=unique_basins,
        per_fold_counts=per_fold_counts,
        all_folds_present=all_folds_present,
        min_fold_count=min_fold_count,
        zero_folds=zero_folds,
        count_ok=count_ok,
        basin_ok=basin_ok,
        sparse_ok=sparse_ok,
        active_expert=active_expert,
        fallback_reason=fallback_reason,
    )


def to_dict_branch(b: BranchSupport) -> dict:
    return {
        "name": b.name,
        "total_count": b.total_count,
        "unique_basins": b.unique_basins,
        "per_fold_counts": b.per_fold_counts,
        "all_folds_present": b.all_folds_present,
        "min_fold_count": b.min_fold_count,
        "zero_folds": b.zero_folds,
        "count_ok": b.count_ok,
        "basin_ok": b.basin_ok,
        "sparse_ok": b.sparse_ok,
        "active_expert": b.active_expert,
        "fallback_reason": b.fallback_reason,
    }


def load_train(root: Path, cfg: dict) -> pd.DataFrame:
    train = normalize_keys(pd.read_csv(root / cfg["data"]["train_base_path"]), dayfirst=True)
    tc = normalize_keys(pd.read_csv(root / cfg["data"]["tc_train_path"]), dayfirst=True)
    train = train.merge(tc[[LAT_COL, LON_COL, DATE_COL, "pet"]], on=KEYS, how="left")

    required = [cfg["cv"]["fold_column"], "basin_id", DRP_COL, "dist_main_km", "basin_area_km2", "log_upstream_area_km2", "pet"]
    missing = [c for c in required if c not in train.columns]
    if missing:
        raise ValueError(f"Missing required train columns: {missing}")
    return train


def load_valid(root: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    template = pd.read_csv(root / cfg["data"]["template_path"])
    template_n = normalize_keys(template, dayfirst=True)

    tc = normalize_keys(pd.read_csv(root / cfg["data"]["tc_valid_path"]), dayfirst=True)
    geo = normalize_keys(pd.read_csv(root / cfg["data"]["geo_features_path"]), dayfirst=True)

    valid = template_n[[LAT_COL, LON_COL, DATE_COL]].copy()
    valid = valid.merge(tc[[LAT_COL, LON_COL, DATE_COL, "pet"]], on=KEYS, how="left")
    valid = valid.merge(
        geo[[LAT_COL, LON_COL, "dist_main_km", "basin_area_km2", "log_upstream_area_km2"]],
        on=[LAT_COL, LON_COL],
        how="left",
    )

    return template, valid


def ensure_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def run_controlled_training(train_df: pd.DataFrame, cfg: dict, guard: GuardConfig, feat_cols: list[str]) -> tuple[dict, pd.DataFrame]:
    fold_col = cfg["cv"]["fold_column"]
    folds = sorted(train_df[fold_col].dropna().unique().tolist())

    y = train_df[DRP_COL].astype(float).values
    oof_global = np.full(len(train_df), np.nan)
    oof_gated = np.full(len(train_df), np.nan)
    oof_route = np.array(["" for _ in range(len(train_df))], dtype=object)

    global_scores = []
    gated_scores = []
    gate_rows = []

    for fold in folds:
        tr_mask = train_df[fold_col] != fold
        va_mask = train_df[fold_col] == fold

        tr = train_df.loc[tr_mask].copy()
        va = train_df.loc[va_mask].copy()

        y_tr = tr[DRP_COL].astype(float).values
        y_va = va[DRP_COL].astype(float).values

        # Global model always fitted
        global_model = make_model(cfg["models"]["global_drp"])
        global_model.fit(tr[feat_cols], y_tr)
        pred_global = np.maximum(global_model.predict(va[feat_cols]), float(cfg["inference"]["drp_clip_min"]))

        # Branch support evaluated on fold-train only
        high_support = evaluate_branch_support(tr, fold_col, "pet_high", guard)
        low_support = evaluate_branch_support(tr, fold_col, "pet_low", guard)

        pred_gated = pred_global.copy()
        route = np.array(["global" for _ in range(len(va))], dtype=object)

        hi_mask_va = va["pet_high"].values
        lo_mask_va = va["pet_low"].values

        if high_support.active_expert and hi_mask_va.any():
            tr_hi = tr.loc[tr["pet_high"]]
            m_hi = make_model(cfg["models"]["expert_high"])
            m_hi.fit(tr_hi[feat_cols], tr_hi[DRP_COL].astype(float).values)
            pred_hi = np.maximum(m_hi.predict(va.loc[hi_mask_va, feat_cols]), float(cfg["inference"]["drp_clip_min"]))
            pred_gated[hi_mask_va] = pred_hi
            route[hi_mask_va] = "expert_high"
        elif hi_mask_va.any():
            route[hi_mask_va] = "fallback_global_high"

        if low_support.active_expert and lo_mask_va.any():
            tr_lo = tr.loc[tr["pet_low"]]
            m_lo = make_model(cfg["models"]["expert_low"])
            m_lo.fit(tr_lo[feat_cols], tr_lo[DRP_COL].astype(float).values)
            pred_lo = np.maximum(m_lo.predict(va.loc[lo_mask_va, feat_cols]), float(cfg["inference"]["drp_clip_min"]))
            pred_gated[lo_mask_va] = pred_lo
            route[lo_mask_va] = "expert_low"
        elif lo_mask_va.any():
            route[lo_mask_va] = "fallback_global_low"

        sc_global = float(r2_score(y_va, pred_global))
        sc_gated = float(r2_score(y_va, pred_gated))

        global_scores.append(sc_global)
        gated_scores.append(sc_gated)

        idx_va = np.where(va_mask.values)[0]
        oof_global[idx_va] = pred_global
        oof_gated[idx_va] = pred_gated
        oof_route[idx_va] = route

        gate_rows.append(
            {
                "fold": int(fold),
                "global_r2": sc_global,
                "gated_r2": sc_gated,
                "pet_high_active": bool(high_support.active_expert),
                "pet_low_active": bool(low_support.active_expert),
                "pet_high_fallback_reason": high_support.fallback_reason,
                "pet_low_fallback_reason": low_support.fallback_reason,
                "pet_high_per_fold": high_support.per_fold_counts,
                "pet_low_per_fold": low_support.per_fold_counts,
            }
        )

    cv_report = {
        "drp_global_cv": {
            "mean": float(np.mean(global_scores)),
            "std": float(np.std(global_scores)),
            "folds": [float(x) for x in global_scores],
        },
        "drp_gated_cv": {
            "mean": float(np.mean(gated_scores)),
            "std": float(np.std(gated_scores)),
            "folds": [float(x) for x in gated_scores],
        },
        "overall_mean_cv": float(np.mean(gated_scores)),
    }

    oof = train_df[[LAT_COL, LON_COL, DATE_COL, DRP_COL, "pet_high", "pet_low"]].copy()
    oof["pred_global_drp"] = oof_global
    oof["pred_gated_drp"] = oof_gated
    oof["route"] = oof_route

    gate_report = {
        "fold_gate_summary": gate_rows,
        "branches_active_all_folds": {
            "pet_high": bool(all(r["pet_high_active"] for r in gate_rows)),
            "pet_low": bool(all(r["pet_low_active"] for r in gate_rows)),
        },
    }

    return {"cv_report": cv_report, "gate_report": gate_report}, oof


def fit_final_and_predict(train_df: pd.DataFrame, valid_df: pd.DataFrame, cfg: dict, guard: GuardConfig, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    global_model = make_model(cfg["models"]["global_drp"])
    global_model.fit(train_df[feat_cols], train_df[DRP_COL].astype(float).values)
    pred_global = np.maximum(global_model.predict(valid_df[feat_cols]), float(cfg["inference"]["drp_clip_min"]))

    fold_col = cfg["cv"]["fold_column"]
    high_support = evaluate_branch_support(train_df, fold_col, "pet_high", guard)
    low_support = evaluate_branch_support(train_df, fold_col, "pet_low", guard)

    pred_gated = pred_global.copy()
    hi_mask = valid_df["pet_high"].values
    lo_mask = valid_df["pet_low"].values

    if high_support.active_expert and hi_mask.any():
        tr_hi = train_df.loc[train_df["pet_high"]]
        m_hi = make_model(cfg["models"]["expert_high"])
        m_hi.fit(tr_hi[feat_cols], tr_hi[DRP_COL].astype(float).values)
        pred_gated[hi_mask] = np.maximum(m_hi.predict(valid_df.loc[hi_mask, feat_cols]), float(cfg["inference"]["drp_clip_min"]))

    if low_support.active_expert and lo_mask.any():
        tr_lo = train_df.loc[train_df["pet_low"]]
        m_lo = make_model(cfg["models"]["expert_low"])
        m_lo.fit(tr_lo[feat_cols], tr_lo[DRP_COL].astype(float).values)
        pred_gated[lo_mask] = np.maximum(m_lo.predict(valid_df.loc[lo_mask, feat_cols]), float(cfg["inference"]["drp_clip_min"]))

    gate_status = {
        "pet_high_active_final": bool(high_support.active_expert),
        "pet_low_active_final": bool(low_support.active_expert),
        "pet_high_fallback_reason": high_support.fallback_reason,
        "pet_low_fallback_reason": low_support.fallback_reason,
    }

    return pred_global, pred_gated, gate_status


def build_submission(template_raw: pd.DataFrame, anchor: pd.DataFrame, pred_global: np.ndarray, pred_gated: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict]:
    out = template_raw.copy(deep=True)
    out[TA_COL] = anchor[TA_COL].values
    out[EC_COL] = anchor[EC_COL].values

    if cfg["inference"]["blend_with_anchor"].get("enabled", True):
        alpha = float(cfg["inference"]["blend_with_anchor"]["alpha"])
        drp = anchor[DRP_COL].values.astype(float) + alpha * (pred_gated - pred_global)
    else:
        drp = pred_gated

    drp = np.maximum(drp, float(cfg["inference"]["drp_clip_min"]))
    out[DRP_COL] = drp

    out = out[SUBMISSION_COLS].copy()

    if out.shape != (200, 6):
        raise RuntimeError(f"Unexpected submission shape: {out.shape}")
    if out.isna().sum().sum() > 0:
        raise RuntimeError("Submission has NaN values.")
    if (out[DRP_COL] < 0).any():
        raise RuntimeError("Submission has negative DRP values.")

    corr = float(np.corrcoef(anchor[DRP_COL].values.astype(float), out[DRP_COL].values.astype(float))[0, 1])
    mae = float(np.mean(np.abs(anchor[DRP_COL].values.astype(float) - out[DRP_COL].values.astype(float))))
    stats = {
        "drp_mean": float(out[DRP_COL].mean()),
        "drp_std": float(out[DRP_COL].std()),
        "drp_min": float(out[DRP_COL].min()),
        "drp_max": float(out[DRP_COL].max()),
        "corr_to_anchor": corr,
        "mae_to_anchor": mae,
    }
    return out, stats


def write_readiness_markdown(path: Path, payload: dict) -> None:
    hi = payload["support"]["pet_high"]
    lo = payload["support"]["pet_low"]
    guard = payload["guard_thresholds"]

    lines = [
        "# PET-Gated DRP Expert Readiness Report",
        "",
        f"Generated at: {payload['generated_at']}",
        f"Config: {payload['config_path']}",
        "",
        "## Safety Verdict",
        f"- safe_to_run_controlled_training: {payload['safe_to_run_controlled_training']}",
        "",
        "## Guard Thresholds",
        f"- min_total_count_per_branch: {guard['min_total_count_per_branch']}",
        f"- min_unique_basins_per_branch: {guard['min_unique_basins_per_branch']}",
        f"- min_fold_count_per_branch: {guard['min_fold_count_per_branch']}",
        "",
        "## Branch Support",
        f"- pet_high: total={hi['total_count']}, basins={hi['unique_basins']}, per_fold={hi['per_fold_counts']}, active_expert={hi['active_expert']}",
        f"- pet_low: total={lo['total_count']}, basins={lo['unique_basins']}, per_fold={lo['per_fold_counts']}, active_expert={lo['active_expert']}",
        "",
        "## Fallback Logic",
        "- Always train/use global DRP model.",
        "- Route by PET gate only: pet > 198.8 => pet_high branch, else pet_low branch.",
        "- Activate branch expert only when all guard checks pass.",
        "- If a branch fails any guard, route that branch to global model (hard fallback).",
        "- If both branches fail, run global-only DRP path.",
        "",
        "## Planned Artifact Paths (when training is executed later)",
    ]
    for k, v in payload["planned_artifacts"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Launch Command (later)",
        f"- {payload['launch_command']}",
        "",
        "## Notes",
        "- This run was dry-run only. No model training or submission generation occurred.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_exp_regime_expert_drp_v1.yml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-train", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / args.config
    if not cfg_path.exists():
        cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    prepared_only = bool(cfg.get("execution", {}).get("prepared_only", True))
    if args.allow_train and prepared_only:
        raise RuntimeError("Training blocked: execution.prepared_only=true in config.")

    train = add_feature_pack(load_train(root, cfg), cfg)
    fold_col = cfg["cv"]["fold_column"]
    folds = sorted(train[fold_col].dropna().unique().tolist())
    if len(folds) != int(cfg["cv"]["n_folds"]):
        raise RuntimeError(f"Expected {cfg['cv']['n_folds']} folds, found {len(folds)}")

    guard = GuardConfig(
        min_total_count=int(cfg["strategy"]["fallback_global_if_count_below"]),
        min_unique_basins=int(cfg["strategy"]["fallback_global_if_unique_basins_below"]),
        min_fold_count=int(cfg["strategy"]["fallback_global_if_min_fold_count_below"]),
    )

    high = evaluate_branch_support(train, fold_col, "pet_high", guard)
    low = evaluate_branch_support(train, fold_col, "pet_low", guard)
    safe_to_run = bool(high.active_expert and low.active_expert)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Readiness mode by default
    if args.dry_run or not args.allow_train:
        run_id = f"regime_expert_drp_v1_readiness_{timestamp}"
        out_dir = root / cfg["output"]["exp_dir"] / run_id
        out_dir.mkdir(parents=True, exist_ok=False)

        payload = {
            "generated_at": ts(),
            "config_path": str(cfg_path),
            "prepared_only": prepared_only,
            "safe_to_run_controlled_training": safe_to_run,
            "guard_thresholds": {
                "min_total_count_per_branch": guard.min_total_count,
                "min_unique_basins_per_branch": guard.min_unique_basins,
                "min_fold_count_per_branch": guard.min_fold_count,
            },
            "support": {
                "pet_high": to_dict_branch(high),
                "pet_low": to_dict_branch(low),
            },
            "fallback_logic": {
                "route_pet_high": "expert_high if active else global",
                "route_pet_low": "expert_low if active else global",
                "if_both_inactive": "global_only",
            },
            "planned_artifacts": {
                "run_dir": str(root / cfg["output"]["exp_dir"] / f"regime_expert_drp_v1_{timestamp}"),
                "cv_report": str(root / cfg["output"]["exp_dir"] / f"regime_expert_drp_v1_{timestamp}" / "cv_report.json"),
                "gate_report": str(root / cfg["output"]["exp_dir"] / f"regime_expert_drp_v1_{timestamp}" / "gate_report.json"),
                "oof_predictions": str(root / cfg["output"]["exp_dir"] / f"regime_expert_drp_v1_{timestamp}" / "oof_predictions.csv"),
                "metadata": str(root / cfg["output"]["exp_dir"] / f"regime_expert_drp_v1_{timestamp}" / "metadata.json"),
                "submission": str(root / cfg["output"]["exp_dir"] / f"regime_expert_drp_v1_{timestamp}" / f"{cfg['output']['submission_prefix']}.csv"),
            },
            "launch_command": ".venv\\Scripts\\python.exe src/run_regime_expert_drp_v1.py --config config_exp_regime_expert_drp_v1.yml --allow-train",
        }

        json_path = out_dir / "readiness_report.json"
        md_path = out_dir / "readiness_report.md"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        write_readiness_markdown(md_path, payload)

        print("=" * 72)
        print("PET-GATED DRP EXPERT READINESS (DRY-RUN)")
        print("=" * 72)
        print(f"safe_to_run_controlled_training: {safe_to_run}")
        print(f"pet_high active_expert: {high.active_expert} | per_fold={high.per_fold_counts}")
        print(f"pet_low  active_expert: {low.active_expert} | per_fold={low.per_fold_counts}")
        print(f"readiness_report_json: {json_path}")
        print(f"readiness_report_md:   {md_path}")
        print("No training executed.")
        print("=" * 72)
        return 0

    # Controlled training mode
    if not safe_to_run:
        raise RuntimeError("Safety guards failed. Training aborted.")

    run_id = f"regime_expert_drp_v1_{timestamp}"
    out_dir = root / cfg["output"]["exp_dir"] / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    feat_cols = get_feature_cols(cfg)
    train = ensure_features(train, feat_cols)

    reports, oof = run_controlled_training(train, cfg, guard, feat_cols)

    template_raw, valid = load_valid(root, cfg)
    valid = add_feature_pack(valid, cfg)
    valid = ensure_features(valid, feat_cols)

    pred_global, pred_gated, gate_final = fit_final_and_predict(train, valid, cfg, guard, feat_cols)

    anchor = pd.read_csv(root / cfg["anchor"]["submission_path"])
    submission, sub_stats = build_submission(template_raw, anchor, pred_global, pred_gated, cfg)

    cv_report = {
        "generated_at": ts(),
        "run_id": run_id,
        "drp": reports["cv_report"],
    }
    gate_report = {
        "generated_at": ts(),
        "run_id": run_id,
        "global_fallback_logic": {
            "route_pet_high": "expert_high if active else global",
            "route_pet_low": "expert_low if active else global",
            "if_both_inactive": "global_only",
        },
        "guard_thresholds": {
            "min_total_count_per_branch": guard.min_total_count,
            "min_unique_basins_per_branch": guard.min_unique_basins,
            "min_fold_count_per_branch": guard.min_fold_count,
        },
        "branch_support_full_train": {
            "pet_high": to_dict_branch(high),
            "pet_low": to_dict_branch(low),
        },
        "fold_gate_summary": reports["gate_report"]["fold_gate_summary"],
        "branches_active_all_folds": reports["gate_report"]["branches_active_all_folds"],
        "final_fit_branch_status": gate_final,
    }

    metadata = {
        "generated_at": ts(),
        "run_id": run_id,
        "config_path": str(cfg_path),
        "feature_columns": feat_cols,
        "prepared_only_config": prepared_only,
        "safe_to_run_before_training": safe_to_run,
        "drp_cv_gated_mean": reports["cv_report"]["drp_gated_cv"]["mean"],
        "drp_cv_global_mean": reports["cv_report"]["drp_global_cv"]["mean"],
        "overall_mean_cv": reports["cv_report"]["overall_mean_cv"],
        "submission_stats": sub_stats,
        "submission_file": f"{cfg['output']['submission_prefix']}.csv",
    }

    (out_dir / "oof_predictions.csv").write_text(oof.to_csv(index=False), encoding="utf-8")
    with open(out_dir / "cv_report.json", "w", encoding="utf-8") as f:
        json.dump(cv_report, f, indent=2)
    with open(out_dir / "gate_report.json", "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=2)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    submission.to_csv(out_dir / f"{cfg['output']['submission_prefix']}.csv", index=False)

    print("=" * 72)
    print("PET-GATED DRP EXPERT CONTROLLED RUN")
    print("=" * 72)
    print(f"run_dir: {out_dir}")
    print(f"DRP CV (gated mean): {reports['cv_report']['drp_gated_cv']['mean']:.6f}")
    print(f"Overall mean CV: {reports['cv_report']['overall_mean_cv']:.6f}")
    print(f"Branches active all folds: {reports['gate_report']['branches_active_all_folds']}")
    print(f"submission: {out_dir / (cfg['output']['submission_prefix'] + '.csv')}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
