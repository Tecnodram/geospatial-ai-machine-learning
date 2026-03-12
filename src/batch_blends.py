#!/usr/bin/env python
# coding: utf-8

import os, argparse, datetime
import numpy as np
import pandas as pd

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

KEYS = ["Latitude","Longitude","Sample Date"]
TARGETS = ["Total Alkalinity","Electrical Conductance","Dissolved Reactive Phosphorus"]

def safe_clip_drp(x):
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = np.nan
    med = np.nanmedian(x)
    x = np.nan_to_num(x, nan=med)
    return np.maximum(x, 0.0)

def strict_sanity(df, template_raw, name):
    assert df.shape == (200,6), f"{name}: shape {df.shape}"
    assert not df[TARGETS].isna().any().any(), f"{name}: NaNs"
    assert df[KEYS].equals(template_raw[KEYS]), f"{name}: KEYS mismatch template raw"
    assert (df["Dissolved Reactive Phosphorus"] >= 0).all(), f"{name}: DRP negativa"

def score_proxy(df):
    # determinístico y conservador (penaliza explosiones)
    drp = df["Dissolved Reactive Phosphorus"].astype(float).values
    ec  = df["Electrical Conductance"].astype(float).values
    ta  = df["Total Alkalinity"].astype(float).values

    drp_rng = float(np.nanpercentile(drp, 99) - np.nanpercentile(drp, 1))
    ec_rng  = float(np.nanpercentile(ec, 99) - np.nanpercentile(ec, 1))
    ta_rng  = float(np.nanpercentile(ta, 99) - np.nanpercentile(ta, 1))

    penalty = 0.0
    penalty += max(0.0, drp_rng - 25.0) * 0.9
    penalty += max(0.0, ec_rng  - 900.0) * 0.02
    penalty += max(0.0, ta_rng  - 280.0) * 0.05

    drp_med = float(np.nanmedian(drp))
    penalty += abs(drp_med - 25.0) * 0.3

    return 1000.0 - penalty

def read_last_run_path(cache_dir: str):
    p = os.path.join(cache_dir, "last_run_path.txt")
    if os.path.exists(p):
        try:
            txt = open(p, "r", encoding="utf-8").read().strip()
            return txt if txt else None
        except Exception:
            return None
    return None

def append_leaderboard_log(csv_path: str, row: dict):
    # Creates file if missing
    cols = [
        "timestamp",
        "run_id",
        "recommended_name",
        "recommended_file",
        "proxy",
        "notes",
    ]
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    raw_dir = cfg["project"]["raw_dir"]
    out_dir = cfg["project"]["out_dir"]
    cache_dir = cfg["project"].get("cache_dir", "cache")
    # out_dir kept for legacy candidate lookups only; no new files written there.

    template_raw = pd.read_csv(os.path.join(raw_dir, cfg["io"]["template_name"]))

    # Load base submission from last run directory (immutable path).
    run_path = read_last_run_path(cache_dir)
    base_candidates = []
    if run_path and os.path.isdir(run_path):
        run_sub = os.path.join(run_path, "submission.csv")
        if os.path.exists(run_sub):
            base_candidates.append(run_sub)
        else:
            # Legacy: run pre-dates immutable pipeline; try old fixed name in run dir.
            legacy_run = os.path.join(run_path, "submission_V5_2_OOFTE_fixkeys.csv")
            if os.path.exists(legacy_run):
                base_candidates.append(legacy_run)

    # Legacy fallback: fixed-name file in out_dir from pre-immutable runs.
    if not base_candidates:
        legacy_out = os.path.join(out_dir, "submission_V5_2_OOFTE_fixkeys.csv")
        if os.path.exists(legacy_out):
            base_candidates.append(legacy_out)

    # Add other candidates listed in config (historical blends, anchor, etc.).
    for f in cfg["batch"]["candidates"]:
        if os.path.exists(f):
            base_candidates.append(f)

    base_candidates = list(dict.fromkeys(base_candidates))
    if not base_candidates:
        raise RuntimeError(
            "No candidates found for batch. Run train_pipeline.py first to generate "
            "a submission in the run directory, or check batch.candidates in config."
        )

    loaded = {os.path.basename(p).replace(".csv",""): pd.read_csv(p) for p in base_candidates}

    # pick base = newest preferred
    # Prefer canonical immutable name, then legacy fixed name, then first available.
    if "submission" in loaded:
        base_key = "submission"
    elif "submission_V5_2_OOFTE_fixkeys" in loaded:
        base_key = "submission_V5_2_OOFTE_fixkeys"
    else:
        base_key = list(loaded.keys())[0]
    base = loaded[base_key].copy()

    variants = {}
    variants[f"{base_key}_as_is"] = base

    # Rankblend DRP V44+V43 si existen
    if cfg["batch"]["make_rankblend_drp"]:
        k44 = "submission_V4_4_DRP_tuned_ET_fixorder"
        k43 = "submission_V4_3_ETonly_allTargets__winsor_allTargets_fixorder"
        if k44 in loaded and k43 in loaded:
            v44 = loaded[k44]; v43 = loaded[k43]
            assert v44[KEYS].equals(v43[KEYS]), "V44/V43 KEYS mismatch"

            drp1 = v44["Dissolved Reactive Phosphorus"].astype(float).values
            drp2 = v43["Dissolved Reactive Phosphorus"].astype(float).values
            r1 = pd.Series(drp1).rank(pct=True).values
            r2 = pd.Series(drp2).rank(pct=True).values
            r = 0.80*r1 + 0.20*r2
            ref = np.sort(drp1)
            idx = np.clip((r*(len(ref)-1)).astype(int), 0, len(ref)-1)
            drp_rankblend = np.maximum(ref[idx], 0.0)

            sub = v44.copy()
            sub["Dissolved Reactive Phosphorus"] = drp_rankblend
            variants["rankblend_DRP_V44V43_80_20"] = sub

    # Balanced blend con B si existe (conservador)
    if cfg["batch"]["make_balanced_blend"]:
        kb = "submission_B_ensemble_log_EC_DRP"
        k44 = "submission_V4_4_DRP_tuned_ET_fixorder"
        if (k44 in loaded) and (kb in loaded):
            v44 = loaded[k44]; b = loaded[kb]
            assert v44[KEYS].equals(b[KEYS]), "V44/B KEYS mismatch"

            sub = v44.copy()
            sub["Electrical Conductance"] = (
                0.85*v44["Electrical Conductance"].astype(float).values +
                0.15*b["Electrical Conductance"].astype(float).values
            )
            sub["Dissolved Reactive Phosphorus"] = safe_clip_drp(
                0.60*v44["Dissolved Reactive Phosphorus"].astype(float).values +
                0.40*b["Dissolved Reactive Phosphorus"].astype(float).values
            )
            variants["blend_balanced_V44B"] = sub

    # save + recommend — write to run directory only (immutable; out_dir never touched)
    blend_dir = run_path if (run_path and os.path.isdir(run_path)) else out_dir
    os.makedirs(blend_dir, exist_ok=True)
    report = []
    for name, df in variants.items():
        strict_sanity(df, template_raw, name)
        out = os.path.join(blend_dir, f"{name}.csv")
        df.to_csv(out, index=False)
        proxy = score_proxy(df)
        report.append((proxy, name, out))
        print(f"✅ Saved {name} -> {out} | proxy={proxy:.2f}")

    report.sort(key=lambda x: -x[0])
    best = report[0]
    best_proxy, best_name, best_path = best

    # -------- experiment folder integration --------
    # If train_pipeline ran, it wrote cache/last_run_path.txt.
    run_path = read_last_run_path(cache_dir)
    run_id = os.path.basename(run_path) if run_path else ""

    if run_path and os.path.isdir(run_path):
        # Record recommendation metadata (best_path already lives in run_path).
        try:
            import json as _json
            rec = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "recommended_name": best_name,
                "recommended_file": best_path,
                "proxy": float(best_proxy),
            }
            with open(os.path.join(run_path, "batch_recommendation.json"), "w", encoding="utf-8") as f:
                _json.dump(rec, f, ensure_ascii=False, indent=2)
            print(f"📌 Recommendation recorded in run folder: {best_path}")
        except Exception as e:
            print("⚠️ No pude escribir batch_recommendation.json:", e)

    # -------- leaderboard log (root) --------
    log_path = "leaderboard_log.csv"
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "recommended_name": best_name,
        "recommended_file": best_path,
        "proxy": float(best_proxy),
        "notes": "auto-recommend from batch_blends",
    }
    try:
        append_leaderboard_log(log_path, row)
        print(f"🧾 Updated {log_path}")
    except Exception as e:
        print("⚠️ No pude actualizar leaderboard_log.csv:", e)

    print("\n" + "="*80)
    print("RECOMENDACIÓN AUTOMÁTICA:")
    print("📌 Subir:", best_name)
    print("📄 Archivo:", best_path)
    print("🔎 Proxy:", f"{best_proxy:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()