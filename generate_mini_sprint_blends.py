"""
Mini-sprint blend generator.
All outputs → experiments/mini_sprint_blends/
TA/EC taken from V4.4 anchor for every blend.
Blend only DRP column.

Usage:
    python generate_mini_sprint_blends.py
    python generate_mini_sprint_blends.py --et1400_path experiments/<exp_id>/submission.csv
"""
import argparse
import os
import json
import datetime
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference files
# ---------------------------------------------------------------------------
V44_PATH   = "submissions/submission_V4_4_DRP_tuned_ET_fixorder.csv"  # LB=0.3039
S9010_PATH = "submissions_batch/archive/submission_BLEND_safe_DRPlite_90_10.csv"  # LB=0.2989
S8515_PATH = "submissions_batch/archive/submission_BLEND_safe_DRPlite_85_15.csv"  # LB=0.2949
RUN134704  = "experiments/exp_20260311_134704/submission_V5_2_OOFTE_fixkeys.csv"

DRP = "Dissolved Reactive Phosphorus"
OUT_DIR = "experiments/mini_sprint_blends"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path):
    df = pd.read_csv(path)
    assert df.shape[0] == 200, f"Unexpected row count in {path}: {df.shape}"
    return df


def drp_stats(series, anchor_drp):
    corr = series.corr(anchor_drp)
    mae  = (series - anchor_drp).abs().mean()
    return dict(
        min=float(series.min()),
        max=float(series.max()),
        median=float(series.median()),
        corr_to_anchor=float(corr),
        mae_to_anchor=float(mae),
    )


def linear_blend(base_drp, other_drp, w_other):
    """(1-w)×base + w×other"""
    return (1 - w_other) * base_drp + w_other * other_drp


def rank_blend(base_drp, other_drp, w_other):
    """Rank-interpolation blend."""
    n = len(base_drp)
    r1 = base_drp.rank()
    r2 = other_drp.rank()
    blended_rank = (1 - w_other) * r1 + w_other * r2
    sorted_base = np.sort(base_drp.values)
    rank_int = np.clip(blended_rank - 1, 0, n - 1).values
    lo = np.floor(rank_int).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = rank_int - lo
    interpolated = sorted_base[lo] + frac * (sorted_base[hi] - sorted_base[lo])
    return pd.Series(interpolated, index=base_drp.index)


def save_blend(template_df, drp_series, name, audit_meta, out_dir):
    out = template_df.copy()
    out[DRP] = drp_series.values
    path = os.path.join(out_dir, f"{name}.csv")
    out.to_csv(path, index=False)
    audit_meta["output_path"] = path
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--et1400_path", default=None,
                        help="Path to ET-1400 run submission.csv (optional)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    anchor   = load(V44_PATH)
    s9010    = load(S9010_PATH)
    s8515    = load(S8515_PATH)
    run134   = load(RUN134704)

    anchor_drp  = anchor[DRP]
    s9010_drp   = s9010[DRP]
    s8515_drp   = s8515[DRP]
    run134_drp  = run134[DRP]

    # Template has TA/EC from anchor
    template = anchor.copy()

    audit = {}

    # ------------------------------------------------------------------
    # Blend 1: linear 90/10 v44 + 134704
    # ------------------------------------------------------------------
    name = "blend1_v44_90_134704_10_linear"
    drp = linear_blend(anchor_drp, run134_drp, 0.10)
    meta = {"source_a": "v44(0.90)", "source_b": "134704(0.10)", "type": "linear",
            "lb_source_a": 0.3039}
    meta.update(drp_stats(drp, anchor_drp))
    save_blend(template, drp, name, meta, OUT_DIR)
    audit[name] = meta
    print(f"[DONE] {name}: DRP=[{meta['min']:.2f},{meta['max']:.2f}]  corr={meta['corr_to_anchor']:.3f}  mae={meta['mae_to_anchor']:.3f}")

    # ------------------------------------------------------------------
    # Blend 2: linear 80/20 v44 + 134704
    # ------------------------------------------------------------------
    name = "blend2_v44_80_134704_20_linear"
    drp = linear_blend(anchor_drp, run134_drp, 0.20)
    meta = {"source_a": "v44(0.80)", "source_b": "134704(0.20)", "type": "linear",
            "lb_source_a": 0.3039}
    meta.update(drp_stats(drp, anchor_drp))
    save_blend(template, drp, name, meta, OUT_DIR)
    audit[name] = meta
    print(f"[DONE] {name}: DRP=[{meta['min']:.2f},{meta['max']:.2f}]  corr={meta['corr_to_anchor']:.3f}  mae={meta['mae_to_anchor']:.3f}")

    # ------------------------------------------------------------------
    # Blend 3: linear 70/30 v44 + 134704
    # ------------------------------------------------------------------
    name = "blend3_v44_70_134704_30_linear"
    drp = linear_blend(anchor_drp, run134_drp, 0.30)
    meta = {"source_a": "v44(0.70)", "source_b": "134704(0.30)", "type": "linear",
            "lb_source_a": 0.3039}
    meta.update(drp_stats(drp, anchor_drp))
    save_blend(template, drp, name, meta, OUT_DIR)
    audit[name] = meta
    print(f"[DONE] {name}: DRP=[{meta['min']:.2f},{meta['max']:.2f}]  corr={meta['corr_to_anchor']:.3f}  mae={meta['mae_to_anchor']:.3f}")

    # ------------------------------------------------------------------
    # Blend 4: rank-blend 80/20 v44 + 134704
    # ------------------------------------------------------------------
    name = "blend4_v44_80_134704_20_rank"
    drp = rank_blend(anchor_drp, run134_drp, 0.20)
    meta = {"source_a": "v44(0.80)", "source_b": "134704(0.20)", "type": "rank",
            "lb_source_a": 0.3039}
    meta.update(drp_stats(drp, anchor_drp))
    save_blend(template, drp, name, meta, OUT_DIR)
    audit[name] = meta
    print(f"[DONE] {name}: DRP=[{meta['min']:.2f},{meta['max']:.2f}]  corr={meta['corr_to_anchor']:.3f}  mae={meta['mae_to_anchor']:.3f}")

    # ------------------------------------------------------------------
    # Blend 5: rank-blend 70/30 v44 + 134704
    # ------------------------------------------------------------------
    name = "blend5_v44_70_134704_30_rank"
    drp = rank_blend(anchor_drp, run134_drp, 0.30)
    meta = {"source_a": "v44(0.70)", "source_b": "134704(0.30)", "type": "rank",
            "lb_source_a": 0.3039}
    meta.update(drp_stats(drp, anchor_drp))
    save_blend(template, drp, name, meta, OUT_DIR)
    audit[name] = meta
    print(f"[DONE] {name}: DRP=[{meta['min']:.2f},{meta['max']:.2f}]  corr={meta['corr_to_anchor']:.3f}  mae={meta['mae_to_anchor']:.3f}")

    # ------------------------------------------------------------------
    # Blend 6: v44 90% + ET-1400 10% linear  (requires new run)
    # ------------------------------------------------------------------
    if args.et1400_path and os.path.exists(args.et1400_path):
        et1400 = load(args.et1400_path)
        et1400_drp = et1400[DRP]
        et1400_corr = et1400_drp.corr(anchor_drp)
        print(f"\n[INFO] ET-1400 DRP stats: [{et1400_drp.min():.2f},{et1400_drp.max():.2f}]  med={et1400_drp.median():.2f}  corr_to_v44={et1400_corr:.3f}")

        name = "blend6_v44_90_et1400_10_linear"
        drp = linear_blend(anchor_drp, et1400_drp, 0.10)
        meta = {"source_a": "v44(0.90)", "source_b": f"et1400({args.et1400_path})(0.10)",
                "type": "linear", "lb_source_a": 0.3039}
        meta.update(drp_stats(drp, anchor_drp))
        save_blend(template, drp, name, meta, OUT_DIR)
        audit[name] = meta
        print(f"[DONE] {name}: DRP=[{meta['min']:.2f},{meta['max']:.2f}]  corr={meta['corr_to_anchor']:.3f}  mae={meta['mae_to_anchor']:.3f}")

        # Also 80/20
        name = "blend7_v44_80_et1400_20_linear"
        drp = linear_blend(anchor_drp, et1400_drp, 0.20)
        meta = {"source_a": "v44(0.80)", "source_b": f"et1400({args.et1400_path})(0.20)",
                "type": "linear", "lb_source_a": 0.3039}
        meta.update(drp_stats(drp, anchor_drp))
        save_blend(template, drp, name, meta, OUT_DIR)
        audit[name] = meta
        print(f"[DONE] {name}: DRP=[{meta['min']:.2f},{meta['max']:.2f}]  corr={meta['corr_to_anchor']:.3f}  mae={meta['mae_to_anchor']:.3f}")
    else:
        print("[SKIP] blend6/7 — ET-1400 path not provided or not found yet.")
        print("       Re-run with:  --et1400_path experiments/<exp_id>/submission.csv")

    # ------------------------------------------------------------------
    # Save audit report
    # ------------------------------------------------------------------
    audit_path = os.path.join(OUT_DIR, "audit_report.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[AUDIT] Report written to {audit_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Name':<45} {'corr_to_v44':>12} {'mae_to_v44':>11} {'DRP min':>8} {'DRP max':>8}")
    print("=" * 90)
    for name, m in audit.items():
        print(f"{name:<45} {m['corr_to_anchor']:>12.4f} {m['mae_to_anchor']:>11.3f} {m['min']:>8.2f} {m['max']:>8.2f}")
    print("=" * 90)
    print("\nNote: v44 anchor LB=0.3039. Blends closest to v44 (corr→1, mae→0) tend to be safest.")
    print("All blends use TA/EC from v44. Only DRP is blended.")


if __name__ == "__main__":
    main()
