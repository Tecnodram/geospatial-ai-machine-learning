from pathlib import Path
import json
import pandas as pd


def safe_load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


def safe_float(value):
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def get_overall_mean(cv_data, ta, ec, drp):
    # Prefer explicit overall fields if present; otherwise compute from target means.
    for key in ("overall_mean", "mean", "overall"):
        v = cv_data.get(key)
        if isinstance(v, (int, float)):
            return float(v)

    vals = pd.Series([ta, ec, drp], dtype="float64")
    if vals.notna().any():
        return float(vals.mean(skipna=True))
    return float("nan")


def extract_row(exp_dir):
    cv_path = exp_dir / "cv_report.json"
    cfg_path = exp_dir / "config_snapshot.json"

    row = {
        "run_id": exp_dir.name,
        "TA": float("nan"),
        "EC": float("nan"),
        "DRP": float("nan"),
        "Mean": float("nan"),
        "te": None,
        "drp_mode": None,
        "drp_model": None,
        "n_est": None,
        "ext_file": None,
    }

    try:
        cv_data = safe_load_json(cv_path)
        if isinstance(cv_data, dict):
            ta = safe_float(cv_data.get("Total Alkalinity", {}).get("mean"))
            ec = safe_float(cv_data.get("Electrical Conductance", {}).get("mean"))
            drp = safe_float(cv_data.get("Dissolved Reactive Phosphorus", {}).get("mean"))

            row["TA"] = ta
            row["EC"] = ec
            row["DRP"] = drp
            row["Mean"] = get_overall_mean(cv_data, ta, ec, drp)
    except Exception as e:
        print(f"[WARN] Failed parsing cv_report for {exp_dir.name}: {e}")

    try:
        cfg_data = safe_load_json(cfg_path)
        if isinstance(cfg_data, dict):
            row["te"] = cfg_data.get("te", {}).get("enabled")
            row["drp_mode"] = (
                cfg_data.get("targets", {})
                .get("y_mode_by_target", {})
                .get("Dissolved Reactive Phosphorus")
            )
            row["drp_model"] = (
                cfg_data.get("model", {})
                .get("cv_by_target", {})
                .get("Dissolved Reactive Phosphorus", {})
                .get("name")
            )
            row["n_est"] = cfg_data.get("model", {}).get("et_cv", {}).get("n_estimators")
            ext_path = cfg_data.get("project", {}).get("external_path")
            row["ext_file"] = Path(ext_path).name if ext_path else None
    except Exception as e:
        print(f"[WARN] Failed parsing config_snapshot for {exp_dir.name}: {e}")

    return row


def main():
    base = Path("experiments")
    if not base.exists():
        print("[ERROR] experiments/ folder not found.")
        return

    cv_reports = sorted(base.rglob("cv_report.json"))
    if not cv_reports:
        print("[INFO] No cv_report.json files found under experiments/.")
        return

    rows = []
    for cv_file in cv_reports:
        exp_dir = cv_file.parent
        rows.append(extract_row(exp_dir))

    df = pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "TA",
            "EC",
            "DRP",
            "Mean",
            "te",
            "drp_mode",
            "drp_model",
            "n_est",
            "ext_file",
        ],
    )

    df = df.sort_values("Mean", ascending=False, na_position="last").reset_index(drop=True)

    print("\n=== FULL EXPERIMENT AUDIT (sorted by Mean desc) ===")
    print(df.to_string(index=False))

    print("\n=== TOP 5 BEST EXPERIMENTS ===")
    print(df.head(5).to_string(index=False))

    print("\n=== WORST 3 EXPERIMENTS ===")
    print(df.tail(3).to_string(index=False))

    out_csv = base / "audit_full_history.csv"
    try:
        df.to_csv(out_csv, index=False)
        print(f"\nSaved audit CSV: {out_csv}")
    except Exception as e:
        print(f"[ERROR] Could not write CSV {out_csv}: {e}")


if __name__ == "__main__":
    main()
