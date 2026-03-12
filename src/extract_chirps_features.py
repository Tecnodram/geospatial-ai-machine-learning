import os
import ee
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
PROJECT_ID = "ey-water-quality-2026-489102"
RAW_DIR = r"data/raw"
OUT_DIR = r"data/external"

TRAIN_WQ = "water_quality_training_dataset.csv"
VALID_TEMPLATE = "submission_template.csv"

OUT_TRAIN = os.path.join(OUT_DIR, "chirps_features_training.csv")
OUT_VALID = os.path.join(OUT_DIR, "chirps_features_validation.csv")

BATCH_SIZE = 300
SCALE = 5566  # CHIRPS ~0.05 degrees (~5.6 km)

LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"

# =========================
# INIT
# =========================
os.makedirs(OUT_DIR, exist_ok=True)

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    print("Earth Engine no estaba autenticado. Lanzando ee.Authenticate()...")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

# =========================
# HELPERS
# =========================
def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out[LAT_COL], errors="coerce")
    out[LON_COL] = pd.to_numeric(out[LON_COL], errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    return out

def chunker(df, size):
    for i in range(0, len(df), size):
        yield i, df.iloc[i:i + size]

def extract_for_rows(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae CHIRPS daily precipitation para cada fila (lat, lon, date).
    Devuelve:
      Latitude, Longitude, Sample Date, chirps_ppt
    """
    df = normalize_keys(df_in)
    df = df.dropna(subset=[LAT_COL, LON_COL, DATE_COL]).copy()
    df["_row_id"] = np.arange(len(df))

    all_rows = []

    for start_idx, part in chunker(df, BATCH_SIZE):
        end_idx = start_idx + len(part) - 1
        print(f"Batch {start_idx}-{end_idx} ...", flush=True)

        feats = []
        for _, r in part.iterrows():
            pt = ee.Geometry.Point([float(r[LON_COL]), float(r[LAT_COL])])

            start_date = ee.Date(r[DATE_COL].strftime("%Y-%m-%d"))
            end_date = start_date.advance(1, "day")

            img = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterDate(start_date, end_date)
                .select("precipitation")
                .first()
            )

            base_feat = ee.Feature(
                pt,
                {
                    "_row_id": int(r["_row_id"]),
                    LAT_COL: float(r[LAT_COL]),
                    LON_COL: float(r[LON_COL]),
                    DATE_COL: r[DATE_COL].strftime("%Y-%m-%d"),
                },
            )

            feat = ee.Algorithms.If(
                img,
                ee.Feature(
                    base_feat.geometry(),
                    ee.Dictionary(base_feat.toDictionary()).combine(
                        img.reduceRegion(
                            reducer=ee.Reducer.first(),
                            geometry=pt,
                            scale=SCALE,
                        )
                    ),
                ),
                base_feat,
            )

            feats.append(feat)

        fc = ee.FeatureCollection(feats)
        data = fc.getInfo()

        rows = [f["properties"] for f in data["features"]]
        out = pd.DataFrame(rows)

        if "precipitation" in out.columns:
            out = out.rename(columns={"precipitation": "chirps_ppt"})
        elif "first" in out.columns:
            out = out.rename(columns={"first": "chirps_ppt"})
        else:
            out["chirps_ppt"] = np.nan

        all_rows.append(out)
        print(f"  -> got {len(out)} rows", flush=True)

    result = pd.concat(all_rows, ignore_index=True)

    result[LAT_COL] = pd.to_numeric(result[LAT_COL], errors="coerce")
    result[LON_COL] = pd.to_numeric(result[LON_COL], errors="coerce")
    result[DATE_COL] = pd.to_datetime(result[DATE_COL], errors="coerce")
    result["chirps_ppt"] = pd.to_numeric(result["chirps_ppt"], errors="coerce")

    result = result.drop(columns=["_row_id"], errors="ignore")

    return result[[LAT_COL, LON_COL, DATE_COL, "chirps_ppt"]]

# =========================
# LOAD INPUTS
# =========================
train = pd.read_csv(os.path.join(RAW_DIR, TRAIN_WQ))
valid = pd.read_csv(os.path.join(RAW_DIR, VALID_TEMPLATE))

train = normalize_keys(train)
valid = normalize_keys(valid)

train_keys = train[[LAT_COL, LON_COL, DATE_COL]].copy()
valid_keys = valid[[LAT_COL, LON_COL, DATE_COL]].copy()

print("Train rows:", len(train_keys))
print("Valid rows:", len(valid_keys))

# =========================
# EXTRACT
# =========================
train_out = extract_for_rows(train_keys)
valid_out = extract_for_rows(valid_keys)

# =========================
# SAVE
# =========================
train_out.to_csv(OUT_TRAIN, index=False)
valid_out.to_csv(OUT_VALID, index=False)

print("\n✅ Saved:")
print(OUT_TRAIN, train_out.shape)
print(OUT_VALID, valid_out.shape)

print("\nTrain missingness:")
print(train_out.isna().mean().sort_values(ascending=False))

print("\nValid missingness:")
print(valid_out.isna().mean().sort_values(ascending=False))

print("\nTrain preview:")
print(train_out.head())