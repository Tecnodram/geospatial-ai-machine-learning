import os
import ee
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
PROJECT_ID = "ey-water-quality-2026-489102"
RAW_DIR = r"C:\Projects\ey-water-quality-2026\data\raw"
OUT_PATH = r"data/external_geofeatures.csv"

BATCH_SIZE = 300   # si falla por tiempo/memoria, baja a 150
SCALE = 90         # escala segura (SRTM ~30m, SoilGrids ~250m, WorldCover 10m)

# =========================
# INIT
# =========================
ee.Initialize(project=PROJECT_ID)
os.makedirs("data", exist_ok=True)

# =========================
# 1) Coordenadas únicas
# =========================
train = pd.read_csv(os.path.join(RAW_DIR, "water_quality_training_dataset.csv"))
valid = pd.read_csv(os.path.join(RAW_DIR, "submission_template.csv"))

coords = pd.concat(
    [train[["Latitude", "Longitude"]], valid[["Latitude", "Longitude"]]],
    ignore_index=True
).drop_duplicates().reset_index(drop=True)

print("Unique coordinate points:", len(coords))

# =========================
# 2) Datasets (con selección de 1 banda)
# =========================

# DEM + slope
dem = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation")
terrain = ee.Terrain.products(dem)
slope = terrain.select("slope").rename("slope")

# SoilGrids: tienen varias bandas por profundidad -> seleccionamos 0–5 cm
soil_ph = (
    ee.Image("projects/soilgrids-isric/phh2o_mean")
    .select("phh2o_0-5cm_mean")
    .rename("soil_ph_0_5")
)

soil_clay = (
    ee.Image("projects/soilgrids-isric/clay_mean")
    .select("clay_0-5cm_mean")
    .rename("soil_clay_0_5")
)

soil_soc = (
    ee.Image("projects/soilgrids-isric/soc_mean")
    .select("soc_0-5cm_mean")
    .rename("soil_soc_0_5")
)

# WorldCover: 1 sola banda "Map"
landcover = (
    ee.Image("ESA/WorldCover/v100/2020")
    .select("Map")
    .rename("landcover")
)

# Stack final: todas 1 banda cada una
stack = dem.addBands([slope, soil_ph, soil_clay, soil_soc, landcover])

def chunker(df, size):
    for i in range(0, len(df), size):
        yield i, df.iloc[i:i+size]

# =========================
# 3) Extracción por lotes
# =========================
all_rows = []

for start_idx, part in chunker(coords, BATCH_SIZE):
    end_idx = start_idx + len(part) - 1
    print(f"Batch {start_idx}–{end_idx} ...", flush=True)

    feats = []
    for _, r in part.iterrows():
        pt = ee.Geometry.Point([float(r["Longitude"]), float(r["Latitude"])])
        feats.append(ee.Feature(pt))

    fc = ee.FeatureCollection(feats)

    # Reducer.first: toma el primer pixel (punto)
    results = stack.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=SCALE
    )

    data = results.getInfo()
    rows = [f["properties"] for f in data["features"]]

    out = pd.DataFrame(rows)
    out["Latitude"] = part["Latitude"].values
    out["Longitude"] = part["Longitude"].values

    all_rows.append(out)
    print(f"  -> got {len(out)} rows", flush=True)

external_df = pd.concat(all_rows, ignore_index=True)

# =========================
# 4) Guardar + sanity
# =========================
external_df.to_csv(OUT_PATH, index=False)

print("✅ Saved:", OUT_PATH)
print("shape:", external_df.shape)

# Checa NA rates (normal que haya algunos NA si cae en agua/edge, luego imputamos)
print("Top NA rates:")
print(external_df.isna().mean().sort_values(ascending=False).head(12))
print("\nColumns:", external_df.columns.tolist())