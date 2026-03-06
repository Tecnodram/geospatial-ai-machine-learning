import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# ==============================
# Helpers
# ==============================

def find_one(pattern: str) -> Path:
    hits = list(Path("data/hydrology").rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No encontré {pattern} dentro de data/hydrology")
    if len(hits) > 1:
        print(f"⚠️ Encontré múltiples {pattern}, usando: {hits[0]}")
    return hits[0]

# ==============================
# Paths
# ==============================

TRAIN_PATH = Path("data/raw/water_quality_training_dataset.csv")
VALID_PATH = Path("data/raw/submission_template.csv")

# Rutas reales confirmadas por ti
RIVERS_PATH = Path(r"data/hydrology/HydroRIVERS_v10_af_shp/HydroRIVERS_v10_af.shp")
BASINS_PATH = Path(r"data/hydrology/hybas_af_lev01-12_v1c/hybas_af_lev08_v1c.shp")

# Si quieres autodetección en vez de rutas fijas, usa estas dos líneas y comenta las de arriba:
# RIVERS_PATH = find_one("HydroRIVERS_v10_af.shp")
# BASINS_PATH = find_one("hybas_af_lev08_v1c.shp")

OUT_PATH = Path("data/external_geofeatures.csv")

# ==============================
# Validación de existencia
# ==============================

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"No existe {TRAIN_PATH}")

if not VALID_PATH.exists():
    raise FileNotFoundError(f"No existe {VALID_PATH}")

if not RIVERS_PATH.exists():
    raise FileNotFoundError(f"No existe {RIVERS_PATH}")

if not BASINS_PATH.exists():
    raise FileNotFoundError(f"No existe {BASINS_PATH}")

print("Usando archivos:")
print(" TRAIN :", TRAIN_PATH)
print(" VALID :", VALID_PATH)
print(" RIVERS:", RIVERS_PATH)
print(" BASINS:", BASINS_PATH)

# ==============================
# Load points
# ==============================

train = pd.read_csv(TRAIN_PATH)
valid = pd.read_csv(VALID_PATH)

required_cols = {"Latitude", "Longitude"}
for name, df in [("train", train), ("valid", valid)]:
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {name}: {missing}")

points = pd.concat([
    train[["Latitude", "Longitude"]],
    valid[["Latitude", "Longitude"]]
]).drop_duplicates().reset_index(drop=True)

gdf_points = gpd.GeoDataFrame(
    points.copy(),
    geometry=gpd.points_from_xy(points["Longitude"], points["Latitude"]),
    crs="EPSG:4326"
)

print(f"Puntos únicos: {len(gdf_points)}")

# ==============================
# Load basins
# ==============================

basins = gpd.read_file(BASINS_PATH)
print("Columnas HydroBASINS:", basins.columns.tolist()[:20], "...")

# Detectar columnas útiles
candidate_id_cols = ["HYBAS_ID", "hybas_id", "BASIN_ID", "MAIN_BAS", "NEXT_DOWN"]
candidate_area_cols = ["AREA_SQKM", "area_sqkm", "SUB_AREA", "BAS_AREA"]

basin_id_col = None
for c in candidate_id_cols:
    if c in basins.columns:
        basin_id_col = c
        break

if basin_id_col is None:
    raise ValueError("No encontré una columna de ID de cuenca tipo HYBAS_ID en HydroBASINS.")

basin_area_col = None
for c in candidate_area_cols:
    if c in basins.columns:
        basin_area_col = c
        break

if basin_area_col is None:
    print("⚠️ No encontré AREA_SQKM. Calcularé área geométrica en km².")
    basins = basins.to_crs("EPSG:3857")
    basins["basin_area_km2"] = basins.geometry.area / 1e6
    basins = basins.to_crs("EPSG:4326")
else:
    basins["basin_area_km2"] = pd.to_numeric(basins[basin_area_col], errors="coerce")

basins = basins[[basin_id_col, "basin_area_km2", "geometry"]].copy()
basins = basins.rename(columns={basin_id_col: "basin_id"})
basins = basins.to_crs("EPSG:4326")

# Spatial join punto -> cuenca
points_basin = gpd.sjoin(
    gdf_points,
    basins,
    how="left",
    predicate="within"
)

# Quitar columnas extra del spatial join si aparecen
for col in ["index_right"]:
    if col in points_basin.columns:
        points_basin.drop(columns=[col], inplace=True)

# ==============================
# Distance to river
# ==============================

rivers = gpd.read_file(RIVERS_PATH)
print("Columnas HydroRIVERS:", rivers.columns.tolist()[:20], "...")

# Proyección métrica para distancias
rivers_m = rivers.to_crs("EPSG:3857")
points_m = gdf_points.to_crs("EPSG:3857")

# Unión geométrica para distancia mínima
river_union = rivers_m.unary_union

points_m["dist_to_river_m"] = points_m.geometry.distance(river_union)
points_basin["dist_to_river_m"] = points_m["dist_to_river_m"].values

# ==============================
# Output
# ==============================

features = points_basin[[
    "Latitude",
    "Longitude",
    "basin_id",
    "basin_area_km2",
    "dist_to_river_m"
]].copy()

# Tipos seguros
features["basin_id"] = pd.to_numeric(features["basin_id"], errors="coerce")
features["basin_area_km2"] = pd.to_numeric(features["basin_area_km2"], errors="coerce")
features["dist_to_river_m"] = pd.to_numeric(features["dist_to_river_m"], errors="coerce")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUT_PATH, index=False)

print("\n✔ Hydro features created")
print("\nPrimeras filas:")
print(features.head())

print("\nDescribe numérico:")
print(features[["basin_id", "basin_area_km2", "dist_to_river_m"]].describe())

print("\nMissingness:")
print(features.isna().mean())

print(f"\n✅ Guardado en: {OUT_PATH} | shape={features.shape}")