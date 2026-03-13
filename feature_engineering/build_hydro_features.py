import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import numpy as np

# ==============================
# Helpers
# ==============================

def pick_existing(columns, candidates, required=False, label="column"):
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise ValueError(f"No encontré {label}. Candidatas: {candidates}")
    return None

# ==============================
# Paths
# ==============================

TRAIN_PATH = Path("data/raw/water_quality_training_dataset.csv")
VALID_PATH = Path("data/raw/submission_template.csv")

RIVERS_PATH = Path(r"data/hydrology/HydroRIVERS_v10_af_shp/HydroRIVERS_v10_af.shp")
BASINS_PATH = Path(r"data/hydrology/hybas_af_lev01-12_v1c/hybas_af_lev08_v1c.shp")

OUT_PATH = Path("data/external_geofeatures_hydro_v2.csv")

# ==============================
# Validation
# ==============================

for p in [TRAIN_PATH, VALID_PATH, RIVERS_PATH, BASINS_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")

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
# Load HydroBASINS
# ==============================

basins = gpd.read_file(BASINS_PATH)
print("\nColumnas HydroBASINS:")
print(basins.columns.tolist())

col_hybas_id  = pick_existing(basins.columns, ["HYBAS_ID"], required=True, label="HYBAS_ID")
col_area      = pick_existing(basins.columns, ["AREA_SQKM", "SUB_AREA", "BAS_AREA"], required=False, label="basin area")
col_up_area   = pick_existing(basins.columns, ["UP_AREA"], required=False, label="UP_AREA")
col_dist_main = pick_existing(basins.columns, ["DIST_MAIN"], required=False, label="DIST_MAIN")
col_dist_sink = pick_existing(basins.columns, ["DIST_SINK"], required=False, label="DIST_SINK")
col_order     = pick_existing(basins.columns, ["ORDER"], required=False, label="ORDER")
col_pfaf      = pick_existing(basins.columns, ["PFAF_ID"], required=False, label="PFAF_ID")

# área de cuenca
if col_area is None:
    print("⚠️ No encontré columna de área. Calculo área geométrica en km².")
    basins_tmp = basins.to_crs("EPSG:3857")
    basins["basin_area_km2"] = basins_tmp.geometry.area / 1e6
else:
    basins["basin_area_km2"] = pd.to_numeric(basins[col_area], errors="coerce")

# columnas opcionales
basins["upstream_area_km2"] = pd.to_numeric(basins[col_up_area], errors="coerce") if col_up_area else np.nan
basins["dist_main_km"] = pd.to_numeric(basins[col_dist_main], errors="coerce") if col_dist_main else np.nan
basins["dist_sink_km"] = pd.to_numeric(basins[col_dist_sink], errors="coerce") if col_dist_sink else np.nan
basins["basin_order"] = pd.to_numeric(basins[col_order], errors="coerce") if col_order else np.nan
basins["pfaf_id"] = pd.to_numeric(basins[col_pfaf], errors="coerce") if col_pfaf else np.nan
basins["basin_id"] = pd.to_numeric(basins[col_hybas_id], errors="coerce")

basins_keep = basins[[
    "basin_id",
    "basin_area_km2",
    "upstream_area_km2",
    "dist_main_km",
    "dist_sink_km",
    "basin_order",
    "pfaf_id",
    "geometry"
]].copy()

basins_keep = basins_keep.to_crs("EPSG:4326")

# spatial join punto -> cuenca
points_basin = gpd.sjoin(
    gdf_points,
    basins_keep,
    how="left",
    predicate="within"
)

if "index_right" in points_basin.columns:
    points_basin = points_basin.drop(columns=["index_right"])

# ==============================
# Load HydroRIVERS
# ==============================

rivers = gpd.read_file(RIVERS_PATH)
print("\nColumnas HydroRIVERS:")
print(rivers.columns.tolist())

col_len       = pick_existing(rivers.columns, ["LENGTH_KM"], required=False, label="LENGTH_KM")
col_dist_dn   = pick_existing(rivers.columns, ["DIST_DN_KM"], required=False, label="DIST_DN_KM")
col_dist_up   = pick_existing(rivers.columns, ["DIST_UP_KM"], required=False, label="DIST_UP_KM")
col_catch     = pick_existing(rivers.columns, ["CATCH_SKM"], required=False, label="CATCH_SKM")
col_upland    = pick_existing(rivers.columns, ["UPLAND_SKM"], required=False, label="UPLAND_SKM")
col_discharge = pick_existing(rivers.columns, ["DIS_AV_CMS"], required=False, label="DIS_AV_CMS")
col_ord_flow  = pick_existing(rivers.columns, ["ORD_FLOW"], required=False, label="ORD_FLOW")
col_ord_stra  = pick_existing(rivers.columns, ["ORD_STRA"], required=False, label="ORD_STRA")
col_ord_clas  = pick_existing(rivers.columns, ["ORD_CLAS"], required=False, label="ORD_CLAS")

rivers_keep_cols = ["geometry"]
for c in [col_len, col_dist_dn, col_dist_up, col_catch, col_upland, col_discharge, col_ord_flow, col_ord_stra, col_ord_clas]:
    if c is not None:
        rivers_keep_cols.append(c)

rivers = rivers[rivers_keep_cols].copy()

rename_map = {}
if col_len:       rename_map[col_len] = "river_length_km"
if col_dist_dn:   rename_map[col_dist_dn] = "river_dist_dn_km"
if col_dist_up:   rename_map[col_dist_up] = "river_dist_up_km"
if col_catch:     rename_map[col_catch] = "river_catch_skm"
if col_upland:    rename_map[col_upland] = "river_upland_skm"
if col_discharge: rename_map[col_discharge] = "river_discharge_cms"
if col_ord_flow:  rename_map[col_ord_flow] = "river_order_flow"
if col_ord_stra:  rename_map[col_ord_stra] = "river_order_stra"
if col_ord_clas:  rename_map[col_ord_clas] = "river_order_clas"

rivers = rivers.rename(columns=rename_map)

# Proyección métrica para nearest join y distancia
rivers_m = rivers.to_crs("EPSG:3857")
points_m = gdf_points.to_crs("EPSG:3857")

# nearest river attributes
nearest = gpd.sjoin_nearest(
    points_m,
    rivers_m,
    how="left",
    distance_col="dist_to_river_m"
)

# quitar index_right si aparece
if "index_right" in nearest.columns:
    nearest = nearest.drop(columns=["index_right"])

# ==============================
# Merge basin + river features
# ==============================

river_feature_cols = [c for c in nearest.columns if c not in ["Latitude", "Longitude", "geometry"]]
river_features = nearest[["Latitude", "Longitude"] + river_feature_cols].copy()

basin_feature_cols = [c for c in points_basin.columns if c not in ["geometry"]]
basin_features = points_basin[basin_feature_cols].copy()

features = basin_features.merge(
    river_features,
    on=["Latitude", "Longitude"],
    how="left",
    validate="one_to_one"
)

# asegurar tipos numéricos en variables feature
for c in features.columns:
    if c not in ["Latitude", "Longitude"]:
        features[c] = pd.to_numeric(features[c], errors="coerce")

# ingeniería hidrológica simple adicional
if "basin_area_km2" in features.columns and "dist_to_river_m" in features.columns:
    features["log_basin_area_km2"] = np.log1p(features["basin_area_km2"].clip(lower=0))
    features["log_dist_to_river_m"] = np.log1p(features["dist_to_river_m"].clip(lower=0))

if "upstream_area_km2" in features.columns:
    features["log_upstream_area_km2"] = np.log1p(features["upstream_area_km2"].clip(lower=0))

if "river_discharge_cms" in features.columns:
    features["log_river_discharge_cms"] = np.log1p(features["river_discharge_cms"].clip(lower=0))

if "river_catch_skm" in features.columns and "basin_area_km2" in features.columns:
    features["catch_to_basin_ratio"] = features["river_catch_skm"] / (features["basin_area_km2"] + 1e-6)

# === hydrology V3 engineered proxies ===
# Topographic Wetness Index proxy: logs of upslope area per slope (radians)
if "upstream_area_km2" in features.columns and "slope" in features.columns:
    upstream_m2 = features["upstream_area_km2"] * 1e6
    slope_rad = np.deg2rad(features["slope"]).clip(lower=1e-6)
    tan_slope = np.tan(slope_rad).clip(lower=1e-6)
    features["twi_proxy"] = np.log1p(upstream_m2 / tan_slope)

# Flow/discharge proxy: amplify upslope area by basin/river order
# use whichever order column exists (basin_order preferred)
order_col = None
for c in ["basin_order", "river_order_flow", "river_order_stra", "river_order_clas"]:
    if c in features.columns:
        order_col = c
        break
if "upstream_area_km2" in features.columns and order_col is not None:
    features["upslope_flow_proxy"] = features["upstream_area_km2"] * features[order_col]

# salida final: sin duplicados
features = features.drop_duplicates(subset=["Latitude", "Longitude"]).reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUT_PATH, index=False)

print("\n✔ Hydro features V2 created")
print("\nPrimeras filas:")
print(features.head())

print("\nColumnas finales:")
print(features.columns.tolist())

print("\nDescribe numérico:")
print(features.describe())

print("\nMissingness top 20:")
print(features.isna().mean().sort_values(ascending=False).head(20))

print(f"\n✅ Guardado en: {OUT_PATH} | shape={features.shape}")