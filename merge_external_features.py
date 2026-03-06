import pandas as pd

old_path = r"data/external_geofeatures.csv"
hydro_path = r"data/external_geofeatures_hydro_only.csv"
out_path = r"data/external_geofeatures_plus_hydro.csv"

old = pd.read_csv(old_path)
hydro = pd.read_csv(hydro_path)

# columnas hydro a conservar
hydro_cols = [
    "Latitude",
    "Longitude",
    "basin_id",
    "basin_area_km2",
    "dist_to_river_m"
]

hydro = hydro[hydro_cols]

merged = old.merge(
    hydro,
    on=["Latitude","Longitude"],
    how="left",
    validate="one_to_one"
)

print("OLD shape:", old.shape)
print("HYDRO shape:", hydro.shape)
print("MERGED shape:", merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nMissingness:")
print(merged.isna().mean().sort_values(ascending=False).head(10))

merged.to_csv(out_path,index=False)

print("\n✅ Saved:", out_path)