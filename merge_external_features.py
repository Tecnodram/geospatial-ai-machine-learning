import pandas as pd
import numpy as np

old_path = r"data/external_geofeatures.csv"
hydro_path = r"data/external_geofeatures_hydro_v2.csv"
out_path = r"data/external_geofeatures_plus_hydro_v2.csv"

old = pd.read_csv(old_path)
hydro = pd.read_csv(hydro_path)

merged = old.merge(
    hydro,
    on=["Latitude", "Longitude"],
    how="left",
    validate="one_to_one"
)

print("OLD shape:", old.shape)
print("HYDRO shape:", hydro.shape)
print("MERGED shape:", merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nMissingness:")
print(merged.isna().mean().sort_values(ascending=False).head(20))
# === hydrology V3 proxies injected into plus file ===
# Topographic wetness proxy (need slope and upslope area)
if "slope" in merged.columns and "upstream_area_km2" in merged.columns:
    upstream_m2 = merged["upstream_area_km2"].fillna(0) * 1e6
    slope_rad = np.deg2rad(merged["slope"].fillna(0)).clip(lower=1e-6)
    tan_slope = np.tan(slope_rad).clip(lower=1e-6)
    merged["twi_proxy"] = np.log1p(upstream_m2 / tan_slope)

# Flow/discharge proxy: use first available order column
order_col = None
for c in ["basin_order", "river_order_flow", "river_order_stra", "river_order_clas"]:
    if c in merged.columns:
        order_col = c
        break
if "upstream_area_km2" in merged.columns and order_col is not None:
    merged["upslope_flow_proxy"] = merged["upstream_area_km2"].fillna(0) * merged[order_col].fillna(0)
merged.to_csv(out_path, index=False)
print(f"\n✅ Saved -> {out_path}")