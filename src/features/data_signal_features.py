#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Sample Date"


def _consecutive_dry_days(series: pd.Series, dry_threshold: float = 0.1) -> pd.Series:
    vals = series.fillna(0.0).values.astype(float)
    out = np.zeros(len(vals), dtype=float)
    streak = 0
    for i, v in enumerate(vals):
        out[i] = streak
        if v <= dry_threshold:
            streak += 1
        else:
            streak = 0
    return pd.Series(out, index=series.index)


def add_rainfall_antecedent_package(df: pd.DataFrame, skipped: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "chirps_ppt" not in out.columns:
        skipped.append("Rainfall package skipped: chirps_ppt missing")
        return out

    out = out.sort_values([LAT_COL, LON_COL, DATE_COL]).copy()
    g = out.groupby([LAT_COL, LON_COL])["chirps_ppt"]

    def sum_prior(window: int):
        # Strictly prior: shift(1) prevents future leakage at sample timestamp.
        return g.transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())

    def max_prior(window: int):
        return g.transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).max())

    out["rain_3d_sum"] = sum_prior(3)
    out["rain_7d_sum"] = sum_prior(7)
    out["rain_14d_sum"] = sum_prior(14)
    out["rain_30d_sum"] = sum_prior(30)
    out["rain_60d_sum"] = sum_prior(60)
    out["rain_90d_sum"] = sum_prior(90)

    out["rain_3d_max"] = max_prior(3)
    out["rain_7d_max"] = max_prior(7)

    out["dry_days_before_sample"] = (
        out.groupby([LAT_COL, LON_COL])["chirps_ppt"].apply(_consecutive_dry_days).reset_index(level=[0, 1], drop=True)
    )

    r30 = out["rain_30d_sum"].fillna(0.0)
    base30 = (
        out.groupby([LAT_COL, LON_COL])["rain_30d_sum"]
        .transform(lambda x: x.shift(1).rolling(window=90, min_periods=5).mean())
        .fillna(r30.mean())
    )
    out["rain_anomaly_30d"] = r30 - base30

    if "pet" in out.columns:
        pet_g = out.groupby([LAT_COL, LON_COL])["pet"]
        pet30 = pet_g.transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).sum())
        pet60 = pet_g.transform(lambda x: x.shift(1).rolling(window=60, min_periods=1).sum())
        out["rain_to_pet_30d"] = out["rain_30d_sum"] / (pet30 + 1.0)
        out["rain_to_pet_60d"] = out["rain_60d_sum"] / (pet60 + 1.0)
    else:
        skipped.append("rain_to_pet_* skipped: pet missing")

    return out


def add_moisture_dryness_package(df: pd.DataFrame, skipped: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "rain_30d_sum" not in out.columns:
        skipped.append("Moisture package skipped: rain_30d_sum missing")
        return out

    if "pet" not in out.columns:
        skipped.append("Moisture package partially skipped: pet missing")
        out["pet_30d_sum"] = np.nan
        out["deficit_30d"] = np.nan
        out["humidity_proxy_30d"] = np.nan
        out["dryness_index_30d"] = np.nan
        return out

    out = out.sort_values([LAT_COL, LON_COL, DATE_COL]).copy()
    pet_g = out.groupby([LAT_COL, LON_COL])["pet"]
    out["pet_30d_sum"] = pet_g.transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).sum())
    out["deficit_30d"] = out["rain_30d_sum"] - out["pet_30d_sum"]
    out["humidity_proxy_30d"] = out["rain_30d_sum"] / (out["pet_30d_sum"] + 1.0)
    out["dryness_index_30d"] = out["pet_30d_sum"] / (out["rain_30d_sum"] + 1.0)

    if "dry_days_before_sample" not in out.columns:
        out["dry_days_before_sample"] = (
            out.groupby([LAT_COL, LON_COL])["chirps_ppt"].apply(_consecutive_dry_days).reset_index(level=[0, 1], drop=True)
        )
    return out


def _landcover_masks(landcover: pd.Series):
    lc = pd.to_numeric(landcover, errors="coerce").fillna(-1).astype(int)
    cropland = lc.eq(40).values
    urban = lc.eq(50).values
    natural = (~cropland) & (~urban) & (lc.values >= 0)
    return cropland.astype(float), urban.astype(float), natural.astype(float)


def add_landuse_pressure(df: pd.DataFrame, skipped: list[str]) -> pd.DataFrame:
    out = df.copy()
    required = {LAT_COL, LON_COL, "landcover"}
    if not required.issubset(set(out.columns)):
        skipped.append("Landuse pressure skipped: required columns missing")
        return out

    coords = out[[LAT_COL, LON_COL]].astype(float).values
    if len(coords) == 0:
        skipped.append("Landuse pressure skipped: empty dataframe")
        return out

    coords_rad = np.deg2rad(coords)
    tree = BallTree(coords_rad, metric="haversine")

    cropland_m, urban_m, natural_m = _landcover_masks(out["landcover"])

    def frac_for_radius(radius_km: float, mask_vals: np.ndarray) -> np.ndarray:
        r = radius_km / 6371.0
        ind = tree.query_radius(coords_rad, r=r)
        frac = np.zeros(len(out), dtype=float)
        for i, neigh in enumerate(ind):
            if len(neigh) == 0:
                frac[i] = np.nan
            else:
                frac[i] = float(np.mean(mask_vals[neigh]))
        return frac

    for radius in [1, 5, 10]:
        out[f"cropland_fraction_{radius}km"] = frac_for_radius(radius, cropland_m)
        out[f"urban_fraction_{radius}km"] = frac_for_radius(radius, urban_m)

    for radius in [1, 5]:
        out[f"natural_fraction_{radius}km"] = frac_for_radius(radius, natural_m)

    return out


def add_upstream_pressure(df: pd.DataFrame, skipped: list[str]) -> pd.DataFrame:
    out = df.copy()

    def col(name: str) -> pd.Series:
        return pd.to_numeric(out.get(name, pd.Series(np.nan, index=out.index)), errors="coerce")

    up = col("upstream_area_km2")
    if up.isna().all():
        skipped.append("Upstream pressure partially skipped: upstream_area_km2 missing")

    out["upstream_cropland_pressure"] = col("cropland_fraction_5km") * up
    out["upstream_urban_pressure"] = col("urban_fraction_5km") * up
    out["upstream_soil_clay_pressure"] = col("soil_clay_0_5") * up
    out["upstream_soc_pressure"] = col("soil_soc_0_5") * up
    out["upstream_rain_30d_pressure"] = col("rain_30d_sum") * up
    out["upstream_rain_60d_pressure"] = col("rain_60d_sum") * up
    out["flow_contact_proxy"] = up / (col("river_discharge_cms") + 1.0)
    return out
