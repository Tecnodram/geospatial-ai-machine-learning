# ============================================================
# PATCH 1 — Cyclical temporal features
# Archivo: src/train_pipeline.py
# Función: add_valid_lag1_and_time (línea ~590)
#
# BUSCA este bloque (ya existe en el código):
#     df["year"] = df[DATE_COL].dt.year
#     df["month"] = df[DATE_COL].dt.month
#     df["dayofyear"] = df[DATE_COL].dt.dayofyear
#
# REEMPLAZA CON:
# ============================================================

    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["dayofyear"] = df[DATE_COL].dt.dayofyear

    # Cyclical encoding — captura estacionalidad sin discontinuidades
    df["sin_doy"]   = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    df["cos_doy"]   = np.cos(2 * np.pi * df["dayofyear"] / 365.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)


# ============================================================
# PATCH 2 — Nuevas interacciones físicas
# Archivo: src/train_pipeline.py
# Función: enrich_features (línea ~400)
# Agregar AL FINAL de la función, antes del return df
#
# BUSCA: return df   (al final de enrich_features)
# INSERTA ANTES:
# ============================================================

    # === NUEVAS INTERACCIONES FÍSICA-DISCIPLINADA ===

    # DRP — fósforo agrícola (fuente difusa)
    if "cropland_fraction" in df.columns:
        if "chirps_ppt" in df.columns:
            df["cropland_x_ppt"] = df["cropland_fraction"] * df["chirps_ppt"]
        if "upstream_area_km2" in df.columns:
            df["cropland_x_area"] = df["cropland_fraction"] * df["upstream_area_km2"]
        if "slope" in df.columns:
            df["cropland_x_slope"] = df["cropland_fraction"] * df["slope"]

    # DRP — urban runoff
    if "urban_fraction" in df.columns and "upstream_area_km2" in df.columns:
        df["urban_x_area"] = df["urban_fraction"] * df["upstream_area_km2"]

    # TA — alcalinidad geológica
    if "soil_clay_0_5" in df.columns and "elevation" in df.columns:
        df["clay_x_elevation"] = df["soil_clay_0_5"] * df["elevation"]
    if "soil_ph_0_5" in df.columns and "upstream_area_km2" in df.columns:
        df["ph_x_area"] = df["soil_ph_0_5"] * df["upstream_area_km2"]

    # EC — conductividad (concentración de solutos)
    if "upstream_area_km2" in df.columns and "slope" in df.columns:
        df["area_x_slope"] = df["upstream_area_km2"] * df["slope"]
    if "soil_organic_carbon_0_5" in df.columns and "soil_clay_0_5" in df.columns:
        df["organic_x_clay"] = df["soil_organic_carbon_0_5"] * df["soil_clay_0_5"]

    # Ciclos estacionales × hidrología
    if "sin_doy" in df.columns:
        if "chirps_ppt" in df.columns:
            df["sin_doy_x_ppt"] = df["sin_doy"] * df["chirps_ppt"]
        if "upstream_area_km2" in df.columns:
            df["sin_doy_x_area"] = df["sin_doy"] * df["upstream_area_km2"]

    return df


# ============================================================
# PATCH 3 — Soporte LightGBM en build_model_from_cfg
# Archivo: src/train_pipeline.py
#
# PASO A) Agregar import al inicio del archivo (línea ~1, cerca de otros imports):
# ============================================================

try:
    from lightgbm import LGBMRegressor
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


# ============================================================
# PASO B) En build_model_from_cfg, ANTES de la línea:
#   raise ValueError(f"Unknown model: {model_name}")
#
# Agregar este bloque:
# ============================================================

    if model_name == "LGBMRegressor":
        if not _LGBM_AVAILABLE:
            raise ImportError("LightGBM no instalado. Ejecuta: pip install lightgbm")
        by_t = cfg.get("model", {}).get(f"{which}_by_target", {}) or {}
        spec  = by_t.get(target) or {}
        params = spec.get("params", {}) or {}
        return LGBMRegressor(**params)


# ============================================================
# PATCH 4 — Lags adicionales (lag2, lag3)
# Archivo: src/train_pipeline.py
# Función: add_valid_lag1_and_time
#
# BUSCA el bloque del lag1:
#     for c in lag_cols:
#         if c in df.columns:
#             lag = df.groupby([LAT_COL, LON_COL])[c].shift(1)
#             df[f"{c}_lag1"] = lag
#             df.loc[~has_lag1, f"{c}_lag1"] = np.nan
#
# REEMPLAZA CON:
# ============================================================

    for c in lag_cols:
        if c in df.columns:
            # lag1 (ya existía)
            lag1 = df.groupby([LAT_COL, LON_COL])[c].shift(1)
            df[f"{c}_lag1"] = lag1
            df.loc[~has_lag1, f"{c}_lag1"] = np.nan

            # lag2 — 2 observaciones atrás
            lag2 = df.groupby([LAT_COL, LON_COL])[c].shift(2)
            df[f"{c}_lag2"] = lag2

            # Rolling mean 3 obs — memoria corta
            roll3 = df.groupby([LAT_COL, LON_COL])[c].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )
            df[f"{c}_roll3"] = roll3
