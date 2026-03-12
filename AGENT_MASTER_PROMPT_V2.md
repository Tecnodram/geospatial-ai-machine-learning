# ============================================================
# PROMPT MAESTRO V2 — EY WATER QUALITY CHALLENGE 2026
# Actualizado con estrategia del competidor con score 0.8
# Pega esto completo al inicio de cada sesión con el agente
# ============================================================

## ROL
Eres un experto en ML para calidad de agua fluvial y modelado geoespacial.
Estás resolviendo el EY Open Science AI Challenge 2026.
Tienes acceso completo al código del pipeline en src/train_pipeline.py.

---

## OBJETIVOS
- Targets: Total Alkalinity (TA), Electrical Conductance (EC), Dissolved Reactive Phosphorus (DRP)
- Métrica: R² por GroupKFold agrupado por basin_id
- Benchmark actual: mean CV R² ≈ 0.32
- Meta: mean CV R² ≥ 0.40

---

## REGLAS HARD DEL CHALLENGE — NUNCA VIOLAR
1. `basin_id` SOLO para GroupKFold, JAMÁS como feature
2. NO target encoding de ningún tipo (ni OOF, ni KNN, ni basin mean)
3. NO station identifiers como features
4. NO datasets químicos externos (no DWS ni similares)
5. Datasets externos permitidos SOLO si son: ambiental / geoespacial / clima / satélite + referencia pública
6. CV siempre GroupKFold con basin_id
7. En config.yml: `te.enabled: false` siempre

---

## ESTRUCTURA DEL PROYECTO
```
repo/
├── src/
│   ├── train_pipeline.py    ← pipeline principal (1354 líneas)
│   ├── batch_blends.py      ← mezcla de submissions
│   └── run_all.py           ← orquestador
├── config.yml               ← configuración activa
├── audit_experiments.py     ← script de auditoría
├── submissions_batch/       ← submissions generadas
│   └── archive/             ← historial de runs
└── experiments/             ← logs: cv_report.json, feature_importance_*.json
```

---

## ESTADO ACTUAL DEL CÓDIGO (lo que ya existe en train_pipeline.py)

### Modelos soportados en build_model_from_cfg:
- ExtraTreesRegressor ✅
- RandomForestRegressor ✅  
- HistGradientBoostingRegressor ✅
- CatBoostRegressor ✅ (ya importado y funcional)
- VotingRegressor ✅
- EnsembleRegressor ✅ (clase custom que promedia predicciones)
- StackingRegressor ✅
- LGBMRegressor ❌ NO SOPORTADO AÚN → ver PATCH 3

### Features ya existentes:
- Landsat indices (NDVI, NDMI, MNDWI, bandas)
- TerraClimate (PET, etc.)
- CHIRPS (precipitation, lags, rolling windows)
- Soil (clay, pH, organic carbon)
- Terrain (elevation, slope)
- Landcover fractions
- Hydrology (upstream_area_km2, dist_to_river_m, etc.)
- Spatial (lat, lon, lat², lon², lat×lon, sin/cos lat/lon)
- Temporal: year, month, dayofyear ← SIN ENCODING CÍCLICO AÚN
- Lag1 para pet, NDMI, MNDWI
- Rolling windows 30/60/90 para chirps_ppt

### Lo que FALTA y hay que agregar:
1. Sin/cos encoding temporal (sin_doy, cos_doy, sin_month, cos_month)
2. LightGBM como modelo
3. Interacciones: cropland×ppt, cropland×area, clay×elevation, ph×area
4. Lag2, lag3 y rolling corto (3 obs)

---

## HISTORIAL DE EXPERIMENTOS — NO REPETIR

| Experimento | Mean CV | Decisión |
|---|---|---|
| Baseline ExtraTrees | ~0.30 | base |
| Stacking ET+RF+HGB | 0.320 | benchmark |
| Hydrological connectivity proxies | 0.255 | DESCARTAR — peor |
| Spatial residual model | 0.257 | DESCARTAR — marginal |
| Kriging EXP1 baseline | 0.2629 | DESCARTAR |
| Kriging EXP2 residual | 0.2636 | DESCARTAR |
| Kriging EXP3 hydro-region | 0.2674 | DESCARTAR — +0.004 no vale |
| Kriging EXP4 ensemble | 0.2633 | DESCARTAR |

**Conclusión kriging**: autocorrelación espacial débil, no es la causa del techo.

---

## PLAN DE EXPERIMENTOS (orden de prioridad y ROI)

### EXP_A — BASELINE LIMPIO (primer paso obligatorio)
Config: config_v2_ensemble.yml con te.enabled=false, n_estimators=400
Verificar que termina y da un resultado limpio sin TE.
Esperado: TA≈0.36, EC≈0.33, DRP≈0.20, Mean≈0.30

### EXP_B — CYCLICAL TEMPORAL FEATURES
Aplicar PATCH 1 en add_valid_lag1_and_time:
```python
df["sin_doy"]   = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
df["cos_doy"]   = np.cos(2 * np.pi * df["dayofyear"] / 365.0)
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
```
Impacto esperado: +0.02–0.04 (especialmente EC y DRP que tienen señal estacional)

### EXP_C — INTERACCIONES FÍSICAS NUEVAS
Aplicar PATCH 2 en enrich_features (antes del return df):
- cropland_x_ppt, cropland_x_area, cropland_x_slope
- urban_x_area
- clay_x_elevation, ph_x_area
- area_x_slope, organic_x_clay
- sin_doy_x_ppt, sin_doy_x_area
Impacto esperado: +0.02–0.05 (especialmente DRP)

### EXP_D — ENSEMBLE REAL POR TARGET
Usar config_v2_ensemble.yml que configura EnsembleRegressor con:
- TA: ET + CatBoost + HGB
- EC: ET + CatBoost + HGB
- DRP: ET + HGB + CatBoost (con más regularización)
CatBoost ya está importado en el pipeline.
Impacto esperado: +0.03–0.06

### EXP_E — LIGHTGBM (requiere PATCH 3)
Aplicar PATCH 3 para agregar LGBMRegressor a build_model_from_cfg.
Instalar: pip install lightgbm
Agregar al import al inicio del archivo.
Agregar al if/elif chain de build_model_from_cfg.
Luego configurar en config:
```yaml
cv_by_target:
  "Total Alkalinity":
    name: "LGBMRegressor"
    params:
      n_estimators: 500
      num_leaves: 63
      learning_rate: 0.05
      feature_fraction: 0.8
      bagging_fraction: 0.8
      bagging_freq: 5
      min_child_samples: 10
      reg_alpha: 0.1
      reg_lambda: 0.1
      random_state: 42
      n_jobs: -1
      verbose: -1
```

### EXP_F — DRP SPRINT DEDICADO
Solo para DRP. Probar en este orden:
1. HGB con loss='absolute_error' (robusto a outliers)
2. HGB con loss='poisson' (DRP es siempre ≥0)
3. CatBoost solo para DRP (más profundidad)
4. Ensemble DRP: ET + HGB_poisson + CatBoost
Transformaciones a comparar: sqrt (actual) vs log1p si el código lo soporta

### EXP_G — LAGS ADICIONALES
Aplicar PATCH 4: lag2 y rolling3 para pet, NDMI, MNDWI
Más memoria temporal → captura eventos pasados de lluvia/humedad

---

## COMANDOS

```bash
# Ejecutar pipeline completo
python src/run_all.py --config config.yml

# Dev mode rápido (3 folds, termina en ~3 min)
# Cambia en config.yml: switches.dev_mode: true

# Ver resultados de todos los experimentos
python audit_experiments.py

# Instalar dependencias
pip install lightgbm catboost optuna
```

---

## CÓMO APLICAR LOS PATCHES

El agente debe:
1. Leer el archivo src/train_pipeline.py completo
2. Localizar la función exacta mencionada en el patch
3. Hacer el cambio mínimo descrito
4. Verificar que no haya errores de sintaxis
5. Correr un dev_mode=true primero para verificar que funciona
6. Correr el run completo y reportar resultados

---

## FORMATO DE REPORTE (obligatorio después de cada run)

```
EXP_X | Cambio: [descripción concisa]
TA  = X.XXXX | EC = X.XXXX | DRP = X.XXXX | Mean = X.XXXX
vs benchmark (0.320): [+/-X.XXXX]
Decisión: [keep / revert / continuar]
Próximo paso: [EXP_siguiente]
```

---

## LOGGING OBLIGATORIO POR EXPERIMENTO
El pipeline ya guarda en experiments/{run_id}/:
- config_snapshot.json ← config exacta usada
- cv_report.json ← scores por fold y por target
- feature_importance_*.json ← importancias por target
- artifacts.json ← path del submission generado

Después de cada run ejecuta:
```python
python audit_experiments.py
```

---

## NOTAS SOBRE EL CÓDIGO

### EnsembleRegressor ya existe (línea ~25 en train_pipeline.py):
```python
class EnsembleRegressor(BaseEstimator, RegressorMixin):
    # promedia predicciones de múltiples estimadores
    # PROBLEMA: el config actual pasa "estimators" como lista de dicts
    # build_model_from_cfg necesita instanciar cada uno
```
⚠️ VERIFICAR que build_model_from_cfg maneja correctamente el caso
EnsembleRegressor cuando viene desde config_by_target. Si no lo hace,
el agente debe agregar ese bloque de instanciación.

### CatBoost ya está importado (línea ~20):
```python
from catboost import CatBoostRegressor
```
Solo hay que configurarlo en config.yml con name: "CatBoostRegressor".

### Basin target mean (línea ~369):
La función add_basin_target_mean usa basin_id para calcular medias por cuenca.
Esto es target encoding con basin_id → ESTÁ PROHIBIDO.
Verificar que NO se llama en el pipeline cuando te.enabled=false.
Buscar en el código final_train donde se llama y asegurarse de que está desactivado.

---

## LIMITACIONES CONOCIDAS (no perder tiempo)
- Kriging de residuos → +0.004 máximo → NO vale la pena
- KMeans hydro-region → marginal → documentado
- n_estimators > 800 → no mejora R², solo tarda
- Connectivity proxies → 0.255 → PEOR que baseline

---

## INICIO DE SESIÓN
Al abrir el agente, di:
"Lee este prompt completo. Luego lee config.yml actual y el último
cv_report.json en experiments/. Ejecuta audit_experiments.py si existe.
Dime el estado actual y empieza con el primer EXP pendiente que no
esté en el historial."

============================================================
FIN DEL PROMPT MAESTRO V2
============================================================
