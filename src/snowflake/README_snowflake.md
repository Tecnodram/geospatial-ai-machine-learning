# Snowflake Minimal Integration (EY Water Quality)

This folder provides a low-risk Snowflake layer for:
- loading raw CSV assets,
- building model-ready Snowflake tables,
- registering experiment metadata.

The current local training/submission pipeline remains unchanged:
- local training: `python src/run_all.py --config config.yml`
- local submission generation stays in `src/train_pipeline.py` and `src/batch_blends.py`

## 1) Environment variables

Set Snowflake credentials before running scripts:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- optional: `SNOWFLAKE_ROLE`

## 2) Load core datasets to Snowflake RAW

From repository root:

```powershell
python src/snowflake/load_to_snowflake.py --database EY_WQ --warehouse COMPUTE_WH
```

Notes:
- Script uses `write_pandas` with `overwrite=True` and `auto_create_table=True`.
- Missing local files are skipped with warnings.
- To include extra external tables (for example `external_geofeatures_plus_hydro_v2.csv`),
  load them as additional RAW tables with the same naming convention.

## 3) Build model-ready tables + experiment registry table

Run SQL in Snowflake Worksheet:

- `src/snowflake/build_feature_tables.sql`

This creates:
- `EY_WQ.MART.TRAIN_MODELREADY`
- `EY_WQ.MART.VALID_MODELREADY`
- `EY_WQ.AUDIT.EXPERIMENT_REGISTRY`

## 4) Register a local experiment run

```powershell
python src/snowflake/register_experiment.py --run-path experiments/exp_YYYYMMDD_HHMMSS --notes "baseline check"
```

## Cost/ops guidance

- Use a small warehouse (`XSMALL`) while iterating.
- Suspend warehouse when not in use.
- Prefer table rebuild only when upstream raw tables changed.
