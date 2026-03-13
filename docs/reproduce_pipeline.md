# Reproduce Pipeline

## 1. Environment Setup

Prerequisites:
- Windows, Linux, or macOS
- Python 3.11+
- Git

From repository root:

```bash
git clone <your-repo-url>
cd ey-water-quality-2026
python -m venv .venv
```

Activate environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Required Data Assets

Place or validate the following files:

Raw tabular inputs:
- data/raw/water_quality_training_dataset.csv
- data/raw/submission_template.csv
- data/raw/landsat_features_training.csv
- data/raw/landsat_features_validation.csv
- data/raw/terraclimate_features_training.csv
- data/raw/terraclimate_features_validation.csv

Engineered external features:
- data/external/chirps_features_training.csv
- data/external/chirps_features_validation.csv
- data/external_geofeatures_plus_hydro_v2.csv

Hydrology source files for regeneration workflows:
- HydroBASINS shapefiles under data/hydrology/
- HydroRIVERS shapefiles under data/hydrology/

## 3. Optional Credentialed Services

Google Earth Engine (if regenerating CHIRPS/geospatial feature extracts):
- Authenticate Earth Engine locally.
- Ensure project ID values in extraction scripts are valid for your account.

Snowflake (optional experiment archival):
- Provide SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD.
- Optional role: SNOWFLAKE_ROLE.
- Credentials may be stored in .snowflake.env.

## 4. Configuration

Default run config is config.yml.

Check and adjust:
- project.raw_dir
- project.external_path
- project.out_dir
- project.exp_dir
- model and target transform blocks

## 5. Feature Extraction (Optional Rebuild)

If external feature CSVs are missing or need refresh:

```bash
python feature_engineering/extract_chirps_features.py
python feature_engineering/build_hydro_features.py
python feature_engineering/external_geofeatures.py
```

Note: script-specific paths may need local adaptation depending on your machine and data placement.

## 6. Train and Generate Predictions

Run full pipeline:

```bash
python src/run_all.py --config config.yml
```

This executes:
- src/train_pipeline.py
- src/batch_blends.py

Outputs:
- experiments/exp_*/ with run artifacts
- submissions_batch/ and/or run-local submission CSVs
- leaderboard_log.csv updates

## 7. Reproduce Reference Closure Run

Reference best indexed run:
- experiments/exp_20260307_003919/

Validate existence of:
- config_snapshot.json
- cv_report.json
- feature_manifest_*.json
- submission_V5_2_OOFTE_fixkeys.csv

## 8. Snowflake Logging (Optional)

Register existing run to Snowflake registry:

```bash
python src/snowflake/register_experiment.py --run-path experiments/exp_20260307_003919 --notes "portfolio closure baseline"
```

## 9. Determinism and Validation Checklist

- Keep random_state fixed in configs.
- Use the same fold/group settings.
- Do not alter key columns or date parsing semantics.
- Compare CV metrics against stored cv_report.json for sanity.
