# Data Structure

## Directory Layout

- data/raw/
  - water_quality_training_dataset.csv
  - submission_template.csv
  - landsat_features_training.csv
  - landsat_features_validation.csv
  - terraclimate_features_training.csv
  - terraclimate_features_validation.csv

- data/external/
  - chirps_features_training.csv
  - chirps_features_validation.csv

- data/
  - external_geofeatures_plus_hydro_v2.csv
  - external_geofeatures_hydro_v2.csv
  - external_geofeatures.csv

- data/hydrology/
  - HydroBASINS shapefiles and layers
  - HydroRIVERS shapefiles and layers

## Key Join Columns

All primary datasets are aligned on:
- Latitude
- Longitude
- Sample Date

## Notes

- The model pipeline expects exact file names configured in config.yml.
- If paths or names change, update config.yml before running training.
- Keep raw and engineered datasets versioned or checksummed for strict reproducibility.
