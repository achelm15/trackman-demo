# TrackMan pipeline (DAB)

The Databricks Asset Bundle version: deploys the schema, volume, and jobs, and runs the
pipeline end to end. For the standalone notebooks, see [`../notebooks/`](../notebooks/).

## Layout

```
databricks.yml                    # bundle config + catalog variable + target
resources/
  trackman.schema.yml             # trackman schema
  raw_trackman_data.volume.yml    # landing volume
  trackman_ingest.job.yml         # bronze -> silver -> (train if needed) -> score
  trackman_train.job.yml          # manual training / retraining
src/
  ingest_to_bronze.py
  transform_to_silver.py
  train_strike_model.py
  score_to_gold.py
  check_model.py                  # branch helper: does @prod exist?
```

The sample CSV lives at the repo root (`../Track_Combo.csv`).

## Jobs

`trackman_ingest` runs the pipeline end to end. After silver, it checks whether a `@prod`
strike model exists: if so it scores directly, and if not it trains one first, then
scores. The branch uses a condition task, so the pipeline is self-sufficient on a cold
start.

`trackman_train` retrains on demand. It registers the new version as `@challenger`; review
it, then promote to `@prod` manually (`client.set_registered_model_alias(...)` or the UI). The
ingest job scores with `@prod`, so a challenger has no effect until you promote it. On a cold
start, the ingest branch instead registers its bootstrap model directly as `@prod` so scoring
has something to load.

## Setup

Requires the Databricks CLI and a configured profile. The bundle's `default` target
points at a profile named `DEFAULT`; change it in `databricks.yml` to your own, or pass
`-p <profile>` on each command.

The `catalog` variable defaults to `main`. Override it to target a different catalog:

```bash
databricks bundle deploy --var catalog=my_catalog
```

## Run it

Run these from this `dab/` folder:

```bash
# 1. Deploy the schema, volume, and jobs
databricks bundle deploy

# 2. Upload the sample CSV to the volume (adjust catalog/schema if changed)
databricks fs cp ../Track_Combo.csv \
  "dbfs:/Volumes/main/trackman/raw_trackman_data/Track_Combo.csv"

# 3. Run the pipeline (trains a model on the first run, scores new pitches every run)
databricks bundle run trackman_ingest

# Retrain on demand
databricks bundle run trackman_train
```

## Notes

- The bronze layer builds its `VARIANT` from raw string columns rather than Auto Loader's
  `singleVariantColumn`, which would coerce values such as the clock-style `Tilt` "8:15"
  into a timestamp.
- The training notebook installs `xgboost`, `optuna`, and `shap`, which are not on the
  base serverless image.
