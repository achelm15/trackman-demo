# TrackMan pipeline (standalone notebooks)

The same pipeline as [`../dab/`](../dab/), as notebooks you import into a Databricks
workspace and run by hand. No bundle and no job orchestration, so you run each notebook
yourself and provision the schema and volume with `00_setup`.

## Notebooks

Run them in order:

1. `00_setup.py` — creates the `trackman` schema and `raw_trackman_data` volume.
2. `01_ingest_to_bronze.py` — Auto Loader reads the CSV into a `VARIANT` bronze table.
3. `02_transform_to_silver.py` — casts the fields into a typed silver table.
4. `03_train_strike_model.py` — XGBoost + Optuna, logs plots to MLflow, registers `@challenger`.
5. `04_score_to_gold.py` — incrementally scores new silver pitches with the `@prod` model,
   upserting into a gold table (streaming read from silver with a checkpoint).

Each notebook takes catalog/schema/table names as widgets, defaulting to `main` /
`trackman`. Change the widgets to target a different catalog.

## How to run

1. Import the notebooks into your workspace (Workspace > Import, or the CLI:
   `databricks workspace import-dir notebooks /Users/you@example.com/trackman`).
2. Run `00_setup` once.
3. Upload the sample CSV to the volume it created:
   ```bash
   databricks fs cp ../Track_Combo.csv \
     "dbfs:/Volumes/main/trackman/raw_trackman_data/Track_Combo.csv"
   ```
4. Run `01` and `02` to build bronze and silver.
5. Run `03` to train. It registers the best version as `@challenger`. Review it, then promote
   to `@prod` (run `alias=prod`, use the printed `set_registered_model_alias(...)` call, or the
   UI). `04` loads `@prod`, so it needs a promoted model.
6. Run `04` to score.

Re-run `03` whenever you want to retrain; it registers a new `@challenger`, which has no effect
until you promote it. `04` scores incrementally, so a promoted model only applies to pitches
added after it. To re-score history, delete the gold checkpoint and table, then rerun `04`.

## Notes

- The bronze notebook builds its `VARIANT` from raw string columns rather than Auto
  Loader's `singleVariantColumn`, which would coerce values such as the clock-style
  `Tilt` "8:15" into a timestamp.
- `03_train_strike_model` installs `xgboost`, `optuna`, and `shap`; `04_score_to_gold`
  installs `xgboost`. These are not on the base serverless image.
