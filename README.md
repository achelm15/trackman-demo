# TrackMan pitch pipeline

Ingests raw TrackMan pitch data into a medallion lakehouse on Databricks and trains a
model to predict called strikes.

The data comes from optical tracking of college baseball scrimmages: one row per pitch,
with plate location, velocity, spin, movement, and hit metrics. A sample file,
`Track_Combo.csv`, is at the repo root.

## Two ways to run it

- [`dab/`](dab/) — a Databricks Asset Bundle. Deploys the schema, volume, and jobs, and
  runs the pipeline end to end (including a branch that trains a model on the first run).
  Use this to deploy the whole thing with `databricks bundle deploy`.
- [`notebooks/`](notebooks/) — the same logic as standalone notebooks you import and run
  by hand. No bundle, no job orchestration. Use this to step through the pipeline or drop
  a notebook into an existing project.

Both read and write the same tables and expect the same `Track_Combo.csv`. Each folder
has its own README with instructions.

## Pipeline

1. Bronze: Auto Loader reads the CSV from a volume and writes each row as one `VARIANT`
   column.
2. Silver: casts the pitch and hit fields into a typed table, upserting with MERGE.
3. Model: XGBoost tuned with Optuna predicts called strike vs. ball, tracked in MLflow.
   Each run logs a calibration plot and a SHAP beeswarm plot; the best model registers to
   Unity Catalog under `@prod`.
4. Scoring: scores every silver pitch with the `@prod` model into a gold table.

"Strike or ball" means the umpire's call on a taken pitch. Swinging strikes, fouls, and
balls in play are excluded, since those are not called balls or strikes.
