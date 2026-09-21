# TrackMan pitch pipeline

Ingests raw TrackMan pitch data into a medallion lakehouse on Databricks and trains a
model to predict called strikes.

The data comes from optical tracking of college baseball scrimmages: one row per pitch,
with plate location, velocity, spin, movement, and hit metrics. A sample file,
`Track_Combo.csv`, is at the repo root.

## Data source

`Track_Combo.csv` is from:

Pifer, Nathan David (2024), "Optical Tracking Data from College Baseball Scrimmages",
Mendeley Data, V3. DOI: [10.17632/xfnz6mkdzm.3](https://doi.org/10.17632/xfnz6mkdzm.3)

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The CSV is included
here unmodified; the notebooks transform it into bronze, silver, and gold tables.

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
   Each run logs a calibration plot, a SHAP beeswarm plot, and a strike-zone heatmap. The best
   model registers to Unity Catalog as `@challenger`; review it, then promote to `@prod`.
4. Scoring: incrementally scores new silver pitches with the `@prod` model, upserting into a
   gold table (streaming read from silver with a checkpoint).

"Strike or ball" means the umpire's call on a taken pitch. Swinging strikes, fouls, and
balls in play are excluded, since those are not called balls or strikes.
