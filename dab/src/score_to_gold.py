# Databricks notebook source
# MAGIC %md
# MAGIC # Score pitches → gold called-strike predictions
# MAGIC
# MAGIC Loads the registered `@prod` strike model and scores every pitch in `trackman_silver`,
# MAGIC writing a per-pitch called-strike probability to a gold table for dashboards, apps, and Genie.
# MAGIC
# MAGIC It loads the native flavor (`mlflow.xgboost.load_model`) for `predict_proba`, since the
# MAGIC pyfunc/`spark_udf` path returns labels only. The dataset is small, so scoring in-driver is fine.

# COMMAND ----------

# MAGIC %pip install --quiet xgboost mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Parameters
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "trackman", "Schema")
dbutils.widgets.text("silver_table", "trackman_silver", "Silver table")
dbutils.widgets.text("model_name", "strike_predictor", "Registered model name")
dbutils.widgets.text("gold_table", "gold_strike_predictions", "Gold predictions table")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SILVER_TABLE = dbutils.widgets.get("silver_table")
MODEL_NAME = dbutils.widgets.get("model_name")
GOLD_TABLE = dbutils.widgets.get("gold_table")

FULL_TABLE = f"{CATALOG}.{SCHEMA}.{SILVER_TABLE}"
FULL_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
FULL_GOLD = f"{CATALOG}.{SCHEMA}.{GOLD_TABLE}"

# COMMAND ----------

import mlflow
import pyspark.sql.functions as F
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")

# Same feature set the model was trained on
FEATURES = [
    "plate_loc_height", "plate_loc_side",
    "rel_speed", "zone_speed", "effective_velo",
    "rel_height", "rel_side", "extension",
    "vert_rel_angle", "horz_rel_angle",
    "spin_rate", "spin_axis",
    "vert_break", "induced_vert_break", "horz_break",
    "pfx_x", "pfx_z",
    "vert_appr_angle", "horz_appr_angle",
]
ID_COLS = [
    "game_date", "pitcher_id", "pitcher_team", "batter_id", "batter_team",
    "inning", "inning_half", "pa_of_inning", "pitch_of_pa",
    "tagged_pitch_type", "pitch_call",
]

# Resolve the @prod version for lineage, then load that exact artifact
prod = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(FULL_MODEL, "prod")
MODEL_VERSION = prod.version
model = mlflow.xgboost.load_model(f"models:/{FULL_MODEL}@prod")
print(f"Loaded {FULL_MODEL} v{MODEL_VERSION}")

# COMMAND ----------

# Score every pitch (XGBoost handles the rare missing feature natively)
pdf = spark.table(FULL_TABLE).select(*ID_COLS, *FEATURES).toPandas()

proba = model.predict_proba(pdf[FEATURES])[:, 1]
pdf["strike_probability"] = proba
pdf["predicted_is_strike"] = (proba >= 0.5).astype(int)
# Actual umpire call for taken pitches (NULL for swung-at pitches)
pdf["actual_is_strike"] = pdf["pitch_call"].map(
    {"StrikeCalled": 1, "BallCalled": 0, "BallinDirt": 0}
)

result = (
    spark.createDataFrame(pdf)
    .withColumn("model_version", F.lit(MODEL_VERSION))
    .withColumn("scored_at", F.current_timestamp())
)

(
    result.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_GOLD)
)

print(f"Wrote {result.count()} rows to {FULL_GOLD}")

# COMMAND ----------

# Quick sanity check: accuracy on taken pitches
display(
    spark.sql(f"""
        SELECT
            count(*)                                                    AS taken_pitches,
            round(avg(CASE WHEN predicted_is_strike = actual_is_strike THEN 1 ELSE 0 END), 3) AS accuracy,
            round(avg(strike_probability), 3)                           AS avg_pred_prob
        FROM {FULL_GOLD}
        WHERE actual_is_strike IS NOT NULL
    """)
)

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "gold_table": FULL_GOLD,
    "model_version": MODEL_VERSION,
    "rows_scored": result.count(),
}))
