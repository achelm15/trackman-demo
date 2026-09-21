# Databricks notebook source
# MAGIC %md
# MAGIC # Score pitches -> gold called-strike predictions (incremental)
# MAGIC
# MAGIC Streams new silver pitches, scores them with the registered `@prod` model, and MERGEs the
# MAGIC results into a gold table. A checkpoint tracks progress, so each run scores only the pitches
# MAGIC added since the last run.
# MAGIC
# MAGIC It loads the native flavor (`mlflow.xgboost.load_model`) for `predict_proba`, since the
# MAGIC pyfunc/`spark_udf` path returns labels only.
# MAGIC
# MAGIC Retraining does not re-score history: the checkpoint only advances over new silver rows. To
# MAGIC re-score everything with a new model, delete the gold checkpoint and table, then rerun.

# COMMAND ----------

# MAGIC %pip install --quiet xgboost mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Parameters
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "trackman", "Schema")
dbutils.widgets.text("silver_table", "trackman_silver", "Silver table")
dbutils.widgets.text("volume", "raw_trackman_data", "Volume (for checkpoint)")
dbutils.widgets.text("model_name", "strike_predictor", "Registered model name")
dbutils.widgets.text("gold_table", "gold_strike_predictions", "Gold predictions table")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SILVER_TABLE = dbutils.widgets.get("silver_table")
VOLUME = dbutils.widgets.get("volume")
MODEL_NAME = dbutils.widgets.get("model_name")
GOLD_TABLE = dbutils.widgets.get("gold_table")

FULL_TABLE = f"{CATALOG}.{SCHEMA}.{SILVER_TABLE}"
FULL_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
FULL_GOLD = f"{CATALOG}.{SCHEMA}.{GOLD_TABLE}"
CHECKPOINT_LOCATION_GOLD = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/_checkpoints/{GOLD_TABLE}"

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
# Natural pitch key for the gold upsert (same as silver)
KEY_COLS = [
    "game_date", "pitcher_id", "batter_id",
    "inning", "inning_half", "pa_of_inning", "pitch_of_pa",
]

# Resolve the @prod version for lineage, then load that exact artifact
prod = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(FULL_MODEL, "prod")
MODEL_VERSION = prod.version
model = mlflow.xgboost.load_model(f"models:/{FULL_MODEL}@prod")
print(f"Loaded {FULL_MODEL} v{MODEL_VERSION}")

# COMMAND ----------

def score_and_upsert(batch_df, batch_id):
    if not batch_df.take(1):  # nothing new this microbatch
        return

    pdf = batch_df.select(*ID_COLS, *FEATURES).toPandas()
    proba = model.predict_proba(pdf[FEATURES])[:, 1]
    pdf["strike_probability"] = proba
    pdf["predicted_is_strike"] = (proba >= 0.5).astype(int)
    # Actual umpire call for taken pitches (NULL for swung-at pitches)
    pdf["actual_is_strike"] = pdf["pitch_call"].map(
        {"StrikeCalled": 1, "BallCalled": 0, "BallinDirt": 0}
    )

    out = (
        spark.createDataFrame(pdf)
        .withColumn("model_version", F.lit(MODEL_VERSION))
        .withColumn("scored_at", F.current_timestamp())
    )

    # First run: create the gold table from the batch schema so MERGE has a target
    if not spark.catalog.tableExists(FULL_GOLD):
        out.limit(0).write.format("delta").saveAsTable(FULL_GOLD)

    out.createOrReplaceTempView("temp_gold_batch")
    on_clause = " AND ".join(f"g.{k} <=> u.{k}" for k in KEY_COLS)
    spark.sql(f"""
        MERGE INTO {FULL_GOLD} AS g
        USING `temp_gold_batch` AS u
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    spark.catalog.dropTempView("temp_gold_batch")

# COMMAND ----------

# Stream new silver rows and upsert their scores into gold.
# ignoreChanges: silver is written by MERGE, so its Delta log has non-append commits; this lets
# the stream read them, and the idempotent gold MERGE keeps re-emitted rows from duplicating.
(
    spark.readStream
    .format("delta")
    .option("ignoreChanges", "true")
    .table(FULL_TABLE)
    .writeStream
    .foreachBatch(score_and_upsert)
    .outputMode("update")
    .option("checkpointLocation", CHECKPOINT_LOCATION_GOLD)
    .trigger(availableNow=True)
    .start()
    .awaitTermination()
)

# COMMAND ----------

# Verify: gold row count + accuracy on taken pitches (spans all scored rows, not a held-out set)
print(f"Gold row count: {spark.table(FULL_GOLD).count()}")
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
    "gold_rows": spark.table(FULL_GOLD).count(),
}))
