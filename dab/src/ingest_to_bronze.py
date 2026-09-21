# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest TrackMan Data: Raw → Bronze
# MAGIC
# MAGIC Uses Auto Loader (`cloudFiles`) to incrementally ingest the raw TrackMan CSV(s) from the Unity Catalog volume into a bronze Delta table with a single `VARIANT` column.
# MAGIC
# MAGIC Each source row becomes one `VARIANT` value (`data`), so the full record is kept without a fixed schema at ingest. File metadata sits alongside it for lineage.

# COMMAND ----------

# Parameters (override via job/DAB params or the widget bar)
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "trackman", "Schema")
dbutils.widgets.text("volume", "raw_trackman_data", "Raw volume")
dbutils.widgets.text("bronze_table", "trackman_bronze", "Bronze table")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
BRONZE_TABLE_NAME = dbutils.widgets.get("bronze_table")

# COMMAND ----------

import pyspark.sql.functions as F
from datetime import datetime

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.{BRONZE_TABLE_NAME}"
DATA_LOCATION = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Auto Loader schema + checkpoint state, kept separate from the raw data
CHECKPOINT_LOCATION_BRONZE = f"{DATA_LOCATION}/_checkpoints/{BRONZE_TABLE_NAME}"
SCHEMA_LOCATION_BRONZE = f"{DATA_LOCATION}/_schemas/{BRONZE_TABLE_NAME}"

print(f"Source:  {DATA_LOCATION}/*.csv")
print(f"Target:  {BRONZE_TABLE}")

# COMMAND ----------

current_run = datetime.now()

# Read every column as a raw string, then build the VARIANT ourselves. Auto Loader's
# singleVariantColumn coerces values (e.g. Tilt "8:15" -> a timestamp); this preserves them.
raw = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", SCHEMA_LOCATION_BRONZE)
    .option("header", "true")
    .option("cloudFiles.inferColumnTypes", "false")
    .load(f"{DATA_LOCATION}/*.csv")
)

# Every column except the rescue column becomes a VARIANT key (column names -> keys).
source_cols = [c for c in raw.columns if c != "_rescued_data"]
cols_sql = ", ".join(f"`{c}`" for c in source_cols)
data_variant = F.expr(f"to_variant_object(struct({cols_sql}))")

query = (
    raw
    .withColumn("data", data_variant)
    .withColumn("file_path", F.col("_metadata.file_path"))
    .withColumn("file_name", F.col("_metadata.file_name"))
    .withColumn("file_size", F.col("_metadata.file_size"))
    .withColumn("file_modification_time", F.col("_metadata.file_modification_time"))
    .withColumn("file_batch_time", F.lit(current_run))
    .withColumn("last_update_time", F.current_timestamp())
    .select(
        "data",
        "file_path",
        "file_name",
        "file_size",
        "file_modification_time",
        "file_batch_time",
        "last_update_time",
    )
    .writeStream
    .option("checkpointLocation", CHECKPOINT_LOCATION_BRONZE)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(BRONZE_TABLE)
)

query.awaitTermination()

# COMMAND ----------

# Verify: row count + a sample with a few fields pulled from the VARIANT
print(f"Row count: {spark.table(BRONZE_TABLE).count()}")

display(
    spark.sql(f"""
        SELECT
            data:PitchNo::int        AS pitch_no,
            data:Date::string        AS game_date,
            data:PitcherTeam::string AS pitcher_team,
            data:RelSpeed::double    AS release_speed,
            file_name,
            last_update_time,
            data
        FROM {BRONZE_TABLE}
        LIMIT 20
    """)
)
