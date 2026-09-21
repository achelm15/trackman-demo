# Databricks notebook source
# MAGIC %md
# MAGIC # Transform TrackMan Data: Bronze → Silver
# MAGIC
# MAGIC Streams the bronze `VARIANT` table, casts the TrackMan pitch and hit fields into a typed
# MAGIC silver table, and upserts (MERGE) so re-runs are idempotent.
# MAGIC
# MAGIC The bronze record is flat (one `VARIANT` object per pitch), so fields are pulled directly
# MAGIC with `data:Column::type`.

# COMMAND ----------

# Parameters (override via job/DAB params or the widget bar)
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "trackman", "Schema")
dbutils.widgets.text("volume", "raw_trackman_data", "Raw volume (for checkpoints)")
dbutils.widgets.text("bronze_table", "trackman_bronze", "Bronze table")
dbutils.widgets.text("silver_table", "trackman_silver", "Silver table")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
BRONZE_TABLE_NAME = dbutils.widgets.get("bronze_table")
SILVER_TABLE_NAME = dbutils.widgets.get("silver_table")

# COMMAND ----------

import pyspark.sql.functions as F

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.{BRONZE_TABLE_NAME}"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.{SILVER_TABLE_NAME}"
DATA_LOCATION = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
CHECKPOINT_LOCATION_SILVER = f"{DATA_LOCATION}/_checkpoints/{SILVER_TABLE_NAME}"
TEMP_VIEW = "temp_trackman_bronze_stream"

print(f"Source:  {BRONZE_TABLE}")
print(f"Target:  {SILVER_TABLE}")

# COMMAND ----------

# Stream in the bronze table
df_raw = (
    spark.readStream
    .format("delta")
    .table(BRONZE_TABLE)
)

# COMMAND ----------

# Temp view so we can extract with Spark SQL
df_raw.createOrReplaceTempView(TEMP_VIEW)

# COMMAND ----------

# Extract + cast fields from the VARIANT. try_cast -> NULL on blank cells (rather than error);
# Date is M/D/YYYY; Top/Bottom needs the bracket path data:['Top/Bottom'].
df = spark.sql(f"""
    SELECT
    -- Identifiers / context
    try_cast(data:PitchNo AS int)          AS pitch_no,
    to_date(data:Date::string, 'M/d/yyyy') AS game_date,
    data:PitcherId::string                 AS pitcher_id,
    data:PitcherThrows::string             AS pitcher_throws,
    data:PitcherTeam::string               AS pitcher_team,
    data:BatterId::string                  AS batter_id,
    data:BatterSide::string                AS batter_side,
    data:BatterTeam::string                AS batter_team,
    data:CatcherId::string                 AS catcher_id,
    data:CatcherThrows::string             AS catcher_throws,
    try_cast(data:Inning AS int)           AS inning,
    data:['Top/Bottom']::string            AS inning_half,
    try_cast(data:PAofInning AS int)       AS pa_of_inning,
    try_cast(data:PitchofPA AS int)        AS pitch_of_pa,
    try_cast(data:Outs AS int)             AS outs,
    try_cast(data:Balls AS int)            AS balls,
    try_cast(data:Strikes AS int)          AS strikes,
    data:Count::string                     AS count,

    -- Classification / outcome
    data:TaggedPitchType::string           AS tagged_pitch_type,
    data:AutoPitchType::string             AS auto_pitch_type,
    data:PitchCall::string                 AS pitch_call,
    data:KorBB::string                     AS kor_bb,
    data:TaggedHitType::string             AS tagged_hit_type,
    data:AutoHitType::string               AS auto_hit_type,
    data:PlayResult::string                AS play_result,
    try_cast(data:OutsOnPlay AS int)       AS outs_on_play,
    try_cast(data:RunsScored AS int)       AS runs_scored,

    -- Pitch tracking
    try_cast(data:RelSpeed AS double)         AS rel_speed,
    try_cast(data:VertRelAngle AS double)     AS vert_rel_angle,
    try_cast(data:HorzRelAngle AS double)     AS horz_rel_angle,
    try_cast(data:SpinRate AS double)         AS spin_rate,
    try_cast(data:SpinAxis AS double)         AS spin_axis,
    data:Tilt::string                         AS tilt,
    try_cast(data:RelHeight AS double)        AS rel_height,
    try_cast(data:RelSide AS double)          AS rel_side,
    try_cast(data:Extension AS double)        AS extension,
    try_cast(data:VertBreak AS double)        AS vert_break,
    try_cast(data:InducedVertBreak AS double) AS induced_vert_break,
    try_cast(data:HorzBreak AS double)        AS horz_break,
    try_cast(data:PlateLocHeight AS double)   AS plate_loc_height,
    try_cast(data:PlateLocSide AS double)     AS plate_loc_side,
    try_cast(data:ZoneSpeed AS double)        AS zone_speed,
    try_cast(data:VertApprAngle AS double)    AS vert_appr_angle,
    try_cast(data:HorzApprAngle AS double)    AS horz_appr_angle,
    try_cast(data:ZoneTime AS double)         AS zone_time,
    try_cast(data:EffectiveVelo AS double)    AS effective_velo,
    try_cast(data:pfxx AS double)             AS pfx_x,
    try_cast(data:pfxz AS double)             AS pfx_z,

    -- Hit tracking
    try_cast(data:ExitSpeed AS double)           AS exit_speed,
    try_cast(data:Angle AS double)               AS launch_angle,
    try_cast(data:Direction AS double)           AS direction,
    try_cast(data:HitSpinRate AS double)         AS hit_spin_rate,
    try_cast(data:Distance AS double)            AS distance,
    try_cast(data:LastTrackedDistance AS double) AS last_tracked_distance,
    try_cast(data:Bearing AS double)             AS bearing,
    try_cast(data:HangTime AS double)            AS hang_time,

    -- File / lineage metadata (carried from bronze)
    file_name,
    file_path,
    file_modification_time,
    file_batch_time,
    last_update_time,
    current_timestamp()                    AS silver_processed_time
    FROM {TEMP_VIEW}
""")

# COMMAND ----------

# MERGE upsert for idempotent re-runs. TrackMan has no unique pitch id and PitchNo restarts
# per game, so the key is date + matchup + inning position.
MERGE_KEYS = [
    "game_date",
    "pitcher_id",
    "batter_id",
    "inning",
    "inning_half",
    "pa_of_inning",
    "pitch_of_pa",
]

def upsert_to_silver(batch_df, batch_id):
    batch_temp_view = "temp_trackman_silver_batch"

    # First run: create the target table from the batch schema
    if not spark.catalog.tableExists(SILVER_TABLE):
        batch_df.limit(0).write.format("delta").saveAsTable(SILVER_TABLE)

    batch_df.createOrReplaceTempView(batch_temp_view)

    # <=> is null-safe so keys with NULLs still match
    on_clause = " AND ".join(f"silver.{k} <=> updates.{k}" for k in MERGE_KEYS)

    spark.sql(f"""
        MERGE INTO {SILVER_TABLE} AS silver
        USING `{batch_temp_view}` AS updates
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

    spark.catalog.dropTempView(batch_temp_view)

# COMMAND ----------

# Write stream with foreachBatch
(
    df.writeStream
    .foreachBatch(upsert_to_silver)
    .outputMode("update")
    .option("checkpointLocation", CHECKPOINT_LOCATION_SILVER)
    .trigger(availableNow=True)
    .start()
    .awaitTermination()
)

# COMMAND ----------

# Verify
print(f"Silver row count: {spark.table(SILVER_TABLE).count()}")
display(spark.table(SILVER_TABLE).limit(20))
