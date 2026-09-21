# Databricks notebook source
# MAGIC %md
# MAGIC # Setup: schema and volume
# MAGIC
# MAGIC Creates the `trackman` schema and `raw_trackman_data` volume that the pipeline notebooks
# MAGIC read and write. The DAB version provisions these from resource YAML; without the bundle,
# MAGIC run this once first, then upload `Track_Combo.csv` into the volume.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "trackman", "Schema")
dbutils.widgets.text("volume", "raw_trackman_data", "Raw volume")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

print(f"Ready: {CATALOG}.{SCHEMA}, volume {VOLUME}")
print(f"Upload Track_Combo.csv to: /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/")
