# Databricks notebook source
# MAGIC %md
# MAGIC # Check whether the strike model exists
# MAGIC
# MAGIC Sets a job task value `model_exists` ("true"/"false") used by a downstream condition task
# MAGIC to branch: if the `@prod` model exists, skip straight to scoring; if not, train first.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "trackman", "Schema")
dbutils.widgets.text("model_name", "strike_predictor", "Registered model name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
FULL_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")

exists = False
try:
    mv = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(FULL_MODEL, "prod")
    exists = True
    print(f"Found {FULL_MODEL}@prod -> v{mv.version}")
except Exception as e:
    print(f"No @prod model for {FULL_MODEL}: {e}")

# The condition task compares this against the string "true"
dbutils.jobs.taskValues.set(key="model_exists", value=str(exists).lower())
dbutils.notebook.exit(str(exists).lower())
