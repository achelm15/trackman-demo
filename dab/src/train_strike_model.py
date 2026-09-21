# Databricks notebook source
# MAGIC %md
# MAGIC # Called strike vs. ball (XGBoost + Optuna)
# MAGIC
# MAGIC Predicts whether a taken pitch is called a strike (`is_strike = 1`) or a ball
# MAGIC (`is_strike = 0`) from its plate location and pitch characteristics.
# MAGIC
# MAGIC - Data: `trackman_silver`, filtered to called pitches (`StrikeCalled`, `BallCalled`, `BallinDirt`).
# MAGIC - Tuning: Optuna, with each trial a nested MLflow run.
# MAGIC - Every run logs a calibration plot and a SHAP beeswarm plot (matplotlib).
# MAGIC - Best params are retrained and registered to Unity Catalog with the `@prod` alias.

# COMMAND ----------

# MAGIC %pip install --quiet xgboost optuna shap
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Parameters
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "trackman", "Schema")
dbutils.widgets.text("silver_table", "trackman_silver", "Silver table")
dbutils.widgets.text("model_name", "strike_predictor", "Registered model name")
dbutils.widgets.text("n_trials", "30", "Optuna trials")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SILVER_TABLE = dbutils.widgets.get("silver_table")
MODEL_NAME = dbutils.widgets.get("model_name")
N_TRIALS = int(dbutils.widgets.get("n_trials"))

FULL_TABLE = f"{CATALOG}.{SCHEMA}.{SILVER_TABLE}"
FULL_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Experiment lives in the running user's home so the notebook is portable across workspaces
USERNAME = spark.sql("SELECT current_user()").collect()[0][0]
EXPERIMENT_PATH = f"/Users/{USERNAME}/trackman_strike_predictor"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Build the modeling dataset

# COMMAND ----------

import pyspark.sql.functions as F

# Location + pitch characteristics
FEATURES = [
    # plate location
    "plate_loc_height", "plate_loc_side",
    # velocity / release geometry
    "rel_speed", "zone_speed", "effective_velo",
    "rel_height", "rel_side", "extension",
    "vert_rel_angle", "horz_rel_angle",
    # spin & movement
    "spin_rate", "spin_axis",
    "vert_break", "induced_vert_break", "horz_break",
    "pfx_x", "pfx_z",
    # approach angles
    "vert_appr_angle", "horz_appr_angle",
]
LABEL = "is_strike"

sdf = (
    spark.table(FULL_TABLE)
    .filter(F.col("pitch_call").isin("StrikeCalled", "BallCalled", "BallinDirt"))
    .withColumn(LABEL, (F.col("pitch_call") == F.lit("StrikeCalled")).cast("int"))
    .select(*FEATURES, LABEL)
)

pdf = sdf.toPandas()
print(f"rows: {len(pdf)}  |  strike rate: {pdf[LABEL].mean():.3f}")
print(pdf[LABEL].value_counts().to_dict())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train / test split

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = pdf[FEATURES]
y = pdf[LABEL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Imbalance handling: weight the positive (strike) class
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
SCALE_POS_WEIGHT = neg / pos
print(f"train={len(X_train)} test={len(X_test)} scale_pos_weight={SCALE_POS_WEIGHT:.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Plot helpers: calibration + SHAP, logged to every run

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve
import mlflow


def log_calibration_plot(model, X_eval, y_eval):
    """Reliability curve of predicted probability vs. observed strike frequency."""
    prob = model.predict_proba(X_eval)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_eval, prob, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed strike frequency")
    ax.set_title("Calibration curve — called strike")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/calibration_curve.png")
    plt.close(fig)


def log_shap_beeswarm(model, X_eval):
    """SHAP TreeExplainer beeswarm — per-pitch SHAP values across all features."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_eval)  # Explanation object for the native beeswarm API

    shap.plots.beeswarm(explanation, max_display=len(FEATURES), show=False)
    fig = plt.gcf()
    plt.title("SHAP beeswarm — called strike")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/shap_beeswarm.png")
    plt.close(fig)

    # Also log the ranked mean |SHAP| values as a table artifact
    mean_abs = np.abs(explanation.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    lines = ["feature,mean_abs_shap"] + [
        f"{X_eval.columns[i]},{mean_abs[i]:.6f}" for i in order
    ]
    mlflow.log_text("\n".join(lines), "plots/shap_importance.csv")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Hyperparameter tuning with Optuna

# COMMAND ----------

import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# Autolog params/metrics per trial; log the model only in the best run
mlflow.xgboost.autolog(log_models=False, log_input_examples=False, silent=True)


def evaluate_and_log(model):
    """Compute held-out metrics and log calibration + SHAP artifacts to the active run."""
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "test_auc": roc_auc_score(y_test, prob),
        "test_logloss": log_loss(y_test, prob),
        "test_brier": brier_score_loss(y_test, prob),
        "test_accuracy": accuracy_score(y_test, pred),
    }
    mlflow.log_metrics(metrics)
    log_calibration_plot(model, X_test, y_test)
    log_shap_beeswarm(model, X_test)
    return metrics


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    with mlflow.start_run(nested=True):
        model = XGBClassifier(
            **params,
            scale_pos_weight=SCALE_POS_WEIGHT,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ).fit(X_train, y_train)
        metrics = evaluate_and_log(model)
        return metrics["test_auc"]


with mlflow.start_run(run_name="hpo") as parent:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_test_auc", study.best_value)

print("Best AUC:", study.best_value)
print("Best params:", study.best_params)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Retrain best params, log artifacts, register to Unity Catalog

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name="best") as best_run:
    best_model = XGBClassifier(
        **study.best_params,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)

    metrics = evaluate_and_log(best_model)
    mlflow.log_params(study.best_params)

    signature = infer_signature(X_test, best_model.predict_proba(X_test)[:, 1])
    info = mlflow.xgboost.log_model(
        best_model,
        name="model",
        signature=signature,
        input_example=X_test.head(5),
        registered_model_name=FULL_MODEL,
    )

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(FULL_MODEL, "prod", info.registered_model_version)
print(f"Registered {FULL_MODEL} v{info.registered_model_version} as @prod")
print("Test metrics:", metrics)

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "model": FULL_MODEL,
    "model_version": info.registered_model_version,
    "best_test_auc": study.best_value,
    "test_metrics": metrics,
    "n_rows": len(pdf),
}))
