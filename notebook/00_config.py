# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: shared config
# MAGIC Every credential, input path, output table, and tuning parameter for the three market indicator
# MAGIC pipelines. Each notebook loads it with `%run ./00_config`.
# MAGIC
# MAGIC | Task | Notebooks | Widget input |
# MAGIC |---|---|---|
# MAGIC | Marginal return index | `01_marginal_preprocessing` → `02_marginal_feature_engineering` → `03_marginal_index` → `04_marginal_monitoring` | – |
# MAGIC | Visit share index (independent) | `05_visit_preprocessing` → `06_visit_feature_engineering` → `07_visit_index` → `08_visit_monitoring` | – |
# MAGIC | Overall index score | `09_overall_index` → `10_overall_monitoring` | run ids from `03_marginal_index` and `07_visit_index` |
# MAGIC
# MAGIC Each notebook passes its MLflow run id to the next one through a widget or task value.
# MAGIC The monitoring notebooks only report and can be skipped.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import json
import os
import pickle
import tempfile

import mlflow
import pandas as pd
from pyspark.sql import functions as F
from roost.connector.core.types import WriteMode

pd.set_option("display.max_columns", None)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Project

# COMMAND ----------

TEAM = "fit-ds"
PROJECT = "market_indicator"
SCHEMA = "market_indicator"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Credentials and query runner
# MAGIC `run_query` stands in for the old `funQueryRedshift`. When the connector is replaced, change only
# MAGIC this function. The stages just pass it SQL.

# COMMAND ----------

REDSHIFT = {
    "host": "rd_datamart.rdanalytics.cfahome.com",
    "port": 5439,
    "dbname": "rd_datamart",
    "user": "su.bae",
    # TODO: read from a secret scope, e.g. dbutils.secrets.get("<scope>", "redshift-password")
    "password": "",
    "aws_profile": "rd",
}


def run_query(sql, profile_name=REDSHIFT["aws_profile"]):
    """Run a SQL query and return a pandas DataFrame. Replace the body when the connector changes."""
    import boto3
    import psycopg2

    # The original helper opened an AWS session but did not pass its credentials on to Redshift.
    boto3.Session(profile_name=profile_name)
    conn = psycopg2.connect(
        host=REDSHIFT["host"],
        user=REDSHIFT["user"],
        port=REDSHIFT["port"],
        password=REDSHIFT["password"],
        dbname=REDSHIFT["dbname"],
    )
    try:
        return pd.read_sql(sql, con=conn)
    finally:
        conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inputs
# MAGIC Sources the pipelines read but don't produce. Tabular inputs are Unity Catalog tables with the
# MAGIC same schema as the old CSV/Excel/parquet files. Only the sales forecast model files stay in a Volume.

# COMMAND ----------

INPUT_DIR = "/Volumes/TODO_catalog/market_indicator/inputs"  # TODO: point at the Volume holding the model files

INPUT_PATHS = {
    "sales_model_lower": f"{INPUT_DIR}/sales_forecast_model_l.cbm",
    "sales_model_median": f"{INPUT_DIR}/sales_forecast_model_m.cbm",
    "sales_model_upper": f"{INPUT_DIR}/sales_forecast_model_u.cbm",
}

# Unity Catalog input tables (<catalog>.<schema>.<table>). TODO: replace with the real names.
INPUT_TABLES = {
    # Marginal return index
    "impact_training": "TODO_catalog.TODO_schema.distance_output",                       # was distance_output.parquet
    "pipeline_sales_forecast": "TODO_catalog.TODO_schema.sales_forecast_pipelines",      # was sales_forecast_pipelines.csv
    "integrated_strategy_sites": "TODO_catalog.TODO_schema.rd_integrated_strategy",      # was rd-integrated-strategy-2025-08-21.csv
    "sales_forecast_features": "TODO_catalog.TODO_schema.sales_batch_output_4year",      # was sales_batch_output_4year.csv
    "pipeline_operators": "TODO_catalog.TODO_schema.pipeline_2026_2027_notes",           # was 2026-2027 Pipeline.xlsx, "Pipeline Notes" sheet
    # Visit share index
    "visit_data": "TODO_catalog.TODO_schema.visit_data_placer",                          # Placer visits, M/D/YYYY start_date
    "visit_data_updated": "TODO_catalog.TODO_schema.visit_data_placer_2026_updated",     # Placer visits through 2026
}


def read_input_table(name):
    """A Unity Catalog input table from INPUT_TABLES, as pandas."""
    return spark.table(INPUT_TABLES[name]).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output tables and run artifacts
# MAGIC Tables live in `<write catalog>.market_indicator` and are appended to with the producing stage's
# MAGIC run id. Non-tabular state (nested dicts, models) is logged as MLflow artifacts on the stage run.

# COMMAND ----------

TABLES = {
    # Marginal return index: preprocessing (preprocessing_run_id)
    "mri_location_attributes": "mri_location_attributes",
    "mri_strategy_attributes": "mri_strategy_attributes",
    "mri_distance": "mri_distance",
    "mri_demographics": "mri_demographics",
    "mri_monthly_sales": "mri_monthly_sales",
    "mri_pipeline_sales": "mri_pipeline_sales",
    "mri_esf_sales": "mri_esf_sales",
    "mri_active_deals": "mri_active_deals",
    "mri_network_locations": "mri_network_locations",
    "mri_operator_selection": "mri_operator_selection",
    "mri_pipeline_operators": "mri_pipeline_operators",
    "mri_impact_history": "mri_impact_history",
    "mri_strategy_sites": "mri_strategy_sites",
    "mri_sales_forecast_features": "mri_sales_forecast_features",
    # Marginal return index: feature engineering (feature_run_id)
    "mri_impact_training": "mri_impact_training",
    "mri_impact_test_predictions": "mri_impact_test_predictions",
    # Marginal return index: index (index_run_id)
    "mri_simulation_results": "mri_simulation_results",
    "mri_simulated_network": "mri_simulated_network",
    "mri_submarket_index": "mri_submarket_index",
    # Visit share index
    "vsi_visits": "vsi_visits",                      # preprocessing
    "vsi_visits_updated": "vsi_visits_updated",      # preprocessing
    "vsi_store_master": "vsi_store_master",          # preprocessing
    "vsi_panel": "vsi_panel",                        # feature engineering
    "vsi_impact_coefficients": "vsi_impact_coefficients",  # index
    "vsi_event_catalog": "vsi_event_catalog",        # index
    "vsi_submarket_index": "vsi_submarket_index",    # index
    # Overall index score (index_run_id)
    "overall_index_score": "overall_index_score",
    "overall_quadrant_transitions": "overall_quadrant_transitions",
    # Monitoring (monitoring_run_id), shared by the three pipelines
    "monitoring_metrics": "monitoring_metrics",
    "monitoring_runs": "monitoring_runs",
}

ARTIFACTS = {
    "distance_matrix": "distance_matrix.pkl",
    "candidates": "candidates.pkl",
    "existing_stores": "existing_stores.pkl",
    "ownership_map": "ownership_map.pkl",
    "build_out_cost_by_region": "build_out_cost_by_region.pkl",
    "sales_by_store": "sales_by_store.pkl",
    "impact_model": "impact_catboost_model.cbm",
    "vsi_impact_model_summary": "impact_model_summary.txt",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table and artifact helpers

# COMMAND ----------

_INVALID_COLUMN_CHARS = " ,;{}()\n\t=#"


def _spark_ready(pdf):
    """Make a pandas frame writable to Delta.

    Replaces invalid characters in column names. Serializes list/dict/set cells to JSON. Casts
    mixed-type object columns to strings and periods to strings. Gives all-null object columns a
    string type.
    """
    pdf = pdf.copy()
    pdf.columns = ["".join("_" if c in _INVALID_COLUMN_CHARS else c for c in str(col)) for col in pdf.columns]
    for col in pdf.columns:
        s = pdf[col]
        if isinstance(s.dtype, pd.PeriodDtype):
            pdf[col] = s.astype(str)
        elif s.dtype == object:
            values = s.dropna()
            types = {type(v) for v in values}
            if not types:
                pdf[col] = s.astype("string")
            elif any(issubclass(t, (list, dict, set, tuple)) for t in types):
                pdf[col] = s.apply(lambda v: None if v is None else json.dumps(list(v) if isinstance(v, set) else v, default=str))
            elif len(types) > 1:
                pdf[col] = s.apply(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return pdf


def write_run_table(connector, pdf, table, run_col, run_id, schema_fqn):
    """Append a pandas frame to <schema_fqn>.<table> tagged with the stage run id."""
    output = spark.createDataFrame(_spark_ready(pdf)).withColumn(run_col, F.lit(run_id))
    connector.write(output, destination=f"{schema_fqn}.{TABLES[table]}", mode=WriteMode.APPEND)
    mlflow.log_metric(f"{table}_rows", len(pdf))


def read_run_table(table, run_col, run_id, schema_fqn):
    """Rows of <schema_fqn>.<table> written by one stage run, as pandas."""
    return (
        spark.sql(
            f"select * from identifier(:table) where {run_col} = :run_id",
            args={"table": f"{schema_fqn}.{TABLES[table]}", "run_id": run_id},
        )
        .drop(run_col)
        .toPandas()
    )


def log_pickle(obj, artifact):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ARTIFACTS[artifact])
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        mlflow.log_artifact(path, artifact_path="state")


def load_pickle(run_id, artifact):
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"state/{ARTIFACTS[artifact]}")
    with open(path, "rb") as f:
        return pickle.load(f)


def artifact_path(run_id, artifact):
    """Local copy of a run artifact (e.g. a model file)."""
    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"state/{ARTIFACTS[artifact]}")


def upstream_param(run_id, name):
    """Parameter logged by an upstream run, e.g. the preprocessing_run_id behind a feature run."""
    return mlflow.get_run(run_id).data.params[name]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Marginal return index: preprocessing and feature engineering

# COMMAND ----------

SALES_HISTORY_START = "2016-01-01"

# Impact model (fit inside feature engineering; logged as a run artifact, not registered)
IMPACT_ERROR_RANGE = (-40, 20)       # keep training rows with -40 <= error < 20
IMPACT_TEST_SIZE = 0.2
IMPACT_SPLIT_SEED = 20
IMPACT_MODEL_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.005,
    "depth": 5,
    "loss_function": "RMSE",
    "random_seed": 42,
    "bootstrap_type": "No",
}

# Network geometry
DRIVE_TIME_BUCKET_MIN = 5            # boundary buckets are 0-5, 5-10, 10-15 minutes
ACTIVE_DEAL_RADIUS = 0.2             # matched with haversine_distance, which returns km

# Candidate sales forecast and opening assumptions
FORECAST_YEAR = 2028                 # candidate sales range comes from sites forecast to open this year
CANDIDATE_OPEN_DATE = "2029-01-01"   # assumed opening date when computing candidate boundaries
REC_YEAR_OFFSET = 3                  # prod_year = rec_year + offset
NO_REC_YEAR = 2050                   # prod_year when a site has no rec_year

# Existing network defaults
DEFAULT_PIPELINE_OPEN_DATE = "2026-06-01"  # stores with no open date yet
DEFAULT_PIPELINE_OPEN_YEAR = "2028"
OWNERSHIP_REFERENCE_YEAR = 2028            # years since an operator last received a store
DEFAULT_STORE_SALES = 7_500_000
DEFAULT_DENSITY_CLASS = 4
DEFAULT_COMPETITOR_COUNT = 2

# Build-out cost by region
BUILD_OUT_COST_BY_REGION = {
    "Atlantic Region": {"avg_pct_change": 20.70117896976387, "last_mean": 4790833.241538461,
                        "last_median": 5319967.99, "std": 1635191.1907595405 / 2},
    "Midwest Region": {"avg_pct_change": 13.433963230558328, "last_mean": 5630067.5424,
                       "last_median": 5536303.74, "std": 585191.9791360076},
    "Northeast Region": {"avg_pct_change": 12.143127494548215, "last_mean": 6356554.516428572,
                         "last_median": 6028749.255, "std": 1142610.4311770445 / 2},
    "Southeast Region": {"avg_pct_change": 13.575353503908829, "last_mean": 4912179.8963636365,
                         "last_median": 5474106.095, "std": 1618499.7705548638 / 2},
    "Southwest Region": {"avg_pct_change": 21.9675073858253, "last_mean": 5262843.995789474,
                         "last_median": 5268997.01, "std": 1730145.868958881 / 2},
    "West Region": {"avg_pct_change": 20.08356087949052, "last_mean": 7540748.607777778,
                    "last_median": 7528923.26, "std": 1173953.1845786439 / 2},
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Marginal return index: Monte Carlo simulation

# COMMAND ----------

# Openings per year: {year: [total cap, SSAM cap]}. 82 stores already opened in 2026.
SCENARIOS = {
    "Current Plan": {
        2026: [0, 8], 2027: [0, 22], 2028: [0, 20],
        2029: [119 + 20 + 5 - 51, 20], 2030: [117 + 20 + 5 - 9, 20], 2031: [119 + 20 + 5 - 6, 20],
        2032: [109 + 20 + 5 - 1, 20], 2033: [119 + 10 + 5, 10], 2034: [119 + 10 + 5, 10],
        2035: [119 + 10 + 5, 10],
    },
}
PIPELINE_YEARS = [2026, 2027, 2028]  # years where known pipeline stores open before simulated sites
PIPELINE_OPEN_PROB = 0.9             # chance a pipeline store opens in its planned year
SIM_START_YEAR = 2028
SIM_YEARS = 8
MC_RUNS = 1
MC_MAX_WORKERS = 6
PROB_MODE = True                     # use prod buckets for pool selection
REGION_CAP = 0.25                    # per-region share of the yearly cap

# Candidate pool selection
PROBABILITY_BUCKETS = {">75": 0.9, "50-75": 0.65, "25-50": 0.4, "<25": 0.2}
DEFAULT_BUCKET_PROB = 0.65
POOL_BUFFER_PCT_RANGE = (100, 140)   # pool size = cap * (1 + buffer/100)

# Sales and build-cost sampling
SALES_INFLATION = 0.04               # per year from SALES_BASE_YEAR
SALES_BASE_YEAR = 2025
SALES_MID_HIGH_DEFLATOR = 1.1        # q50 and q90 are divided by this
BUILD_COST_GROWTH = 0.046            # per year from BUILD_COST_BASE_YEAR
BUILD_COST_BASE_YEAR = 2024

# Impact on existing stores
IMPACT_NOISE_RANGE = (0.02, 0.05)    # added to the impact model prediction
IMPACT_WINDOW = 3                    # years of impact history used to categorize owners
IMPACT_CONSECUTIVE_PCT = 0.05        # >= 2 years at this level -> consecutive
IMPACT_SINGLE_PCT = 0.15             # any year at this level -> single
OWNER_MAX_DRIVE_MIN = 20             # existing stores within this drive time are considered

# Single-unit operator: (max drive minutes, probability of receiving the new store)
SUO_ASSIGNMENT_PROBS = [(5, 0.70), (10, 0.5), (15, 0.35), (20, 0.15)]
# Two-unit operator
MUO2_ASSIGNMENT_MAX_MIN = 10
MUO2_ASSIGNMENT_PROB = 0.05

# IRR = build_cost * a + sales * b + impact_pct * c + intercept
IRR_COEF = {"build_cost": -2.592859e-8, "sales": 2.395230e-8, "impact_pct": -1.968792e-1, "intercept": 0.1835}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Marginal return index: submarket AUV and MRI

# COMMAND ----------

ACTUAL_SALES_START = "2020-01-01"
CURRENT_DATE = "2026-06-01"          # openings on/after this date apply impacts to sales history
AUV_INFLATION_RATE = 3               # percent per year when projecting AUV
AUV_PROJECTION_BASE_YEAR = 2026
MIN_MONTHS_FOR_ANNUAL = 10           # a year needs more than this many months of sales
MIN_EXISTING_AUV = 3_000_000         # smaller AUVs are dropped from existing/young averages
YOUNG_STORE_MAX_AGE = 2
SUBMARKET_YEARS = range(2020, 2034)

MRI_WEIGHTS = {
    "auv_score": 0.50,
    "growth_score": 0.30,
    "market_consistency_score": 0.10,
    "future_store_share_score": 0.10,
}
MRI_MATURITY_YEARS = 3.0             # store is fully mature after this many years
MRI_FULL_CONFIDENCE_STORES = 10      # store count giving full count confidence
MRI_CONFIDENCE_CLIP = (0.2, 1.0)
MRI_GROWTH_WEIGHTS = {"matured": 0.7, "young_store": 0.3}
MRI_OUTLIER_QUANTILES = (0.01, 0.99)

MRI_PLOT_YEAR = 2025

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visit share index

# COMMAND ----------

# Column names in the Placer visit data
STORE_ID = "id"
STORE_TYPE = "name"
MONTH = "month"
LAT = "lat"
LON = "lng"
VISITS = "visits"
DMA = "dma"
SUBMARKET = "submarket"

OUR_STORE_TYPE = "Chick-fil-A"

RADIUS_LIST = [1, 3, 5, 10]
EVENT_PRE_MONTHS = 6
EVENT_POST_MONTHS = 12
VISIT_GEO_UNIT = "submarket"
VISIT_EVENT_RADIUS = 5
CATALOG_PRE_MONTHS = 3
CATALOG_POST_MONTHS = 3
OPENING_NONZERO_STREAK = 3           # opening = first month of a 3-month nonzero streak
CLOSING_ZERO_STREAK = 6              # closing = first month of a 6-month zero streak
PLACER_MATCH_RADIUS_MILES = 0.5      # Placer POI <-> location_number match radius

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overall index score

# COMMAND ----------

QUADRANT_ORDER = ["Invest / Expand", "Build Share", "Defend / Optimize", "Deprioritize"]
