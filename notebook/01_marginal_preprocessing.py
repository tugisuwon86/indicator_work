# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: marginal return index, preprocessing
# MAGIC Pulls every source the marginal return index needs (Redshift queries and Unity Catalog input tables). Light
# MAGIC cleaning only, then each source is saved to its own `mri_*` table keyed by `preprocessing_run_id`.
# MAGIC Previously the first part of "Optimization step 1".

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installs

# COMMAND ----------

# Roost intentionally follows the latest release. Pin model/platform packages in the job environment.
# MAGIC %pip install --upgrade roost boto3 psycopg2-binary

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Config

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import mlflow
import pandas as pd
from roost.context import RoostContext
from roost.connector import RoostConnector
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Context and run selection

# COMMAND ----------

ctx = RoostContext(team=TEAM, defer_to_prod=True)
PIPELINE = "marginal_return_index"
READ_CATALOG = ctx.read_catalog
SCHEMA_FQN = f"{ctx.write_catalog}.{SCHEMA}"

mlflow.set_tracking_uri("databricks")
connector = RoostConnector(
    context=ctx,
    format_handler=PySparkFormatHandler(),
    system=DatabricksSystem(DatabricksConnectionConfig(managed_ownership=False)),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL queries

# COMMAND ----------

SQL_LOCATION_ATTRIBUTES = """select * from rd_datamart.conf.location_attributes"""

SQL_STRATEGY_ATTRIBUTES = """
SELECT
    id AS location_num,
    location_name,
    subtype,
    status AS location_type,
    strategy_tier AS location_status,
    state,
    dma,
    submarket,
    region
FROM
    conf.integrated_strategy
WHERE
    subtype IN ('FSU', 'DTO')
    AND status IN ('green', 'yellow')
"""

SQL_DISTANCE = """
SELECT
    CASE
        WHEN o_store IS NULL THEN o_strategy_id
        ELSE LPAD(o_store::CHAR(5), 5, '0')
    END AS location_num,
    CASE
        WHEN d_store IS NULL THEN d_strategy_id
        ELSE LPAD(d_store::CHAR(5), 5, '0')
    END AS dest_location_num,
    distance_miles AS distance_miles,
    distance_minutes AS distance_mins
FROM
    sbx_gis.mom_odcm_stores_pipeline_strategy_2
"""

SQL_DEMOGRAPHICS = """SELECT
    DISTINCT
    CASE
        WHEN store IS NULL THEN strategy_id
        ELSE LPAD(store::CHAR(5), 5, '0')
    END AS location_num,
    density_class::int AS density_classification_1,
    competitors AS competitor_count_1
  FROM sbx_gis.mom_primary_market_area_demographics_2
  WHERE distance_from_cfa_location = 1"""

SQL_MONTHLY_SALES = f"""
SELECT
    DATE_TRUNC('month', pos_business_dt)::DATE AS month,
    location_id as location_number,
    SUM(total_amt) AS total_net_sales,
    SUM(total_transactions) AS trans_cnt
FROM restaurant_performance.sales_qtr_hour_intervals
WHERE pos_business_dt >= '{SALES_HISTORY_START}'
GROUP BY
    DATE_TRUNC('month', pos_business_dt)::DATE,
    location_id
ORDER BY
    location_id,
    month;
"""

# Alternative monthly sales source (Databricks), kept for reference:
# SELECT
#     location_num,
#     date_trunc('month', trans_date) AS month,
#     sum(sls_net) AS monthly_sales,
#     count(distinct trans_date) as trans_count,
#     sum(trans_cnt) AS monthly_trans
# FROM
#     dx_prod.restfin_lh.daily_sales_extended
# WHERE
#     trans_date >= date('2016-01-01')
#     and trans_date < date_trunc('month', current_date)
# GROUP BY 1, 2

SQL_ESF = """SELECT
    location_number, forecasted_month as month, sales_forecast_30
  FROM rd_datamart.conf.enterprise_sales_forecast_ensemble
  WHERE forecast_run_dt = (select max(forecast_run_dt) from rd_datamart.conf.enterprise_sales_forecast_ensemble WHERE model = 'young store')
  AND model = 'young store'

  UNION

  SELECT
    location_number, forecasted_month as month, sales_forecast_30
  FROM rd_datamart.conf.enterprise_sales_forecast_ensemble
  WHERE forecast_run_dt = (select max(forecast_run_dt) from rd_datamart.conf.enterprise_sales_forecast_ensemble WHERE model = 'ensemble-no-gocf')
  AND model = 'ensemble-no-gocf'
  """

SQL_ACTIVE_DEALS = """select * from sbx_gis.mom_location_pipeline_strategy_2 where pipeline_status = 'Active Deal'"""

SQL_NETWORK_LOCATIONS = """select * from rd_datamart.conf.location_attributes where status in ('future', 'open', 'remodel', 'closed') and location_type_code in ('FSU', 'DTO')"""

SQL_OPERATOR_SELECTION = """select * from rd_datamart.people.operator_selection_data"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pull and save the sources

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "preprocessing"}) as run:
    PREPROCESSING_RUN_ID = run.info.run_id
    spark.sql("create schema if not exists identifier(:schema)", args={"schema": SCHEMA_FQN})

    def save(pdf, table):
        write_run_table(connector, pdf, table, "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)

    # Locations
    loc_all = run_query(SQL_LOCATION_ATTRIBUTES)
    save(loc_all, "mri_location_attributes")
    save(run_query(SQL_STRATEGY_ATTRIBUTES), "mri_strategy_attributes")
    save(run_query(SQL_NETWORK_LOCATIONS), "mri_network_locations")

    # Geography
    save(run_query(SQL_DISTANCE), "mri_distance")
    save(run_query(SQL_DEMOGRAPHICS), "mri_demographics")
    save(run_query(SQL_ACTIVE_DEALS), "mri_active_deals")

    # Sales history with location attributes
    monthly_sales = run_query(SQL_MONTHLY_SALES)
    monthly_sales["location_number"] = monthly_sales["location_number"].apply(lambda x: str(x).zfill(5))
    monthly_sales["month"] = pd.to_datetime(monthly_sales["month"])
    monthly_sales = pd.merge(monthly_sales, loc_all[["location_number", "opened_dt", "status", "location_type_code"]], on="location_number", how="left")
    monthly_sales = monthly_sales[["location_number", "opened_dt", "status", "location_type_code", "month", "total_net_sales", "trans_cnt"]]
    monthly_sales.columns = ["location_num", "open_date", "location_status", "location_type", "month", "total_net_sales", "trans_cnt"]
    save(monthly_sales, "mri_monthly_sales")

    # Enterprise sales forecast (latest young store and ensemble-no-gocf runs)
    save(run_query(SQL_ESF), "mri_esf_sales")

    # Pipeline store sales forecasts
    pipeline_sales = read_input_table("pipeline_sales_forecast")
    pipeline_sales["location_status"] = "future"
    pipeline_sales = pipeline_sales[["location_num", "open_date", "location_status", "location_type", "forecast_annual_sales"]]
    pipeline_sales["location_num"] = pipeline_sales["location_num"].apply(lambda x: str(int(x)).zfill(5))
    save(pipeline_sales, "mri_pipeline_sales")

    # Operators
    save(run_query(SQL_OPERATOR_SELECTION), "mri_operator_selection")
    pipeline_operators = read_input_table("pipeline_operators")
    pipeline_operators = pipeline_operators[["Location #", "Operator Selection"]]
    pipeline_operators["location_number"] = pipeline_operators["Location #"].apply(lambda x: str(int(x)).zfill(5) if pd.notna(x) else "")
    pipeline_operators = pipeline_operators.rename(columns={"Location #": "location_raw", "Operator Selection": "operator_selection"})
    save(pipeline_operators, "mri_pipeline_operators")

    # Impact history, strategy sites, and candidate sales forecast features
    save(read_input_table("impact_training"), "mri_impact_history")
    sites = read_input_table("integrated_strategy_sites")
    sites["location_number"] = sites["GlobalID"].apply(lambda x: x.replace("{", "").replace("}", ""))
    save(sites, "mri_strategy_sites")
    save(read_input_table("sales_forecast_features"), "mri_sales_forecast_features")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="preprocessing_run_id", value=PREPROCESSING_RUN_ID)
