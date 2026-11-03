# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: visit share index, preprocessing
# MAGIC Loads the Placer visit data from Unity Catalog (`INPUT_TABLES` in `00_config`) and flags each store's opening and closing months. Saves the
# MAGIC flagged visits, the store master, and the 2026-updated visits keyed by `preprocessing_run_id`.
# MAGIC This pipeline is independent of the marginal return index.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installs

# COMMAND ----------

# Roost intentionally follows the latest release. Pin model/platform packages in the job environment.
# MAGIC %pip install --upgrade roost

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
PIPELINE = "visit_share_index"
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
# MAGIC ### Functions

# COMMAND ----------

def load_store_data(input_table):
    """Placer visits with a month column (start_date is M/D/YYYY). Drops rows missing keys or visits."""
    df = read_input_table(input_table)
    df[MONTH] = df["start_date"].apply(lambda x: x.split("/")[-1] + "-" + x.split("/")[0].zfill(2) + "-" + x.split("/")[1].zfill(2))
    df[MONTH] = pd.to_datetime(df[MONTH])
    df[MONTH] = df[MONTH].dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=[STORE_ID, STORE_TYPE, MONTH, LAT, LON, VISITS])
    df[VISITS] = pd.to_numeric(df[VISITS], errors="coerce")
    df = df.dropna(subset=[VISITS])
    return df


def add_consecutive_flags(g, value_col="visits"):
    g = g.sort_values(MONTH).copy()
    g["is_nonzero"] = g[value_col].fillna(0) > 0
    g["is_zero"] = g[value_col].fillna(0) <= 0

    g["nonzero_streak"] = g["is_nonzero"].groupby((g["is_nonzero"] != g["is_nonzero"].shift()).cumsum()).cumcount() + 1
    g.loc[~g["is_nonzero"], "nonzero_streak"] = 0

    g["zero_streak"] = g["is_zero"].groupby((g["is_zero"] != g["is_zero"].shift()).cumsum()).cumcount() + 1
    g.loc[~g["is_zero"], "zero_streak"] = 0
    return g


def detect_openings_and_closings(df):
    """Store master with opening month (first month of a nonzero streak) and closing month (first month of a zero streak)."""
    first_data_month = df[MONTH].min()

    df2 = df.groupby(STORE_ID, group_keys=False).apply(add_consecutive_flags, value_col=VISITS).reset_index(drop=True)

    store_rows = []
    for store_id, g in df2.groupby(STORE_ID):
        g = g.sort_values(MONTH).copy()
        store_type = g[STORE_TYPE].iloc[0]

        opening_candidates = g[g["nonzero_streak"] >= OPENING_NONZERO_STREAK]
        if len(opening_candidates) > 0:
            opening_month = opening_candidates.iloc[0][MONTH] - pd.DateOffset(months=OPENING_NONZERO_STREAK - 1)
        else:
            opening_month = pd.NaT

        closing_candidates = g[g["zero_streak"] >= CLOSING_ZERO_STREAK]
        if len(closing_candidates) > 0:
            closing_month = closing_candidates.iloc[0][MONTH] - pd.DateOffset(months=CLOSING_ZERO_STREAK - 1)
        else:
            closing_month = pd.NaT

        store_rows.append({
            STORE_ID: store_id,
            "store_type": store_type,
            "opening_month": opening_month,
            "closing_month": closing_month,
            "is_new_store": pd.notna(opening_month) and opening_month > first_data_month,
            "is_closed": pd.notna(closing_month),
            "lat": g[LAT].iloc[0],
            "lon": g[LON].iloc[0],
            "dma": g[DMA].iloc[0],
            "submarket": g[SUBMARKET].iloc[0],
            "is_our_store": store_type == OUR_STORE_TYPE,
            "is_competitor": store_type != OUR_STORE_TYPE,
        })

    return pd.DataFrame(store_rows), df2

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load visits and detect openings

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "preprocessing"}) as run:
    PREPROCESSING_RUN_ID = run.info.run_id
    spark.sql("create schema if not exists identifier(:schema)", args={"schema": SCHEMA_FQN})

    def save(pdf, table):
        write_run_table(connector, pdf, table, "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)

    # Visits for the historical impact analysis
    visits = load_store_data("visit_data")
    store_master, visits_with_flags = detect_openings_and_closings(visits)
    save(visits_with_flags, "vsi_visits")
    save(store_master, "vsi_store_master")
    mlflow.log_metrics({
        "zero_visit_rows": int((visits[VISITS] == 0).sum()),
        "new_stores": int(store_master["is_new_store"].sum()),
        "closed_stores": int(store_master["is_closed"].sum()),
    })

    # Visits through 2026, for the visit share index
    visits_updated = read_input_table("visit_data_updated")
    visits_updated["year"] = pd.to_datetime(visits_updated["start_date"]).dt.year
    visits_updated[VISITS] = pd.to_numeric(visits_updated[VISITS], errors="coerce").fillna(0)
    save(visits_updated, "vsi_visits_updated")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="preprocessing_run_id", value=PREPROCESSING_RUN_ID)
