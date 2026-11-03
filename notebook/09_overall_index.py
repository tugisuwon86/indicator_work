# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: overall index score
# MAGIC Combines the Marginal Return Index and the Visit Share Index into a strategy quadrant for each
# MAGIC submarket and year, and records how submarkets move between quadrants. It needs one finished
# MAGIC index run from `03_marginal_index` and `07_visit_index`.
# MAGIC
# MAGIC | Quadrant | MRI vs year median | VSI vs year median |
# MAGIC |---|---|---|
# MAGIC | Invest / Expand | at or above | at or above |
# MAGIC | Build Share | at or above | below |
# MAGIC | Defend / Optimize | below | at or above |
# MAGIC | Deprioritize | below | below |
# MAGIC
# MAGIC **Tables**: `overall_index_score`, `overall_quadrant_transitions`.

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
PIPELINE = "overall_index_score"
READ_CATALOG = ctx.read_catalog
SCHEMA_FQN = f"{ctx.write_catalog}.{SCHEMA}"

mlflow.set_tracking_uri("databricks")
connector = RoostConnector(
    context=ctx,
    format_handler=PySparkFormatHandler(),
    system=DatabricksSystem(DatabricksConnectionConfig(managed_ownership=False)),
)

dbutils.widgets.text("mri_index_run_id", "")
dbutils.widgets.text("vsi_index_run_id", "")
MRI_INDEX_RUN_ID = dbutils.widgets.get("mri_index_run_id").strip()
VSI_INDEX_RUN_ID = dbutils.widgets.get("vsi_index_run_id").strip()
if not MRI_INDEX_RUN_ID or not VSI_INDEX_RUN_ID:
    raise ValueError("Provide successful mri_index_run_id and vsi_index_run_id")
for upstream in (MRI_INDEX_RUN_ID, VSI_INDEX_RUN_ID):
    if mlflow.get_run(upstream).info.status != "FINISHED":
        raise ValueError("Choose successful upstream runs")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score the submarkets

# COMMAND ----------

def assign_quadrant(row):
    mri, vsi = row["final_mri"], row["visit_share_index"]
    if pd.isna(mri) or pd.isna(vsi):
        return "Insufficient Data"
    if mri >= row["mri_threshold"] and vsi >= row["vsi_threshold"]:
        return "Invest / Expand"
    elif mri >= row["mri_threshold"] and vsi < row["vsi_threshold"]:
        return "Build Share"
    elif mri < row["mri_threshold"] and vsi >= row["vsi_threshold"]:
        return "Defend / Optimize"
    return "Deprioritize"


with mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "index"}) as run:
    INDEX_RUN_ID = run.info.run_id
    mlflow.log_params({"mri_index_run_id": MRI_INDEX_RUN_ID, "vsi_index_run_id": VSI_INDEX_RUN_ID})

    def save(pdf, table):
        write_run_table(connector, pdf, table, "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)

    mr = read_run_table("mri_submarket_index", "index_run_id", MRI_INDEX_RUN_ID, SCHEMA_FQN)
    vs = read_run_table("vsi_submarket_index", "index_run_id", VSI_INDEX_RUN_ID, SCHEMA_FQN)

    # 1. Join MRI and VSI
    overall = mr.merge(vs[["submarket", "year", "visit_share_index"]], on=["submarket", "year"], how="left")
    overall["final_mri"] = pd.to_numeric(overall["final_mri"], errors="coerce")
    overall["visit_share_index"] = pd.to_numeric(overall["visit_share_index"], errors="coerce")

    # 2. Thresholds are the medians within each year
    thresholds = (
        overall.groupby("year")
        .agg(mri_threshold=("final_mri", "median"), vsi_threshold=("visit_share_index", "median"))
        .reset_index()
    )
    overall = overall.merge(thresholds, on="year", how="left")

    # 3. Quadrant
    overall["quadrant"] = overall.apply(assign_quadrant, axis=1)

    # 4. Transitions between consecutive years
    overall = overall.sort_values(["submarket", "year"]).copy()
    overall["prev_year"] = overall.groupby("submarket")["year"].shift(1)
    overall["prev_quadrant"] = overall.groupby("submarket")["quadrant"].shift(1)
    save(overall, "overall_index_score")

    transitions = overall[overall["prev_quadrant"].notna() & (overall["year"] == overall["prev_year"] + 1)].copy()
    transitions["year_transition"] = transitions["prev_year"].astype(int).astype(str) + " → " + transitions["year"].astype(int).astype(str)
    transition_counts = (
        transitions.groupby(["year_transition", "prev_quadrant", "quadrant"])
        .size()
        .reset_index(name="count")
    )
    save(transition_counts, "overall_quadrant_transitions")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="index_run_id", value=INDEX_RUN_ID)
