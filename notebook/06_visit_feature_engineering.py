# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: visit share index, feature engineering
# MAGIC Builds the our-store monthly panel. For each store-month it adds counts and distance-weighted
# MAGIC scores for nearby new and historical openings (our stores and competitors) in the same geo unit,
# MAGIC plus YoY visit metrics.
# MAGIC
# MAGIC **Table**: `vsi_panel` keyed by `feature_run_id`.

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
import numpy as np
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

dbutils.widgets.text("preprocessing_run_id", "")
PREPROCESSING_RUN_ID = dbutils.widgets.get("preprocessing_run_id").strip()
if not PREPROCESSING_RUN_ID:
    raise ValueError("Provide a successful preprocessing_run_id")
if mlflow.get_run(PREPROCESSING_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Functions

# COMMAND ----------

def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def build_our_store_panel(df, store_master):
    our_ids = store_master.loc[store_master["is_our_store"], STORE_ID]
    panel = df[df[STORE_ID].isin(our_ids)].copy()
    panel = panel.rename(columns={
        STORE_ID: "our_store_id", STORE_TYPE: "our_store_type", LAT: "our_lat", LON: "our_lon",
        VISITS: "our_visits", DMA: "dma", SUBMARKET: "submarket",
    })
    return panel[["our_store_id", MONTH, "our_store_type", "our_lat", "our_lon", "our_visits", "dma", "submarket"]]


def _nearby_opening_features(panel, store_master, geo_unit, prefix, active_only):
    """Count/weight new stores (ours and competitors) near each our-store month in the same geo unit.

    active_only=True drops openings that have closed by that month ("new_" features).
    active_only=False counts every historical opening ("active_" features).
    """
    results = []
    openings = store_master[store_master["is_new_store"]].copy()

    for _, row in panel.iterrows():
        month = row[MONTH]
        mask = (
            (openings[geo_unit] == row[geo_unit])
            & (openings[STORE_ID] != row["our_store_id"])
            & (openings["opening_month"] <= month)
        )
        if active_only:
            mask &= openings["closing_month"].isna() | (openings["closing_month"] > month)
        candidates = openings[mask].copy()

        base = row.to_dict()
        for r in RADIUS_LIST:
            base[f"{prefix}_comp_{r}mi"] = 0
            base[f"{prefix}_our_{r}mi"] = 0
        if active_only:
            base["nearest_new_comp_dist"] = np.nan
            base["nearest_new_our_dist"] = np.nan
        base[f"{prefix}_comp_weighted_score"] = 0.0
        base[f"{prefix}_our_weighted_score"] = 0.0

        if len(candidates) > 0:
            candidates["distance_miles"] = haversine_miles(row["our_lat"], row["our_lon"], candidates["lat"].values, candidates["lon"].values)
            comp = candidates[candidates["is_competitor"]]
            our = candidates[candidates["is_our_store"]]

            for r in RADIUS_LIST:
                base[f"{prefix}_comp_{r}mi"] = (comp["distance_miles"] <= r).sum()
                base[f"{prefix}_our_{r}mi"] = (our["distance_miles"] <= r).sum()
            if len(comp) > 0:
                if active_only:
                    base["nearest_new_comp_dist"] = comp["distance_miles"].min()
                base[f"{prefix}_comp_weighted_score"] = (1 / comp["distance_miles"].clip(lower=0.1)).sum()
            if len(our) > 0:
                if active_only:
                    base["nearest_new_our_dist"] = our["distance_miles"].min()
                base[f"{prefix}_our_weighted_score"] = (1 / our["distance_miles"].clip(lower=0.1)).sum()

        results.append(base)

    return pd.DataFrame(results)


def add_opening_features(panel, store_master, geo_unit="submarket"):
    """New openings still open at each month."""
    return _nearby_opening_features(panel, store_master, geo_unit, prefix="new", active_only=True)


def add_active_opening_features(panel, store_master, geo_unit="submarket"):
    """Every historical opening up to each month (cumulative impact)."""
    return _nearby_opening_features(panel, store_master, geo_unit, prefix="active", active_only=False)


def add_yoy_metrics(panel):
    panel = panel.sort_values(["our_store_id", MONTH]).copy()
    panel["mom_pct"] = panel.groupby("our_store_id")["our_visits"].pct_change()
    panel["rolling3"] = panel.groupby("our_store_id")["our_visits"].transform(lambda s: s.rolling(3).mean())
    panel["rolling3_pct"] = panel.groupby("our_store_id")["rolling3"].pct_change()
    panel["quarter"] = panel[MONTH].dt.to_period("Q")
    panel["visits_lag_12"] = panel.groupby("our_store_id")["our_visits"].shift(12)
    panel["yoy_visit_change"] = (panel["our_visits"] - panel["visits_lag_12"]) / panel["visits_lag_12"]
    panel["log_visits"] = np.log1p(panel["our_visits"])
    return panel

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the features

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "feature_engineering"}) as run:
    FEATURE_RUN_ID = run.info.run_id
    mlflow.log_param("preprocessing_run_id", PREPROCESSING_RUN_ID)
    mlflow.log_params({"geo_unit": VISIT_GEO_UNIT, "radius_list": RADIUS_LIST})

    visits = read_run_table("vsi_visits", "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)
    store_master = read_run_table("vsi_store_master", "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)
    analysis_df = visits[visits[VISITS] > 0].copy()

    panel = build_our_store_panel(analysis_df, store_master)
    panel = add_opening_features(panel, store_master, geo_unit=VISIT_GEO_UNIT)
    panel2 = add_active_opening_features(panel, store_master, geo_unit=VISIT_GEO_UNIT)
    active_cols = [c for c in panel2.columns if c.startswith("active_")]
    panel = panel.merge(panel2[["our_store_id", MONTH] + active_cols], on=["our_store_id", MONTH], how="left")
    # The regression target needs the YoY metrics. The original skipped this and hit KeyError 'yoy_visit_change'.
    panel = add_yoy_metrics(panel)

    write_run_table(connector, panel, "vsi_panel", "feature_run_id", FEATURE_RUN_ID, SCHEMA_FQN)

# COMMAND ----------

dbutils.jobs.taskValues.set(key="feature_run_id", value=FEATURE_RUN_ID)
