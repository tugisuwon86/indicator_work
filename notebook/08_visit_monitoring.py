# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: visit share index, monitoring
# MAGIC Reviews one index run and the feature and preprocessing runs behind it. Holds every
# MAGIC check and report from the old visit share analysis notebook. Metrics and a completion marker
# MAGIC are saved last.

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

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from roost.context import RoostContext
from roost.connector import RoostConnector
from roost.connector.core.types import WriteMode
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
)
from sklearn.neighbors import BallTree

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

dbutils.widgets.text("index_run_id", "")
INDEX_RUN_ID = dbutils.widgets.get("index_run_id").strip()
if not INDEX_RUN_ID:
    raise ValueError("Provide a successful index_run_id")
if mlflow.get_run(INDEX_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")
FEATURE_RUN_ID = upstream_param(INDEX_RUN_ID, "feature_run_id")
PREPROCESSING_RUN_ID = upstream_param(INDEX_RUN_ID, "preprocessing_run_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL queries (exploration only)

# COMMAND ----------

SQL_LOCATION_COORDINATES = """select location_number, latitude, longitude from rd_datamart.conf.location_attributes"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the run outputs

# COMMAND ----------

visits = read_run_table("vsi_visits", "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)
visits_updated = read_run_table("vsi_visits_updated", "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)
store_master = read_run_table("vsi_store_master", "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)
panel = read_run_table("vsi_panel", "feature_run_id", FEATURE_RUN_ID, SCHEMA_FQN)
coefficients = read_run_table("vsi_impact_coefficients", "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)
event_catalog = read_run_table("vsi_event_catalog", "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)
submarket_visit_index = read_run_table("vsi_submarket_index", "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Preprocessing checks: raw visit data

# COMMAND ----------

display(visits.head())
print(visits[STORE_TYPE].nunique(), "store types:", sorted(set(visits[STORE_TYPE])))
print(visits.shape, "rows;", (visits[VISITS] == 0).sum(), "zero-visit rows", f"({(visits[VISITS] == 0).mean():.1%})")
print(len(store_master), "stores;", int(store_master["is_new_store"].sum()), "new;", int(store_master["is_closed"].sum()), "closed")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Match Placer stores to location numbers
# MAGIC Nearest location within `PLACER_MATCH_RADIUS_MILES` for each Placer POI.

# COMMAND ----------

df_a = visits.groupby(STORE_ID).agg({LAT: "first", LON: "first", DMA: "first"}).reset_index()
df_a.columns = ["id", "lat", "lon", "dma"]
display(df_a[df_a["lat"].isna()].head())
df_a = df_a[df_a["lat"].notna()].copy()

loc = run_query(SQL_LOCATION_COORDINATES)
loc.columns = ["location_num", "lat", "lon"]
df_b = loc[loc["lat"].notna()].copy()

tree = BallTree(np.radians(df_b[["lat", "lon"]].astype(float).values), metric="haversine")
distance, index = tree.query(np.radians(df_a[["lat", "lon"]].astype(float).values), k=1)
df_a["distance_miles"] = distance[:, 0] * 3958.8
df_a["location_num"] = df_b.iloc[index[:, 0]]["location_num"].to_numpy()
df_a.loc[df_a["distance_miles"] > PLACER_MATCH_RADIUS_MILES, "location_num"] = np.nan

print(df_a["location_num"].notna().sum(), "Placer stores matched")
display(df_a[df_a["location_num"].notna()].sort_values("location_num").head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature engineering checks: panel

# COMMAND ----------

print(panel.shape, "store-months;", panel["our_store_id"].nunique(), "our stores")
display(panel.head())
display(panel[[c for c in panel.columns if c.startswith(("new_", "active_"))]].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Index checks

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impact regression

# COMMAND ----------

inference_metrics = mlflow.get_run(INDEX_RUN_ID).data.metrics
print({k: v for k, v in inference_metrics.items() if k.startswith("impact_model")})
display(coefficients)
print(mlflow.artifacts.load_text(f"runs:/{INDEX_RUN_ID}/state/{ARTIFACTS['vsi_impact_model_summary']}")[:5000])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visit share for one submarket

# COMMAND ----------

a = visits_updated[visits_updated[SUBMARKET] == "Atlanta, GA-West Cobb County"]
for i in sorted(a["year"].unique()):
    b = a[a["year"] == i]
    cfa = b.loc[b[STORE_TYPE] == OUR_STORE_TYPE, VISITS].sum()
    total = b[VISITS].sum()
    print(i, cfa, total, cfa / total * 100)
display(submarket_visit_index[submarket_visit_index["submarket"] == "Atlanta, GA-West Cobb County"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Opening impact reports

# COMMAND ----------

# MAGIC %md
# MAGIC #### Event study: visits around nearby openings

# COMMAND ----------

def build_event_study_dataset(panel, store_master, geo_unit="submarket", radius=5):
    """Our-store panel rows within EVENT_PRE_MONTHS/EVENT_POST_MONTHS of each nearby opening."""
    events = []
    openings = store_master[store_master["is_new_store"]].copy()
    our_stores = panel[["our_store_id", "our_lat", "our_lon", geo_unit]].drop_duplicates().copy()

    for _, opening in openings.iterrows():
        opening_id = opening[STORE_ID]
        opening_type = opening["store_type"]
        opening_month = opening["opening_month"]

        affected = our_stores[(our_stores[geo_unit] == opening[geo_unit]) & (our_stores["our_store_id"] != opening_id)].copy()
        if len(affected) == 0:
            continue
        affected["distance_miles"] = haversine_miles(affected["our_lat"].values, affected["our_lon"].values, opening["lat"], opening["lon"])
        affected = affected[affected["distance_miles"] <= radius]

        for _, a in affected.iterrows():
            g = panel[panel["our_store_id"] == a["our_store_id"]].copy()
            g["event_store_id"] = opening_id
            g["event_store_type"] = opening_type
            g["event_is_competitor"] = opening_type != OUR_STORE_TYPE
            g["event_is_our_store"] = opening_type == OUR_STORE_TYPE
            g["event_opening_month"] = opening_month
            g["distance_miles"] = a["distance_miles"]
            g["event_month"] = (g[MONTH].dt.year - opening_month.year) * 12 + (g[MONTH].dt.month - opening_month.month)
            g = g[(g["event_month"] >= -EVENT_PRE_MONTHS) & (g["event_month"] <= EVENT_POST_MONTHS)]
            events.append(g)

    if len(events) == 0:
        return pd.DataFrame()
    return pd.concat(events, ignore_index=True)


def summarize_event_study(event_df):
    return (
        event_df.groupby(["event_is_competitor", "event_month"])
        .agg(
            avg_yoy_change=("yoy_visit_change", "mean"),
            median_yoy_change=("yoy_visit_change", "median"),
            avg_visits=("our_visits", "mean"),
            n=("our_store_id", "nunique"),
        )
        .reset_index()
    )


event_df = build_event_study_dataset(panel, store_master, geo_unit=VISIT_GEO_UNIT, radius=VISIT_EVENT_RADIUS)

# Pre/post averages per (opening, our store)
event_study_catalog = (
    event_df.groupby(["event_store_id", "our_store_id", "event_store_type", "event_is_competitor", "distance_miles"])
    .agg(
        opening_month=("event_opening_month", "first"),
        pre_visits=("our_visits", lambda x: x[event_df.loc[x.index, "event_month"].between(-6, -1)].mean()),
        post_visits=("our_visits", lambda x: x[event_df.loc[x.index, "event_month"].between(1, 6)].mean()),
        pre_yoy=("yoy_visit_change", lambda x: x[event_df.loc[x.index, "event_month"].between(-6, -1)].mean()),
        post_yoy=("yoy_visit_change", lambda x: x[event_df.loc[x.index, "event_month"].between(1, 6)].mean()),
    )
    .reset_index()
)
event_study_catalog["visit_change_pct"] = (event_study_catalog["post_visits"] - event_study_catalog["pre_visits"]) / event_study_catalog["pre_visits"]
event_study_catalog["yoy_change"] = event_study_catalog["post_yoy"] - event_study_catalog["pre_yoy"]
display(event_study_catalog.head())
display(summarize_event_study(event_df))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Event curve

# COMMAND ----------

summary = (
    event_df.groupby(["event_is_competitor", "event_month"])
    .agg(avg=("yoy_visit_change", "mean"), std=("yoy_visit_change", "std"), n=("our_store_id", "nunique"))
    .reset_index()
)
summary["se"] = summary["std"] / np.sqrt(summary["n"])
summary["lower"] = summary["avg"] - 1.96 * summary["se"]
summary["upper"] = summary["avg"] + 1.96 * summary["se"]

plt.figure(figsize=(10, 5))
for flag, label in [(True, "Competitor Opening"), (False, "Our Store Opening")]:
    g = summary[summary.event_is_competitor == flag]
    plt.plot(g.event_month, g.avg, marker="o", label=label)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Months Relative to Opening")
plt.ylabel("Average YoY Change")
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Distance effect and submarket summary (event catalog)

# COMMAND ----------

display(event_catalog.head())

event_catalog["distance_bin"] = pd.cut(
    event_catalog.distance_miles, bins=[0, 1, 3, 5, 10, np.inf], labels=["0-1", "1-3", "3-5", "5-10", "10+"]
)
distance_summary = (
    event_catalog.groupby(["event_type", "distance_bin"])
    .agg(avg=("visit_change_pct", "mean"), median=("visit_change_pct", "median"), count=("visit_change_pct", "size"))
)
print(distance_summary)

submarket_summary = (
    event_catalog.groupby(["submarket", "event_type"])
    .agg(avg=("visit_change_pct", "mean"), count=("visit_change_pct", "size"))
    .reset_index()
)
display(submarket_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Publish monitoring metrics
# MAGIC Coverage of the batch. No alert limits are set yet, so nothing here fails the job.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "monitoring"}) as run:
    MONITORING_RUN_ID = run.info.run_id
    mlflow.log_param("index_run_id", INDEX_RUN_ID)

    metrics = {
        "submarket_years": len(submarket_visit_index),
        "submarkets": submarket_visit_index["submarket"].nunique(),
        "visit_share_index_missing": int(submarket_visit_index["visit_share_index"].isna().sum()),
        "events": len(event_catalog),
        "panel_rows": len(panel),
        "placer_stores_matched": int(df_a["location_num"].notna().sum()),
    }
    mlflow.log_metrics(metrics)
    metrics_df = spark.createDataFrame(
        [(PIPELINE, k, float(v)) for k, v in metrics.items()],
        "pipeline string, metric_name string, metric_value double",
    )
    output = metrics_df.withColumn("monitoring_run_id", F.lit(MONITORING_RUN_ID)).withColumn("index_run_id", F.lit(INDEX_RUN_ID))
    connector.write(output, destination=f"{SCHEMA_FQN}.{TABLES['monitoring_metrics']}", mode=WriteMode.APPEND)
    # Publish the completion marker last, after all monitoring results are saved.
    completed = spark.createDataFrame(
        [(MONITORING_RUN_ID, INDEX_RUN_ID, PIPELINE, "completed")],
        "monitoring_run_id string, index_run_id string, pipeline string, status string",
    )
    connector.write(completed, destination=f"{SCHEMA_FQN}.{TABLES['monitoring_runs']}", mode=WriteMode.APPEND)

# COMMAND ----------

dbutils.jobs.taskValues.set(key="monitoring_run_id", value=MONITORING_RUN_ID)
