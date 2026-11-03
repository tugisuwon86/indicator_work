# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: visit share index
# MAGIC 1. Historical impact of nearby openings on our store visits: the fixed-effects regression
# MAGIC    and an event catalog of pre/post visit changes.
# MAGIC 2. Submarket visit share index (VSI), read by `09_overall_index`.
# MAGIC
# MAGIC Reads the `feature_run_id` from the feature engineering notebook.
# MAGIC
# MAGIC **Tables**: `vsi_impact_coefficients`, `vsi_event_catalog`, `vsi_submarket_index`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installs

# COMMAND ----------

# Roost intentionally follows the latest release. Pin model/platform packages in the job environment.
# MAGIC %pip install --upgrade roost statsmodels

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
import statsmodels.formula.api as smf
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

dbutils.widgets.text("feature_run_id", "")
FEATURE_RUN_ID = dbutils.widgets.get("feature_run_id").strip()
if not FEATURE_RUN_ID:
    raise ValueError("Provide a successful feature_run_id")
if mlflow.get_run(FEATURE_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")
PREPROCESSING_RUN_ID = upstream_param(FEATURE_RUN_ID, "preprocessing_run_id")

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


def run_impact_model(panel):
    """OLS of yoy_visit_change on nearby new/active opening counts with store and month fixed effects.
    Standard errors are clustered by store."""
    model_df = panel.dropna(subset=["yoy_visit_change"]).copy()
    model_df["month_str"] = model_df[MONTH].dt.strftime("%Y-%m")
    formula = """
        yoy_visit_change
        ~ new_comp_3mi
        + new_comp_5mi
        + new_our_3mi
        + new_our_5mi
        + active_comp_3mi
        + active_comp_5mi
        + active_our_3mi
        + active_our_5mi
        + new_comp_weighted_score
        + new_our_weighted_score
        + C(our_store_id)
        + C(month_str)
    """
    return smf.ols(formula=formula, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df["our_store_id"]})


def build_event_catalog(analysis_df, store_master, geo_unit="submarket", radius=10, pre_months=3, post_months=3):
    """One row per (opening, affected our-store): average visits before vs after the opening."""
    events = []
    our_stores = store_master[store_master.is_our_store]
    new_events = store_master[store_master.is_new_store]

    event_id = 0
    for _, event in new_events.iterrows():
        event_id += 1
        event_store = event[STORE_ID]
        event_month = event["opening_month"]
        event_type = "Competitor Open" if event["is_competitor"] else "Our Open"

        candidates = our_stores[our_stores[geo_unit] == event[geo_unit]].copy()
        candidates = candidates[candidates[STORE_ID] != event_store]
        if len(candidates) == 0:
            continue
        candidates["distance"] = haversine_miles(candidates["lat"].values, candidates["lon"].values, event["lat"], event["lon"])
        candidates = candidates[candidates.distance <= radius]

        for _, affected in candidates.iterrows():
            affected_store = affected[STORE_ID]
            hist = analysis_df[analysis_df[STORE_ID] == affected_store].sort_values(MONTH)
            pre = hist[(hist[MONTH] >= event_month - pd.DateOffset(months=pre_months)) & (hist[MONTH] < event_month)]
            post = hist[(hist[MONTH] > event_month) & (hist[MONTH] <= event_month + pd.DateOffset(months=post_months))]
            if len(pre) < 2 or len(post) < 2:
                continue
            pre_avg = pre[VISITS].mean()
            post_avg = post[VISITS].mean()
            events.append({
                "event_id": event_id,
                "event_type": event_type,
                "event_store_id": event_store,
                "affected_store_id": affected_store,
                "distance_miles": affected.distance,
                "submarket": affected["submarket"],
                "dma": affected["dma"],
                "opening_month": event_month,
                "pre_avg_visits": pre_avg,
                "post_avg_visits": post_avg,
                "visit_change_pct": (post_avg - pre_avg) / pre_avg,
            })

    return pd.DataFrame(events)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score the impact and the visit share index

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "index"}) as run:
    INDEX_RUN_ID = run.info.run_id
    mlflow.log_params({"feature_run_id": FEATURE_RUN_ID, "preprocessing_run_id": PREPROCESSING_RUN_ID,
                       "event_radius": VISIT_EVENT_RADIUS})

    def save(pdf, table):
        write_run_table(connector, pdf, table, "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)

    panel = read_run_table("vsi_panel", "feature_run_id", FEATURE_RUN_ID, SCHEMA_FQN)
    store_master = read_run_table("vsi_store_master", "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)
    visits = read_run_table("vsi_visits", "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)
    analysis_df = visits[visits[VISITS] > 0].copy()

    # 1. Historical impact regression
    model_result = run_impact_model(panel)
    coefficients = pd.DataFrame({
        "term": model_result.params.index,
        "coef": model_result.params.values,
        "std_err": model_result.bse.values,
        "p_value": model_result.pvalues.values,
    })
    coefficients = coefficients[~coefficients["term"].str.startswith(("C(our_store_id)", "C(month_str)"))]
    save(coefficients, "vsi_impact_coefficients")
    mlflow.log_text(model_result.summary().as_text(), f"state/{ARTIFACTS['vsi_impact_model_summary']}")
    mlflow.log_metrics({"impact_model_r2": model_result.rsquared, "impact_model_nobs": model_result.nobs})

    # 2. Event catalog
    event_catalog = build_event_catalog(
        analysis_df=analysis_df, store_master=store_master, geo_unit=VISIT_GEO_UNIT, radius=VISIT_EVENT_RADIUS,
        pre_months=CATALOG_PRE_MONTHS, post_months=CATALOG_POST_MONTHS,
    )
    save(event_catalog, "vsi_event_catalog")

    # 3. Visit share index
    # Rebuilt step, please confirm. No notebook built this output before. It is our-store visits as
    # a share of all visits per submarket and year, scored as a within-year percentile (0-100).
    visits_updated = read_run_table("vsi_visits_updated", "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)
    visit_totals = visits_updated.groupby([SUBMARKET, "year"])[VISITS].sum().rename("total_visits")
    our_visits = visits_updated[visits_updated[STORE_TYPE] == OUR_STORE_TYPE].groupby([SUBMARKET, "year"])[VISITS].sum().rename("our_visits")

    submarket_visit_index = pd.concat([visit_totals, our_visits], axis=1).fillna({"our_visits": 0}).reset_index()
    submarket_visit_index = submarket_visit_index.rename(columns={SUBMARKET: "submarket"})
    submarket_visit_index["visit_share_pct"] = np.where(
        submarket_visit_index["total_visits"] > 0,
        submarket_visit_index["our_visits"] / submarket_visit_index["total_visits"] * 100,
        np.nan,
    )
    submarket_visit_index["visit_share_index"] = submarket_visit_index.groupby("year")["visit_share_pct"].rank(pct=True) * 100
    save(submarket_visit_index, "vsi_submarket_index")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="index_run_id", value=INDEX_RUN_ID)
