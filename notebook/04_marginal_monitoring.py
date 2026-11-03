# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: marginal return index, monitoring
# MAGIC Reviews one index run and the feature and preprocessing runs behind it. Holds every
# MAGIC check and report that used to be spread across the optimization notebooks, grouped by the notebook
# MAGIC whose output it checks. Metrics and a completion marker are saved last.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installs

# COMMAND ----------

# Roost intentionally follows the latest release. Pin model/platform packages in the job environment.
# MAGIC %pip install --upgrade roost seaborn boto3 psycopg2-binary

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

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from roost.context import RoostContext
from roost.connector import RoostConnector
from roost.connector.core.types import WriteMode
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

dbutils.widgets.text("index_run_id", "")
INDEX_RUN_ID = dbutils.widgets.get("index_run_id").strip()
if not INDEX_RUN_ID:
    raise ValueError("Provide a successful index_run_id")
if mlflow.get_run(INDEX_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")
FEATURE_RUN_ID = upstream_param(INDEX_RUN_ID, "feature_run_id")
PREPROCESSING_RUN_ID = upstream_param(INDEX_RUN_ID, "preprocessing_run_id")


def read_preprocessed(table):
    return read_run_table(table, "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)


def read_features(table):
    return read_run_table(table, "feature_run_id", FEATURE_RUN_ID, SCHEMA_FQN)


def read_inference(table):
    return read_run_table(table, "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL queries (exploration only)

# COMMAND ----------

SQL_CHECK_INTEGRATED_STRATEGY = """select * from conf.integrated_strategy"""
SQL_CHECK_ESF_SAMPLE = """select * from rd_datamart.conf.enterprise_sales_forecast_ensemble limit 10"""
SQL_CHECK_ESF_RUNS = """SELECT
    distinct model, forecast_run_dt
  FROM rd_datamart.conf.enterprise_sales_forecast_ensemble
  """

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the run outputs

# COMMAND ----------

loc_all = read_preprocessed("mri_location_attributes")
monthly_sales = read_preprocessed("mri_monthly_sales")
esf_sales = read_preprocessed("mri_esf_sales")
pipeline_sales = read_preprocessed("mri_pipeline_sales")
pipeline_operators = read_preprocessed("mri_pipeline_operators")
strategy = read_preprocessed("mri_strategy_attributes")

impact_training = read_features("mri_impact_training")
impact_eval = read_features("mri_impact_test_predictions")
candidates = load_pickle(FEATURE_RUN_ID, "candidates")
existing_stores = load_pickle(FEATURE_RUN_ID, "existing_stores")
ownership_map = load_pickle(FEATURE_RUN_ID, "ownership_map")

simulation_results = read_inference("mri_simulation_results")
simulated_network = read_inference("mri_simulated_network")
submarket_df_w_index = read_inference("mri_submarket_index")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Preprocessing checks

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2026 openings and pipeline opening years

# COMMAND ----------

loc_all["year"] = loc_all["opened_dt"].apply(lambda x: str(x)[:4])
display(loc_all[loc_all["year"] == "2026"].sort_values("opened_dt").tail(20))
print(Counter(loc_all[loc_all["year"] == "2026"].subtype))

pipes = loc_all[loc_all["status"] == "future"].copy()
pipes["open"] = pipes["projected_open_dt"].apply(lambda x: str(x)[:4])
print(Counter(pipes.open))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Sales history

# COMMAND ----------

print(min(monthly_sales.month), max(monthly_sales.month), set(monthly_sales.location_status))
display(monthly_sales[monthly_sales["location_num"] == "01864"].tail())
display(pipeline_sales.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Enterprise sales forecast (ESF)

# COMMAND ----------

display(run_query(SQL_CHECK_ESF_SAMPLE).head())
display(esf_sales.head())
print(min(esf_sales.month), max(esf_sales.month))
display(run_query(SQL_CHECK_ESF_RUNS))

esf_check = esf_sales.copy()
esf_check.columns = ["location_num", "month", "total_net_sales"]
display(monthly_sales[monthly_sales["location_num"] == "05677"].sort_values("month").head(10))
display(esf_check[esf_check["location_num"] == "05677"].sort_values("month").head(20))

loc_check = loc_all[loc_all["status"].isin(["open", "remodel", "closed"]) & loc_all["subtype_id"].isin(["FSU", "DTO"])]
print(set(loc_all.status))
display(pd.merge(loc_check[["location_number", "status", "subtype_id"]], esf_check, left_on="location_number", right_on="location_num", how="left").head())
display(loc_check[loc_check["location_number"] == "01864"].head())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Integrated strategy

# COMMAND ----------

sites_ = run_query(SQL_CHECK_INTEGRATED_STRATEGY)
display(sites_.head())
print(set(sites_.strategy_tier))
display(strategy.head(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature engineering checks

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impact model fit

# COMMAND ----------

feature_metrics = mlflow.get_run(FEATURE_RUN_ID).data.metrics
print({k: v for k, v in feature_metrics.items() if k.startswith("impact_")})
print(impact_training.shape)
display(impact_training.head())

y_train = impact_eval.loc[impact_eval["dataset_split"] == "train", "target"].values
test = impact_eval[impact_eval["dataset_split"] == "test"]
y_test, predictions = test["target"].values, test["prediction"].values

for values, label in [(y_train, "y_train"), (y_test, "y_test")]:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, alpha=0.5, label=label)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Target distribution ({label})")
    plt.legend()
    plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(predictions, y_test, alpha=0.5)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual")
min_val, max_val = min(min(predictions), min(y_test)), max(max(predictions), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val])  # diagonal reference line
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Candidates, existing stores, and ownership

# COMMAND ----------

print({k: v for k, v in feature_metrics.items() if not k.startswith("impact_")})
print(len(candidates), "candidates;", len(existing_stores), "existing stores;", len(ownership_map), "operators")
print("5-character candidate ids:", [x for x in candidates.keys() if len(x) == 5])
display(pipeline_operators.head())
print(existing_stores.get("05243"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Simulation checks

# COMMAND ----------

display(simulation_results.head(20))

new_openings = simulated_network[simulated_network["opened_dt"].notna()
                                 & (pd.to_datetime(simulated_network["opened_dt"], errors="coerce") >= pd.to_datetime(CURRENT_DATE))]
print(len(new_openings), f"openings on/after {CURRENT_DATE}")
display(new_openings.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Openings by year

# COMMAND ----------

loc = loc_all[loc_all["status"].isin(["future", "open", "remodel", "closed"]) & loc_all["subtype_id"].isin(["FSU", "DTO"])]
loc_att = {row.location_number: (row.subtype_id, row.submarket, row.dma) for _, row in loc.iterrows()}
loc_att.update({row.location_num: (row.subtype, row.submarket, row.dma) for _, row in strategy.iterrows()})

for site_id in simulated_network["site_id"]:
    if site_id in loc_att and loc_att[site_id][1] == "Abilene, TX-South":
        print(site_id, loc_att[site_id])

history = defaultdict(int)
history_ = defaultdict(int)
for year in range(2027, 2034):
    for _, row in simulated_network.iterrows():
        if str(row["opened_dt"]).split("-")[0] == str(year):
            if row["site_id"] in loc_att and loc_att[row["site_id"]][0] in ("FSU", "DTO"):
                history[str(row["opened_dt"])[:7]] += 1
                history_[str(row["opened_dt"])[:4]] += 1
print(dict(history))
print(dict(history_))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Submarket AUV checks

# COMMAND ----------

display(submarket_df_w_index[submarket_df_w_index["submarket"] == "Abilene, TX-South"].head(15))
display(submarket_df_w_index[submarket_df_w_index["submarket"] == "Abilene, TX-North"].head(20))
display(submarket_df_w_index[submarket_df_w_index["submarket"] == "Albany, NY | Albany South"][["year", "maturity", "number_of_existing_stores", "number_of_new_stores", "number_of_pipeline", "number_of_strategy"]].head(10))
display(submarket_df_w_index[submarket_df_w_index["submarket"] == "Atlanta, GA-Smyrna-Mableton"][["year", "maturity", "number_of_existing_stores", "number_of_new", "number_of_pipeline", "number_of_strategy"]].head(10))
print(sorted(submarket_df_w_index.columns))
print(submarket_df_w_index.submarket.nunique(), "submarkets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MRI reports

# COMMAND ----------

# MAGIC %md
# MAGIC #### Final MRI distribution by year

# COMMAND ----------

for s in sorted(set(submarket_df_w_index.year)):
    temp = submarket_df_w_index[submarket_df_w_index["year"] == s]
    sns.displot(data=temp, x="final_mri", kde=True)
    plt.title(s)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Example markets
# MAGIC Top MRI, top growth, top confidence, and bottom MRI submarkets.

# COMMAND ----------

df_plot = submarket_df_w_index[submarket_df_w_index["year"] == MRI_PLOT_YEAR].copy()

# examples within MRI_PLOT_YEAR
selected = pd.concat([
    df_plot.nlargest(1, "final_mri"),
    df_plot.nlargest(1, "growth_score"),
    df_plot.nlargest(1, "market_confidence"),
    df_plot.nsmallest(1, "final_mri"),
]).drop_duplicates("submarket")
display(selected)

# distinct examples across all years
selected_markets = []
selected_markets.append(submarket_df_w_index.loc[submarket_df_w_index["final_mri"].idxmax(), "submarket"])
growth = submarket_df_w_index[~submarket_df_w_index["submarket"].isin(selected_markets)]
selected_markets.append(growth.loc[growth["growth_score"].idxmax(), "submarket"])
conf = submarket_df_w_index[~submarket_df_w_index["submarket"].isin(selected_markets)]
selected_markets.append(conf.loc[conf["market_confidence"].idxmax(), "submarket"])
worst = submarket_df_w_index[~submarket_df_w_index["submarket"].isin(selected_markets)]
selected_markets.append(worst.loc[worst["final_mri"].idxmin(), "submarket"])
print(selected_markets)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Component radar for the example markets

# COMMAND ----------

cols = ["auv_score", "growth_score", "market_consistency_score", "future_store_share_score"]
labels = ["AUV", "Growth", "Consistency", "Future\nStores"]
plot_df = df_plot[df_plot["submarket"].isin(selected_markets)].set_index("submarket")

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
for market in plot_df.index:
    values = plot_df.loc[market, cols].values.tolist()
    values += [values[0]]
    ax.plot(angles, values, linewidth=2, label=market)
    ax.fill(angles, values, alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
plt.title(f"Market Component Comparison ({MRI_PLOT_YEAR})")
plt.legend(bbox_to_anchor=(1.2, 1))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Final MRI vs market confidence
# MAGIC Labels only the example markets. The original labeled every row and hung the notebook.

# COMMAND ----------

plt.figure(figsize=(10, 8))
plt.scatter(df_plot["market_confidence"], df_plot["final_mri"], alpha=0.6)
for _, row in df_plot[df_plot["submarket"].isin(selected_markets)].iterrows():
    plt.scatter(row["market_confidence"], row["final_mri"], s=150)
    plt.text(row["market_confidence"] + 0.01, row["final_mri"], row["submarket"], fontsize=9)
plt.xlabel("Market Confidence")
plt.ylabel("Final MRI")
plt.title(f"Final MRI vs Market Confidence ({MRI_PLOT_YEAR})")
plt.grid(alpha=0.3)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Market opportunity map

# COMMAND ----------

plot_df = df_plot.copy()
plot_df["bubble_size"] = plot_df["final_mri"] * 8

plt.figure(figsize=(12, 9))
scatter = plt.scatter(
    plot_df["price_normalized_auv"], plot_df["growth_rate_selected"], s=plot_df["bubble_size"],
    c=plot_df["market_confidence"], cmap="viridis", alpha=0.7, edgecolors="black", linewidth=0.5,
)
for _, row in plot_df.nlargest(10, "final_mri").iterrows():  # label top MRI markets
    plt.text(row["price_normalized_auv"], row["growth_rate_selected"], row["submarket"], fontsize=8)
plt.colorbar(scatter, label="Market Confidence")
plt.xlabel("Price-Normalized AUV")
plt.ylabel("Sales Growth Rate (%)")
plt.title(f"Market Opportunity Map ({MRI_PLOT_YEAR})")
plt.grid(alpha=0.3)
plt.axvline(plot_df["price_normalized_auv"].median(), color="gray", linestyle="--", alpha=0.5)
plt.axhline(plot_df["growth_rate_selected"].median(), color="gray", linestyle="--", alpha=0.5)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Publish monitoring metrics
# MAGIC Coverage of the batch. No alert limits are set yet, so nothing here fails the job.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "monitoring"}) as run:
    MONITORING_RUN_ID = run.info.run_id
    mlflow.log_param("index_run_id", INDEX_RUN_ID)

    metrics = {
        "submarket_years": len(submarket_df_w_index),
        "submarkets": submarket_df_w_index["submarket"].nunique(),
        "final_mri_missing": int(submarket_df_w_index["final_mri"].isna().sum()),
        "final_mri_mean": float(submarket_df_w_index["final_mri"].mean()),
        "simulated_openings": int(simulated_network["simulated"].sum()),
        "simulation_avg_irr": float(simulation_results["avg_IRR"].mean()),
        "candidates": len(candidates),
        "existing_stores": len(existing_stores),
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
