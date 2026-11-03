# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: overall index score, monitoring
# MAGIC Reviews one overall index run. Covers both index distributions, the quadrant
# MAGIC thresholds and counts, the opportunity matrix, transition matrices, and strategy migration.
# MAGIC Metrics and a completion marker are saved last.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installs

# COMMAND ----------

# Roost intentionally follows the latest release. Pin model/platform packages in the job environment.
# MAGIC %pip install --upgrade roost seaborn plotly

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
import pandas as pd
import plotly.graph_objects as go
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
PIPELINE = "overall_index_score"
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
MRI_INDEX_RUN_ID = upstream_param(INDEX_RUN_ID, "mri_index_run_id")
VSI_INDEX_RUN_ID = upstream_param(INDEX_RUN_ID, "vsi_index_run_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the run outputs

# COMMAND ----------

overall = read_run_table("overall_index_score", "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)
transition_counts = read_run_table("overall_quadrant_transitions", "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)
mr = read_run_table("mri_submarket_index", "index_run_id", MRI_INDEX_RUN_ID, SCHEMA_FQN)
vs = read_run_table("vsi_submarket_index", "index_run_id", VSI_INDEX_RUN_ID, SCHEMA_FQN)

print(mr.shape, vs.shape, overall.shape)
display(mr.head())
display(vs.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Index distributions by year

# COMMAND ----------

def summarize_by_year(df, col, title, ylabel):
    summary = (
        df.groupby("year")[col]
        .agg(
            count="count", mean="mean", median="median", std="std", min="min",
            p10=lambda x: x.quantile(0.10), p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75), p90=lambda x: x.quantile(0.90), max="max",
        )
        .round(2)
    )
    display(summary)

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df, x="year", y=col, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7},
    )
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


summarize_by_year(mr, "final_mri", "Final MRI Distribution by Year", "Final MRI")
summarize_by_year(vs, "visit_share_index", "Final VSI Distribution by Year", "Final VSI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Thresholds and quadrant counts

# COMMAND ----------

print("Thresholds by year")
display(overall[["year", "mri_threshold", "vsi_threshold"]].drop_duplicates().sort_values("year"))
print("Quadrant counts by year")
display(overall.groupby(["year", "quadrant"]).size().unstack(fill_value=0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Opportunity matrix by year

# COMMAND ----------

quadrant_styles = {"Invest / Expand": "o", "Build Share": "^", "Defend / Optimize": "s", "Deprioritize": "X"}

for year in sorted(overall["year"].dropna().unique()):
    temp = overall[overall["year"] == year].dropna(subset=["final_mri", "visit_share_index"]).copy()
    if temp.empty:
        continue
    mri_threshold = temp["mri_threshold"].iloc[0]
    vsi_threshold = temp["vsi_threshold"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 8))
    for quadrant, marker in quadrant_styles.items():
        q = temp[temp["quadrant"] == quadrant]
        ax.scatter(q["visit_share_index"], q["final_mri"], label=f"{quadrant} (n={len(q)})", marker=marker, s=65, alpha=0.65)

    ax.axvline(vsi_threshold, linestyle="--", linewidth=1.5)
    ax.axhline(mri_threshold, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Visit Share Index (Market Capture)")
    ax.set_ylabel("Marginal Return Index (Incremental Economics)")
    ax.set_title(f"Submarket Opportunity Matrix — {year}", fontsize=16, fontweight="bold")

    # threshold labels at the axis edges (the original referenced undefined xmax/ymax)
    xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.text(vsi_threshold, ymax, f" VSI Median = {vsi_threshold:.1f}", ha="left", va="top")
    ax.text(xmax, mri_threshold, f"MRI Median = {mri_threshold:.1f} ", ha="right", va="bottom")

    ax.grid(alpha=0.15)
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Quadrant transitions

# COMMAND ----------

display(transition_counts)

for period in sorted(transition_counts["year_transition"].unique()):
    temp = transition_counts[transition_counts["year_transition"] == period]
    matrix = temp.pivot_table(index="prev_quadrant", columns="quadrant", values="count", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(index=QUADRANT_ORDER, columns=QUADRANT_ORDER, fill_value=0)
    matrix.index.name = "From"
    matrix.columns.name = "To"
    print(f"\n{period}")
    display(matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Strategy migration (Sankey)

# COMMAND ----------

plot_df = overall[overall["quadrant"].isin(QUADRANT_ORDER)].copy()
years = sorted(plot_df["year"].unique())

# one node per year x quadrant, labeled with its submarket count
nodes = [(year, quadrant) for year in years for quadrant in QUADRANT_ORDER]
node_index = {node: i for i, node in enumerate(nodes)}
node_counts = plot_df.groupby(["year", "quadrant"])["submarket"].nunique().to_dict()
node_labels = [f"{year}<br>{quadrant}<br>n={node_counts.get((year, quadrant), 0):,}" for year, quadrant in nodes]

# flows between consecutive years
sources, targets, values = [], [], []
for year1, year2 in zip(years[:-1], years[1:]):
    if year2 != year1 + 1:
        continue
    left = plot_df[plot_df["year"] == year1][["submarket", "quadrant"]].rename(columns={"quadrant": "q1"})
    right = plot_df[plot_df["year"] == year2][["submarket", "quadrant"]].rename(columns={"quadrant": "q2"})
    counts = left.merge(right, on="submarket", how="inner").groupby(["q1", "q2"]).size().reset_index(name="count")
    for _, row in counts.iterrows():
        sources.append(node_index[(year1, row["q1"])])
        targets.append(node_index[(year2, row["q2"])])
        values.append(row["count"])

fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(pad=20, thickness=20, label=node_labels),
    link=dict(source=sources, target=targets, value=values),
))
fig.update_layout(title="Submarket Opportunity Strategy Migration", font_size=11, height=700)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Publish monitoring metrics
# MAGIC Coverage of the batch. No alert limits are set yet, so nothing here fails the job.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "monitoring"}) as run:
    MONITORING_RUN_ID = run.info.run_id
    mlflow.log_param("index_run_id", INDEX_RUN_ID)

    metrics = {
        "submarket_years": len(overall),
        "insufficient_data_share": float((overall["quadrant"] == "Insufficient Data").mean()),
        **{f"quadrant_{q.split(' ')[0].lower()}": int((overall["quadrant"] == q).sum()) for q in QUADRANT_ORDER},
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
