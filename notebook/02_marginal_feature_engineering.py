# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: marginal return index, feature engineering
# MAGIC Builds the simulation state from the preprocessing run: distance matrix, impact model, store
# MAGIC sales, candidate sites, existing network, and operator ownership.
# MAGIC Previously the rest of "Optimization step 1".
# MAGIC
# MAGIC The impact model is fitted here as a helper for the simulation. It is logged as a run artifact,
# MAGIC not registered.
# MAGIC
# MAGIC **Run artifacts** (`state/`): `distance_matrix`, `candidates`, `existing_stores`, `ownership_map`,
# MAGIC `build_out_cost_by_region`, `sales_by_store`, `impact_model`.
# MAGIC **Tables**: `mri_impact_training`, `mri_impact_test_predictions`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installs

# COMMAND ----------

# Roost intentionally follows the latest release. Pin model/platform packages in the job environment.
# MAGIC %pip install --upgrade roost catboost

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

import math
import os
import tempfile
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from roost.context import RoostContext
from roost.connector import RoostConnector
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
)
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

dbutils.widgets.text("preprocessing_run_id", "")
PREPROCESSING_RUN_ID = dbutils.widgets.get("preprocessing_run_id").strip()
if not PREPROCESSING_RUN_ID:
    raise ValueError("Provide a successful preprocessing_run_id")
if mlflow.get_run(PREPROCESSING_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")


def read_preprocessed(table):
    return read_run_table(table, "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helpers

# COMMAND ----------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Straight-line distance between two points in decimal degrees, in kilometers."""
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_boundaries(store, store_open_dt, open_meta, fsu_dto_only):
    """Count stores opened before `store_open_dt` near `store`, by drive-time bucket and by years before.

    open_meta: {location: (open date, subtype)}
    Returns boundary counts (0-5/5-10/10-15 min), year counts (0/1/2 years before),
    distances (miles) of stores within 15 min, and nearby stores within 10 min.
    """
    boundary, years, distances, nearby_stores = [0, 0, 0], [0, 0, 0], [], []
    for nearby in distance_matrix.get(store, {}):
        if nearby not in open_meta or open_meta[nearby][0] is None or open_meta[nearby][0] >= store_open_dt:
            continue
        if fsu_dto_only and open_meta[nearby][1] not in ("FSU", "DTO"):
            continue
        miles, minutes = distance_matrix[store][nearby]
        bucket = int(minutes // DRIVE_TIME_BUCKET_MIN)
        if bucket < 3:
            boundary[bucket] += 1
            distances += [miles]
            if bucket < 2:
                nearby_stores += [nearby]
        diff = int((store_open_dt - open_meta[nearby][0]).days // 365)
        if diff < 3:
            years[diff] += 1
    return {"boundary": boundary, "years": years, "distances": distances, "nearby_stores": nearby_stores}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the preprocessed sources

# COMMAND ----------

loc_all = read_preprocessed("mri_location_attributes")
loc_network = read_preprocessed("mri_network_locations")
distance = read_preprocessed("mri_distance")
demo = read_preprocessed("mri_demographics")
demo_by_loc = demo.rename(columns={"location_num": "location_number"})
monthly_sales = read_preprocessed("mri_monthly_sales")
pipeline_sales = read_preprocessed("mri_pipeline_sales")
impact = read_preprocessed("mri_impact_history")
sites = read_preprocessed("mri_strategy_sites")
active_deal_dots = read_preprocessed("mri_active_deals")
features = read_preprocessed("mri_sales_forecast_features")
operator = read_preprocessed("mri_operator_selection")
pipelines_df = read_preprocessed("mri_pipeline_operators")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the features

# COMMAND ----------

# This stage spans several cells, so the run is closed explicitly at the end.
# A run left open by an earlier failed attempt is marked FAILED so downstream stages reject it.
if mlflow.active_run():
    mlflow.end_run(status="FAILED")
mlflow_run = mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "feature_engineering"})
FEATURE_RUN_ID = mlflow_run.info.run_id
mlflow.log_param("preprocessing_run_id", PREPROCESSING_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Distance matrix
# MAGIC `distance_matrix[a][b] = (miles, minutes)`, symmetric.

# COMMAND ----------

distance_matrix = {}
for _, rows in distance.iterrows():
    if rows.location_num not in distance_matrix:
        distance_matrix[rows.location_num] = {}
    if rows.dest_location_num not in distance_matrix:
        distance_matrix[rows.dest_location_num] = {}
    distance_matrix[rows.location_num][rows.dest_location_num] = (rows.distance_miles, rows.distance_mins)
    distance_matrix[rows.dest_location_num][rows.location_num] = (rows.distance_miles, rows.distance_mins)

log_pickle(distance_matrix, "distance_matrix")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Impact model
# MAGIC Predicts the sales error (%) of an existing store after a new store opens nearby.
# MAGIC Features: `forecast, open_year, distance, density_new, density_exist, competitor_count_1,
# MAGIC boundary_1-3, year_1-3, closest, closer_stores`.

# COMMAND ----------

# Opening date and subtype for every location, from actual open dates
opened_meta = {row["location_number"]: (row["opened_dt"], row["subtype_id"]) for _, row in loc_all.iterrows()}

impact_ = pd.merge(impact, demo[["location_num", "density_classification_1"]], left_on="store1", right_on="location_num", how="left")
impact_ = impact_.rename(columns={"density_classification_1": "density_new"})
impact_ = pd.merge(impact_, demo, left_on="store2", right_on="location_num", how="left")
impact_ = impact_.rename(columns={"density_classification_1": "density_exist"})
impact_ = impact_.drop(["location_num_x", "location_num_y"], axis=1)
for c in ["density_new", "density_exist"]:
    impact_ = impact_.fillna(value={c: 0})
    impact_[c] = impact_[c].apply(lambda x: str(int(x)))

output = []
for _, row in impact_.iterrows():
    impacted_store, new_store = row.store2, row.store1
    new_store_meta = opened_meta[new_store]
    if new_store_meta[0] is not None:
        closest, closer = 1000, 0
        boundary, years = [0, 0, 0], [0, 0, 0]
        if impacted_store in distance_matrix:
            for nearby in distance_matrix[impacted_store]:
                if nearby in opened_meta and opened_meta[nearby][0] is not None and opened_meta[nearby][0] < new_store_meta[0]:
                    if opened_meta[nearby][1] in ("FSU", "DTO"):
                        closest = min(closest, distance_matrix[impacted_store][nearby][0])
                        if distance_matrix[impacted_store][nearby][0] < row["distance"]:
                            closer += 1
                        if distance_matrix[impacted_store][nearby][1] // DRIVE_TIME_BUCKET_MIN < 3:
                            boundary[int(distance_matrix[impacted_store][nearby][1] // DRIVE_TIME_BUCKET_MIN)] += 1
                        diff = int((new_store_meta[0] - opened_meta[nearby][0]).days // 365)
                        if diff < 3:
                            years[diff] += 1
            closest = [0, 1][row["distance"] < closest]
        else:
            print("not in distance matrix:", impacted_store)
            closest = 1
        output += [[row.store1, row.store2, row.forecast, row.opened_dt[:4], row.distance, row.density_new, row.density_exist, row.competitor_count_1]
                   + boundary + years + [closest, closer] + [row.error]]

IMPACT_FEATURES = ["forecast", "open_year", "distance", "density_new", "density_exist", "competitor_count_1",
                   "boundary_1", "boundary_2", "boundary_3", "year_1", "year_2", "year_3", "closest", "closer_stores"]
output_df = pd.DataFrame(output, columns=["store1", "store2"] + IMPACT_FEATURES + ["error"])
output_df = output_df.replace([np.inf, -np.inf], np.nan)
# drop extremes
output_df = output_df[(output_df["error"] >= IMPACT_ERROR_RANGE[0]) & (output_df["error"] < IMPACT_ERROR_RANGE[1])]
output_df = output_df.dropna()
write_run_table(connector, output_df, "mri_impact_training", "feature_run_id", FEATURE_RUN_ID, SCHEMA_FQN)

# COMMAND ----------

X, y = output_df[IMPACT_FEATURES].values, output_df["error"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=IMPACT_TEST_SIZE, random_state=IMPACT_SPLIT_SEED)

impact_model = CatBoostRegressor(**IMPACT_MODEL_PARAMS)
impact_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)
with tempfile.TemporaryDirectory() as d:
    model_file = os.path.join(d, ARTIFACTS["impact_model"])
    impact_model.save_model(model_file)
    mlflow.log_artifact(model_file, artifact_path="state")

predictions = impact_model.predict(X_test)
mlflow.log_params({f"impact_{k}": v for k, v in IMPACT_MODEL_PARAMS.items()})
mlflow.log_metrics({
    "impact_r2": r2_score(y_test, predictions),
    "impact_mse": mean_squared_error(y_test, predictions),
    "impact_mape": mean_absolute_percentage_error(y_test, predictions),
})

# Held-out and training targets, for the fit plots in monitoring
impact_eval = pd.concat([
    pd.DataFrame({"dataset_split": "test", "target": y_test, "prediction": predictions}),
    pd.DataFrame({"dataset_split": "train", "target": y_train, "prediction": np.nan}),
])
write_run_table(connector, impact_eval, "mri_impact_test_predictions", "feature_run_id", FEATURE_RUN_ID, SCHEMA_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Store sales
# MAGIC Annualized sales per store from the last year of history. Pipeline stores use their forecast.

# COMMAND ----------

sales = monthly_sales.copy()
sales["year"] = sales["month"].apply(lambda x: str(x)[:4])
sales = sales.groupby(["location_num", "year"]).agg(annual_sales=("total_net_sales", "sum"), count=("month", "count")).reset_index()

# Rows are in year order, so the last year seen wins
sales_by_store = {}
for _, row in sales.iterrows():
    sales_by_store[row.location_num] = round(row.annual_sales / int(row["count"]) * 12, 2)
for _, row in pipeline_sales.iterrows():
    sales_by_store[row.location_num] = round(row.forecast_annual_sales, 2)

log_pickle(sales_by_store, "sales_by_store")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Candidate sales forecast and build-out cost
# MAGIC Sales range (lower, median, upper quantile models) for sites forecast to open in `FORECAST_YEAR`.

# COMMAND ----------

categorical = [
    "dma", "density_classification_1", "month_of_opening", "ssam_flag", "no_dt_flag", "subtype",
]
integer = [
    "total_population_1", "worker_population_1", "daytime_population_1", "total_households_1",
    "competitor_count_1",
    "boundary_5", "boundary_10", "boundary_15", "boundary_20", "boundary_25", "boundary_30",
    "months_6m", "months_12m", "months_18m", "months_24m", "months_30m", "months_36m",
]
numerical = [
    "avg_population_growth_1", "total_households_per_sq_mile_1", "med_household_income_1",
    "med_disposable_income_1", "avg_disposable_income_1", "annual_budget_expenditure_1",
    "annual_budget_expenditure_index_1", "med_home_value_1", "avg_home_value_1", "food_away_from_home_1",
    "bigbox_1", "bigbox_2", "bigbox_3", "competitor_1", "competitor_2", "competitor_3",
    "highway_5", "highway_10", "highway_5min", "highway_10min", "close_highway_dist",
    "avg_monthly_sales_closest_1year", "growth_rate_closest", "avg_monthly_sales_mean_1year",
    "growth_rate_mean", "avg_monthly_sales_25th_1year", "growth_rate_25th",
    "avg_monthly_sales_75th_1year", "growth_rate_75th",
]
traffic = [
    "median_average_daily_traffic_1", "max_average_daily_traffic_1",
    "median_average_daily_traffic_2", "max_average_daily_traffic_2",
    "median_average_daily_traffic_3", "max_average_daily_traffic_3",
]

model_lower = CatBoostRegressor()
model_lower.load_model(INPUT_PATHS["sales_model_lower"])
model_median = CatBoostRegressor()
model_median.load_model(INPUT_PATHS["sales_model_median"])
model_upper = CatBoostRegressor()
model_upper.load_model(INPUT_PATHS["sales_model_upper"])

features_ = features.reset_index(drop=True)
features_[integer] = features_[integer].astype(int)
features_[categorical] = features_[categorical].astype(str)
features_[numerical] = features_[numerical].astype(float)
features_[traffic] = features_[traffic].astype(float)
feature_list = categorical + numerical + traffic
x = features_[feature_list].values.reshape(-1, len(feature_list))

sales_range = {}  # location -> annual (lower, median, upper)
for index, row in features_.iterrows():
    if int(str(row.opened_dt)[:4]) == FORECAST_YEAR:
        low, mid, high = model_lower.predict(x[index]), model_median.predict(x[index]), model_upper.predict(x[index])
        sales_range[row.location_num] = (np.exp(low) * 12, np.exp(mid) * 12, np.exp(high) * 12)

log_pickle(BUILD_OUT_COST_BY_REGION, "build_out_cost_by_region")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Candidate sites
# MAGIC Integrated strategy sites with a `FORECAST_YEAR` sales range and a region, excluding active deals.

# COMMAND ----------

# Remove sites that sit on an active deal (already chosen)
active_deals = []
for _, row in sites[["location_number", "NAME", "LATITUDE", "LONGITUDE"]].iterrows():
    for _, row_ in active_deal_dots[["name", "latitude", "longitude"]].iterrows():
        # haversine_distance returns km, so this radius is km
        if haversine_distance(row.LATITUDE, row.LONGITUDE, row_.latitude, row_.longitude) < ACTIVE_DEAL_RADIUS:
            active_deals += [row.location_number]
            break

# Opening date per location; future stores use the projected date
open_meta_all = {}
for _, row in loc_all.iterrows():
    opening_date = row["projected_open_dt"] if row["status"] == "future" else row["opened_dt"]
    open_meta_all[row["location_number"]] = (opening_date, row["subtype_id"])

sites = pd.merge(sites, demo_by_loc[["location_number", "density_classification_1", "competitor_count_1"]], on="location_number", how="left")

candidate_open_dt = datetime.strptime(CANDIDATE_OPEN_DATE, "%Y-%m-%d").date()
boundaries = {}
for _, row in sites.iterrows():
    store = row.GlobalID.replace("{", "").replace("}", "")
    boundaries[store] = compute_boundaries(store, candidate_open_dt, open_meta_all, fsu_dto_only=False)

candidates = {}
no_region = 0
for _, row in sites.iterrows():
    if row.location_number in sales_range:
        if row.REGION is not None and str(row.REGION) != "nan":
            rec_prob = row.rec_probability  # {'25-50', '50-75', '<25', '>75'}
            rec_year = row.rec_year         # 2025, 2026, 2027, nan
            if row.location_number not in active_deals:
                candidates[row.location_number] = {
                    "site_id": row.GlobalID.replace("{", "").replace("}", ""),
                    "region": row.REGION,
                    "store_type": row.SSAM,
                    "prod_bucket": rec_prob if str(rec_prob) != "nan" else "",
                    "prod_year": int(rec_year) + REC_YEAR_OFFSET if str(rec_year) != "nan" else NO_REC_YEAR,
                    "impact_pct": 0.0,
                    "density_classification": row.density_classification_1,
                    "competitor_count": row.competitor_count_1,
                    "boundary": boundaries[row["location_number"]]["boundary"],
                    "years": boundaries[row["location_number"]]["years"],
                    "distances": boundaries[row["location_number"]]["distances"],
                    "nearby_stores": boundaries[row["location_number"]]["nearby_stores"],
                    "sales_forecast": sorted(sales_range[row.location_number]),
                }
        else:
            no_region += 1

log_pickle(candidates, "candidates")
mlflow.log_metrics({"active_deals_excluded": len(active_deals), "candidates": len(candidates), "candidates_without_region": no_region})

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. Existing stores
# MAGIC FSU/DTO network (open, remodel, closed, future) with sales, demographics, and nearby-store features.

# COMMAND ----------

loc_network = pd.merge(loc_network, demo_by_loc[["location_number", "density_classification_1", "competitor_count_1"]], on="location_number", how="left")

open_meta_network = {}
for _, row in loc_network.iterrows():
    opening_date = row["projected_open_dt"] if row["status"] == "future" else row["opened_dt"]
    open_meta_network[row["location_number"]] = (opening_date, row["subtype_id"])

default_open_dt = datetime.strptime(DEFAULT_PIPELINE_OPEN_DATE, "%Y-%m-%d").date()
boundaries = {}
for _, row in loc_network.iterrows():
    store = row.location_number
    store_open_dt = open_meta_network[store][0]
    if store_open_dt is None:
        store_open_dt = default_open_dt
    boundaries[store] = compute_boundaries(store, store_open_dt, open_meta_network, fsu_dto_only=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7. Operator ownership
# MAGIC `ownership_map[operator_id] = [set of stores, years since the operator last received a store]`.
# MAGIC Pipeline stores without an operator are matched by name to the operator selection data and the
# MAGIC pipeline workbook. Anything left gets a new operator.

# COMMAND ----------

existing_stores, ownership_map = {}, {}
store_open_date = {}
loc_included = set()
operator_id_name_mapping = {}
for index, row in loc_network.sort_values("opened_dt").iterrows():
    owner = row["operator_id"]
    if row.opened_dt is None:  # pipeline stores
        store_open_date[row.location_number] = (
            str(row.projected_open_dt)[:4]
            if str(row.projected_open_dt) != "nan" and row.projected_open_dt is not None
            else DEFAULT_PIPELINE_OPEN_YEAR
        )
    existing_stores[row["location_number"]] = {
        "site_id": row["location_number"],
        "owner_id": owner,
        "opened_dt": row["projected_open_dt"] if row["status"] == "future" else row["opened_dt"],
        "sales": sales_by_store.get(row["location_number"], DEFAULT_STORE_SALES),
        "density_classification": row.density_classification_1 if str(row.density_classification_1) != "nan" else DEFAULT_DENSITY_CLASS,
        "competitor_count": row.competitor_count_1 if str(row.competitor_count_1) != "nan" else DEFAULT_COMPETITOR_COUNT,
        "boundary": boundaries[row["location_number"]]["boundary"],
        "years": boundaries[row["location_number"]]["years"],
        "distances": boundaries[row["location_number"]]["distances"],
        "nearby_stores": boundaries[row["location_number"]]["nearby_stores"],
    }
    if str(row["operator_id"]) != "nan":
        if row["operator_id"] not in ownership_map:
            ownership_map[row["operator_id"]] = [set(), 2]
        if row["location_number"] in store_open_date:
            ownership_map[row["operator_id"]][1] = min(2, OWNERSHIP_REFERENCE_YEAR - int(store_open_date[row["location_number"]]))
        operator_id_name_mapping[row["operator"]] = row["operator_id"]
        ownership_map[row["operator_id"]][0].add(row["location_number"])
        loc_included.add(row["location_number"])

operator_matches = pd.merge(
    loc_network[["location_number", "operator_id", "operator"]],
    operator[["location_number", "person_system_id", "selected_candidate_name"]],
    on="location_number", how="left",
)

# COMMAND ----------

operator_next = max(ownership_map.keys()) + 1
print("next operator id:", operator_next)

# Pipeline stores with a selected candidate in the operator selection data
for _, row in operator_matches[(operator_matches["operator_id"].isna()) & (operator_matches["person_system_id"].notna())].iterrows():
    name = row.selected_candidate_name.split()
    not_found = True
    for operator_name, operator_id in operator_id_name_mapping.items():
        if operator_name is not None and name[0] in operator_name and name[1] in operator_name:
            print(operator_name, name, row.location_number)
            if operator_id not in ownership_map:
                ownership_map[operator_id] = [set(), 2]
            ownership_map[operator_id][0].add(row.location_number)
            if row["location_number"] in store_open_date:
                ownership_map[operator_id][1] = min(2, OWNERSHIP_REFERENCE_YEAR - int(store_open_date[row["location_number"]]))
            loc_included.add(row.location_number)
            not_found = False
            existing_stores[row.location_number]["owner_id"] = operator_id
            break
    if not_found:
        ownership_map[operator_next] = [set(), -1]
        ownership_map[operator_next][0].add(row.location_number)
        if row["location_number"] in store_open_date:
            # NOTE: updates the last matched operator_id, not operator_next (kept as in the original)
            ownership_map[operator_id][1] = min(2, OWNERSHIP_REFERENCE_YEAR - int(store_open_date[row["location_number"]]))
        operator_next += 1

# Pipeline stores with an operator named in the pipeline workbook
for _, row in pipelines_df[pipelines_df["operator_selection"].notna()].iterrows():
    name = row["operator_selection"].split()
    not_found = True
    if row["location_number"] in existing_stores:
        for operator_name, operator_id in operator_id_name_mapping.items():
            if operator_name is not None and name[0] in operator_name and len(name) >= 2 and name[1] in operator_name:
                print(operator_name, name, row.location_number)
                if operator_id not in ownership_map:
                    ownership_map[operator_id] = [set(), 2]
                ownership_map[operator_id][0].add(row.location_number)
                loc_included.add(row.location_number)
                if row["location_number"] in store_open_date:
                    ownership_map[operator_id][1] = min(2, OWNERSHIP_REFERENCE_YEAR - int(store_open_date[row["location_number"]]))
                not_found = False
                existing_stores[row.location_number]["owner_id"] = operator_id
                break
        if not_found:
            ownership_map[operator_next] = [set(), -1]
            ownership_map[operator_next][0].add(row.location_number)
            if row["location_number"] in store_open_date:
                # NOTE: updates the last matched operator_id, not operator_next (kept as in the original)
                ownership_map[operator_id][1] = min(2, OWNERSHIP_REFERENCE_YEAR - int(store_open_date[row["location_number"]]))
            operator_next += 1

# Everything else gets a new operator
for index, row in loc_network.sort_values("opened_dt").iterrows():
    if row.location_number not in loc_included:
        # NOTE: the store is not added to the new operator's store set (kept as in the original)
        ownership_map[operator_next] = [set(), -1]
        if row["location_number"] in store_open_date:
            ownership_map[operator_next][1] = min(2, OWNERSHIP_REFERENCE_YEAR - int(store_open_date[row["location_number"]]))
        existing_stores[row.location_number]["owner_id"] = operator_next
        print(row.location_number, row.projected_open_dt, row.operator)
        operator_next += 1

log_pickle(ownership_map, "ownership_map")
log_pickle(existing_stores, "existing_stores")
mlflow.log_metrics({"existing_stores": len(existing_stores), "operators": len(ownership_map)})

# COMMAND ----------

mlflow.end_run()
dbutils.jobs.taskValues.set(key="feature_run_id", value=FEATURE_RUN_ID)
