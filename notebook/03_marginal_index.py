# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # Market indicator: marginal return index
# MAGIC Runs the greedy and Monte Carlo site selection simulation on the feature run's state. Then it
# MAGIC builds submarket AUV history and projections and scores each submarket-year with the Marginal
# MAGIC Return Index (MRI). Previously "Optimization step 2".
# MAGIC
# MAGIC Reads the `feature_run_id` from the feature engineering notebook.
# MAGIC
# MAGIC **Tables**: `mri_simulation_results`, `mri_simulated_network`, and `mri_submarket_index`
# MAGIC (read by `09_overall_index`).

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

import copy
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from dateutil.relativedelta import relativedelta
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

dbutils.widgets.text("feature_run_id", "")
FEATURE_RUN_ID = dbutils.widgets.get("feature_run_id").strip()
if not FEATURE_RUN_ID:
    raise ValueError("Provide a successful feature_run_id")
if mlflow.get_run(FEATURE_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")
PREPROCESSING_RUN_ID = upstream_param(FEATURE_RUN_ID, "preprocessing_run_id")


def read_preprocessed(table):
    return read_run_table(table, "preprocessing_run_id", PREPROCESSING_RUN_ID, SCHEMA_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the feature run state

# COMMAND ----------

candidates = load_pickle(FEATURE_RUN_ID, "candidates")
existing_stores = load_pickle(FEATURE_RUN_ID, "existing_stores")
ownership_map = load_pickle(FEATURE_RUN_ID, "ownership_map")
distance_matrix = load_pickle(FEATURE_RUN_ID, "distance_matrix")
build_out_cost_by_region = load_pickle(FEATURE_RUN_ID, "build_out_cost_by_region")

impact_model = CatBoostRegressor()
impact_model.load_model(artifact_path(FEATURE_RUN_ID, "impact_model"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulator
# MAGIC Flow per Monte Carlo run:
# MAGIC - Open known pipeline stores in `PIPELINE_YEARS` and record their impact on nearby stores.
# MAGIC - For each simulated year:
# MAGIC   - Sample sales and build cost for every candidate, estimate impact on nearby existing stores, compute IRR.
# MAGIC   - Select the pool using probability buckets and a random buffer.
# MAGIC   - Greedily pick sites under the region cap and the SSAM cap.
# MAGIC   - Assign owners using the operator rules, then update impact history and operator metrics.
# MAGIC   - Add the opened sites to the network.

# COMMAND ----------

# -------------------------
# Sampling
# -------------------------
def sample_sales_for_site(row, year):
    """Triangular sample from the (q10, q50, q90) annual sales forecast, inflated to `year`."""
    q10, q50, q90 = sorted(row)
    growth = (1 + SALES_INFLATION) ** max(0, year - SALES_BASE_YEAR)
    q10, q50, q90 = q10 * growth, q50 * growth, q90 * growth
    q10, q50, q90 = sorted([q10, q50 / SALES_MID_HIGH_DEFLATOR, q90 / SALES_MID_HIGH_DEFLATOR])
    sample = float(np.random.triangular(q10, q50, q90))
    return max(sample, 0.0)


def sample_build_cost_for_site(region, year, build_cost_lookup):
    """Lognormal sample around the region's median build cost, grown to `year`."""
    median, std = build_cost_lookup[region]["last_median"], build_cost_lookup[region]["std"]
    median = median * (1 + BUILD_COST_GROWTH) ** max(0, year - BUILD_COST_BASE_YEAR)
    sigma2 = np.log(1 + (std ** 2) / (median ** 2))
    mu = np.log(median) - 0.5 * sigma2
    sigma = np.sqrt(max(sigma2, 1e-12))
    return float(np.random.lognormal(mu, sigma))


def impact_pct_by_distance(features, exist_model):
    # features: forecast, open_year, distance, density_new, density_exist, competitor_count_1,
    #           boundary_1-3, year_1-3, closest, closer_stores
    return exist_model.predict([features]) + random.uniform(*IMPACT_NOISE_RANGE)


def impact_on_existing(new_id, new_site, ex_id, ex_site, year, distance_matrix, impact_model):
    """Share of `ex_site` sales lost when `new_site` opens (0 when the model predicts a gain)."""
    miles, minutes = distance_matrix[new_id][ex_id]
    closest = [0, 1][minutes < min(new_site["distances"])]
    closer = len([x for x in new_site["distances"] if x < minutes])
    base_pct = impact_pct_by_distance(
        [ex_site["sales"], year, miles, str(int(new_site["density_classification"])),
         str(int(ex_site["density_classification"])), ex_site["competitor_count"]]
        + ex_site["boundary"] + ex_site["years"] + [closest, closer],
        impact_model,
    )[0]
    return -base_pct / 100 if base_pct < 0 else 0.0


def irr_estimate(build_cost, sales, impact_pct):
    return (IRR_COEF["build_cost"] * build_cost + IRR_COEF["sales"] * sales
            + IRR_COEF["impact_pct"] * impact_pct + IRR_COEF["intercept"])

# COMMAND ----------

# -------------------------------
# Owner impact categories
# -------------------------------
def categorize_owner_impacts_single(impact_values):
    """Category for one store's impact history: consecutive, single, or other."""
    if sum(i >= IMPACT_CONSECUTIVE_PCT for i in impact_values) >= 2:
        return "consecutive"
    elif any(i >= IMPACT_SINGLE_PCT for i in impact_values):
        return "single"
    return "other"


def categorize_owner_impacts(owner_map, owner_id, sites, site, existing_impact_history, impact):
    """Category for the owner of `site` over the last IMPACT_WINDOW years including this one.

    Single-unit owners: single_{single, consecutive, combined, other}.
    Multi-unit owners: also multiple_enterprise, when the owner's stores together are hit in 2+ years.
    """
    window = IMPACT_WINDOW
    if len(owner_map[owner_id][0]) > 1:  # multi-unit operator
        t = "multiple_"
        impact_values = existing_impact_history[site][-window:-1] + [impact]
        output = []
        if any(i >= IMPACT_SINGLE_PCT for i in impact_values):
            output += [t + "single"]
        elif sum(i >= IMPACT_CONSECUTIVE_PCT for i in impact_values) >= 2 and len(impact_values) >= 2:
            output += [t + "consecutive"]

        # enterprise
        impact_values = [0, 0, 0]
        for s in sites:
            temp = existing_impact_history[s][-window:]
            for i in range(min(3, len(temp))):
                impact_values[i] = max(existing_impact_history[s][-(i + 1)], impact_values[i])
        if sum(i >= IMPACT_CONSECUTIVE_PCT for i in impact_values) >= 2 and t + "consecutive" not in output and t + "single" not in output:
            output += [t + "enterprise"]
        output = list(set(output))
        if len(output) > 1:
            return t + "combined"
        elif len(output) == 1:
            return output[0]
    else:
        t = "single_"
        output = []
        impact_values = existing_impact_history[site][-window:-1] + [impact]
        if any(i >= IMPACT_SINGLE_PCT for i in impact_values):
            output += [t + "single"]
        elif sum(i >= IMPACT_CONSECUTIVE_PCT for i in impact_values) >= 2 and len(impact_values) >= 2:
            output += [t + "consecutive"]

        if len(output) > 1:
            return t + "combined"
        elif len(output) == 1:
            return output[0]
    return ["single_", "multiple_"][len(owner_map[owner_id][0]) > 1] + "other"


def owner_assignment(owner_stores, driving_minutes=30):
    """Does the owner of an impacted store receive the new store?

    Single-unit operators: probability falls with drive time (SUO_ASSIGNMENT_PROBS).
    Two-unit operators: MUO2_ASSIGNMENT_PROB within MUO2_ASSIGNMENT_MAX_MIN.
    Larger operators: never.
    """
    prob = random.random()
    if len(owner_stores) == 1:
        for max_minutes, p in SUO_ASSIGNMENT_PROBS:  # first drive-time band that fits
            if driving_minutes <= max_minutes:
                return [prob <= p, "single"]
        return [False, "single"]
    elif len(owner_stores) == 2:
        if driving_minutes <= MUO2_ASSIGNMENT_MAX_MIN and prob <= MUO2_ASSIGNMENT_PROB:
            return [True, "multiple"]
        return [False, "multiple"]
    return [False, "multiple"]

# COMMAND ----------

# -------------------------------
# Pool selection
# -------------------------------
def _bucket_probs(sites):
    site_probs, sids = [], []
    for sid, s in sites.items():
        site_probs += [PROBABILITY_BUCKETS.get(s["prod_bucket"], DEFAULT_BUCKET_PROB)]
        sids += [sid]
    site_probs = np.array(site_probs, dtype=float)
    return site_probs / site_probs.sum(), sids


def _draw_by_probability(sites, n, chosen):
    """Draw n sites without replacement weighted by bucket probability, highest probability first."""
    site_probs, sids = _bucket_probs(sites)
    selected_indices = np.random.choice(len(sites), n, replace=False, p=site_probs)
    sorted_indices = selected_indices[np.argsort(site_probs[selected_indices])[::-1]]
    for i in sorted_indices:
        chosen[sids[i]] = sites[sids[i]]
    return chosen


def select_from_pool(candidates, year, cap, chosen_stores_list, prob_mode=True):
    """Select the pool for `year`. Returns (chosen_sites, leftovers)."""
    year_sites = {sid: s for sid, s in candidates.items() if s.get("prod_year") <= year if sid not in chosen_stores_list}
    leftovers = {sid: s for sid, s in candidates.items() if sid not in chosen_stores_list and sid not in year_sites}

    # enlarge the pool by a random buffer for a more robust selection
    buffer = random.randint(*POOL_BUFFER_PCT_RANGE)
    cap = int(cap * (1 + buffer / 100))

    if len(year_sites) < cap:
        next_year_sites = {sid: s for sid, s in candidates.items() if s.get("prod_year") == year + 1}
        if len(next_year_sites) + len(year_sites) < cap:
            return year_sites | next_year_sites | leftovers, {}
        chosen = year_sites.copy()  # everything from the current year first
        chosen = _draw_by_probability(next_year_sites, cap - len(chosen), chosen)
    else:
        chosen = _draw_by_probability(year_sites, cap, {})

    return chosen, {sid: s for sid, s in candidates.items() if sid not in chosen_stores_list and sid not in chosen}

# COMMAND ----------

# -------------------------------
# Simulation
# -------------------------------
def score_sites(sites, year, existing_sites, existing_impact_history, impact_history, distance_matrix, build_cost_lookup, impact_model):
    """Sample sales and cost, estimate impact, and compute IRR for each site. Fills impact_history[sid]."""
    scored = []
    for sid, site in sites.items():
        sales = sample_sales_for_site(site["sales_forecast"], year)
        build_cost = sample_build_cost_for_site(site["region"], year, build_cost_lookup)
        impact_pct, impact_score, loss_sales = 0.0, 0, 0
        impact_history[sid] = {}
        for id_ in site["nearby_stores"]:
            try:
                if id_ in existing_sites:
                    ex_site = existing_sites[id_]
                    pct = impact_on_existing(sid, site, id_, ex_site, year, distance_matrix, impact_model)
                    impact_pct += pct
                    impact_history[sid][id_] = pct
                    loss_sales += pct * ex_site["sales"]
                    # +1 when the existing store would be single/consecutive impacted
                    site_category = categorize_owner_impacts_single(existing_impact_history[id_][-IMPACT_WINDOW:-1] + [pct])
                    impact_score += [0, 1]["other" not in site_category]
            except Exception:
                continue
        irr = irr_estimate(build_cost, sales, impact_pct)
        scored.append((sid, impact_score, irr, sales, build_cost, impact_pct, loss_sales))
    return scored


def run_single_mc(candidates_original, existing_stores_original, ownership_map_original, yearly_caps, distance_matrix,
                  build_cost_lookup, region_cap=0.25, years=10, impact_model=None, prob_mode=True, run_number=0):
    results = []
    window = IMPACT_WINDOW

    existing_impact_history = defaultdict(list)

    candidates = copy.deepcopy(candidates_original)
    existing_sites = copy.deepcopy(existing_stores_original)
    ownership_map = copy.deepcopy(ownership_map_original)

    # Known pipeline openings
    pipelines_opened = {}
    for year in PIPELINE_YEARS:
        pipelines_opened[year] = yearly_caps.get(year, 0)[0]
        for sid in existing_sites.keys():
            existing_impact_history[sid] += [0]
        cap = yearly_caps.get(year, 0)[0]
        for sid, site in existing_sites.items():
            if site["opened_dt"] is not None and int(str(site["opened_dt"])[:4]) == year and random.random() <= PIPELINE_OPEN_PROB:
                for id_ in site["nearby_stores"]:
                    try:
                        if id_ in existing_sites:
                            pct = impact_on_existing(sid, site, id_, existing_sites[id_], year, distance_matrix, impact_model)
                            existing_impact_history[id_][-1] += pct
                    except Exception:
                        continue
                cap -= 1
                pipelines_opened[year] += 1
                if cap == 0:
                    break
    print(pipelines_opened)

    chosen_stores_list = set()
    current_chosen_ids = set()
    for year in range(SIM_START_YEAR, SIM_START_YEAR + years):
        cap = yearly_caps.get(year, 0)[0]
        if year in pipelines_opened:
            cap -= pipelines_opened[year]
        cap = max(0, cap)
        ssam_cap = yearly_caps.get(year, 0)[1]
        print(year, cap, ssam_cap)

        # one more year since each operator last received a store
        for key in ownership_map.keys():
            ownership_map[key][1] += 1

        # roll impact history and refresh boundary/years features with last year's openings
        for sid in existing_sites.keys():
            existing_impact_history[sid] += [0]
            if year > SIM_START_YEAR:
                nearby_store = existing_sites[sid]["nearby_stores"]
                boundary, years_ = existing_sites[sid]["boundary"], existing_sites[sid]["years"][1:] + [0]
                add_distance, add_nearby = [], []
                for id_ in nearby_store:
                    if id_ in current_chosen_ids:
                        miles, minutes = distance_matrix[sid][id_]
                        if miles // 5 < 3:
                            boundary[int(miles // 5)] += 1
                            years_[-1] += 1
                            add_distance += [miles]
                        if miles // 5 < 4:
                            add_nearby += [id_]
                existing_sites[sid]["boundary"] = boundary
                existing_sites[sid]["years"] = years_
                existing_sites[sid]["distances"] += add_distance
                existing_sites[sid]["nearby_stores"] += add_nearby

        chosen_pool, remainders = select_from_pool(candidates, year, cap, chosen_stores_list, prob_mode=prob_mode)

        impact_history = {}
        scored = score_sites(chosen_pool, year, existing_sites, existing_impact_history, impact_history,
                             distance_matrix, build_cost_lookup, impact_model)
        scored_temp = score_sites(remainders, year, existing_sites, existing_impact_history, impact_history,
                                  distance_matrix, build_cost_lookup, impact_model)

        chosen, current_chosen_ids = [], set()

        # Hypothetical owner categories if every dot in the pool and remainder opened.
        # Counts the dots that would break a growth principle.
        extra_operator_metrics = defaultdict(int)
        for sid, impact_score, irr, sales, build_cost, impact_pct, loss_sales in scored + scored_temp:
            temp_impact_history = existing_impact_history.copy()
            for site in impact_history[sid].keys():
                temp_impact_history[site][-1] += impact_history[sid][site]
            category, assigned = set(), False
            for site in impact_history[sid].keys():
                distance, minutes = distance_matrix[sid][site]
                if minutes <= OWNER_MAX_DRIVE_MIN:
                    site_category = categorize_owner_impacts(ownership_map, existing_sites[site]["owner_id"], impact_history[sid], site, temp_impact_history, impact_history[sid][site])
                    if not assigned:
                        owner = existing_sites[site]["owner_id"]
                        owner_assigned, owner_type = owner_assignment(ownership_map[owner][0], minutes)
                        if owner_assigned:
                            assigned = True
                    if site_category not in ("single_other", "multiple_other"):
                        category.add(site_category)
            for site in impact_history[sid].keys():
                temp_impact_history[site][-1] -= impact_history[sid][site]
            category = ", ".join(sorted(category))
            if category.strip() != "":
                extra_operator_metrics[category] += 1

        # region and SSAM caps
        region_limit = defaultdict(int)
        ssam_limit = defaultdict(int)
        for sid, impact_score, irr, sales, build_cost, impact_pct, loss_sales in scored:
            region, ssam = chosen_pool[sid]["region"], chosen_pool[sid]["store_type"]
            if region_limit[region] <= (int(cap * region_cap) + 1):
                if (ssam == "Yes" and ssam_limit[ssam] <= int(ssam_cap)) or (ssam == "No"):
                    region_limit[region] += 1
                    ssam_limit[ssam] += 1
                    chosen += [(sid, impact_score, irr, sales, build_cost, impact_pct, loss_sales)]
            if len(chosen) == cap:
                break

        impact_stores_list, irrs, sales_, costs_, impacts_, losses_ = set(), [], [], [], [], []
        owner_assignments = defaultdict(str)
        site_category_map = defaultdict(str)
        impact_scores = []  # 1 if impacted; 0 if not
        operator_metrics = {"new": 0, "single": 0, "multiple": 0, "single_consecutive": 0, "single_single": 0, "single_combined": 0, "single_other": 0,
                            "multiple_consecutive": 0, "multiple_single": 0, "multiple_enterprise": 0, "multiple_combined": 0, "multiple_other": 0}
        force_zero_impact = set()  # existing stores whose operator receives the new site: impact is zero

        consecutives = {"single": [], "multiple": []}  # impact percentages for consecutive cases
        multi_operators_case = {"new": 0, "single": 0}  # new/single-unit receivers when a multi-unit operator is impacted
        order = 1
        for sid, impact_score, irr, sales, build_cost, impact_pct, loss_sales in chosen:
            muo_impacted, suo_received = False, False

            chosen_stores_list.add(sid)
            current_chosen_ids.add(sid)
            new_site_assigned = False
            operator_received, operator_type = "", ""
            for site in impact_history[sid].keys():
                distance, minutes = distance_matrix[sid][site]
                if minutes <= OWNER_MAX_DRIVE_MIN:
                    impact_stores_list.add(site)
                    existing_impact_history[site][-1] += impact_history[sid][site]
                    site_category = categorize_owner_impacts(ownership_map, existing_sites[site]["owner_id"], impact_history[sid], site, existing_impact_history, impact_history[sid][site])
                    site_category_map[site] = site_category
                    if not new_site_assigned:
                        owner = existing_sites[site]["owner_id"]
                        owner_assigned, owner_type = owner_assignment(ownership_map[owner][0], minutes)
                        if owner_assigned:
                            operator_received, operator_type = owner, owner_type
                            ownership_map[operator_received][1] = 0
                            ownership_map[operator_received][0].add(sid)
                            if owner_type == "single":
                                operator_metrics["single"] += 1
                                suo_received = True
                            elif owner_type == "multiple":
                                operator_metrics["multiple"] += 1
                            new_site_assigned = True
                            force_zero_impact.add(site)
                        else:
                            if "consecutive" in site_category:
                                consecutives[owner_type] += [sum(existing_impact_history[site][-window:-1] + [impact_history[sid][site]])]
                            if owner_type == "multiple" and site_category != "other":
                                muo_impacted = True
                        owner_assignments[site] = site_category

            if not new_site_assigned:  # new operator
                operator_metrics["new"] += 1
                operator_received = max(ownership_map.keys()) + 1
                ownership_map[operator_received] = [set(), 0]

            if muo_impacted:  # multi-unit operator impacted but did not receive the store
                if not new_site_assigned:
                    multi_operators_case["new"] += 1
                elif suo_received:
                    multi_operators_case["single"] += 1

            existing_sites[sid] = {"site_id": sid, "owner_id": operator_received, "opened_dt": str(year) + f"-01-{str(order).zfill(2)}",
                                   "sales": sales, "density_classification": candidates[sid]["density_classification"],
                                   "competitor_count": candidates[sid]["competitor_count"], "boundary": candidates[sid]["boundary"],
                                   "years": candidates[sid]["years"],
                                   "distances": candidates[sid]["distances"], "nearby_stores": candidates[sid]["nearby_stores"],
                                   "impacts": impact_history[sid]}
            order += 1
            del candidates[sid]
            irrs.append(irr)
            impact_scores.append(impact_score)
            sales_.append(sales)
            costs_.append(build_cost)
            impacts_.append(impact_pct)
            losses_.append(loss_sales)
            if len(irrs) == cap:
                break

        # impacted stores whose operator did not receive a new store
        for sid in impact_stores_list:
            if sid not in force_zero_impact:
                operator_metrics[site_category_map[sid]] += 1

        for sid in force_zero_impact:
            existing_impact_history[sid][-1] = 0

        results.append({
            "run_number": run_number,
            "year": year,
            "avg_IRR": np.mean(irrs) if irrs else 0,
            "std_IRR": np.std(irrs) if irrs else 0,
            "irr_q25": np.percentile(irrs, 25) if irrs else 0,
            "irr_q50": np.percentile(irrs, 50) if irrs else 0,
            "irr_q75": np.percentile(irrs, 75) if irrs else 0,
            "irr": irrs,
            "sales": sales_,
            "build_cost": costs_,
            "impact_pct": impacts_,
            "impact_loss": losses_,
            "region": region_limit,
            "ssam": ssam_limit,
            "avg_impact_pct": np.mean(impact_scores) if impact_scores else 0,
            "impact_score": impact_scores,
            "number_of_impacted_stores": len(impact_stores_list),
            "new_owner_sites": operator_metrics["new"],
            "suo_impacted": sum([y for x, y in operator_metrics.items() if "single" in x]),
            "suo_not_receive_single": operator_metrics["single_single"],
            "suo_not_receive_consecutive": operator_metrics["single_consecutive"],
            "suo_not_receive_combined": operator_metrics["single_combined"],
            "suo_not_receive_other": operator_metrics["single_other"],
            "muo_impacted": sum([y for x, y in operator_metrics.items() if "multiple" in x]),
            "muo_not_receive_single": operator_metrics["multiple_single"],
            "muo_not_receive_consecutive": operator_metrics["multiple_consecutive"],
            "muo_not_receive_enterpise": operator_metrics["multiple_enterprise"],
            "muo_not_receive_combined": operator_metrics["multiple_combined"],
            "muo_not_receive_other": operator_metrics["multiple_other"],
            "total_sites": len(irrs),
            "network_size": len(existing_sites),
            "consecutive": consecutives,
            "muo_analysis": multi_operators_case,
            "new_metric": extra_operator_metrics,
            "pool_size": len(chosen_pool) + len(remainders),
            "openings": existing_sites,
        })

    return results


def run_monte_carlo(candidates, existing_sites, ownership_map, scenarios, distance_matrix, build_cost_lookup,
                    region_cap=0.25, years=10, runs=100, impact_model=None, prob_mode=True):
    all_runs = []
    with ThreadPoolExecutor(max_workers=MC_MAX_WORKERS) as executor:
        futures = {}
        for label, caps in scenarios.items():
            for r in range(runs):
                future = executor.submit(
                    run_single_mc, candidates.copy(), existing_sites.copy(), ownership_map.copy(), caps,
                    distance_matrix, build_cost_lookup, region_cap, years, impact_model, prob_mode, r,
                )
                futures[future] = (label, r)
        for future in as_completed(futures):
            label, r = futures[future]
            all_runs.extend([dict(res, yearly_caps=label) for res in future.result()])
    return pd.DataFrame(all_runs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Submarket AUV and MRI functions

# COMMAND ----------

def annual_sales(store, year, predicted_sales):
    """Annual sales for `store` in `year`. Uses the sales records when available. Otherwise
    projects the last full year or the predicted sales with AUV_INFLATION_RATE."""
    temp_sales_history = {}
    if store in sales_records:
        for month, sale in sales_records[store].items():
            temp_sales_history.setdefault(str(month)[:4], []).append(sale)
        if str(year) in temp_sales_history:
            if len(temp_sales_history[str(year)]) > MIN_MONTHS_FOR_ANNUAL:
                total_sales = sum(temp_sales_history[str(year)]) / len(temp_sales_history[str(year)]) * 12
            else:
                total_sales = 0
        else:
            last_year = max(temp_sales_history.keys())
            if int(last_year) < int(year) - 1 or (store in closed and int(year) > int(str(closed[store])[:4])):
                total_sales = 0
            else:
                sales_ = sum(temp_sales_history[str(last_year)]) / len(temp_sales_history[str(last_year)]) * 12
                total_sales = sales_ * ((1 + AUV_INFLATION_RATE / 100) ** (int(year) - int(last_year)))
    else:
        total_sales = predicted_sales * ((1 + AUV_INFLATION_RATE / 100) ** (int(year) - AUV_PROJECTION_BASE_YEAR))
    return total_sales


def flatten_and_clean(values):
    """List of positive floats (drops None, NaN, and values <= 0)."""
    if values is None:
        return []
    if isinstance(values, float) and np.isnan(values):
        return []
    if not isinstance(values, list):
        values = [values]
    cleaned = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if float(v) > 0:
            cleaned.append(float(v))
    return cleaned


def add_details(df, key):
    """Add AUV summaries, combined AUV lists, maturity, and year-over-year growth rates."""
    df["existing_auv_med"] = df["existing_auv"].apply(lambda x: np.median(flatten_and_clean(x)))
    df["young_store_auv_med"] = df["young_store_auv"].apply(lambda x: np.median(flatten_and_clean(x)))

    df["combined_auv"] = df.apply(
        lambda row: flatten_and_clean(row["existing_auv"]) + flatten_and_clean(row["young_store_auv"])
        + flatten_and_clean(row["new_auv"]) + flatten_and_clean(row["pipeline_auv"]) + flatten_and_clean(row["strategy_auv"]),
        axis=1,
    )
    df["existing_combined_auv"] = df.apply(
        lambda row: flatten_and_clean(row["existing_auv"]) + flatten_and_clean(row["young_store_auv"]), axis=1
    )
    df["combined_auv_avg"] = df["combined_auv"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    df["combined_auv_med"] = df["combined_auv"].apply(lambda x: np.median(x) if len(x) > 0 else np.nan)
    df["new_combined_auv"] = df.apply(
        lambda row: flatten_and_clean(row["new_auv"]) + flatten_and_clean(row["pipeline_auv"]) + flatten_and_clean(row["strategy_auv"]),
        axis=1,
    )

    df["maturity_avg"] = df["maturity"].apply(lambda x: np.mean(flatten_and_clean(x)))

    # sales growth
    df = df.sort_values([key, "year"])
    df["young_store_auv_avg"] = df["young_store_auv"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    df["existing_auv_avg"] = df["existing_combined_auv"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    df["matured_auv_avg"] = df["existing_auv"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    df["new_auv_avg"] = df["new_combined_auv"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

    df["sales_growth_rate_matured"] = df.groupby(key)["matured_auv_avg"].pct_change() * 100
    df["sales_growth_rate_young_store"] = df.groupby(key)["young_store_auv_avg"].pct_change() * 100
    df["sales_growth_rate_existing"] = df.groupby(key)["existing_auv_avg"].pct_change() * 100
    df["sales_growth_rate_new"] = df.groupby(key)["new_auv_avg"].pct_change() * 100
    df["sales_growth_rate_combined"] = df.groupby(key)["combined_auv_avg"].pct_change() * 100

    df["number_of_new_stores"] = df["number_of_new"] + df["number_of_pipeline"] + df["number_of_strategy"]
    return df


def array_stats(arr):
    """Mean, median, std, cv, and count of an array of AUVs."""
    empty = pd.Series({"mean": np.nan, "median": np.nan, "std": np.nan, "cv": np.nan, "count": 0})
    if not isinstance(arr, (list, tuple, np.ndarray)) or len(arr) == 0:
        return empty
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return empty
    mean = arr.mean()
    std = arr.std(ddof=0)
    return pd.Series({"mean": mean, "median": np.median(arr), "std": std, "cv": std / mean if mean > 0 else np.nan, "count": len(arr)})


def maturity_score(arr):
    """Mean store maturity, where a store is fully mature after MRI_MATURITY_YEARS."""
    if not isinstance(arr, (list, tuple, np.ndarray)) or len(arr) == 0:
        return np.nan
    arr = np.asarray(arr, dtype=float)
    return np.nanmean(np.minimum(arr / MRI_MATURITY_YEARS, 1.0))


def safe_divide(a, b):
    return np.where((b > 0) & (~pd.isna(b)), a / b, np.nan)


def score_by_year(df, source_col, output_col, ascending=True):
    """Within-year percentile score (0-100). Neutral 50 when a year has no variation."""
    df[output_col] = np.nan
    for year, idx in df.groupby("year").groups.items():
        s = df.loc[idx, source_col]
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            df.loc[idx, output_col] = 50
            continue
        pct = s.rank(pct=True)
        df.loc[idx, output_col] = pct * 100 if ascending else (1 - pct) * 100
    return df


def build_mri(df):
    """raw_mri: weighted within-year percentile scores (MRI_WEIGHTS).
    final_mri: raw_mri pulled toward the year's system average when market confidence is low."""
    df = df.copy()

    # AUV statistics
    combined = df["combined_auv"].apply(array_stats)
    combined.columns = [f"combined_auv_{c}" for c in combined.columns]
    existing = df["existing_combined_auv"].apply(array_stats)
    existing.columns = [f"existing_combined_auv_{c}" for c in existing.columns]
    df = pd.concat([df, combined, existing], axis=1)

    # Maturity
    df["maturity_confidence"] = df["maturity"].apply(maturity_score)

    # Store counts
    count_cols = ["number_of_existing_stores", "number_of_new", "number_of_pipeline", "number_of_strategy"]
    for c in count_cols:
        df[c] = df[c].fillna(0)
    df["total_store_count"] = df[count_cols].sum(axis=1)
    df["future_store_count"] = df["number_of_new"] + df["number_of_pipeline"] + df["number_of_strategy"]
    df["future_store_share"] = safe_divide(df["future_store_count"], df["total_store_count"])

    # Price-normalized AUV
    df["price_group_median"] = df.groupby(["type", "year"])["combined_auv_mean"].transform("median")
    df["price_normalized_auv"] = safe_divide(df["combined_auv_mean"], df["price_group_median"])

    # Growth
    df["growth_rate_selected"] = (
        MRI_GROWTH_WEIGHTS["matured"] * df["sales_growth_rate_matured"].fillna(df["sales_growth_rate_combined"])
        + MRI_GROWTH_WEIGHTS["young_store"] * df["sales_growth_rate_young_store"].fillna(df["sales_growth_rate_combined"])
    )
    lower, upper = df["growth_rate_selected"].quantile(MRI_OUTLIER_QUANTILES[0]), df["growth_rate_selected"].quantile(MRI_OUTLIER_QUANTILES[1])
    df["growth_rate_selected"] = df["growth_rate_selected"].clip(lower, upper)

    # Market consistency
    df["auv_cv"] = df["combined_auv_cv"]
    lower, upper = df["auv_cv"].quantile(MRI_OUTLIER_QUANTILES[0]), df["auv_cv"].quantile(MRI_OUTLIER_QUANTILES[1])
    df["auv_cv_capped"] = df["auv_cv"].clip(lower, upper)

    # Confidence
    df["store_count_confidence"] = np.minimum(1, np.sqrt(df["total_store_count"] / MRI_FULL_CONFIDENCE_STORES))
    df["market_confidence"] = np.sqrt(df["store_count_confidence"] * df["maturity_confidence"])

    # Scores (within year)
    df = score_by_year(df, "price_normalized_auv", "auv_score")
    df = score_by_year(df, "growth_rate_selected", "growth_score")
    df = score_by_year(df, "future_store_share", "future_store_share_score")
    df = score_by_year(df, "auv_cv_capped", "market_consistency_score", ascending=False)

    score_cols = list(MRI_WEIGHTS.keys())
    df[score_cols] = df[score_cols].fillna(50)

    # Raw and final MRI
    df["raw_mri"] = sum(weight * df[col] for col, weight in MRI_WEIGHTS.items())
    df["system_avg_mri"] = df.groupby("year")["raw_mri"].transform("mean")
    df["market_confidence"] = df["market_confidence"].fillna(0.5).clip(*MRI_CONFIDENCE_CLIP)
    df["final_mri"] = df["market_confidence"] * df["raw_mri"] + (1 - df["market_confidence"]) * df["system_avg_mri"]

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score the submarkets

# COMMAND ----------

if mlflow.active_run():
    mlflow.end_run(status="FAILED")
mlflow_run = mlflow.start_run(tags={"project": PROJECT, "pipeline": PIPELINE, "stage": "index"})
INDEX_RUN_ID = mlflow_run.info.run_id
mlflow.log_params({"feature_run_id": FEATURE_RUN_ID, "preprocessing_run_id": PREPROCESSING_RUN_ID,
                   "sim_years": SIM_YEARS, "mc_runs": MC_RUNS, "region_cap": REGION_CAP})


def save(pdf, table):
    write_run_table(connector, pdf, table, "index_run_id", INDEX_RUN_ID, SCHEMA_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Run the simulation

# COMMAND ----------

df_prob = run_monte_carlo(candidates, existing_stores, ownership_map, SCENARIOS, distance_matrix, build_out_cost_by_region,
                          REGION_CAP, SIM_YEARS, MC_RUNS, impact_model, PROB_MODE)

# Network after the last simulated year (of the last finished run)
last = df_prob.openings.values[-1]

save(df_prob.drop(columns=["openings"]), "mri_simulation_results")
save(pd.DataFrame([
    {"site_id": sid, "owner_id": site["owner_id"], "opened_dt": None if site["opened_dt"] is None else str(site["opened_dt"]),
     "sales": site["sales"], "simulated": "impacts" in site, "impacts": site.get("impacts", {})}
    for sid, site in last.items()
]), "mri_simulated_network")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Sales records
# MAGIC Monthly sales per store: actual history first, ESF where there is no history.

# COMMAND ----------

actual_sales = read_preprocessed("mri_monthly_sales")
actual_sales = actual_sales[actual_sales["month"] >= ACTUAL_SALES_START]
esf_sales = read_preprocessed("mri_esf_sales")

actual_sales = actual_sales[["location_num", "month", "total_net_sales"]].copy()
esf_sales.columns = ["location_num", "month", "total_net_sales"]
actual_sales["source"] = "history"
esf_sales["source"] = "esf"
sales = pd.concat([actual_sales, esf_sales])

sales_records = {}
for _, row in sales.iterrows():
    if row.location_num not in sales_records:
        sales_records[row.location_num] = {}
    if str(row.month)[:10] in sales_records[row.location_num] and row.source == "history":
        sales_records[row.location_num][str(row.month)[:10]] = row.total_net_sales
    elif str(row.month)[:10] not in sales_records[row.location_num]:
        sales_records[row.location_num][str(row.month)[:10]] = row.total_net_sales

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Location lookup
# MAGIC `loc_att[location] = (subtype, submarket, dma)` for network stores and strategy sites.
# MAGIC `closed[location] = closed date`.

# COMMAND ----------

loc_attributes = read_preprocessed("mri_location_attributes")

loc = loc_attributes[loc_attributes["status"].isin(["future", "open", "remodel", "closed"])]
loc = loc[["location_number", "subtype_id", "submarket", "dma"]]
loc = loc[loc["subtype_id"].isin(["FSU", "DTO"])]

closed_loc = loc_attributes[loc_attributes["status"] == "closed"][["location_number", "closed_dt"]]
closed = {row.location_number: row.closed_dt for _, row in closed_loc.iterrows()}

strategy = read_preprocessed("mri_strategy_attributes")

loc_att = {}
for _, row in loc.iterrows():
    loc_att[row.location_number] = (row.subtype_id, row.submarket, row.dma)
for _, row in strategy.iterrows():
    loc_att[row.location_num] = (row.subtype, row.submarket, row.dma)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Apply simulated impacts to sales
# MAGIC Each new opening on or after `CURRENT_DATE` cuts nearby store sales by its impact %
# MAGIC in year 1 and by half that in year 2.

# COMMAND ----------

for key, value in last.items():
    if "impacts" in value and value["impacts"] != {}:
        if value["opened_dt"] >= CURRENT_DATE:
            for nearby, pct in value["impacts"].items():
                if nearby in sales_records:
                    dt = datetime.strptime(value["opened_dt"][:7] + "-01", "%Y-%m-%d")
                    for i in range(12):  # first year
                        new_date = str(dt + relativedelta(months=i))[:10]
                        if new_date in sales_records[nearby]:
                            sales_records[nearby][new_date] = sales_records[nearby][new_date] * (1 - pct)
                    for i in range(12, 24):  # second year
                        new_date = str(dt + relativedelta(months=i))[:10]
                        if new_date in sales_records[nearby]:
                            sales_records[nearby][new_date] = sales_records[nearby][new_date] * (1 - pct / 2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Submarket AUV by year
# MAGIC Each FSU/DTO store goes into a submarket-year bucket as `existing` (opened in an earlier year)
# MAGIC or as `new`, `pipeline`, or `strategy` (opened that year).

# COMMAND ----------

def store_entry(value):
    return (value["site_id"], value["opened_dt"], value["sales"], value["impacts"] if "impacts" in value else {})


submarkets = {}
for year in SUBMARKET_YEARS:
    for key, value in last.items():
        if str(value["opened_dt"]).split("-")[0] <= str(year):
            if key in loc_att and loc_att[key][0] in ("FSU", "DTO"):
                submarket = loc_att[key][1]
                if submarket not in submarkets:
                    submarkets[submarket] = {}
                if year not in submarkets[submarket]:
                    submarkets[submarket][year] = {"existing": [], "new": [], "pipeline": [], "strategy": []}

                if str(value["opened_dt"])[:4] < str(year):
                    bucket = "existing"
                elif str(value["opened_dt"])[:4] < str(CURRENT_DATE):  # new store opened before today
                    bucket = "new"
                elif key.isdigit():
                    bucket = "pipeline"
                else:
                    bucket = "strategy"
                submarkets[submarket][year][bucket] += [store_entry(value)]

submarket_output = []
for submarket in submarkets.keys():
    visited = set()
    for year in sorted(submarkets[submarket].keys()):
        existing_auv, young_store_auv, maturity = [], [], []
        for store, open_dt, sales, impact in submarkets[submarket][year]["existing"]:
            matured_year = int(year) - int(str(open_dt)[:4])
            annual_sales_ = annual_sales(store, str(year), sales)
            if annual_sales_ >= MIN_EXISTING_AUV:
                if matured_year <= YOUNG_STORE_MAX_AGE:
                    young_store_auv += [annual_sales_]
                else:
                    existing_auv += [annual_sales_]
            if annual_sales_ > 0 or store not in visited or year == min(submarkets[submarket].keys()):
                maturity += [int(year) - int(str(open_dt)[:4])]
            visited.add(store)
        new_auv = [annual_sales(store, str(year), sales) for store, open_dt, sales, impact in submarkets[submarket][year]["new"]]
        pipeline_auv = [annual_sales(store, str(year), sales) for store, open_dt, sales, impact in submarkets[submarket][year]["pipeline"]]
        strategy_auv = [annual_sales(store, str(year), sales) for store, open_dt, sales, impact in submarkets[submarket][year]["strategy"]]
        submarket_output += [["submarket", submarket, year, existing_auv, young_store_auv, maturity, len(existing_auv) + len(young_store_auv),
                              new_auv, len(new_auv), pipeline_auv, len(pipeline_auv), strategy_auv, len(strategy_auv)]]

submarket_df = pd.DataFrame(submarket_output, columns=[
    "type", "submarket", "year", "existing_auv", "young_store_auv", "maturity", "number_of_existing_stores",
    "new_auv", "number_of_new", "pipeline_auv", "number_of_pipeline", "strategy_auv", "number_of_strategy",
])
submarket_df = add_details(submarket_df, "submarket")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. Marginal Return Index

# COMMAND ----------

submarket_df_w_index = build_mri(submarket_df)
save(submarket_df_w_index, "mri_submarket_index")

mlflow.log_metrics({
    "submarkets": submarket_df_w_index["submarket"].nunique(),
    "simulated_openings": int(sum(1 for site in last.values() if "impacts" in site)),
})
mlflow.end_run()

# COMMAND ----------

dbutils.jobs.taskValues.set(key="index_run_id", value=INDEX_RUN_ID)
