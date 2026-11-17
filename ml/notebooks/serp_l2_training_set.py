# Databricks notebook source
# DBTITLE 1,SERP L2 training set: Feature View lookups -> model matrix, one row per search
# MAGIC %md
# MAGIC # SERP L2 training set: Feature View lookups → model matrix, one row per search
# MAGIC
# MAGIC 1. **Spine:** each sampled search (`serp_l2_searches`) × its L1 top 200 (`serp_l1_candidates`), with the
# MAGIC    product's price/brand/category from `products_search` (the index source) and a graded label from
# MAGIC    `serp_impressions`: cart 2, click 1, else 0. The last `models.serp_l2.valid_days` of train become `valid`.
# MAGIC 2. **`create_training_set(features=...)`:** session features from the Kafka `StreamSource`, user and item features
# MAGIC    from Delta, point-in-time at the search, plus the candidate fields as `RequestSource` features
# MAGIC    (`SERP_L2_REQUEST_COLUMNS`). `l1_rank`, `brand`, `category` and `is_guest` reuse the features QAC registered
# MAGIC    (a RequestSource feature only requires its own column); `l1_score` and `price` are new.
# MAGIC 3. **Model matrix** (`rts_ml.serp_l2.model_matrix`, the same function the logged model calls), packed per search
# MAGIC    in L1 order and padded to `models.serp_l2.candidates`: `x` (candidates × features, flattened), `label`, `mask`.
# MAGIC    Written to `serp_l2_matrix` with the split and the evaluation slices (first vs 2nd+ search, guest, intent).
# MAGIC
# MAGIC Searches are processed in hash buckets so no pandas frame holds all ~3M candidate rows. The `TrainingSet` itself
# MAGIC is rebuilt from `serp_l2_spine` by `serp_l2_dcnv2` with the same `rts_ml.l2_training.training_features`, for
# MAGIC `fe.log_model`.
# MAGIC
# MAGIC Checks: every search has exactly k candidates; session, user and item features joined (a missing model input
# MAGIC fails `model_matrix` on the first bucket); the matrix is finite; the share of searches whose engaged product is
# MAGIC among the candidates (the L2 ceiling).
# MAGIC
# MAGIC **Compute:** Serverless, environment version 4+.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install "databricks-feature-engineering>=0.18.2" pyyaml -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load config
import json
import sys

import numpy as np
from pyspark.sql import functions as F

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
ROOT = "/Workspace" + "/".join(_nb_path.split("/")[:-3])  # this notebook is <ROOT>/ml/notebooks/<name>
sys.path[:0] = [f"{ROOT}/shared/src", f"{ROOT}/ml/src"]
from rts_ml.l2_training import search_slices, training_features
from rts_ml.serp_l2 import MODEL_COLUMNS, pack_searches
from rts_ml.serp_l2_training import spine as build_spine
from rts_shared.config import full_name, load_config
from rts_shared.features import feature_keys
from rts_shared.retrieval import SERP_L2_REQUEST_COLUMNS

from databricks.feature_engineering import FeatureEngineeringClient

dbutils.widgets.text("config_path", f"{ROOT}/shared/configs/dev.yml")  # jobs pass configs/<bundle target>.yml
CFG = load_config(dbutils.widgets.get("config_path"))
L2 = CFG["models"]["serp_l2"]
N = L2["candidates"]
SPINE, MATRIX = full_name(CFG, "serp_l2_spine"), full_name(CFG, "serp_l2_matrix")
BUCKETS = 8
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Spine: searches x L1 top k, product fields, graded labels (rts_ml.serp_l2_training)
build_spine(spark, CFG, N, L2["valid_days"], BUCKETS).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SPINE)
spine = spark.table(SPINE)

per_search = spine.groupBy("search_id").agg(F.count("*").alias("n"), F.max("label").alias("best"))
shape = per_search.agg(F.min("n").alias("min_n"), F.max("n").alias("max_n"),
                       F.avg((F.col("best") > 0).cast("double")).alias("engaged_in_candidates")).first().asDict()
gaps = spine.agg(F.count_if(F.col("in_products").isNull()).alias("missing_product"),
                 F.count_if(F.col("price").isNull()).alias("null_price"),
                 F.min(F.when(F.col("split") == "valid", F.col("event_time"))).alias("valid_from")).first().asDict()
# A few indexed products have no category, so no synthetic price; the model reads that as missing, as at serving.
print(shape, gaps)
assert shape["min_n"] == shape["max_n"] == N, f"searches should have exactly {N} candidates: {shape}"
assert gaps["missing_product"] == 0, f"{gaps['missing_product']} candidates missing from products_search"

# COMMAND ----------

# DBTITLE 1,Features: lookups (session, user, item) + candidate RequestSource features
FEATURES = training_features(fe, CFG, feature_keys(CFG), spine, SERP_L2_REQUEST_COLUMNS)
# Extra spine columns (split, bucket, keys) pass through to load_df.
joined = fe.create_training_set(df=spine, features=FEATURES, label="label", exclude_columns=[]).load_df()
slices = (spine.select("search_id", "split").distinct().join(search_slices(spark, CFG), "search_id")
          .toPandas().set_index("search_id"))

# COMMAND ----------

# DBTITLE 1,Model matrix, packed per search (bucket by bucket)
SCHEMA = ("search_id string, split string, search string, guest string, intent string, "
          "x array<float>, label array<float>, mask array<float>")
spark.sql(f"DROP TABLE IF EXISTS {MATRIX}")
coverage, finite = [], True
for b in range(BUCKETS):
    pdf = joined.where(F.col("bucket") == b).toPandas()
    # Real history, not just non-null: counts are 0 (not null) without history.
    guest = pdf["is_guest"] == 1
    coverage.append({"rows": len(pdf), "earlier_search": int((pdf["sess_n_searches_30m"].fillna(0) > 0).sum()),
                     "session_engaged": int(pdf["sess_last_category"].notna().sum()),
                     "user_history": int((pdf["user_n_clicks_30d"].fillna(0) > 0).sum()),
                     "guest_user_history": int((guest & (pdf["user_n_clicks_30d"].fillna(0) > 0)).sum()),
                     "item_history": int((pdf["item_clicks_7d"].fillna(0) > 0).sum()),
                     "brand": int(pdf["brand"].notna().sum())})
    ids, x, y, mask = pack_searches(pdf, n=N)
    finite &= bool(np.isfinite(x).all())
    s = slices.loc[ids]
    out = [(i, *meta, xi.ravel().tolist(), yi.tolist(), mi.tolist())
           for i, *meta, xi, yi, mi in zip(ids, s["split"], s["search"], s["guest"], s["intent"], x, y, mask)]
    spark.createDataFrame(out, SCHEMA).write.mode("append").saveAsTable(MATRIX)
    print(f"bucket {b}: {len(pdf):,} rows, {len(out):,} searches")
    del pdf, x, out

# COMMAND ----------

# DBTITLE 1,Checks and summary
rows = sum(c["rows"] for c in coverage)
share = {k: round(sum(c[k] for c in coverage) / rows, 4) for k in coverage[0] if k != "rows"}  # of all rows
matrix = spark.table(MATRIX)
counts = {r["split"]: r["count"] for r in matrix.groupBy("split").count().collect()}
summary = {"candidates_per_search": N, "features": len(MODEL_COLUMNS), "spine_rows": rows, "searches_by_split": counts,
           "engaged_product_in_candidates": round(shape["engaged_in_candidates"], 4), "null_price_rows": gaps["null_price"],
           "history_share": share, "valid_from": str(gaps["valid_from"]), "matrix_finite": finite}
print(json.dumps(summary, indent=2))
assert finite, "the model matrix has NaN or inf values"
assert sum(counts.values()) == per_search.count(), "some searches were not packed"
assert set(counts) == {"train", "valid", "test"}, f"splits: {counts}"
assert share["earlier_search"] > 0 and share["session_engaged"] > 0 and share["user_history"] > 0 \
    and share["item_history"] > 0, f"session, user or item features did not join: {share}"
assert share["guest_user_history"] == 0, "guest rows have user history: user features joined on a null key?"
dbutils.notebook.exit(json.dumps(summary))
