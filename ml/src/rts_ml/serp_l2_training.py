"""SERP L2 training spine (Spark): sampled searches x their L1 top k, with product fields and graded labels.
Checked by running `serp_l2_training_set` (no Spark in local tests)."""

from __future__ import annotations

from datetime import timedelta

from pyspark.sql import functions as F

from rts_shared.config import table
from rts_shared.retrieval import SERP_INDEX_PK


def spine(spark, cfg: dict, candidates: int, valid_days: int, buckets: int):
    """Each search in `serp_l2_searches` x its L1 top `candidates`, with the product's price, brand and category
    (`in_products` flags a match), label cart 2 > click 1 > 0, and a hash `bucket`. The last `valid_days` of train
    become `valid`."""
    searches = spark.table(table(cfg, "serp_l2_searches"))
    cutoff = searches.where("split = 'train'").agg(F.max("event_time")).first()[0] - timedelta(days=valid_days)
    searches = searches.withColumn("split", F.when((F.col("split") == "train") & (F.col("event_time") >= F.lit(cutoff)),
                                                   "valid").otherwise(F.col("split")))
    products = (spark.table(table(cfg, "products_search")).select(SERP_INDEX_PK, "price", "brand", "category")
                .withColumn("in_products", F.lit(True)))
    labels = (spark.table(table(cfg, "serp_impressions")).where("clicked OR carted")
              .select("search_id", SERP_INDEX_PK, F.when(F.col("carted"), 2).otherwise(1).alias("label")))
    l1 = (spark.table(table(cfg, "serp_l1_candidates")).where(F.col("rank") <= candidates)
          .select("query_id", F.col("rank").cast("int").alias("l1_rank"), SERP_INDEX_PK,
                  F.col("score").alias("l1_score")))
    return (searches.join(l1, "query_id").join(products, SERP_INDEX_PK, "left")
            .join(labels, ["search_id", SERP_INDEX_PK], "left")
            .select("search_id", "session_id", "user_id", "event_time", "split", SERP_INDEX_PK, "l1_rank", "l1_score",
                    F.col("price").cast("double").alias("price"), "brand", "category", "in_products",
                    F.col("user_id").isNull().cast("int").alias("is_guest"),
                    F.coalesce("label", F.lit(0)).cast("int").alias("label"),
                    F.abs(F.xxhash64("search_id") % buckets).alias("bucket")))
