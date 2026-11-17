"""QAC L2 training spine (Spark): sampled searches x the prefixes typed x the L1 completions, labelled by whether
the completion is the query submitted. Checked by running `qac_l2_lgbm` (no Spark in local tests)."""

from __future__ import annotations

from pyspark.sql import functions as F

from rts_shared.config import table
from rts_shared.retrieval import QAC_L2_CANDIDATE_COLUMNS, normalize


def searches(spark, cfg: dict):
    """One row per search with a non-empty submitted query, normalized like the L1 table's keys."""
    impressions = spark.table(table(cfg, "qac_impressions"))
    # Normalized on the driver with the same function as the L1 table, so prefixes match its keys.
    queries = [r["query"] for r in impressions.select("query").distinct().collect()]
    submitted = spark.createDataFrame([(q, normalize(q)) for q in queries], "query string, submitted string")
    return (impressions.join(submitted, "query").where("submitted != ''")
            .select("search_id", "session_id", "user_id", "event_time", "split", "submitted"))


def sample(searches, split: str, n: int):
    """A reproducible sample of `n` searches from one split."""
    return searches.where(F.col("split") == split).orderBy(F.sha2(F.col("search_id"), 256)).limit(n)


def typed(searches, prefix_lens):
    """One row per (search, prefix length) the submitted query is long enough for, with the `prefix` typed."""
    return (searches.withColumn("n", F.explode(F.array(*[F.lit(n) for n in prefix_lens])))
            .where(F.length("submitted") >= F.col("n")).withColumn("prefix", F.expr("substring(submitted, 1, n)")))


def spine(spark, cfg: dict, searches, prefix_lens, candidates: int):
    """Typed prefixes x their L1 top `candidates` completions: one ranking group per (search, prefix length), the
    completion equal to the submitted query labelled 1."""
    l1 = (spark.table(table(cfg, "qac_prefix_completions")).where(F.col("rank") <= candidates)
          .select("prefix", F.col("rank").alias("l1_rank"), "completion",
                  *[c for c in QAC_L2_CANDIDATE_COLUMNS if c != "l1_rank"]))
    return (typed(searches, prefix_lens).join(l1, "prefix")
            .select(F.concat_ws(":", "search_id", F.col("n").cast("string")).alias("group_id"), "search_id",
                    "session_id", "user_id", "event_time", *QAC_L2_CANDIDATE_COLUMNS,
                    F.col("user_id").isNull().cast("int").alias("is_guest"),
                    (F.col("completion") == F.col("submitted")).cast("int").alias("label")))
