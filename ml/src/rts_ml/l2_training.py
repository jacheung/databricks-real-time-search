"""Spark and Feature Engineering code both L2 training sets share. Local tests have no Spark (no JVM), so this is
checked by running the training notebooks."""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from rts_shared.config import full_name, table


def training_features(fe, cfg: dict, lookup_names, request_df, request_columns) -> list:
    """The looked-up features by name, plus one RequestSource feature per request column, named after it (a
    RequestSource feature's value is that column of the request). Definitions persist in UC across runs, and QAC
    and SERP share some, so existing ones are reused."""
    from databricks.feature_engineering.entities import ColumnSelection, RequestSource

    uc = cfg["unity_catalog"]
    source = RequestSource.from_dataframe(dataframe=request_df.select(*request_columns))

    def candidate_feature(column):
        try:
            return fe.create_feature(source=source, function=ColumnSelection(column), catalog_name=uc["catalog"],
                                     schema_name=uc["schema"], name=column)
        except Exception as error:
            if "already exists" not in str(error).lower():
                raise
            return fe.get_feature(full_name=full_name(cfg, column))

    return ([fe.get_feature(full_name=full_name(cfg, n)) for n in lookup_names]
            + [candidate_feature(c) for c in request_columns])


def search_slices(spark, cfg: dict):
    """Per search: `search` (first or 2nd+ in its session), `guest` and `intent` (what the simulator gave the
    session). For the evaluation report, never features."""
    nth = F.row_number().over(Window.partitionBy("session_id").orderBy("event_time"))
    intent = spark.table(table(cfg, "sim_sessions")).select(
        "session_id", F.when(F.col("intent_category").isNull(), "none")
        .when(F.col("intent_brand").isNull(), "category").otherwise("category+brand").alias("intent"))
    return (spark.table(table(cfg, "qac_impressions")).withColumn("search_index", nth)
            .join(intent, "session_id", "left")
            .select("search_id", F.when(F.col("search_index") == 1, "first").otherwise("2nd+").alias("search"),
                    F.when(F.col("user_id").isNull(), "guest").otherwise("logged-in").alias("guest"), "intent"))
