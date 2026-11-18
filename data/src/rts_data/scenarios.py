"""Demo scenario candidates (Spark): test-split sessions with a planted brand intent, cut before their last search.
Checked by running `demo_scenarios` (no Spark in local tests); `rts_data.demo.pick_scenarios` chooses among them."""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from rts_shared.config import table


def session_starts(spark, cfg: dict):
    """(before, candidates): `before` is each session's events before its last search, with the session's intent;
    `candidates` one row per session that qualifies: 3+ distinct searches before the last with no query repeated back
    to back, 3+ clicks and one on the intent brand, and a last search that is new to the session and in the intent
    category (so the demo shows L2 generalising from intent, not replaying a repeat)."""
    sessions = (spark.table(table(cfg, "sim_sessions"))
                .where("split = 'test' AND intent_brand IS NOT NULL")
                .select("session_id", "user_id", "intent_category", "intent_brand", "intent_price_target"))
    # user_id comes from sim_sessions (the replay sets it on every event).
    events = (spark.table(table(cfg, "session_events")).drop("user_id")
              .join(sessions.select("session_id"), "session_id"))
    last_search = (events.where("event_type = 'search'").groupBy("session_id")
                   .agg(F.max_by("query", "event_time").alias("next_query"),
                        F.max_by("category", "event_time").alias("next_category"), F.max("event_time").alias("cut")))
    before = (events.join(last_search, "session_id").where(F.col("event_time") < F.col("cut"))
              .join(sessions, "session_id"))
    searches = (before.where("event_type = 'search'")
                .withColumn("prev_query", F.lag("query").over(Window.partitionBy("session_id").orderBy("event_time"))))
    search_stats = (searches.groupBy("session_id")
                    .agg(F.count_distinct("query").alias("distinct_queries"),
                         F.count_if(F.col("query") == F.col("prev_query")).alias("repeats"),
                         F.count_if(F.col("query") == F.col("next_query")).alias("next_seen")))
    candidates = (before.groupBy("session_id", "user_id", "intent_category", "intent_brand", "intent_price_target",
                                 "next_query", "next_category")
                  .agg(F.count_if(F.col("event_type") == "search").alias("searches"),
                       F.count_if(F.col("event_type") == "click").alias("clicks"),
                       F.count_if((F.col("event_type") == "click")
                                  & (F.lower("brand") == F.lower("intent_brand"))).alias("brand_clicks"),
                       F.count_if((F.col("event_type") == "click")
                                  & (F.col("category") == F.col("intent_category"))).alias("category_clicks"))
                  .join(search_stats, "session_id")
                  .where("distinct_queries >= 3 AND repeats = 0 AND next_seen = 0 AND next_category = intent_category "
                         "AND clicks >= 3 AND brand_clicks >= 1"))
    return before, candidates
