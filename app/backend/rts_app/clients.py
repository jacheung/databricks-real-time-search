"""The real backend behind `create_app` (rts_app.app.Backend): Lakebase, model serving, AI Search and Kafka.

Runs as the app's service principal. Lakebase OAuth tokens expire hourly, so pooled connections take a fresh token
whenever they (re)connect. Every SQL statement the app runs is here.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from rts_app.service import rows_to_dicts
from rts_shared.config import full_name
from rts_shared.kafka import producer as kafka_producer
from rts_shared.lakebase import branch, features_database
from rts_shared.retrieval import SERP_INDEX_COLUMNS, qac_l2_scores, serp_l2_scores


def _utc(row: dict) -> dict:
    return {k: v.replace(tzinfo=timezone.utc) if isinstance(v, datetime) and v.tzinfo is None else v
            for k, v in row.items()}


class Clients:
    def __init__(self, cfg: dict, w: WorkspaceClient):
        self.cfg, self.w = cfg, w
        self.schema = cfg["unity_catalog"]["schema"]
        self.embed_endpoint = cfg["models"]["query_embedding"]["endpoint"]
        self.qac_endpoint = cfg["models"]["qac_l2"]["endpoint"]
        self.serp_endpoint = cfg["models"]["serp_l2"]["endpoint"]
        self.index = full_name(cfg, cfg["ai_search"]["index"])
        endpoint = next(e for e in w.postgres.list_endpoints(parent=branch(cfg))
                        if "READ_WRITE" in str(e.status.endpoint_type))
        token = {"value": None, "at": 0.0}

        def password() -> str:
            if time.time() - token["at"] > 45 * 60:  # tokens last an hour
                credential = w.postgres.generate_database_credential(endpoint=endpoint.name)
                token.update(value=credential.token, at=time.time())
            return token["value"]

        class TokenConnection(psycopg.Connection):
            @classmethod
            def connect(cls, conninfo: str = "", **kwargs):
                kwargs["password"] = password()
                return super().connect(conninfo, **kwargs)

        def pool(dbname, min_size, max_size):
            # PGUSER is the app's Postgres role, injected by the app's Lakebase resource.
            return ConnectionPool(
                kwargs={"host": endpoint.status.hosts.host, "dbname": dbname, "user": os.environ["PGUSER"],
                        "sslmode": "require", "autocommit": True, "row_factory": dict_row},
                connection_class=TokenConnection, min_size=min_size, max_size=max_size, open=True)

        self.pool = pool(cfg["lakebase"]["database"], 2, 8)
        # Feature Store online tables are in their own database, named after the catalog (rts_shared.lakebase).
        self.features_pool = pool(features_database(cfg), 1, 4)
        self._producer, self._producer_lock = None, threading.Lock()

    def _table(self, key: str) -> str:
        return f'"{self.schema}"."{self.cfg["tables"][key]}"'

    def _query(self, pool, sql, params=()):
        with pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    # Lakebase, app database.
    def prefix_rows(self, prefix):
        return self._query(self.pool, f"SELECT * FROM {self._table('qac_prefix_online')} WHERE prefix = %s "
                                      "ORDER BY rank", (prefix,))

    def prefix_rows_for(self, prefixes):
        return self._query(self.pool, f"SELECT * FROM {self._table('qac_prefix_online')} WHERE prefix = ANY(%s)",
                           (list(prefixes),))

    def load_prefixes(self, n, candidates):
        # Prefixes with a full L1 top `candidates` (popular ones, as typed): every request carries the real payload.
        prefixes = [r["prefix"] for r in self._query(
            self.pool, f"SELECT prefix FROM {self._table('qac_prefix_online')} WHERE length(prefix) <= 6 "
                       "GROUP BY prefix HAVING count(*) >= %s ORDER BY random() LIMIT %s", (candidates, n))]
        by_prefix: dict[str, list[dict]] = {}
        for r in self._query(self.pool, f"SELECT * FROM {self._table('qac_prefix_online')} WHERE prefix = ANY(%s) "
                                        "AND rank <= %s ORDER BY prefix, rank", (prefixes, candidates)):
            by_prefix.setdefault(r["prefix"], []).append(r)
        return list(by_prefix.values())

    def scenarios(self):
        return self._query(self.pool, f"SELECT * FROM {self._table('demo_scenarios_online')} ORDER BY scenario_id")

    def personas(self):
        return self._query(self.pool, f"SELECT * FROM {self._table('demo_personas_online')} ORDER BY persona_id")

    # Lakebase, Feature Store database.
    def online_table_columns(self, prefix, columns):
        # pg_catalog, not information_schema: the latter hides tables this role has no privilege on.
        return self._query(
            self.features_pool,
            "SELECT n.nspname, c.relname, a.attname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_attribute a ON a.attrelid = c.oid "
            "WHERE n.nspname = %s AND c.relkind IN ('r', 'p') AND c.relname LIKE %s "
            "AND a.attname = ANY(%s)", (self.schema, f"{prefix}%", list(columns)))

    def rows(self, table, key, value):
        # `table` and `key` come from online_table_columns and the catalog, never from a request.
        return [_utc(r) for r in self._query(self.features_pool, f"SELECT * FROM {table} WHERE {key} = %s", (value,))]

    def distinct(self, table, column, n):
        return [r[column] for r in self._query(self.features_pool, f"SELECT DISTINCT {column} FROM {table} LIMIT %s",
                                               (n,))]

    # Model serving (route-optimized: the data-plane client uses the app SP's OAuth token) and AI Search.
    def embed(self, text):
        return self.w.serving_endpoints_data_plane.query(name=self.embed_endpoint,
                                                         dataframe_records=[{"text": text}]).predictions[0]

    def query_index(self, **kw):
        r = self.w.vector_search_indexes.query_index(index_name=self.index, columns=list(SERP_INDEX_COLUMNS), **kw)
        return rows_to_dicts([c.name for c in r.manifest.columns], r.result.data_array)

    def qac_l2(self, rows, session_id, user_id):
        return qac_l2_scores(self.w, self.qac_endpoint, rows, session_id, user_id)

    def serp_l2(self, hits, session_id, user_id):
        return serp_l2_scores(self.w, self.serp_endpoint, hits, session_id, user_id)

    def qac_l2_sender(self, workers):
        # Its own client, with a connection pool sized for the load test's workers.
        client = WorkspaceClient(config=Config(max_connection_pools=workers, max_connections_per_pool=workers))
        return lambda records: client.serving_endpoints_data_plane.query(name=self.qac_endpoint,
                                                                          dataframe_records=records)

    # Kafka.
    def producer(self):
        with self._producer_lock:  # requests run on FastAPI's thread pool: build it once
            if self._producer is None:
                self._producer = kafka_producer(self.cfg, self.w)
            return self._producer
