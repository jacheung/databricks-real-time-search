"""The app's routes and what runs behind them: QAC, SERP, scenario Play and its freshness check, the persona check,
benchmarks and the QAC L2 load test. Built by `create_app` over a backend (`Backend`), so tests run it on an
in-memory one; main.py wires the real Lakebase, serving, AI Search and Kafka clients (rts_app.clients).
"""

from __future__ import annotations

import itertools
import logging
import random
import string
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Literal, Protocol

from fastapi import FastAPI, HTTPException, Query

from rts_app.load import LoadRun, summarize
from rts_app.service import (RecentValues, expected_session_features, freshness_breakdown, l2_inputs, landed,
                             latency_stats, online_tables, persona_lift, play_scenario, qac, ranked_top,
                             scenario_timeline, search, session_features, time_calls, trending, user_feature_values,
                             with_timeout)
from rts_shared.features import online_table_prefix
from rts_shared.retrieval import SERP_INDEX_TEXT, qac_l2_records

log = logging.getLogger("rts_app")

QAC_N = 10
SERP_N = 40  # 4 UI pages of 10; L2 reranks every L1 hit either way, so paging needs no extra call
TRENDING_N = 20  # more than two rows' worth; the UI shows two rows
FRESHNESS_POLL_S, FRESHNESS_GIVE_UP_S = 0.01, 15
L2_WORKERS = 8  # per surface: timed-out calls finish on their pool without holding the request
# Persona check: broad prefixes and queries, where the candidates span categories so user history can reorder them.
CHECK_PREFIXES = [*"abcdefghijklmnopqrstuvwxyz", "ba", "be", "bo", "ca", "ch", "co", "ha", "he", "ho", "ki", "la", "li",
                  "ma", "mo", "pa", "pl", "sa", "sh", "sk", "sp", "st", "to", "tr", "wa", "wh"]
CHECK_QUERIES = ["case", "bag", "charger", "brush", "light", "stand", "cleaner", "set", "cover", "kit", "holder",
                 "gift"]
CHECK_MIN_LIFT = 2.0


class Backend(Protocol):
    """Everything the app reads or calls outside its process. Rows are dicts; timestamps are timezone-aware UTC."""

    # Lakebase, app database (synced tables).
    def prefix_rows(self, prefix: str) -> list[dict]: ...             # QAC L1: every rank for one prefix
    def prefix_rows_for(self, prefixes: list[str]) -> list[dict]: ...  # every rank for several prefixes
    def load_prefixes(self, n: int, candidates: int) -> list[list[dict]]: ...  # random short prefixes with a full top
    def scenarios(self) -> list[dict]: ...
    def personas(self) -> list[dict]: ...

    # Lakebase, Feature Store database (online tables).
    def online_table_columns(self, prefix: str, columns: list[str]) -> list[dict]: ...  # pg_catalog rows
    def rows(self, table: str, key: str, value: str) -> list[dict]: ...                # one entity's rows
    def distinct(self, table: str, column: str, n: int) -> list[str]: ...

    # Model serving and AI Search.
    def embed(self, text: str) -> list[float]: ...
    def query_index(self, **kw) -> list[dict]: ...
    def qac_l2(self, rows: list[dict], session_id: str | None, user_id: str | None) -> list[float]: ...
    def serp_l2(self, hits: list[dict], session_id: str | None, user_id: str | None) -> list[float]: ...
    def qac_l2_sender(self, workers: int) -> Callable[[list[dict]], object]: ...  # load test: its own client

    # Kafka.
    def producer(self): ...


def create_app(backend: Backend, cfg: dict) -> FastAPI:
    lb, app_cfg, ai = cfg["lakebase"], cfg["app"], cfg["ai_search"]
    qac_cfg = cfg["models"]["qac_l2"]
    qac_timeout_s, serp_timeout_s = app_cfg["timeouts_ms"]["qac_l2"] / 1000, app_cfg["timeouts_ms"]["serp_l2"] / 1000
    inputs, sessions = l2_inputs(cfg), session_features(cfg)
    # One pool per surface, so a slow SERP call can't queue QAC keystrokes into their timeout.
    qac_pool, serp_pool = ThreadPoolExecutor(L2_WORKERS), ThreadPoolExecutor(L2_WORKERS)
    l2_times = RecentValues()  # full L2 call time per /api/qac request, read by the UI after a timeout
    played = RecentValues(200)  # what each Play sent, so /freshness knows when every table has caught up
    table_cache = {"session": {}, "user": {}}
    load = {"run": None, "phases": [], "info": {}}
    app = FastAPI(title="real-time-search")

    def qac_l2_scores(rows, session_id, user_id, request_id=None):
        # The endpoint looks up session and user features by key; an unknown or missing session scores as no history.
        # On the QAC pool it finishes and logs its full time even after the request has timed out and moved on.
        t = time.perf_counter()
        try:
            return backend.qac_l2(rows, session_id, user_id)
        except Exception:
            log.exception("qac_l2 call failed")
            raise
        finally:
            ms = (time.perf_counter() - t) * 1000
            log.info("qac_l2 call %.1f ms, %d candidates", ms, len(rows))
            if request_id:
                l2_times.record(request_id, round(ms, 2))

    def serp_l2_scores(hits, session_id, user_id):
        # Same lookups as QAC L2 (session and user features by key), plus item features by product_id.
        t = time.perf_counter()
        try:
            return backend.serp_l2(hits, session_id, user_id)
        except Exception:
            log.exception("serp_l2 call failed")
            raise
        finally:
            log.info("serp_l2 call %.1f ms, %d candidates", (time.perf_counter() - t) * 1000, len(hits))

    def tables(group: str, columns: dict[str, str]) -> dict[str, str]:
        """Feature View online tables by kind. Feature Views names them with a generated suffix that changes if the
        features are re-materialized, so each is looked up by a feature only it holds rather than set in config."""
        cache = table_cache[group]
        if not cache:
            prefix = online_table_prefix(group)
            try:  # all kinds or none: a partial cache would stick
                cache.update(online_tables(columns, backend.online_table_columns(prefix, list(columns.values()))))
            except LookupError as error:
                raise HTTPException(503, f"{prefix}: {error}") from error
        return cache

    def session_tables():
        return tables("session", {k: names[0] for k, names in sessions.items()})

    def user_tables():
        return tables("user", {f: f for f in inputs["batch"]})

    def scenarios():
        return {s["scenario_id"]: s for s in backend.scenarios()}

    @app.get("/api/config")
    def config():
        return {"qac_debounce_ms": app_cfg["qac_debounce_ms"], "qac_n": QAC_N, "l2_inputs": inputs,
                "serp_budget_ms": cfg["budgets_ms"]["serp_total_p95"]}

    @app.get("/api/qac")
    def get_qac(q: str = "", session_id: str | None = None, user_id: str | None = None):
        request_id = uuid.uuid4().hex[:12]

        def rerank(rows):
            return with_timeout(qac_pool, lambda: qac_l2_scores(rows, session_id, user_id, request_id), qac_timeout_s)

        return (qac(q, backend.prefix_rows, lb["qac_max_prefix_len"], QAC_N, rerank, qac_cfg["candidates"])
                | {"session_id": session_id, "user_id": user_id, "request_id": request_id})

    @app.get("/api/qac/{request_id}/l2_time")
    def qac_l2_time(request_id: str):
        """The L2 call's full time, including any time after the request timed out; 404 while it's still running."""
        ms = l2_times.get(request_id)
        if ms is None:
            raise HTTPException(404, "L2 call still running or unknown")
        return {"l2_full_ms": ms}

    @app.get("/api/search")
    def get_search(q: str = "", mode: Literal["HYBRID", "ANN"] = "HYBRID", session_id: str | None = None,
                   user_id: str | None = None):
        # Keyword matching covers search_text only, as in serp_l1_ai_search's recall run.
        def rerank(hits):
            return with_timeout(serp_pool, lambda: serp_l2_scores(hits, session_id, user_id), serp_timeout_s)

        return (search(q, backend.embed, backend.query_index, mode, ai["serp_l1_k"], [SERP_INDEX_TEXT], rerank, SERP_N)
                | {"session_id": session_id, "user_id": user_id})

    @app.get("/api/bench/qac_l1")
    def bench_qac_l1(n: int = Query(100, ge=1, le=500)):
        """QAC L1 (Lakebase prefix lookup) latency from inside the app, back-to-back, cycling short and long
        prefixes."""
        prefixes = itertools.cycle(["a", "ca", "cam", "usb", "dog f", "iphone c", "nike shoes", "s", "lap",
                                    "wireless e"])
        return {"lookup": latency_stats(time_calls(lambda: backend.prefix_rows(next(prefixes)), n))}

    @app.get("/api/bench/qac_l2")
    def bench_qac_l2(n: int = Query(100, ge=1, le=500), session_id: str | None = None, user_id: str | None = None,
                     gap_ms: float = Query(0, ge=0, le=2000)):
        """QAC L2 call latency from inside the app, no timeout: 1 vs `candidates` real L1 rows for "ca". Back-to-back
        by default; `gap_ms` idles before each call, like typing pace."""
        rows = backend.prefix_rows("ca")
        sizes = sorted({1, min(qac_cfg["candidates"], len(rows))})
        return {f"candidates_{k}": latency_stats(time_calls(lambda: qac_l2_scores(rows[:k], session_id, user_id), n,
                                                             gap_ms=gap_ms))
                for k in sizes} | {"session_id": session_id, "user_id": user_id, "gap_ms": gap_ms}

    # QAC L2 load test (scripts/load_qac_l2.py drives it): open-loop QPS phases on their own client and thread pool,
    # so the demo's pools aren't involved. Requests are real L1 top-`candidates` rows for random short prefixes, with
    # real user and session keys, so every call does the endpoint's online-store lookups. One run at a time.
    def load_requests(n_prefixes: int = 300, n_keys: int = 500):
        requests = backend.load_prefixes(n_prefixes, qac_cfg["candidates"])
        users = backend.distinct(next(iter(user_tables().values())), "user_id", n_keys)
        keys = backend.distinct(session_tables()["window"], "session_id", n_keys)
        # Sessions age out of the online table; unknown ones score as no history, like a new visitor.
        return requests, users, keys or [f"load-{uuid.uuid4().hex[:10]}" for _ in range(n_keys)]

    @app.post("/api/load/qac_l2")
    def start_load_qac_l2(phases: str = Query(..., description="qps:seconds,qps:seconds,..."),
                          workers: int = Query(128, ge=1, le=512)):
        if load["run"] and not load["run"].done:
            raise HTTPException(409, "a load run is already running")
        plan = [(float(q), float(s)) for q, s in (p.split(":") for p in phases.split(","))]
        if any(q <= 0 or q > 3000 for q, _ in plan) or sum(s for _, s in plan) > 3600:
            raise HTTPException(400, "each phase needs 0 < QPS <= 3000, and a run lasts at most an hour")
        requests, users, keys = load_requests()
        send = backend.qac_l2_sender(workers)

        def call():
            send(qac_l2_records(random.choice(requests), random.choice(keys), random.choice(users)))

        load.update(run=LoadRun(call, plan, workers), phases=plan,
                    info={"started": datetime.now(timezone.utc).isoformat(), "workers": workers,
                          "prefixes": len(requests), "users": len(users), "sessions": len(keys),
                          "candidates_mean": round(sum(map(len, requests)) / len(requests), 1)})
        load["run"].start()
        return load["info"] | {"phases": plan}

    @app.get("/api/load/qac_l2")
    def load_qac_l2(window_s: float = Query(10, gt=0)):
        run = load["run"]
        if run is None:
            raise HTTPException(404, "no load run yet")
        out = load["info"] | {"phases": load["phases"], "done": run.done, "n": len(run.records),
                              "last_error": run.last_error}
        if run.done:
            out["summary"] = summarize(list(run.records), load["phases"], window_s, app_cfg["timeouts_ms"]["qac_l2"])
        return out

    @app.delete("/api/load/qac_l2")
    def stop_load_qac_l2():
        if load["run"]:
            load["run"].stop()
        return {"stopping": bool(load["run"])}

    @app.get("/api/scenarios")
    def list_scenarios():
        return [{k: s[k] for k in ("scenario_id", "label", "next_query")} for s in scenarios().values()]

    @app.get("/api/personas")
    def list_personas():
        """Who the demo signs in as (data's demo_personas): a name, the simulated user and what their history holds."""
        return [{k: p[k] for k in ("persona_id", "name", "user_id", "category", "category_share",
                                   "category_events_30d", "n_clicks_30d", "median_price_30d")}
                for p in backend.personas()]

    @app.get("/api/trending")
    def get_trending():
        """Home-page trending searches, from the QAC prefix table: the top completions for every one-letter prefix."""
        return trending(backend.prefix_rows_for(list(string.ascii_lowercase)), TRENDING_N)

    @app.get("/api/checks/personas")
    def check_personas():
        """Is Sign in worth showing? With no session, scores the same L1 candidates as a guest and as each persona;
        `ok` when the persona's category share in the top results (QAC top 5, SERP top 10) at least doubles. Step 0
        (2026-11-05) measured ~3-4x for QAC. If it fails, check the stale batch features gotcha in data/PLAN.md
        first. Here rather than in a notebook: route-optimized endpoints need an OAuth token, which the app's SP has.
        L2 is called directly, off the demo's pools and without their timeouts: this checks lift, not latency."""
        n = qac_cfg["candidates"]
        pools = [rows[:n] for rows in map(backend.prefix_rows, CHECK_PREFIXES)
                 if len({r["category"] for r in rows[:n]}) >= 2]
        hits = [h for h in (search(q, backend.embed, backend.query_index, "HYBRID", ai["serp_l1_k"],
                                   [SERP_INDEX_TEXT])["l1"] for q in CHECK_QUERIES) if h]

        def tops(user_id):
            return ([ranked_top(rows, qac_l2_scores(rows, None, user_id), 5) for rows in pools],
                    [ranked_top(hs, serp_l2_scores(hs, None, user_id), 10) for hs in hits])

        guest_qac, guest_serp = tops(None)
        report = []
        for p in backend.personas():
            qac_tops, serp_tops = tops(p["user_id"])
            report.append({"name": p["name"], "user_id": p["user_id"], "category": p["category"],
                           "qac_top5": persona_lift(guest_qac, qac_tops, p["category"], CHECK_MIN_LIFT),
                           "serp_top10": persona_lift(guest_serp, serp_tops, p["category"], CHECK_MIN_LIFT)})
        return {"ok": bool(report) and all(r["qac_top5"]["ok"] and r["serp_top10"]["ok"] for r in report),
                "prefixes": len(pools), "queries": len(hits), "min_lift": CHECK_MIN_LIFT, "personas": report}

    @app.post("/api/scenarios/{scenario_id}/play")
    def play(scenario_id: str, user_id: str | None = None):
        """Replays the scenario's clickstream as whoever is signed in (`user_id`, or a guest): identity comes from the
        sign-in, not the scenario."""
        scenario = scenarios().get(scenario_id)
        if scenario is None:
            raise HTTPException(404, f"no scenario {scenario_id}")
        session_id = f"demo-{uuid.uuid4().hex[:10]}"
        sent, kafka_ack_ms = play_scenario({**scenario, "user_id": user_id}, session_id, backend.producer(),
                                           cfg["kafka"]["topic"], datetime.now(timezone.utc))
        played.record(session_id, {"expected": expected_session_features(sent),
                                   "last_event_time": sent[-1]["event_time"], "kafka_ack_ms": kafka_ack_ms})
        return {"session_id": session_id, "user_id": user_id, "events": len(sent), "kafka_ack_ms": kafka_ack_ms,
                "next_query": scenario["next_query"], "timeline": scenario_timeline(sent),
                "intent": {k: scenario[f"intent_{k}"] for k in ("category", "brand", "price_target")}}

    @app.get("/api/sessions/{session_id}/freshness")
    def freshness(session_id: str):
        """Waits until every session table reflects the replayed events, reading them every FRESHNESS_POLL_S, then
        returns the stage and per-feature breakdown. The poll interval bounds only the "app read it" stage."""
        sent = played.get(session_id)
        if sent is None:
            raise HTTPException(404, f"session {session_id} was not replayed by this app process")
        names, seen = session_tables(), {}
        deadline = time.monotonic() + FRESHNESS_GIVE_UP_S
        while len(seen) < len(names) and time.monotonic() < deadline:
            for kind, name in names.items():
                if kind in seen:
                    continue
                rows = backend.rows(name, "session_id", session_id)
                if landed(kind, sent["expected"], rows, sent["last_event_time"]):
                    seen[kind] = {"rows": rows, "detected_at": datetime.now(timezone.utc)}
            time.sleep(FRESHNESS_POLL_S)
        if len(seen) < len(names):
            return {"status": "timeout", "landed": sorted(seen), "waited_s": FRESHNESS_GIVE_UP_S}
        return {"status": "ready"} | freshness_breakdown(seen, sent["last_event_time"], sent["kafka_ack_ms"], sessions)

    @app.get("/api/users/{user_id}/features")
    def user_features(user_id: str):
        """The user's batch (hourly) feature values as the online store holds them, read the way the L2 endpoint
        looks them up; for display, so read once per Play rather than per keystroke."""
        return {"user_id": user_id,
                "features": user_feature_values([backend.rows(name, "user_id", user_id)
                                                 for name in user_tables().values()])}

    return app
