"""The app's routes over an in-memory backend: everything main.py wires to Lakebase, serving, AI Search and Kafka."""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rts_app.app as app_module
from rts_app.app import create_app
from rts_shared.config import load_config

CFG = load_config(str(Path(__file__).resolve().parents[2] / "shared" / "configs" / "example.yml"))
PREFIX_ROWS = {
    "ca": [{"prefix": "ca", "rank": i, "completion": c, "prefix_len": 2, "n_tokens": 1, "is_word_boundary": True,
            "pop_30d": 10 - i, "pop_7d": 5, "ctr": 0.5, "category": cat, "category_share": 0.9, "brand": None,
            "brand_share": 0.0, "median_clicked_price": 20.0}
           for i, (c, cat) in enumerate([("camera", "Camera & Photo"), ("cat food", "Pet Supplies"),
                                         ("candle", "Home & Kitchen")], start=1)],
}
PREFIX_ROWS["c"] = [{**r, "prefix": "c"} for r in PREFIX_ROWS["ca"]]  # trending reads one-letter prefixes
# L1 puts ten Toys ahead of ten Electronics, so a guest's top 10 holds no Electronics.
HITS = [{"product_id": f"B{i:02d}", "title": f"Bag {i}", "brand": "sony",
         "category": "Toys" if i <= 10 else "Electronics", "price": 1.0 * i, "score": 1 - i / 100} for i in range(1, 21)]
SCENARIO = {"scenario_id": "sc1", "label": "Toys shopper", "next_query": "lego", "user_id": "u-ignored",
            "intent_category": "Toys", "intent_brand": "lego", "intent_price_target": 20.0,
            "events": json.dumps([
                {"offset_ms": 0, "search_id": "x", "event_type": "search", "query_id": 1, "query": "lego",
                 "product_id": None, "category": None, "brand": None, "price": None, "device": "desktop"},
                {"offset_ms": 900, "search_id": "x", "event_type": "click", "query_id": 1, "query": "lego",
                 "product_id": "P1", "category": "Toys", "brand": "lego", "price": 20.0, "device": "desktop"}])}
PERSONAS = [{"persona_id": "p1", "name": "Ada", "user_id": "u1", "category": "Electronics", "category_share": 0.8,
             "category_events_30d": 20, "n_clicks_30d": 30, "median_price_30d": 50.0}]
SESSION_TABLES = {"sess_n_clicks_30m": "rts_session_online_a", "sess_cat_cnt_30m": "rts_session_online_b",
                  "sess_last_category": "rts_session_online_c"}


class FakeProducer:
    def __init__(self):
        self.sent = []

    def produce(self, topic, key, value):
        self.sent.append(json.loads(value))

    def flush(self, timeout=None):
        return 0


class FakeBackend:
    """In-memory Lakebase, serving, AI Search and Kafka. `landed` decides whether replayed events show up online."""

    def __init__(self):
        self.kafka = FakeProducer()
        self.qac_delay_s = self.serp_delay_s = 0.0
        self.qac_calls, self.serp_calls = [], []
        self.landed = True
        self.tables = dict(SESSION_TABLES)

    def prefix_rows(self, prefix):
        return PREFIX_ROWS.get(prefix, [])

    def prefix_rows_for(self, prefixes):
        return [r for p in prefixes for r in PREFIX_ROWS.get(p, [])]

    def scenarios(self):
        return [SCENARIO]

    def personas(self):
        return PERSONAS

    def online_table_columns(self, prefix, columns):
        return [{"nspname": "s", "relname": t, "attname": c} for c, t in self.tables.items()
                if c in columns and t.startswith(prefix)]

    def rows(self, table, key, value):
        if not self.landed or key != "session_id":
            return []
        events = [e for e in self.kafka.sent if e["session_id"] == value]
        last = datetime.fromisoformat(events[-1]["event_time"].replace("Z", "+00:00"))
        at = {"event_time": last, "output_time": last + timedelta(milliseconds=20),
              "commit_time": last + timedelta(milliseconds=30)}
        if table.endswith('_a"'):
            return [{"sess_n_searches_30m": 1, "sess_n_clicks_30m": 1, "sess_n_carts_30m": 0, **at}]
        if table.endswith('_b"'):
            return [{"category": "Toys", "sess_cat_cnt_30m": 1, **at}]
        return [{"sess_last_event_type": "click", "sess_last_brand": "lego", "sess_last_category": "Toys",
                 "sess_last_price": 20.0, "commit_time": at["commit_time"]}]

    def embed(self, text):
        return [0.1, 0.2]

    def query_index(self, **kw):
        return [dict(h) for h in HITS]

    def qac_l2(self, rows, session_id, user_id):
        self.qac_calls.append(user_id)
        time.sleep(self.qac_delay_s)
        return [float(r["rank"]) for r in rows]  # reverses the L1 order

    def serp_l2(self, hits, session_id, user_id):
        self.serp_calls.append(user_id)
        time.sleep(self.serp_delay_s)
        # Signed-in users get their category first, as a persona's history would.
        return [1.0 if user_id and h["category"] == "Electronics" else 0.0 for h in hits]

    def producer(self):
        return self.kafka


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def client(backend):
    return TestClient(create_app(backend, CFG))


def test_config_tells_the_ui_its_debounce_and_l2_inputs(client):
    out = client.get("/api/config").json()
    assert out["qac_debounce_ms"] == CFG["app"]["qac_debounce_ms"]
    assert "sess_n_clicks_30m" in out["l2_inputs"]["stream"]


def test_qac_reranks_with_l2_and_records_the_full_l2_time(client):
    out = client.get("/api/qac", params={"q": "Ca", "session_id": "s1"}).json()
    assert [c["completion"] for c in out["l1"]] == ["camera", "cat food", "candle"]
    assert out["l2"]["status"] == "ok"
    assert [c["completion"] for c in out["l2"]["items"]] == ["candle", "cat food", "camera"]
    assert client.get(f"/api/qac/{out['request_id']}/l2_time").json()["l2_full_ms"] >= 0


def test_a_slow_qac_l2_falls_back_to_l1_and_its_time_arrives_later(client, backend):
    backend.qac_delay_s = 0.2
    out = client.get("/api/qac", params={"q": "ca"}).json()
    assert out["l2"] == {"status": "timeout"}
    path = f"/api/qac/{out['request_id']}/l2_time"
    assert client.get(path).status_code == 404  # still running on the QAC pool
    time.sleep(0.3)
    assert client.get(path).json()["l2_full_ms"] >= 200


def test_slow_serp_calls_do_not_hold_up_qac(client, backend):
    backend.serp_delay_s = 0.5
    searches = [threading.Thread(target=client.get, args=("/api/search",), kwargs={"params": {"q": "bag"}})
                for _ in range(16)]
    for t in searches:
        t.start()
    time.sleep(0.05)  # every SERP worker is now busy
    assert client.get("/api/qac", params={"q": "ca"}).json()["l2"]["status"] == "ok"
    for t in searches:
        t.join()


def test_search_reranks_every_hit(client):
    out = client.get("/api/search", params={"q": "bag", "user_id": "u1"}).json()
    assert out["l2"]["status"] == "ok" and out["embedding"] == "custom"
    assert [h["product_id"] for h in out["l2"]["items"]][:2] == ["B11", "B12"]  # ties keep the L1 order


def test_play_replays_as_whoever_is_signed_in_and_unknown_scenarios_404(client, backend):
    assert client.post("/api/scenarios/nope/play").status_code == 404
    out = client.post("/api/scenarios/sc1/play", params={"user_id": "u7"}).json()
    assert out["events"] == 2 and out["next_query"] == "lego"
    assert {e["user_id"] for e in backend.kafka.sent} == {"u7"}  # not the scenario's user


def test_freshness_waits_for_every_session_table_then_breaks_the_time_down(client):
    session_id = client.post("/api/scenarios/sc1/play").json()["session_id"]
    out = client.get(f"/api/sessions/{session_id}/freshness").json()
    assert out["status"] == "ready"
    assert [s["name"] for s in out["stages"]][0] == "Kafka write"
    assert {f["feature"] for f in out["features"]} >= {"sess_n_clicks_30m", "sess_cat_cnt_30m[Toys]"}


def test_freshness_gives_up_when_the_features_never_land(client, backend, monkeypatch):
    monkeypatch.setattr(app_module, "FRESHNESS_GIVE_UP_S", 0.05)
    backend.landed = False
    session_id = client.post("/api/scenarios/sc1/play").json()["session_id"]
    out = client.get(f"/api/sessions/{session_id}/freshness").json()
    assert out == {"status": "timeout", "landed": [], "waited_s": 0.05}
    assert client.get("/api/sessions/never-played/freshness").status_code == 404


def test_missing_online_tables_503_until_they_exist_and_are_not_cached_half_found(client, backend):
    del backend.tables["sess_last_category"]
    session_id = client.post("/api/scenarios/sc1/play").json()["session_id"]
    assert client.get(f"/api/sessions/{session_id}/freshness").status_code == 503
    backend.tables["sess_last_category"] = "rts_session_online_c"  # re-materialized
    assert client.get(f"/api/sessions/{session_id}/freshness").json()["status"] == "ready"


def test_persona_check_compares_each_persona_with_a_guest(client, backend):
    out = client.get("/api/checks/personas").json()
    persona = out["personas"][0]
    assert persona["name"] == "Ada" and persona["serp_top10"]["ok"]
    assert None in backend.serp_calls and "u1" in backend.serp_calls  # guest and persona both scored
    assert out["queries"] == len(app_module.CHECK_QUERIES)


def test_persona_check_falls_back_to_the_managed_embedding(client, backend):
    def broken(text):
        raise RuntimeError("embedding endpoint down")
    backend.embed = broken
    assert client.get("/api/checks/personas").json()["queries"] == len(app_module.CHECK_QUERIES)


def test_scenarios_personas_and_trending(client):
    assert client.get("/api/scenarios").json() == [{"scenario_id": "sc1", "label": "Toys shopper",
                                                    "next_query": "lego"}]
    assert client.get("/api/personas").json()[0]["user_id"] == "u1"
    assert client.get("/api/trending").json()[0] == {"query": "camera", "category": "Camera & Photo"}


def test_load_runs_reject_bad_plans(client):
    assert client.post("/api/load/qac_l2", params={"phases": "0:10"}).status_code == 400
    assert client.get("/api/load/qac_l2").status_code == 404
