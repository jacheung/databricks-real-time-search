from datetime import timedelta
from pathlib import Path

import pytest

from rts_shared.config import load_config
from rts_shared.features import ENGAGED, ENGAGED_EVENTS, feature_keys, feature_specs, online_table_prefix

ROOT = Path(__file__).resolve().parents[2]  # real-time-search/


@pytest.fixture(scope="module")
def cfg():
    return load_config(str(ROOT / "shared" / "configs" / "example.yml"))


@pytest.fixture(scope="module")
def specs(cfg):
    return feature_specs(cfg)


def by_name(specs):
    return {s.name: s for s in specs}


def test_names_are_unique_and_lowercase_identifiers(specs):
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    assert all(n.isidentifier() and n == n.lower() for n in names)


def test_category_affinity_is_keyed_on_category(specs):
    # One feature per entity keyed on (entity, category): each candidate row looks up the count for its own
    # category, instead of one column per category.
    names = by_name(specs)
    session, user = names["sess_cat_cnt_30m"], names["user_cat_cnt_30d"]
    assert session.entity == ["session_id", "category"] and session.filter == ENGAGED
    assert user.entity == ["user_id", "category"] and user.filter == f"user_id IS NOT NULL AND {ENGAGED}"
    assert not [n for n in names if n.startswith(("sess_cat_", "user_cat_")) and n not in (session.name, user.name)]


def test_entities_and_groups(specs):
    assert {s.group for s in specs} == {"session", "user", "item"}
    assert all(s.entity[0] == {"session": "session_id", "user": "user_id", "item": "product_id"}[s.group]
               for s in specs)


def test_windows_follow_config(cfg, specs):
    fs = cfg["feature_store"]
    windows = {s.group: s.window for s in specs if s.op != "ColumnSelection"}
    assert windows == {"session": timedelta(minutes=fs["session_window_minutes"]),
                       "user": timedelta(days=fs["user_window_days"]),
                       "item": timedelta(days=fs["item_window_days"])}
    assert all(s.window is None for s in specs if s.op == "ColumnSelection")


def test_user_features_skip_guests(specs):
    assert all("user_id IS NOT NULL" in s.filter for s in specs if s.group == "user")


def test_price_tier_and_latest_intent(specs):
    names = by_name(specs)
    assert {names[f"sess_{op}_price_30m"].op for op in ("avg", "min", "max", "std")} == {"Avg", "Min", "Max", "StddevPop"}
    assert names["user_median_price_30d"].op == "ApproxPercentile"
    assert names["user_median_price_30d"].params == {"percentile": 0.5}
    for col in ("category", "brand", "price"):
        latest = names[f"sess_last_{col}"]
        assert (latest.op, latest.input, latest.filter) == ("ColumnSelection", col, ENGAGED)
    assert names["sess_last_event_type"].filter is None


def test_counts(specs):
    groups = [s.group for s in specs]
    assert groups.count("session") == 3 + 4 + 1 + 4
    assert groups.count("user") == 3 + 1
    assert groups.count("item") == 2


def test_stream_prefix_applies_to_session_features_only(cfg):
    stream = by_name(feature_specs(cfg, session_prefix="value."))
    cat = stream["sess_cat_cnt_30m"]
    assert cat.entity == ["value.session_id", "value.category"]
    assert cat.input == "value.product_id" and cat.timeseries == "value.event_time"
    assert cat.filter == "value.event_type IN ('click', 'add_to_cart')"
    assert stream["sess_last_category"].input == "value.category"
    assert stream["sess_last_event_type"].filter is None
    user = stream["user_cat_cnt_30d"]
    assert user.entity == ["user_id", "category"] and user.timeseries == "event_time"
    assert user.filter == f"user_id IS NOT NULL AND {ENGAGED}"


def test_unprefixed_specs_are_unchanged(specs):
    assert all(s.timeseries == "event_time" and not s.input.startswith("value.") for s in specs)


def test_names_and_keys_are_unchanged(cfg):
    # Trained models, Feature Views and online tables are bound to these names: a rename or re-key means
    # re-materializing and redeploying, so it must fail here first.
    session, user, item = ("session_id",), ("user_id",), ("product_id",)
    assert feature_keys(cfg) == {
        "sess_n_clicks_30m": session, "sess_n_carts_30m": session, "sess_n_searches_30m": session,
        "sess_avg_price_30m": session, "sess_min_price_30m": session, "sess_max_price_30m": session,
        "sess_std_price_30m": session, "sess_cat_cnt_30m": (*session, "category"),
        "sess_last_category": session, "sess_last_brand": session, "sess_last_price": session,
        "sess_last_event_type": session,
        "user_median_price_30d": user, "user_n_clicks_30d": user, "user_n_carts_30d": user,
        "user_cat_cnt_30d": (*user, "category"),
        "item_clicks_7d": item, "item_carts_7d": item,
    }


def test_lookup_keys_drop_the_stream_prefix(cfg):
    # ML looks session features up by request-side keys, whatever the source's field names.
    stream = {s.name: tuple(k.removeprefix(s.prefix) for k in s.entity) for s in feature_specs(cfg, "value.")}
    assert stream == feature_keys(cfg)


def test_kinds_follow_the_online_tables(specs):
    # Each session Feature View lands in three online tables: windowed counts and prices, the per-category count,
    # and the latest values.
    kinds = {}
    for s in specs:
        kinds.setdefault((s.group, s.kind), []).append(s.name)
    assert kinds[("session", "category")] == ["sess_cat_cnt_30m"]
    assert sorted(kinds[("session", "last")]) == ["sess_last_brand", "sess_last_category", "sess_last_event_type",
                                                  "sess_last_price"]
    assert len(kinds[("session", "window")]) == 7
    assert kinds[("user", "category")] == ["user_cat_cnt_30d"]


def test_engaged_is_clicks_and_carts():
    assert ENGAGED_EVENTS == ("click", "add_to_cart")
    assert ENGAGED == "event_type IN ('click', 'add_to_cart')"


def test_online_table_prefix():
    assert online_table_prefix("session") == "rts_session_online"
    assert online_table_prefix("user") == "rts_user_online"
