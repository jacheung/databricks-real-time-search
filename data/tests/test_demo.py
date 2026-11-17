from rts_data.demo import live_offline_views, pick_scenarios


def cand(session_id, category, logged_in, brand_clicks, category_clicks=0):
    return {"session_id": session_id, "intent_category": category, "user_id": "u1" if logged_in else None,
            "brand_clicks": brand_clicks, "category_clicks": category_clicks}


def test_pick_scenarios_alternates_logged_in_and_guest_one_per_category_strongest_first():
    candidates = [cand("s1", "Cameras", True, 9), cand("s2", "Cameras", False, 8), cand("s3", "Audio", False, 7),
                  cand("s4", "Audio", True, 3), cand("s5", "Toys", True, 5), cand("s6", "Books", False, 1)]
    picked = pick_scenarios(candidates, n=4)
    assert [r["session_id"] for r in picked] == ["s1", "s3", "s5", "s6"]
    assert len({r["intent_category"] for r in picked}) == 4


def test_pick_scenarios_takes_the_strongest_session_per_category_and_login():
    # Ties on brand clicks break on category clicks, then session id, so reruns pick the same sessions.
    candidates = [cand("s9", "Cameras", True, 4, 1), cand("s2", "Cameras", True, 4, 3),
                  cand("s1", "Cameras", True, 4, 3)]
    assert [r["session_id"] for r in pick_scenarios(candidates, n=1)] == ["s1"]


def test_pick_scenarios_returns_fewer_when_categories_run_out():
    assert len(pick_scenarios([cand("s1", "Cameras", True, 2), cand("s2", "Cameras", False, 1)], n=3)) == 1


def test_live_offline_views_skips_views_whose_offline_table_was_deleted():
    # Deleting a materialization drops its offline table but leaves the _latest_view behind (2026-11-10).
    views = ["rts_user_offline_a_latest_view", "rts_user_offline_b_latest_view"]
    assert live_offline_views(views, exists=lambda t: t == "rts_user_offline_a") == ["rts_user_offline_a_latest_view"]
