"""Demo selection rules (pandas-free, Spark-free, so tested locally): which sessions become replayable scenarios and
which user-feature views are live. The notebooks `demo_scenarios` and `demo_personas` call these."""

from __future__ import annotations


def pick_scenarios(candidates: list[dict], n: int) -> list[dict]:
    """Up to `n` scenarios from candidate sessions (`session_id`, `intent_category`, `user_id`, `brand_clicks`,
    `category_clicks`): the strongest per (intent category, logged in or not), then logged-in and guest picks
    alternately, strongest first, never two in one intent category."""
    strength = lambda r: (-r["brand_clicks"], -r["category_clicks"], r["session_id"])  # noqa: E731
    best = {}
    for r in sorted(candidates, key=strength):
        best.setdefault((r["intent_category"], r["user_id"] is not None), r)
    pools = {flag: sorted((r for (_, logged_in), r in best.items() if logged_in == flag), key=strength)
             for flag in (True, False)}
    picked, used = [], set()
    while len(picked) < n and any(pools.values()):
        for flag in (True, False):
            while pools[flag] and pools[flag][0]["intent_category"] in used:
                pools[flag].pop(0)
            if pools[flag] and len(picked) < n:
                r = pools[flag].pop(0)
                picked.append(r)
                used.add(r["intent_category"])
    return picked


def live_offline_views(views: list[str], exists) -> list[str]:
    """The `<offline table>_latest_view` names whose offline table `exists(name)`: deleting a materialization drops
    its offline table but leaves the view behind (2026-11-10)."""
    return [v for v in views if exists(v.removesuffix("_latest_view"))]
