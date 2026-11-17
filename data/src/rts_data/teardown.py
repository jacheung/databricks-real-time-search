"""Tear down the clickstream features and the Kafka Stream (the `teardown` notebook): materializations first, then
each feature, then the Stream. Only "not found" means absent; any other error is reported, not skipped."""

from __future__ import annotations

from databricks.sdk.errors import NotFound


def _absent(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, NotFound) or "not found" in message or "does not exist" in message


def teardown(fe, feature_names, stream: str, delete: bool) -> dict:
    """`fe` is a FeatureEngineeringClient. With `delete=False` it only lists what it would delete."""
    result = {"confirm": delete, "features": [], "materializations": 0, "stream": None, "errors": []}
    for name in feature_names:
        try:
            fe.get_feature(full_name=name)
        except Exception as error:
            if not _absent(error):
                result["errors"].append(f"{name}: {type(error).__name__}: {str(error)[:200]}")
            continue
        result["features"].append(name)
        # A feature with a live materialization strands it, so materializations go first. Batch features have an
        # offline + online pair: deleting the offline one removes the online one too, and the online one can't be
        # deleted on its own. Streaming features are online-only.
        mats = list(fe.list_materialized_features(feature_name=name))
        offline = [m for m in mats if not m.is_online]
        result["materializations"] += len(offline or mats)
        if not delete:
            continue
        try:
            for m in offline or mats:
                fe.delete_materialized_feature(materialized_feature=m)
            assert not list(fe.list_materialized_features(feature_name=name)), "materializations left"
            fe.delete_feature(full_name=name)
        except Exception as error:
            result["errors"].append(f"{name}: {type(error).__name__}: {str(error)[:200]}")
    try:
        fe.get_stream(name=stream)
    except Exception as error:
        if not _absent(error):
            result["errors"].append(f"stream: {type(error).__name__}: {str(error)[:200]}")
        return result
    result["stream"] = stream
    if delete:
        try:
            fe.delete_stream(name=stream)
        except Exception as error:
            result["errors"].append(f"stream: {type(error).__name__}: {str(error)[:200]}")
    return result
