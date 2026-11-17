from types import SimpleNamespace

from databricks.sdk.errors import NotFound, PermissionDenied

from rts_data.teardown import teardown


class FakeFe:
    """Features and the stream as Feature Engineering holds them; `broken` names raise `error` on lookup."""

    def __init__(self, features=(), stream=None, broken=(), error=None):
        self.features = {f: [SimpleNamespace(is_online=False), SimpleNamespace(is_online=True)] for f in features}
        self.stream, self.broken, self.error, self.calls = stream, set(broken), error, []

    def get_feature(self, full_name):
        if full_name in self.broken:
            raise self.error
        if full_name not in self.features:
            raise NotFound(f"Feature '{full_name}' not found.")

    def list_materialized_features(self, feature_name):
        return list(self.features.get(feature_name, []))

    def delete_materialized_feature(self, materialized_feature):
        # Deleting the offline materialization removes its online copy too.
        self.calls.append("delete_materialization")
        for mats in self.features.values():
            if materialized_feature in mats:
                mats.clear()

    def delete_feature(self, full_name):
        self.calls.append(("delete_feature", full_name))

    def get_stream(self, name):
        if name in self.broken:
            raise self.error
        if name != self.stream:
            raise NotFound(f"Stream '{name}' does not exist.")

    def delete_stream(self, name):
        self.calls.append(("delete_stream", name))


def test_dry_run_lists_what_exists_and_deletes_nothing():
    fe = FakeFe(features=["c.s.a"], stream="c.s.stream")
    result = teardown(fe, ["c.s.a", "c.s.b"], "c.s.stream", delete=False)
    assert result == {"confirm": False, "features": ["c.s.a"], "materializations": 1, "stream": "c.s.stream",
                      "errors": []}
    assert fe.calls == []


def test_deletes_materializations_before_the_feature_then_the_stream():
    fe = FakeFe(features=["c.s.a"], stream="c.s.stream")
    teardown(fe, ["c.s.a"], "c.s.stream", delete=True)
    assert fe.calls == ["delete_materialization", ("delete_feature", "c.s.a"), ("delete_stream", "c.s.stream")]


def test_an_error_that_is_not_not_found_is_reported_not_treated_as_absent():
    # It used to treat any lookup error as "absent", so a permissions error skipped the feature and reported success.
    fe = FakeFe(features=["c.s.a"], stream="c.s.stream", broken=["c.s.a", "c.s.stream"],
                error=PermissionDenied("no USE SCHEMA"))
    result = teardown(fe, ["c.s.a"], "c.s.stream", delete=True)
    assert result["features"] == [] and result["stream"] is None
    assert [e.split(":")[0] for e in result["errors"]] == ["c.s.a", "stream"]
    assert fe.calls == []


def test_not_found_by_message_counts_as_absent():
    # The Feature Engineering client may wrap the SDK error; its message still says so.
    fe = FakeFe(broken=["c.s.a"], error=RuntimeError("Feature 'c.s.a' not found."))
    assert teardown(fe, ["c.s.a"], "c.s.stream", delete=True)["errors"] == []
