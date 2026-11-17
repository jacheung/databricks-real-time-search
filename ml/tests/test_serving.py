from types import SimpleNamespace

from databricks.sdk.service.serving import ServingModelWorkloadType

from rts_ml.serving import ensure_endpoint


class FakeEndpoints:
    def __init__(self, serving=None, first_deploy=None):
        # serving: the served entity (see `entity`) or None.
        # first_deploy: the endpoint exists but never got a config: "UPDATE_FAILED" or "IN_PROGRESS".
        self.serving, self.first_deploy, self.calls = serving, first_deploy, []

    def list(self):
        return [SimpleNamespace(name="ep")] if self.serving or self.first_deploy else []

    def get(self, name):
        if self.first_deploy:
            return SimpleNamespace(config=None,
                                   state=SimpleNamespace(config_update=SimpleNamespace(value=self.first_deploy)))
        return SimpleNamespace(config=SimpleNamespace(served_entities=[self.serving]))

    def create(self, name, config, route_optimized):
        self.calls.append(("create", name, route_optimized))

    def update_config(self, name, served_entities):
        self.calls.append(("update", name))

    def delete(self, name):
        self.calls.append(("delete", name))


def entity(version, scale_to_zero=False, size="Small", workload_type=ServingModelWorkloadType.CPU):
    return SimpleNamespace(entity_name="main.s.model", entity_version=version, scale_to_zero_enabled=scale_to_zero,
                           workload_size=size, workload_type=workload_type)


def test_creates_a_missing_endpoint():
    w = SimpleNamespace(serving_endpoints=FakeEndpoints())
    assert ensure_endpoint(w, "ep", entity("2"), route_optimized=True) == "created"
    assert w.serving_endpoints.calls == [("create", "ep", True)]


def test_updates_an_endpoint_serving_another_version():
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(serving=entity("1")))
    assert ensure_endpoint(w, "ep", entity("2"), route_optimized=True) == "updated"
    assert w.serving_endpoints.calls == [("update", "ep")]


def test_leaves_an_endpoint_already_serving_the_version():
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(serving=entity("2")))
    assert ensure_endpoint(w, "ep", entity("2"), route_optimized=True) == "unchanged"
    assert w.serving_endpoints.calls == []


def test_restores_scale_to_zero_someone_else_turned_on():
    # e2-demo-fe's cleanup bot turns scale-to-zero on overnight; cold starts make L2 time out to the L1 order.
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(serving=entity("2", scale_to_zero=True)))
    assert ensure_endpoint(w, "ep", entity("2"), route_optimized=True) == "updated"
    assert w.serving_endpoints.calls == [("update", "ep")]


def test_applies_a_new_workload_size():
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(serving=entity("2", size="Small")))
    assert ensure_endpoint(w, "ep", entity("2", size="Medium"), route_optimized=True) == "updated"
    assert w.serving_endpoints.calls == [("update", "ep")]


def test_applies_a_new_workload_type():
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(serving=entity("2")))
    gpu = entity("2", workload_type=ServingModelWorkloadType.GPU_SMALL)
    assert ensure_endpoint(w, "ep", gpu, route_optimized=True) == "updated"
    assert w.serving_endpoints.calls == [("update", "ep")]


def test_recreates_an_endpoint_whose_first_deploy_failed():
    # 2026-11-10: the online store was purged, both L2 endpoints failed their first deploy and were left with no
    # config, so reruns crashed instead of deploying again.
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(first_deploy="UPDATE_FAILED"))
    assert ensure_endpoint(w, "ep", entity("2"), route_optimized=True) == "created"
    assert w.serving_endpoints.calls == [("delete", "ep"), ("create", "ep", True)]


def test_leaves_a_first_deploy_in_progress_to_wait_ready():
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(first_deploy="IN_PROGRESS"))
    assert ensure_endpoint(w, "ep", entity("2"), route_optimized=True) == "unchanged"
    assert w.serving_endpoints.calls == []
