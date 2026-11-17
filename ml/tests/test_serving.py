from types import SimpleNamespace

import pytest
from databricks.sdk.service.serving import ServingModelWorkloadType

from rts_ml.serving import deploy, ensure_endpoint, version_to_serve


class FakeEndpoints:
    def __init__(self, serving=None, first_deploy=None, route_optimized=True):
        # serving: the served entity (see `entity`) or None.
        # first_deploy: the endpoint exists but never got a config: "UPDATE_FAILED" or "IN_PROGRESS".
        self.serving, self.first_deploy, self.route_optimized, self.calls = serving, first_deploy, route_optimized, []

    def list(self):
        return [SimpleNamespace(name="ep")] if self.serving or self.first_deploy else []

    def get(self, name):
        if self.first_deploy:
            return SimpleNamespace(config=None, route_optimized=self.route_optimized,
                                   state=SimpleNamespace(config_update=SimpleNamespace(value=self.first_deploy)))
        ready = SimpleNamespace(ready=SimpleNamespace(value="READY"), config_update=SimpleNamespace(value="NOT_UPDATING"))
        return SimpleNamespace(config=SimpleNamespace(served_entities=[self.serving]), state=ready,
                               route_optimized=self.route_optimized, endpoint_url=f"https://host/{name}")

    def create(self, name, config, route_optimized):
        self.calls.append(("create", name, route_optimized))
        self.serving, self.first_deploy, self.route_optimized = config.served_entities[0], None, route_optimized

    def update_config(self, name, served_entities):
        self.calls.append(("update", name))
        self.serving = served_entities[0]

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


def registry(*versions):
    return SimpleNamespace(model_versions=SimpleNamespace(
        list=lambda full_name: [SimpleNamespace(version=v) for v in versions]))


def test_serves_this_runs_version_even_when_a_later_one_exists():
    # A later version (another run, or a manual registration) must not be deployed in place of this run's.
    assert version_to_serve(registry(1, 2, 3), "main.s.model", "2") == "2"


def test_serves_the_latest_version_when_the_endpoint_task_runs_alone():
    # pre_demo.sh runs the endpoint tasks with --only: the training task didn't run, so there's no task value.
    assert version_to_serve(registry(2, 10, 9), "main.s.model", "") == "10"


def test_refuses_to_change_route_optimization_in_place():
    # route_optimized is fixed when the endpoint is created; changing it means deleting the endpoint (L2 goes down).
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(serving=entity("2"), route_optimized=False))
    with pytest.raises(ValueError, match="route_optimized"):
        ensure_endpoint(w, "ep", entity("2"), route_optimized=True)
    assert w.serving_endpoints.calls == []


BLOCK = {"endpoint": "ep", "route_optimized": True, "workload_type": "GPU_SMALL", "workload_size": "Small",
         "scale_to_zero_enabled": False}


def test_deploy_serves_the_version_with_the_configured_workload():
    w = SimpleNamespace(serving_endpoints=FakeEndpoints(serving=entity("1")))
    summary = deploy(w, BLOCK, "main.s.model", "2")
    served = w.serving_endpoints.serving
    assert (served.entity_version, served.workload_type, served.workload_size, served.scale_to_zero_enabled) == \
        ("2", ServingModelWorkloadType.GPU_SMALL, "Small", False)
    assert summary == {"endpoint": "ep", "model": "main.s.model", "version": "2", "action": "updated",
                       "route_optimized": True, "url": "https://host/ep"}


def test_deploy_creates_a_missing_endpoint_route_optimized_from_config():
    w = SimpleNamespace(serving_endpoints=FakeEndpoints())
    assert deploy(w, {**BLOCK, "route_optimized": False}, "main.s.model", "2")["action"] == "created"
    assert w.serving_endpoints.calls == [("create", "ep", False)]
