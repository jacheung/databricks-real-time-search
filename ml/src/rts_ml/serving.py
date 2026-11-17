"""Model serving endpoints the notebooks deploy: create if missing, move to a new version, or leave as is."""

from __future__ import annotations

import time

from databricks.sdk.service.serving import EndpointCoreConfigInput


def ensure_endpoint(w, name: str, entity, route_optimized: bool) -> str:
    """Serve `entity` (a ServedEntityInput) on `name`; returns "created", "updated" or "unchanged"."""
    def create():
        w.serving_endpoints.create(name=name, config=EndpointCoreConfigInput(name=name, served_entities=[entity]),
                                   route_optimized=route_optimized)
        return "created"

    if not any(e.name == name for e in w.serving_endpoints.list()):
        return create()
    endpoint = w.serving_endpoints.get(name)
    if endpoint.config is None:  # never deployed: its first config is still building, or failed
        if endpoint.state.config_update.value != "UPDATE_FAILED":
            return "unchanged"  # wait_ready waits for it
        # Nothing was ever served, so nothing is lost. 2026-11-10: the online store was purged and both L2
        # endpoints were left like this.
        w.serving_endpoints.delete(name)
        return create()
    served = endpoint.config.served_entities or []
    # scale_to_zero too: e2-demo-fe's cleanup bot turns it on overnight, and cold starts time L2 out.
    key = lambda s: (s.entity_name, s.entity_version, s.workload_type, s.workload_size,  # noqa: E731
                     bool(s.scale_to_zero_enabled))
    if [key(s) for s in served] == [key(entity)]:
        return "unchanged"
    w.serving_endpoints.update_config(name=name, served_entities=[entity])
    return "updated"


def wait_ready(w, name: str, polls: int = 80, interval_s: int = 30):
    """Bounded wait (serverless sessions can outlive long blocking calls); returns the ready endpoint."""
    for _ in range(polls):
        endpoint = w.serving_endpoints.get(name)
        if endpoint.state.ready.value == "READY" and endpoint.state.config_update.value == "NOT_UPDATING":
            return endpoint
        if endpoint.state.config_update.value == "UPDATE_FAILED":
            raise RuntimeError(f"{name} config update failed; see its build logs")
        time.sleep(interval_s)
    raise TimeoutError(f"{name} not ready after {polls * interval_s // 60} min: {endpoint.state}")
