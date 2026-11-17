"""QAC L2 load test: step, spike and soak against the QAC L2 endpoint, generated inside the app.

The route-optimized endpoint takes only OAuth, so the app (its SP) sends the load, in-region, through
`/api/load/qac_l2` (open-loop, `rts_app.load`). This script only starts runs, polls them and scrapes the endpoint's
metrics export once a minute for the server-side view (latency, queue time, provisioned concurrency, CPU). From the
repo root:

    uv run --no-project --with "databricks-sdk>=0.81" --with pyyaml python -I scripts/load_qac_l2.py \\
        shared/configs/fevm.yml

1. Warm-up, then step test: each step `STEP_S`; stops at the knee (p99 over 2x the first step's, errors over 1%,
   or achieved QPS short of target) or when the app can't keep up (its send lag, so the client is the limit).
2. Spike: 10% of capacity, then capacity within one second. 3. Soak: 70% of capacity for `SOAK_S`.
Writes /tmp/rts_load/qac_l2_<time>.json and prints a markdown summary, including users at the measured capacity.
The demo app shares its process with the load: don't demo while this runs.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", p) for p in ("shared/src", "app/backend")]
from rts_app.load import qps_per_user, users_for_qps  # noqa: E402
from rts_shared.config import load_config  # noqa: E402

from databricks.sdk import WorkspaceClient  # noqa: E402

APP = "rts-search"
STEPS = [25, 50, 100, 150, 200, 300, 400, 600, 800, 1200]
STEP_S, SPIKE_LOW_S, SPIKE_HIGH_S, SOAK_S = 180, 120, 180, 1800
KNEE_P99_X, MAX_ERROR_RATE, MAX_SEND_LAG_MS = 2.0, 0.01, 10.0
# (label, keystrokes before a completion is picked, searches per active user per minute)
USER_ASSUMPTIONS = [("heavy", 6, 3), ("typical", 4, 2), ("light", 3, 1)]
KEYSTROKE_S = 0.3   # ~40 wpm; a typing user sends 1/KEYSTROKE_S QPS while typing


class App:
    def __init__(self, w: WorkspaceClient):
        self.w, self.url = w, w.apps.get(APP).url.rstrip("/")

    def call(self, method: str, path: str, timeout: float = 300):
        req = urllib.request.Request(self.url + path, method=method, headers=self.w.config.authenticate())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    def run(self, phases: list[tuple[float, float]], window_s: float) -> dict:
        spec = ",".join(f"{q:g}:{s:g}" for q, s in phases)
        print(f"  run {spec}", flush=True)
        self.call("POST", f"/api/load/qac_l2?phases={spec}")
        time.sleep(sum(s for _, s in phases))
        while not (out := self.call("GET", f"/api/load/qac_l2?window_s={window_s:g}"))["done"]:
            time.sleep(5)
        if out["last_error"]:
            print(f"  last error: {out['last_error']}")
        return out


_LINE = re.compile(r'^(\w+?)(_bucket)?\{(.*)\}\s+(\S+)\s+(\d+)$')


def parse_metrics(text: str) -> dict[int, dict]:
    """Prometheus text -> {minute ts ms: {gauge: value, histogram: {le: cumulative count}}}."""
    out: dict[int, dict] = {}
    for line in text.splitlines():
        if not (m := _LINE.match(line.strip())):
            continue
        name, bucket, labels, value, ts = m.groups()
        row = out.setdefault(int(ts), {})
        if bucket:
            le = re.search(r'le="([^"]+)"', labels).group(1)
            row.setdefault(name, {})[float(le)] = float(value)
        else:
            row[name] = float(value)
    return out


def hist_pctl(buckets: dict[float, float], q: float) -> float | None:
    """Upper bound of the first bucket holding the q-th request (bucket width is 5 ms)."""
    total = buckets.get(float("inf")) or max(buckets.values(), default=0)
    if not total:
        return None
    return next(le for le in sorted(buckets) if buckets[le] >= q * total)


class Scraper(threading.Thread):
    def __init__(self, w: WorkspaceClient, endpoint: str):
        super().__init__(daemon=True)
        self.w, self.endpoint, self.minutes, self._stop = w, endpoint, {}, threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                r = self.w.api_client.do("GET", f"/api/2.0/serving-endpoints/{self.endpoint}/metrics", raw=True,
                                         headers={"Accept": "text/plain; version=0.0.4; charset=utf-8"})
                self.minutes.update(parse_metrics(r["contents"].read().decode()))
            except Exception as e:  # noqa: BLE001 - a missed scrape loses one minute, not the run
                print(f"  metrics scrape failed: {e!r}")
            self._stop.wait(60)

    def stop(self):
        self._stop.set()

    def table(self) -> list[dict]:
        rows = []
        for ts in sorted(self.minutes):
            m = self.minutes[ts]
            row = {"minute": datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%H:%M")}
            for h in ("request_latency_ms", "model_prediction_latency_ms", "model_queue_time_ms"):
                if h in m:
                    row[f"{h}_p50"], row[f"{h}_p99"] = hist_pctl(m[h], .5), hist_pctl(m[h], .99)
            for g in ("request_count_total", "request_5xx_count_total", "provisioned_concurrent_requests_total",
                      "cpu_usage_percentage"):
                if g in m:
                    row[g] = round(m[g], 1)
            rows.append(row)
        return rows


def knee_reason(step: dict, base_p99: float) -> str | None:
    if step["send_lag_p99"] > MAX_SEND_LAG_MS:
        return f"client limit: the app's send lag p99 {step['send_lag_p99']} ms"
    if step["errors"] > MAX_ERROR_RATE * step["n"]:
        return f"errors {step['errors']}/{step['n']}"
    if step["qps"] < 0.95 * step["target_qps"]:
        return f"achieved {step['qps']} of {step['target_qps']} QPS"
    if step.get("p99", 1e9) > KNEE_P99_X * base_p99:
        return f"p99 {step.get('p99')} ms > {KNEE_P99_X}x the first step's {base_p99} ms"
    return None


def main(config_path: str):
    cfg = load_config(config_path)
    w = WorkspaceClient(profile=cfg["workspace"]["profile"])
    app, budget = App(w), cfg["app"]["timeouts_ms"]["qac_l2"]
    scraper = Scraper(w, cfg["models"]["qac_l2"]["endpoint"])
    scraper.start()
    report = {"started": datetime.now(timezone.utc).isoformat(), "budget_ms": budget, "steps": []}

    print("warm-up")
    report["warmup"] = app.run([(25, 30)], 30)["summary"]["phases"][0]

    print("1/3 step test")
    capacity, capacity_in_budget = None, None
    for qps in STEPS:
        out = app.run([(qps, STEP_S)], STEP_S)
        step = out["summary"]["phases"][0]
        report.setdefault("payload", {k: out[k] for k in ("workers", "prefixes", "users", "sessions",
                                                          "candidates_mean")})
        report["steps"].append(step)
        print(f"  {qps} QPS: {step}")
        if reason := knee_reason(step, report["steps"][0].get("p99", 1e9)):
            report["knee"] = {"qps": qps, "reason": reason}
            break
        capacity = qps
        if step.get("p99", 1e9) <= budget:
            capacity_in_budget = qps
    report["capacity_qps"], report["capacity_in_budget_qps"] = capacity, capacity_in_budget
    if not capacity:
        raise SystemExit(f"no step passed: {report.get('knee')}")

    print(f"2/3 spike to {capacity} QPS")
    low = max(5, round(capacity * 0.1))
    report["spike"] = app.run([(low, SPIKE_LOW_S), (capacity, SPIKE_HIGH_S)], 5)["summary"]

    soak_qps = round(capacity * 0.7)
    print(f"3/3 soak at {soak_qps} QPS for {SOAK_S // 60} min")
    report["soak"] = app.run([(soak_qps, SOAK_S)], 60)["summary"]

    time.sleep(70)   # the last minute's metrics
    scraper.stop()
    report["server_minutes"] = scraper.table()
    report["users"] = [{"assumption": label, "keystrokes": k, "searches_per_min": s,
                        "qps_per_user": round(qps_per_user(k, s), 3),
                        "users_at_capacity": users_for_qps(capacity, k, s),
                        "users_at_capacity_in_budget": users_for_qps(capacity_in_budget, k, s)
                        if capacity_in_budget else None}
                       for label, k, s in USER_ASSUMPTIONS]
    report["typing_at_once_at_capacity"] = round(capacity * KEYSTROKE_S)

    os.makedirs("/tmp/rts_load", exist_ok=True)
    path = f"/tmp/rts_load/qac_l2_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"wrote {path}")
    print(json.dumps({k: report[k] for k in ("payload", "capacity_qps", "capacity_in_budget_qps", "knee", "users",
                                             "typing_at_once_at_capacity") if k in report}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "shared/configs/fevm.yml")
