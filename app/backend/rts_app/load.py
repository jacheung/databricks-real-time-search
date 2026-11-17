"""Open-loop load generator for an L2 endpoint, run inside the app (route-optimized endpoints need the app SP's OAuth).

Requests leave on a fixed schedule whatever the responses do, and each latency is measured from its scheduled time:
a slow endpoint (or a backed-up client) shows as latency instead of quietly lowering the load. `send_lag` (start minus
schedule) says whether the client kept up; when it grows, the numbers measure the app, not the endpoint.
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

Phase = tuple[float, float]   # (QPS, seconds)
Record = tuple[int, float, float, float, bool]   # (phase, scheduled offset s, latency ms, send lag ms, ok)


def schedule(phases: list[Phase]) -> list[tuple[float, int]]:
    """(offset s, phase) for every request: evenly spaced at each phase's rate, phases back to back."""
    out, t0 = [], 0.0
    for p, (qps, seconds) in enumerate(phases):
        out += [(t0 + i / qps, p) for i in range(round(qps * seconds))]
        t0 += seconds
    return out


def pctl(xs_sorted: list[float], q: float) -> float:
    """Nearest-rank percentile of a sorted list (as `service.latency_stats`)."""
    return xs_sorted[max(0, math.ceil(q * len(xs_sorted)) - 1)]


def qps_per_user(keystrokes: float, searches_per_min: float) -> float:
    """Average QAC L2 QPS per active user: one call per keystroke (the debounce is shorter than a keystroke)."""
    return keystrokes * searches_per_min / 60


def users_for_qps(qps: float, keystrokes: float, searches_per_min: float) -> int:
    return round(qps / qps_per_user(keystrokes, searches_per_min))


def _stats(recs: list[Record], seconds: float, slow_ms: float) -> dict:
    ok = sorted(r[2] for r in recs if r[4])
    lag = sorted(r[3] for r in recs)
    out = {"n": len(recs), "qps": round(len(recs) / seconds, 1) if seconds else 0, "errors": len(recs) - len(ok),
           "slow": sum(x > slow_ms for x in ok)}
    if ok:
        out |= {q: round(pctl(ok, v), 1) for q, v in (("p50", .5), ("p95", .95), ("p99", .99), ("p999", .999))}
        out["max"] = round(ok[-1], 1)
    if lag:
        out["send_lag_p99"] = round(pctl(lag, .99), 1)
    return out


def summarize(records: list[Record], phases: list[Phase], window_s: float, slow_ms: float) -> dict:
    """Per phase and per `window_s` window: count, achieved QPS, errors, requests over `slow_ms` (the app's L2
    timeout: those fall back to the L1 order), latency percentiles from the schedule, and the client's send lag."""
    by_phase = [[r for r in records if r[0] == p] for p in range(len(phases))]
    windows: dict[int, list[Record]] = {}
    for r in records:
        windows.setdefault(int(r[1] // window_s), []).append(r)
    return {"phases": [_stats(rs, s, slow_ms) | {"target_qps": q, "seconds": s}
                       for rs, (q, s) in zip(by_phase, phases)],
            "windows": [{"t": k * window_s} | _stats(windows[k], window_s, slow_ms) for k in sorted(windows)]}


class LoadRun:
    """Runs `call` on the schedule for `phases` on a thread pool of `workers`."""

    def __init__(self, call: Callable[[], object], phases: list[Phase], workers: int):
        self.call, self.phases, self.records = call, phases, []
        self.last_error, self.done = None, False
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout: float | None = None):
        self._thread.join(timeout)

    def _one(self, t_sched: float, offset: float, phase: int):
        start = time.perf_counter()
        try:
            self.call()
            ok = True
        except Exception as e:  # noqa: BLE001 - every failure counts as an error; the last one is kept for the report
            ok, self.last_error = False, repr(e)[:300]
        end = time.perf_counter()
        self.records.append((phase, offset, (end - t_sched) * 1000, (start - t_sched) * 1000, ok))

    def _run(self):
        t0 = time.perf_counter() + 0.1
        try:
            for offset, phase in schedule(self.phases):
                if self._stop.is_set():
                    break
                t = t0 + offset
                if (wait := t - time.perf_counter()) > 0:
                    time.sleep(wait)
                self._pool.submit(self._one, t, offset, phase)
        finally:
            self._pool.shutdown(wait=True)
            self.done = True
