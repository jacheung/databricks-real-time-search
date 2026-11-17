import time

import pytest

from rts_app.load import LoadRun, pctl, qps_per_user, schedule, summarize, users_for_qps


def test_schedule_is_open_loop_at_each_phase_rate():
    s = schedule([(10, 1), (20, 0.5)])
    assert len(s) == 20
    assert [p for _, p in s] == [0] * 10 + [1] * 10
    assert s[1][0] == pytest.approx(0.1)
    assert s[10][0] == pytest.approx(1.0)       # phase 1 starts when phase 0 ends
    assert s[11][0] == pytest.approx(1.05)


def test_pctl_is_nearest_rank():
    xs = list(range(1, 1001))
    assert pctl(xs, 0.5) == 500 and pctl(xs, 0.99) == 990 and pctl(xs, 0.999) == 999 and pctl([7], 0.99) == 7


def test_users_from_typing_assumptions():
    assert qps_per_user(keystrokes=4, searches_per_min=2) == pytest.approx(8 / 60)
    assert users_for_qps(400, keystrokes=4, searches_per_min=2) == 3000


def test_summarize_per_phase_and_window():
    # (phase, scheduled offset s, latency from schedule ms, send lag ms, ok)
    recs = [(0, i / 10, 10.0 + i, 0.1, True) for i in range(10)] + [(1, 1 + i / 10, 50.0, 0.2, i != 0) for i in range(10)]
    out = summarize(recs, phases=[(10, 1), (10, 1)], window_s=1, slow_ms=36)
    p0, p1 = out["phases"]
    assert p0["qps"] == 10 and p0["n"] == 10 and p0["errors"] == 0 and p0["p50"] == 14.0 and p0["slow"] == 0
    assert p1["errors"] == 1 and p1["slow"] == 9    # failed requests aren't counted again as slow
    assert [w["t"] for w in out["windows"]] == [0, 1] and out["windows"][1]["errors"] == 1


def test_run_measures_latency_from_the_schedule_so_a_backed_up_client_shows():
    # 2 workers, 20 ms calls, 200 QPS: the client can't keep up and requests queue. Open-loop latency counts the queue.
    run = LoadRun(lambda: time.sleep(0.02), [(200, 0.25)], workers=2)
    run.start()
    run.join(timeout=10)
    assert run.done and len(run.records) == 50
    lat = sorted(r[2] for r in run.records)
    assert lat[-1] > 200                            # the last request waited behind ~48 others


def test_run_counts_errors_and_can_stop():
    def boom():
        raise RuntimeError("503")
    run = LoadRun(boom, [(100, 0.1)], workers=4)
    run.start()
    run.join(timeout=10)
    assert run.records and all(not r[4] for r in run.records) and "503" in run.last_error

    slow = LoadRun(lambda: None, [(10, 60)], workers=2)
    slow.start()
    slow.stop()
    slow.join(timeout=5)
    assert slow.done and len(slow.records) < 600
