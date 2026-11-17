"""The demo scripts name jobs, tasks and the app by hand; a rename in a bundle must fail here, not in a demo."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def bundle_jobs() -> dict[str, set[str]]:
    jobs = {}
    for path in ROOT.glob("*/resources/*.yml"):
        for key, job in ((yaml.safe_load(path.read_text()) or {}).get("resources", {}).get("jobs") or {}).items():
            jobs[key] = {t["task_key"] for t in job.get("tasks", [])}
    return jobs


def test_every_bundle_run_in_the_scripts_names_a_real_job_and_tasks():
    jobs = bundle_jobs()
    runs = [(m["job"], m["only"]) for p in sorted((ROOT / "scripts").glob("*.sh"))
            for m in re.finditer(r"bundle run (?P<job>[\w-]+)(?:[^\n;&|]*?--only (?P<only>[\w,]+))?", p.read_text())]
    assert len(runs) >= 5  # the parser found pre_demo.sh's runs: it isn't passing vacuously
    for job, only in runs:
        assert job in jobs, f"scripts run {job}, which no bundle defines"
        assert not only or set(only.split(",")) <= jobs[job], f"{job} has no task {only}"


def test_the_scripts_app_name_is_the_bundles():
    (app,) = [a["name"] for p in ROOT.glob("app/resources/*.yml")
              for a in ((yaml.safe_load(p.read_text()) or {}).get("resources", {}).get("apps") or {}).values()]
    named = [m.group(1) for p in (ROOT / "scripts").iterdir() if p.suffix in (".sh", ".py")
             for m in re.finditer(r"""^APP\s*=\s*["']?([\w-]+)""", p.read_text(), re.M)]
    assert len(named) >= 3 and set(named) == {app}, named
