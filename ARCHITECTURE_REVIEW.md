# Architecture review (2026-11-17)

Places where a small interface could hide more of the work (deepening opportunities), found by one review across all
tracks, plus the order to do them in. Branch `architecture`. Each track ticks off its own section here; status stays in
[PLAN.md](PLAN.md).

## Resume here (updated 2026-11-17, end of session 1)

- **Where things are:** phase 1 (#1, #2) and the app track (bugs, #3, #4) are merged into `main` and pushed
  (`6624409`). Work happens in the worktree `~/GitHub/databricks-real-time-search-arch` on branch `architecture`
  (at `main` plus this note). One track at a time, in this worktree only.
- **Next:** the ml track (bugs, #6, #5) is done on `architecture`, unmerged. Open: training on time-shifted history (Bugs) blocks
  retraining on fevm until the data is rebuilt. QAC v2 is registered, not deployed (v1 serves). Then data: the
  `clickstream_sim` bug, then #8.
- **How we work (agreed with the user):**
  - Present the design choices before coding, test first, one commit per item.
  - Ask before merging into `main` or pushing. The repo is public.
  - Report results in numbers (lines, modules, tests), not just a description.
- **Tests:**
  - Python, from the repo root: `uv run --no-project --with pytest --with pyyaml --with pandas --with numpy --with
    lightgbm --with scikit-learn --with mlflow-skinny --with databricks-sdk --with fastapi --with httpx python -m
    pytest -q` (214 pass).
  - Torch tests: a separate process with `--with torch ... ml/tests_torch`.
  - Frontend: `cd app/frontend && npm test` (15 pass).
- **Live checks:** on fevm, profile `fevm-jon`, bundle target `fevm`. The `e2-demo-fe` login has expired.
  - Copy the gitignored `shared/configs/{dev,fevm}.yml` from the main checkout before deploying.
  - As of 2026-11-17 the app, all three endpoints and the session pipeline are **running** (the user chose to leave
    them on). Ask before turning anything off, and say so when testing is done.
  - App routes are checked over HTTP with `WorkspaceClient(profile=...).config.authenticate()` headers, as
    `scripts/pre_demo_checks.py` (`_app_call`) does.
- **Measured so far** (`32ef8db` → `6624409`, the user's load test excluded):
  - Production code: +815 / −630 lines.
  - `main.py`: 455 → 44 lines. `App.jsx`: 990 → 901 lines.
  - Python tests: 169 → 204. Frontend tests: 0 → 15. Routes with a test: 0 → 13 of 16.
  - Hardcoded feature names in the app backend: 16 → 5. Places that call an L2 endpoint directly: 4 → 2.

## Order

**Phase 1: `shared/`, in the `architecture` worktree, one candidate at a time (PLAN.md rule 1).** The cross-track
candidates rewrite code inside every track, so they land before the track work starts.
- [x] #1 Feature catalog, with every consumer moved onto it in the same commit. Feature names stay exactly as they are
  (a test asserts it): a rename means re-materializing Feature Views and redeploying endpoints.
  Decided (2026-11-17): the whole spec moves to `rts_shared.features` (the app can't import `rts_data`). The app
  computes its session tables and feature lists from it; ml keeps its model input columns as literals (a trained model
  freezes them), guarded by tests. Still literal on purpose: the expected values in the app's freshness check (it
  restates what each feature counts; names are tested), the `sess_last_event_type` override (open ask to data),
  notebook checks, `demo_personas`, `scripts/_common.sh`.
- [x] #2 L2 request contract. After #1: its "model input columns ⊆ catalog + request columns" test needs the catalog.
  Decided (2026-11-17): `rts_shared.retrieval` owns the whole call (`l2_scores`, `qac_l2_scores`, `serp_l2_scores`,
  taking `w` like `rts_shared.kafka`): request, endpoint query, one numeric score per candidate or `ValueError`. The
  app keeps its timing, logging and timeouts around it; both ml benchmarks use it. The column tuples sent to the
  endpoint are unchanged.
- [x] ~~#7 Event shape~~: dropped (2026-11-17); see #7 below.
- [x] Merge into `main` (the user approves this). Merged 2026-11-17 (`7323240`), with the QAC L2 load test.

**Phase 2: one track at a time in the `architecture` worktree (decided 2026-11-17: live checks serialize anyway).
Order: app, then ml, then data. Originally: one worktree per track, branched from `main`, in parallel.** PLAN.md's track worktree names
(`databricks-blueprints-rts-app`/`-ml`) predate the repo split: reuse them and merge `main` if they still exist.
- app: bug commits (done), then #3 (done), then #4 (done; not yet deployed). Live on fevm 2026-11-17 (app bundle, `bundle run rts_app`):
  every route 200, QAC L2 36 ms, SERP L2 96 ms, Play → freshness ready in 820 ms, persona check ok for all three.
  Browser check of the UI (QAC error message) still to do.
- ml: bug commits (done), #6 (done), then #5.
- data: bug commits, then #8.

**Bugs:** each is its own commit in its track's worktree, before that track's refactor, test first (so it ships even if
the refactor stalls; the refactor keeps the test).

**Limits on parallel work:**
- Only one worktree at a time deploys a given bundle (rule 3). Refactors and unit tests run in parallel freely.
- Integration checks (browser, scenario Play, persona check) run after merging into `main` (rule 5). Tracks merge back
  one at a time; the others merge `main` before continuing.
- Live checks need the endpoints, the session pipeline and Lakebase on: batch them, then turn things off.

## Candidates

### Cross-track (`shared/`)

**#1 Feature catalog** (owner: data; used by ml and app)
- Files: `data/src/rts_data/feature_specs.py`, `shared/src/rts_shared/features.py`, `data/notebooks/feature_views.py`,
  `data/notebooks/demo_personas.py`, `ml/src/rts_ml/qac_l2.py`, `ml/src/rts_ml/serp_l2.py`,
  `app/backend/rts_app/service.py`, `app/backend/rts_app/main.py`, `scripts/_common.sh`.
- Problem: names are built twice (`feature_specs()`, `feature_keys()`), kept equal only by a test. `_30m`/`_30d`/`_7d`
  names are hardcoded in both models' input columns, the app's freshness expectations and a shell script. Source and
  window per group, and the `rts_{group}_online` table names, live only in the `feature_views` notebook. "Engaged" is
  defined three times. A window change in config passes every test and breaks serving.
- Solution: one module owns each feature's name, keys, group, window and online table; everything else reads it.
- Benefits: a rename or window change is one edit; a test can check each model's input columns against the catalog.
- Open: this reopens `features.py`'s "names only, no definitions" contract. Decide how much of the definition moves.

**#2 L2 request contract** (owner: ml; used by app)
- Files: `shared/src/rts_shared/retrieval.py`, `app/backend/rts_app/main.py`, `app/backend/rts_app/service.py`.
- Problem: `qac_l2_records`/`serp_l2_records` slice column tuples by position (`[1:]`, `[2:]`), so inserting a column
  changes the request silently. SERP checks the endpoint returns one score per hit; QAC doesn't. `SERP_INDEX_COLUMNS`
  omits the `score` that `serp_l2_records` reads.
- Solution: one module per surface owns the whole L2 call: request from L1 rows, response checks, scores in L1 order.
- Benefits: same checks on both surfaces; the request format is tested in one place against a fake endpoint.

**#7 Event shape** (owner: data; touches `shared/events.py`). **Dropped (2026-11-17); don't re-suggest unless the
reasons below stop holding.**
- Why dropped: the Spark writer's `'Z'` is true because `kafka_stream.py` sets `spark.sql.session.timeZone` to UTC,
  so both writers emit the same ms UTC string. `shared/` can't depend on pyspark (the app ships it without Spark),
  and the two patterns are in different formatting languages, so a shared constant removes nothing. Event ids only
  need to be unique (dedup keys on them; nothing parses them). The two scenario cut-offs serve different purposes:
  `demo_scenarios` stops before the search QAC/SERP predicts; Play stops at the last click or cart so one event
  times all three session tables (app logic, #3). The one real duplicate is a data bug below.
- Files: `shared/src/rts_shared/events.py`, `data/notebooks/kafka_stream.py`, `data/src/rts_data/datagen.py`,
  `shared/src/rts_shared/scenarios.py`.
- Problem: Kafka JSON is written twice, by `event_payload` (tested) and a Spark `date_format` (untested); ms precision
  was already a bug once (2026-10-30). The event id format is copied three times. Scenario cut-off rules are split
  between `demo_scenarios` and the app's `play_scenario`.
- Solution: one module owns the event shape for Python and Spark, ids and timestamps included.
- Benefits: one round-trip test covers simulator → Kafka → scenario replay.

### app

**#3 Request handling in `main.py`** (done 2026-11-17). `create_app(backend, cfg)` in `rts_app/app.py` holds every
route; `Backend` (14 methods: Lakebase reads, serving, AI Search, Kafka) is the seam, with `rts_app/clients.py` as
the real adapter and an in-memory fake in `app/tests/test_app.py`. QAC and SERP each get their own L2 pool. The
persona check stays synchronous and off the pools without timeouts (it checks lift; ~150 calls at QAC's 36 ms
timeout would nearly always fail), and its SERP L1 now goes through `search()` with the embed fallback.
- Files: `app/backend/rts_app/main.py` (no tests), `app/backend/rts_app/service.py`, `app/tests/test_service.py`.
- Problem: importing `main.py` creates a `WorkspaceClient`, lists Lakebase endpoints, opens pools and starts a thread,
  so it can't be tested. `check_personas` rebuilds SERP L1 and calls L2 with no timeout (~38 × (1 + personas) calls).
  QAC and SERP share one 8-worker `L2_POOL`, so a slow SERP call eats into QAC's 36 ms timeout. Six endpoints each
  hold their own Lakebase SQL.
- Solution: QAC, SERP, Play → freshness and the persona check behind one seam with two adapters (real clients, and
  fakes built from `test_service.py`'s); `main.py` shrinks to wiring.
- Benefits: the freshness loop, the real timeout path, 404/503 paths and the persona check become testable.

**#4 QAC and SERP fetching and timing in `App.jsx`** (done 2026-11-17). `useQac` and `useSearch` in
`src/search.js` own fetch, stale-response drops, errors and timing (`qacTiming` / `serpTiming`); Vitest covers them
with a fake `fetch`. `PAGES` is gone (pages follow the response), and stages are matched by name.
- Problem: stale-response handling written twice; QAC has no error path; server timings re-summed in four places;
  `PAGES = 4` must match `SERP_N / PAGE_SIZE`; stage names matched by index; no frontend tests.
- Solution: one hook per surface returning result, status and timings; rendering only reads them.
- Benefits: sequencing and errors in one place, testable with a fake `fetch`. Lower leverage than #3.

### ml

**#5 L2 training sets**
- Files: `ml/src/rts_ml/qac_l2.py`, `ml/src/rts_ml/serp_l2.py`, `ml/notebooks/qac_l2_lgbm.py`,
  `ml/notebooks/serp_l2_training_set.py`, `ml/notebooks/serp_l2_dcnv2.py`, `ml/tests_torch/test_serp_l2_dcn.py`.
- Problem: everything before the shared feature → matrix transform is untested notebook code (labels, joins, splits,
  packing; feature list copied into three notebooks). The DCN notebook logs its signature from a second training set.
  `add_crosses` is copied; the shared L2 helpers live in `qac_l2.py`. QAC and SERP made opposite calls on
  `sess_last_event_type`, and nothing asserts QAC's is safe. The torch test re-implements the packing.
- Solution: one module per surface builds the training set; notebooks become thin callers.
- Benefits: labels, the L1 join and packing become testable; the torch test runs production code.
- Done (2026-11-17), unmerged. Decided: Spark builders move to modules but are checked live only (no JVM locally);
  QAC drops `sess_last_event_type`; the live check runs the training tasks only. Modules: `rts_ml.l2` (shared
  crosses, LambdaRank, report), `rts_ml.l2_training` (training features, evaluation slices), `qac_l2_training`,
  `serp_l2_training` (spines and labels); `serp_l2.pack_searches` / `unpack`, `qac_l2.fit_valid_split` (tested).
  Lines: notebooks 684 → 571 (−113), modules 212 → 371 (+159), net +46: not the cut expected; the gain is that the
  feature list, slices, packing and splits exist once and the torch test runs production packing. Tests 211 → 213.
  Live on fevm: QAC v2 trained: same groups (66,934 / 7,240 / 29,973), same L1 metrics and ceiling (0.819) as v1;
  MRR lift +0.0578 (v1 +0.0584), success@1 +0.0629 (v1 +0.063). Not deployed. SERP training set ran; `serp_l2_dcnv2`
  failed its raw-vs-packed parity check (max diff 2.35), so no new SERP version. A diagnostic showed the stored matrix
  equals the full-spine training set bit for bit (the refactor is not the cause), but Feature Engineering returns
  different user/item (Delta, sliding-window) features for a 5-search spine than for the full spine, and SQL counts
  from `session_events` side with neither consistently (user clicks: small spine right; item clicks: full spine right).
  Cause: `session_events` was time-shifted after the labels (see Bugs); the check caught it.

**#6 L2 endpoint deploys**
- Files: `ml/src/rts_ml/serving.py`, `ml/notebooks/l2_endpoint.py`, `shared/src/rts_shared/lakebase.py`,
  `ml/notebooks/serp_l1_ai_search.py`.
- Problem: `ensure_endpoint` compares only model name, version and scale-to-zero, so a workload size or type change
  reports "unchanged". `l2_endpoint` deploys the highest registered version, not this run's. The wait-until-ready loop
  is written three times.
- Solution: one module takes this run's model and the config and makes the endpoint match.
- Benefits: `test_serving.py`'s fakes cover the whole interface.
- Done (2026-11-17). Decided: `rts_ml.serving.deploy(w, block, model_name, version)` builds the served entity from
  config, ensures, waits and checks the served version; both endpoint notebooks call it. A `route_optimized` mismatch
  raises (it can't change in place; recreating means L2 downtime). The three wait loops stay separate: they poll
  different resources and states. Lines: `serving.py` 56 → 76, `l2_endpoint.py` 82 → 66, `serp_query_embedding.py`
  73 → 59 (net −10). Live on fevm: `--only` QAC L2 and query embedding both "unchanged".

### data

**#8 Demo operations logic**
- Files: `data/notebooks/time_shift.py`, `teardown.py`, `demo_scenarios.py`, `demo_personas.py`,
  `scripts/pre_demo.sh`, `scripts/pre_demo_checks.py`, `scripts/_common.sh`.
- Problem: scenario and persona selection rules and the orphan filter are untested notebook code. `teardown` treats any
  error as "absent". Scripts hardcode job and task keys that must match `ml/resources/*.yml`. `pre_demo_checks` and
  `time_shift` disagree on "fresh" (UTC vs local date).
- Solution: selection and freshness rules into `rts_data`; scripts read job names from the bundles or config.
- Benefits: lower leverage, but it's the code that broke in the online store purge.

## Bugs (fix directly, test first, one commit each)

- [x] app: `_online_tables` fills its cache key by key, then raises 503 on a later key; the next call returns the
  partial cache and freshness raises `KeyError` (500). `main.py:151-167`.
- [x] app: `bench_qac_l1` defined twice (removed in the load test commit); `import itertools` twice. `main.py`.
- [x] app: QAC doesn't check the endpoint returned one score per candidate. `service.py:38`. Fixed by #2.
- [x] app: QAC fetch has no error path; a 500 body is stored as the result and the render throws. `App.jsx:736-741`.
- [x] ml: `ensure_endpoint` ignores workload size and type. `serving.py:27-30`.
- [x] ml: `l2_endpoint` deploys the highest registered version, not this run's. `l2_endpoint.py:62`.
  Decided (2026-11-17): training tasks set task value `version` (`lgbm_version`); endpoint tasks read it from
  `train_task` with `default=""` and serve the latest when run alone. Same fix in `serp_query_embedding`. Live on
  fevm: `--only qac_l2_endpoint` fell back to v1, "unchanged". The task-value path itself runs on the next retrain.
- [ ] ml/data: training ran on time-shifted history (found 2026-11-17 by `serp_l2_dcnv2`'s parity check). `time_shift`
  moved `session_events` +120 h (UPDATEs 2026-11-16 23:37 and 2026-11-17 18:25 UTC); labels and the Kafka copy keep
  their dates, which `time_shift.py` says means don't retrain. Training sets over the full spine compute user/item
  features from the shifted source (5 days off their labels); small spines inside the materialized range (window ends
  2026-10-28 onward) read the pre-shift offline tables (inferred from one user and one item: offline 3 = unshifted
  count 3, full spine 6 = shifted count). QAC v2 (registered today, not deployed) trained on the shifted features;
  nothing in QAC checked. Not an FE bug and not #5. Diagnostic: `real-time-search/scratch/serp_l2_parity_diag`.
  Done 2026-11-17: QAC v2 deleted (v1 serves); `rts_ml.l2_training.require_unshifted_history` (pure part
  `rts_shared.config.history_shift`, tested) stops both training notebooks first; live on fevm it fails with
  "shifted 5 days". Still open (the user's call: not now): rebuild the data before the next retrain.
- [ ] data: `clickstream_sim.py:90` redefines the required event fields instead of importing `REQUIRED_FIELDS`.
