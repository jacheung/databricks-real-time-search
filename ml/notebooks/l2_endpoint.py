# Databricks notebook source
# DBTITLE 1,L2: serving endpoint (QAC or SERP)
# MAGIC %md
# MAGIC # L2: serving endpoint (QAC or SERP)
# MAGIC
# MAGIC Serves the version this job run registered (a task value from `train_task`; the latest registered version when
# MAGIC this task runs alone) of `models.<model_key>.registered_name` on `models.<model_key>.endpoint`
# MAGIC (route-optimized, workload type and size from config). `variant=lgbm` serves the SERP LightGBM comparator from
# MAGIC `models.serp_l2.lgbm` instead (CPU, for the latency comparison). The models were logged with `fe.log_model`, so
# MAGIC the endpoint looks up the session (Kafka stream), user and item (Delta) features in the online store itself;
# MAGIC callers send only keys and the request fields (`rts_shared.retrieval.qac_l2_records` / `serp_l2_records`).
# MAGIC
# MAGIC Check: the endpoint is ready and serving that version. Queries are checked elsewhere: route-optimized endpoints
# MAGIC accept only OAuth tokens, which notebook credentials can't provide. The app (its service principal uses OAuth)
# MAGIC and a local SDK client (CLI OAuth login) can query it.
# MAGIC
# MAGIC Reruns are safe: an endpoint already serving that version is reused. The endpoint bills while it exists
# MAGIC (`scale_to_zero_enabled` from config).
# MAGIC
# MAGIC **Compute:** Serverless, environment version 4+.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install "databricks-sdk>=0.81" pyyaml -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load config
import json
import sys

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
ROOT = "/Workspace" + "/".join(_nb_path.split("/")[:-3])  # this notebook is <ROOT>/ml/notebooks/<name>
sys.path[:0] = [f"{ROOT}/shared/src", f"{ROOT}/ml/src"]
from rts_ml.serving import deploy, version_to_serve
from rts_shared.config import full_name, load_config, model_name

from databricks.sdk import WorkspaceClient

dbutils.widgets.text("config_path", f"{ROOT}/shared/configs/dev.yml")  # jobs pass configs/<bundle target>.yml
dbutils.widgets.dropdown("model_key", "qac_l2", ["qac_l2", "serp_l2"])
dbutils.widgets.dropdown("variant", "", ["", "lgbm"])  # lgbm: the SERP LightGBM comparator
dbutils.widgets.text("train_task", "")  # the job task that registered the model; empty: serve the latest version
CFG = load_config(dbutils.widgets.get("config_path"))
KEY, VARIANT, TRAIN_TASK = (dbutils.widgets.get(k) for k in ("model_key", "variant", "train_task"))
L2 = CFG["models"][KEY]
MODEL_NAME = model_name(CFG, KEY)
if VARIANT:
    assert KEY == "serp_l2", f"variant {VARIANT!r} exists only for serp_l2"
    L2 = {**L2, **L2[VARIANT]}  # its own name, endpoint and workload; route optimization etc. from serp_l2
    MODEL_NAME = full_name(CFG, L2["registered_name"])
w = WorkspaceClient()

# COMMAND ----------

# DBTITLE 1,Serve this run's model version; check: ready and serving it
run_version = dbutils.jobs.taskValues.get(taskKey=TRAIN_TASK, key=f"{VARIANT}_version" if VARIANT else "version",
                                         default="", debugValue="") if TRAIN_TASK else ""
summary = deploy(w, L2, MODEL_NAME, version_to_serve(w, MODEL_NAME, run_version))
print(json.dumps(summary, indent=2))
dbutils.notebook.exit(json.dumps(summary))
