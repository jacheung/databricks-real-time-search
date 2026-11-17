# Databricks notebook source
# DBTITLE 1,SERP query embedding endpoint (custom GPU)
# MAGIC %md
# MAGIC # SERP query embedding endpoint (custom GPU)
# MAGIC
# MAGIC Serves `models.query_embedding` (gte-large-en-v1.5, the same model as `ai_search.embedding_endpoint`) on a
# MAGIC dedicated GPU endpoint. The app embeds each search query here and passes `query_vector` (with `query_text` for
# MAGIC the keyword half of hybrid) instead of letting AI Search call the shared pay-per-token endpoint, whose tail
# MAGIC latency is wide (serving-bench: p95 853 ms at concurrency 1). Same model, so the index needs no rebuild.
# MAGIC
# MAGIC Serves the version `query_embedding_model` registered in this job run, or the latest registered version when
# MAGIC this task runs alone (logged on Serverless GPU with Express `env_pack`, raw vectors); this notebook only serves
# MAGIC it, so it runs on serverless CPU.
# MAGIC
# MAGIC Check: ready and serving that version. Route-optimized endpoints accept only OAuth tokens, which notebook
# MAGIC credentials can't provide, so vector parity with pay-per-token is checked by
# MAGIC `ml/scripts/check_query_embedding.py` (SDK + CLI OAuth login) and latency from the app.
# MAGIC
# MAGIC Bills while it exists and can't be stopped, only deleted: delete it when not testing.
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
from rts_shared.config import load_config, model_name

from databricks.sdk import WorkspaceClient

dbutils.widgets.text("config_path", f"{ROOT}/shared/configs/dev.yml")  # jobs pass configs/<bundle target>.yml
CFG = load_config(dbutils.widgets.get("config_path"))
QE = CFG["models"]["query_embedding"]
MODEL_NAME = model_name(CFG, "query_embedding")
w = WorkspaceClient()

# COMMAND ----------

# DBTITLE 1,Serve this run's model version; check: ready and serving it (Express deployment: ~5 min, no image build)
version = version_to_serve(w, MODEL_NAME, dbutils.jobs.taskValues.get(taskKey="query_embedding_model", key="version",
                                                                    default="", debugValue=""))
summary = {**deploy(w, QE, MODEL_NAME, version), "workload_type": QE["workload_type"]}
print(json.dumps(summary, indent=2))
dbutils.notebook.exit(json.dumps(summary))
