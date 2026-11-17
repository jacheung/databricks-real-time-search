# Databricks notebook source
# DBTITLE 1,Tear down the clickstream features and the Kafka Stream
# MAGIC %md
# MAGIC # Tear down the clickstream features and the Kafka Stream
# MAGIC
# MAGIC Deletes, in this order, everything `kafka_stream` and `feature_views` create from config:
# MAGIC 1. Each feature's materializations (offline and online), then the feature.
# MAGIC 2. The Feature Engineering Stream (`kafka.stream`).
# MAGIC
# MAGIC Run it before `clickstream_job` with `rebuild_data=true`: rebuilding replaces `session_events`, which breaks
# MAGIC live materializations reading it. Tables, the online store, the Kafka connection and the topic are kept.
# MAGIC
# MAGIC With `confirm=no` (the default) it only lists what it would delete.
# MAGIC
# MAGIC **Compute:** Serverless, environment version 4+.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install "databricks-feature-engineering>=0.16.0" pyyaml -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load config
import json
import sys

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
ROOT = "/Workspace" + "/".join(_nb_path.split("/")[:-3])  # this notebook is <ROOT>/data/notebooks/<name>
sys.path[:0] = [f"{ROOT}/shared/src", f"{ROOT}/data/src"]
from rts_data.teardown import teardown
from rts_shared.config import full_name, load_config
from rts_shared.features import feature_specs

from databricks.feature_engineering import FeatureEngineeringClient

dbutils.widgets.text("config_path", f"{ROOT}/shared/configs/dev.yml")  # jobs pass configs/<bundle target>.yml
dbutils.widgets.dropdown("confirm", "no", ["no", "yes"])  # yes = delete; no = list only
CFG = load_config(dbutils.widgets.get("config_path"))
DELETE = dbutils.widgets.get("confirm") == "yes"
STREAM = full_name(CFG, CFG["kafka"]["stream"])
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Features (materializations first), then the Stream (rts_data.teardown)
result = teardown(fe, [full_name(CFG, spec.name) for spec in feature_specs(CFG)], STREAM, DELETE)

# COMMAND ----------

# DBTITLE 1,Summary
print(json.dumps(result, indent=2))
assert not result["errors"], f"teardown errors: {result['errors']}"
dbutils.notebook.exit(json.dumps(result))
