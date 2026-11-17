"""FastAPI app: QAC from the Lakebase prefix table, SERP from the AI Search index, demo scenarios replayed to Kafka,
and the React UI.

Wiring only: the routes are in rts_app.app, the clients they use in rts_app.clients. Both serving endpoints are
route-optimized; their first call opens a connection (~400 ms), so startup warms them in the background.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from databricks.sdk import WorkspaceClient
from fastapi.staticfiles import StaticFiles

from rts_app.app import create_app
from rts_app.clients import Clients
from rts_shared.config import load_config

ROOT = Path(__file__).resolve().parents[3]  # app/backend/rts_app/main.py -> source root
CFG = load_config(str(ROOT / os.environ.get("RTS_CONFIG", "shared/configs/dev.yml")))
logging.basicConfig(level=logging.INFO)

backend = Clients(CFG, WorkspaceClient())
app = create_app(backend, CFG)


def _warm():
    try:
        backend.embed("warm up")
        backend.qac_l2(backend.prefix_rows("ca")[:1], None, None)
        backend.serp_l2(backend.query_index(query_type="ANN", query_vector=backend.embed("warm up"), num_results=1),
                        None, None)
    except Exception:
        logging.exception("endpoint warm-up failed; the first requests pay the connection cost")


threading.Thread(target=_warm, daemon=True).start()

_dist = ROOT / "app" / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
