"""FastAPI entrypoint.

Design notes:
* Routers are imported fail-fast (no try/except) — a broken module now
  prevents boot instead of silently serving a chatless API.
* Logging: configured once here (`REGURAG_LOG_LEVEL`, default INFO).
* Ops: request-timing middleware; `/api/health` reports app version; `/data`
  serves the documents directory read-only so `pdf_links` resolve.
"""

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.approved_api import router as approved_router
from backend.api.backup_api import router as backup_router
from backend.api.chat_api import router as chat_router
from backend.api.conflicts_api import router as conflicts_router

APP_VERSION = "0.2.0"

LOG = logging.getLogger("regurag")


def _setup_logging() -> None:
    level = os.environ.get("REGURAG_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


_setup_logging()

app = FastAPI(title="ReguRAG_MM Backend", version=APP_VERSION)

# Restrict CORS to the frontend origins actually served on loopback
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_timing(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    LOG.info("http %s %s -> %d %.0fms", request.method, request.url.path, response.status_code, (time.monotonic() - t0) * 1000)
    return response


@app.get("/api/health")
def api_health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/health")
def health_alias():
    return {"status": "ok"}


@app.get("/api/ingest/status")
def ingest_status():
    from backend.services.ingestion_pipeline import get_status

    return get_status()


@app.post("/api/documents/refresh")
def documents_refresh():
    from backend.services.ingestion_pipeline import start_refresh

    return start_refresh()


app.include_router(conflicts_router)
app.include_router(chat_router)
app.include_router(backup_router)
app.include_router(approved_router)


# Serve the documents directory (read-only) so pdf_links like
# /data/<file>#page=N resolve in the browser. Mounted only when the
# configured documents path exists.
def _mount_documents() -> None:
    try:
        from backend.core.config import get_config

        docs_path = Path(get_config().system.documents_path)
        if docs_path.exists():
            app.mount("/data", StaticFiles(directory=str(docs_path)), name="data")
            LOG.info("static /data mounted at %s", docs_path)
    except Exception:
        LOG.warning("static /data mount skipped", exc_info=True)


_mount_documents()
