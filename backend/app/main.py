"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app import __app_version__, db
from backend.app.api import projects as projects_api
from backend.app.api import system as system_api
from backend.app.config import settings
from backend.app.services.event_bus import bus
from backend.app.services.job_manager import get_manager
from backend.app.utils.logs import get_logger

logger = get_logger("spatialforge.main")

# backend/app/main.py -> parents[0]=backend/app, [1]=backend, [2]=repo root
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus.attach_loop(asyncio.get_running_loop())
    db.get_conn()          # initialize schema
    get_manager().start()  # single background worker
    logger.info("SpatialForge %s ready (data: %s)", __app_version__, settings.data_dir)
    yield
    db.close_conn()


def create_app() -> FastAPI:
    app = FastAPI(
        title="SpatialForge API",
        description=(
            "Transform ordinary photographs into interactive 3D worlds: "
            "Structure-from-Motion reconstruction, Gaussian Splatting and "
            "novel-view visualization."
        ),
        version=__app_version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                       "http://localhost:4173", "http://127.0.0.1:4173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system_api.router)
    app.include_router(projects_api.router)

    # Serve the built frontend when it exists (production-style single port).
    if FRONTEND_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
                  name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            # API paths must never fall through to the SPA: unknown API routes
            # are errors, and traversal-ish paths stay out of the SPA handler
            if full_path.startswith("api/") or ".." in full_path:
                raise HTTPException(status_code=404, detail="Not found")
            candidate = FRONTEND_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
