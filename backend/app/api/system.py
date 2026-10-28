"""System endpoints: health, hardware capabilities, pipeline stage metadata."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app import __app_version__
from backend.app.config import capabilities, settings
from backend.app.services.job_manager import STAGES

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __app_version__, "dataDir": str(settings.data_dir)}


@router.get("/system/capabilities")
def system_capabilities(refresh: bool = Query(False)) -> dict:
    return capabilities(force=refresh)


@router.get("/system/stages")
def pipeline_stages() -> list[dict]:
    return STAGES
