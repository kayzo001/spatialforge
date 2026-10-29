"""REST API routes for projects, uploads, jobs, artifacts and live events."""

from __future__ import annotations

import asyncio
import json
import shutil

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from backend.app import db
from backend.app.config import settings
from backend.app.reconstruction.pipeline import run_reconstruction
from backend.app.schemas import ProjectCreate, ReconstructRequest
from backend.app.services import project_store
from backend.app.services.event_bus import bus
from backend.app.services.job_manager import get_manager
from backend.app.services.upload_service import UploadError, ingest_uploads

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _get_project_or_404(project_id: str) -> dict:
    row = db.get_project(project_id)
    if row is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return row


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

@router.post("", status_code=201)
def create_project(body: ProjectCreate) -> dict:
    row = db.create_project(body.name, body.description, body.mode, body.preset)
    settings.ensure_project_dirs(row["id"])
    return project_store.project_out(row)


@router.get("")
def list_projects() -> list[dict]:
    return [project_store.project_out(r) for r in db.list_projects()]


@router.get("/{project_id}")
def get_project(project_id: str) -> dict:
    _get_project_or_404(project_id)
    detail = project_store.project_detail(project_id)
    if detail is None:
        raise HTTPException(404, "Project not found")
    return detail


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str) -> None:
    row = _get_project_or_404(project_id)
    if row["status"] in ("queued", "processing"):
        raise HTTPException(409, "Cancel the running job before deleting the project")
    db.delete_project(project_id)
    shutil.rmtree(settings.project_dir(project_id), ignore_errors=True)


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

_MAX_READ = settings.max_upload_mb * 1024 * 1024


@router.post("/{project_id}/images")
async def upload_images(
    project_id: str, files: list[UploadFile] = File(...)  # noqa: B008
) -> dict:
    _get_project_or_404(project_id)
    payload: list[tuple[str, bytes]] = []
    for f in files:
        if f.size is not None and f.size > _MAX_READ + 1024:
            raise HTTPException(413, f"{f.filename} exceeds {settings.max_upload_mb} MB")
        data = await f.read()
        if len(data) > _MAX_READ + 1024:
            raise HTTPException(413, f"{f.filename} exceeds {settings.max_upload_mb} MB")
        payload.append((f.filename or "unnamed", data))
    try:
        report = ingest_uploads(project_id, payload)
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    return report


# ---------------------------------------------------------------------------
# Reconstruction job
# ---------------------------------------------------------------------------

@router.post("/{project_id}/reconstruct", status_code=202)
def reconstruct(project_id: str, body: ReconstructRequest) -> dict:
    row = _get_project_or_404(project_id)
    if row["status"] in ("queued", "processing"):
        raise HTTPException(409, "A reconstruction job is already queued or running")
    if row["image_count"] < 3:
        raise HTTPException(400, "Upload at least 3 images before reconstructing")
    params = {
        "preset": body.preset or row["preset"] or "balanced",
        "mode": body.mode or row["mode"],
        "eval_fraction": body.eval_fraction,
    }
    db.update_project(project_id, preset=params["preset"], mode=params["mode"])

    def _job(ctx) -> None:
        project = db.get_project(project_id)  # type: ignore[arg-type]
        run_reconstruction(ctx, project, params)

    try:
        job = get_manager().submit(project_id, "reconstruction", _job)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"jobId": job["id"], "status": "queued"}


@router.post("/{project_id}/cancel", status_code=202)
def cancel(project_id: str) -> dict:
    _get_project_or_404(project_id)
    requested = get_manager().request_cancel(project_id)
    return {"cancelRequested": requested}


@router.get("/{project_id}/status")
def status(project_id: str) -> dict:
    row = _get_project_or_404(project_id)
    job = db.active_job(project_id) or db.latest_job(project_id)
    return {
        "project": project_store.project_out(row),
        "job": project_store.job_out(job) if job else None,
    }


@router.get("/{project_id}/metrics")
def metrics(project_id: str) -> list[dict]:
    _get_project_or_404(project_id)
    return db.get_metrics(project_id)


@router.get("/{project_id}/artifacts")
def artifacts(project_id: str) -> list[dict]:
    _get_project_or_404(project_id)
    return [project_store.artifact_out(r) for r in db.list_artifacts(project_id)]


@router.get("/{project_id}/artifacts/{kind}/{filename}")
def artifact_file(project_id: str, kind: str, filename: str) -> FileResponse:
    _get_project_or_404(project_id)
    if ".." in filename or "/" in filename or "\\" in filename or "\x00" in filename:
        raise HTTPException(400, "Invalid artifact filename")
    path = db.artifact_path(project_id, kind, filename)
    if path is None:
        raise HTTPException(404, "Artifact not found")
    # Re-runs overwrite artifacts in place under the same URL; force the
    # browser to revalidate (cheap: ETag/Last-Modified) instead of reusing
    # a stale point cloud or camera file.
    return FileResponse(path, filename=filename, headers={"Cache-Control": "no-cache"})


@router.get("/{project_id}/images/{image_id}/file")
def image_file(project_id: str, image_id: str) -> FileResponse:
    _get_project_or_404(project_id)
    path = db.image_path(project_id, image_id)
    if path is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(path, filename=path.name)


@router.get("/{project_id}/logs")
def logs(project_id: str, after_id: int = Query(0), limit: int = Query(500, le=2000)) -> dict:
    _get_project_or_404(project_id)
    rows = db.list_logs(project_id, after_id=after_id, limit=limit)
    return {
        "logs": [
            {"id": r["id"], "ts": r["ts"], "level": r["level"], "message": r["message"]}
            for r in rows
        ],
        "lastId": rows[-1]["id"] if rows else after_id,
    }


# ---------------------------------------------------------------------------
# Server-Sent Events
# ---------------------------------------------------------------------------

@router.get("/{project_id}/events")
async def events(project_id: str) -> StreamingResponse:
    _get_project_or_404(project_id)
    queue = await bus.subscribe(project_id)

    async def stream():
        try:
            yield f"data: {json.dumps({'type': 'connected', 'projectId': project_id})}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:  # client disconnected
            pass
        finally:
            bus.unsubscribe(project_id, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )
