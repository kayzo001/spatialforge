"""Single-worker background job execution.

Design constraints (per requirements):
- heavy jobs must never block HTTP handling
- only ONE reconstruction/training job runs at a time (GPU/CPU resource
  locking); additional submissions queue FIFO
- cancellation is cooperative: the pipeline checks a threading.Event between
  stages and passes it to subprocess runners, which terminate process trees
- all state transitions are persisted, so status survives restarts (a job
  found 'running' at startup is marked failed as unknown-interrupt)
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from typing import Any

from backend.app import db
from backend.app.services.event_bus import bus
from backend.app.utils.logs import get_logger

logger = get_logger("spatialforge.jobs")

JobFn = Callable[["JobContext"], Any]

STAGES: list[dict[str, str]] = [
    {"id": "upload", "label": "Upload"},
    {"id": "validation", "label": "Validation"},
    {"id": "preprocessing", "label": "Preprocessing"},
    {"id": "feature_extraction", "label": "Feature Extraction"},
    {"id": "matching", "label": "Matching"},
    {"id": "sparse_reconstruction", "label": "Sparse Reconstruction"},
    {"id": "point_processing", "label": "Point Processing"},
    {"id": "training", "label": "Neural Training"},
    {"id": "evaluation", "label": "Evaluation"},
    {"id": "export", "label": "Export"},
]


class JobContext:
    """Passed to pipeline functions: progress reporting + cancellation."""

    def __init__(self, job_id: str, project_id: str, cancel_event: threading.Event):
        self.job_id = job_id
        self.project_id = project_id
        self.cancel_event = cancel_event
        self.logs: list[tuple[str, str]] = []

    # -- progress reporting -------------------------------------------------
    def stage(self, stage_id: str, message: str | None = None) -> None:
        db.update_job(self.job_id, stage=stage_id)
        bus.publish(self.project_id, "stage", {"stage": stage_id, "message": message})
        self.log(message or f"Stage: {stage_id}")

    def progress(self, value: float, message: str | None = None) -> None:
        value = max(0.0, min(1.0, value))
        db.update_job(self.job_id, progress=value)
        bus.publish(self.project_id, "progress", {"progress": value, "message": message})

    def log(self, message: str, level: str = "info") -> None:
        ts = db.add_log(self.project_id, level, message, self.job_id)
        bus.publish(self.project_id, "log", {"level": level, "message": message, "ts": ts})
        logger.info("[%s] %s", self.project_id, message)

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise JobCanceled()


class JobCanceled(RuntimeError):
    pass


class JobManager:
    _instance: JobManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, str, JobFn]] = queue.Queue()
        self._cancel_events: dict[str, threading.Event] = {}
        self._worker: threading.Thread | None = None
        self._started = False

    @classmethod
    def instance(cls) -> JobManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._recover_orphans()
        self._worker = threading.Thread(target=self._run_loop, name="spatialforge-jobs",
                                        daemon=True)
        self._worker.start()

    def _recover_orphans(self) -> None:
        # Jobs found mid-flight at startup can no longer be running.
        for row in db.query(
            "SELECT id, project_id FROM jobs WHERE status IN ('queued','running')"
        ):
            db.update_job(row["id"], status="failed", error="Interrupted by server restart",
                          finished_at=db._now())
            db.update_project(row["project_id"], status="failed",
                              error="Interrupted by server restart")

    def submit(self, project_id: str, kind: str, fn: JobFn) -> dict[str, Any]:
        active = db.active_job(project_id)
        if active is not None:
            raise RuntimeError("A job is already queued or running for this project")
        job = db.create_job(project_id, kind)
        db.update_project(project_id, status="queued", error=None)
        self._queue.put((project_id, job["id"], fn))
        bus.publish(project_id, "status", {"status": "queued", "jobId": job["id"]})
        return job

    def request_cancel(self, project_id: str) -> bool:
        event = self._cancel_events.get(project_id)
        if event is None:
            return False
        event.set()
        return True

    def _run_loop(self) -> None:
        while True:
            project_id, job_id, fn = self._queue.get()
            cancel_event = threading.Event()
            self._cancel_events[project_id] = cancel_event
            ctx = JobContext(job_id, project_id, cancel_event)

            db.update_job(job_id, status="running", started_at=db._now())
            db.update_project(project_id, status="processing")
            bus.publish(project_id, "status", {"status": "processing", "jobId": job_id})
            try:
                fn(ctx)
            except JobCanceled:
                db.update_job(job_id, status="canceled", finished_at=db._now())
                db.update_project(project_id, status="canceled")
                bus.publish(project_id, "status", {"status": "canceled", "jobId": job_id})
                ctx.log("Job canceled", "warning")
            except Exception as exc:  # noqa: BLE001 - job boundary
                logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
                db.update_job(job_id, status="failed", error=str(exc), finished_at=db._now())
                db.update_project(project_id, status="failed", error=str(exc))
                bus.publish(project_id, "status", {"status": "failed", "error": str(exc)})
                ctx.log(f"Job failed: {exc}", "error")
            else:
                db.update_job(job_id, status="completed", progress=1.0,
                              finished_at=db._now())
                db.update_project(project_id, status="completed", error=None)
                bus.publish(project_id, "status", {"status": "completed", "jobId": job_id})
                bus.publish(project_id, "done", {})
            finally:
                self._cancel_events.pop(project_id, None)


def get_manager() -> JobManager:
    mgr = JobManager.instance()
    mgr.start()
    return mgr
