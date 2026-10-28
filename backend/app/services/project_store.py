"""Serialization of DB rows into API contract shapes."""

from __future__ import annotations

import json
from typing import Any

from backend.app import db

STAGE_ORDER = [
    "upload", "validation", "preprocessing", "feature_extraction", "matching",
    "sparse_reconstruction", "point_processing", "training", "evaluation", "export",
]


def project_out(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "status": row["status"],
        "mode": row["mode"],
        "preset": row["preset"],
        "error": row["error"],
        "isDemo": bool(row["is_demo"]),
        "imageCount": row["image_count"],
        "stats": json.loads(row["stats_json"]) if row["stats_json"] else None,
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def image_out(row: dict[str, Any]) -> dict[str, Any]:
    quality = json.loads(row["quality_json"]) if row["quality_json"] else {}
    return {
        "id": row["id"],
        "filename": row["filename"],
        "width": row["width"],
        "height": row["height"],
        "fileSize": row["file_size"],
        "split": row["split"],
        "quality": quality,
        "url": f"/api/projects/{row['project_id']}/images/{row['id']}/file",
    }


def artifact_out(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": row["kind"],
        "filename": row["filename"],
        "format": row["format"],
        "sizeBytes": row["size_bytes"],
        "url": f"/api/projects/{row['project_id']}/artifacts/{row['kind']}/{row['filename']}",
    }


def job_out(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "kind": row["kind"],
        "status": row["status"],
        "stage": row["stage"],
        "progress": row["progress"],
        "error": row["error"],
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
    }


def available_stages(project_id: str) -> list[str]:
    """Stages that have real artifacts (drives frontend viewer modes)."""
    kinds = {a["kind"] for a in db.list_artifacts(project_id)}
    stages: list[str] = []
    if any(k in kinds for k in ("sparse_cloud", "refined_cloud", "cameras")):
        stages.append("sparse_reconstruction")
    if "refined_cloud" in kinds:
        stages.append("point_processing")
    if "splats" in kinds:
        stages.append("training")
    if "evaluation" in {m["kind"] for m in db.get_metrics(project_id)}:
        stages.append("evaluation")
    return stages


def project_detail(project_id: str) -> dict[str, Any] | None:
    row = db.get_project(project_id)
    if row is None:
        return None
    job = db.active_job(project_id) or db.latest_job(project_id)
    return {
        **project_out(row),
        "images": [image_out(r) for r in db.list_images(project_id)],
        "job": job_out(job) if job else None,
        "artifacts": [artifact_out(r) for r in db.list_artifacts(project_id)],
        "metrics": db.get_metrics(project_id),
        "availableStages": available_stages(project_id),
    }
