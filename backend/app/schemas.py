"""Pydantic schemas: the explicit API contract shared with the TypeScript frontend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["draft", "queued", "processing", "completed", "failed", "canceled"]
StageId = Literal[
    "upload", "validation", "preprocessing", "feature_extraction", "matching",
    "sparse_reconstruction", "point_processing", "training", "evaluation", "export",
]
ExecutionMode = Literal["cpu", "gpu", "colab"]
Preset = Literal["fast_preview", "balanced", "high_quality"]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    mode: ExecutionMode = "cpu"
    preset: Preset = "balanced"


class ReconstructRequest(BaseModel):
    preset: Preset | None = None
    mode: ExecutionMode | None = None
    eval_fraction: float = Field(default=0.1, ge=0.0, le=0.4)


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    status: Status
    mode: ExecutionMode
    preset: Preset | None = None
    error: str | None = None
    isDemo: bool = False
    imageCount: int = 0
    stats: dict[str, Any] | None = None
    createdAt: str
    updatedAt: str


class ImageQuality(BaseModel):
    blur_score: float | None = None
    is_likely_blurry: bool = False
    is_duplicate_of: str | None = None
    too_dark: bool = False
    warnings: list[str] = []


class ImageOut(BaseModel):
    id: str
    filename: str
    width: int | None = None
    height: int | None = None
    fileSize: int | None = None
    split: str = "train"
    quality: ImageQuality = ImageQuality()
    url: str


class JobOut(BaseModel):
    id: str
    projectId: str
    kind: str
    status: Literal["queued", "running", "completed", "failed", "canceled"]
    stage: StageId | None = None
    progress: float = 0.0
    error: str | None = None
    startedAt: str | None = None
    finishedAt: str | None = None


class LogEntry(BaseModel):
    id: int
    ts: str
    level: str
    message: str


class ArtifactOut(BaseModel):
    kind: str
    filename: str
    format: str
    sizeBytes: int
    url: str


class MetricsBlock(BaseModel):
    kind: str
    data: dict[str, Any]
    createdAt: str


class ProjectDetail(ProjectOut):
    images: list[ImageOut] = []
    job: JobOut | None = None
    artifacts: list[ArtifactOut] = []
    metrics: list[MetricsBlock] = []
    availableStages: list[StageId] = []


class SseEvent(BaseModel):
    type: Literal["stage", "progress", "log", "metrics", "status", "done", "error"]
    projectId: str
    data: dict[str, Any]


class HealthOut(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    dataDir: str
