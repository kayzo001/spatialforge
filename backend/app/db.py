"""SQLite persistence layer.

Uses the standard library sqlite3 with a small DAO surface. A single connection
guarded by a lock is sufficient for SpatialForge's single-worker workload and
keeps the dependency footprint minimal. All timestamps are UTC ISO strings.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.config import settings

_CONN: sqlite3.Connection | None = None
_LOCK = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    mode TEXT NOT NULL DEFAULT 'cpu',
    preset TEXT,
    error TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    image_count INTEGER NOT NULL DEFAULT 0,
    config_json TEXT,
    stats_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    sha256 TEXT,
    quality_json TEXT,
    split TEXT NOT NULL DEFAULT 'train',
    status TEXT NOT NULL DEFAULT 'original',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id);
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    job_id TEXT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_project ON logs(project_id, id);
CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, kind)
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    filename TEXT NOT NULL,
    format TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, kind, filename)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def get_conn() -> sqlite3.Connection:
    global _CONN
    with _LOCK:
        if _CONN is None:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            _CONN = sqlite3.connect(str(settings.db_path), check_same_thread=False)
            _CONN.row_factory = sqlite3.Row
            _CONN.execute("PRAGMA journal_mode=WAL")
            _CONN.execute("PRAGMA foreign_keys=ON")
            _CONN.executescript(SCHEMA)
            _CONN.commit()
        return _CONN


def close_conn() -> None:
    global _CONN
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
            _CONN = None


def execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    with _LOCK:
        conn = get_conn()
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with _LOCK:
        cur = get_conn().execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def query_one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Project helpers
# ---------------------------------------------------------------------------

def create_project(name: str, description: str, mode: str, preset: str | None,
                   is_demo: bool = False) -> dict[str, Any]:
    pid = new_id()
    now = _now()
    execute(
        "INSERT INTO projects (id, name, description, status, mode, preset, is_demo, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (pid, name, description, "draft", mode, preset, int(is_demo), now, now),
    )
    return get_project(pid)  # type: ignore[return-value]


def get_project(project_id: str) -> dict[str, Any] | None:
    return query_one("SELECT * FROM projects WHERE id = ?", (project_id,))


def list_projects() -> list[dict[str, Any]]:
    return query("SELECT * FROM projects ORDER BY created_at DESC")


def update_project(project_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    execute(f"UPDATE projects SET {cols} WHERE id = ?", (*fields.values(), project_id))


def delete_project(project_id: str) -> None:
    for table in ("images", "jobs", "logs", "metrics", "artifacts"):
        execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
    execute("DELETE FROM projects WHERE id = ?", (project_id,))


# ---------------------------------------------------------------------------
# Images / jobs / logs / metrics / artifacts helpers
# ---------------------------------------------------------------------------

def add_image(project_id: str, filename: str, filepath: str, width: int, height: int,
              file_size: int, sha256: str, quality: dict | None = None,
              status: str = "original") -> dict[str, Any]:
    img_id = new_id()
    quality_json = json.dumps(quality or {})
    execute(
        "INSERT OR REPLACE INTO images (id, project_id, filename, filepath, width, height, "
        "file_size, sha256, quality_json, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (img_id, project_id, filename, filepath, width, height, file_size, sha256,
         quality_json, status, _now()),
    )
    return query_one("SELECT * FROM images WHERE id = ?", (img_id,))  # type: ignore[return-value]


def list_images(project_id: str) -> list[dict[str, Any]]:
    return query(
        "SELECT * FROM images WHERE project_id = ? ORDER BY filename", (project_id,)
    )


def set_image_split(project_id: str, filename: str, split: str) -> None:
    execute(
        "UPDATE images SET split = ? WHERE project_id = ? AND filename = ?",
        (split, project_id, filename),
    )


def create_job(project_id: str, kind: str) -> dict[str, Any]:
    job_id = new_id()
    execute(
        "INSERT INTO jobs (id, project_id, kind, status, created_at) VALUES (?,?,?,?,?)",
        (job_id, project_id, kind, "queued", _now()),
    )
    return query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))  # type: ignore[return-value]


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def active_job(project_id: str) -> dict[str, Any] | None:
    return query_one(
        "SELECT * FROM jobs WHERE project_id = ? AND status IN ('queued','running') "
        "ORDER BY created_at DESC LIMIT 1",
        (project_id,),
    )


def latest_job(project_id: str) -> dict[str, Any] | None:
    return query_one(
        "SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
        (project_id,),
    )


def add_log(project_id: str, level: str, message: str, job_id: str | None = None) -> str:
    ts = _now()
    execute(
        "INSERT INTO logs (project_id, job_id, ts, level, message) VALUES (?,?,?,?,?)",
        (project_id, job_id, ts, level, message),
    )
    return ts


def list_logs(project_id: str, after_id: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
    return query(
        "SELECT * FROM logs WHERE project_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
        (project_id, after_id, limit),
    )


def upsert_metrics(project_id: str, kind: str, data: dict) -> None:
    execute(
        "INSERT INTO metrics (id, project_id, kind, data_json, created_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(project_id, kind) "
        "DO UPDATE SET data_json = excluded.data_json, created_at = excluded.created_at",
        (new_id(), project_id, kind, json.dumps(data), _now()),
    )


def get_metrics(project_id: str) -> list[dict[str, Any]]:
    rows = query(
        "SELECT kind, data_json, created_at FROM metrics WHERE project_id = ?",
        (project_id,),
    )
    return [{"kind": r["kind"], "data": json.loads(r["data_json"]),
             "createdAt": r["created_at"]} for r in rows]


def add_artifact(project_id: str, kind: str, filename: str, fmt: str,
                 size_bytes: int) -> None:
    execute(
        "INSERT INTO artifacts (project_id, kind, filename, format, size_bytes, created_at) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(project_id, kind, filename) DO UPDATE SET "
        "size_bytes = excluded.size_bytes, created_at = excluded.created_at",
        (project_id, kind, filename, fmt, size_bytes, _now()),
    )


def list_artifacts(project_id: str) -> list[dict[str, Any]]:
    return query(
        "SELECT project_id, kind, filename, format, size_bytes, created_at FROM artifacts "
        "WHERE project_id = ? ORDER BY kind, filename",
        (project_id,),
    )


def artifact_path(project_id: str, kind: str, filename: str) -> Path | None:
    """Resolve an artifact to a real file, guarding against path traversal."""
    row = query_one(
        "SELECT filename FROM artifacts WHERE project_id = ? AND kind = ? AND filename = ?",
        (project_id, kind, filename),
    )
    if row is None:
        return None
    root = (settings.project_dir(project_id) / "artifacts").resolve()
    target = (root / row["filename"]).resolve()
    if root not in target.parents and target != root:
        return None
    return target if target.is_file() else None


def image_path(project_id: str, image_id: str) -> Path | None:
    row = query_one(
        "SELECT filepath FROM images WHERE project_id = ? AND id = ?", (project_id, image_id)
    )
    if row is None:
        return None
    root = settings.project_dir(project_id).resolve()
    target = Path(row["filepath"]).resolve()
    if root not in target.parents and target != root:
        return None
    return target if target.is_file() else None
