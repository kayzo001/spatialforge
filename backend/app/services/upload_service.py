"""Image ingestion service (Stage 1): files, ZIP archives, local directories.

Every image is validated (readable, resolution, blur, darkness) and
fingerprinted (SHA-256 for exact dupes, dhash for near-dupes) before it is
persisted to `originals/` untouched. Returns a structured report consumed by
the API and the UI's dataset-validation panel.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from backend.app import db
from backend.app.config import settings
from backend.app.reconstruction.preprocessing.validation import (
    dataset_summary,
    detect_duplicates,
    dhash,
    inspect_image,
)
from backend.app.utils.files import sanitize_filename
from backend.app.utils.security import UnsafeZipError, extract_images_from_zip


class UploadError(ValueError):
    pass


def _save_stream(data: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def _dedupe_filename(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    stem, ext = dest.stem, dest.suffix
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{stem}_{counter}{ext}"
        counter += 1
    return dest


def _ingest_paths(project_id: str, paths: list[Path], source_label: str,
                  move: bool = True) -> dict[str, Any]:
    originals_dir = settings.project_dir(project_id) / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    added: list[dict] = []
    rejected: list[dict] = []
    per_file_limit = settings.max_upload_mb * 1024 * 1024
    max_images = settings.max_images_per_project

    for path in paths:
        existing_count = len(db.list_images(project_id)) + len(added)
        if existing_count >= max_images:
            rejected.append({"filename": path.name, "reason":
                             f"Project image limit reached ({max_images})."})
            continue
        file_size = path.stat().st_size
        if file_size > per_file_limit:
            rejected.append({"filename": path.name,
                             "reason": f"File exceeds {settings.max_upload_mb} MB limit."})
            continue

        report = inspect_image(str(path))
        if not report.pop("readable", True):
            rejected.append({"filename": path.name, "reason": report["warnings"][0]})
            continue

        sha = _sha(path)
        dest = _dedupe_filename(originals_dir, sanitize_filename(path.name))
        if move:
            shutil.move(str(path), str(dest))
        else:
            shutil.copy2(str(path), str(dest))

        row = db.add_image(
            project_id=project_id, filename=dest.name, filepath=str(dest),
            width=report["width"], height=report["height"],
            file_size=file_size, sha256=sha, quality=report,
        )
        added.append({"id": row["id"], "filename": dest.name, "quality": report})

    # near-duplicate detection across all images in the project
    all_rows = db.list_images(project_id)
    hashes = {r["filename"]: dhash(r["filepath"]) for r in all_rows}
    dup_map = detect_duplicates(hashes)
    for r in all_rows:
        quality = json.loads(r["quality_json"]) if r["quality_json"] else {}
        dup_of = dup_map.get(r["filename"])
        if dup_of:
            quality["is_duplicate_of"] = dup_of
            warnings = quality.get("warnings", [])
            if not any(w.startswith("Near-duplicate") for w in warnings):
                warnings.append(f"Near-duplicate of {dup_of} (perceptual hash).")
            quality["warnings"] = warnings
            db.execute("UPDATE images SET quality_json = ? WHERE id = ?",
                       (json.dumps(quality), r["id"]))
        added_note = next((a for a in added if a["filename"] == r["filename"]), None)
        if added_note:
            added_note["quality"] = quality

    db.execute(
        "UPDATE projects SET image_count = ? WHERE id = ?",
        (len(all_rows), project_id),
    )
    summary = dataset_summary({
        r["filename"]: json.loads(r["quality_json"]) if r["quality_json"] else {}
        for r in all_rows
    })
    return {
        "source": source_label,
        "added": added,
        "rejected": rejected,
        "summary": summary,
    }


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ingest_uploads(project_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Ingest in-memory uploads: list of (filename, bytes). ZIPs are extracted."""
    staging = settings.project_dir(project_id) / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    collected: list[Path] = []
    errors: list[str] = []

    try:
        for name, data in files:
            ext = Path(name).suffix.lower()
            if ext == ".zip":
                tmp = staging / f"zip_{sanitize_filename(Path(name).stem)}_{len(collected)}.zip"
                _save_stream(data, tmp)
                try:
                    extracted = extract_images_from_zip(
                        tmp, staging, image_exts(),
                        max_files=settings.max_images_per_project,
                        max_total_bytes=settings.max_project_mb * 1024 * 1024,
                    )
                    collected.extend(extracted)
                except UnsafeZipError as exc:
                    errors.append(f"{name}: {exc}")
                finally:
                    tmp.unlink(missing_ok=True)
            elif ext in image_exts():
                p = staging / f"file_{len(collected)}_{sanitize_filename(name)}"
                _save_stream(data, p)
                collected.append(p)
            else:
                errors.append(f"{name}: unsupported file type {ext or '(none)'}")

        report = _ingest_paths(project_id, collected, "upload")
        report["errors"] = errors
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ingest_directory(project_id: str, directory: Path) -> dict[str, Any]:
    """Ingest images from a local directory (demo/CLI path).

    Files are copied (not moved): the source dataset stays usable for
    further demo runs.
    """
    paths = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in image_exts())
    if not paths:
        raise UploadError(f"No supported images found in {directory.name}")
    return _ingest_paths(project_id, paths, f"directory:{directory.name}", move=False)


def image_exts() -> set[str]:
    from backend.app.config import CAMERA_EXTS

    return set(CAMERA_EXTS)
