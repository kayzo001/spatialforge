"""Filesystem helpers: hashing and path safety."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._\- ]+")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sanitize_filename(name: str, fallback: str = "image") -> str:
    """Strip directories and unsafe characters; keep the file extension."""
    name = os.path.basename(name.replace("\\", "/"))
    stem = Path(name).stem or fallback
    ext = Path(name).suffix.lower()
    clean = _UNSAFE_CHARS.sub("_", stem).strip("._ ") or fallback
    clean_ext = _UNSAFE_CHARS.sub("_", ext)
    return f"{clean[:120]}{clean_ext}"


def human_size(num_bytes: int | float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def inside_directory(target: Path, root: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
