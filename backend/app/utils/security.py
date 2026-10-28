"""ZIP archive handling with hardening against common archive attacks.

Guards implemented:
- path traversal ("../", absolute paths, drive letters, UNC prefixes)
- zip bombs (member count, total uncompressed size, per-member compression ratio)
- disallowed file types
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from backend.app.utils.files import sanitize_filename


class UnsafeZipError(ValueError):
    """Raised when an archive violates the safety policy."""


def _is_dangerous_name(name: str) -> bool:
    p = PurePosixPath(name)
    if p.is_absolute() or name.startswith(("/", "\\")):
        return True
    if ":" in name or name.startswith("//"):
        return True
    for part in p.parts:
        if part == ".." or part == "":
            if part == "..":
                return True
    return False


def extract_images_from_zip(
    zip_path: Path,
    dest_dir: Path,
    allowed_exts: set[str],
    max_files: int = 500,
    max_total_bytes: int = 4 * 1024**3,
    max_ratio: float = 200.0,
) -> list[Path]:
    """Safely extract image members of a zip archive. Returns extracted paths.

    Raises UnsafeZipError on any policy violation; partial extraction is
    avoided by validating all members up-front (except ratio, which needs the
    compressed size and is validated per member before writing).
    """
    extracted: list[Path] = []
    total_bytes = 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_dir = dest_dir.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        if len(members) > max_files:
            raise UnsafeZipError(
                f"Archive contains {len(members)} files (limit {max_files})."
            )

        for info in members:
            if _is_dangerous_name(info.filename):
                raise UnsafeZipError(f"Unsafe path in archive: {info.filename!r}")

            ext = Path(info.filename).suffix.lower()
            if ext not in allowed_exts:
                continue  # silently skip non-image entries (e.g. __MACOSX metadata)

            if info.compress_size > 0 and info.file_size / info.compress_size > max_ratio:
                raise UnsafeZipError(
                    f"Suspicious compression ratio for {info.filename!r}; possible zip bomb."
                )

            total_bytes += info.file_size
            if total_bytes > max_total_bytes:
                raise UnsafeZipError(
                    f"Archive expands beyond the {max_total_bytes // (1024**2)} MB limit."
                )

        for info in members:
            ext = Path(info.filename).suffix.lower()
            if ext not in allowed_exts:
                continue
            target = dest_dir / sanitize_filename(info.filename)
            with zf.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            extracted.append(target)

    return extracted
