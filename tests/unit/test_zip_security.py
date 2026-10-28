"""Security tests for ZIP extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from backend.app.utils.security import UnsafeZipError, extract_images_from_zip

ALLOWED = {".jpg", ".png"}


def _zip(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    zpath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return zpath


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64  # minimal-ish placeholder content


def test_extracts_only_allowed_extensions(tmp_path):
    z = _zip(tmp_path, {
        "a.jpg": PNG_BYTES, "b.txt": b"ignore", "c.PNG": PNG_BYTES,
        "__MACOSX/junk": b"x",
    })
    out = tmp_path / "out"
    extracted = extract_images_from_zip(z, out, ALLOWED)
    # extensions are normalized to lowercase by the sanitizer
    assert sorted(p.name for p in extracted) == ["a.jpg", "c.png"]


def test_rejects_path_traversal(tmp_path):
    z = _zip(tmp_path, {"../../evil.jpg": PNG_BYTES})
    with pytest.raises(UnsafeZipError, match="Unsafe path"):
        extract_images_from_zip(z, tmp_path / "out", ALLOWED)


def test_rejects_absolute_paths(tmp_path):
    z = _zip(tmp_path, {"/etc/evil.jpg": PNG_BYTES})
    with pytest.raises(UnsafeZipError):
        extract_images_from_zip(z, tmp_path / "out", ALLOWED)


def test_rejects_windows_drive_paths(tmp_path):
    z = _zip(tmp_path, {"C:/evil.jpg": PNG_BYTES})
    with pytest.raises(UnsafeZipError):
        extract_images_from_zip(z, tmp_path / "out", ALLOWED)


def test_rejects_file_count_limit(tmp_path):
    entries = {f"f{i}.jpg": PNG_BYTES for i in range(6)}
    z = _zip(tmp_path, entries)
    with pytest.raises(UnsafeZipError, match="limit"):
        extract_images_from_zip(z, tmp_path / "out", ALLOWED, max_files=5)


def test_rejects_zip_bomb_ratio(tmp_path):
    zpath = tmp_path / "bomb.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bomb.jpg", b"\x00" * 500_000)
    with pytest.raises(UnsafeZipError, match="ratio"):
        extract_images_from_zip(zpath, tmp_path / "out", ALLOWED, max_ratio=100.0)


def test_extracts_within_dedicated_names(tmp_path):
    z = _zip(tmp_path, {"sub/dir/pic.jpg": PNG_BYTES, "we ird%name.jpg": PNG_BYTES})
    out = tmp_path / "out"
    extracted = extract_images_from_zip(z, out, ALLOWED)
    for p in extracted:
        assert p.resolve().parent == out.resolve()
