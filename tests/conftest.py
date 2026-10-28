"""Shared test fixtures.

Sets an isolated data directory BEFORE any backend import so tests never
touch the developer's real data. Each test gets a wiped database.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_TMP_DATA = tempfile.mkdtemp(prefix="spatialforge-tests-")
os.environ["SPATIALFORGE_DATA_DIR"] = _TMP_DATA

import pytest  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture()
def data_dir() -> Path:
    return Path(_TMP_DATA)


@pytest.fixture(autouse=True)
def fresh_database():
    from backend.app import db

    db.close_conn()
    yield
    db.close_conn()


@pytest.fixture(autouse=True)
def _wipe_data(fresh_database):
    for child in Path(_TMP_DATA).iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    yield


@pytest.fixture()
def sample_jpeg_bytes() -> bytes:
    """A small sharp test image with plenty of edges."""
    import io

    import numpy as np

    rng = np.random.default_rng(3)
    img = np.tile(np.linspace(20, 230, 240).astype(np.uint8), (180, 1))
    for _ in range(60):
        x, y = int(rng.integers(0, 200)), int(rng.integers(0, 140))
        img[y:y + rng.integers(6, 20), x:x + rng.integers(6, 20)] = rng.integers(0, 255)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture()
def sample_images_dir(tmp_path: Path, sample_jpeg_bytes: bytes) -> Path:
    d = tmp_path / "images"
    d.mkdir()
    for i in range(5):
        (d / f"img_{i:02d}.jpg").write_bytes(sample_jpeg_bytes)
    return d
