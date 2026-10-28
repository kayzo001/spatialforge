"""Validation/quality-assessment tests."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.app.reconstruction.preprocessing.validation import (
    blur_score,
    dataset_summary,
    detect_duplicates,
    dhash,
    inspect_image,
)


def _write(tmp_path, name, arr):
    p = tmp_path / name
    Image.fromarray(arr).save(p, "JPEG", quality=95)
    return p


def sharp_image(width=640, height=480, seed=5):
    rng = np.random.default_rng(seed)
    img = np.tile(np.linspace(30, 220, width).astype(np.uint8), (height, 1))
    for _ in range(120):
        x, y = int(rng.integers(0, width - 30)), int(rng.integers(0, height - 30))
        img[y:y + 12, x:x + 12] = rng.integers(0, 255)
    return np.stack([img] * 3, axis=-1).astype(np.uint8)


def test_sharp_image_scores_higher_than_blurred(tmp_path):
    sharp = _write(tmp_path, "sharp.jpg", sharp_image())
    # large-kernel box blur via PIL
    im = Image.fromarray(sharp_image()).filter(__import__("PIL.ImageFilter", fromlist=["BoxBlur"]).BoxBlur(6))
    blurred_path = tmp_path / "blurred.jpg"
    im.save(blurred_path, "JPEG", quality=95)

    s_sharp = blur_score(str(sharp))
    s_blur = blur_score(str(blurred_path))
    assert s_sharp is not None and s_blur is not None
    assert s_sharp > s_blur * 5
    assert inspect_image(str(sharp))["is_likely_blurry"] is False
    assert inspect_image(str(blurred_path))["is_likely_blurry"] is True


def test_unreadable_image_reports_not_readable(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"not an image at all")
    report = inspect_image(str(p))
    assert report["readable"] is False
    assert report["warnings"]


def test_low_resolution_warning(tmp_path):
    tiny = _write(tmp_path, "tiny.jpg", sharp_image(200, 120))
    report = inspect_image(str(tiny))
    assert any("resolution" in w for w in report["warnings"])


def test_detect_duplicates_finds_copy():
    h = dhash(str(_write(__import__("pathlib").Path(__import__("tempfile").mkdtemp()),
                         "a.jpg", sharp_image())))
    assert h is not None
    dup_map = detect_duplicates({"a.jpg": h, "b.jpg": h, "c.jpg": None})
    assert dup_map == {"b.jpg": "a.jpg"}


def test_dataset_summary_flags_few_images():
    reports = {f"i{n}.jpg": {"is_likely_blurry": False, "is_duplicate_of": None}
               for n in range(5)}
    summary = dataset_summary(reports)
    assert summary["image_count"] == 5
    assert any("Fewer than 10" in w for w in summary["warnings"])


def test_dataset_summary_flags_blur_share():
    reports = {f"i{n}.jpg": {"is_likely_blurry": n < 4, "is_duplicate_of": None}
               for n in range(10)}
    summary = dataset_summary(reports)
    assert "4 of 10 images appear blurry" in summary["warnings"][0]
