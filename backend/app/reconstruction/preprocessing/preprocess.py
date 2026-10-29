"""Stage 2 - non-destructive image preprocessing.

Originals are preserved untouched in `originals/`; processed derivatives
(EXIF-normalized, downscaled) are written to `images/` for COLMAP. Processed
filenames are sanitized and de-duplicated so COLMAP sees a clean flat folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from backend.app.utils.files import sanitize_filename

Image.MAX_IMAGE_PIXELS = 200_000_000


def preprocess_image(src: Path, dest_dir: Path, max_edge: int = 1600,
                     jpeg_quality: int = 92) -> dict:
    """EXIF-normalize and downscale one image. Returns processing metadata."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        width, height = im.size
        scale = 1.0
        if max(im.size) > max_edge:
            scale = max_edge / max(im.size)
            new_size = (round(width * scale), round(height * scale))
            # LANCZOS keeps texture detail that SIFT benefits from
            im = im.resize(new_size, Image.LANCZOS)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        out_path = dest_dir / (Path(sanitize_filename(src.name)).stem + ".jpg")
        # ensure unique processed name
        counter = 1
        while out_path.exists() and counter < 10000:
            out_path = dest_dir / f"{Path(sanitize_filename(src.name)).stem}_{counter}.jpg"
            counter += 1
        im.save(out_path, "JPEG", quality=jpeg_quality)
    return {
        "source": str(src),
        "processed": str(out_path),
        "processed_name": out_path.name,
        "original_size": [width, height],
        "processed_size": [im.size[0], im.size[1]],
        "scale": round(scale, 5),
    }


def preprocess_dataset(original_paths: list[Path], dest_dir: Path,
                       max_edge: int = 1600) -> list[dict]:
    """Regenerate `dest_dir` from the originals.

    `dest_dir` only ever holds derived copies, so it is emptied first. Without
    this, a re-run found last run's files, the uniqueness counter in
    preprocess_image() kicked in, and every photo appeared twice
    (view_000.jpg + view_000_1.jpg) - COLMAP then registered 72/36 images.
    """
    if dest_dir.exists():
        for stale in dest_dir.iterdir():
            if stale.is_file():
                stale.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for src in original_paths:
        try:
            results.append(preprocess_image(src, dest_dir, max_edge))
        except Exception as exc:
            results.append({"source": str(src), "error": str(exc)})
    return results


def mean_brightness(path: Path) -> float | None:
    img = cv2_imread_gray(str(path))
    return float(img.mean()) if img is not None else None


def cv2_imread_gray(path: str):
    import cv2

    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
