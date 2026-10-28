"""Image validation and quality assessment for Stage 1/2.

Produces a per-image quality report: resolution checks, blur estimation via
Laplacian variance, dark-frame detection, and perceptual-hash duplicate
detection. Scores are heuristics -- they surface *warnings*, never rejects on
their own (the user keeps full control over their dataset).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000  # decompression-bomb guard

MIN_WIDTH = 320
MIN_HEIGHT = 320
BLUR_THRESHOLD = 60.0  # Laplacian variance below this => likely blurry
DARK_THRESHOLD = 30.0  # mean pixel value below this => too dark


def load_gray(path: str) -> np.ndarray | None:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return img


def blur_score(path: str) -> float | None:
    """Variance of the Laplacian; higher = sharper. None when unreadable."""
    gray = load_gray(path)
    if gray is None:
        return None
    small = cv2.resize(gray, (min(gray.shape[1], 1024), min(gray.shape[0], 1024)))
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def dhash(path: str, hash_size: int = 8) -> str | None:
    """Difference hash (64-bit) for near-duplicate detection."""
    try:
        gray = load_gray(path)
        if gray is None:
            return None
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return "".join("1" if v else "0" for v in diff.flatten())
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b, strict=False))


def inspect_image(path: str) -> dict:
    """Full quality inspection for a single image file."""
    warnings: list[str] = []
    try:
        with Image.open(path) as im:
            width, height = im.size
            im.close()
    except Exception as exc:
        return {
            "width": None, "height": None, "blur_score": None,
            "is_likely_blurry": False, "is_duplicate_of": None, "too_dark": False,
            "warnings": [f"Unreadable image: {exc}"], "readable": False,
        }

    score = blur_score(path)
    is_blurry = bool(score is not None and score < BLUR_THRESHOLD)
    if is_blurry:
        warnings.append("Image appears blurry (low Laplacian variance).")

    gray = load_gray(path)
    mean_val = float(gray.mean()) if gray is not None else None
    too_dark = bool(mean_val is not None and mean_val < DARK_THRESHOLD)
    if too_dark:
        warnings.append("Image is very dark; matching may fail.")

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        warnings.append(f"Low resolution ({width}x{height}); at least {MIN_WIDTH}px recommended.")

    return {
        "width": width,
        "height": height,
        "blur_score": round(score, 2) if score is not None else None,
        "is_likely_blurry": is_blurry,
        "is_duplicate_of": None,  # filled by the duplicate pass below
        "too_dark": too_dark,
        "warnings": warnings,
        "readable": True,
    }


def detect_duplicates(hashes: dict[str, str | None], max_distance: int = 4) -> dict[str, str]:
    """Map filename -> original filename for perceptual near-duplicates.

    The first occurrence is always treated as the original; later images whose
    dhash is within `max_distance` bits are flagged as duplicates of it.
    """
    duplicates: dict[str, str] = {}
    seen: list[tuple[str, str]] = []
    for name in sorted(hashes):
        h = hashes[name]
        if not h:
            continue
        for other_name, other_h in seen:
            if hamming(h, other_h) <= max_distance:
                duplicates[name] = other_name
                break
        else:
            seen.append((name, h))
    return duplicates


def dataset_summary(reports: dict[str, dict]) -> dict:
    """Dataset-level warnings, e.g. insufficient overlap heuristics."""
    warnings: list[str] = []
    n = len(reports)
    blurry = [n_ for n_, r in reports.items() if r.get("is_likely_blurry")]
    dups = [n_ for n_, r in reports.items() if r.get("is_duplicate_of")]
    if n and len(blurry) / n > 0.3:
        warnings.append(
            f"{len(blurry)} of {n} images appear blurry; sharpness strongly affects reconstruction."
        )
    if dups:
        warnings.append(f"{len(dups)} near-duplicate images detected; they add no new viewpoints.")
    if 0 < n < 10:
        warnings.append(
            "Fewer than 10 images: coverage will be limited. 20-80 well-overlapping views are recommended."
        )
    return {"image_count": n, "blurry": blurry, "duplicates": dups, "warnings": warnings}
