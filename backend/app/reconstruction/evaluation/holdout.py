"""Deterministic train/eval splitting for held-out novel-view evaluation."""

from __future__ import annotations

import hashlib


def choose_eval_images(filenames: list[str], eval_fraction: float) -> list[str]:
    """Pick evaluation images deterministically (hash-based, stable across runs).

    `eval_fraction=0` returns an empty eval set (train on everything).
    With >=10 images the eval set is at least 1 image when fraction > 0.
    """
    if eval_fraction <= 0 or not filenames:
        return []
    target = max(1, round(len(filenames) * eval_fraction)) if len(filenames) >= 10 else 0
    if target == 0:
        return []
    scored = sorted(
        (hashlib.sha256(name.encode()).hexdigest(), name) for name in filenames
    )
    return sorted(name for _, name in scored[:target])
