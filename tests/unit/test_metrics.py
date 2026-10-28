"""Image-quality metrics and holdout split tests."""

from __future__ import annotations

import numpy as np

from backend.app.reconstruction.evaluation.holdout import choose_eval_images
from backend.app.reconstruction.evaluation.metrics import compare_images, psnr, ssim


def _gradient_image(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    img = np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))
    img += rng.integers(-8, 8, img.shape).astype(np.uint8)
    return np.stack([img] * 3, axis=-1).astype(np.uint8)


def test_identical_images_perfect_scores():
    img = _gradient_image()
    p = psnr(img, img)
    assert p == float("inf")
    s = ssim(img, img)
    assert s > 0.999


def test_noisy_image_degrades_scores():
    a = _gradient_image(seed=1)
    rng = np.random.default_rng(2)
    b = np.clip(a.astype(np.float64) + rng.normal(0, 12, a.shape), 0, 255).astype(np.uint8)
    p = psnr(a, b)
    assert 15.0 < p < 35.0  # finite, meaningful degradation
    s = ssim(a, b)
    assert 0.1 < s < 0.99


def test_compare_images_reports_lpips_none_without_torch():
    result = compare_images(_gradient_image(seed=3), _gradient_image(seed=4))
    assert "psnr" in result and "ssim" in result
    # lpips must be None (never a fabricated number) when torch/lpips is absent
    try:
        import lpips  # noqa: F401
        torch_present = True
    except ImportError:
        torch_present = False
    if not torch_present:
        assert result["lpips"] is None


def test_holdout_deterministic_and_fractional():
    names = [f"img_{i:03d}.jpg" for i in range(50)]
    a = choose_eval_images(names, 0.1)
    b = choose_eval_images(names, 0.1)
    assert a == b  # deterministic
    assert len(a) == 5
    assert set(a).issubset(names)
    assert choose_eval_images(names, 0.0) == []
    assert choose_eval_images([], 0.2) == []


def test_holdout_small_dataset_returns_empty():
    names = [f"img_{i}.jpg" for i in range(5)]
    assert choose_eval_images(names, 0.1) == []
