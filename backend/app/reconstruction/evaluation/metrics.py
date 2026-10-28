"""Quantitative image-quality metrics.

PSNR and SSIM are implemented directly with NumPy/OpenCV so they are always
available in CPU mode. LPIPS requires torch+lpips and is used only when those
are importable. Missing capabilities return None and are surfaced as
"Not available" in the UI -- never as invented numbers.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_float(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr.astype(np.float64) / 255.0


def psnr(img_a: np.ndarray, img_b: np.ndarray, data_range: float = 1.0) -> float:
    a, b = to_float(img_a), to_float(img_b)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def _gaussian_kernel_1d(sigma: float = 1.5, radius: int = 5) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def ssim(img_a: np.ndarray, img_b: np.ndarray, data_range: float = 1.0) -> float:
    """Single-channel SSIM (Wang et al. 2004) with an 11x11 Gaussian window.

    Operates on grayscale by default, matching common practice for NVS papers.
    """
    if img_a.ndim == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        data_range = 1.0
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    k = _gaussian_kernel_1d(1.5, 5)
    kernel2d = (k[:, None] * k[None, :]).astype(np.float32)

    def blur(x):
        return cv2.filter2D(x.astype(np.float32), -1, kernel2d, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = blur(a), blur(b)
    mu_a_sq, mu_b_sq, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    sigma_a = blur(a * a) - mu_a_sq
    sigma_b = blur(b * b) - mu_b_sq
    sigma_ab = blur(a * b) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a_sq + mu_b_sq + C1) * (sigma_a + sigma_b + C2)
    return float(np.mean(num / den))


def lpips_score(img_a: np.ndarray, img_b: np.ndarray) -> float | None:
    """LPIPS (Zhang et al. 2018) when torch+lpips are installed; else None."""
    try:
        import lpips  # type: ignore
        import torch  # type: ignore
    except Exception:
        return None
    try:
        loss_fn = lpips.LPIPS(net="alex", verbose=False)  # type: ignore[attr-defined]
        def to_t(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1)[None]
            return t * 2 - 1
        with torch.no_grad():
            return float(loss_fn(to_t(img_a), to_t(img_b)).item())
    except Exception:
        return None


def compare_images(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """Full comparison dict. Values may be None where unavailable."""
    result = {
        "psnr": round(psnr(img_a, img_b), 4),
        "ssim": round(ssim(img_a, img_b), 6),
        "lpips": None,
    }
    lp = lpips_score(img_a, img_b)
    if lp is not None:
        result["lpips"] = round(lp, 6)
    return result
