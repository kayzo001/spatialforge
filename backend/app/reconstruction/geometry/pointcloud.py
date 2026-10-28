"""Point-cloud processing (Stage 4).

Primary implementation uses Open3D (statistical outlier removal + voxel
downsampling). A pure-NumPy fallback with equivalent semantics keeps the
backend functional on platforms without Open3D wheels. The active backend is
reported in the output so the UI can disclose it honestly.
"""

from __future__ import annotations

import numpy as np

try:  # Open3D is an optional (recommended) dependency
    import open3d as o3d  # type: ignore

    OPEN3D_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    o3d = None  # type: ignore[assignment]
    OPEN3D_AVAILABLE = False


def _knn_mean_distances(xyz: np.ndarray, k: int, chunk: int = 256) -> np.ndarray:
    """Mean distance from each point to its k nearest neighbours (NumPy)."""
    n = xyz.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d2 = ((xyz[start:end, None, :] - xyz[None, :, :]) ** 2).sum(-1)
        # exclude self-matches (the diagonal block of this chunk)
        rows = np.arange(start, end)
        d2[rows - start, rows] = np.inf
        k_eff = min(k, n - 1)
        idx = np.argpartition(d2, k_eff - 1, axis=1)[:, :k_eff]
        out[start:end] = np.sqrt(np.take_along_axis(d2, idx, axis=1)).mean(axis=1)
    return out


def _numpy_statistical_outlier_removal(
    xyz: np.ndarray, rgb: np.ndarray, nb_neighbors: int, std_ratio: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = xyz.shape[0]
    k = min(nb_neighbors, max(1, n - 1))
    if n > 150_000:  # keep the O(n^2/chunk) fallback tractable
        rng = np.random.default_rng(0)
        sel = rng.choice(n, size=150_000, replace=False)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[sel] = _knn_mean_distances(xyz[sel], k) < (
            _knn_mean_distances(xyz[sel], k).mean() + std_ratio * _knn_mean_distances(xyz[sel], k).std()
        )
        mask = keep_mask
    else:
        dists = _knn_mean_distances(xyz, k)
        threshold = dists.mean() + std_ratio * dists.std()
        mask = dists < threshold
    removed = int((~mask).sum())
    return xyz[mask], rgb[mask], {"removed_by_outlier_removal": removed}


def process_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float | None = None,
    outlier_std_ratio: float = 2.0,
    nb_neighbors: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Outlier removal + optional voxel downsampling. Returns (xyz, rgb, stats)."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    stats: dict = {"backend": "open3d" if OPEN3D_AVAILABLE else "numpy-fallback",
                   "input_points": int(xyz.shape[0])}

    if xyz.shape[0] == 0:
        return xyz, rgb, stats

    if OPEN3D_AVAILABLE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
        before = len(pcd.points)
        # returns (filtered_pcd, kept_indices) across open3d versions
        pcd, _kept = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                    std_ratio=outlier_std_ratio)
        stats["removed_by_outlier_removal"] = int(before - len(pcd.points))
        if voxel_size and voxel_size > 0:
            before = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size)
            stats["removed_by_voxel_downsample"] = int(before - len(pcd.points))
        out_xyz = np.asarray(pcd.points, dtype=np.float64)
        out_rgb = (np.asarray(pcd.colors) * 255.0).round().astype(np.uint8)
    else:
        out_xyz, out_rgb, st = _numpy_statistical_outlier_removal(
            xyz, rgb, nb_neighbors, outlier_std_ratio
        )
        stats.update(st)
        if voxel_size and voxel_size > 0:
            key = np.floor(out_xyz / voxel_size).astype(np.int64)
            _, first_idx = np.unique(key, axis=0, return_index=True)
            before = out_xyz.shape[0]
            out_xyz = out_xyz[first_idx]
            out_rgb = out_rgb[first_idx]
            stats["removed_by_voxel_downsample"] = int(before - out_xyz.shape[0])

    stats["output_points"] = int(out_xyz.shape[0])
    if out_xyz.shape[0]:
        mn, mx = out_xyz.min(axis=0), out_xyz.max(axis=0)
        stats["bounds_min"] = [round(float(v), 4) for v in mn]
        stats["bounds_max"] = [round(float(v), 4) for v in mx]
        stats["extent"] = [round(float(v), 4) for v in (mx - mn)]
    return out_xyz, out_rgb, stats


def decimate_to_budget(xyz: np.ndarray, rgb: np.ndarray, max_points: int = 1_500_000):
    """Uniform random decimation for lightweight viewer representations."""
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, rgb, n
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    return xyz[idx], rgb[idx], max_points
