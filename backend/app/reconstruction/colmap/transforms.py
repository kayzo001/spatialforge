"""Coordinate-system transformations between COLMAP, SpatialForge and three.js.

Conventions (see docs/coordinates.md for the full write-up):

- COLMAP model: world-to-camera (w2c) pose, OpenCV camera frame
  (x right, y down, z forward), quaternion order (w, x, y, z).
- SpatialForge internal / three.js: camera-to-world (c2w) pose in the OpenGL
  camera frame (x right, y up, z backward), as a 4x4 matrix in column-vector
  convention: p_world = C2W @ p_camera.

The w2c->c2w conversion inverts the rigid transform; the OpenCV->OpenGL frame
change flips the camera's local y and z axes: C2W_GL = C2W_CV @ diag(1, -1, -1).
The camera *center* is unaffected (it lies in both representations).
"""

from __future__ import annotations

import numpy as np

from backend.app.reconstruction.colmap.parser import Camera, Reconstruction

_FLIP_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Hamilton (w, x, y, z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w, x, y, z) quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def w2c_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP w2c (OpenCV frame) -> 4x4 c2w (OpenCV frame)."""
    R = quat_to_rotmat(qvec)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ np.asarray(tvec, dtype=np.float64)
    return c2w


def c2w_cv_to_gl(c2w_cv: np.ndarray) -> np.ndarray:
    """OpenCV-convention c2w -> OpenGL-convention c2w (flip camera y/z axes)."""
    return np.asarray(c2w_cv, dtype=np.float64) @ _FLIP_CV_TO_GL


def colmap_pose_to_c2w_gl(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """One-shot conversion from a COLMAP image pose to the three.js c2w matrix."""
    return c2w_cv_to_gl(w2c_to_c2w(qvec, tvec))


def camera_center(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w, dtype=np.float64)[:3, 3]


def intrinsics(camera: Camera) -> dict[str, float]:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "model": camera.model,
    }


def fov_degrees(width: int, height: int, fx: float, fy: float) -> tuple[float, float]:
    import math

    fov_x = 2.0 * math.degrees(math.atan(width / (2.0 * fx)))
    fov_y = 2.0 * math.degrees(math.atan(height / (2.0 * fy)))
    return fov_x, fov_y


def build_cameras_json(recon: Reconstruction, max_images: int | None = None) -> list[dict]:
    """Serialized camera poses for the frontend viewer (OpenGL c2w convention)."""
    out: list[dict] = []
    images = list(recon.images.values())
    if max_images:
        images = images[:max_images]
    for img in sorted(images, key=lambda i: i.name):
        cam = recon.cameras.get(img.camera_id)
        if cam is None:
            continue
        c2w = colmap_pose_to_c2w_gl(img.qvec, img.tvec)
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        fov_x, fov_y = fov_degrees(cam.width, cam.height, fx, fy)
        out.append({
            "id": img.image_id,
            "name": img.name,
            "width": cam.width,
            "height": cam.height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "c2w": np.round(c2w, 8).tolist(),
            "center": np.round(camera_center(c2w), 6).tolist(),
            "fov_x_deg": round(fov_x, 3),
            "fov_y_deg": round(fov_y, 3),
        })
    return out


def similarity_alignment(c2w_list_a: list[np.ndarray], c2w_list_b: list[np.ndarray]):
    """Umeyama/Procrustes similarity alignment between two sets of camera centers.

    COLMAP reconstructions have arbitrary scale and coordinate gauge, so
    comparing estimated vs ground-truth poses requires a 7-DoF alignment
    (rotation + translation + uniform scale). Returns (s, R, t, aligned_a).
    """
    A = np.stack([camera_center(m) for m in c2w_list_a])
    B = np.stack([camera_center(m) for m in c2w_list_b])
    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - mu_a, B - mu_b
    cov = B0.T @ A0 / len(A)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt  # rotates A onto B
    var_a = (A0 ** 2).sum() / len(A)
    scale = float(np.trace(np.diag(D) @ S) / var_a) if var_a > 0 else 1.0
    t = mu_b - scale * R @ mu_a
    out = []
    for m in c2w_list_a:
        mm = np.eye(4)
        # scale applies to positions only; the rotation block must stay orthonormal
        mm[:3, :3] = R @ m[:3, :3]
        mm[:3, 3] = scale * (R @ m[:3, 3]) + t
        out.append(mm)
    return scale, R, t, out
