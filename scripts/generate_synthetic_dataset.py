#!/usr/bin/env python3
"""Generate a synthetic multi-view dataset with GROUND-TRUTH camera poses.

Renders a textured cube on a textured ground plane from orbiting viewpoints
using pure NumPy/OpenCV perspective texture mapping. Because the poses are
known exactly, this dataset is:

- a license-clean sample dataset (generated locally, MIT) for demo mode
- a fixture for coordinate-convention tests (estimated vs GT poses after
  similarity alignment)

Usage:
    python scripts/generate_synthetic_dataset.py [--out data/samples/synthetic_cube]
        [--views 36] [--width 800] [--height 600]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# z-up world (right-handed in coordinates: x cross y = z). Camera bases follow
# the OpenCV convention (x right, y down, z forward) and stay proper, which
# COLMAP requires - an improper basis makes SfM converge to a mirrored scene.
WORLD_UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------

def make_face_texture(rng: np.random.Generator, size: int = 512) -> np.ndarray:
    """Rich random texture: background, shapes, fine checker border."""
    tex = np.full((size, size, 3), rng.uniform(120, 220, 3), dtype=np.float32)
    for _ in range(rng.integers(6, 12)):
        color = rng.uniform(20, 255, 3)
        pt1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        pt2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        if rng.random() < 0.5:
            cv2.rectangle(tex, pt1, pt2, color, -1)
        else:
            cv2.circle(tex, pt1, int(rng.integers(8, size // 4)), color, -1)
    for _ in range(rng.integers(40, 90)):  # high-frequency detail for SIFT
        p = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        r = int(rng.integers(1, 4))
        cv2.circle(tex, p, r, rng.uniform(0, 255, 3), -1)
    cell = size // 16
    for i in range(16):
        for j in range(16):
            if (i + j) % 2 == 0:
                tex[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] *= 0.35
    return tex


def make_ground_texture(size: int = 1024) -> np.ndarray:
    tex = np.full((size, size, 3), (150, 148, 140), dtype=np.float32)
    cell = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                tex[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = (90, 88, 82)
    rng = np.random.default_rng(7)
    for _ in range(600):
        p = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.circle(tex, p, int(rng.integers(1, 5)), rng.uniform(40, 240, 3), -1)
    return tex


# ---------------------------------------------------------------------------
# Geometry and cameras
# ---------------------------------------------------------------------------

def cube_faces(half: float = 0.5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (corners 4x3 CCW from outside, outward normal) for each face."""
    s = half
    V = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ])
    faces = [
        ([0, 3, 2, 1], np.array([0.0, 0.0, -1.0])),  # z = -s
        ([4, 5, 6, 7], np.array([0.0, 0.0, 1.0])),   # z = +s
        ([0, 1, 5, 4], np.array([0.0, -1.0, 0.0])),  # y = -s
        ([3, 7, 6, 2], np.array([0.0, 1.0, 0.0])),   # y = +s
        ([0, 4, 7, 3], np.array([-1.0, 0.0, 0.0])),  # x = -s
        ([1, 2, 6, 5], np.array([1.0, 0.0, 0.0])),   # x = +s
    ]
    return [(V[idx], n) for idx, n in faces]


def orbit_camera(angle_deg: float, radius: float, elevation_deg: float,
                 target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera position + OpenCV w2c (R, t) looking at `target` (z-up world).

    Basis construction keeps [x_cam, y_cam, z_cam] right-handed in
    coordinates: z_cam = gaze, x_cam = gaze x world_up (image right),
    y_cam = z_cam x x_cam (image down).
    """
    a = math.radians(angle_deg)
    e = math.radians(elevation_deg)
    center = target + radius * np.array([
        math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)
    ])
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    x_cam = np.cross(forward, WORLD_UP)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(forward, x_cam)  # image-down axis
    R = np.stack([x_cam, y_cam, forward], axis=0)  # world -> camera rotation
    t = -R @ center
    return center, (R, t)


def project(points_w: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    pc = (R @ points_w.T).T + t
    uv = (K @ (pc / pc[:, 2:3]).T).T
    return uv[:, :2], pc[:, 2]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_view(R: np.ndarray, t: np.ndarray, K: np.ndarray, size: tuple[int, int],
                faces, ground_quad: np.ndarray, ground_tex: np.ndarray,
                face_textures: list[np.ndarray], brightness: float) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 210, dtype=np.float32)

    def draw_textured_quad(tex: np.ndarray, corners_w: np.ndarray) -> None:
        uv, depth = project(corners_w, R, t, K)
        if (depth <= 1e-6).any():
            return
        src = np.array([[0, 0], [tex.shape[1], 0],
                        [tex.shape[1], tex.shape[0]], [0, tex.shape[0]]],
                       dtype=np.float32)
        H, _ = cv2.findHomography(src, uv.astype(np.float32), cv2.RANSAC, 3.0)
        if H is None:
            return
        warped = cv2.warpPerspective(
            tex, H, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # composite: warpPerspective outside the quad produced garbage; mask it
        poly = uv.astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        m3 = mask[:, :, None].astype(np.float32) / 255.0
        canvas[:] = canvas * (1 - m3) + warped * m3

    # painter's algorithm: ground first, then visible cube faces (convex body)
    draw_textured_quad(ground_tex, ground_quad)
    camera_center = -R.T @ t
    for idx, (corners, normal) in enumerate(faces):
        if float(normal @ (camera_center - corners.mean(axis=0))) <= 0:
            continue
        tex = face_textures[idx].copy()
        tex *= brightness  # mild per-view exposure variation
        draw_textured_quad(tex, corners)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def generate(out_dir: Path, views: int = 36, width: int = 800, height: int = 600,
             seed: int = 1234) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fx = fy = 0.9 * width
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    faces = cube_faces(0.5)
    face_textures = [make_face_texture(rng) for _ in faces]
    ground_tex = make_ground_texture()
    # half=1.4 keeps all four corners in front of every orbit camera (the
    # radius is 2.1), so the quad never crosses the near plane.
    ground_half = 1.4
    ground_quad = np.array([
        [-ground_half, -ground_half, -0.75], [ground_half, -ground_half, -0.75],
        [ground_half, ground_half, -0.75], [-ground_half, ground_half, -0.75],
    ], dtype=np.float64)

    gt_images = []
    radius = 2.1
    for i in range(views):
        angle = 360.0 * i / views
        elevation = float(np.clip(18 + 16 * math.sin(math.radians(angle * 2)), 5, 40))
        center, (R, t) = orbit_camera(angle, radius, elevation, np.zeros(3))
        brightness = float(rng.uniform(0.85, 1.1))
        img = render_view(R, t, K, (width, height), faces, ground_quad,
                          ground_tex, face_textures, brightness)
        name = f"view_{i:03d}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        c2w = np.eye(4)
        c2w[:3, :3] = R.T          # camera axes as columns (OpenCV convention)
        c2w[:3, 3] = center        # camera position in world
        gt_images.append({
            "name": name,
            "camera_center": [round(v, 6) for v in center],
            "c2w_opencv": [[round(v, 8) for v in row] for row in c2w],
        })
        if (i + 1) % 6 == 0:
            print(f"  rendered {i + 1}/{views}")

    meta = {
        "description": "Synthetic orbit dataset generated by "
                       "scripts/generate_synthetic_dataset.py (MIT license).",
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "width": width, "height": height,
                       "model": "PINHOLE"},
        "world_up": [0, 0, 1],
        "note": "c2w_opencv is camera-to-world in the OpenCV convention "
                "(x right, y down, z forward). COLMAP gauge/scale may differ; "
                "compare after similarity alignment.",
        "images": gt_images,
    }
    (out_dir / "gt_poses.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Wrote {views} images + gt_poses.json to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/samples/synthetic_cube")
    ap.add_argument("--views", type=int, default=36)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()
    generate(Path(args.out), args.views, args.width, args.height)


if __name__ == "__main__":
    main()
