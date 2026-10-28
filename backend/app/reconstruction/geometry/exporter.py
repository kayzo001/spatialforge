"""Artifact exporters: binary PLY, camera JSON, scene metadata.

PLY files are written in binary little-endian form (x, y, z, red, green, blue
[, error]) which every mainstream viewer and the frontend loader understand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.app.reconstruction.colmap.transforms import build_cameras_json

PLY_HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "comment SpatialForge export\n"
    "element vertex {count}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "{extra_props}"
    "end_header\n"
)


def write_ply_binary(path: Path, xyz: np.ndarray, rgb: np.ndarray,
                     error: np.ndarray | None = None) -> int:
    """Write a binary PLY. Returns the vertex count."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz/rgb length mismatch")

    if error is not None:
        error = np.ascontiguousarray(error.reshape(-1), dtype=np.float32)
        dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3), ("error", "<f4")])
        extra = "property float error\n"
        arr = np.empty(xyz.shape[0], dtype=dtype)
        arr["xyz"] = xyz
        arr["rgb"] = rgb
        arr["error"] = error
    else:
        extra = ""
        dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
        arr = np.empty(xyz.shape[0], dtype=dtype)
        arr["xyz"] = xyz
        arr["rgb"] = rgb

    header = PLY_HEADER.format(count=xyz.shape[0], extra_props=extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
    return int(xyz.shape[0])


def write_cameras_json(path: Path, recon) -> int:
    cameras = build_cameras_json(recon)
    path.write_text(json.dumps(cameras, indent=1), encoding="utf-8")
    return len(cameras)


def write_scene_meta(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
