"""Pure-Python parser for COLMAP sparse reconstruction models (binary format).

Implements the documented binary layout of cameras.bin, images.bin and
points3D.bin (see COLMAP docs "Output format" and
colmap/src/colmap/scene/reconstruction_io.cc). Avoiding pycolmap keeps the
backend installable everywhere COLMAP's CLI can run.

Coordinate conventions (see docs/coordinates.md):
- COLMAP stores world-to-camera (w2c) poses with an OpenCV camera frame
  (x right, y down, z forward) and quaternions in (w, x, y, z) order.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ModelParseError(ValueError):
    pass


# COLMAP camera model id -> (name, number of parameters). Source:
# colmap/src/colmap/sensor/models.h (CameraModelTypeId).
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 15),
}

# Models whose first parameter is a single shared focal length f (fx = fy = f)
# and whose center follows at params[1], params[2]:
#   SIMPLE_PINHOLE(f, cx, cy), SIMPLE_RADIAL(f, cx, cy, k), RADIAL(f, cx, cy, k1, k2),
#   SIMPLE_RADIAL_FISHEYE(f, cx, cy, k), RADIAL_FISHEYE(f, cx, cy, k1, k2)
SINGLE_FOCAL_MODELS = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                       "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]

    @property
    def fx(self) -> float:
        return self.params[0]

    @property
    def fy(self) -> float:
        return self.params[0] if self.model in SINGLE_FOCAL_MODELS else self.params[1]

    @property
    def cx(self) -> float:
        return self.params[1] if self.model in SINGLE_FOCAL_MODELS else self.params[2]

    @property
    def cy(self) -> float:
        return self.params[2] if self.model in SINGLE_FOCAL_MODELS else self.params[3]


@dataclass
class CameraImage:
    image_id: int
    qvec: np.ndarray  # (4,) wxyz, world->camera
    tvec: np.ndarray  # (3,), world->camera
    camera_id: int
    name: str
    points2d: np.ndarray  # (N, 2)
    point3d_ids: np.ndarray  # (N,) int64, -1 = unmatched


@dataclass
class PointCloud:
    ids: np.ndarray  # (N,) uint64
    xyz: np.ndarray  # (N, 3) float64
    rgb: np.ndarray  # (N, 3) uint8
    error: np.ndarray  # (N,) float64 (mean reprojection error in px)
    track_length: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


@dataclass
class Reconstruction:
    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, CameraImage] = field(default_factory=dict)
    points: PointCloud | None = None

    @property
    def num_points(self) -> int:
        return len(self.points) if self.points else 0

    def stats(self) -> dict:
        p = self.points
        return {
            "num_cameras": len(self.cameras),
            "num_images_registered": len(self.images),
            "num_points3d": self.num_points,
            "mean_track_length": float(p.track_length.mean()) if p is not None and len(p) else 0.0,
            "mean_reprojection_error_px": float(p.error.mean()) if p is not None and len(p) else 0.0,
            "num_observations": int(p.track_length.sum()) if p is not None and len(p) else 0,
        }


# ---------------------------------------------------------------------------
# Header-width handling
# ---------------------------------------------------------------------------
# COLMAP 3.x writes the entry count of cameras.bin/images.bin as uint32;
# COLMAP >= 4.0 writes it as uint64 (points3D.bin has always used uint64).
# We try the 4.x layout first and verify that the parse consumes the buffer
# exactly; the 3.x layout is the fallback. This keeps the parser compatible
# with both release lines without a version probe.

_COUNT_FORMATS = ("<Q", "<I")


def _try_parse(data: bytes, count_fmt: str, parse_fn):
    """Parse with the given count format; require exact end-of-buffer."""
    (count,) = struct.unpack_from(count_fmt, data, 0)
    header_size = 8 if count_fmt == "<Q" else 4
    if count > 100_000_000:
        raise ModelParseError(f"implausible count {count}")
    result, offset = parse_fn(data, header_size, count)
    if offset != len(data):
        raise ModelParseError("layout mismatch: trailing/missing bytes")
    return result


# ---------------------------------------------------------------------------
# cameras.bin
# ---------------------------------------------------------------------------

def _parse_cameras(data: bytes, offset: int, count: int) -> tuple[dict[int, Camera], int]:
    cameras: dict[int, Camera] = {}
    for _ in range(count):
        camera_id, model_id = struct.unpack_from("<II", data, offset)
        offset += 8
        width, height = struct.unpack_from("<QQ", data, offset)
        offset += 16
        if model_id not in CAMERA_MODELS:
            raise ModelParseError(f"Unknown camera model id {model_id}")
        if not (0 < width < 10**6 and 0 < height < 10**6):
            raise ModelParseError(f"Implausible camera size {width}x{height}")
        model_name, num_params = CAMERA_MODELS[model_id]
        params = list(struct.unpack_from(f"<{num_params}d", data, offset))
        offset += 8 * num_params
        cameras[camera_id] = Camera(camera_id, model_name, int(width), int(height), params)
    return cameras, offset


def read_cameras_bin(path: Path) -> dict[int, Camera]:
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ModelParseError("cameras.bin truncated")
    last_err: Exception | None = None
    for count_fmt in _COUNT_FORMATS:
        try:
            return _try_parse(data, count_fmt, _parse_cameras)
        except (ModelParseError, struct.error) as exc:
            last_err = exc
    raise ModelParseError(f"cameras.bin could not be parsed ({last_err})")


# ---------------------------------------------------------------------------
# images.bin
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# images.bin
# ---------------------------------------------------------------------------

def _parse_images(data: bytes, offset: int, count: int) -> tuple[dict[int, CameraImage], int]:
    images: dict[int, CameraImage] = {}
    for _ in range(count):
        image_id = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        qvec = np.array(struct.unpack_from("<4d", data, offset), dtype=np.float64)
        offset += 32
        if abs(float(np.linalg.norm(qvec)) - 1.0) > 0.05:
            raise ModelParseError("quaternion is not unit norm - layout mismatch")
        tvec = np.array(struct.unpack_from("<3d", data, offset), dtype=np.float64)
        offset += 24
        camera_id = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        end = data.index(b"\x00", offset)
        name = data[offset:end].decode("utf-8", errors="replace")
        offset = end + 1
        (num_points,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        if num_points:
            # layout is interleaved per point: (x f64, y f64, point3D_id i64)
            block = np.frombuffer(
                data,
                dtype=np.dtype([("xy", "<f8", (2,)), ("id", "<i8")]),
                count=int(num_points),
                offset=offset,
            )
            offset += int(num_points) * 24
            raw = np.ascontiguousarray(block["xy"])
            ids = np.ascontiguousarray(block["id"])
        else:
            raw = np.zeros((0, 2), dtype=np.float64)
            ids = np.zeros((0,), dtype=np.int64)
        images[image_id] = CameraImage(int(image_id), qvec, tvec, int(camera_id), name, raw, ids)
    return images, offset


def read_images_bin(path: Path) -> dict[int, CameraImage]:
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ModelParseError("images.bin truncated")
    last_err: Exception | None = None
    for count_fmt in _COUNT_FORMATS:
        try:
            return _try_parse(data, count_fmt, _parse_images)
        except (ModelParseError, struct.error) as exc:
            last_err = exc
    raise ModelParseError(f"images.bin could not be parsed ({last_err})")


# ---------------------------------------------------------------------------
# points3D.bin
# ---------------------------------------------------------------------------

def read_points3d_bin(path: Path) -> PointCloud:
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ModelParseError("points3D.bin truncated")
    (count,) = struct.unpack_from("<Q", data, 0)
    if count > 100_000_000:
        raise ModelParseError(f"implausible point count {count}")
    offset = 8

    ids = np.empty(count, dtype=np.uint64)
    xyz = np.empty((count, 3), dtype=np.float64)
    rgb = np.empty((count, 3), dtype=np.uint8)
    error = np.empty(count, dtype=np.float64)
    track_len = np.empty(count, dtype=np.int64)

    for i in range(count):
        ids[i] = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        xyz[i] = struct.unpack_from("<3d", data, offset)
        offset += 24
        rgb[i] = np.frombuffer(data, dtype=np.uint8, count=3, offset=offset)
        offset += 3
        error[i] = struct.unpack_from("<d", data, offset)[0]
        offset += 8
        (tl,) = struct.unpack_from("<Q", data, offset)
        track_len[i] = tl
        offset += 8
        offset += int(tl) * 8  # skip track entries (image_id u32 + point2D_idx u32)

    return PointCloud(ids, xyz, rgb, error, track_len)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def read_reconstruction(model_dir: Path) -> Reconstruction:
    """Read a sparse model directory containing cameras.bin/images.bin/points3D.bin."""
    model_dir = Path(model_dir)
    cameras_file = model_dir / "cameras.bin"
    images_file = model_dir / "images.bin"
    points_file = model_dir / "points3D.bin"

    missing = [p.name for p in (cameras_file, images_file) if not p.is_file()]
    if missing:
        raise ModelParseError(
            f"Sparse model is missing {', '.join(missing)} in {model_dir}; "
            "the reconstruction did not complete."
        )

    recon = Reconstruction()
    recon.cameras = read_cameras_bin(cameras_file)
    recon.images = read_images_bin(images_file)
    if points_file.is_file():
        recon.points = read_points3d_bin(points_file)
    return recon
