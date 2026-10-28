"""COLMAP binary model parser tests using hand-crafted fixture files."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from backend.app.reconstruction.colmap.parser import (
    ModelParseError,
    read_cameras_bin,
    read_images_bin,
    read_points3d_bin,
    read_reconstruction,
)

SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, OPENCV = 0, 1, 2, 4


def _cameras_bin() -> bytes:
    out = struct.pack("<I", 2)
    # camera 1: SIMPLE_PINHOLE(f=1000, cx=320, cy=240) -> 3 params
    out += struct.pack("<IIQQ", 1, SIMPLE_PINHOLE, 640, 480)
    out += struct.pack("<3d", 1000.0, 320.0, 240.0)
    # camera 2: PINHOLE(fx=900, fy=950, cx=300, cy=250) -> 4 params
    out += struct.pack("<IIQQ", 2, PINHOLE, 800, 600)
    out += struct.pack("<4d", 900.0, 950.0, 300.0, 250.0)
    return out


def _images_bin() -> bytes:
    out = struct.pack("<I", 1)
    q = (1.0, 0.0, 0.0, 0.0)  # identity rotation
    t = (0.1, 0.2, 0.3)
    name = b"view_000.jpg"
    out += struct.pack("<I", 10)              # image id
    out += struct.pack("<4d", *q)
    out += struct.pack("<3d", *t)
    out += struct.pack("<I", 1)               # camera id
    out += name + b"\x00"
    out += struct.pack("<Q", 2)               # two points2D
    out += struct.pack("<ddq", 10.0, 20.0, 1)   # matched to point 1
    out += struct.pack("<ddq", 30.0, 40.0, -1)  # unmatched
    return out


def _points3d_bin() -> bytes:
    out = struct.pack("<Q", 2)
    for pid, xyz, rgb, err, track in [
        (1, (0.0, 0.0, 1.0), (255, 0, 0), 0.5, [(10, 0)]),
        (2, (1.0, 1.0, 2.0), (0, 255, 0), 1.5, [(10, 1), (11, 0)]),
    ]:
        out += struct.pack("<Q", pid)
        out += struct.pack("<3d", *xyz)
        out += bytes(rgb)
        out += struct.pack("<d", err)
        out += struct.pack("<Q", len(track))
        for img_id, idx in track:
            out += struct.pack("<II", img_id, idx)
    return out


def test_parse_cameras_intrinsics(tmp_path):
    p = tmp_path / "cameras.bin"
    p.write_bytes(_cameras_bin())
    cams = read_cameras_bin(p)
    assert set(cams) == {1, 2}
    c1, c2 = cams[1], cams[2]
    assert c1.model == "SIMPLE_PINHOLE"
    # single-focal model: fx == fy == params[0], center follows
    assert (c1.fx, c1.fy, c1.cx, c1.cy) == (1000.0, 1000.0, 320.0, 240.0)
    # pinhole: separate focal lengths
    assert (c2.fx, c2.fy, c2.cx, c2.cy) == (900.0, 950.0, 300.0, 250.0)


def test_parse_images(tmp_path):
    p = tmp_path / "images.bin"
    p.write_bytes(_images_bin())
    imgs = read_images_bin(p)
    assert set(imgs) == {10}
    img = imgs[10]
    assert img.name == "view_000.jpg"
    assert img.camera_id == 1
    assert np.allclose(img.qvec, [1, 0, 0, 0])
    assert np.allclose(img.tvec, [0.1, 0.2, 0.3])
    assert img.points2d.shape == (2, 2)
    assert list(img.point3d_ids) == [1, -1]


def test_parse_points3d(tmp_path):
    p = tmp_path / "points3D.bin"
    p.write_bytes(_points3d_bin())
    pc = read_points3d_bin(p)
    assert len(pc) == 2
    assert np.allclose(pc.xyz[0], [0, 0, 1])
    assert list(pc.rgb[0]) == [255, 0, 0]
    assert pc.error[1] == 1.5
    assert list(pc.track_length) == [1, 2]


def test_read_reconstruction_and_stats(tmp_path):
    (tmp_path / "cameras.bin").write_bytes(_cameras_bin())
    (tmp_path / "images.bin").write_bytes(_images_bin())
    (tmp_path / "points3D.bin").write_bytes(_points3d_bin())
    recon = read_reconstruction(tmp_path)
    stats = recon.stats()
    assert stats["num_cameras"] == 2
    assert stats["num_images_registered"] == 1
    assert stats["num_points3d"] == 2
    assert stats["mean_track_length"] == 1.5
    assert stats["mean_reprojection_error_px"] == 1.0


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelParseError):
        read_reconstruction(tmp_path)


def test_truncated_file_raises(tmp_path):
    p = tmp_path / "cameras.bin"
    p.write_bytes(b"\x01\x02")
    with pytest.raises(ModelParseError):
        read_cameras_bin(p)


def _cameras_bin_u64_header() -> bytes:
    """COLMAP >= 4.0 layout: count is uint64."""
    out = struct.pack("<Q", 1)
    out += struct.pack("<IIQQ", 7, PINHOLE, 640, 480)
    out += struct.pack("<4d", 700.0, 700.0, 320.0, 240.0)
    return out


def test_parse_cameras_colmap4_u64_header(tmp_path):
    p = tmp_path / "cameras.bin"
    p.write_bytes(_cameras_bin_u64_header())
    cams = read_cameras_bin(p)
    assert set(cams) == {7}
    assert (cams[7].fx, cams[7].fy, cams[7].cx, cams[7].cy) == (700.0, 700.0, 320.0, 240.0)


def _images_bin_u64_header() -> bytes:
    out = struct.pack("<Q", 1)
    out += struct.pack("<I", 3)
    out += struct.pack("<4d", 1.0, 0.0, 0.0, 0.0)
    out += struct.pack("<3d", 0.0, 0.0, 1.0)
    out += struct.pack("<I", 1)
    out += b"shot.jpg\x00"
    out += struct.pack("<Q", 0)
    return out


def test_parse_images_colmap4_u64_header(tmp_path):
    p = tmp_path / "images.bin"
    p.write_bytes(_images_bin_u64_header())
    imgs = read_images_bin(p)
    assert set(imgs) == {3}
    assert imgs[3].name == "shot.jpg"
