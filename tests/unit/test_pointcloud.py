"""Point-cloud processing and PLY export tests."""

from __future__ import annotations

import numpy as np

import backend.app.reconstruction.geometry.pointcloud as pc_mod
from backend.app.reconstruction.geometry.exporter import write_ply_binary
from backend.app.reconstruction.geometry.pointcloud import (
    decimate_to_budget,
    process_point_cloud,
)


def _cluster(rng, center, n, spread=0.02):
    return center + rng.normal(0, spread, (n, 3))


def test_outlier_removed_and_stats_reported():
    rng = np.random.default_rng(0)
    xyz = np.vstack([_cluster(rng, [0, 0, 0], 300),
                     _cluster(rng, [1, 1, 1], 300),
                     [[50.0, 50.0, 50.0]]])  # isolated outlier
    rgb = np.zeros((601, 3), dtype=np.uint8)
    out_xyz, out_rgb, stats = process_point_cloud(xyz, rgb, voxel_size=None)
    assert stats["input_points"] == 601
    assert stats["output_points"] == 600  # outlier removed
    assert stats["backend"] in ("open3d", "numpy-fallback")


def test_voxel_downsample_collapses_cluster():
    rng = np.random.default_rng(1)
    xyz = rng.normal(0, 0.001, (500, 3))  # all points inside one voxel
    rgb = np.zeros((500, 3), dtype=np.uint8)
    _, _, stats = process_point_cloud(xyz, rgb, voxel_size=0.1)
    assert stats["output_points"] <= 8


def test_numpy_fallback_matches_open3d_semantics(monkeypatch):
    """Forcing the fallback backend must still remove the outlier."""
    monkeypatch.setattr(pc_mod, "OPEN3D_AVAILABLE", False)
    rng = np.random.default_rng(2)
    xyz = np.vstack([_cluster(rng, [0, 0, 0], 200),
                     [[10.0, 0.0, 0.0]]])
    rgb = np.zeros((201, 3), dtype=np.uint8)
    out_xyz, _, stats = process_point_cloud(xyz, rgb, voxel_size=None)
    assert stats["backend"] == "numpy-fallback"
    assert stats["output_points"] == 200


def test_decimate_respects_budget():
    rng = np.random.default_rng(3)
    xyz = rng.normal(0, 1, (10_000, 3))
    rgb = np.zeros((10_000, 3), dtype=np.uint8)
    out_xyz, out_rgb, kept = decimate_to_budget(xyz, rgb, max_points=1_000)
    assert kept == 1_000
    assert out_xyz.shape[0] == 1_000
    out_xyz, _, kept = decimate_to_budget(xyz[:500], rgb[:500], max_points=1_000)
    assert kept == 500


def _parse_ply_header(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.read(1)
    return header.decode("ascii")


def test_ply_write_roundtrip(tmp_path):
    xyz = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=np.float64)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    err = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    p = tmp_path / "cloud.ply"
    n = write_ply_binary(p, xyz, rgb, error=err)
    assert n == 3

    header = _parse_ply_header(p)
    assert "format binary_little_endian 1.0" in header
    assert "element vertex 3" in header
    assert "property float error" in header

    with open(p, "rb") as f:
        data = f.read()
    vertex_data = data.split(b"end_header\n", 1)[1]
    dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3), ("error", "<f4")])
    verts = np.frombuffer(vertex_data, dtype=dtype)
    assert np.allclose(verts["xyz"][1], [1, 2, 3], atol=1e-5)
    assert list(verts["rgb"][0]) == [255, 0, 0]
    assert verts["error"][2] == 0.30000001192092896  # f32


def test_ply_write_without_error_scalar(tmp_path):
    xyz = np.zeros((4, 3), dtype=np.float64)
    rgb = np.zeros((4, 3), dtype=np.uint8)
    p = tmp_path / "plain.ply"
    write_ply_binary(p, xyz, rgb)
    header = _parse_ply_header(p)
    assert "property float error" not in header
    with open(p, "rb") as f:
        data = f.read()
    vertex_data = data.split(b"end_header\n", 1)[1]
    assert len(vertex_data) == 4 * (3 * 4 + 3)  # xyz f32 + rgb u1
