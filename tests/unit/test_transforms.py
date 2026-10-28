"""Coordinate-convention and camera-pose correctness tests."""

from __future__ import annotations

import numpy as np

from backend.app.reconstruction.colmap.parser import Camera, CameraImage, PointCloud, Reconstruction
from backend.app.reconstruction.colmap.transforms import (
    build_cameras_json,
    c2w_cv_to_gl,
    camera_center,
    colmap_pose_to_c2w_gl,
    quat_to_rotmat,
    rotmat_to_quat,
    similarity_alignment,
    w2c_to_c2w,
)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:  # sign ambiguity is irrelevant for rotations
            q = -q
        R = quat_to_rotmat(q)
        q2 = rotmat_to_quat(R)
        assert np.allclose(q, q2, atol=1e-9) or np.allclose(q, -q2, atol=1e-9)
        assert np.allclose(quat_to_rotmat(q2), R, atol=1e-9)
    assert np.allclose(quat_to_rotmat([1, 0, 0, 0]), np.eye(3))


def test_identity_pose_conversion():
    """Camera at origin looking down +z (OpenCV). c2w must keep the origin."""
    c2w = w2c_to_c2w(np.array([1.0, 0, 0, 0]), np.zeros(3))
    assert np.allclose(c2w, np.eye(4))


def test_camera_center_preserved_through_flip():
    """The OpenCV->OpenGL axis flip must not move the camera center."""
    rng = np.random.default_rng(1)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    t = rng.normal(size=3)
    c2w_cv = w2c_to_c2w(q, t)
    c2w_gl = c2w_cv_to_gl(c2w_cv)
    assert np.allclose(camera_center(c2w_cv), camera_center(c2w_gl), atol=1e-12)


def test_known_pose_camera_at_negative_z():
    """Camera at C=(0,0,-5) looking toward world +z: t must be (0,0,5).

    World origin must project to (cx, cy) with z=5 in camera coordinates.
    """
    R = np.eye(3)
    C = np.array([0.0, 0.0, -5.0])
    t = -R @ C
    c2w = w2c_to_c2w(np.array([1.0, 0, 0, 0]), t)
    assert np.allclose(c2w[:3, 3], [0, 0, -5])
    # roundtrip: w2c back from c2w
    R_back = c2w[:3, :3].T
    t_back = -R_back @ c2w[:3, 3]
    assert np.allclose(t_back, [0, 0, 5])
    p_cam = R_back @ np.zeros(3) + t_back
    assert np.allclose(p_cam, [0, 0, 5])  # origin is 5 units in front


def test_colmap_pose_to_gl_matches_manual_composition():
    rng = np.random.default_rng(2)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    t = rng.normal(size=3)
    manual = c2w_cv_to_gl(w2c_to_c2w(q, t))
    direct = colmap_pose_to_c2w_gl(q, t)
    assert np.allclose(manual, direct)


def test_build_cameras_json_opengl_convention():
    """Z axis of the c2w must point from the scene toward the camera
    (OpenGL camera looks down -z). Camera sits at (0,0,-5) gazing +z."""
    cam = Camera(1, "PINHOLE", 640, 480, [600.0, 600.0, 320.0, 240.0])
    img = CameraImage(1, np.array([1.0, 0, 0, 0]), np.array([0.0, 0, 5.0]),
                      1, "a.jpg", np.zeros((0, 2)), np.zeros((0,), dtype=np.int64))
    recon = Reconstruction(cameras={1: cam}, images={1: img},
                           points=PointCloud(np.zeros(0), np.zeros((0, 3)),
                                             np.zeros((0, 3), dtype=np.uint8),
                                             np.zeros(0), np.zeros(0)))
    cams = build_cameras_json(recon)
    assert len(cams) == 1
    entry = cams[0]
    assert entry["name"] == "a.jpg"
    c2w = np.array(entry["c2w"])
    # camera center at (0,0,5): world origin is in front -> scene direction
    # is -z of the camera frame
    scene_dir = c2w[:3, :3] @ np.array([0, 0, -1.0])
    center = c2w[:3, 3]
    assert np.allclose(scene_dir, -center / np.linalg.norm(center))
    assert (entry["fx"], entry["fy"]) == (600.0, 600.0)


def test_similarity_alignment_recovers_transform():
    rng = np.random.default_rng(3)
    poses_a = []
    for _ in range(6):
        m = np.eye(4)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        m[:3, :3] = quat_to_rotmat(q)
        m[:3, 3] = rng.uniform(-2, 2, 3)
        poses_a.append(m)
    # ground truth: scale 3.5, rotation about y, translation offset
    ang = np.deg2rad(37)
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0],
                  [-np.sin(ang), 0, np.cos(ang)]])
    s = 3.5
    t = np.array([10.0, -4.0, 7.0])
    poses_b = []
    for m in poses_a:
        mm = np.eye(4)
        mm[:3, :3] = R @ m[:3, :3]
        mm[:3, 3] = s * (R @ m[:3, 3]) + t
        poses_b.append(mm)
    s_est, R_est, t_est, aligned = similarity_alignment(poses_a, poses_b)
    assert abs(s_est - s) < 1e-9
    assert np.allclose(R_est, R, atol=1e-9)
    assert np.allclose(t_est, t, atol=1e-9)
    for a, b in zip(aligned, poses_b, strict=False):
        assert np.allclose(a, b, atol=1e-9)
