# Coordinate conventions

Camera math is where 3D pipelines silently break. This document fixes the
conventions used everywhere in SpatialForge and how they map to COLMAP,
Nerfstudio and three.js.

## The two representations

**COLMAP (input):** per-image *world-to-camera* pose (w2c) with an OpenCV
camera frame and unit quaternions in (w, x, y, z) order:

```
p_cam = R @ p_world + t        # R rows = camera axes expressed in world
camera center: C = -Rᵀ t
```

OpenCV camera frame: **x right, y down, z forward** (into the scene).

**SpatialForge / three.js (internal & viewer):** per-image *camera-to-world*
pose (c2w) in the OpenGL camera frame:

```
p_world = C2W @ p_cam          # column-vector convention, 4x4
```

OpenGL camera frame: **x right, y up, z backward** (three.js cameras look
down their local −z).

## The conversion (implemented in `reconstruction/colmap/transforms.py`)

1. Invert the rigid transform: `C2W_cv = [Rᵀ | −Rᵀ t]`.
2. Flip the camera's y and z axes (OpenCV → OpenGL):
   `C2W_gl = C2W_cv @ diag(1, −1, −1, 1)`.

The camera *center* is identical in both representations (the origin is on
the flip plane), which the unit tests assert explicitly.

## Why this matters (and is tested)

A reconstruction has arbitrary gauge (scale + rigid offset), so "is the pose
right?" is only answerable against **ground truth after similarity
alignment**. The synthetic dataset generator renders images from known poses;
the integration test then:

1. runs the full COLMAP pipeline,
2. converts estimated w2c → c2w (OpenGL),
3. aligns estimated camera centers to ground truth with a 7-DoF
   Umeyama/Procrustes fit (rotation, translation, uniform scale),
4. asserts mean rotation error < 3° and center error < 10% of the orbit
   radius.

Measured on this machine: **36/36 cameras, mean rotation error 0.32°, mean
center error 3 cm on a 2.1 m orbit**.

## Renderer notes

- `cameras.json` stores the OpenGL c2w matrices plus intrinsics (fx, fy, cx,
  cy) and vertical FOV. The viewer builds frustums directly from these.
- Nerfstudio does **not** need a converted file: the pipeline writes the
  undistorted COLMAP layout (`images/` + `sparse/`), which Nerfstudio's
  ColmapDataParser reads natively (it applies its own OpenCV→OpenGL
  conversion internally).
- PLY point clouds are exported in COLMAP world coordinates as-is; the
  viewer auto-frames them via bounding sphere.
