---
license: mit
library_name: spatialforge
tags:
- structure-from-motion
- 3d-reconstruction
- gaussian-splatting
- colmap
---

# SpatialForge demo scene — `<scene name>`

Precomputed reconstruction artifacts published by SpatialForge. Fill in every
placeholder before uploading — this template exists so published scenes can
never be mistaken for verified training runs.

## Contents

| file | description |
|---|---|
| `sparse_cloud.ply` | raw sparse point cloud from COLMAP SfM (x, y, z, r, g, b, error) |
| `refined_cloud.ply` | statistically filtered + voxel-downsampled cloud |
| `cameras.json` | per-image intrinsics + camera-to-world poses (OpenGL convention) |
| `scene_meta.json` | reconstruction metadata and artifact manifest |
| `splats.ply` | (optional) trained 3D Gaussian Splat scene from Splatfacto |
| `eval.json` | (optional) ns-eval metrics on held-out views |

## Reconstruction summary

- images uploaded: `<n>`
- images registered: `<n>`
- sparse points: `<n>`
- mean reprojection error: `<x>` px
- training: `<not run / preset, iterations>`
- evaluation (held-out): `<PSNR/SSIM/LPIPS or "not available">`

## Provenance

- Source dataset: `<dataset name, origin URL, license, and required citation>`
- Pipeline: COLMAP SfM + SpatialForge, `manifest.json` has exact details
- Coordinates: `cameras.json` stores camera-to-world matrices in the OpenGL
  convention — see SpatialForge `docs/coordinates.md`

## Intended use & limitations

Demonstration and education. Reconstruction quality depends entirely on the
capture; no warranty of metric accuracy beyond the numbers above, which are
reproduced from the artifact manifest.
