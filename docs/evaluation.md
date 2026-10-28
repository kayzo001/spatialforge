# Evaluation & benchmarks

## What is measured

| metric | source | availability |
|---|---|---|
| registered images / ratio | COLMAP model parsing | always (CPU mode) |
| sparse point count, track length, observations | `points3D.bin` parsing | always |
| mean reprojection error | COLMAP per-point error field | always |
| PSNR | NumPy implementation (`evaluation/metrics.py`) | always |
| SSIM | NumPy/OpenCV Gaussian-window implementation (Wang et al. 2004) | always |
| LPIPS | torch + lpips package | only when installed |
| novel-view PSNR/SSIM/LPIPS | `ns-eval` on held-out cameras | GPU training path |

The UI renders **"Not available"** for missing metrics; it never invents
numbers.

## Holdout protocol

`evaluation/holdout.py` selects evaluation images deterministically
(SHA-256 order, stable across runs; default fraction 0.1, only for ≥ 10
images). During training the Nerfstudio datamanager applies its own
train/eval split of the undistorted dataset, and `ns-eval` reports metrics on
held-out cameras — training-image reconstruction quality is therefore never
presented as novel-view performance.

## Known metric limitations

- **PSNR/SSIM** are implemented in-house (NumPy/OpenCV). The SSIM follows the
  reference formulation (11×11 Gaussian window, C1/C2 stabilizers) on
  grayscale, matching common NVS-paper practice, but values can differ in the
  third decimal from scikit-image depending on border handling.
- **LPIPS** loads an AlexNet backbone — a perceptual metric, not a fidelity
  metric; it is only shown when the dependency exists.
- The **viewer's** rendered quality is WebGL-limited and not a benchmark.

## Reproducing

```bash
# geometric stats for any completed project live in the API:
curl http://127.0.0.1:8000/api/projects/<id>/metrics
```

`scripts/prepare_demo.py` + the integration test
(`tests/integration/test_pipeline_e2e.py`) reproduce the full pipeline
end-to-end, including ground-truth pose verification.

## Verified measurements (this machine)

Environment: Windows 10, i5-12500H (12 cores), 15.6 GB RAM, COLMAP 4.2.1
(CPU build), Open3D 0.19.

Dataset: synthetic textured cube + ground, 36 views, 800×600
(`scripts/generate_synthetic_dataset.py`), exhaustive CPU matching.

| measurement | value |
|---|---|
| images registered | 36/36 |
| sparse points | ~3,200 |
| mean reprojection error | 0.41 px |
| pipeline runtime (CPU, incl. matching) | ~21 s |
| camera pose error vs GT (after 7-DoF alignment) | mean 0.32° rotation, 3 cm on a 2.1 m orbit |
| Open3D refinement | 2,984 → 1,953 points (voxel 0.038) |

No GPU-training benchmark is reported because no Nerfstudio/CUDA environment
was available on the verification machine; that path is exercised in Colab
(`docs/colab.md`) and its numbers belong there once produced.
