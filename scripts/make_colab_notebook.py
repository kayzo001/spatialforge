#!/usr/bin/env python3
"""Regenerates notebooks/SpatialForge_Colab.ipynb (development helper)."""

import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [
        md("""# SpatialForge on Google Colab — GPU reconstruction & Gaussian Splatting

Run the GPU-heavy parts of the SpatialForge pipeline on a Colab GPU, then
download the artifacts back to your laptop (or into the SpatialForge web app).

**Workflow**
1. Detect the assigned runtime hardware (GPU model, CUDA, disk).
2. Install COLMAP + a pinned Nerfstudio/Splatfacto environment (10-20 min).
3. Upload your photo set (or mount Google Drive).
4. Run Structure-from-Motion (COLMAP) -> sparse point cloud + camera poses.
5. Train **Splatfacto** (3D Gaussian Splatting) on the reconstructed scene.
6. Evaluate (PSNR / SSIM / LPIPS on held-out views) with `ns-eval`.
7. Export `splats.ply` + metrics and download everything.

> Every cell is defensive: if the runtime has no GPU or a package version
> changed, you get a clear error instead of silent garbage. Cell 1 MUST run first."""),
        code("""# @title 1. Runtime hardware detection (run this first)
import os, platform, subprocess, sys, shutil, json

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

print("python:", sys.version.split()[0], "| platform:", platform.machine())
gpu = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader")
if not gpu:
    raise RuntimeError(
        "No GPU detected. Go to Runtime > Change runtime type and select a GPU "
        "(T4 is sufficient for the balanced preset)."
    )
print("GPU:", gpu)
free_gb = sh("df -BG /content | tail -1 | awk '{print $4}'")
print("disk free:", free_gb)
cc = float(gpu.split(",")[-1])
assert cc >= 7.5, f"compute capability {cc} too old for Splatfacto (needs >= 7.5)"
print("OK")"""),
        code("""# @title 2. Install system dependencies + COLMAP (apt)
!apt-get update -qq
!apt-get install -y -qq colmap ffmpeg git-lfs > /dev/null
import subprocess
out = subprocess.run(["colmap", "--help"], capture_output=True, text=True).stdout
print(out.splitlines()[0] if out else "colmap installed (version unknown)")"""),
        md("""### 3. PyTorch + Nerfstudio (pinned, compatible versions)

Nerfstudio versions move fast; these pins are a tested combination for
Splatfacto on CUDA 12.x runtimes (Colab T4/L4/A100). If Colab's preinstalled
torch already matches the required CUDA build, the reinstall is skipped.

Known-good pins (update this table if you change anything):

| package | pin | note |
|---|---|---|
| torch | 2.4.1+cu121 | matches Colab CUDA 12 driver stack |
| nerfstudio | 1.1.3 | splatfacto stable |
| gsplat | per nerfstudio dep range | CUDA rasterizer |"""),
        code("""# @title 3. Install PyTorch (cu121) + Nerfstudio  (10-20 min)
import importlib.metadata as im, subprocess, sys

def pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

NEED_TORCH = "2.4.1"
torch_ver = None
try:
    torch_ver = im.version("torch")
except importlib.metadata.PackageNotFoundError:
    pass

if not torch_ver or not torch_ver.startswith(NEED_TORCH):
    pip("torch==2.4.1+cu121", "torchvision==0.19.1+cu121",
        "--index-url", "https://download.pytorch.org/whl/cu121")
else:
    print("torch already", torch_ver)

import torch
assert torch.cuda.is_available(), "CUDA unavailable after torch install"
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda,
      "| device:", torch.cuda.get_device_name(0))

pip("nerfstudio==1.1.3")
!ns-train --help > /dev/null && echo "nerfstudio CLI OK"
!git lfs install"""),
        md("""### 4. Bring your photo set

**Option A — upload a ZIP** of 20-150 sharp, overlapping photos (JPEG/PNG).
**Option B — Google Drive**: put the photos in a Drive folder and set
`DRIVE_FOLDER` below. Images are validated before anything expensive runs."""),
        code("""# @title 4a. Upload a ZIP of photos (or 4b. mount Drive)
from google.colab import files
import zipfile, pathlib

UPLOAD_ZIP = True  # @param {type:"boolean"}
DRIVE_FOLDER = ""  # @param {type:"string"}
DATA = pathlib.Path("/content/data"); DATA.mkdir(exist_ok=True)

if UPLOAD_ZIP:
    print("Choose a .zip with your photos:")
    up = files.upload()
    zpath = DATA / next(iter(up))
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        imgs = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        bad = [n for n in names if n.startswith(("/", chr(92))) or ".." in n]
        assert not bad, f"unsafe paths in zip: {bad}"
        zf.extractall(DATA / "images")
    zpath.unlink()
else:
    from google.colab import drive
    drive.mount("/content/drive")
    src = pathlib.Path("/content/drive/MyDrive") / DRIVE_FOLDER
    assert src.is_dir(), f"Drive folder not found: {src}"
    import shutil as _sh
    _sh.copytree(src, DATA / "images", dirs_exist_ok=True)

n = len(list((DATA / "images").rglob("*.jpg")) + list((DATA / "images").rglob("*.png")))
print(f"{n} images ready in {DATA / 'images'}")
assert n >= 3, "need at least three images"
"""),
        code("""# @title 5. Structure-from-Motion (COLMAP)
import pathlib, subprocess

DATA = pathlib.Path("/content/data")
IMAGES = DATA / "images"
COLMAP_DIR = DATA / "colmap"; COLMAP_DIR.mkdir(exist_ok=True)
DB = COLMAP_DIR / "database.db"
SPARSE = COLMAP_DIR / "sparse"; SPARSE.mkdir(exist_ok=True)
UNDIST = COLMAP_DIR / "undistorted"

def colmap(*args):
    r = subprocess.run(["colmap", *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"COLMAP {args[0]} failed ({r.returncode})")
    return r

n_images = len(list(IMAGES.glob("*.jpg")) + list(IMAGES.glob("*.png")))
colmap("feature_extractor", "--database_path", DB, "--image_path", IMAGES,
       "--FeatureExtraction.use_gpu", "1", "--SiftExtraction.max_num_features", "8192")
print("features extracted")

matcher = "exhaustive_matcher" if n_images <= 120 else "sequential_matcher"
colmap(matcher, "--database_path", DB, "--FeatureMatching.use_gpu", "1")
print("matched")

colmap("mapper", "--database_path", DB, "--image_path", IMAGES, "--output_path", SPARSE)
models = sorted(SPARSE.glob("*"))
assert models, "COLMAP registered no images - capture more overlap, avoid blur"
print(f"sparse model(s): {[m.name for m in models]}")

colmap("image_undistorter", "--image_path", IMAGES,
       "--input_path", models[0], "--output_path", UNDIST, "--output_type", "COLMAP")
print("undistorted training-ready layout at", UNDIST)"""),
        md("""### 6. Train Gaussian Splats (Splatfacto)

Presets (iterations): **fast_preview** 7k · **balanced** 15k · **high_quality** 30k.
On a T4, *balanced* takes roughly 20-45 min depending on resolution and image
count. Checkpoints are saved periodically, so you can stop early."""),
        code("""# @title 6. Splatfacto training
PRESET = "balanced"  # @param ["fast_preview", "balanced", "high_quality"]
import pathlib, subprocess

ITER = {"fast_preview": 7000, "balanced": 15000, "high_quality": 30000}[PRESET]
DOWNSCALE = {"fast_preview": 3, "balanced": 2, "high_quality": 1}[PRESET]
OUT = pathlib.Path("/content/outputs"); OUT.mkdir(exist_ok=True)

cmd = ["ns-train", "splatfacto",
       "--data", "/content/data/colmap/undistorted",
       "--output-dir", str(OUT),
       "--timestamp-name", "spatialforge",
       "--max-num-iterations", str(ITER),
       "--num-downscales", str(DOWNSCALE),
       "--viewer.quit-on-train-completion", "True"]
print(" ".join(cmd))
r = subprocess.run(cmd)
if r.returncode != 0:
    raise RuntimeError(f"ns-train failed ({r.returncode}). Free VRAM and retry "
                       "with the fast_preview preset.")"""),
        code("""# @title 7. Evaluate on held-out views (PSNR / SSIM / LPIPS)
import glob, json, pathlib, subprocess

configs = sorted(glob.glob("/content/outputs/**/config.yml", recursive=True),
                 key=lambda p: pathlib.Path(p).stat().st_mtime)
assert configs, "no training config found"
config = configs[-1]
print("using config:", config)

r = subprocess.run(["ns-eval", "--load-config", config,
                    "--output-path", "/content/eval.json"], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-1500:], r.stderr[-1500:])
    raise RuntimeError("ns-eval failed")
metrics = json.loads(pathlib.Path("/content/eval.json").read_text())
print(json.dumps({k: metrics[k] for k in metrics if k in ("psnr", "ssim", "lpips")}, indent=2))"""),
        code("""# @title 8. Export splats.ply + package artifacts for download
import glob, json, pathlib, shutil, subprocess

EXPORT = pathlib.Path("/content/export"); EXPORT.mkdir(exist_ok=True)
config = sorted(glob.glob("/content/outputs/**/config.yml", recursive=True),
                key=lambda p: pathlib.Path(p).stat().st_mtime)[-1]
r = subprocess.run(["ns-export", "gaussian-splat", "--load-config", config,
                    "--output-dir", str(EXPORT)], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-1500:], r.stderr[-1500:])
    raise RuntimeError("ns-export failed")

splat = max(EXPORT.glob("*.ply"), key=lambda p: p.stat().st_size)
pkg = pathlib.Path("/content/spatialforge_artifacts"); pkg.mkdir(exist_ok=True)
shutil.copy(splat, pkg / "splats.ply")
shutil.copytree("/content/data/colmap/undistorted/sparse", pkg / "sparse_model",
                dirs_exist_ok=True)
if pathlib.Path("/content/eval.json").exists():
    shutil.copy("/content/eval.json", pkg / "eval.json")
(pkg / "manifest.json").write_text(json.dumps({
    "generator": "SpatialForge Colab notebook",
    "splats_file": "splats.ply",
    "coordinate_convention": "COLMAP / OpenGL c2w, see docs/coordinates.md",
    "eval": json.loads(pathlib.Path("/content/eval.json").read_text())
            if pathlib.Path("/content/eval.json").exists() else None,
}, indent=2))
import subprocess as sp
sp.run(["zip", "-qr", "/content/spatialforge_artifacts.zip", "spatialforge_artifacts"],
       cwd="/content", check=True)
print("packaged:", splat.name, f"{splat.stat().st_size/1e6:.1f} MB")"""),
        code("""# @title 9. Download results (back to your laptop / into the web app)
from google.colab import files
files.download("/content/spatialforge_artifacts.zip")
print("In SpatialForge, place splats.ply into data/projects/<id>/artifacts/, "
      "register it (kind='splats'), or use scripts/hf_download.py for shared scenes.")"""),
        md("""## Troubleshooting

| symptom | fix |
|---|---|
| `No GPU detected` | Runtime → Change runtime type → GPU (T4 is enough) |
| torch/CUDA mismatch | rerun cell 3; it pins torch 2.4.1+cu121 |
| `ns-train` OOM | use `fast_preview`, or lower input resolution |
| COLMAP registers no images | more overlap, sharper photos, textured scenes |
| Colab disconnects mid-training | checkpoints live only in the session; copy `/content/outputs` to Drive early |
| version drift | pins live in cell 3; check the nerfstudio changelog before bumping |"""),
    ],
}

with open("notebooks/SpatialForge_Colab.ipynb", "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
print(f"wrote notebooks/SpatialForge_Colab.ipynb ({len(nb['cells'])} cells)")
