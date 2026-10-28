"""End-to-end integration test: real COLMAP, real pipeline, real artifacts.

Generates a small synthetic dataset with ground-truth poses, runs the full
CPU reconstruction through the backend pipeline, then verifies:

- reconstruction completes and writes real artifacts
- registered camera poses match ground truth after similarity alignment
  (this is the coordinate-correctness fixture required by the spec)

Skipped explicitly (not silently passed) when COLMAP is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.app.config import capabilities

pytestmark = pytest.mark.integration

_colmap = capabilities()["dependencies"]["colmap"]["available"]
if not _colmap:
    pytest.skip("COLMAP not available on this machine", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "generate_synthetic_dataset", REPO_ROOT / "scripts" / "generate_synthetic_dataset.py"
)
gen = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(gen)  # type: ignore[union-attr]


class _Ctx:
    job_id = "integration"
    project_id = "integration"
    cancel_event = None

    def __init__(self):
        import threading

        self.cancel_event = threading.Event()
        self.project_id = None

    def stage(self, stage_id, message=None):
        print(f"[stage] {stage_id}")

    def progress(self, value, message=None):
        pass

    def log(self, message, level="info"):
        print(f"[{level}] {message}")

    def check_cancel(self):
        pass


def test_full_pipeline_matches_ground_truth_poses(tmp_path, data_dir):
    from backend.app import db
    from backend.app.config import settings
    from backend.app.reconstruction.colmap.parser import read_reconstruction
    from backend.app.reconstruction.colmap.transforms import (
        colmap_pose_to_c2w_gl,
        similarity_alignment,
    )
    from backend.app.reconstruction.pipeline import run_reconstruction
    from backend.app.services.upload_service import ingest_directory

    # 1. small synthetic dataset with ground-truth poses
    dataset = tmp_path / "synthetic"
    gen.generate(dataset, views=36, width=800, height=600)
    gt = json.loads((dataset / "gt_poses.json").read_text())
    gt_by_name = {i["name"]: np.array(i["c2w_opencv"]) for i in gt["images"]}

    # 2. run the real pipeline
    project = db.create_project("Integration", "", mode="cpu", preset="balanced")
    settings.ensure_project_dirs(project["id"])
    ingest_directory(project["id"], dataset)
    ctx = _Ctx()
    ctx.project_id = project["id"]
    scene = run_reconstruction(ctx, db.get_project(project["id"]),
                               {"preset": "balanced", "mode": "cpu",
                                "eval_fraction": 0.0})

    # 3. artifacts exist and are non-trivial
    art = settings.project_dir(project["id"]) / "artifacts"
    assert (art / "sparse_cloud.ply").stat().st_size > 1000
    assert (art / "refined_cloud.ply").stat().st_size > 500
    assert (art / "cameras.json").is_file()
    assert scene["images_registered"] >= 30

    # 4. pose correctness vs ground truth (after 7-DoF alignment)
    model = sorted((settings.project_dir(project["id"]) / "colmap" / "sparse").iterdir())
    model_dir = max(model, key=lambda d: (d / "images.bin").stat().st_size)
    recon = read_reconstruction(model_dir)

    names, gt_poses, est_poses = [], [], []
    for img in recon.images.values():
        if img.name in gt_by_name:
            c2w_gl = colmap_pose_to_c2w_gl(img.qvec, img.tvec)
            names.append(img.name)
            est_poses.append(c2w_gl @ np.diag([1, -1, -1, 1.0]))
            gt_poses.append(gt_by_name[img.name])
    assert len(names) >= 30

    _, _, _, aligned = similarity_alignment(est_poses, gt_poses)
    rot_errs, center_errs = [], []
    for est, gt_pose in zip(aligned, gt_poses, strict=False):
        r_err = est[:3, :3].T @ gt_pose[:3, :3]
        rot_errs.append(np.degrees(np.arccos(np.clip((np.trace(r_err) - 1) / 2, -1, 1))))
        center_errs.append(np.linalg.norm(est[:3, 3] - gt_pose[:3, 3]))

    assert np.mean(rot_errs) < 3.0, f"mean rotation error {np.mean(rot_errs):.3f} deg"
    orbit_radius = float(np.mean([np.linalg.norm(p[:3, 3]) for p in gt_poses]))
    assert np.mean(center_errs) < 0.1 * orbit_radius, (
        f"mean center error {np.mean(center_errs):.4f} vs radius {orbit_radius}"
    )

    # 5. metrics persisted and honest
    metrics = {m["kind"]: m["data"] for m in db.get_metrics(project["id"])}
    assert metrics["colmap"]["num_points3d"] > 500
    assert metrics["colmap"]["mean_reprojection_error_px"] < 1.5
    assert metrics["training"]["status"] == "not_requested"  # CPU mode: honest
