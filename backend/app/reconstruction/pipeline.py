"""Reconstruction pipeline orchestration (Stages 2-6).

`run_reconstruction(ctx, params)` executes every stage of the CPU geometric
pipeline, publishing live stage/progress/log events through the JobContext,
persisting artifacts and metrics, and - when a CUDA GPU with Nerfstudio is
available and the project requests it - continuing into Gaussian-Splatting
training. Failure at any stage fails the job with an actionable message;
camera poses are never fabricated.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from backend.app import db
from backend.app.config import capabilities, settings
from backend.app.reconstruction.colmap import commands as cm
from backend.app.reconstruction.colmap.parser import ModelParseError, read_reconstruction
from backend.app.reconstruction.colmap.runner import ColmapError, run_colmap
from backend.app.reconstruction.gaussian.nerfstudio import train_splatfacto
from backend.app.reconstruction.gaussian.presets import get_preset
from backend.app.reconstruction.geometry.exporter import (
    write_cameras_json,
    write_ply_binary,
    write_scene_meta,
)
from backend.app.reconstruction.geometry.pointcloud import (
    OPEN3D_AVAILABLE,
    decimate_to_budget,
    process_point_cloud,
)
from backend.app.reconstruction.preprocessing.preprocess import preprocess_dataset
from backend.app.services.job_manager import JobContext
from backend.app.utils.files import human_size
from backend.app.utils.logs import get_logger

logger = get_logger("spatialforge.pipeline")


class PipelineError(RuntimeError):
    """Reconstruction failure with a user-facing explanation."""


def _find_model_dir(root: Path) -> Path | None:
    """Return the largest COLMAP sparse model under `root`.

    The mapper may emit several models (sparse/0, sparse/1, ...); fragments
    can be tiny. The model with the biggest images.bin is the primary one.
    """
    candidates: list[Path] = []
    if (root / "cameras.bin").is_file():
        candidates.append(root)
    elif root.is_dir():
        candidates.extend(d for d in sorted(root.iterdir())
                          if (d / "cameras.bin").is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "images.bin").stat().st_size)


def run_reconstruction(ctx: JobContext, project: dict, params: dict) -> dict[str, Any]:
    t_start = time.time()
    project_id = project["id"]
    dirs = settings.ensure_project_dirs(project_id)
    images_dir: Path = dirs["images"]
    colmap_dir: Path = dirs["colmap"]
    artifacts_dir: Path = dirs["artifacts"]

    preset = get_preset(params.get("preset") or project.get("preset") or "balanced")
    eval_fraction = float(params.get("eval_fraction", 0.1))
    do_training = (params.get("mode") or project.get("mode") or "cpu") == "gpu"

    exe = capabilities().get("dependencies", {}).get("colmap", {}).get("path")
    if not exe:
        raise PipelineError(
            "COLMAP is not installed or not found. Install it via "
            "scripts/install_colmap.py (see docs/installation.md), then retry."
        )
    from pathlib import Path as _P

    colmap_exe = _P(exe)
    from backend.app.reconstruction.colmap.commands import profile_for_executable

    profile = profile_for_executable(str(colmap_exe))

    image_rows = db.list_images(project_id)
    if len(image_rows) < 3:
        raise PipelineError(
            f"Only {len(image_rows)} image(s) uploaded; at least 3 overlapping views "
            "are required for reconstruction."
        )

    # ---- Stage 2: preprocessing -----------------------------------------
    ctx.stage("preprocessing", "Normalizing EXIF and resizing images")
    ctx.progress(0.02)
    originals = [Path(r["filepath"]) for r in image_rows]
    processed = preprocess_dataset(originals, images_dir, settings.max_image_size)
    failed = [p for p in processed if "error" in p]
    if failed:
        ctx.log(f"{len(failed)} image(s) failed preprocessing and were skipped", "warning")
    ok_processed = [p for p in processed if "error" not in p]
    if len(ok_processed) < 3:
        raise PipelineError("Too few images survived preprocessing to reconstruct.")
    (colmap_dir / "processed_map.json").write_text(
        json.dumps(ok_processed, indent=1), encoding="utf-8")
    ctx.log(f"Preprocessed {len(ok_processed)} images "
            f"(longest edge <= {settings.max_image_size}px)")
    ctx.progress(0.15)

    # eval split is chosen from processed names, deterministically
    from backend.app.reconstruction.evaluation.holdout import choose_eval_images

    eval_names = set(choose_eval_images([p["processed_name"] for p in ok_processed],
                                        eval_fraction))
    for p in ok_processed:
        db.set_image_split(project_id, Path(p["source"]).name,
                           "eval" if p["processed_name"] in eval_names else "train")
    if eval_names:
        ctx.log(f"Held out {len(eval_names)} image(s) for evaluation: "
                f"{sorted(eval_names)}")

    # ---- Stage 3: COLMAP SfM ---------------------------------------------
    database_path = colmap_dir / "database.db"
    if database_path.exists() and not settings.use_cache:
        database_path.unlink()
    for stale in ("sparse", "undistorted"):
        target = colmap_dir / stale
        if target.exists() and not settings.use_cache:
            shutil.rmtree(target, ignore_errors=True)

    ctx.stage("feature_extraction", "Extracting SIFT features (CPU)")
    try:
        run_colmap(
            cm.feature_extractor_cmd(str(database_path), str(images_dir), profile),
            colmap_exe, log=ctx.log, cancel_event=ctx.cancel_event,
        )
    except ColmapError as exc:
        raise PipelineError(f"Feature extraction failed: {exc}. "
                            "Check that images are readable.") from exc
    ctx.progress(0.28)

    ctx.stage("matching", "Matching features between images")
    n_images = len(ok_processed)
    try:
        run_colmap(
            cm.matching_strategy(n_images, str(database_path), profile,
                                 settings.exhaustive_match_max_images),
            colmap_exe, log=ctx.log, cancel_event=ctx.cancel_event,
        )
    except ColmapError as exc:
        raise PipelineError(
            f"Feature matching failed: {exc}. Low-overlap datasets are the usual cause."
        ) from exc
    ctx.progress(0.5)

    ctx.stage("sparse_reconstruction", "Estimating camera poses and sparse geometry")
    sparse_root = colmap_dir / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=True)  # COLMAP 4.x requires it to exist
    try:
        run_colmap(
            cm.mapper_cmd(str(database_path), str(images_dir), str(sparse_root)),
            colmap_exe, log=ctx.log, cancel_event=ctx.cancel_event,
        )
    except ColmapError as exc:
        raise PipelineError(
            f"Sparse reconstruction failed: {exc}. Try more overlapping images, "
            "avoid motion blur, and ensure the scene has enough texture."
        ) from exc

    model_dir = _find_model_dir(sparse_root)
    if model_dir is None:
        raise PipelineError(
            "COLMAP registered no images. Common causes: insufficient viewpoint "
            "overlap, textureless/reflective scenes, or blurry images. "
            "Recommendation: capture 30-80 sharp, well-overlapping photos."
        )

    try:
        recon = read_reconstruction(model_dir)
    except ModelParseError as exc:
        raise PipelineError(f"Could not parse the COLMAP model: {exc}") from exc

    registered = len(recon.images)
    registration_ratio = registered / max(1, n_images)
    ctx.log(f"Registered {registered}/{n_images} images "
            f"({registration_ratio:.0%}), {recon.num_points} sparse points")
    if registered < 3:
        raise PipelineError(
            f"Only {registered} image(s) registered (3 minimum). The scene likely "
            "lacks overlap or texture. See docs/troubleshooting.md."
        )
    if registration_ratio < 0.5:
        ctx.log(
            f"Only {registration_ratio:.0%} of images registered; results will be "
            "incomplete. Consider recapturing the missing viewpoints.", "warning")
    ctx.progress(0.68)

    # undistorted copy in the Nerfstudio-ready layout (images/ + sparse/)
    undistorted = colmap_dir / "undistorted"
    if not undistorted.exists() or not settings.use_cache:
        ctx.log("Undistorting images into the training-ready layout")
        undistorted.mkdir(parents=True, exist_ok=True)
        try:
            run_colmap(
                cm.image_undistorter_cmd(str(images_dir), str(model_dir),
                                         str(undistorted)),
                colmap_exe, log=ctx.log, cancel_event=ctx.cancel_event,
            )
        except ColmapError as exc:
            ctx.log(f"Undistortion failed ({exc}); continuing with distorted frames",
                    "warning")
            undistorted = None  # type: ignore[assignment]

    # ---- Stage 4: artifacts + point processing ----------------------------
    ctx.stage("point_processing", "Writing artifacts and refining the point cloud")
    sparse_ply = artifacts_dir / "sparse_cloud.ply"
    if recon.points is not None and len(recon.points):
        count = write_ply_binary(
            sparse_ply, recon.points.xyz, recon.points.rgb, recon.points.error)
        db.add_artifact(project_id, "sparse_cloud", sparse_ply.name, "ply",
                        sparse_ply.stat().st_size)
        ctx.log(f"Exported sparse cloud: {count} points ({human_size(sparse_ply.stat().st_size)})")
    else:
        ctx.log("No sparse points were reconstructed; skipping cloud artifacts", "warning")

    cameras_json = artifacts_dir / "cameras.json"
    write_cameras_json(cameras_json, recon)
    db.add_artifact(project_id, "cameras", cameras_json.name, "json",
                    cameras_json.stat().st_size)

    colmap_metrics = recon.stats()
    colmap_metrics["images_uploaded"] = n_images
    colmap_metrics["registration_ratio"] = round(registration_ratio, 4)

    refined_stats: dict[str, Any] = {"backend": "skipped"}
    if recon.points is not None and len(recon.points):
        extent = max((recon.points.xyz.max(axis=0) - recon.points.xyz.min(axis=0)).max(), 1e-6)
        voxel = min(0.5, max(0.005, extent / 512.0))
        refined_xyz, refined_rgb, refined_stats = process_point_cloud(
            recon.points.xyz, recon.points.rgb, voxel_size=voxel)
        refined_ply = artifacts_dir / "refined_cloud.ply"
        count = write_ply_binary(refined_ply, refined_xyz, refined_rgb)
        db.add_artifact(project_id, "refined_cloud", refined_ply.name, "ply",
                        refined_ply.stat().st_size)
        ctx.log(f"Refined cloud: {refined_stats['input_points']} -> {count} points "
                f"(voxel {voxel:.4f}, backend {refined_stats['backend']})")
        # lightweight viewer copy for very large clouds
        view_xyz, view_rgb, kept = decimate_to_budget(refined_xyz, refined_rgb, 1_500_000)
        viewer_ply = artifacts_dir / "viewer_cloud.ply"
        if kept < count:
            write_ply_binary(viewer_ply, view_xyz, view_rgb)
            db.add_artifact(project_id, "viewer_cloud", viewer_ply.name, "ply",
                            viewer_ply.stat().st_size)
    ctx.progress(0.8)

    db.upsert_metrics(project_id, "colmap", {
        **colmap_metrics, "point_processing": refined_stats,
        "open3d_available": OPEN3D_AVAILABLE,
    })

    # ---- Stage 5: neural training (only when explicitly requested + capable) -
    training_available = capabilities()["modes"]["local_gpu_training"]
    if do_training and not training_available:
        ctx.log(
            "GPU training requested but Nerfstudio/CUDA are unavailable on this "
            "machine. Geometric reconstruction artifacts are complete; use the "
            "bundled Colab notebook for Gaussian-Splatting training.", "warning")
        db.upsert_metrics(project_id, "training", {
            "status": "not_available",
            "reason": "Nerfstudio + CUDA not detected on this machine",
        })
    elif do_training:
        ctx.stage("training", "Training Gaussian Splats (Splatfacto)")
        data_for_training = undistorted if undistorted else colmap_dir / "sparse"
        try:
            result = train_splatfacto(
                data_dir=data_for_training,
                output_dir=settings.project_dir(project_id) / "outputs",
                preset=preset, log=ctx.log, cancel_event=ctx.cancel_event,
            )
            splat_ply = result.pop("splat_ply", None)
            if splat_ply:
                src = Path(splat_ply)
                dest = artifacts_dir / "splats.ply"
                shutil.copy2(src, dest)
                db.add_artifact(project_id, "splats", dest.name, "ply",
                                dest.stat().st_size)
                ctx.log(f"Exported Gaussian splats: {human_size(dest.stat().st_size)}")
            db.upsert_metrics(project_id, "training", result)
            ctx.progress(0.95)
        except Exception as exc:  # noqa: BLE001 - training failure shouldn't void geometry
            ctx.log(f"Gaussian training failed: {exc}", "error")
            db.upsert_metrics(project_id, "training", {"status": "failed",
                                                       "error": str(exc)})
    else:
        db.upsert_metrics(project_id, "training", {
            "status": "not_requested",
            "reason": "CPU mode: geometric reconstruction only. Run in GPU mode "
                      "or via the Colab notebook for Gaussian Splatting.",
        })

    # ---- Stage 6: finalize -------------------------------------------------
    ctx.stage("export", "Finalizing scene metadata")
    scene_meta = {
        "project_id": project_id,
        "images_uploaded": n_images,
        "images_registered": registered,
        "num_points3d": colmap_metrics["num_points3d"],
        "eval_images": sorted(eval_names),
        "preset": preset.key,
        "training_available_locally": training_available,
        "artifacts": {
            "sparse_cloud": sparse_ply.name if sparse_ply.exists() else None,
            "refined_cloud": (artifacts_dir / "refined_cloud.ply").name
            if (artifacts_dir / "refined_cloud.ply").exists() else None,
            "cameras": cameras_json.name,
            "splats": (artifacts_dir / "splats.ply").name
            if (artifacts_dir / "splats.ply").exists() else None,
        },
        "coordinate_convention": "c2w OpenGL (three.js); see docs/coordinates.md",
    }
    write_scene_meta(artifacts_dir / "scene_meta.json", scene_meta)
    db.add_artifact(project_id, "scene_meta", "scene_meta.json", "json",
                    (artifacts_dir / "scene_meta.json").stat().st_size)

    db.update_project(project_id, stats_json=json.dumps({
        "images_registered": registered,
        "images_uploaded": n_images,
        "num_points3d": colmap_metrics["num_points3d"],
        "mean_reprojection_error_px": colmap_metrics["mean_reprojection_error_px"],
        "duration_s": round(time.time() - t_start, 1),
    }))

    ctx.log(f"Reconstruction finished in {time.time() - t_start:.1f}s")
    ctx.progress(1.0)
    return scene_meta
