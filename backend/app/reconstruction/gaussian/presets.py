"""User-facing training presets for Splatfacto (Nerfstudio).

Preset values are conservative starting points, tuned for 8 GB-class GPUs and
scaled down when the detected GPU has less memory. None of these values
promise a runtime; actual performance depends on the dataset and hardware.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingPreset:
    key: str
    label: str
    description: str
    max_num_iterations: int
    num_downscales: int            # additional image pyramid levels nerfstudio generates
    steps_per_eval_batch: int
    steps_per_eval_image: int
    steps_per_save: int
    cull_threshold: float
    densify_grad_thresh: float
    # Conservative cap used to request a smaller cap on low-VRAM GPUs (3DGS
    # grows the Gaussian count during densification):
    max_scene_num_gaussians: int | None


PRESETS: dict[str, TrainingPreset] = {
    "fast_preview": TrainingPreset(
        key="fast_preview",
        label="Fast Preview",
        description="Quick iteration pass with reduced resolution and iterations.",
        max_num_iterations=7_000,
        num_downscales=3,
        steps_per_eval_batch=2_000,
        steps_per_eval_image=2_000,
        steps_per_save=3_500,
        cull_threshold=0.008,
        densify_grad_thresh=0.0005,
        max_scene_num_gaussians=1_500_000,
    ),
    "balanced": TrainingPreset(
        key="balanced",
        label="Balanced",
        description="Balance of speed, geometry and appearance quality.",
        max_num_iterations=15_000,
        num_downscales=2,
        steps_per_eval_batch=2_500,
        steps_per_eval_image=2_500,
        steps_per_save=7_500,
        cull_threshold=0.006,
        densify_grad_thresh=0.0004,
        max_scene_num_gaussians=3_000_000,
    ),
    "high_quality": TrainingPreset(
        key="high_quality",
        label="High Quality",
        description="Prioritizes reconstruction detail; slowest option.",
        max_num_iterations=30_000,
        num_downscales=1,
        steps_per_eval_batch=3_000,
        steps_per_eval_image=3_000,
        steps_per_save=10_000,
        cull_threshold=0.005,
        densify_grad_thresh=0.00035,
        max_scene_num_gaussians=6_000_000,
    ),
}


def get_preset(key: str) -> TrainingPreset:
    return PRESETS.get(key, PRESETS["balanced"])


def downscale_preset_for_vram(preset: TrainingPreset, vram_mb: int | None) -> TrainingPreset:
    """On small GPUs (<6 GB) swap to the lighter preset rather than risk OOM.

    Returns a preset object; possibly the same one when resources suffice.
    """
    if vram_mb is None:
        return preset
    if vram_mb < 6144 and preset.key != "fast_preview":
        lighter = PRESETS["fast_preview"]
        return TrainingPreset(
            key=preset.key,
            label=preset.label,
            description=preset.description + " (auto-reduced for a small GPU)",
            max_num_iterations=min(preset.max_num_iterations, 10_000),
            num_downscales=lighter.num_downscales,
            steps_per_eval_batch=lighter.steps_per_eval_batch,
            steps_per_eval_image=lighter.steps_per_eval_image,
            steps_per_save=max(preset.max_num_iterations // 2, lighter.steps_per_save),
            cull_threshold=lighter.cull_threshold,
            densify_grad_thresh=lighter.densify_grad_thresh,
            max_scene_num_gaussians=lighter.max_scene_num_gaussians,
        )
    return preset
