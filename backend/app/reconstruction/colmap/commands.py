"""Pure COLMAP command-line builders.

Keeping command construction separate from execution makes the exact CLI
invocations unit-testable without COLMAP installed.

COLMAP 4.x renamed option groups versus 3.x:
    3.x: --SiftExtraction.* / --SiftMatching.*
    4.x: --FeatureExtraction.* / --FeatureMatching.*
`ColmapProfile` abstracts this; `profile_for_executable()` detects the major
version from the binary once per executable. All commands explicitly request
CPU SIFT (use_gpu 0) so behavior is deterministic on machines without CUDA.
"""

from __future__ import annotations

import functools
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ColmapProfile:
    """Option-group names that differ between COLMAP 3.x and 4.x.

    In 3.x everything SIFT-related lives under --SiftExtraction./--SiftMatching.
    In 4.x the *extractor-level* options (use_gpu, max_image_size) moved to
    --FeatureExtraction. and matcher-level ones to --FeatureMatching., while
    SIFT parameters themselves stayed under --SiftExtraction.
    """

    extraction_gpu_prefix: str     # owns .use_gpu / .max_image_size
    matching_prefix: str           # owns .use_gpu / .guided_matching


LEGACY_3X = ColmapProfile("SiftExtraction", "SiftMatching")
MODERN_4X = ColmapProfile("FeatureExtraction", "FeatureMatching")


@functools.lru_cache(maxsize=8)
def profile_for_executable(executable: str) -> ColmapProfile:
    """Detect 3.x vs 4.x option groups from the binary's version banner."""
    import platform

    flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0  # type: ignore[attr-defined]
    try:
        out = subprocess.run(
            [str(executable), "--help"], capture_output=True, text=True, timeout=30,
            creationflags=flags,
        )
        banner = (out.stdout or "") + (out.stderr or "")
        m = re.search(r"COLMAP\s+(\d+)\.", banner)
        if m:
            return MODERN_4X if int(m.group(1)) >= 4 else LEGACY_3X
    except Exception:
        pass
    return MODERN_4X  # latest releases are the safe default


def feature_extractor_cmd(
    database_path: str,
    image_dir: str,
    profile: ColmapProfile = MODERN_4X,
    use_gpu: bool = False,
    single_camera: bool = False,
    max_features: int = 8192,
    max_image_size: int = 3200,
) -> list[str]:
    cmd = [
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        f"--{profile.extraction_gpu_prefix}.use_gpu", "1" if use_gpu else "0",
        "--SiftExtraction.max_num_features", str(max_features),
        f"--{profile.extraction_gpu_prefix}.max_image_size", str(max_image_size),
    ]
    if single_camera:
        cmd += ["--ImageReader.single_camera", "1"]
    return cmd


def exhaustive_matcher_cmd(database_path: str, profile: ColmapProfile = MODERN_4X,
                           use_gpu: bool = False,
                           guided: bool = False) -> list[str]:
    return [
        "exhaustive_matcher",
        "--database_path", database_path,
        f"--{profile.matching_prefix}.use_gpu", "1" if use_gpu else "0",
        f"--{profile.matching_prefix}.guided_matching", "1" if guided else "0",
    ]


def sequential_matcher_cmd(database_path: str, profile: ColmapProfile = MODERN_4X,
                           overlap: int = 10, use_gpu: bool = False) -> list[str]:
    return [
        "sequential_matcher",
        "--database_path", database_path,
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.quadratic_overlap", "1",
        f"--{profile.matching_prefix}.use_gpu", "1" if use_gpu else "0",
    ]


def vocab_tree_matcher_cmd(database_path: str, vocab_tree_path: str,
                           profile: ColmapProfile = MODERN_4X,
                           num_images: int = 100, use_gpu: bool = False) -> list[str]:
    return [
        "vocab_tree_matcher",
        "--database_path", database_path,
        "--VocabTreeMatching.vocab_tree_path", vocab_tree_path,
        "--VocabTreeMatching.num_images", str(num_images),
        f"--{profile.matching_prefix}.use_gpu", "1" if use_gpu else "0",
    ]


def mapper_cmd(database_path: str, image_dir: str, output_path: str) -> list[str]:
    # Default mapper settings are used: option names inside --Mapper.* changed
    # between releases, so we avoid hard-coding them here.
    return [
        "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", output_path,
    ]


def image_undistorter_cmd(image_dir: str, input_model: str, output_path: str) -> list[str]:
    return [
        "image_undistorter",
        "--image_path", image_dir,
        "--input_path", input_model,
        "--output_path", output_path,
        "--output_type", "COLMAP",
    ]


def matching_strategy(image_count: int, database_path: str,
                      profile: ColmapProfile = MODERN_4X,
                      exhaustive_max: int = 120,
                      vocab_tree_path: str | None = None) -> list[str]:
    """Choose a sensible matcher: exhaustive for small sets, otherwise
    sequential with quadratic overlap (safe default for capture sequences) or
    a vocab-tree when a tree file is available."""
    if image_count <= exhaustive_max:
        return exhaustive_matcher_cmd(database_path, profile)
    if vocab_tree_path:
        return vocab_tree_matcher_cmd(database_path, vocab_tree_path, profile)
    return sequential_matcher_cmd(database_path, profile)
