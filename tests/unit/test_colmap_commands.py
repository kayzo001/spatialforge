"""COLMAP command construction tests (no COLMAP required)."""

from __future__ import annotations

from backend.app.reconstruction.colmap.commands import (
    LEGACY_3X,
    MODERN_4X,
    exhaustive_matcher_cmd,
    feature_extractor_cmd,
    image_undistorter_cmd,
    mapper_cmd,
    matching_strategy,
    sequential_matcher_cmd,
    vocab_tree_matcher_cmd,
)


def test_feature_extractor_modern_profile_forces_cpu():
    cmd = feature_extractor_cmd("db.db", "images", MODERN_4X)
    joined = " ".join(cmd)
    # 4.x: extractor-level options moved to --FeatureExtraction.*
    assert "--FeatureExtraction.use_gpu 0" in joined
    assert "--FeatureExtraction.max_image_size 3200" in joined
    # SIFT parameters stay under --SiftExtraction in every version
    assert "--SiftExtraction.max_num_features 8192" in joined


def test_feature_extractor_legacy_profile():
    cmd = feature_extractor_cmd("db.db", "images", LEGACY_3X, single_camera=True)
    joined = " ".join(cmd)
    assert "--SiftExtraction.use_gpu 0" in joined
    assert "--SiftExtraction.max_image_size 3200" in joined
    assert "--ImageReader.single_camera 1" in joined


def test_matchers_use_version_specific_prefix():
    assert "--FeatureMatching.use_gpu 0" in " ".join(
        exhaustive_matcher_cmd("db.db", MODERN_4X))
    assert "--SiftMatching.use_gpu 0" in " ".join(
        exhaustive_matcher_cmd("db.db", LEGACY_3X))
    assert "--SequentialMatching.overlap 10" in " ".join(
        sequential_matcher_cmd("db.db", MODERN_4X))
    assert "--VocabTreeMatching.vocab_tree_path tree" in " ".join(
        vocab_tree_matcher_cmd("db.db", "tree", MODERN_4X))


def test_mapper_has_no_version_specific_options():
    cmd = mapper_cmd("db.db", "images", "out")
    assert cmd[:2] == ["mapper", "--database_path"]
    assert "--output_path" in cmd and "out" in cmd
    assert not any(o.startswith("--Mapper.") for o in cmd)


def test_undistorter_outputs_colmap_type():
    cmd = image_undistorter_cmd("images", "sparse/0", "undistorted")
    joined = " ".join(cmd)
    assert "--output_type COLMAP" in joined
    assert "--input_path sparse/0" in joined


def test_matching_strategy_small_dataset_is_exhaustive():
    cmd = matching_strategy(50, "db.db", MODERN_4X, exhaustive_max=120)
    assert cmd[0] == "exhaustive_matcher"


def test_matching_strategy_large_dataset_prefers_vocab_tree():
    cmd = matching_strategy(500, "db.db", MODERN_4X, exhaustive_max=120,
                            vocab_tree_path="tree.bin")
    assert cmd[0] == "vocab_tree_matcher"
    cmd = matching_strategy(500, "db.db", MODERN_4X, exhaustive_max=120)
    assert cmd[0] == "sequential_matcher"
