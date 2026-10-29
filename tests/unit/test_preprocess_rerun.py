"""Re-running preprocessing must not duplicate images (regression test)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from backend.app.reconstruction.preprocessing.preprocess import preprocess_dataset


def _make_originals(folder: Path, n: int = 4) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"view_{i:03d}.png"
        Image.new("RGB", (64, 48), (i * 40, 80, 120)).save(p)
        paths.append(p)
    return paths


def test_rerun_does_not_create_suffixed_duplicates(tmp_path: Path) -> None:
    originals = _make_originals(tmp_path / "originals")
    images = tmp_path / "images"

    first = preprocess_dataset(originals, images, max_edge=1600)
    second = preprocess_dataset(originals, images, max_edge=1600)

    names = sorted(p.name for p in images.iterdir())
    assert names == [f"view_{i:03d}.jpg" for i in range(4)]
    assert [r["processed_name"] for r in first] == [r["processed_name"] for r in second]


def test_same_stem_inputs_still_get_unique_names(tmp_path: Path) -> None:
    src = tmp_path / "originals"
    src.mkdir()
    a = src / "shot.png"
    b = src / "shot.jpg"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(a)
    Image.new("RGB", (32, 32), (40, 50, 60)).save(b)

    out = preprocess_dataset([a, b], tmp_path / "images")

    assert sorted(r["processed_name"] for r in out) == ["shot.jpg", "shot_1.jpg"]


def test_colmap_cache_invalidated_when_inputs_change(tmp_path: Path, monkeypatch) -> None:
    from backend.app.reconstruction import pipeline

    monkeypatch.setattr(pipeline.settings, "use_cache", True)
    a = [{"processed_name": "view_000.jpg", "processed_size": [800, 600]}]
    b = a + [{"processed_name": "view_000_1.jpg", "processed_size": [800, 600]}]

    assert not pipeline._colmap_cache_valid(tmp_path, a)  # no fingerprint yet
    pipeline._write_colmap_fingerprint(tmp_path, b)
    assert pipeline._colmap_cache_valid(tmp_path, b)
    assert not pipeline._colmap_cache_valid(tmp_path, a)  # image set changed

    monkeypatch.setattr(pipeline.settings, "use_cache", False)
    assert not pipeline._colmap_cache_valid(tmp_path, b)
