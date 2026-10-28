#!/usr/bin/env python3
"""Download an openly licensed multi-view sample dataset.

Primary source: the "South Building" dataset (188 images) hosted by the
COLMAP authors at https://demuc.de/colmap/ - provided free for research use;
cite "Schonberger and Frahm, Structure-from-Motion Revisited, CVPR 2016" when
using it. If the primary mirror is unreachable, the script lists manual
alternatives (Middlebury multi-view: https://vision.middlebury.edu/mview/data/,
free for research; cite Seitz et al., "A Comparison and Evaluation of
Multi-View Stereo Reconstruction Algorithms", CVPR 2006).

Usage:
    python scripts/fetch_sample_dataset.py [--out data/samples/south_building]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

CANDIDATES = [
    ("South Building (COLMAP authors, free for research)",
     "https://demuc.de/colmap/South-Building.zip"),
]

ALTERNATIVES = """
No mirror was reachable. Manual options:
  1. South Building:  https://demuc.de/colmap/        (free for research)
  2. Middlebury temple datasets:
     https://vision.middlebury.edu/mview/data/        (free for research)
  3. Generate a fully open synthetic dataset instead:
     python scripts/generate_synthetic_dataset.py
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/samples/south_building")
    args = ap.parse_args()
    out = Path(args.out)
    if out.is_dir() and any(out.glob("*.jpg")):
        print(f"Dataset already present at {out}")
        return

    for label, url in CANDIDATES:
        print(f"Trying {label}\n  {url}")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = Path(tmp) / "dataset.zip"
                with urlopen(url, timeout=1200) as resp, open(zip_path, "wb") as f:  # noqa: S310
                    shutil.copyfileobj(resp, f, length=1 << 20)
                print("Extracting...")
                with zipfile.ZipFile(zip_path) as zf:
                    names = zf.namelist()
                    images = [n for n in names if n.lower().endswith((".jpg", ".png"))]
                    zf.extractall(out)
                # normalize: ensure images sit directly under out/
                out_images = list(out.rglob("*.jpg")) + list(out.rglob("*.png"))
                if out_images and not list(out.glob("*.jpg")):
                    for img in out_images:
                        shutil.move(str(img), str(out / img.name))
                    for p in sorted(out.rglob("*"), reverse=True):
                        if p.is_dir() and not any(p.iterdir()):
                            p.rmdir()
                print(f"Done: {len(images)} images in {out}\n"
                      "License: free for research - cite the source above.")
                return
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")

    print(ALTERNATIVES)
    sys.exit(1)


if __name__ == "__main__":
    main()
