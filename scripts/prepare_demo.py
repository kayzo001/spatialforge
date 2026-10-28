#!/usr/bin/env python3
"""Prepare the demo project end-to-end (dataset -> reconstruction -> artifacts).

Steps:
1. Ensures a dataset exists (downloads an open one, or generates the
   synthetic cube which has no license constraints).
2. Runs the full CPU reconstruction pipeline through the backend CLI.
3. Prints how to inspect the result in the web app.

Usage:
    python scripts/prepare_demo.py [--synthetic] [--dataset DIR]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def ensure_dataset(synthetic: bool, dataset: str | None) -> Path:
    if dataset:
        p = Path(dataset)
        if not p.is_dir():
            sys.exit(f"Dataset directory not found: {p}")
        return p
    if synthetic:
        p = REPO_ROOT / "data" / "samples" / "synthetic_cube"
        if not p.is_dir():
            print("Generating synthetic dataset...")
            subprocess.run([PY, str(REPO_ROOT / "scripts" / "generate_synthetic_dataset.py")],
                           check=True)
        return p
    p = REPO_ROOT / "data" / "samples" / "south_building"
    if p.is_dir() and any(p.glob("*.jpg")):
        return p
    print("Fetching open sample dataset (or fall back to --synthetic)...")
    result = subprocess.run([PY, str(REPO_ROOT / "scripts" / "fetch_sample_dataset.py")])
    if result.returncode != 0:
        print("Download unavailable; generating the synthetic dataset instead.")
        return ensure_dataset(synthetic=True, dataset=None)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="use the generated synthetic cube dataset (no download)")
    ap.add_argument("--dataset", help="use a specific directory of images")
    ap.add_argument("--preset", default="balanced")
    args = ap.parse_args()

    dataset = ensure_dataset(args.synthetic, args.dataset)
    print(f"Running demo reconstruction on {dataset}")
    subprocess.run([
        PY, "-m", "backend.app.cli", "demo", str(dataset),
        "--name", f"Demo - {dataset.name}", "--preset", args.preset,
    ], check=True, cwd=REPO_ROOT)
    print("\nStart the app (make dev:backend / make dev:frontend or "
          "python -m backend.app.cli serve) and open the Dashboard to inspect it.")


if __name__ == "__main__":
    main()
