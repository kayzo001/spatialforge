#!/usr/bin/env python3
"""Download a published SpatialForge demo scene from the Hugging Face Hub.

Creates a read-only demo project in the local app with the downloaded
artifacts, clearly labeled as an externally published demo.

Usage:
    python scripts/hf_download.py --repo-id <user>/<repo>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    from backend.app import db
    from backend.app.config import settings

    local = Path(snapshot_download(args.repo_id, repo_type=args.repo_type))
    manifest_path = local / "manifest.json"
    if not manifest_path.is_file():
        sys.exit("Not a SpatialForge scene: manifest.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    project = db.create_project(
        name=f"HF demo - {args.repo_id.split('/')[-1]}",
        description=f"Downloaded from https://huggingface.co/{args.repo_id}",
        mode="cpu", preset=None, is_demo=True,
    )
    settings.ensure_project_dirs(project["id"])
    artifacts_dir = settings.project_dir(project["id"]) / "artifacts"

    for kind, info in manifest["artifacts"].items():
        src = local / info["filename"]
        if not src.is_file():
            print(f"[warn] missing in repo: {info['filename']}")
            continue
        dest = artifacts_dir / info["filename"]
        dest.write_bytes(src.read_bytes())
        db.add_artifact(project["id"], kind, info["filename"],
                        src.suffix.lstrip("."), src.stat().st_size)

    if "metrics" in manifest:
        for kind, data in manifest["metrics"].items():
            db.upsert_metrics(project["id"], kind, data)

    print(f"Imported as project {project['id']} - open the app and select it.")


if __name__ == "__main__":
    main()
