"""SpatialForge command-line interface.

Usage:
  python -m backend.app.cli serve                  # start API + frontend host
  python -m backend.app.cli capabilities           # print hardware capabilities
  python -m backend.app.cli demo [images_dir]      # create + reconstruct a demo project
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from backend.app import db
from backend.app.config import capabilities, settings
from backend.app.utils.logs import get_logger

logger = get_logger("spatialforge.cli")


def cmd_serve(_args: argparse.Namespace) -> None:
    import uvicorn

    db.get_conn()
    uvicorn.run("backend.app.main:app", host=settings.host, port=settings.port,
                log_level=settings.log_level.lower())


def cmd_capabilities(_args: argparse.Namespace) -> None:
    print(json.dumps(capabilities(force=True), indent=2))


class _CliJobContext:
    """Minimal JobContext stand-in that prints progress to stdout."""

    def __init__(self) -> None:
        self.job_id = "cli"
        self.project_id = "cli"
        self.cancel_event = threading.Event()

    def stage(self, stage_id: str, message: str | None = None) -> None:
        print(f"[stage] {stage_id}: {message or ''}")

    def progress(self, value: float, message: str | None = None) -> None:
        print(f"[{value:5.0%}] {message or ''}")

    def log(self, message: str, level: str = "info") -> None:
        print(f"[{level}] {message}")

    def check_cancel(self) -> None:
        pass


def cmd_demo(args: argparse.Namespace) -> None:
    from backend.app.reconstruction.pipeline import run_reconstruction
    from backend.app.services.upload_service import ingest_directory

    db.get_conn()
    images_dir = Path(args.images_dir).resolve() if args.images_dir else None
    if images_dir is None:
        default = settings.data_dir / "samples" / "synthetic_cube"
        if not default.is_dir():
            print("No dataset found. Generate one first:\n"
                  "  python scripts/generate_synthetic_dataset.py")
            sys.exit(2)
        images_dir = default

    name = args.name or f"Demo - {images_dir.name}"
    project = db.create_project(name, "Created via CLI demo", mode="cpu",
                                preset=args.preset, is_demo=True)
    settings.ensure_project_dirs(project["id"])
    print(f"Project {project['id']}: ingesting {images_dir}")
    report = ingest_directory(project["id"], images_dir)
    print(f"Ingested {len(report['added'])} images "
          f"({len(report['rejected'])} rejected)")

    ctx = _CliJobContext()
    project = db.get_project(project["id"])  # type: ignore[arg-type]
    result = run_reconstruction(ctx, project, {"preset": args.preset, "mode": "cpu",
                                               "eval_fraction": 0.0})
    db.update_project(project["id"], status="completed", error=None)
    print("\nReconstruction complete. Artifacts:")
    for kind, fname in result.get("artifacts", {}).items():
        if fname:
            print(f"  {kind}: {fname}")
    print(f"Open the app and select project {project['id']} to inspect the scene.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spatialforge",
                                     description="SpatialForge 3D reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="start the API server")
    sub.add_parser("capabilities", help="print hardware/dependency capabilities")
    demo = sub.add_parser("demo", help="ingest + reconstruct a local demo dataset")
    demo.add_argument("images_dir", nargs="?", help="directory with input photos")
    demo.add_argument("--name", help="project name")
    demo.add_argument("--preset", default="balanced",
                      choices=["fast_preview", "balanced", "high_quality"])
    args = parser.parse_args(argv)

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "capabilities":
        cmd_capabilities(args)
    elif args.command == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
