#!/usr/bin/env python3
"""Download and install a COLMAP Windows/Linux release binary into tools/colmap.

Usage:
    python scripts/install_colmap.py [--version latest] [--dir tools/colmap]

- Queries the GitHub releases API for the newest COLMAP release.
- On Windows picks the *no-CUDA* build by default (smaller, CPU feature
  extraction/matching is sufficient for reconstruction; CUDA build optional).
- On Linux prints distribution install instructions (apt/dnf packages are the
  reliable path there).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.github.com/repos/colmap/colmap/releases/latest"


def latest_release() -> dict:
    with urlopen(API_URL, timeout=30) as r:  # noqa: S310 - fixed https URL
        return json.loads(r.read().decode("utf-8"))


def pick_asset(release: dict) -> tuple[str, str] | None:
    """Return (asset_name, download_url) for the current platform."""
    assets = release.get("assets", [])
    win = [a for a in assets if "windows" in a["name"].lower()
           and a["name"].lower().endswith(".zip")]
    if win:
        nocuda = [a for a in win if "nocuda" in a["name"].lower()]
        chosen = nocuda[0] if nocuda else win[0]
        return chosen["name"], chosen["browser_download_url"]
    return None


def install_windows(url: str, dest: Path) -> Path:
    print(f"Downloading {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "colmap.zip"
        with urlopen(url, timeout=600) as resp, open(zip_path, "wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out)
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # release zips usually contain a single top-level folder
        entries = [p for p in Path(tmp).iterdir() if p.is_dir()]
        src = entries[0] if entries else Path(tmp)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
    exe = dest / "bin" / "colmap.exe"
    if not exe.is_file():  # some layouts put colmap.exe at the root
        exe = dest / "colmap.exe"
    if not exe.is_file():
        raise RuntimeError("colmap.exe not found after extraction")
    return exe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO_ROOT / "tools" / "colmap"))
    args = ap.parse_args()
    dest = Path(args.dir)

    if sys.platform != "win32":
        print("Linux/macOS: install COLMAP via the system package manager, e.g.\n"
              "  Ubuntu/Debian: sudo apt install colmap\n"
              "  Fedora:        sudo dnf install colmap\n"
              "  macOS (brew):  brew install colmap\n"
              "or build from source: https://colmap.github.io/install.html")
        sys.exit(0)

    print("Querying latest COLMAP release...")
    release = latest_release()
    print(f"Latest release: {release['tag_name']}")
    asset = pick_asset(release)
    if asset is None:
        print("No Windows asset found; download manually from "
              "https://github.com/colmap/colmap/releases")
        sys.exit(1)

    name, url = asset
    print(f"Selected asset: {name}")
    exe = install_windows(url, dest)
    print(f"Installed: {exe}")
    out = subprocess.run([str(exe), "--help"], capture_output=True, text=True)
    print(out.stdout.splitlines()[0] if out.stdout else "(no version output)")
    print("\nSpatialForge auto-detects tools/colmap/bin/colmap.exe; "
          "or set SPATIALFORGE_COLMAP_EXECUTABLE in .env")


if __name__ == "__main__":
    main()
