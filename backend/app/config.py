"""Application settings and hardware capability detection.

SpatialForge is laptop-first: it must run end-to-end on CPU. GPU acceleration
is optional and detected at runtime -- nothing here assumes CUDA is present.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

CAMERA_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class Settings(BaseSettings):
    """Runtime configuration, overridable through environment / .env file."""

    model_config = SettingsConfigDict(
        env_prefix="SPATIALFORGE_", env_file=str(REPO_ROOT / ".env"), extra="ignore"
    )

    data_dir: Path = REPO_ROOT / "data"
    host: str = "127.0.0.1"
    port: int = 8000
    colmap_executable: str = ""
    max_image_size: int = 1600
    max_images_per_project: int = 300
    max_upload_mb: int = 25
    max_project_mb: int = 2000
    exhaustive_match_max_images: int = 120
    use_cache: bool = True
    log_level: str = "INFO"

    # --- derived paths -----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "spatialforge.db"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    def project_dir(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def ensure_project_dirs(self, project_id: str) -> dict[str, Path]:
        root = self.project_dir(project_id)
        dirs = {
            "root": root,
            "originals": root / "originals",
            "images": root / "images",
            "colmap": root / "colmap",
            "artifacts": root / "artifacts",
            "renders": root / "artifacts" / "renders",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs


settings = Settings()


# ---------------------------------------------------------------------------
# COLMAP discovery
# ---------------------------------------------------------------------------

def _is_executable(path: Path) -> bool:
    """On Windows COLMAP ships as colmap.exe / colmap.bat inside its bin dir."""
    return path.is_file() and os.access(path, os.X_OK)


def _windows_candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    for base in (
        REPO_ROOT / "tools" / "colmap",
        Path.home() / ".spatialforge" / "colmap",
        Path("C:/Program Files/COLMAP"),
        Path("C:/Program Files/COLMAP-dev"),
    ):
        for rel in ("bin/colmap.exe", "bin/colmap.bat", "colmap.bat", "colmap.exe"):
            candidates.append(base / rel)
    return candidates


def find_colmap() -> Path | None:
    """Locate a usable COLMAP executable without assuming it is installed."""
    if settings.colmap_executable:
        p = Path(settings.colmap_executable)
        if _is_executable(p):
            return p
    which = shutil.which("colmap")
    if which:
        return Path(which)
    if platform.system() == "Windows":
        for cand in _windows_candidate_paths():
            if _is_executable(cand):
                return cand
    return None


def colmap_version(executable: Path) -> str | None:
    try:
        out = subprocess.run(
            [str(executable), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        m = re.search(r"COLMAP\s+([\w.\-]+)", out.stdout + out.stderr)
        return m.group(1) if m else "unknown"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hardware capability detection (never assumes CUDA)
# ---------------------------------------------------------------------------

def _query_nvidia_smi() -> list[dict]:
    """Return a list of NVIDIA GPUs using nvidia-smi, or [] when unavailable."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        for p in (Path("C:/Windows/System32/nvidia-smi.exe"),):
            if _is_executable(p):
                smi = str(p)
                break
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    gpus.append({
                        "name": parts[0],
                        "vram_mb": int(float(parts[1])),
                        "compute_capability": parts[2],
                        "driver_version": parts[3],
                    })
                except ValueError:
                    continue
        return gpus
    except Exception:
        return []


def _probe_python_package(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "present")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _capabilities_snapshot() -> dict:
    gpus = _query_nvidia_smi()

    torch_version = _probe_python_package("torch")
    cuda_available = False
    if torch_version:
        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            pass

    colmap_path = find_colmap()
    colmap_ver = colmap_version(colmap_path) if colmap_path else None

    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        cpu_count = psutil.cpu_count(logical=True)
        disk_gb = round(psutil.disk_usage(str(settings.data_dir if settings.data_dir.exists() else REPO_ROOT)).free / (1024**3), 1)
    except Exception:
        ram_gb, cpu_count, disk_gb = None, os.cpu_count(), None

    nerfstudio_available = shutil.which("ns-train") is not None

    local_gpu_ready = bool(cuda_available and nerfstudio_available)
    cpu_ready = colmap_path is not None

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
        },
        "hardware": {
            "cpu_model": platform.processor() or None,
            "cpu_cores": cpu_count,
            "ram_gb": ram_gb,
            "gpus": gpus,
            "disk_free_gb": disk_gb,
        },
        "dependencies": {
            "colmap": {"available": colmap_path is not None, "version": colmap_ver,
                       "path": str(colmap_path) if colmap_path else None},
            "open3d": {"available": _probe_python_package("open3d") is not None,
                       "version": _probe_python_package("open3d")},
            "torch": {"available": torch_version is not None, "version": torch_version,
                      "cuda_available": cuda_available},
            "nerfstudio": {"available": nerfstudio_available},
            "lpips": {"available": _probe_python_package("lpips") is not None},
        },
        "modes": {
            "cpu_reconstruction": cpu_ready,
            "local_gpu_training": local_gpu_ready,
            "colab_gpu_training": True,  # provided via the bundled notebook
        },
        "notes": _capability_notes(cpu_ready, local_gpu_ready),
    }


def _capability_notes(cpu_ready: bool, gpu_ready: bool) -> list[str]:
    notes: list[str] = []
    if not cpu_ready:
        notes.append(
            "COLMAP was not found on this machine. Install it (see docs/installation.md "
            "or run: python scripts/install_colmap.py) to enable reconstruction."
        )
    if not gpu_ready:
        notes.append(
            "Local Gaussian-Splatting training requires PyTorch with CUDA and Nerfstudio, "
            "which are not detected. CPU reconstruction still works; use the bundled Google "
            "Colab notebook for neural training."
        )
    return notes


def capabilities(force: bool = False) -> dict:
    if force:
        _capabilities_snapshot.cache_clear()
    return _capabilities_snapshot()
