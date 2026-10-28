"""Nerfstudio Splatfacto training orchestration (GPU path).

This module builds and runs `ns-train splatfacto`, `ns-eval` and
`ns-export gaussian-splat` against the COLMAP undistorted dataset layout that
Stage 3 produces (`<project>/colmap/undistorted/{images,sparse}`).

The local-GPU path is only executable when `ns-train` and a CUDA device are
present; every other environment receives a clear "not available" signal
instead of a fabricated result. Command construction is unit-tested without
Nerfstudio installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from backend.app.reconstruction.colmap.runner import CancelledError, _kill_process_tree
from backend.app.reconstruction.gaussian.presets import TrainingPreset
from backend.app.utils.logs import get_logger

logger = get_logger("spatialforge.gaussian")

LogCallback = Callable[[str], None]


def nerfstudio_available() -> bool:
    return shutil.which("ns-train") is not None


def check_cuda_torch() -> tuple[bool, str | None]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            return True, f"{name} ({mem} MB)"
        return False, "torch is installed but no CUDA device is visible"
    except ImportError:
        return False, "torch is not installed in this environment"
    except Exception as exc:  # pragma: no cover
        return False, f"CUDA probe failed: {exc}"


def splatfacto_train_cmd(
    data_dir: str,
    output_dir: str,
    preset: TrainingPreset,
    timestamp_name: str = "spatialforge",
    machine_seed: int = 42,
) -> list[str]:
    """Build the ns-train splatfacto command for the COLMAP-layout dataset."""
    return [
        "ns-train", "splatfacto",
        "--data", data_dir,
        "--output-dir", output_dir,
        "--timestamp-name", timestamp_name,
        "--machine.seed", str(machine_seed),
        "--max-num-iterations", str(preset.max_num_iterations),
        "--num-downscales", str(preset.num_downscales),
        "--steps-per-eval-batch", str(preset.steps_per_eval_batch),
        "--steps-per-eval-image", str(preset.steps_per_eval_image),
        "--steps-per-save", str(preset.steps_per_save),
        "--pipeline.model.cull-alpha-thresh", str(preset.cull_threshold),
        "--pipeline.model.densify-grad-thresh", str(preset.densify_grad_thresh),
        "--viewer.quit-on-train-completion", "True",
    ]


def eval_cmd(load_config: str, output_path: str) -> list[str]:
    return ["ns-eval", "--load-config", load_config, "--output-path", output_path]


def export_splat_cmd(load_config: str, output_dir: str) -> list[str]:
    return ["ns-export", "gaussian-splat", "--load-config", load_config,
            "--output-dir", output_dir]


# Regex over nerfstudio's stdout progress lines, e.g.:
# "Training Progress:  43%|####2 | 6450/15000 [05:12<06:53, 20.7it/s, loss=0.021]"
TRAIN_LINE = re.compile(
    r"(?P<step>\d+)/(?P<total>\d+).*?(?:loss[=:]\s*(?P<loss>[\d.eE+-]+))?"
)


def parse_progress_line(line: str) -> dict | None:
    if "Training Progress" not in line and "it/s" not in line:
        return None
    m = TRAIN_LINE.search(line)
    if not m:
        return None
    out: dict = {"step": int(m.group("step")), "total": int(m.group("total"))}
    if m.group("loss"):
        try:
            out["loss"] = float(m.group("loss"))
        except ValueError:
            pass
    return out


def _run(cmd: list[str], log: LogCallback, cancel_event: threading.Event,
         timeout_s: float | None = None) -> str:
    exe = shutil.which(cmd[0])
    if exe is None:
        raise RuntimeError(f"{cmd[0]} is not installed or not on PATH")
    full = [exe] + cmd[1:]
    log(f"$ {' '.join(full)}")
    proc = subprocess.Popen(  # noqa: S603
        full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    output_lines: list[str] = []
    assert proc.stdout is not None

    def reader() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            output_lines.append(line.rstrip())
            log(line.rstrip())

    threading.Thread(target=reader, daemon=True).start()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(proc.pid)
        raise RuntimeError(f"{cmd[0]} timed out") from exc
    if cancel_event.is_set():
        _kill_process_tree(proc.pid)
        raise CancelledError("Canceled by user")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed (exit {proc.returncode}): {'|'.join(output_lines[-12:])}"
        )
    return "\n".join(output_lines)


def train_splatfacto(
    data_dir: Path,
    output_dir: Path,
    preset: TrainingPreset,
    log: LogCallback,
    cancel_event: threading.Event,
    vram_mb: int | None = None,
) -> dict:
    """Run full GPU training: train -> eval -> export. Returns paths + metrics.

    Raises RuntimeError with an actionable message when Nerfstudio/CUDA are
    missing. Never fabricates metrics; eval numbers come from ns-eval output.
    """
    ok, why = check_cuda_torch()
    if not ok:
        raise RuntimeError(f"GPU training unavailable: {why}. Use the Colab notebook.")
    if not nerfstudio_available():
        raise RuntimeError(
            "GPU training unavailable: Nerfstudio (ns-train) not found. "
            "Install it or use the Colab notebook."
        )

    from backend.app.reconstruction.gaussian.presets import downscale_preset_for_vram

    preset = downscale_preset_for_vram(preset, vram_mb)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(splatfacto_train_cmd(str(data_dir), str(output_dir), preset), log, cancel_event)

    config_yml = _find_latest_config(output_dir)
    if config_yml is None:
        raise RuntimeError("Training finished but no config.yml was produced")

    result: dict = {
        "preset": preset.key,
        "iterations": preset.max_num_iterations,
        "config": config_yml.name,
        "status": "trained",
    }

    eval_json = output_dir / "eval.json"
    try:
        _run(eval_cmd(str(config_yml), str(eval_json)), log, cancel_event)
        if eval_json.is_file():
            result["evaluation"] = json.loads(eval_json.read_text(encoding="utf-8"))
    except RuntimeError as exc:
        log(f"[warn] ns-eval failed: {exc}")

    export_dir = output_dir / "export"
    try:
        _run(export_splat_cmd(str(config_yml), str(export_dir)), log, cancel_event)
        ply = _find_splat_ply(export_dir)
        if ply:
            result["splat_ply"] = str(ply)
    except RuntimeError as exc:
        log(f"[warn] ns-export failed: {exc}")

    return result


def _find_latest_config(output_dir: Path) -> Path | None:
    configs = sorted(output_dir.glob("**/config.yml"), key=lambda p: p.stat().st_mtime)
    return configs[-1] if configs else None


def _find_splat_ply(export_dir: Path) -> Path | None:
    cands = sorted(export_dir.glob("**/*.ply"), key=lambda p: p.stat().st_size, reverse=True)
    return cands[0] if cands else None
