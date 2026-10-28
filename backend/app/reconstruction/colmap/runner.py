"""COLMAP subprocess execution with structured logging and cancellation."""

from __future__ import annotations

import os
import platform
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from backend.app.utils.logs import get_logger

logger = get_logger("spatialforge.colmap")

LogCallback = Callable[[str], None]


class ColmapError(RuntimeError):
    """COLMAP exited with a non-zero code."""

    def __init__(self, message: str, returncode: int, tail: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.tail = tail


class CancelledError(RuntimeError):
    """The user requested cancellation of the running process."""


def _creation_flags() -> int:
    if platform.system() == "Windows":
        return subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return 0


def _kill_process_tree(pid: int) -> None:
    try:
        import psutil

        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except Exception:
        try:
            proc = subprocess.Popen
            del proc
            os.kill(pid, 9)
        except Exception:
            pass


def run_colmap(
    args: list[str],
    executable: Path,
    log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
    timeout_s: float | None = None,
) -> str:
    """Run a COLMAP command, streaming its output line by line.

    Raises ColmapError on non-zero exit and CancelledError when
    `cancel_event` is set (the whole process tree is terminated).
    """
    cmd = [str(executable)] + args
    env = os.environ.copy()
    exe_dir = str(Path(executable).resolve().parent)
    env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")

    emit = log or (lambda _msg: None)
    emit(f"$ {' '.join(cmd)}")
    logger.info("Running COLMAP: %s", " ".join(args[:2]))

    proc = subprocess.Popen(  # noqa: S603 - args are built internally, not from raw user input
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=exe_dir,
        creationflags=_creation_flags(),
    )

    output_lines: list[str] = []
    assert proc.stdout is not None

    def _reader() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            output_lines.append(line)
            emit(line)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(proc.pid)
        raise ColmapError(f"COLMAP timed out after {timeout_s:.0f}s", -1) from exc

    while reader.is_alive():
        if cancel_event is not None and cancel_event.is_set():
            _kill_process_tree(proc.pid)
            raise CancelledError("Canceled by user")
        reader.join(timeout=0.2)

    if cancel_event is not None and cancel_event.is_set():
        _kill_process_tree(proc.pid)
        raise CancelledError("Canceled by user")

    tail = "\n".join(output_lines[-30:])
    if proc.returncode != 0:
        raise ColmapError(
            f"COLMAP {' '.join(args[:1])} failed with exit code {proc.returncode}",
            proc.returncode,
            tail,
        )
    return tail
