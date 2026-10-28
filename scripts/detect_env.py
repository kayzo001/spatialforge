#!/usr/bin/env python3
"""Print a full environment report (OS, CPU, RAM, GPU, dependencies).

Run before installing anything:
    python scripts/detect_env.py

Never assumes CUDA is present; reports exactly what was detected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from backend.app.config import capabilities

    report = capabilities(force=True)
    print(json.dumps(report, indent=2))
    print("\nSummary:")
    modes = report["modes"]
    print(f"  CPU reconstruction (COLMAP): {'YES' if modes['cpu_reconstruction'] else 'NO - install COLMAP'}")
    print(f"  Local GPU training (Nerfstudio+CUDA): {'YES' if modes['local_gpu_training'] else 'no (use Colab)'}")
    print("  Colab GPU training: available via notebooks/SpatialForge_Colab.ipynb")
    for note in report["notes"]:
        print(f"  * {note}")


if __name__ == "__main__":
    main()
