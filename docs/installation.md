# Installation

SpatialForge is laptop-first: the complete geometric reconstruction pipeline
(upload → validation → SfM → point cloud → viewer) runs **entirely on CPU**.
GPU support is optional and detected at runtime — nothing assumes CUDA exists.

## 0. Inspect your machine first

```bash
python scripts/detect_env.py
```

prints OS, CPU, RAM, GPU(s), COLMAP/Open3D/torch availability and which modes
are usable. Run this before installing anything else.

## 1. Requirements

| component | version | notes |
|---|---|---|
| OS | Windows 10/11, Linux, macOS | tested on Windows 10 x64 |
| Python | 3.10 – 3.12 | Open3D wheels currently cover up to 3.12 |
| Node.js | ≥ 18 | only needed to *build* the frontend |
| COLMAP | 3.8+ or 4.x | required for reconstruction (CPU build is fine) |
| CUDA GPU | optional | only for local Gaussian-Splatting training |

## 2. Python backend

```bash
python -m venv .venv
# Windows (Git Bash / PowerShell):
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-geometry.txt -r requirements-dev.txt
# Linux/macOS:
.venv/bin/python -m pip install -r requirements.txt -r requirements-geometry.txt -r requirements-dev.txt
```

- `requirements.txt` — FastAPI stack, OpenCV, Pillow, NumPy (all CPU-friendly)
- `requirements-geometry.txt` — Open3D (optional: a NumPy fallback is used if absent)
- `requirements-dev.txt` — pytest, httpx, ruff

## 3. COLMAP

**Windows (no admin needed):**

```bash
python scripts/install_colmap.py
```

downloads the latest official release into `tools/colmap/`, which SpatialForge
auto-detects (add `SPATIALFORGE_COLMAP_EXECUTABLE` to `.env` to point elsewhere).

**Ubuntu/Debian:** `sudo apt install colmap` · **Fedora:** `sudo dnf install colmap` ·
**macOS:** `brew install colmap`

Verify: `python scripts/detect_env.py` → `CPU reconstruction (COLMAP): YES`.

## 4. Frontend

```bash
cd frontend
npm install
npm run build      # production build; the backend serves frontend/dist automatically
# or for development with hot reload:
npm run dev        # http://localhost:5173, /api proxied to :8000
```

## 5. Run it

```bash
# backend (from the repo root)
.venv/Scripts/python -m uvicorn backend.app.main:app --port 8000
# or equivalently:
.venv/Scripts/python -m backend.app.cli serve
```

- Development: open http://localhost:5173 (Vite dev server)
- Production-style: open http://127.0.0.1:8000 (backend serves the built SPA)
- API docs: http://127.0.0.1:8000/docs (OpenAPI / Swagger)

## 6. First reconstruction (demo)

```bash
python scripts/generate_synthetic_dataset.py     # license-clean sample data
python scripts/prepare_demo.py                   # dataset -> reconstruction -> demo project
```

then open the Dashboard — the demo project shows real camera poses, a sparse
point cloud, and reprojection statistics computed from actual COLMAP output.

To use your own photos: Dashboard → **New Reconstruction** → drop images or a
ZIP → choose preset/mode → Start.

## 7. Optional: local GPU training (Nerfstudio + CUDA)

Only needed for Gaussian-Splatting training on your own GPU. The environment
must match your CUDA driver; there is deliberately no single `pip install`
line because the correct PyTorch build depends on the driver.

```bash
# 1. PyTorch with CUDA matching your driver (example for CUDA 12.1):
.venv/Scripts/python -m pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
# 2. Nerfstudio (see https://docs.nerf.studio for current system deps)
.venv/Scripts/python -m pip install nerfstudio==1.1.3
# 3. verify
python scripts/detect_env.py   # -> Local GPU training: YES
```

If this is not set up, the app runs in CPU mode and directs you to the bundled
Google Colab notebook (`notebooks/SpatialForge_Colab.ipynb`) for training.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md).
