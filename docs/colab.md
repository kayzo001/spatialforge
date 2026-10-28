# Running on Google Colab

`notebooks/SpatialForge_Colab.ipynb` runs the GPU-heavy half of the pipeline
on a free Colab GPU and hands the artifacts back to your laptop.

## Workflow

1. **Cell 1 — hardware detection.** Verifies a GPU is assigned and its
   compute capability is sufficient (≥ 7.5 for Splatfacto). Fails loudly
   otherwise.
2. **Cell 2 — system deps.** `apt` installs COLMAP, ffmpeg, git-lfs.
3. **Cell 3 — pinned Python env.** torch 2.4.1+cu121 + nerfstudio 1.1.3
   (skips the reinstall if Colab's torch already matches). 10-20 min.
4. **Cell 4 — your data.** Upload a ZIP (path-traversal checked) or mount
   Drive. ≥ 3 images required.
5. **Cell 5 — COLMAP SfM.** Feature extraction + matching (GPU SIFT),
   incremental mapper, undistortion into the Nerfstudio-ready layout.
6. **Cell 6 — Splatfacto training.** Presets: fast_preview (7k iters),
   balanced (15k), high_quality (30k). T4 handles balanced.
7. **Cell 7 — ns-eval.** PSNR / SSIM / LPIPS on held-out cameras.
8. **Cell 8 — export.** `splats.ply` + sparse model + eval.json, zipped.
9. **Cell 9 — download.**

## Importing results into the web app

Copy `splats.ply` into `data/projects/<project_id>/artifacts/` and register
it (kind `splats`) — the viewer's "Gaussian splats" mode then renders the
scene with @mkkellogg/gaussian-splats-3d. Alternatively publish/share scenes
via the Hugging Face scripts (`scripts/hf_upload.py` / `scripts/hf_download.py`).

## Known limitations

- Colab sessions are ephemeral: checkpoints under `/content/outputs` die with
  the session. Copy to Drive early for long runs.
- Colab's preinstalled packages change over time; cell 3 pins a tested
  combination and skips/reinstalls accordingly. If the CUDA stack drifts far
  enough, adjust the pins (they are documented in the cell).
- The notebook is **not** verified against every Colab GPU type; T4 and L4
  are the reference runtimes.
