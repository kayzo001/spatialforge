# Troubleshooting

## COLMAP problems

**"COLMAP is not installed or not found"**
Run `python scripts/install_colmap.py` (Windows) or install via the system
package manager. Verify with `python scripts/detect_env.py`. Custom location?
Set `SPATIALFORGE_COLMAP_EXECUTABLE=/path/to/colmap(.exe)` in `.env`.

**"COLMAP registered no images" / very few registered**
The usual causes, in order of likelihood:
1. **Insufficient overlap** — consecutive photos should share ~60-80% of
   content. Capture in a ring or spiral around the object.
2. **Blurry images** — the validation panel flags these; reshoot with faster
   shutter.
3. **Textureless/reflective subjects** — blank walls, glass, mirrors break
   SIFT matching. Add texture where possible.
4. **Too few images** — use ≥ 20 for robust results (3 is the hard minimum).

**"Only N% of images registered"** — the pipeline continues with a warning;
the missing views likely have weak overlap. Recapture those viewpoints.

**Mapper produces tiny fragmented models** — COLMAP may output several model
directories; SpatialForge automatically uses the largest one.

## Matching / reconstruction quality

- Varying depth of field between shots, zooming between shots, or exposure
  pumping all hurt matching. Lock focus/exposure if your camera allows.
- Feature extraction runs on CPU (`use_gpu 0`) for deterministic behavior on
  machines without CUDA; it is not slower for typical dataset sizes.

## Upload problems

- **"File exceeds N MB"** — per-file limit is configurable via
  `SPATIALFORGE_MAX_UPLOAD_MB`.
- **ZIP rejected** — the extractor refuses archives with path traversal,
  > 500 files, or suspicious compression ratios (zip bombs). Extract
  locally and upload the images directly if your archive trips a guard.
- **"Unreadable image"** — the file is corrupt or an unsupported variant
  (e.g. CMYK JPEGs from some scanners convert fine; progressive WebP with
  animation does not).

## Web app

- **Viewer shows nothing** — check the Artifacts panel; the display modes are
  enabled only when the corresponding artifact actually exists.
- **Live progress stalls** — SSE runs over the same origin; if you proxy the
  backend, make sure buffering is disabled for `text/event-stream`.
- **Status "failed: interrupted by server restart"** — jobs do not survive
  restarts by design; press Re-run.

## GPU training

- `"GPU training unavailable: torch is not installed"` — expected on CPU
  setups; use the Colab notebook or install the GPU stack
  (docs/installation.md §7).
- Nerfstudio OOM — reduce image resolution (num_downscales), switch to the
  fast_preview preset, or close other GPU consumers. On 4 GB cards expect
  only small scenes at reduced resolution.
- After training, `ns-eval`/`ns-export` failures do not void the trained
  model — the checkpoint remains in `outputs/` and can be re-exported
  manually.

## Still stuck?

Open an issue with the output of `python scripts/detect_env.py` and the
last ~50 lines from the project's Processing-logs tab.
