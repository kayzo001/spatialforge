# Changelog

## Unreleased

### Fixed
- **Duplicate images on re-run.** Preprocessing left the previous run's
  derivatives in `images/`, so every re-run added `*_1.jpg` copies and COLMAP
  registered each photo twice (for example "72/36", 200%). `images/` is now
  regenerated from the originals on every run.
- **Stale COLMAP cache.** With caching on, the database and sparse model were
  reused even when the input images changed. The cache is now keyed to a
  fingerprint of the processed image set (`colmap/inputs.json`).
- **Camera frustums drawn as long spikes.** The viewer loaded the row-major
  `c2w` matrices from `cameras.json` as column-major, which transposed every
  pose and placed all camera markers at the origin. Camera centers now match
  `cameras.json` exactly.
- **COLMAP 3.x detection.** Version detection always returned the 4.x option
  profile, so feature extraction failed on COLMAP 3.x (`unrecognised option
  '--FeatureExtraction.use_gpu'`).
- **"COLMAP was not detected" on New Reconstruction.** The page never checked
  capabilities on load, and the two "not available" labels were swapped.
- **Analytics "undefined (200%)".** The registered-images card read a field
  that does not exist.
- **Stale page and artifacts after updates or re-runs.** `index.html` and
  artifact files are now served with `Cache-Control: no-cache`.
- **Log console mixed runs.** After a re-run the console appended to, and past
  500 lines only showed, older runs. It now shows the current job only
  (`GET /api/projects/{id}/logs?job_id=…`).
- **Viewer collapsed on narrow windows** (zero-height canvas, 0 fps).

### Changed
- Dashboard cards show the first photo as a thumbnail; long names wrap.
- Pipeline stages that did not run are shown as *skipped*, not done.
- Smaller, scene-scaled camera frustums; larger default point size; color
  modes labelled Color / Cyan / Depth.
- Responsive workspace layout and a sidebar that collapses to icons on small
  screens.
- Dependencies: three 0.186.1 (+ @types/three), autoprefixer 10.6.1,
  postcss 8.5.31 (the Dependabot "npm-minor" group).
- New screenshots in `assets/` (the previous frustum screenshot showed the
  transposed-pose bug).
