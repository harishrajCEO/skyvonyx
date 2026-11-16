# Skyvonyx — rules for AI coding assistants

Project: AI-assisted, human-verified labels for satellite imagery. Pre-revenue, free tools, Colab GPU.
Context document: `Downloads/skyvonyx_master_context.md` (not in this repo). Read it before changing strategy-adjacent code.

## Priorities (in order)
1. First paid pilot. 2. Measured accuracy on unseen data. 3. Correct georeferenced output. 4. Everything else.

## Hard rules
- Never claim accuracy that was not measured. Reports and summaries say "NOT MEASURED" until a gold set is checked.
- Every geospatial output keeps CRS + affine transform. GeoJSON (RFC 7946) is WGS84; also write native CRS.
- Never load a whole GeoTIFF into memory: windowed reads via `raster.RasterSource`.
- Train and infer at the same tile size. Split train/test by scene, never by tile.
- Human approval (`APPROVED` flag) before anything reaches a customer. Code never emails, shares, or serves unreviewed output (the API enforces this).
- One expert resident on the GPU at a time (`max_resident_experts=1`). A new model goes through `ResidencyManager`, with a real `unload()`.
- Every stage reads from and writes to `RunMemory`; a new stage gets a fingerprint entry in `JobConfig.stage_fingerprint`.
- No secrets in code or notebooks. No customer imagery in git (`.gitignore` covers `*.tif`, `jobs/`, `memory/`).
- Every training run declares `dataset_ids`; never mark a model commercial-ok by hand. `licence_status()` fails closed, and only a person who read the licence may flip `verified` in `DATASET_LICENCES` (with a row in `datasets/LICENCES.md`).
- Training and inference use the same `tile` and `sensor`. Chips smaller than a tile are padded, never upscaled.
- Civil geospatial monitoring only (see `docs/ACCEPTABLE_USE.md`). No face recognition, no tracking of individuals, no CCTV ingestion.
- Licences: ultralytics AGPL-3.0; DOTA academic; RF-DETR Plus sizes PML 1.0 (base Apache-2.0); SAM Apache-2.0; SAHI MIT. Flag any new dependency's licence in the PR.

## Layout
See README. Stage order lives in `config.STAGE_ORDER`; the router is `pipeline.run_scene`; experts register in `experts/__init__.py`.

## Before you say "done"
Run `python tests/test_pipeline.py`. For anything touching a real expert or rasterio, run notebook cell 6 on Colab and state what was not executed.
