# Skyvonyx backend

Staged satellite-image labelling: **SAHI slicing → YOLOv8 → SAHI re-stitch → SAM 2 masks → georeferenced export**, run so that only one model is ever resident on the GPU. Built to fit a free Colab T4 and to survive its disconnects.

```
 stage     device   what                                      resident in VRAM
 ───────   ──────   ───────────────────────────────────────   ──────────────────
 plan      CPU      SAHI slice geometry + contrast limits     nothing
 detect    GPU      YOLOv8 over every non-empty tile          detector
 stitch    CPU      SAHI GreedyNMM (IoS) merge across tiles   nothing (detector evicted)
 segment   GPU      SAM 2 box-prompted masks, per crop        segmenter
 export    CPU      pixel→CRS, GeoJSON (WGS84 + native),      nothing
                    COCO, preview, draft report
```

Each stage runs over the whole scene, writes its result to `memory/`, and the next stage reads it back. A detector and a segmenter never share the GPU, and a job that dies during SAM 2 restarts at SAM 2.

## Layout

```
skyvonyx/            the package
  config.py          JobConfig + per-stage fingerprints (what invalidates what)
  memory.py          disk-backed stage hand-off, resume, run ledger
  residency.py       expert residency manager: max_resident=1, load/evict, VRAM accounting
  raster.py          windowed GeoTIFF reads, stretch, empty-tile test (never loads a whole scene)
  slicing.py         stage 1: SAHI get_slice_bboxes (builtin fallback)
  stitching.py       stage 3: SAHI GreedyNMM/NMS re-stitch (builtin fallback)
  geo.py             stage 5: affine → CRS, mask → polygon, GeoJSON/COCO/preview
  pipeline.py        the router: runs the five stages, builds summary.json
  report.py          REPORT_DRAFT.md (accuracy: NOT MEASURED, licences, per-stage timings)
  queue_worker.py    Drive-folder queue, human review gate, requeue after disconnect
  experts/           yolov8, rfdetr, sam2, sam1, fake - one interface, lazy imports
  datasets.py        labelled scenes (DOTA/COCO) -> YOLO tiles, same slicer+stretch as inference, split by scene
  evaluate.py        scene-level precision/recall/F1/AP50 vs a gold COCO set; reviewer agreement
api/main.py          FastAPI intake + status + approved-only delivery (no models here)
notebooks/skyvonyx_colab_worker.ipynb   the GPU worker, cells 1-8
notebooks/skyvonyx_train_detector.ipynb train your own detector, measure it twice, publish to models/best.pt
scripts/clone_repos.sh, bootstrap_colab.sh
tests/test_pipeline.py                  56 offline checks, fake experts, ~15 s
tests/test_training_eval.py             13 checks: DOTA parsing, scene splits, tiling, scoring
vendor/              shallow clones (gitignored): sahi, rf-detr, sam2, segment-anything, gods-eye-view, ECC
```

## Run it

**Tests (no GPU, no weights):**
```
python tests/test_pipeline.py && python tests/test_training_eval.py
```

**Measure accuracy (the only way to replace "NOT MEASURED"):**
```
python -m skyvonyx.evaluate --pred jobs/review/<job>/output --gold gold_coco.json     --conf 0.25 --out accuracy.json --note "12 Sentinel-1 IW scenes, Mumbai port, Jun-Sep 2026, labelled by A, checked by B"
```
Writes `accuracy.json` and `accuracy.md`. Score reviewer B's labels against reviewer A's with the same command to get agreement.

**Colab worker:** open `notebooks/skyvonyx_colab_worker.ipynb`, set `SKYVONYX_REPO` in cell 2 to wherever you pushed this repo, run cells 1-8. Cell 6 runs real YOLOv8 + SAM 2 on a synthetic UTM scene and prints the scheduler report - you should see the detector load, get evicted, then SAM 2 load, with VRAM numbers for each.

**Submit a job:** drop a folder into `MyDrive/skyvonyx/jobs/inbox/<job_id>/` with `input/*.tif`, a `job.json`, and an empty `READY` file created last. Or `POST /jobs` to the API. Or `queue_worker.create_job(...)`.

```json
{"job_id": "galaxeye_pilot_01", "customer": "galaxeye", "sensor": "optical",
 "detector": "yolov8", "detector_weights": "/content/drive/MyDrive/skyvonyx/models/best.pt",
 "segmenter": "sam2",  "segmenter_weights": "/content/drive/MyDrive/skyvonyx/models/sam2.1_hiera_small.pt",
 "tile": 1024, "overlap": 0.2, "conf": 0.25, "merge_metric": "IOS", "merge_threshold": 0.5,
 "segment_min_score": 0.35, "max_resident_experts": 1}
```

**API (CPU host, e.g. Hugging Face Spaces):**
```
pip install -r requirements-api.txt
SKYVONYX_ROOT=/data/skyvonyx SKYVONYX_REVIEWER_TOKEN=change-me uvicorn api.main:app --port 7860
```
`POST /jobs` (multipart) · `GET /jobs/{id}` · `GET /jobs/{id}/summary` · `GET /jobs/{id}/geojson` · `GET /jobs/{id}/results/{file}` · `POST /jobs/{id}/review`.
Results and GeoJSON are served **only for jobs in `done/`** (a human created `APPROVED`); review-stage access needs `X-Skyvonyx-Token`. The API and the worker share nothing but the queue directory.

## Memory, in two senses

1. **VRAM.** `ResidencyManager(max_resident=1)` - activating the segmenter evicts the detector (`unload()` + `gc.collect()` + `empty_cache()`), and every activation records VRAM before/after/peak into `summary.json["scheduler"]`. Set `max_resident_experts: 2` in a job to keep both loaded on a bigger GPU.
2. **Disk.** `RunMemory` keys every stage payload by scene content + a stage fingerprint. Change `conf` and detect/stitch/segment/export recompute but the slice plan is reused; change `merge_threshold` and only stitch onward recomputes. The ledger (`memory/<job>/ledger.jsonl`) is the per-stage timing record the master context asks for.

## What the outputs mean

- `*_wgs84.geojson` - RFC 7946 lon/lat, for web maps and QGIS. `*_native.geojson` - the scene's CRS, for measurement.
- `*_PIXEL_COORDS_NOT_GEOREFERENCED.geojson` - the input had no transform (PNG/JPG). Not a geospatial deliverable; the report says so.
- Every feature carries `score`, `geometry_source` (`sam_mask` or `detector_box`), `px_bbox`, and `status: unreviewed`.
- `REPORT_DRAFT.md` says **accuracy NOT MEASURED**. Only a reviewer with a gold subset may change that.

## Licences (not legal advice)

| component | licence | consequence |
|---|---|---|
| ultralytics / YOLOv8 | AGPL-3.0 | fine for pilots and delivering labels; a hosted commercial service needs an enterprise licence or `"detector": "rfdetr"` |
| weights fine-tuned on DOTA | DOTA academic terms | treat as a commercial risk; sell on Sentinel-derived or customer-owned data |
| SAM 2 | Apache-2.0 | verify the checkpoint you ship |
| SAHI | MIT | — |
| RF-DETR | base sizes Apache-2.0, Plus sizes PML 1.0 | check `vendor/rf-detr/LICENSE` per size |
| gods-eye-view | MIT (fork of bilawalsidhu/gods-eye-view) | keep the attribution; its CCTV/ALPR layers are outside Skyvonyx's stated scope |

## Not tested here

Everything in `tests/` runs with fake experts on CPU. The real-model code paths (`experts/detector_yolo.py`, `experts/segmenter_sam2.py`, `experts/segmenter_sam1.py`, `experts/detector_rfdetr.py`) and rasterio GeoTIFF I/O were written against the vendored source but **not executed** on this machine - no GPU, no rasterio. Notebook cell 6 is the test for them. Run it before trusting anything.
