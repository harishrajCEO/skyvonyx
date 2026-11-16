"""The staged router: SAHI -> YOLOv8 -> re-stitch -> SAM 2 -> export.

This is the "sparse MoE" execution model you asked for. Each stage owns the
machine for its whole pass over the scene, and the experts it does not need are
not resident:

  plan     CPU    slice geometry + contrast limits      no model in VRAM
  detect   GPU    YOLOv8 over every tile, batched       detector resident, SAM 2 not loaded
  stitch   CPU    SAHI GREEDYNMM merge across tiles     detector EVICTED, nothing resident
  segment  GPU    SAM 2 box-prompted masks per crop     segmenter resident, detector gone
  export   CPU    pixel -> CRS, GeoJSON/COCO/preview    nothing resident

Why not interleave detect and segment per tile, which would avoid re-reading
the raster? Because that forces both models into VRAM simultaneously, and
because SAM 2 must see the *merged* box - a ship cut across two tiles would
otherwise get two half-masks that no amount of postprocessing can fuse into
one object. Staging is what makes cross-tile merging meaningful.

Every stage result goes through RunMemory, so a Colab disconnect mid-SAM2 costs
the segment stage only; detect is read back from disk on the next run.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import geo, slicing, stitching
from .config import STAGE_ORDER, JobConfig
from .experts import build_detector, build_segmenter, licence_notes
from .memory import RunMemory, content_key
from .raster import RasterSource, is_empty_tile, stretch_limits, to_uint8_rgb
from .residency import ResidencyManager, pick_device, vram_mb

SEG_CHUNK = 16          # boxes per SAM 2 predict() call - caps the mask tensor size
SEG_MARGIN = 48         # px of context around a box so SAM 2 sees the object's edges


# --------------------------------------------------------------------- stage 1
def stage_plan(scene: Path, cfg: JobConfig, mem: RunMemory) -> dict:
    fp = cfg.stage_fingerprint("plan")
    cached = mem.get_json("plan", fp)
    if cached and cfg.resume:
        mem.record("plan", "cached", 0.0, tiles=cached["summary"]["tiles"])
        return cached

    with mem.timed("plan") as detail:
        with RasterSource(scene, cfg.bands) as src:
            wins = slicing.slice_plan(src.width, src.height, cfg.tile, cfg.overlap)
            lo, hi = stretch_limits(src.overview_sample(), cfg.sensor, src.nodata)
            plan = {
                "windows": [list(map(int, w)) for w in wins],
                "stretch_lo": lo.tolist(), "stretch_hi": hi.tolist(),
                "width": src.width, "height": src.height,
                "bands_used": src.bands,
                "georeferenced": src.georeferenced,
                "crs": src.crs_string,
                "affine": list(src.affine) if src.affine else None,
                "nodata": None if src.nodata is None else float(src.nodata),
                "summary": slicing.plan_summary(wins, src.width, src.height, cfg.tile, cfg.overlap),
            }
        detail.update(tiles=len(wins), megapixels=plan["summary"]["megapixels"],
                      georeferenced=plan["georeferenced"])
    mem.put_json("plan", fp, plan)
    return plan


# --------------------------------------------------------------------- stage 2
def stage_detect(scene: Path, cfg: JobConfig, plan: dict, mem: RunMemory,
                 router: ResidencyManager) -> dict:
    """Run the detector over every non-empty tile. Boxes stay tile-local here;
    stage 3 does the shifting, so a re-stitch threshold change does not force
    re-detection."""
    fp = cfg.stage_fingerprint("detect")
    cached = mem.get_arrays("detect", fp)
    cached_meta = mem.get_json("detect_meta", fp)
    if cached is not None and cached_meta is not None and cfg.resume:
        # JSON turns the class-id keys into strings on the way back in
        cached_meta["names"] = {int(k): v for k, v in cached_meta.get("names", {}).items()}
        mem.record("detect", "cached", 0.0, raw_detections=int(len(cached["boxes"])))
        return {**cached, **cached_meta}

    lo = np.array(plan["stretch_lo"], np.float32)
    hi = np.array(plan["stretch_hi"], np.float32)
    windows = [tuple(w) for w in plan["windows"]]

    with mem.timed("detect") as detail:
        boxes_l, scores_l, classes_l, offsets_l = [], [], [], []
        skipped = 0
        t_read = t_infer = 0.0
        with router.active("detector") as detector:
            names = dict(getattr(detector, "names", {}) or {})
            with RasterSource(scene, cfg.bands) as src:
                for start in range(0, len(windows), cfg.batch):
                    batch = windows[start:start + cfg.batch]
                    t0 = time.time()
                    tiles, offs = [], []
                    for (col, row, w, h) in batch:
                        rgb = to_uint8_rgb(src.read(col, row, w, h), lo, hi, cfg.sensor)
                        if is_empty_tile(rgb, cfg.min_tile_std):
                            skipped += 1
                            continue
                        if rgb.shape[0] < cfg.tile or rgb.shape[1] < cfg.tile:
                            # A tile smaller than `tile` (a scene or chip smaller than
                            # one tile) would be upscaled by the detector's letterbox,
                            # showing objects bigger than in training. Pad bottom/right
                            # instead - the same as datasets.TileSpec(pad_to_tile=True).
                            # Boxes stay in tile-local pixels; restitch clips the pad.
                            canvas = np.zeros((max(rgb.shape[0], cfg.tile),
                                               max(rgb.shape[1], cfg.tile), 3), np.uint8)
                            canvas[:rgb.shape[0], :rgb.shape[1]] = rgb
                            rgb = canvas
                        tiles.append(rgb)
                        offs.append((col, row))
                    t_read += time.time() - t0
                    if not tiles:
                        continue
                    t0 = time.time()
                    for (b, s, k), (col, row) in zip(detector(tiles), offs):
                        if len(b):
                            boxes_l.append(np.asarray(b, np.float32).reshape(-1, 4))
                            scores_l.append(np.asarray(s, np.float32).ravel())
                            classes_l.append(np.asarray(k, int).ravel())
                            offsets_l.append(np.tile([col, row], (len(b), 1)).astype(np.float32))
                    t_infer += time.time() - t0

        arrays = {
            "boxes": np.concatenate(boxes_l) if boxes_l else np.zeros((0, 4), np.float32),
            "scores": np.concatenate(scores_l) if scores_l else np.zeros(0, np.float32),
            "classes": np.concatenate(classes_l) if classes_l else np.zeros(0, int),
            "offsets": np.concatenate(offsets_l) if offsets_l else np.zeros((0, 2), np.float32),
        }
        ev = router.last_event("detector")
        meta = {"names": {int(k): str(v) for k, v in names.items()},
                "tiles_total": len(windows), "tiles_skipped_empty": skipped,
                "tile_read_s": round(t_read, 2), "inference_s": round(t_infer, 2),
                "expert_load_s": ev.get("load_s"), "vram_peak_mb": ev.get("vram_peak_mb")}
        detail.update(raw_detections=int(len(arrays["boxes"])),
                      tiles=len(windows) - skipped, skipped=skipped,
                      read_s=round(t_read, 2), infer_s=round(t_infer, 2),
                      vram_peak_mb=ev.get("vram_peak_mb"))

    mem.put_arrays("detect", fp, arrays)
    mem.put_json("detect_meta", fp, meta)
    return {**arrays, **meta}


# --------------------------------------------------------------------- stage 3
def stage_stitch(cfg: JobConfig, plan: dict, det: dict, mem: RunMemory) -> dict:
    fp = cfg.stage_fingerprint("stitch")
    cached = mem.get_arrays("stitch", fp)
    cached_meta = mem.get_json("stitch_meta", fp)
    if cached is not None and cached_meta is not None and cfg.resume:
        mem.record("stitch", "cached", 0.0, merged=int(len(cached["boxes"])))
        return {**cached, **cached_meta}

    with mem.timed("stitch") as detail:
        merged = stitching.restitch(
            det["boxes"], det["scores"], det["classes"], det["offsets"],
            det.get("names", {}), (plan["height"], plan["width"]),
            threshold=cfg.merge_threshold, metric=cfg.merge_metric,
            algorithm=cfg.merge_algorithm,
        )
        detail.update(before=merged["before"], after=merged["after"], backend=merged["backend"])

    arrays = {k: merged[k] for k in ("boxes", "scores", "classes")}
    meta = {"before_merge": merged["before"], "after_merge": merged["after"],
            "backend": merged["backend"], "duplicates_removed": merged["before"] - merged["after"]}
    mem.put_arrays("stitch", fp, arrays)
    mem.put_json("stitch_meta", fp, meta)
    return {**arrays, **meta}


# --------------------------------------------------------------------- stage 4
def segmentation_crops(boxes: np.ndarray, width: int, height: int, tile: int,
                       margin: int = SEG_MARGIN) -> list[tuple[int, int, int, int, list[int]]]:
    """Group merged boxes into crops for SAM 2: (col, row, w, h, box_indices).

    Boxes are bucketed by which `tile`-sized grid cell their centre falls in, so
    nearby objects share one image embedding. The crop is then the union of the
    bucket's boxes plus a margin - not the whole cell - because a small crop is
    upsampled to SAM 2's 1024px input, which gives a 20px ship far more
    effective resolution than a full tile would.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, b in enumerate(boxes):
        key = (int((b[0] + b[2]) / 2) // tile, int((b[1] + b[3]) / 2) // tile)
        buckets.setdefault(key, []).append(i)

    crops = []
    for _, idxs in sorted(buckets.items()):
        bb = boxes[idxs]
        x1 = max(0, int(np.floor(bb[:, 0].min())) - margin)
        y1 = max(0, int(np.floor(bb[:, 1].min())) - margin)
        x2 = min(width, int(np.ceil(bb[:, 2].max())) + margin)
        y2 = min(height, int(np.ceil(bb[:, 3].max())) + margin)
        if x2 > x1 and y2 > y1:
            crops.append((x1, y1, x2 - x1, y2 - y1, idxs))
    return crops


def stage_segment(scene: Path, cfg: JobConfig, plan: dict, stitched: dict,
                  mem: RunMemory, router: ResidencyManager) -> dict:
    fp = cfg.stage_fingerprint("segment")
    cached = mem.get_json("segment", fp)
    if cached is not None and cfg.resume:
        mem.record("segment", "cached", 0.0, masked=len(cached.get("polygons", {})))
        return {"polygons": {int(k): v for k, v in cached.get("polygons", {}).items()},
                **{k: v for k, v in cached.items() if k != "polygons"}}

    boxes = stitched["boxes"]
    scores = stitched["scores"]
    if cfg.segmenter in {"none", ""} or len(boxes) == 0:
        reason = "segmenter disabled" if cfg.segmenter in {"none", ""} else "no detections"
        mem.record("segment", "ok", 0.0, masked=0, note=reason)
        out = {"polygons": {}, "requested": 0, "masked": 0, "skipped_low_score": 0,
               "note": reason}
        mem.put_json("segment", fp, out)
        return out

    # Only mask boxes worth masking: below segment_min_score it is probably a
    # false positive, and SAM 2 time is the most expensive thing in the run.
    eligible = [i for i in range(len(boxes)) if float(scores[i]) >= cfg.segment_min_score]
    skipped_low = len(boxes) - len(eligible)
    truncated = 0
    if len(eligible) > cfg.segment_max_boxes:
        eligible = sorted(eligible, key=lambda i: -float(scores[i]))[:cfg.segment_max_boxes]
        truncated = 1

    if not eligible:
        # Every box is below segment_min_score. Loading SAM 2 here would cost
        # load time and VRAM to segment nothing, and would make the run look as
        # if SAM 2 had been exercised when it had not.
        note = f"all {len(boxes)} boxes below segment_min_score={cfg.segment_min_score}"
        mem.record("segment", "ok", 0.0, masked=0, note=note)
        out = {"polygons": {}, "requested": 0, "masked": 0,
               "skipped_low_score": int(skipped_low), "note": note}
        mem.put_json("segment", fp, out)
        return out

    lo = np.array(plan["stretch_lo"], np.float32)
    hi = np.array(plan["stretch_hi"], np.float32)
    keep = np.array(sorted(eligible), int)
    crops = segmentation_crops(boxes[keep], plan["width"], plan["height"], cfg.tile)

    with mem.timed("segment") as detail:
        polygons: dict[int, list] = {}
        failed = 0
        with router.active("segmenter") as segmenter:
            with RasterSource(scene, cfg.bands) as src:
                for (col, row, w, h, local_idx) in crops:
                    rgb = to_uint8_rgb(src.read(col, row, w, h), lo, hi, cfg.sensor)
                    global_idx = [int(keep[i]) for i in local_idx]
                    local_boxes = boxes[global_idx] - np.array([col, row, col, row], np.float32)
                    for start in range(0, len(global_idx), SEG_CHUNK):
                        chunk_g = global_idx[start:start + SEG_CHUNK]
                        chunk_b = local_boxes[start:start + SEG_CHUNK]
                        try:
                            masks = segmenter.masks_for_boxes(rgb, chunk_b)
                        except Exception as e:      # one bad crop must not lose the scene
                            failed += len(chunk_g)
                            mem.record("segment_crop", "failed", 0.0,
                                       error=f"{type(e).__name__}: {e}"[:200],
                                       crop=[col, row, w, h])
                            continue
                        for gi, mask in zip(chunk_g, masks):
                            rings = geo.mask_to_polygons(mask, offset=(col, row),
                                                         simplify_px=cfg.polygon_simplify_px)
                            if rings:
                                polygons[gi] = rings
        ev = router.last_event("segmenter")
        detail.update(requested=len(keep), masked=len(polygons), crops=len(crops),
                      failed=failed, vram_peak_mb=ev.get("vram_peak_mb"))

    out = {"polygons": polygons, "requested": int(len(keep)), "masked": len(polygons),
           "crops": len(crops), "failed": failed, "skipped_low_score": int(skipped_low),
           "truncated_to_max_boxes": bool(truncated),
           "vram_peak_mb": router.last_event("segmenter").get("vram_peak_mb")}
    mem.put_json("segment", fp, out)
    return out


# --------------------------------------------------------------------- stage 5
def stage_export(scene: Path, cfg: JobConfig, plan: dict, det: dict, stitched: dict,
                 seg: dict, out_dir: Path, mem: RunMemory) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(scene).stem
    names = {int(k): v for k, v in det.get("names", {}).items()}

    with mem.timed("export") as detail:
        boxes, scores, classes = stitched["boxes"], stitched["scores"], stitched["classes"]
        polygons = seg.get("polygons", {})
        affine = tuple(plan["affine"]) if plan["affine"] else None

        features = geo.build_features(boxes, scores, classes, names, affine, polygons)
        transformer = None
        if plan["georeferenced"]:
            with RasterSource(scene, cfg.bands) as src:
                transformer = geo.wgs84_transformer(src.crs)
                written = geo.write_geojson(out_dir, stem, features, plan["crs"], affine, transformer)
                overview = to_uint8_rgb(src.overview_sample(1600),
                                        np.array(plan["stretch_lo"], np.float32),
                                        np.array(plan["stretch_hi"], np.float32), cfg.sensor)
        else:
            written = geo.write_geojson(out_dir, stem, features, None, None, None)
            with RasterSource(scene, cfg.bands) as src:
                overview = to_uint8_rgb(src.overview_sample(1600),
                                        np.array(plan["stretch_lo"], np.float32),
                                        np.array(plan["stretch_hi"], np.float32), cfg.sensor)

        written["coco"] = geo.write_coco(out_dir, stem, plan["width"], plan["height"],
                                         boxes, scores, classes, names, polygons)
        try:
            written["preview"] = geo.save_preview(overview, (plan["width"], plan["height"]),
                                                  boxes, out_dir / f"{stem}_preview.png", polygons)
        except Exception as e:
            mem.record("preview", "failed", 0.0, error=str(e)[:200])

        per_class: dict[str, int] = {}
        for c in classes:
            key = str(names.get(int(c), c))
            per_class[key] = per_class.get(key, 0) + 1
        detail.update(features=len(features), files=len(written))

    return {"files": written, "per_class": per_class, "features": len(features)}


# ------------------------------------------------------------------ per scene
def run_scene(scene: Path, cfg: JobConfig, out_dir: Path, mem: RunMemory,
              router: ResidencyManager) -> dict:
    """Run all five stages for one scene. Returns the per-scene summary block."""
    scene = Path(scene)
    key = content_key(scene)
    t0 = time.time()
    with mem.scene(key):
        plan = stage_plan(scene, cfg, mem)
        det = stage_detect(scene, cfg, plan, mem, router)
        stitched = stage_stitch(cfg, plan, det, mem)
        seg = stage_segment(scene, cfg, plan, stitched, mem, router)
        exported = stage_export(scene, cfg, plan, det, stitched, seg, out_dir, mem)
        stage_rows = [r.row() for r in mem.records if r.scene == key]

    return {
        "file": scene.name,
        "scene_key": key,
        "width": plan["width"], "height": plan["height"],
        "megapixels": plan["summary"]["megapixels"],
        "georeferenced": plan["georeferenced"], "crs": plan["crs"],
        "slicer": plan["summary"]["slicer"],
        "tiles_total": det["tiles_total"], "tiles_skipped_empty": det["tiles_skipped_empty"],
        "detections_before_merge": stitched["before_merge"],
        "detections_after_merge": stitched["after_merge"],
        "duplicates_removed": stitched["duplicates_removed"],
        "merge_backend": stitched["backend"],
        "polygons_from_masks": seg["masked"],
        "segment_requested": seg.get("requested", 0),
        "segment_failed": seg.get("failed", 0),
        "segment_skipped_low_score": seg.get("skipped_low_score", 0),
        "segment_note": seg.get("note", ""),
        "per_class": exported["per_class"],
        "outputs": exported["files"],
        "wall_s": round(time.time() - t0, 2),
        "stages": stage_rows,
    }


# -------------------------------------------------------------------- per job
def build_router(cfg: JobConfig, device=None, verbose: bool = True) -> ResidencyManager:
    """Register the experts named in the job. Nothing is loaded yet - the first
    `active()` call does that, and only one of them is ever resident."""
    device = pick_device(device)
    router = ResidencyManager(max_resident=cfg.max_resident_experts, verbose=verbose)
    router.register("detector", lambda: build_detector(cfg, device))
    if cfg.segmenter not in {"none", ""}:
        router.register("segmenter", lambda: build_segmenter(cfg, device))
    return router


def run_job(job_dir: Path, cfg: JobConfig | None = None, memory_root: Path | None = None,
            device=None, verbose: bool = True) -> dict:
    """Process every image in <job_dir>/input/ and write <job_dir>/output/.

    Does not move the job folder and never delivers anything - queue_worker.py
    owns the folder transitions and the human review gate.
    """
    job_dir = Path(job_dir)
    cfg = cfg or JobConfig.load(job_dir / "job.json")
    out_dir = job_dir / "output"
    out_dir.mkdir(exist_ok=True)
    mem = RunMemory(memory_root or job_dir, cfg.job_id, enabled=True)
    router = build_router(cfg, device=device, verbose=verbose)

    from .config import IMG_EXT
    scenes = sorted(p for p in (job_dir / "input").glob("*") if p.suffix.lower() in IMG_EXT)
    if not scenes:
        raise FileNotFoundError(f"no images in {job_dir / 'input'}")

    t0 = time.time()
    results = []
    try:
        for scene in scenes:
            results.append(run_scene(scene, cfg, out_dir, mem, router))
    finally:
        router.evict_all()      # never leave a model on the GPU between jobs

    summary = {
        "job_id": cfg.job_id,
        "customer": cfg.customer,
        "generated_utc": _now(),
        "pipeline": " -> ".join(STAGE_ORDER),
        "config": cfg.to_dict(),
        "licences": licence_notes(cfg),
        "scheduler": router.report(),
        "stage_seconds": mem.timings(),
        "vram_after_job": vram_mb(),
        "versions": versions(),
        "images": results,
        "totals": {
            "scenes": len(results),
            "detections": sum(r["detections_after_merge"] for r in results),
            "polygons": sum(r["polygons_from_masks"] for r in results),
            "wall_s": round(time.time() - t0, 2),
        },
        "accuracy": "NOT MEASURED - model output before human QA. Do not quote "
                    "precision, recall, IoU or mAP until a reviewer has checked a gold subset.",
    }
    (out_dir / "summary.json").write_text(_json(summary), encoding="utf-8")
    return summary


def versions() -> dict:
    out = {"numpy": np.__version__}
    for mod in ("torch", "ultralytics", "sahi", "rasterio", "cv2", "sam2", "rfdetr"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "installed")
        except Exception:
            pass
    try:
        import torch
        out["cuda"] = torch.version.cuda if torch.cuda.is_available() else "cpu-only"
    except Exception:
        pass
    return out


def _now() -> str:
    from .memory import now_iso
    return now_iso()


def _json(obj) -> str:
    import json

    from .memory import _json_default
    return json.dumps(obj, indent=2, default=_json_default)
