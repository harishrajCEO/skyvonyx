"""Offline test suite - no GPU, no weights, no network.

Covers everything except the model forward pass: slice geometry, empty-tile
skipping, cross-tile re-stitching, polygon export, georeferencing arithmetic,
the residency rules, memory resume, and the queue's review gate.

Run:  python tests/test_pipeline.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyvonyx import geo, queue_worker, slicing, stitching          # noqa: E402
from skyvonyx.config import JobConfig                                # noqa: E402
from skyvonyx.experts.base import normalise_masks                    # noqa: E402
from skyvonyx.experts.fake import FakeDetector, FakeSegmenter        # noqa: E402
from skyvonyx.memory import RunMemory, content_key                   # noqa: E402
from skyvonyx.raster import (is_empty_tile, stretch_limits,          # noqa: E402
                             tile_windows, to_uint8_rgb)
from skyvonyx.residency import ResidencyManager                      # noqa: E402

PASS: list[str] = []


def ok(label: str) -> None:
    PASS.append(label)
    print(f"  ok  {label}")


# ============================================================= 1. slice plan
def test_slicing():
    print("1. slicing")
    wins = tile_windows(2500, 1300, 1024, 0.2)
    covered = np.zeros((1300, 2500), bool)
    for c, r, w, h in wins:
        assert w <= 1024 and h <= 1024 and c + w <= 2500 and r + h <= 1300
        covered[r:r + h, c:c + w] = True
    assert covered.all(), "tiles must cover every pixel"
    ok(f"tiles cover the scene with no gap ({len(wins)} tiles)")

    assert tile_windows(500, 400, 1024, 0.2) == [(0, 0, 500, 400)]
    ok("a scene smaller than one tile yields exactly one window")

    plan = slicing.slice_plan(2500, 1300, 1024, 0.2)
    cov2 = np.zeros((1300, 2500), bool)
    for c, r, w, h in plan:
        assert c >= 0 and r >= 0 and c + w <= 2500 and r + h <= 1300, (c, r, w, h)
        cov2[r:r + h, c:c + w] = True
    assert cov2.all(), "SAHI plan must also cover the scene"
    ok(f"slice_plan stays inside the raster and covers it ({slicing.plan_summary(plan, 2500, 1300, 1024, 0.2)['slicer']})")


# ========================================================== 2. 8-bit render
def test_render():
    print("2. rendering")
    sar = np.random.gamma(1.0, 0.05, (1, 64, 64)).astype(np.float32)
    lo, hi = stretch_limits(sar, "sar")
    rgb = to_uint8_rgb(sar, lo, hi, "sar")
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    ok("single-band SAR renders to 3-channel uint8 via dB stretch")

    flat = np.zeros((3, 32, 32), np.float32)
    assert is_empty_tile(to_uint8_rgb(flat, np.zeros(3, np.float32), np.ones(3, np.float32), "optical"))
    ok("a flat tile is detected as empty and will be skipped")

    with_nodata = np.concatenate([np.full((1, 10, 10), -9999.0, np.float32),
                                 np.random.rand(1, 10, 10).astype(np.float32) * 100])
    lo2, hi2 = stretch_limits(with_nodata[1:2], "optical", nodata=-9999.0)
    assert lo2[0] >= 0, "nodata must not drag the low percentile down"
    ok("nodata values are excluded from the contrast stretch")


# ============================================================ 3. re-stitch
def test_stitching():
    print("3. re-stitch")
    # a box cut at a tile edge sits inside the full box: low IoU, high IoS
    cut = np.array([1000, 900, 1024, 940], np.float32)
    full = np.array([[1000, 900, 1060, 940]], np.float32)
    assert stitching.iou_one_to_many(cut, full)[0] < 0.5
    assert stitching.ios_one_to_many(cut, full)[0] > 0.99
    ok("IoS catches a tile-edge fragment that IoU misses")

    boxes = np.array([[1000, 900, 1024, 940], [1000, 900, 1060, 940],
                      [1000, 900, 1060, 940]], np.float32)
    mb, ms, mc, groups = stitching.greedy_merge(boxes, np.array([.9, .9, .5]),
                                                np.array([0, 0, 1]), 0.5, "IOS")
    assert len(mb) == 2, f"expected one merged box per class, got {len(mb)}"
    assert mb[0].tolist() == [1000, 900, 1060, 940], mb[0]
    assert sorted(groups[0]) == [0, 1]
    ok("greedy merge fuses the fragment into the union box, per class")

    # Full path. One object at scene x 900..1060, seen by two adjacent tiles of
    # a 1024px/20% plan (steps of 819): tile 0 sees it cut off at its right
    # edge, tile 1 sees all of it. Both must collapse to one scene box.
    res = stitching.restitch(
        tile_boxes=np.array([[900, 100, 1024, 140],      # tile 0, local == scene
                             [81, 100, 241, 140]],       # tile 1, local
                            np.float32),
        tile_scores=np.array([0.8, 0.7], np.float32),
        tile_classes=np.array([0, 0]),
        offsets=np.array([[0, 0], [819, 0]], np.float32),
        names={0: "ship"}, shape=(1000, 2000), threshold=0.5, metric="IOS")
    assert res["before"] == 2 and res["after"] == 1, res
    assert res["boxes"][0].tolist() == [900, 100, 1060, 140], res["boxes"][0]
    ok(f"two tiles -> one object at the right scene coordinates ({res['backend']})")

    empty = stitching.restitch(np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0, int),
                              np.zeros((0, 2)), {}, (100, 100))
    assert empty["after"] == 0 and empty["boxes"].shape == (0, 4)
    ok("zero detections is handled without a crash")


# ========================================================== 4. georeferencing
def test_geo():
    print("4. georeferencing")
    # Sentinel-2 style: 10 m pixels, north-up, UTM
    affine = (10.0, 0.0, 300000.0, 0.0, -10.0, 2100000.0)
    x, y = geo.pixel_to_map(affine, np.array([0, 100]), np.array([0, 50]))
    assert (x == [300000, 301000]).all() and (y == [2100000, 2099500]).all()
    ok("pixel -> map uses the full affine (north-up UTM case)")

    feats = geo.build_features(np.array([[100, 50, 110, 60]], np.float32), [0.9], [3],
                               {3: "ship"}, affine)
    ring = feats[0]["geometry"]["coordinates"][0]
    assert ring[0] == [301000.0, 2099500.0], ring[0]
    assert ring[0] == ring[-1], "ring must close"
    assert feats[0]["properties"]["geometry_source"] == "detector_box"
    assert feats[0]["properties"]["status"] == "unreviewed"
    ok("a box becomes a closed georeferenced ring marked unreviewed")

    mask = np.zeros((40, 40), bool)
    mask[10:30, 12:28] = True
    rings = geo.mask_to_polygons(mask, offset=(1000, 2000), simplify_px=1.0)
    assert len(rings) == 1 and len(rings[0]) >= 4
    xs = [p[0] for p in rings[0]]
    ys = [p[1] for p in rings[0]]
    assert 1011 <= min(xs) <= 1013 and 2009 <= min(ys) <= 2011, (min(xs), min(ys))
    ok("a mask becomes a simplified ring shifted into scene coordinates")

    polys = {0: rings}
    feats = geo.build_features(np.array([[1012, 2010, 1028, 2030]], np.float32), [0.9], [0],
                               {0: "x"}, None, polys)
    assert feats[0]["properties"]["geometry_source"] == "sam_mask"
    ok("when SAM produced a mask the feature uses the polygon, not the box")

    rotated = (8.0, 2.0, 500.0, 2.0, -8.0, 900.0)   # non-north-up transform
    rx, ry = geo.pixel_to_map(rotated, np.array([10.0]), np.array([5.0]))
    assert rx[0] == 8 * 10 + 2 * 5 + 500 and ry[0] == 2 * 10 - 8 * 5 + 900
    ok("rotated/sheared transforms are handled (b and d terms are not dropped)")


# ============================================================= 5. residency
def test_residency():
    print("5. residency (one expert at a time)")
    built: list[str] = []
    killed: list[str] = []

    class Dummy:
        def __init__(self, name):
            self.name = name
            built.append(name)

        def unload(self):
            killed.append(self.name)

    router = ResidencyManager(max_resident=1, verbose=False)
    router.register("detector", lambda: Dummy("detector"))
    router.register("segmenter", lambda: Dummy("segmenter"))

    with router.active("detector") as d:
        assert d.name == "detector"
        assert not router.slots["segmenter"].resident
    ok("activating the detector does not load the segmenter")

    with router.active("segmenter") as s:
        assert s.name == "segmenter"
        assert not router.slots["detector"].resident, "detector must be evicted"
    assert killed == ["detector"], killed
    ok("activating the segmenter evicts the detector first")

    with router.active("segmenter"):
        pass
    assert built.count("segmenter") == 1, "a repeat activation must not reload"
    ok("two consecutive stages on one expert pay for a single load")

    router.evict_all()
    assert not any(sl.resident for sl in router.slots.values())
    ok("evict_all leaves nothing resident")

    rep = router.report()
    assert rep["max_resident_experts"] == 1 and rep["experts"]["detector"]["loads"] == 1
    assert len(rep["events"]) == 3
    ok("the router reports loads, evictions and VRAM per activation")

    pair = ResidencyManager(max_resident=2, verbose=False)
    pair.register("a", lambda: Dummy("a"))
    pair.register("b", lambda: Dummy("b"))
    with pair.active("a"):
        pass
    with pair.active("b"):
        assert pair.slots["a"].resident, "max_resident=2 should keep both"
    ok("max_resident=2 keeps both experts loaded (opt-in, for a bigger GPU)")

    try:
        with router.active("nope"):
            pass
    except KeyError as e:
        assert "nope" in str(e)
        ok("an unregistered expert fails loudly")


# ================================================================ 6. memory
def test_memory():
    print("6. memory + resume")
    root = Path(tempfile.mkdtemp())
    try:
        mem = RunMemory(root, "job1")
        with mem.scene("scene_abc"):
            assert mem.get_json("plan", "fp1") is None
            mem.put_json("plan", "fp1", {"tiles": 9})
            assert mem.get_json("plan", "fp1") == {"tiles": 9}
            assert mem.get_json("plan", "fp2") is None, "a different fingerprint must miss"
        ok("stage payloads round-trip and are keyed by fingerprint")

        with mem.scene("scene_abc"):
            mem.put_arrays("detect", "fp1", {"boxes": np.zeros((3, 4), np.float32)})
            back = mem.get_arrays("detect", "fp1")
            assert back is not None and back["boxes"].shape == (3, 4)
        ok("detections round-trip through compressed npz")

        with mem.scene("scene_abc"):
            with mem.timed("detect") as d:
                d["raw_detections"] = 7
            try:
                with mem.timed("segment"):
                    raise RuntimeError("sam exploded")
            except RuntimeError:
                pass
        rows = [json.loads(l) for l in (root / "memory" / "job1" / "ledger.jsonl").read_text().splitlines()]
        assert rows[-2]["stage"] == "detect" and rows[-2]["raw_detections"] == 7
        assert rows[-1]["stage"] == "segment" and rows[-1]["status"] == "failed"
        assert "sam exploded" in rows[-1]["error"]
        ok("the ledger records a row per stage, failures included")

        corrupt = root / "memory" / "job1" / "scene_abc" / "plan.bad.json"
        corrupt.write_text("{not json")
        with mem.scene("scene_abc"):
            assert mem.get_json("plan", "bad") is None
            assert not corrupt.exists(), "a corrupt payload must be discarded"
        ok("a half-written payload from a killed session is dropped, not served")

        f = root / "img.bin"
        f.write_bytes(b"x" * 5000)
        k1 = content_key(f)
        f.write_bytes(b"y" * 5000)
        assert content_key(f) != k1
        ok("content_key changes when the file content changes")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ====================================================== 7. config fingerprints
def test_config():
    print("7. config fingerprints")
    a = JobConfig(job_id="j")
    b = JobConfig(job_id="j", conf=0.5)
    assert a.stage_fingerprint("plan") == b.stage_fingerprint("plan")
    assert a.stage_fingerprint("detect") != b.stage_fingerprint("detect")
    assert a.stage_fingerprint("export") != b.stage_fingerprint("export")
    ok("changing conf reuses the slice plan but invalidates detect and everything after")

    c = JobConfig(job_id="j", merge_threshold=0.9)
    assert a.stage_fingerprint("detect") == c.stage_fingerprint("detect")
    assert a.stage_fingerprint("stitch") != c.stage_fingerprint("stitch")
    ok("changing the merge threshold does not force re-detection")

    d = JobConfig(job_id="j", tile=640)
    assert all(a.stage_fingerprint(s) != d.stage_fingerprint(s)
               for s in ("plan", "detect", "stitch", "segment", "export"))
    ok("changing the tile size invalidates every stage")

    p = Path(tempfile.mkdtemp()) / "job.json"
    p.write_text(json.dumps({"job_id": "x", "conf": 0.4, "typo_setting": 1}))
    loaded = JobConfig.load(p)
    assert loaded.conf == 0.4 and loaded.unknown_keys == ["typo_setting"]
    ok("an unrecognised job.json key is surfaced instead of silently ignored")
    shutil.rmtree(p.parent, ignore_errors=True)


# ============================================================== 8. mask shapes
def test_mask_normalisation():
    print("8. SAM output shapes")
    one = np.ones((1, 20, 20))
    assert len(normalise_masks(one, 1, (20, 20))) == 1
    many = np.ones((5, 1, 20, 20))
    assert len(normalise_masks(many, 5, (20, 20))) == 5
    ok("(C,H,W) for one box and (N,C,H,W) for many both collapse to N masks")
    assert len(normalise_masks(np.ones((20, 20)), 1, (20, 20))) == 1
    ok("a bare (H,W) mask is accepted")
    assert normalise_masks(np.ones((5, 1, 9, 9)), 5, (20, 20)) == []
    ok("masks of the wrong size are rejected rather than mis-georeferenced")


# ========================================================== 9. end-to-end job
def make_scene(path: Path, w=3000, h=2000) -> list[tuple[int, int]]:
    """Noisy scene, a flat strip that must be skipped, and two planted objects -
    one of them straddling a tile boundary so the merge is exercised."""
    rng = np.random.default_rng(7)
    img = (rng.integers(0, 255, (h, w, 3), dtype=np.int16) // 4).astype(np.uint8)
    img[:, 1900:] = 0                     # flat -> empty tiles
    img[900:940, 1000:1060] = 255         # object A: spans the 819px tile step
    img[300:330, 200:240] = 255           # object B: inside one tile
    cv2.imwrite(str(path), img)
    return [(1000, 900), (200, 300)]


def test_end_to_end():
    print("9. end-to-end job through the queue")
    root = Path(tempfile.mkdtemp())
    try:
        scene = root / "scene.png"
        make_scene(scene)

        job = queue_worker.create_job(
            root, "demo_001", [scene],
            detector="fake", segmenter="fake", detector_weights="none",
            tile=1024, overlap=0.2, conf=0.1, batch=4)
        assert (job / "READY").exists()
        ok("create_job stages input + job.json and writes READY last")

        no_ready = queue_worker.create_job(root, "half_002", [scene], detector="fake",
                                           segmenter="fake")
        (no_ready / "READY").unlink()
        stats = queue_worker.tick(root, verbose=False)
        assert stats["processed"] == 1, stats
        assert (root / "jobs" / "inbox" / "half_002").exists(), "job without READY must be left alone"
        ok("one READY job processed; the half-uploaded one is untouched")

        review = root / "jobs" / "review" / "demo_001"
        summary = json.loads((review / "output" / "summary.json").read_text())
        img = summary["images"][0]

        assert img["tiles_skipped_empty"] >= 1, img
        ok(f"empty tiles skipped: {img['tiles_skipped_empty']} of {img['tiles_total']}")

        assert img["detections_before_merge"] > img["detections_after_merge"], img
        assert img["detections_after_merge"] == 2, f"expected the 2 planted objects, got {img}"
        ok(f"{img['detections_before_merge']} raw detections re-stitched to "
           f"{img['detections_after_merge']} objects")

        assert img["polygons_from_masks"] == 2, img
        ok("SAM stage produced a polygon for each object")

        assert img["georeferenced"] is False
        gj_path = review / "output" / "scene_PIXEL_COORDS_NOT_GEOREFERENCED.geojson"
        assert gj_path.exists(), list((review / "output").iterdir())
        ok("a PNG without a CRS is exported under a filename that says so")

        gj = json.loads(gj_path.read_text())
        found = {tuple(f["properties"]["px_bbox"][:2]) for f in gj["features"]}
        assert (1000.0, 900.0) in found, found
        assert all(f["properties"]["geometry_source"] == "sam_mask" for f in gj["features"])
        assert all(f["properties"]["status"] == "unreviewed" for f in gj["features"])
        ok("exported geometry lands on the planted pixel coordinates")

        coco = json.loads((review / "output" / "scene_coco.json").read_text())
        assert len(coco["annotations"]) == 2 and "segmentation" in coco["annotations"][0]
        ok("COCO export carries boxes and segmentation")

        assert (review / "output" / "scene_preview.png").exists()
        report = (review / "output" / "REPORT_DRAFT.md").read_text(encoding="utf-8")
        assert "NOT MEASURED" in report
        assert "PIXEL COORDS ONLY" in report or "WARNING" in report
        assert "AGPL" not in report, "fake detector must not claim an AGPL licence"
        ok("REPORT_DRAFT says NOT MEASURED and warns about the missing CRS")

        sched = summary["scheduler"]
        assert sched["max_resident_experts"] == 1
        assert [e["expert"] for e in sched["events"]] == ["detector", "segmenter"]
        assert sched["events"][1]["evicted"] == ["detector"], sched["events"]
        ok("the run proves the order: detector ran, was evicted, then the segmenter loaded")

        assert set(summary["stage_seconds"]) >= {"plan", "detect", "stitch", "segment", "export"}
        ok(f"per-stage timings recorded: {summary['stage_seconds']}")
        assert "NOT MEASURED" in summary["accuracy"]
        ok("summary refuses to state an accuracy number")

        # ---- the human gate
        assert queue_worker.tick(root, verbose=False)["approved"] == 0
        assert (root / "jobs" / "review" / "demo_001").exists()
        ok("nothing leaves review/ without a human")

        (review / "APPROVED").touch()
        assert queue_worker.tick(root, verbose=False)["approved"] == 1
        done = root / "jobs" / "done" / "demo_001"
        assert done.exists()
        state = json.loads((done / "status.json").read_text())
        assert state["state"] == "approved_ready_to_deliver"
        assert "never sends" in state["note"]
        ok("APPROVED promotes the job to done/, delivery still manual")

        # ---- resume: the Colab-disconnect case. A job that already ran once is
        # pushed back to inbox (what requeue_stale does) and re-run. Its memory
        # travels in the job folder, so every stage should come back cached.
        rerun = queue_worker.create_job(root, "demo_003", [scene], detector="fake",
                                        segmenter="fake", tile=1024, overlap=0.2,
                                        conf=0.1, batch=4)
        queue_worker.tick(root, verbose=False)
        first = json.loads((root / "jobs" / "review" / "demo_003" / "output" /
                            "summary.json").read_text())
        assert not [s for s in first["images"][0]["stages"] if s["status"] == "cached"]

        requeued = queue_worker.move(root / "jobs" / "review" / "demo_003", root, "inbox")
        shutil.rmtree(requeued / "output")          # prove export genuinely re-runs
        queue_worker.tick(root, verbose=False)
        second = json.loads((root / "jobs" / "review" / "demo_003" / "output" /
                             "summary.json").read_text())
        cached = [s for s in second["images"][0]["stages"] if s["status"] == "cached"]
        assert {c["stage"] for c in cached} == {"plan", "detect", "stitch", "segment"}, cached
        assert second["images"][0]["detections_after_merge"] == 2
        assert second["scheduler"]["experts"]["detector"]["loads"] == 0, \
            "a fully cached run must not load any model"
        ok(f"after an interruption {len(cached)} stages came from memory, "
           f"no expert was loaded, and the 2 objects still exported")

        # ---- failure path
        queue_worker.create_job(root, "bad_004", [], detector="fake")
        assert queue_worker.tick(root, verbose=False)["failed"] == 1
        failed = root / "jobs" / "failed" / "bad_004"
        assert "FileNotFoundError" in (failed / "error.txt").read_text()
        ok("a job with no images lands in failed/ with its traceback")

        # ---- rejection path
        rej = root / "jobs" / "review" / "demo_003"
        (rej / "REJECTED").write_text("boxes are off by a tile")
        assert queue_worker.tick(root, verbose=False)["rejected"] == 1
        st = json.loads((root / "jobs" / "failed" / "demo_003" / "status.json").read_text())
        assert "off by a tile" in st["reason"]
        ok("REJECTED moves the job to failed/ and keeps the reason")

        # ---- interrupted session
        stuck = queue_worker.create_job(root, "stuck_005", [scene], detector="fake")
        queue_worker.move(stuck, root, "processing")
        assert queue_worker.requeue_stale(root) == ["stuck_005"]
        assert (root / "jobs" / "inbox" / "stuck_005" / "READY").exists()
        ok("a job abandoned in processing/ is requeued for the next session")

        state = queue_worker.queue_state(root)
        assert "demo_001" in state["done"] and "bad_004" in state["failed"]
        ok(f"queue_state reports the board: { {k: len(v) for k, v in state.items()} }")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ====================================================== 10. fake expert sanity
def test_fakes():
    print("10. test doubles behave like the real experts")
    det = FakeDetector()
    tile = np.zeros((100, 100, 3), np.uint8)
    tile[20:40, 30:60] = 255
    (boxes, scores, classes), = det([tile])
    assert boxes.shape == (1, 4) and classes[0] == 0
    assert boxes[0].tolist() == [30, 20, 60, 40], boxes[0]
    ok("FakeDetector returns xyxy in tile-local pixels")

    seg = FakeSegmenter()
    masks = seg.masks_for_boxes(tile, boxes)
    assert len(masks) == 1 and masks[0].shape == (100, 100) and masks[0].sum() > 0
    ok("FakeSegmenter returns one full-crop boolean mask per box")

    assert det([]) == []
    assert seg.masks_for_boxes(tile, np.zeros((0, 4))) == []
    ok("both handle the empty case")


def main() -> int:
    for fn in (test_slicing, test_render, test_stitching, test_geo, test_residency,
               test_memory, test_config, test_mask_normalisation, test_end_to_end,
               test_fakes):
        fn()
    print(f"\nALL {len(PASS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
