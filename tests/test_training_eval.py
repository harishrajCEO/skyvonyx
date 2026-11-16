"""Offline tests for dataset building and accuracy scoring.

Run:  python tests/test_training_eval.py
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

from skyvonyx import datasets, evaluate          # noqa: E402

PASS: list[str] = []


def ok(label: str) -> None:
    PASS.append(label)
    print(f"  ok  {label}")


def test_dota_loader(tmp: Path):
    print("1. DOTA labels")
    txt = tmp / "P0001.txt"
    txt.write_text("imagesource:GoogleEarth\ngsd:0.5\n"
                   "10 20 50 20 50 40 10 40 ship 0\n"
                   "100 100 140 90 150 130 110 140 plane 1\n"
                   "garbage line\n")
    lab = datasets.load_dota(tmp / "P0001.png", txt)
    assert lab.names == ["ship", "plane"], lab.names
    assert lab.boxes[0].tolist() == [10, 20, 50, 40]
    assert lab.boxes[1].tolist() == [100, 90, 150, 140], "rotated polygon -> enclosing box"
    assert lab.difficult.tolist() == [False, True]
    ok("header lines and junk skipped, polygons become enclosing boxes")
    assert datasets.load_dota(tmp / "P0001.png", txt, keep_difficult=False).names == ["ship"]
    ok("difficult objects can be excluded")


def test_split():
    print("2. scene splits")
    ids = [f"scene_{i}" for i in range(2000)]
    splits = [datasets.scene_split(s) for s in ids]
    assert splits == [datasets.scene_split(s) for s in ids], "must be deterministic"
    share = {k: splits.count(k) / len(ids) for k in ("train", "val", "test")}
    assert 0.12 < share["test"] < 0.18 and 0.12 < share["val"] < 0.18, share
    ok(f"hash split is deterministic and near the target shares {share}")


def test_clip():
    print("3. clipping boxes to tiles")
    boxes = np.array([[100, 100, 200, 200],      # fully inside
                      [950, 100, 1100, 200],     # 74/150 = 49% inside -> dropped
                      [900, 100, 1050, 200],     # 124/150 = 83% inside -> kept, clipped
                      [2000, 2000, 2100, 2100]], np.float32)   # outside
    local, idx = datasets.clip_boxes_to_window(boxes, 0, 0, 1024, 1024, 0.5, 4)
    assert idx.tolist() == [0, 2], idx
    assert local[1].tolist() == [900, 100, 1024, 200]
    ok("edge slivers under min_visible are dropped, the rest clipped to the tile")


def test_build_dataset(tmp: Path):
    print("4. building a YOLO dataset")
    scenes = []
    rng = np.random.default_rng(3)
    for i in range(12):
        img = (rng.integers(0, 255, (1300, 1800, 3)) // 4).astype(np.uint8)
        img[400:440, 500:580] = 255
        p = tmp / f"scene{i:02d}.png"
        cv2.imwrite(str(p), img)
        scenes.append(datasets.SceneLabels(p, np.array([[500, 400, 580, 440],
                                                        [10, 10, 30, 30]], np.float32),
                                           ["ship", "storage-tank"]))
    out = tmp / "ds"
    card = datasets.build_dataset(scenes, out, classes=["ship"], spec=datasets.TileSpec(tile=1024))
    splits = card["splits"]
    assert sum(s["scenes"] for s in splits.values()) == 12
    assert card["dropped_unknown_class"] == 12, "storage-tank is not in the class list"
    ok(f"12 scenes tiled at 1024px: { {k: v['tiles'] for k, v in splits.items()} } tiles per split")

    scene_of = lambda p: p.stem.split("__")[0]
    seen = {s: {scene_of(p) for p in (out / "images" / s).glob("*.png")}
            for s in ("train", "val", "test") if (out / "images" / s).exists()}
    for a in seen:
        for b in seen:
            if a < b:
                assert not seen[a] & seen[b], f"scene leaked across {a}/{b}"
    ok("no scene appears in two splits")

    label = next(p for p in (out / "labels").rglob("*.txt") if p.read_text().strip())
    cls, cx, cy, w, h = label.read_text().split()
    tile_png = out / "images" / label.parent.name / (label.stem + ".png")
    th, tw = cv2.imread(str(tile_png)).shape[:2]
    col, row = map(int, label.stem.split("__")[1].split("_"))
    assert cls == "0"
    assert abs(float(cx) * tw + col - 540) < 1 and abs(float(cy) * th + row - 420) < 1
    ok("YOLO label maps back to the planted object's scene position")

    y = (out / "data.yaml").read_text()
    assert "names:" in y and "0: ship" in y
    gold = json.loads((out / "test_scenes_gold.json").read_text())
    if splits.get("test", {}).get("scenes"):
        assert len(gold["images"]) == splits["test"]["scenes"]
        assert all(a["category_id"] == 0 for a in gold["annotations"])
    ok("data.yaml and full-scene test gold written")


def _coco(path: Path, items: dict, scored: bool):
    coco = {"images": [], "annotations": [], "categories": [{"id": 7, "name": "ship"},
                                                            {"id": 9, "name": "plane"}]}
    for i, (stem, boxes) in enumerate(items.items(), 1):
        coco["images"].append({"id": i, "file_name": stem + ".tif", "width": 1000, "height": 1000})
        for cls, x1, y1, x2, y2, s in boxes:
            a = {"id": len(coco["annotations"]) + 1, "image_id": i,
                 "category_id": 7 if cls == "ship" else 9, "bbox": [x1, y1, x2 - x1, y2 - y1]}
            if scored:
                a["score"] = s
            coco["annotations"].append(a)
    path.write_text(json.dumps(coco))


def test_evaluate(tmp: Path):
    print("5. accuracy scoring")
    gold = {"a": [("ship", 0, 0, 10, 10, 1), ("ship", 50, 50, 60, 60, 1), ("plane", 100, 100, 120, 120, 1)],
            "b": [("ship", 0, 0, 10, 10, 1)]}
    pred = {"a": [("ship", 0, 0, 10, 10, .9),       # TP
                  ("ship", 1, 1, 10, 10, .8),       # duplicate of the same ship -> FP
                  ("ship", 300, 300, 310, 310, .7), # false alarm -> FP
                  ("plane", 101, 101, 121, 121, .6)],  # TP, IoU 0.82
            "b": [("ship", 0, 0, 10, 10, .2)]}      # TP but under conf 0.25
    _coco(tmp / "gold.json", gold, scored=False)
    _coco(tmp / "pred_coco.json", pred, scored=True)
    r = evaluate.evaluate(evaluate.load_predictions(tmp / "pred_coco.json"),
                          evaluate.load_coco_boxes(tmp / "gold.json"), 0.5, 0.25)
    ship, plane = r["per_class"]["ship"], r["per_class"]["plane"]
    assert (ship["gold"], ship["tp"], ship["fp"], ship["fn"]) == (3, 1, 2, 2), ship
    assert ship["precision"] == round(1 / 3, 4) and ship["recall"] == round(1 / 3, 4)
    ok("a duplicate box counts as a false positive, a low-confidence hit as a miss")
    assert plane["tp"] == 1 and plane["fp"] == 0 and 0.8 < plane["mean_iou_of_matches"] < 0.85
    ok("classes are matched by name across differing category ids")
    # AP over all scores: ranks .9 TP, .8 FP, .7 FP, .2 TP with 3 gold ships
    # recall steps 1/3 at p=1, 2/3 at p=0.5 -> AP = 1/3*1 + 1/3*0.5 = 0.5
    assert abs(ship["ap50"] - 0.5) < 1e-4, ship["ap50"]
    ok("AP50 matches the hand-computed VOC value (0.5)")
    assert r["overall"]["tp"] == 2 and r["overall"]["gold"] == 4

    md = evaluate.write_accuracy_report(r, tmp / "acc.md", "synthetic test")
    text = md.read_text()
    assert "| ship | 3 | 1 | 2 | 2 |" in text and "synthetic test" in text
    ok("ACCURACY report table and test-set description written")

    self_r = evaluate.evaluate(evaluate.load_coco_boxes(tmp / "gold.json"),
                               evaluate.load_coco_boxes(tmp / "gold.json"), 0.5, 0.0)
    assert self_r["overall"]["precision"] == 1.0 and self_r["overall"]["recall"] == 1.0
    ok("a labeller scored against themselves agrees 100% (reviewer-agreement path)")


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        test_dota_loader(tmp)
        test_split()
        test_clip()
        test_build_dataset(tmp)
        test_evaluate(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nALL {len(PASS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
