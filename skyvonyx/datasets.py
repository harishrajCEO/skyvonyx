"""Build a YOLO training set that matches how the pipeline sees images.

Two defects in the first Colab notebook are fixed by construction here:

1. TRAIN/INFERENCE MISMATCH. The old model trained on whole DOTA images
   resized to 800 px, then inference sliced scenes into 1024 px tiles - so the
   model saw objects at one scale in training and another in production. This
   module cuts training tiles with the *same* slicer, the same per-scene
   contrast stretch and the same SAR dB conversion as `pipeline.stage_detect`,
   at the same tile size. What the detector trains on is what it will be shown.

2. LEAKY SPLITS. Splitting random tiles puts overlapping crops of one scene in
   both train and test, and the test score then measures memory, not skill.
   Splits here are by scene, assigned by a hash of the scene id, so a scene
   never changes split when more scenes are added later.

Label inputs: DOTA txt (the format you already have) and COCO json (what CVAT
and most SAR ship sets - HRSID, SSDD - ship as). Output: YOLO detect format
plus `data.yaml`, a `dataset_card.json` recording exactly what went in, and a
full-scene COCO gold file for the held-out test scenes so the whole pipeline -
not just the detector on tiles - can be scored afterwards.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .raster import RasterSource, is_empty_tile, stretch_limits, to_uint8_rgb
from .slicing import slice_plan


# ---------------------------------------------------------- licence registry
# Source of truth for "may a model trained on this be sold?". datasets/LICENCES.md
# is the human-readable copy; a test keeps the two in sync.
#
#   commercial: True  -> terms allow commercial use
#               False -> non-commercial / academic only (blocks commercial-ok)
#               None  -> not yet checked (also blocks: unknown is not "ok")
#   verified:   a person on the team read the actual licence text and recorded
#               it in LICENCES.md. Only verified=True + commercial=True passes.
#
# To clear a dataset: read its licence, update LICENCES.md with the quote and
# date, then set commercial/verified here. Do not flip verified without that.
DATASET_LICENCES: dict[str, dict] = {
    "dota":        {"commercial": False, "verified": False,
                    "terms": "reported academic-use-only (Google Earth imagery) - confirm"},
    "xview":       {"commercial": False, "verified": False,
                    "terms": "reported CC BY-NC-SA 4.0 - confirm"},
    "hrsid":       {"commercial": None, "verified": False,
                    "terms": "UNVERIFIED - read the release terms before any commercial use"},
    "ssdd":        {"commercial": None, "verified": False,
                    "terms": "UNVERIFIED - several redistributions exist; check the one you used"},
    "ls-ssdd-v1":  {"commercial": None, "verified": False, "terms": "UNVERIFIED"},
    "xview3-sar":  {"commercial": None, "verified": False, "terms": "UNVERIFIED"},
    "sardet-100k": {"commercial": None, "verified": False,
                    "terms": "UNVERIFIED - aggregates other sets, inherits the strictest terms"},
    "sen1floods11": {"commercial": None, "verified": False, "terms": "UNVERIFIED"},
    "sentinel-1":  {"commercial": True, "verified": True,
                    "terms": "Copernicus Sentinel data: free, full and open incl. commercial; "
                             "attribute 'Contains modified Copernicus Sentinel data [year]'"},
    "sentinel-2":  {"commercial": True, "verified": True,
                    "terms": "Copernicus Sentinel data: as sentinel-1"},
    "skyvonyx-labels": {"commercial": True, "verified": True,
                        "terms": "labels made by the Skyvonyx team; the IMAGERY under them "
                                 "still needs its own entry (e.g. sentinel-1)"},
    "customer-imagery": {"commercial": None, "verified": False,
                         "terms": "governed by the customer contract; default no training reuse"},
}


def licence_status(dataset_ids: list[str]) -> dict:
    """Verdict for a set of training sources. Fails closed.

    commercial_ok is True only if there is at least one source and every source
    is in the registry, verified, and commercial. Anything else - a
    non-commercial set, an unchecked set, a typo'd id, an empty list - blocks.
    """
    blocking, notes = [], {}
    for raw in dataset_ids:
        key = raw.strip().lower()
        entry = DATASET_LICENCES.get(key)
        if entry is None:
            blocking.append(f"{raw}: not in the licence registry")
            continue
        notes[key] = entry["terms"]
        if entry["commercial"] is False:
            blocking.append(f"{key}: non-commercial terms")
        elif entry["commercial"] is None or not entry["verified"]:
            blocking.append(f"{key}: licence not verified")
    if not dataset_ids:
        blocking.append("no dataset_ids declared - provenance unknown")
    return {"commercial_ok": not blocking, "blocking": blocking, "terms": notes,
            "note": "Data terms only. The detector licence (ultralytics AGPL-3.0) is separate."}


# ----------------------------------------------------------------- label I/O
@dataclass
class SceneLabels:
    image: Path
    boxes: np.ndarray                # (n, 4) xyxy, scene pixels
    names: list[str]                 # class name per box
    difficult: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def scene_id(self) -> str:
        return self.image.stem


def load_dota(image: str | Path, label_txt: str | Path, keep_difficult: bool = True) -> SceneLabels:
    """DOTA line: x1 y1 x2 y2 x3 y3 x4 y4 class difficult. Polygons become the
    enclosing horizontal box, which is what the pipeline's detector predicts."""
    boxes, names, diff = [], [], []
    for line in Path(label_txt).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 9 or line.startswith(("imagesource", "gsd")):
            continue
        try:
            xy = np.array(parts[:8], np.float32).reshape(4, 2)
        except ValueError:
            continue
        hard = len(parts) > 9 and parts[9] == "1"
        if hard and not keep_difficult:
            continue
        boxes.append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
        names.append(parts[8])
        diff.append(hard)
    return SceneLabels(Path(image), np.array(boxes, np.float32).reshape(-1, 4), names,
                       np.array(diff, bool))


def load_coco(coco_json: str | Path, images_dir: str | Path) -> list[SceneLabels]:
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    per_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        per_image.setdefault(ann["image_id"], []).append(ann)
    out = []
    for img in coco.get("images", []):
        anns = per_image.get(img["id"], [])
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                  a["bbox"][1] + a["bbox"][3]] for a in anns]
        names = [cats.get(a["category_id"], str(a["category_id"])) for a in anns]
        out.append(SceneLabels(Path(images_dir) / img["file_name"],
                               np.array(boxes, np.float32).reshape(-1, 4), names))
    return out


def load_voc(xml_dir: str | Path, images_dir: str | Path) -> list[SceneLabels]:
    """Pascal VOC XML, one file per image - the format SSDD is commonly distributed in."""
    import xml.etree.ElementTree as ET

    images = {p.stem: p for p in Path(images_dir).iterdir()
              if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}}
    out = []
    for xml in sorted(Path(xml_dir).glob("*.xml")):
        root = ET.parse(xml).getroot()
        stem = Path(root.findtext("filename") or xml.stem).stem
        if stem not in images:
            stem = xml.stem
        if stem not in images:
            continue
        boxes, names = [], []
        for obj in root.iter("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue     # rotated-only annotations: skip rather than guess
            boxes.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            names.append((obj.findtext("name") or "").strip())
        out.append(SceneLabels(images[stem], np.array(boxes, np.float32).reshape(-1, 4), names))
    return out


def manifest_from_dota(images_dir: str | Path, labels_dir: str | Path) -> list[SceneLabels]:
    images = {p.stem: p for p in Path(images_dir).iterdir()
              if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}}
    return [load_dota(images[t.stem], t) for t in sorted(Path(labels_dir).glob("*.txt"))
            if t.stem in images]


# -------------------------------------------------------------------- splits
def scene_split(scene_id: str, val_frac: float = 0.15, test_frac: float = 0.15) -> str:
    """Stable split by hash: adding scenes later never moves an existing one."""
    u = int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


# ------------------------------------------------------------------- tiling
def clip_boxes_to_window(boxes: np.ndarray, col: int, row: int, w: int, h: int,
                         min_visible: float, min_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (tile-local clipped boxes, indices kept).

    A box cut by the tile edge is kept only if at least `min_visible` of it is
    inside: teaching the model that a 10% sliver of a ship is a ship makes it
    fire on every tile edge at inference, and the re-stitch then has to undo it.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, int)
    x1 = np.clip(boxes[:, 0] - col, 0, w)
    y1 = np.clip(boxes[:, 1] - row, 0, h)
    x2 = np.clip(boxes[:, 2] - col, 0, w)
    y2 = np.clip(boxes[:, 3] - row, 0, h)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    vis = (x2 - x1) * (y2 - y1) / area
    keep = np.where((vis >= min_visible) & (x2 - x1 >= min_px) & (y2 - y1 >= min_px))[0]
    return np.stack([x1, y1, x2, y2], 1)[keep].astype(np.float32), keep


def _keep_negative(key: str, frac: float) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


@dataclass
class TileSpec:
    tile: int = 1024
    overlap: float = 0.2
    sensor: str = "optical"
    bands: list = field(default_factory=lambda: [1, 2, 3])
    min_visible: float = 0.5
    min_px: float = 4.0
    min_tile_std: float = 2.0
    negative_frac: float = 0.1     # share of object-free tiles kept, so the model learns "nothing here"
    # Chip datasets (HRSID 800px, SSDD ~500px) are smaller than the tile. The
    # trainer resizes every image's long side to imgsz, so an 800px chip would
    # be blown up 1.28x and the model would learn ships 28% bigger than it will
    # see at inference. Padding to the full tile keeps native scale.
    pad_to_tile: bool = True


def tile_scene(scene: SceneLabels, class_ids: dict[str, int], out_root: Path, split: str,
               spec: TileSpec) -> dict:
    import cv2

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    known = np.array([n in class_ids for n in scene.names], bool)
    boxes = scene.boxes[known] if len(scene.boxes) else scene.boxes
    cls = np.array([class_ids[n] for n, k in zip(scene.names, known) if k], int)
    stats = {"scene": scene.scene_id, "split": split, "tiles": 0, "objects": 0,
             "negatives": 0, "skipped_empty": 0, "dropped_unknown_class": int((~known).sum())}

    with RasterSource(scene.image, spec.bands) as src:
        lo, hi = stretch_limits(src.overview_sample(), spec.sensor, src.nodata)
        for (col, row, w, h) in slice_plan(src.width, src.height, spec.tile, spec.overlap):
            rgb = to_uint8_rgb(src.read(col, row, w, h), lo, hi, spec.sensor)
            if is_empty_tile(rgb, spec.min_tile_std):
                stats["skipped_empty"] += 1
                continue
            local, idx = clip_boxes_to_window(boxes, col, row, w, h, spec.min_visible, spec.min_px)
            name = f"{scene.scene_id}__{col}_{row}"
            if len(local) == 0:
                if not _keep_negative(name, spec.negative_frac):
                    continue
                stats["negatives"] += 1
            out_w, out_h = w, h
            if spec.pad_to_tile and (w < spec.tile or h < spec.tile):
                canvas = np.zeros((max(h, spec.tile), max(w, spec.tile), 3), np.uint8)
                canvas[:h, :w] = rgb
                rgb, out_w, out_h = canvas, canvas.shape[1], canvas.shape[0]
                stats["padded"] = stats.get("padded", 0) + 1
            cv2.imwrite(str(img_dir / f"{name}.png"), rgb[..., ::-1])
            # normalise by the written image size, so padding does not shift labels
            lines = [f"{cls[i]} {(b[0] + b[2]) / 2 / out_w:.6f} {(b[1] + b[3]) / 2 / out_h:.6f} "
                     f"{(b[2] - b[0]) / out_w:.6f} {(b[3] - b[1]) / out_h:.6f}"
                     for b, i in zip(local, idx)]
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            stats["tiles"] += 1
            stats["objects"] += len(lines)
        stats["width"], stats["height"] = src.width, src.height
    return stats


# ------------------------------------------------------------------ dataset
def build_dataset(scenes: list[SceneLabels], out_root: str | Path, classes: list[str],
                  spec: TileSpec | None = None, val_frac: float = 0.15, test_frac: float = 0.15,
                  source: str = "", dataset_ids: list[str] | None = None,
                  group_fn=None) -> dict:
    """Tile every scene into YOLO format, split by scene. Returns the dataset card.

    dataset_ids: keys of DATASET_LICENCES naming every source of images or
        labels. Required to get a licence verdict; omitted means "unknown",
        which can never be commercial-ok.
    group_fn: maps an image stem to the scene it was cut from. Chip datasets
        cut one big scene into many chips; splitting by chip would put
        neighbours from the same scene in train and test. HRSID-style names
        like "P0001_0_800_7200_8000" group with `lambda s: s.split("_")[0]`.
    """
    spec = spec or TileSpec()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    class_ids = {c: i for i, c in enumerate(classes)}

    per_scene = []
    gold = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c} for i, c in enumerate(classes)]}
    for scene in scenes:
        group = group_fn(scene.scene_id) if group_fn else scene.scene_id
        split = scene_split(group, val_frac, test_frac)
        stats = tile_scene(scene, class_ids, out_root, split, spec)
        per_scene.append(stats)
        if split == "test":
            # full-scene gold for scoring the whole pipeline later, in scene pixels
            img_id = len(gold["images"]) + 1
            gold["images"].append({"id": img_id, "file_name": scene.image.name,
                                   "path": str(scene.image),
                                   "width": stats["width"], "height": stats["height"]})
            for b, n in zip(scene.boxes, scene.names):
                if n in class_ids:
                    gold["annotations"].append({
                        "id": len(gold["annotations"]) + 1, "image_id": img_id,
                        "category_id": class_ids[n],
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        "area": float((b[2] - b[0]) * (b[3] - b[1])), "iscrowd": 0})

    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)),
        encoding="utf-8")
    (out_root / "test_scenes_gold.json").write_text(json.dumps(gold), encoding="utf-8")

    totals: dict = {}
    for s in per_scene:
        t = totals.setdefault(s["split"], {"scenes": 0, "tiles": 0, "objects": 0, "negatives": 0})
        t["scenes"] += 1
        t["tiles"] += s["tiles"]
        t["objects"] += s["objects"]
        t["negatives"] += s["negatives"]
    card = {
        "classes": classes, "tile": spec.tile, "overlap": spec.overlap, "sensor": spec.sensor,
        "bands": spec.bands, "min_visible": spec.min_visible, "negative_frac": spec.negative_frac,
        "split_rule": f"sha256(scene_id) - val {val_frac}, test {test_frac}, by scene",
        "source": source, "dataset_ids": list(dataset_ids or []),
        "licence_status": licence_status(dataset_ids or []),
        "split_groups": len({(group_fn or (lambda x: x))(sc.scene_id) for sc in scenes}),
        "splits": totals,
        "dropped_unknown_class": sum(s["dropped_unknown_class"] for s in per_scene),
        "scenes": per_scene,
        "warning": "Train at this tile size and run inference with the same tile, overlap "
                   "and sensor settings, or the numbers measured here do not apply.",
    }
    (out_root / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card
