"""Stage 3 - re-stitch tiled detections into one scene.

Why this stage exists at all: a 1024 px tile with 20% overlap means a ship near
a tile edge is detected twice, and a long ship is detected as two halves. Plain
IoU NMS does not fix the halves - a cut-off box has low IoU with the full box
but sits almost entirely inside it. The metric that catches that is IoS,
intersection over the *smaller* box, which is what SAHI's GREEDYNMM uses.

SAHI does the merge when it is installed (same algorithm the rest of the
ecosystem reports numbers against). The numpy fallback implements the same
greedy class-aware IoS merge so results stay comparable on a bare environment,
and the summary always records which one ran.

CPU only. No expert is resident during this stage.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- overlap
def _areas(b: np.ndarray) -> np.ndarray:
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def _intersection(b: np.ndarray, bs: np.ndarray) -> np.ndarray:
    x1 = np.maximum(b[0], bs[:, 0])
    y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2])
    y2 = np.minimum(b[3], bs[:, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def iou_one_to_many(b: np.ndarray, bs: np.ndarray) -> np.ndarray:
    inter = _intersection(b, bs)
    return inter / np.maximum(_areas(b) + _areas(bs) - inter, 1e-9)


def ios_one_to_many(b: np.ndarray, bs: np.ndarray) -> np.ndarray:
    """Intersection over the smaller box - catches tile-edge fragments."""
    inter = _intersection(b, bs)
    return inter / np.maximum(np.minimum(_areas(b), _areas(bs)), 1e-9)


# ------------------------------------------------------------------ fallbacks
def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                    threshold: float, metric: str = "IOU") -> np.ndarray:
    """Suppress (do not merge). Returns indices to keep, ascending."""
    overlap = ios_one_to_many if metric.upper() == "IOS" else iou_one_to_many
    keep: list[int] = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx])]
        while idx.size:
            i = int(idx[0])
            keep.append(i)
            if idx.size == 1:
                break
            rest = idx[1:]
            idx = rest[overlap(boxes[i], boxes[rest]) < threshold]
    return np.array(sorted(keep), dtype=int)


def greedy_merge(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                 threshold: float = 0.5, metric: str = "IOS"):
    """Class-aware greedy merge: overlapping boxes fuse into their union box.

    Returns (boxes, scores, classes, source_groups) where source_groups[i] lists
    the input indices that went into output box i - kept so a reviewer can trace
    a merged detection back to the tiles it came from.
    """
    n = len(boxes)
    if n == 0:
        return (boxes.reshape(0, 4).astype(np.float32), scores.astype(np.float32),
                classes.astype(int), [])
    overlap = ios_one_to_many if metric.upper() == "IOS" else iou_one_to_many
    out_b, out_s, out_c, out_g = [], [], [], []
    for c in np.unique(classes):
        idx = list(np.where(classes == c)[0][np.argsort(-scores[np.where(classes == c)[0]])])
        while idx:
            i = idx.pop(0)
            box = boxes[i].astype(np.float32).copy()
            score = float(scores[i])
            group = [int(i)]
            changed = True
            while changed and idx:          # grow the union until nothing else touches it
                rest = np.asarray(idx, dtype=int)
                hit = overlap(box, boxes[rest]) >= threshold
                changed = bool(hit.any())
                if changed:
                    grp = boxes[rest[hit]]
                    box = np.array([min(box[0], grp[:, 0].min()), min(box[1], grp[:, 1].min()),
                                    max(box[2], grp[:, 2].max()), max(box[3], grp[:, 3].max())],
                                   np.float32)
                    score = max(score, float(scores[rest[hit]].max()))
                    group += [int(j) for j in rest[hit]]
                    idx = [j for j, h in zip(idx, hit) if not h]
            out_b.append(box)
            out_s.append(score)
            out_c.append(int(c))
            out_g.append(sorted(group))
    return (np.array(out_b, np.float32), np.array(out_s, np.float32),
            np.array(out_c, int), out_g)


# ----------------------------------------------------------------- SAHI merge
def sahi_available() -> bool:
    try:
        import sahi.postprocess.combine  # noqa: F401
        import sahi.prediction           # noqa: F401
        return True
    except ImportError:
        return False


def _sahi_restitch(boxes, scores, classes, names, threshold, metric, algorithm, shape):
    from sahi.postprocess.combine import GreedyNMMPostprocess, NMMPostprocess, NMSPostprocess
    from sahi.prediction import ObjectPrediction

    # Boxes are already shifted to full-scene coordinates, so shift_amount is
    # [0, 0]: we do the offsetting ourselves in restitch() and do not rely on
    # SAHI's shift bookkeeping.
    preds = [
        ObjectPrediction(
            bbox=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            category_id=int(k),
            category_name=str(names.get(int(k), k)),
            score=float(s),
            shift_amount=[0, 0],
            full_shape=[int(shape[0]), int(shape[1])],
        )
        for b, s, k in zip(boxes, scores, classes)
    ]
    cls = {"GREEDYNMM": GreedyNMMPostprocess, "NMM": NMMPostprocess,
           "NMS": NMSPostprocess}[algorithm.upper()]
    merged = cls(match_threshold=threshold, match_metric=metric.upper(),
                 class_agnostic=False)(preds)
    mb = np.array([p.bbox.to_xyxy() for p in merged], np.float32).reshape(-1, 4)
    ms = np.array([p.score.value for p in merged], np.float32)
    mc = np.array([p.category.id for p in merged], int)
    return mb, ms, mc


def restitch(tile_boxes: np.ndarray, tile_scores: np.ndarray, tile_classes: np.ndarray,
             offsets: np.ndarray, names: dict, shape: tuple[int, int],
             threshold: float = 0.5, metric: str = "IOS", algorithm: str = "GREEDYNMM",
             prefer_sahi: bool = True) -> dict:
    """Shift tile-local boxes to scene coordinates, then merge duplicates.

    tile_boxes  (N, 4) xyxy in tile-local pixels
    offsets     (N, 2) the (col_off, row_off) of each box's tile
    Returns a dict with scene-coordinate boxes and the merge provenance.
    """
    n_in = len(tile_boxes)
    if n_in == 0:
        return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
                "classes": np.zeros(0, int), "groups": [],
                "before": 0, "after": 0, "backend": "none (no detections)"}

    boxes = np.asarray(tile_boxes, np.float32).reshape(-1, 4).copy()
    off = np.asarray(offsets, np.float32).reshape(-1, 2)
    boxes += np.concatenate([off, off], axis=1)                 # tile -> scene pixels
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, shape[1])       # a box may poke past the edge
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, shape[0])
    scores = np.asarray(tile_scores, np.float32).ravel()
    classes = np.asarray(tile_classes, int).ravel()

    if prefer_sahi and sahi_available():
        mb, ms, mc = _sahi_restitch(boxes, scores, classes, names, threshold, metric,
                                    algorithm, shape)
        backend = f"sahi.{algorithm.upper()} ({metric.upper()})"
        groups: list[list[int]] = []
    elif algorithm.upper() == "NMS":
        keep = class_aware_nms(boxes, scores, classes, threshold, metric)
        mb, ms, mc = boxes[keep], scores[keep], classes[keep]
        groups = [[int(i)] for i in keep]
        backend = f"builtin NMS ({metric.upper()})"
    else:
        mb, ms, mc, groups = greedy_merge(boxes, scores, classes, threshold, metric)
        backend = f"builtin greedy merge ({metric.upper()})"

    order = np.argsort(-ms) if len(ms) else np.array([], int)
    return {
        "boxes": mb[order], "scores": ms[order], "classes": mc[order],
        "groups": [groups[i] for i in order] if groups else [],
        "before": int(n_in), "after": int(len(mb)), "backend": backend,
    }
