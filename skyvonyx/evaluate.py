"""Measure accuracy against a hand-checked gold set. The only source of numbers.

Every report the pipeline writes says "accuracy NOT MEASURED". This module is
how that line is legitimately replaced: run the pipeline on scenes it never
trained on, have a person label the same scenes (CVAT exports COCO), and score
the machine output against the person.

Scored at the *scene* level, after slicing and re-stitching - so a duplicate
the merge failed to remove counts as a false positive, as it would for the
customer. Tile-level mAP from `model.val()` is useful for training but is not
the number a customer receives.

Reported per class: precision, recall, F1 at a chosen confidence, mean IoU of
the matched pairs, and AP50 (VOC all-point) over every confidence. The same
function scores two reviewers against each other, which gives the
reviewer-agreement number the pilot terms require.

    python -m skyvonyx.evaluate --pred jobs/review/<job>/output \\
        --gold gold_coco.json --iou 0.5 --conf 0.25 --out accuracy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stitching import iou_one_to_many


# -------------------------------------------------------------- loading
_IMG_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _stem(file_name: str) -> str:
    """Strip only an image extension: our COCO export writes the bare stem, a
    CVAT export writes 'scene.v2.tif', and both must key as 'scene.v2'."""
    p = Path(file_name)
    return p.stem if p.suffix.lower() in _IMG_SUFFIXES else p.name


def load_coco_boxes(path: str | Path, min_score: float = 0.0) -> dict[str, list[tuple]]:
    """COCO -> {image stem: [(class_name, x1, y1, x2, y2, score), ...]}.

    Images are matched by file stem and classes by *name*, so a gold set
    exported from CVAT with its own category ids still lines up with ours.
    """
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    stems = {im["id"]: _stem(im["file_name"]) for im in coco.get("images", [])}
    out: dict[str, list[tuple]] = {s: [] for s in stems.values()}
    for a in coco.get("annotations", []):
        score = float(a.get("score", 1.0))
        if score < min_score:
            continue
        x, y, w, h = a["bbox"]
        out.setdefault(stems[a["image_id"]], []).append(
            (cats.get(a["category_id"], str(a["category_id"])), x, y, x + w, y + h, score))
    return out


def load_predictions(path: str | Path) -> dict[str, list[tuple]]:
    """A job output folder (all *_coco.json inside) or one COCO file."""
    path = Path(path)
    files = sorted(path.glob("*_coco.json")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no *_coco.json under {path}")
    merged: dict[str, list[tuple]] = {}
    for f in files:
        for stem, boxes in load_coco_boxes(f).items():
            merged.setdefault(stem, []).extend(boxes)
    return merged


# -------------------------------------------------------------- matching
def match_image(pred: list[tuple], gold: list[tuple], iou_thr: float) -> list[tuple]:
    """Greedy per-class matching, highest score first.

    Returns one row per prediction (class, score, is_tp, iou) plus one row per
    unmatched gold box (class, None, False, 0.0) so recall can be counted.
    """
    rows = []
    for cls in sorted({p[0] for p in pred} | {g[0] for g in gold}):
        p = sorted([x for x in pred if x[0] == cls], key=lambda x: -x[5])
        g = np.array([x[1:5] for x in gold if x[0] == cls], np.float32).reshape(-1, 4)
        used = np.zeros(len(g), bool)
        for x in p:
            box = np.array(x[1:5], np.float32)
            if len(g):
                ious = iou_one_to_many(box, g)
                ious[used] = -1
                j = int(np.argmax(ious))
                if ious[j] >= iou_thr:
                    used[j] = True
                    rows.append((cls, x[5], True, float(ious[j])))
                    continue
            rows.append((cls, x[5], False, 0.0))
        rows += [(cls, None, False, 0.0)] * int((~used).sum())
    return rows


def average_precision(scores: np.ndarray, tps: np.ndarray, n_gold: int) -> float:
    """VOC all-point interpolated AP. 0 when there is no gold to find."""
    if n_gold == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(tps[order])
    fp = np.cumsum(~tps[order])
    recall = tp / n_gold
    precision = tp / np.maximum(tp + fp, 1e-9)
    r = np.concatenate([[0], recall, [1]])
    p = np.concatenate([[0], precision, [0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


# ------------------------------------------------------------ evaluation
def evaluate(pred: dict[str, list[tuple]], gold: dict[str, list[tuple]],
             iou_thr: float = 0.5, conf: float = 0.25) -> dict:
    missing = sorted(set(gold) - set(pred))
    ap_key = f"ap{int(round(iou_thr * 100))}"
    rows = []
    for stem in sorted(gold):
        rows += match_image(pred.get(stem, []), gold[stem], iou_thr)

    per_class = {}
    for cls in sorted({r[0] for r in rows}):
        r = [x for x in rows if x[0] == cls]
        n_gold = sum(1 for x in r if x[1] is None) + sum(1 for x in r if x[2])
        scored = [x for x in r if x[1] is not None]
        scores = np.array([x[1] for x in scored], np.float32)
        tps = np.array([x[2] for x in scored], bool)
        at = [x for x in scored if x[1] >= conf]
        tp = sum(1 for x in at if x[2])
        fp = len(at) - tp
        fn = n_gold - tp
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / n_gold if n_gold else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        ious = [x[3] for x in at if x[2]]
        per_class[cls] = {
            "gold": n_gold, "predicted": len(at), "tp": tp, "fp": fp, "fn": fn,
            "precision": _r(prec), "recall": _r(rec), "f1": _r(f1),
            "mean_iou_of_matches": _r(float(np.mean(ious)) if ious else float("nan")),
            ap_key: _r(average_precision(scores, tps, n_gold)),
        }

    tot = {k: sum(c[k] for c in per_class.values()) for k in ("gold", "tp", "fp", "fn")}
    micro_p = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else float("nan")
    micro_r = tot["tp"] / tot["gold"] if tot["gold"] else float("nan")
    aps = [c[ap_key] for c in per_class.values() if c[ap_key] is not None]
    return {
        "iou_threshold": iou_thr, "confidence_threshold": conf, "ap_metric": ap_key,
        "images_scored": len(gold), "images_without_predictions": missing,
        "overall": {**tot, "precision": _r(micro_p), "recall": _r(micro_r),
                    f"m{ap_key}": _r(float(np.mean(aps)) if aps else float("nan"))},
        "per_class": per_class,
    }


def _r(v: float) -> float | None:
    return None if v != v else round(float(v), 4)


def write_accuracy_report(result: dict, path: Path, test_set_note: str) -> Path:
    o = result["overall"]
    lines = [
        "# Skyvonyx accuracy report",
        "",
        f"Test set: {test_set_note}",
        f"Matching: IoU >= {result['iou_threshold']}, confidence >= {result['confidence_threshold']}, "
        f"{result['images_scored']} scenes, scored after slicing and re-stitching.",
        "",
        "| class | gold | TP | FP | FN | precision | recall | F1 | mean IoU | AP |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cls, c in result["per_class"].items():
        ap = c[result["ap_metric"]]
        lines.append(f"| {cls} | {c['gold']} | {c['tp']} | {c['fp']} | {c['fn']} | "
                     f"{c['precision']} | {c['recall']} | {c['f1']} | {c['mean_iou_of_matches']} | {ap} |")
    lines += ["", f"**Overall:** precision {o['precision']}, recall {o['recall']} "
                  f"({o['tp']} of {o['gold']} objects found, {o['fp']} false alarms).", "",
              "Valid only for imagery like the test set (sensor, resolution, region, season). "
              "Quote these numbers with that description attached, never on their own."]
    if result["images_without_predictions"]:
        lines += ["", "Scenes with no predictions at all: " +
                  ", ".join(result["images_without_predictions"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score pipeline output against a gold COCO file")
    ap.add_argument("--pred", required=True, help="job output dir or a COCO predictions file")
    ap.add_argument("--gold", required=True, help="COCO file of human-verified labels")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="accuracy.json")
    ap.add_argument("--note", default="UNDESCRIBED - say sensor, GSD, region, dates, who labelled")
    a = ap.parse_args(argv)
    result = evaluate(load_predictions(a.pred), load_coco_boxes(a.gold), a.iou, a.conf)
    out = Path(a.out)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_accuracy_report(result, out.with_suffix(".md"), a.note)
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
