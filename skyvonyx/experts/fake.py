"""Deterministic fake experts: the whole pipeline, no GPU, no weights.

Everything except the model forward pass is testable this way - slicing, empty
tile skipping, cross-tile merging, georeferencing, polygon export, the review
gate, resume from memory. That is most of the code and all of the parts that
produce wrong *numbers* rather than obvious crashes.

The fake detector finds bright blobs with connected components, so tests can
plant an object at a known pixel and assert the exported coordinate. The fake
segmenter returns the blob interior, so polygon code gets real contours.
"""
from __future__ import annotations

import numpy as np

from .base import Detection, empty_detection

CLASS_NAMES = {0: "object"}


class FakeDetector:
    """Detects bright blobs (min_pixels or larger) as class 0."""

    def __init__(self, threshold: int = 250, min_pixels: int = 100):
        self.names = dict(CLASS_NAMES)
        self.threshold = threshold
        self.min_pixels = min_pixels
        self.calls = 0
        self.tiles_seen = 0

    def __call__(self, tiles: list[np.ndarray]) -> list[Detection]:
        import cv2
        self.calls += 1
        self.tiles_seen += len(tiles)
        out: list[Detection] = []
        for tile in tiles:
            mask = (tile.min(axis=2) > self.threshold).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            keep = [i for i in range(1, n) if stats[i, 4] >= self.min_pixels]
            if not keep:
                out.append(empty_detection())
                continue
            boxes = np.array([[stats[i, 0], stats[i, 1],
                               stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]]
                              for i in keep], np.float32)
            out.append((boxes, np.full(len(keep), 0.9, np.float32), np.zeros(len(keep), int)))
        return out

    def unload(self) -> None:
        self.names = dict(CLASS_NAMES)


class FakeSegmenter:
    """Returns the bright interior of each box as its mask."""

    def __init__(self, threshold: int = 250):
        self.threshold = threshold
        self.calls = 0
        self.boxes_seen = 0

    def masks_for_boxes(self, image: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        self.calls += 1
        self.boxes_seen += len(boxes)
        h, w = image.shape[:2]
        bright = image.min(axis=2) > self.threshold
        masks = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros((h, w), bool)
            xa, ya = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            xb, yb = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
            if xb > xa and yb > ya:
                m[ya:yb, xa:xb] = bright[ya:yb, xa:xb]
                if not m.any():          # box with nothing bright: fall back to the box
                    m[ya:yb, xa:xb] = True
            masks.append(m)
        return masks

    def unload(self) -> None:
        pass


def build_detector(cfg, device=None) -> FakeDetector:
    return FakeDetector()


def build_segmenter(cfg, device=None) -> FakeSegmenter:
    return FakeSegmenter()
