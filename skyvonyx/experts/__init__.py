"""Expert registry: job.json picks experts by name, nothing imports a model directly.

Imports are deferred into the factories on purpose. Importing ultralytics or
sam2 at module import time would pull torch into every process that touches the
package - including the FastAPI front end, which only moves files around, and
the test suite, which uses fakes.
"""
from __future__ import annotations

from typing import Callable

from .base import Detection, Detector, Segmenter, Tile, empty_detection, normalise_masks

__all__ = ["Detection", "Detector", "Segmenter", "Tile", "empty_detection",
           "normalise_masks", "build_detector", "build_segmenter",
           "DETECTORS", "SEGMENTERS", "licence_notes"]


def _yolo(cfg, device=None):
    from .detector_yolo import build
    return build(cfg, device)


def _rfdetr(cfg, device=None):
    from .detector_rfdetr import build
    return build(cfg, device)


def _sam2(cfg, device=None):
    from .segmenter_sam2 import build
    return build(cfg, device)


def _sam1(cfg, device=None):
    from .segmenter_sam1 import build
    return build(cfg, device)


def _fake_det(cfg, device=None):
    from .fake import build_detector
    return build_detector(cfg, device)


def _fake_seg(cfg, device=None):
    from .fake import build_segmenter
    return build_segmenter(cfg, device)


DETECTORS: dict[str, Callable] = {"yolov8": _yolo, "rfdetr": _rfdetr, "fake": _fake_det}
SEGMENTERS: dict[str, Callable] = {"sam2": _sam2, "sam1": _sam1, "fake": _fake_seg}

# Surfaced in every run summary. Not legal advice - the point is that a
# delivery can never be made without the licence terms sitting next to it.
LICENCES = {
    "yolov8": "ultralytics YOLOv8: AGPL-3.0 (enterprise licence needed for a closed service)",
    "rfdetr": "RF-DETR: base sizes reported Apache-2.0, 'Plus' sizes PML 1.0 - verify per size",
    "fake": "built-in test double, no model",
    "sam2": "SAM 2: Apache-2.0 (verify the checkpoint)",
    "sam1": "SAM 1: Apache-2.0",
    "none": "segmentation disabled",
}


def build_detector(cfg, device=None):
    if cfg.detector not in DETECTORS:
        raise KeyError(f"unknown detector {cfg.detector!r}; have {sorted(DETECTORS)}")
    return DETECTORS[cfg.detector](cfg, device)


def build_segmenter(cfg, device=None):
    if cfg.segmenter in {"none", ""}:
        return None
    if cfg.segmenter not in SEGMENTERS:
        raise KeyError(f"unknown segmenter {cfg.segmenter!r}; have {sorted(SEGMENTERS)}")
    return SEGMENTERS[cfg.segmenter](cfg, device)


def licence_notes(cfg) -> dict:
    return {"detector": LICENCES.get(cfg.detector, "unknown - check before delivery"),
            "segmenter": LICENCES.get(cfg.segmenter, "unknown - check before delivery"),
            "data": "weights fine-tuned on DOTA carry DOTA academic-use terms",
            "sahi": "SAHI: MIT"}
