"""RF-DETR detector expert - the licence escape route from YOLOv8.

Same interface as the YOLO expert, so swapping is a one-word change in
job.json (`"detector": "rfdetr"`). Untrained here: RF-DETR ships COCO weights,
which know "boat" but not "oil tanker" or "flood extent", so this is for when a
fine-tune exists, or for a zero-shot baseline comparison on a shared test set.

LICENCE: check the specific RF-DETR size before commercial use. The base sizes
are reported as Apache-2.0 and the "Plus" sizes under Roboflow's PML 1.0 -
verify against vendor/rf-detr/LICENSE for the exact commit you install.
"""
from __future__ import annotations

import numpy as np

from .base import Detection, empty_detection

_SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
          "base": "RFDETRBase", "large": "RFDETRLarge"}


class RFDetrDetector:
    def __init__(self, weights: str, conf: float, size: str = "nano", device=None):
        import rfdetr
        import torch

        self.torch = torch
        self.conf = conf
        cls_name = _SIZES.get(size, "RFDETRNano")
        cls = getattr(rfdetr, cls_name, None)
        if cls is None:
            raise ImportError(f"{cls_name} not found in the installed rfdetr "
                              f"(have: {[n for n in _SIZES.values() if hasattr(rfdetr, n)]})")
        kwargs = {}
        if weights and weights not in {"", "coco"}:
            kwargs["pretrain_weights"] = weights
        if device is not None:
            kwargs["device"] = f"cuda:{device}" if isinstance(device, int) else device
        self.model = cls(**kwargs)
        classes = getattr(self.model, "class_names", None)
        self.names = (dict(enumerate(classes)) if isinstance(classes, (list, tuple))
                      else dict(classes) if isinstance(classes, dict) else {})

    def __call__(self, tiles: list[np.ndarray]) -> list[Detection]:
        out: list[Detection] = []
        for tile in tiles:
            # rfdetr.predict takes RGB and returns a supervision.Detections
            det = self.model.predict(tile, threshold=self.conf)
            xyxy = getattr(det, "xyxy", None)
            if xyxy is None or len(xyxy) == 0:
                out.append(empty_detection())
                continue
            conf = getattr(det, "confidence", None)
            cls = getattr(det, "class_id", None)
            n = len(xyxy)
            out.append((
                np.asarray(xyxy, np.float32).reshape(-1, 4),
                np.ones(n, np.float32) if conf is None else np.asarray(conf, np.float32),
                np.zeros(n, int) if cls is None else np.asarray(cls, int),
            ))
        return out

    def unload(self) -> None:
        self.model = None


def build(cfg, device=None) -> RFDetrDetector:
    size = getattr(cfg, "rfdetr_size", "nano")
    return RFDetrDetector(cfg.detector_weights, cfg.conf, size=size, device=device)
