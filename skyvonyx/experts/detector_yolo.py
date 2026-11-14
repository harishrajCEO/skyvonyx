"""YOLOv8 detector expert (ultralytics).

LICENCE WARNING: ultralytics is AGPL-3.0. Fine for internal pilots and for
delivering labels; selling a hosted service built on it needs an Ultralytics
enterprise licence or a swap to a permissive detector. Weights fine-tuned on
DOTA additionally carry DOTA's academic-use terms. See experts/detector_rfdetr.py
for the intended escape route.
"""
from __future__ import annotations

import numpy as np

from .base import Detection, empty_detection


class YoloDetector:
    """Batched tile inference. Built by the factory below, evicted by the router."""

    def __init__(self, weights: str, conf: float, imgsz: int, device=None, half: bool = True):
        from ultralytics import YOLO          # AGPL-3.0
        import torch

        self.torch = torch
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        # FP16 only helps on CUDA; on CPU ultralytics raises or silently upcasts.
        self.half = bool(half) and self.device != "cpu"
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(weights)
        self.names = dict(self.model.names)

    def __call__(self, tiles: list[np.ndarray]) -> list[Detection]:
        if not tiles:
            return []
        # ultralytics takes BGR numpy; our tiles are RGB.
        results = self.model.predict(
            [t[..., ::-1] for t in tiles],
            conf=self.conf, imgsz=self.imgsz, device=self.device,
            half=self.half, verbose=False,
        )
        out: list[Detection] = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(empty_detection())
                continue
            out.append((
                b.xyxy.detach().cpu().numpy().astype(np.float32),
                b.conf.detach().cpu().numpy().astype(np.float32),
                b.cls.detach().cpu().numpy().astype(int),
            ))
        return out

    def unload(self) -> None:
        # ultralytics keeps the last Results (with CUDA tensors) on the predictor,
        # so dropping only self.model would leave that alive through the predictor.
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            for attr in ("results", "batch", "plotted_img"):
                if hasattr(predictor, attr):
                    setattr(predictor, attr, None)
        self.model = None


def build(cfg, device=None) -> YoloDetector:
    return YoloDetector(cfg.detector_weights, cfg.conf, cfg.tile, device=device, half=cfg.half)
