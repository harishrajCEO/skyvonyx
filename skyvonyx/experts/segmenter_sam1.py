"""SAM 1 segmenter expert (facebookresearch/segment-anything), Apache-2.0.

Kept as the fallback for when SAM 2 will not install - sam2 needs a recent
torch and a working CUDA extension build, and Colab's base image has broken
that before. Same interface, lighter checkpoints (vit_b is ~375 MB), noticeably
weaker masks on small objects than SAM 2.

Prefer SAM 2. Use this to keep a job moving, and record which one ran: the
summary carries `segmenter` so two deliveries are never silently different.
"""
from __future__ import annotations

import numpy as np

from .base import normalise_masks

_MODEL_FOR = {"sam_vit_h": "vit_h", "sam_vit_l": "vit_l", "sam_vit_b": "vit_b"}


def _guess_variant(ckpt: str) -> str:
    stem = ckpt.replace("\\", "/").split("/")[-1]
    for prefix, variant in _MODEL_FOR.items():
        if stem.startswith(prefix):
            return variant
    return "vit_b"


class Sam1Segmenter:
    def __init__(self, checkpoint: str, variant: str = "", device=None):
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        if not checkpoint:
            raise ValueError("SAM 1 needs a local checkpoint path (segmenter_weights)")
        self.torch = torch
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.device, int):
            self.device = f"cuda:{self.device}"
        variant = variant or _guess_variant(checkpoint)
        sam = sam_model_registry[variant](checkpoint=checkpoint).to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.source = f"{checkpoint} ({variant})"

    def masks_for_boxes(self, image: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        hw = image.shape[:2]
        with self.torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(image))
            bt = self.torch.as_tensor(boxes, device=self.predictor.device)
            # SamPredictor.predict() takes one box only; predict_torch batches.
            transformed = self.predictor.transform.apply_boxes_torch(bt, hw)
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed,
                multimask_output=False)
            out = normalise_masks(masks.detach().cpu().numpy(), len(boxes), hw)
        self.predictor.reset_image()
        return out

    def unload(self) -> None:
        try:
            self.predictor.reset_image()
        except Exception:
            pass
        self.predictor = None


def build(cfg, device=None) -> Sam1Segmenter:
    return Sam1Segmenter(cfg.segmenter_weights, device=device)
