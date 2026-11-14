"""SAM 2 segmenter expert - turns detector boxes into polygon masks.

Runs as its own stage, after the detector has been evicted. Two reasons that
ordering is not negotiable on a free T4:
  - SAM 2's image encoder is the heaviest thing in the pipeline; it wants the
    GPU to itself.
  - A box prompt needs the box, so segmentation cannot start before the
    re-stitch stage has decided which boxes are real.

Crops, not scenes. `set_image()` embeds whatever you hand it, so handing it a
20000x20000 scene would be an instant OOM. The pipeline groups merged boxes by
tile and calls this once per tile crop, batching every box in that crop into a
single `predict()` - one image embedding serves all of them.
"""
from __future__ import annotations

import numpy as np

from .base import normalise_masks

# Checkpoint filename -> hydra config shipped in the sam2 package.
_CONFIG_FOR = {
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}
HF_DEFAULT = "facebook/sam2.1-hiera-small"


def _guess_config(ckpt: str) -> str | None:
    stem = ckpt.replace("\\", "/").split("/")[-1].removesuffix(".pt")
    return _CONFIG_FOR.get(stem)


class Sam2Segmenter:
    """Box-prompted mask prediction. Apache-2.0 (verify the checkpoint you use)."""

    def __init__(self, checkpoint: str = "", config: str = "", device=None,
                 hf_model_id: str = HF_DEFAULT):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.device, int):
            self.device = f"cuda:{self.device}"

        if checkpoint:
            from sam2.build_sam import build_sam2
            cfg = config or _guess_config(checkpoint)
            if not cfg:
                raise ValueError(
                    f"cannot infer a SAM2 config for checkpoint {checkpoint!r}; "
                    f"set segmenter_config explicitly (one of {sorted(_CONFIG_FOR.values())})")
            model = build_sam2(cfg, checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(model)
            self.source = f"{checkpoint} + {cfg}"
        else:
            # No local checkpoint: pull from the Hub (needs network on first use).
            self.predictor = SAM2ImagePredictor.from_pretrained(hf_model_id, device=self.device)
            self.source = f"hf:{hf_model_id}"

        # bf16 on Ampere+, fp16 otherwise; plain float32 on CPU.
        if str(self.device).startswith("cuda"):
            major = torch.cuda.get_device_capability(0)[0]
            self._dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self._dtype = torch.float32

    def masks_for_boxes(self, image: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        """image: (h, w, 3) uint8 RGB crop. boxes: (n, 4) xyxy in crop coordinates."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        hw = image.shape[:2]
        autocast = (self.torch.autocast(device_type="cuda", dtype=self._dtype)
                    if str(self.device).startswith("cuda")
                    else _NullContext())
        with self.torch.inference_mode(), autocast:
            self.predictor.set_image(np.ascontiguousarray(image))
            masks, scores, _ = self.predictor.predict(box=boxes, multimask_output=False)
        out = normalise_masks(masks, len(boxes), hw)
        self.predictor.reset_predictor()   # drop the cached image embedding
        return out

    def unload(self) -> None:
        # The embedding of the last crop lives on the predictor; reset before
        # dropping it or the encoder activations stay on the GPU.
        try:
            self.predictor.reset_predictor()
        except Exception:
            pass
        self.predictor = None


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def build(cfg, device=None) -> Sam2Segmenter:
    return Sam2Segmenter(cfg.segmenter_weights, cfg.segmenter_config, device=device)
