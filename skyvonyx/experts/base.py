"""The expert contract.

Every model in the pipeline is an "expert" with the same shape: it is built
lazily by a factory, it exposes one call, and it can drop every reference it
holds on `unload()` so the residency manager can actually reclaim the VRAM.

Two kinds:
  Detector  (tiles)        -> per-tile (boxes xyxy, scores, class ids)
  Segmenter (image, boxes) -> one boolean mask per box, in image coordinates

Keeping `unload()` honest is the whole trick. A model that leaves a tensor on a
cached attribute (an encoder output, a last-prediction buffer) will not free,
and `torch.cuda.empty_cache()` afterwards reports success while VRAM stays
occupied. Each expert below clears those explicitly.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

Tile = np.ndarray            # (h, w, 3) uint8 RGB
Detection = tuple[np.ndarray, np.ndarray, np.ndarray]   # xyxy (n,4), scores (n,), classes (n,)


@runtime_checkable
class Detector(Protocol):
    names: dict

    def __call__(self, tiles: list[Tile]) -> list[Detection]: ...
    def unload(self) -> None: ...


@runtime_checkable
class Segmenter(Protocol):
    def masks_for_boxes(self, image: Tile, boxes: np.ndarray) -> list[np.ndarray]: ...
    def unload(self) -> None: ...


def empty_detection() -> Detection:
    return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)


def normalise_masks(raw, n_boxes: int, hw: tuple[int, int]) -> list[np.ndarray]:
    """SAM-family predictors return (C,H,W) for one box and (N,C,H,W) for many.

    Collapse both to a list of N boolean (H, W) masks. Getting this wrong is a
    silent failure: you keep only the first box's mask and never notice, because
    the output still looks like a valid polygon set.
    """
    arr = np.asarray(raw)
    if arr.ndim == 4:            # (N, C, H, W) -> best/only mask per box
        arr = arr[:, 0]
    elif arr.ndim == 3:
        arr = arr[:1] if n_boxes == 1 else arr
    elif arr.ndim == 2:          # (H, W)
        arr = arr[None]
    out = [np.asarray(m, bool) for m in arr[:n_boxes]]
    return [m for m in out if m.shape == hw] or []
