"""Skyvonyx backend: staged satellite-image labelling.

SAHI slicing -> YOLOv8 detection -> SAHI re-stitch -> SAM 2 masks -> georeferenced export,
run one expert at a time so the whole stack fits on a free Colab GPU.

    from skyvonyx import queue_worker
    queue_worker.create_job("/content/drive/MyDrive/skyvonyx", "demo_001", ["scene.tif"])
    queue_worker.run_forever("/content/drive/MyDrive/skyvonyx")

Nothing here delivers anything to a customer: jobs stop in review/ until a
person creates an APPROVED file.
"""
from __future__ import annotations

__version__ = "0.2.0"

from .config import STAGE_ORDER, JobConfig

__all__ = ["JobConfig", "STAGE_ORDER", "__version__"]
