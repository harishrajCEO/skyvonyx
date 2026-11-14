"""Job + stage configuration for the Skyvonyx staged pipeline.

One JobConfig describes everything a scene needs: which expert runs each stage,
the slice geometry, thresholds, and how much VRAM the router is allowed to hold.
It is serialised into job.json, hashed into the memory keys, and copied verbatim
into the run summary so a result can always be traced back to its settings.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Stage order is the contract between the router, the memory store and the report.
# Each name is a directory in memory/<job>/<scene>/ and a row in the ledger.
STAGE_ORDER = ("plan", "detect", "stitch", "segment", "export")

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class JobConfig:
    job_id: str
    customer: str = "internal"
    task: str = "detect+segment"

    # ---- experts (resolved through skyvonyx.experts.registry) ----
    detector: str = "yolov8"            # "yolov8" | "rfdetr" | "fake"
    segmenter: str = "sam2"             # "sam2" | "sam1" | "none" | "fake"
    detector_weights: str = "yolov8n.pt"
    segmenter_weights: str = ""         # SAM2 checkpoint path; "" = HF auto-download
    segmenter_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # ---- slicing (SAHI) ----
    tile: int = 1024
    overlap: float = 0.2
    bands: list = field(default_factory=lambda: [1, 2, 3])
    sensor: str = "optical"             # "optical" | "sar"
    min_tile_std: float = 2.0           # below this a tile is treated as empty and skipped

    # ---- detection ----
    conf: float = 0.25
    batch: int = 8
    half: bool = True

    # ---- re-stitch (SAHI postprocess) ----
    merge_metric: str = "IOS"           # IOS handles boxes cut at a tile edge; IOU does not
    merge_threshold: float = 0.5
    merge_algorithm: str = "GREEDYNMM"  # "GREEDYNMM" | "NMS"

    # ---- segmentation ----
    segment_min_score: float = 0.35     # only mask boxes the detector is reasonably sure about
    segment_max_boxes: int = 2000       # hard cap so one bad scene cannot run for hours
    polygon_simplify_px: float = 1.5

    # ---- router / memory ----
    max_resident_experts: int = 1       # the DeepSeek-style rule: one expert in VRAM at a time
    resume: bool = True                 # reuse completed stages from memory after a disconnect
    keep_tiles: bool = False            # keep materialised tiles on disk (debugging only)

    # filled by load(): keys present in job.json that this version does not know.
    # Surfaced in the summary so a typo'd setting is visible instead of ignored.
    unknown_keys: list = field(default_factory=list)

    @property
    def overlap_px(self) -> int:
        return int(round(self.tile * self.overlap))

    def stage_fingerprint(self, stage: str) -> str:
        """Hash of only the settings a stage actually depends on.

        Changing `conf` must invalidate detect/stitch/segment/export but not the
        slice plan, otherwise resume would silently serve stale results.
        """
        deps = {
            "plan": ("tile", "overlap", "bands", "sensor", "min_tile_std"),
            "detect": ("tile", "overlap", "bands", "sensor", "min_tile_std",
                       "detector", "detector_weights", "conf", "half"),
            "stitch": ("merge_metric", "merge_threshold", "merge_algorithm"),
            "segment": ("segmenter", "segmenter_weights", "segmenter_config",
                        "segment_min_score", "segment_max_boxes", "polygon_simplify_px"),
            "export": ("polygon_simplify_px",),
        }[stage]
        payload = {k: getattr(self, k) for k in deps}
        idx = STAGE_ORDER.index(stage)
        if idx > 0:
            # chain upstream in, so a change to `tile` invalidates every later stage too
            payload["_upstream"] = self.stage_fingerprint(STAGE_ORDER[idx - 1])
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: str | Path) -> "JobConfig":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - set(known))
        if "job_id" not in known:
            known["job_id"] = path.parent.name
        # Worker-wide defaults for weights, so a job submitted from the website
        # (which knows nothing about Drive paths) still picks up the fine-tune.
        import os
        env_defaults = {"detector_weights": "SKYVONYX_DEFAULT_DETECTOR_WEIGHTS",
                        "segmenter_weights": "SKYVONYX_DEFAULT_SAM2_CKPT"}
        for key, var in env_defaults.items():
            if key not in raw and os.environ.get(var):
                known[key] = os.environ[var]
        known["unknown_keys"] = unknown
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)
