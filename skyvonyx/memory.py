"""Disk-backed run memory: stage hand-off, resume, and the run ledger.

Two jobs, both of which exist because free Colab has little VRAM and disconnects:

1. STAGE HAND-OFF. Stages never pass big arrays to each other in RAM. `detect`
   writes its boxes to memory and returns; `stitch` reads them back. So the peak
   process memory is one stage's working set, not the whole pipeline's.

2. RESUME. Every stage result is keyed by (scene content hash + stage
   fingerprint). If the session dies after `detect`, the next run reads detect
   from memory and starts at `stitch`. Change a threshold and only the stages
   that depend on it recompute - see JobConfig.stage_fingerprint.

The ledger is the by-product that answers "how long does one scene take, per
stage?" - it records wall time, VRAM before/after, and the counts each stage
produced, for every attempt including failures.

Layout:
  <root>/memory/<job_id>/<scene_key>/<stage>.<ext>   stage payloads
  <root>/memory/<job_id>/ledger.jsonl               append-only stage records
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_key(path: str | Path, head_bytes: int = 1 << 20) -> str:
    """Identify a scene by name + size + first/last MB.

    Full-file hashing of a multi-GB GeoTIFF on Drive costs more than the stage
    it protects, so we hash the ends and the size. Enough to notice "same job
    folder, different image".
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(f"{p.name}:{size}".encode())
    with open(p, "rb") as f:
        h.update(f.read(head_bytes))
        if size > head_bytes * 2:
            f.seek(-head_bytes, 2)
            h.update(f.read(head_bytes))
    return h.hexdigest()[:16]


@dataclass
class StageRecord:
    stage: str
    status: str                 # "cached" | "ok" | "failed"
    seconds: float
    scene: str
    detail: dict

    def row(self) -> dict:
        return {"ts": now_iso(), "stage": self.stage, "status": self.status,
                "seconds": round(self.seconds, 3), "scene": self.scene, **self.detail}


class RunMemory:
    """Content-addressed store for stage payloads plus an append-only ledger."""

    def __init__(self, root: str | Path, job_id: str, enabled: bool = True):
        self.base = Path(root) / "memory" / job_id
        self.base.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.job_id = job_id
        self.records: list[StageRecord] = []
        self._scene = "-"

    # ------------------------------------------------------------------ scene
    @contextmanager
    def scene(self, scene_key: str):
        prev, self._scene = self._scene, scene_key
        (self.base / scene_key).mkdir(parents=True, exist_ok=True)
        try:
            yield self
        finally:
            self._scene = prev

    def _path(self, stage: str, fingerprint: str, ext: str) -> Path:
        return self.base / self._scene / f"{stage}.{fingerprint}.{ext}"

    # ------------------------------------------------------------------- json
    def get_json(self, stage: str, fingerprint: str) -> Any | None:
        p = self._path(stage, fingerprint, "json")
        if not (self.enabled and p.exists()):
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            p.unlink(missing_ok=True)  # a half-written file from a killed session
            return None

    def put_json(self, stage: str, fingerprint: str, obj: Any) -> Path:
        p = self._path(stage, fingerprint, "json")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, default=_json_default), encoding="utf-8")
        tmp.replace(p)  # atomic: a reader never sees a partial payload
        return p

    # ------------------------------------------------------------------- npz
    def get_arrays(self, stage: str, fingerprint: str) -> dict[str, np.ndarray] | None:
        p = self._path(stage, fingerprint, "npz")
        if not (self.enabled and p.exists()):
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        except (OSError, ValueError, EOFError):
            p.unlink(missing_ok=True)
            return None

    def put_arrays(self, stage: str, fingerprint: str, arrays: dict[str, np.ndarray]) -> Path:
        p = self._path(stage, fingerprint, "npz")
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(p)
        return p

    # ----------------------------------------------------------------- ledger
    def record(self, stage: str, status: str, seconds: float, **detail) -> StageRecord:
        rec = StageRecord(stage, status, seconds, self._scene, detail)
        self.records.append(rec)
        with open(self.base / "ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.row()) + "\n")
        return rec

    @contextmanager
    def timed(self, stage: str):
        """Time a stage and always write a ledger row, success or failure.

        Usage:
            with mem.timed("detect") as d:
                ...
                d["tiles"] = 42          # merged into the ledger row
        """
        detail: dict = {}
        t0 = time.time()
        try:
            yield detail
        except Exception as e:
            self.record(stage, "failed", time.time() - t0,
                        error=f"{type(e).__name__}: {e}"[:300], **detail)
            raise
        self.record(stage, detail.pop("_status", "ok"), time.time() - t0, **detail)

    def timings(self) -> dict:
        """Seconds per stage, summed over scenes - the number section 18 asks for."""
        out: dict[str, float] = {}
        for r in self.records:
            out[r.stage] = round(out.get(r.stage, 0.0) + r.seconds, 2)
        out["total_s"] = round(sum(out.values()), 2)
        return out

    def stage_table(self) -> list[dict]:
        return [r.row() for r in self.records]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
