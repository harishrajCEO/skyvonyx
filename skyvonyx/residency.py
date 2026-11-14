"""Expert residency: only one model occupies the GPU at a time.

This is the "work like a sparse MoE" part of the design. YOLOv8 and SAM 2 are
never both resident: the router activates one expert, runs its whole stage over
the entire scene, then evicts it before the next expert loads. On a free Colab
T4 that is the difference between finishing and an OOM - SAM 2 hiera-small plus
a YOLOv8 batch of 8x1024px tiles will not comfortably share 15 GB.

The cost of this choice is honest: evicting means re-loading weights if a stage
runs twice (about 1-3 s for these model sizes, measured in the ledger as
`load_s`). Sequential staging trades that for a far lower peak.

Nothing here is CUDA-specific at the API level - on CPU the manager still
load/unloads, it just reports zeros for VRAM.
"""
from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from typing import Callable

Factory = Callable[[], object]


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def vram_mb() -> dict:
    """Current CUDA memory in MB. All zeros when there is no GPU."""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
        "total_mb": round(total / 1e6, 1),
        "free_mb": round(free / 1e6, 1),
    }


def peak_vram_mb() -> float:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1e6, 1)


def reset_peak() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def release() -> None:
    """Hand freed blocks back to the driver.

    gc.collect() first: a dead model object still referenced by a cycle keeps
    its CUDA storages alive, and empty_cache() then frees nothing.
    """
    gc.collect()
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class Slot:
    """One lazily-loaded expert and its load/unload bookkeeping."""

    def __init__(self, name: str, factory: Factory, on_gpu: bool = True):
        self.name = name
        self.factory = factory
        self.on_gpu = on_gpu
        self.obj: object | None = None
        self.loads = 0
        self.load_s = 0.0
        self.last_used = 0.0

    @property
    def resident(self) -> bool:
        return self.obj is not None

    def load(self) -> object:
        if self.obj is None:
            t0 = time.time()
            self.obj = self.factory()
            self.load_s += time.time() - t0
            self.loads += 1
        self.last_used = time.time()
        return self.obj

    def unload(self) -> None:
        if self.obj is None:
            return
        closer = getattr(self.obj, "unload", None)
        if callable(closer):
            closer()          # let the expert drop its own references first
        self.obj = None
        release()


class ResidencyManager:
    """Keeps at most `max_resident` GPU-bound experts loaded, evicting the rest.

        router = ResidencyManager(max_resident=1)
        router.register("detector", build_yolo)
        router.register("segmenter", build_sam2)

        with router.active("detector") as det:   # segmenter is evicted here
            ...
        with router.active("segmenter") as seg:  # detector is evicted here
            ...
    """

    def __init__(self, max_resident: int = 1, verbose: bool = True):
        self.max_resident = max(1, int(max_resident))
        self.verbose = verbose
        self.slots: dict[str, Slot] = {}
        self.events: list[dict] = []

    def register(self, name: str, factory: Factory, on_gpu: bool = True) -> Slot:
        slot = Slot(name, factory, on_gpu)
        self.slots[name] = slot
        return slot

    def registered(self, name: str) -> bool:
        return name in self.slots

    # ----------------------------------------------------------------- eviction
    def _make_room(self, keep: str) -> list[str]:
        """Evict GPU experts until `keep` can be the (max_resident)-th resident."""
        gpu_resident = [s for n, s in self.slots.items() if s.on_gpu and s.resident and n != keep]
        evicted = []
        # oldest first - with max_resident=1 this evicts everything else
        for slot in sorted(gpu_resident, key=lambda s: s.last_used):
            if len(gpu_resident) - len(evicted) < self.max_resident:
                break
            slot.unload()
            evicted.append(slot.name)
        return evicted

    def evict_all(self) -> None:
        for slot in self.slots.values():
            slot.unload()
        release()

    # ------------------------------------------------------------------ activate
    @contextmanager
    def active(self, name: str):
        """Make one expert resident for the duration of a stage.

        The expert stays loaded on exit - the next `active()` call evicts it.
        That way two consecutive stages on the same expert do not pay for a
        reload, while a different expert still never shares the GPU with it.
        """
        if name not in self.slots:
            raise KeyError(f"no expert registered as {name!r}; have {sorted(self.slots)}")
        slot = self.slots[name]
        evicted = self._make_room(name)
        before = vram_mb()
        reset_peak()
        was_resident = slot.resident
        t0 = time.time()
        obj = slot.load()
        load_s = round(time.time() - t0, 2)
        after_load = vram_mb()
        if self.verbose:
            note = "already resident" if was_resident else f"loaded in {load_s}s"
            ev = f" (evicted {', '.join(evicted)})" if evicted else ""
            print(f"[router] {name}: {note}{ev} | "
                  f"VRAM {before['allocated_mb']} -> {after_load['allocated_mb']} MB", flush=True)
        try:
            yield obj
        finally:
            ev_rec = {
                "expert": name, "evicted": evicted, "reloaded": not was_resident,
                "load_s": load_s, "vram_before_mb": before["allocated_mb"],
                "vram_after_load_mb": after_load["allocated_mb"],
                "vram_peak_mb": peak_vram_mb(), "vram_total_mb": before["total_mb"],
            }
            self.events.append(ev_rec)
            slot.last_used = time.time()

    # -------------------------------------------------------------------- report
    def last_event(self, name: str) -> dict:
        for ev in reversed(self.events):
            if ev["expert"] == name:
                return ev
        return {}

    def report(self) -> dict:
        return {
            "max_resident_experts": self.max_resident,
            "device": _device_name(),
            "experts": {n: {"loads": s.loads, "load_s_total": round(s.load_s, 2),
                            "resident_now": s.resident, "on_gpu": s.on_gpu}
                        for n, s in self.slots.items()},
            "peak_vram_mb": max([e["vram_peak_mb"] for e in self.events], default=0.0),
            "vram_now": vram_mb(),
            "events": self.events,
        }


def _device_name() -> str:
    torch = _torch()
    if torch is None:
        return "no-torch"
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def pick_device(prefer: str | int | None = None) -> str | int:
    torch = _torch()
    if prefer is not None:
        return prefer
    if torch is not None and torch.cuda.is_available():
        return 0
    return "cpu"
