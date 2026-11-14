"""Windowed raster I/O and 8-bit rendering.

A GeoTIFF scene can be several GB. Nothing here ever loads a whole one: every
read is a window, and the contrast stretch is computed from a decimated
overview, not from the full array.

GeoTIFF goes through rasterio (keeps CRS + affine transform). PNG/JPG falls
back to OpenCV and is explicitly NOT georeferenced - the pipeline labels those
outputs as pixel coordinates so a reviewer cannot mistake them for map data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def tile_windows(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """(col_off, row_off, w, h) windows covering the image; last row/col snap to the edge.

    Edge snapping matters: a short final tile would otherwise be padded, and a
    detector trained at `tile` px sees a different object scale in padding.
    """
    step = max(1, int(round(tile * (1 - overlap))))

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, step))
        if s[-1] != n - tile:
            s.append(n - tile)
        return s

    return [(c, r, min(tile, width - c), min(tile, height - r))
            for r in starts(height) for c in starts(width)]


def stretch_limits(sample: np.ndarray, sensor: str, nodata=None,
                   p: tuple[float, float] = (2.0, 98.0)) -> tuple[np.ndarray, np.ndarray]:
    """Per-band percentile limits from a decimated sample.

    Computed once per scene and reused for every tile - if each tile stretched
    itself, identical objects would render differently per tile and the
    detector's scores would wobble across tile boundaries.
    """
    s = np.asarray(sample, dtype=np.float32)
    if sensor == "sar":
        s = 10 * np.log10(np.clip(s, 1e-6, None))  # linear backscatter -> dB
    lo, hi = [], []
    for band in s:
        v = band[np.isfinite(band)]
        if nodata is not None:
            v = v[v != nodata]
        if v.size == 0:
            lo.append(0.0)
            hi.append(1.0)
            continue
        lo.append(float(np.percentile(v, p[0])))
        hi.append(float(np.percentile(v, p[1])))
    return np.array(lo, np.float32), np.array(hi, np.float32)


def to_uint8_rgb(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray, sensor: str) -> np.ndarray:
    """(bands, h, w) -> (h, w, 3) uint8 RGB using precomputed per-band limits."""
    a = np.asarray(arr, dtype=np.float32)
    if sensor == "sar":
        a = 10 * np.log10(np.clip(a, 1e-6, None))
    a = (a - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None]
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if a.shape[0] == 1:
        a = np.repeat(a, 3, axis=0)      # single-band SAR -> grey RGB
    return np.ascontiguousarray(np.transpose(a[:3], (1, 2, 0)))


def is_empty_tile(rgb: np.ndarray, min_std: float = 2.0) -> bool:
    """Flat tile = nodata padding, open water, or blank desert. Skipping these is
    the cheapest speed-up in the pipeline (section 8 of the master context)."""
    return float(rgb.std()) < min_std


class RasterSource:
    """Windowed reader with georeferencing when the format carries it."""

    def __init__(self, path: str | Path, bands: list[int] | None = None):
        self.path = Path(path)
        self._ds = None
        self._arr: np.ndarray | None = None
        self.crs = None
        self.affine: tuple | None = None
        self.nodata = None

        if self.path.suffix.lower() in {".tif", ".tiff"}:
            import rasterio
            self._ds = rasterio.open(self.path)
            self.width, self.height = self._ds.width, self._ds.height
            count = self._ds.count
            self.bands = [b for b in (bands or [1, 2, 3]) if 1 <= b <= count] or [1]
            self.nodata = self._ds.nodata
            t = self._ds.transform
            if self._ds.crs is not None and not t.is_identity:
                self.crs = self._ds.crs
                self.affine = (t.a, t.b, t.c, t.d, t.e, t.f)
        else:
            import cv2
            img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cannot read image: {self.path}")
            self._arr = np.transpose(img[..., ::-1], (2, 0, 1))  # BGR HWC -> RGB (3,h,w)
            self.height, self.width = img.shape[:2]
            self.bands = [1, 2, 3]

    @property
    def georeferenced(self) -> bool:
        return self.affine is not None

    @property
    def crs_string(self) -> str | None:
        return self.crs.to_string() if self.crs is not None else None

    def read(self, col: int, row: int, w: int, h: int) -> np.ndarray:
        """One window as (bands, h, w). The only place pixels enter memory."""
        if self._ds is not None:
            from rasterio.windows import Window
            return self._ds.read(self.bands, window=Window(col, row, w, h))
        return self._arr[:, row:row + h, col:col + w]

    def overview_sample(self, max_side: int = 1024) -> np.ndarray:
        """Decimated whole-scene sample for stretch stats and the preview PNG."""
        if self._ds is not None:
            from rasterio.enums import Resampling
            scale = max(1.0, max(self.width, self.height) / max_side)
            ow, oh = max(1, int(self.width / scale)), max(1, int(self.height / scale))
            # nearest, not average: averaging shrinks the real pixel spread and
            # the 2/98 percentiles come out too tight.
            return self._ds.read(self.bands, out_shape=(len(self.bands), oh, ow),
                                 resampling=Resampling.nearest)
        step = max(1, max(self.width, self.height) // max_side)
        return self._arr[:, ::step, ::step]

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __enter__(self) -> "RasterSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
