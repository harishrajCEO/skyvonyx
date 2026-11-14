"""Stage 1 - SAHI slice plan.

SAHI's `get_slice_bboxes` decides the slice geometry so our tiles match what
SAHI's own postprocess expects to re-stitch. We do not use `sahi.slice_image`,
because that materialises every slice as a file from an already-loaded image -
unusable for a multi-GB GeoTIFF. Instead we take SAHI's plan and read each
window with rasterio when its turn comes.

This stage touches no GPU at all, which is the point: while the plan is being
computed, no detector or segmenter is resident.
"""
from __future__ import annotations

from .raster import tile_windows


def sahi_available() -> bool:
    try:
        import sahi.slicing  # noqa: F401
        return True
    except ImportError:
        return False


def slice_plan(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return (col_off, row_off, w, h) windows for a scene.

    Uses SAHI when installed so the geometry is identical to SAHI's reference
    implementation; falls back to the equivalent local tiler otherwise (same
    step size, same edge snapping) so the pipeline still runs without SAHI.
    """
    if not sahi_available():
        return tile_windows(width, height, tile, overlap)

    from sahi.slicing import get_slice_bboxes
    boxes = get_slice_bboxes(
        image_height=height,
        image_width=width,
        slice_height=tile,
        slice_width=tile,
        auto_slice_resolution=False,   # we pin the tile size to the training size
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )
    wins = []
    for x1, y1, x2, y2 in boxes:
        # SAHI can emit slices that run past the edge on small images; clamp so
        # rasterio is never asked for a window outside the dataset.
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(int(x2), width), min(int(y2), height)
        if x2 > x1 and y2 > y1:
            wins.append((x1, y1, x2 - x1, y2 - y1))
    return wins or tile_windows(width, height, tile, overlap)


def plan_summary(wins: list[tuple[int, int, int, int]], width: int, height: int,
                 tile: int, overlap: float) -> dict:
    return {
        "width": width, "height": height, "megapixels": round(width * height / 1e6, 2),
        "tile": tile, "overlap": overlap,
        "tiles": len(wins),
        "slicer": "sahi.get_slice_bboxes" if sahi_available() else "builtin tile_windows",
    }
