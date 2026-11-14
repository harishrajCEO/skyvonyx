"""Stage 5 - georeferencing and export.

The rule from the master context: a label without a CRS is not a deliverable.
Every geometry leaves here twice - once in the scene's native CRS (what a
GIS analyst wants for measurement) and once in WGS84 lon/lat, because RFC 7946
GeoJSON is defined as WGS84 and a web map silently misplaces anything else.

When the input has no transform (a PNG), outputs are still written but the
filename says PIXEL_COORDS_NOT_GEOREFERENCED. Naming it in the file is the only
way a reviewer cannot mistake it for map data later.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ transforms
def pixel_to_map(affine: tuple, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rasterio.Affine convention: x = a*col + b*row + c, y = d*col + e*row + f.

    Note `col`/`row` are pixel *edges*, not centres - a detection box edge is a
    pixel boundary, so no 0.5 offset is applied.
    """
    a, b, c, d, e, f = affine[:6]
    col = np.asarray(col, np.float64)
    row = np.asarray(row, np.float64)
    return a * col + b * row + c, d * col + e * row + f


def box_ring(x1: float, y1: float, x2: float, y2: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed ring, counter-clockwise in pixel space."""
    return (np.array([x1, x2, x2, x1, x1], np.float64),
            np.array([y1, y1, y2, y2, y1], np.float64))


# -------------------------------------------------------------------- polygons
def mask_to_polygons(mask: np.ndarray, offset: tuple[float, float] = (0.0, 0.0),
                     simplify_px: float = 1.5, min_area_px: float = 4.0) -> list[list[list[float]]]:
    """Boolean mask -> list of closed pixel rings, shifted by `offset`.

    Only outer contours: a hole inside a ship is noise from a reflective deck,
    not a real void, and a GeoJSON ring list with holes confuses most label
    importers. Douglas-Peucker keeps the vertex count sane - an unsimplified
    1024px mask contour can be 3000 points, and 500 of those per scene makes a
    GeoJSON no browser will open.
    """
    import cv2

    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ox, oy = offset
    rings = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        pts = contour
        if simplify_px > 0:
            pts = cv2.approxPolyDP(contour, simplify_px, True)
        pts = pts.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            continue
        ring = [[float(x + ox), float(y + oy)] for x, y in pts]
        ring.append(ring[0])              # GeoJSON rings must close
        rings.append(ring)
    return rings


def ring_to_map(ring: list[list[float]], affine: tuple | None) -> list[list[float]]:
    if affine is None:
        return [[float(x), float(y)] for x, y in ring]
    cols = np.array([p[0] for p in ring])
    rows = np.array([p[1] for p in ring])
    xs, ys = pixel_to_map(affine, cols, rows)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


# ------------------------------------------------------------------- features
def build_features(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                   names: dict, affine: tuple | None,
                   polygons: dict[int, list[list[list[float]]]] | None = None) -> list[dict]:
    """One Feature per detection: polygon geometry when SAM 2 produced a mask,
    otherwise the box. `px_bbox` is always kept so a reviewer can go back to
    the raster, and `geometry_source` records which it is."""
    polygons = polygons or {}
    feats = []
    for i, ((x1, y1, x2, y2), score, cls) in enumerate(zip(boxes, scores, classes)):
        rings = polygons.get(i)
        if rings:
            coords = [ring_to_map(r, affine) for r in rings]
            geom = ({"type": "Polygon", "coordinates": [coords[0]]} if len(coords) == 1
                    else {"type": "MultiPolygon", "coordinates": [[c] for c in coords]})
            source = "sam_mask"
        else:
            cols, rows = box_ring(float(x1), float(y1), float(x2), float(y2))
            ring = ring_to_map([[c, r] for c, r in zip(cols, rows)], affine)
            geom = {"type": "Polygon", "coordinates": [ring]}
            source = "detector_box"
        feats.append({
            "type": "Feature",
            "id": i,
            "geometry": geom,
            "properties": {
                "class_id": int(cls),
                "class_name": str(names.get(int(cls), cls)),
                "score": round(float(score), 4),
                "geometry_source": source,
                "status": "unreviewed",     # a human flips this in QA
                "px_bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            },
        })
    return feats


def write_geojson(out_dir: Path, stem: str, features: list[dict],
                  crs_string: str | None, affine: tuple | None,
                  transform_to_wgs84=None) -> dict:
    """Write native-CRS and WGS84 GeoJSON. Returns the paths written."""
    out_dir = Path(out_dir)
    written = {}
    native = {"type": "FeatureCollection", "features": features}

    if affine is None:
        p = out_dir / f"{stem}_PIXEL_COORDS_NOT_GEOREFERENCED.geojson"
        p.write_text(json.dumps(native), encoding="utf-8")
        return {"pixel": str(p)}

    native["crs"] = {"type": "name", "properties": {"name": crs_string}}
    p = out_dir / f"{stem}_native.geojson"
    p.write_text(json.dumps(native), encoding="utf-8")
    written["native"] = str(p)

    if transform_to_wgs84 is not None:
        wgs = {"type": "FeatureCollection",
               "features": [{**f, "geometry": transform_to_wgs84(f["geometry"])}
                            for f in features]}
        p = out_dir / f"{stem}_wgs84.geojson"
        p.write_text(json.dumps(wgs), encoding="utf-8")
        written["wgs84"] = str(p)
    return written


def wgs84_transformer(crs):
    """Geometry -> WGS84 geometry, or None when rasterio is unavailable."""
    try:
        from rasterio.warp import transform_geom
    except ImportError:
        return None

    def convert(geom: dict) -> dict:
        return transform_geom(crs, "EPSG:4326", geom)

    return convert


# ----------------------------------------------------------------------- COCO
def write_coco(out_dir: Path, stem: str, width: int, height: int,
               boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
               names: dict, polygons: dict | None = None) -> str:
    """COCO detection JSON in pixel coordinates - what most training code eats."""
    polygons = polygons or {}
    cat_ids = sorted({int(c) for c in classes}) or sorted(int(k) for k in names)
    coco = {
        "info": {"description": f"Skyvonyx machine output for {stem}",
                 "note": "UNREVIEWED model output. Not a verified label set."},
        "images": [{"id": 1, "file_name": f"{stem}", "width": int(width), "height": int(height)}],
        "categories": [{"id": int(c), "name": str(names.get(int(c), c))} for c in cat_ids],
        "annotations": [],
    }
    for i, ((x1, y1, x2, y2), score, cls) in enumerate(zip(boxes, scores, classes)):
        w, h = float(x2 - x1), float(y2 - y1)
        ann = {"id": i + 1, "image_id": 1, "category_id": int(cls),
               "bbox": [round(float(x1), 1), round(float(y1), 1), round(w, 1), round(h, 1)],
               "area": round(w * h, 1), "iscrowd": 0, "score": round(float(score), 4)}
        rings = polygons.get(i)
        if rings:
            ann["segmentation"] = [[round(v, 1) for p in ring for v in p] for ring in rings]
        coco["annotations"].append(ann)
    p = Path(out_dir) / f"{stem}_coco.json"
    p.write_text(json.dumps(coco), encoding="utf-8")
    return str(p)


# -------------------------------------------------------------------- preview
def save_preview(overview_rgb: np.ndarray, scene_wh: tuple[int, int],
                 boxes: np.ndarray, path: Path,
                 polygons: dict | None = None) -> str:
    """Decimated scene with detections drawn - the first thing a reviewer opens."""
    import cv2

    img = np.ascontiguousarray(overview_rgb[..., ::-1])     # RGB -> BGR for cv2
    width, height = scene_wh
    fx, fy = img.shape[1] / max(width, 1), img.shape[0] / max(height, 1)
    for x1, y1, x2, y2 in np.asarray(boxes, np.float32).reshape(-1, 4):
        cv2.rectangle(img, (int(x1 * fx), int(y1 * fy)),
                      (int(x2 * fx) + 1, int(y2 * fy) + 1), (0, 0, 255), 1)
    for rings in (polygons or {}).values():
        for ring in rings:
            pts = np.array([[int(x * fx), int(y * fy)] for x, y in ring], np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 1)
    cv2.imwrite(str(path), img)
    return str(path)
