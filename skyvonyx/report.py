"""The draft report that goes into review/ - never straight to a customer.

Two rules from the master context are enforced here rather than left to
discipline:
  - The accuracy section says NOT MEASURED and cannot be filled in by the
    pipeline. Only a reviewer who has checked a gold subset can replace it.
  - The licence line is printed on the deliverable, so nobody ships
    AGPL-derived output to a commercial customer without seeing the warning.

The per-stage timing table is the answer to open question 1 ("how long does one
scene take today, per stage?"), produced automatically on every run.
"""
from __future__ import annotations

from pathlib import Path

TEMPLATE = """# DRAFT - Skyvonyx job {job_id}

**NOT FOR THE CUSTOMER UNTIL A REVIEWER CREATES `APPROVED` IN THIS FOLDER.**

Customer: {customer} | Task: {task} | Sensor: {sensor} | Generated: {generated}

## Pipeline
`{pipeline}`

| Expert | Model | Loads | VRAM peak |
|---|---|---|---|
| detector | {detector} (`{detector_weights}`) | {detector_loads} | {detector_vram} MB |
| segmenter | {segmenter} | {segmenter_loads} | {segmenter_vram} MB |

Experts resident at once: **{max_resident}** on {device}. Stages run one at a
time, so the peak GPU use is one model's, not the stack's.

## Machine output (unreviewed)
{rows}

Totals: {total_scenes} scene(s), {total_detections} detections after merge,
{total_polygons} SAM polygons, {total_wall}s wall clock.

## Time per stage (seconds)
{stage_table}

## Accuracy
**NOT MEASURED.** These are raw model detections before human QA. Do not quote
precision, recall, IoU or mAP anywhere - deck, email, invoice or website -
until a reviewer has checked a gold subset and recorded:
precision, recall, IoU, and second-reviewer agreement.

## Licences on this output
- Detector: {licence_detector}
- Segmenter: {licence_segmenter}
- Slicing: {licence_sahi}
- Training data: {licence_data}

## Georeferencing
{georef_note}

## Reviewer checklist
- [ ] Open `*_wgs84.geojson` over the source scene in QGIS. Do the shapes sit on the objects?
- [ ] Delete false positives, add missed objects, set `status` to `verified` on what you checked.
- [ ] Second reviewer labels one scene independently -> record the agreement number.
- [ ] Confirm imagery rights and the retention date (default: delete 30 days after delivery).
- [ ] Confirm the licence lines above are acceptable for this customer.
- [ ] Create a file named `APPROVED` here (or `REJECTED` with a reason inside).
"""


def _fmt_stage_table(stage_seconds: dict) -> str:
    rows = [f"| {k} | {v} |" for k, v in stage_seconds.items() if k != "total_s"]
    rows.append(f"| **total** | **{stage_seconds.get('total_s', 0)}** |")
    return "| stage | seconds |\n|---|---|\n" + "\n".join(rows)


def _fmt_rows(images: list[dict]) -> str:
    if not images:
        return "_no images processed_"
    out = []
    for r in images:
        geo = "georeferenced" if r["georeferenced"] else "**PIXEL COORDS ONLY**"
        out.append(
            f"- `{r['file']}` ({r['megapixels']} MP, {r['tiles_total']} tiles, "
            f"{r['tiles_skipped_empty']} empty skipped): "
            f"**{r['detections_after_merge']}** detections "
            f"(merged down from {r['detections_before_merge']}, "
            f"-{r['duplicates_removed']} duplicates), "
            f"{r['polygons_from_masks']} polygons, {r['per_class']}, "
            f"{geo}, {r['wall_s']}s"
        )
    return "\n".join(out)


def write_report(summary: dict, out_dir: Path) -> Path:
    cfg = summary["config"]
    sched = summary.get("scheduler", {})
    experts = sched.get("experts", {})
    events = sched.get("events", [])

    def vram_for(name: str):
        vals = [e.get("vram_peak_mb", 0) for e in events if e.get("expert") == name]
        return max(vals) if vals else 0

    not_georef = [r["file"] for r in summary["images"] if not r["georeferenced"]]
    georef_note = (
        "All outputs carry the scene CRS and were written in both native CRS and WGS84."
        if not not_georef else
        "**WARNING** - no CRS/transform on: " + ", ".join(f"`{f}`" for f in not_georef) +
        ". Those outputs are pixel coordinates and are NOT a geospatial deliverable. "
        "Re-run from a GeoTIFF that keeps its transform."
    )

    text = TEMPLATE.format(
        job_id=summary["job_id"], customer=summary.get("customer", "internal"),
        task=cfg.get("task", ""), sensor=cfg.get("sensor", ""),
        generated=summary["generated_utc"], pipeline=summary["pipeline"],
        detector=cfg.get("detector"), detector_weights=cfg.get("detector_weights"),
        detector_loads=experts.get("detector", {}).get("loads", 0),
        detector_vram=vram_for("detector"),
        segmenter=cfg.get("segmenter"),
        segmenter_loads=experts.get("segmenter", {}).get("loads", 0),
        segmenter_vram=vram_for("segmenter"),
        max_resident=sched.get("max_resident_experts", 1), device=sched.get("device", "unknown"),
        rows=_fmt_rows(summary["images"]),
        total_scenes=summary["totals"]["scenes"], total_detections=summary["totals"]["detections"],
        total_polygons=summary["totals"]["polygons"], total_wall=summary["totals"]["wall_s"],
        stage_table=_fmt_stage_table(summary.get("stage_seconds", {})),
        licence_detector=summary["licences"]["detector"],
        licence_segmenter=summary["licences"]["segmenter"],
        licence_sahi=summary["licences"]["sahi"], licence_data=summary["licences"]["data"],
        georef_note=georef_note,
    )
    path = Path(out_dir) / "REPORT_DRAFT.md"
    path.write_text(text, encoding="utf-8")
    return path
