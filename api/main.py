"""FastAPI intake and delivery layer.

This process does NOT run models. It takes uploads, reports status, and serves
approved results. The GPU work happens in the Colab worker watching the same
queue folder. That split is deliberate: the API can then live on a free
CPU-only host (Hugging Face Spaces) while the expensive part runs wherever a
GPU happens to be available, and neither needs to know the other exists -
they only share the queue directory.

The human gate is enforced here as an access rule, not just a convention:

  GET /jobs/{id}/results/...  serves jobs in done/ only (a human approved them)
  review-stage output         needs the reviewer token
  POST /jobs/{id}/review      needs the reviewer token

So there is no code path that hands an unreviewed label set to a customer.

Run:
  export SKYVONYX_ROOT=/data/skyvonyx
  export SKYVONYX_REVIEWER_TOKEN=...        # omit to disable reviewer endpoints
  uvicorn api.main:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from skyvonyx.config import IMG_EXT, JobConfig
from skyvonyx.queue_worker import STAGES, ensure_layout, queue_state

ROOT = Path(os.environ.get("SKYVONYX_ROOT", "./skyvonyx_data")).resolve()
REVIEWER_TOKEN = os.environ.get("SKYVONYX_REVIEWER_TOKEN", "")
MAX_UPLOAD_MB = float(os.environ.get("SKYVONYX_MAX_UPLOAD_MB", "512"))
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

app = FastAPI(
    title="Skyvonyx labelling API",
    version="0.2.0",
    description="Intake, status and approved delivery for the staged "
                "SAHI -> YOLOv8 -> SAM 2 pipeline. Inference runs in the queue worker.",
)


# --------------------------------------------------------------------- helpers
def require_reviewer(x_skyvonyx_token: str = Header(default="")) -> bool:
    """Reviewer endpoints are off unless a token is configured.

    Defaulting to "off" rather than "open" matters: an unset env var on a
    public Space would otherwise let anyone approve a delivery.
    """
    if not REVIEWER_TOKEN:
        raise HTTPException(503, "reviewer endpoints disabled: set SKYVONYX_REVIEWER_TOKEN")
    if x_skyvonyx_token != REVIEWER_TOKEN:
        raise HTTPException(401, "bad or missing X-Skyvonyx-Token")
    return True


def find_job(job_id: str) -> tuple[str, Path]:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(400, "invalid job_id")
    for stage in STAGES:
        candidate = ROOT / "jobs" / stage / job_id
        if candidate.is_dir():
            return stage, candidate
    raise HTTPException(404, f"no job {job_id!r}")


def safe_output_file(job_dir: Path, name: str) -> Path:
    """Resolve a requested filename inside output/ only.

    `name` comes from the URL, so "../../job.json" must not escape. Comparing
    resolved paths is the check that actually holds under symlinks.
    """
    out = (job_dir / "output").resolve()
    target = (out / name).resolve()
    if not target.is_file() or out not in target.parents:
        raise HTTPException(404, f"no output file {name!r}")
    return target


# ---------------------------------------------------------------------- routes
@app.get("/healthz")
def healthz() -> dict:
    ensure_layout(ROOT)
    return {"ok": True, "root": str(ROOT), "reviewer_endpoints": bool(REVIEWER_TOKEN)}


@app.get("/jobs")
def list_jobs() -> dict:
    return {"queue": queue_state(ROOT)}


@app.post("/jobs", status_code=201)
async def create_job(
    files: list[UploadFile] = File(..., description="GeoTIFF preferred; PNG/JPG lose georeferencing"),
    customer: str = Form("internal"),
    job_id: str = Form(""),
    detector: str = Form("yolov8"),
    segmenter: str = Form("sam2"),
    detector_weights: str = Form("yolov8n.pt"),
    sensor: str = Form("optical"),
    conf: float = Form(0.25),
    tile: int = Form(1024),
    overlap: float = Form(0.2),
) -> dict:
    """Stage a job for the worker. READY is written last, after every byte landed."""
    job_id = job_id or f"job_{uuid.uuid4().hex[:10]}"
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(400, "job_id must be alphanumeric with - or _")
    ensure_layout(ROOT)
    if any((ROOT / "jobs" / s / job_id).exists() for s in STAGES):
        raise HTTPException(409, f"job {job_id!r} already exists")

    job_dir = ROOT / "jobs" / "inbox" / job_id
    (job_dir / "input").mkdir(parents=True)
    written = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in IMG_EXT:
                raise HTTPException(415, f"{upload.filename}: accepted types are {sorted(IMG_EXT)}")
            dest = job_dir / "input" / Path(upload.filename).name
            size = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = await upload.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB * 1e6:
                        raise HTTPException(413, f"{upload.filename} exceeds {MAX_UPLOAD_MB} MB")
                    f.write(chunk)
            written.append({"file": dest.name, "bytes": size})

        cfg = {"job_id": job_id, "customer": customer, "detector": detector,
               "segmenter": segmenter, "detector_weights": detector_weights,
               "sensor": sensor, "conf": conf, "tile": tile, "overlap": overlap}
        JobConfig(**cfg)                                  # reject bad values before READY
        (job_dir / "job.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (job_dir / "READY").touch()
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)        # never leave a half-staged job
        raise

    return {"job_id": job_id, "stage": "inbox", "files": written,
            "note": "queued. a GPU worker picks it up, then a human reviews it."}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    stage, job_dir = find_job(job_id)
    status = {}
    if (job_dir / "status.json").is_file():
        status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    body = {"job_id": job_id, "stage": stage, "status": status,
            "deliverable": stage == "done"}
    if stage == "failed" and (job_dir / "error.txt").is_file():
        body["error_tail"] = (job_dir / "error.txt").read_text(encoding="utf-8")[-800:]
    if stage in {"review", "done"} and (job_dir / "output").is_dir():
        body["outputs"] = sorted(p.name for p in (job_dir / "output").iterdir() if p.is_file())
    return body


@app.get("/jobs/{job_id}/summary")
def job_summary(job_id: str) -> JSONResponse:
    """Counts, per-stage timings and the scheduler report. Available at review
    stage too - these are our own numbers, not the customer's deliverable."""
    _, job_dir = find_job(job_id)
    path = job_dir / "output" / "summary.json"
    if not path.is_file():
        raise HTTPException(404, "no summary yet")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/jobs/{job_id}/results/{name}")
def job_result(job_id: str, name: str, x_skyvonyx_token: str = Header(default="")) -> FileResponse:
    """Serve an output file. Approved jobs are open; anything still in review
    needs the reviewer token, so unreviewed labels cannot leak."""
    stage, job_dir = find_job(job_id)
    if stage != "done":
        if not REVIEWER_TOKEN or x_skyvonyx_token != REVIEWER_TOKEN:
            raise HTTPException(
                403, f"job is in {stage!r}, not approved. A reviewer must create APPROVED "
                     "before results are delivered.")
    return FileResponse(safe_output_file(job_dir, name))


@app.get("/jobs/{job_id}/geojson")
def job_geojson(job_id: str, x_skyvonyx_token: str = Header(default="")) -> JSONResponse:
    """The WGS84 FeatureCollection, merged across every scene in the job.

    RFC 7946 lon/lat, which is what a web map expects - this is the endpoint a
    globe or slippy-map front end points at.
    """
    stage, job_dir = find_job(job_id)
    if stage != "done" and (not REVIEWER_TOKEN or x_skyvonyx_token != REVIEWER_TOKEN):
        raise HTTPException(403, f"job is in {stage!r}, not approved")
    out = job_dir / "output"
    files = sorted(out.glob("*_wgs84.geojson"))
    if not files:
        if sorted(out.glob("*NOT_GEOREFERENCED*.geojson")):
            raise HTTPException(
                409, "this job's output is in pixel coordinates (the input had no CRS), "
                     "so there is no WGS84 GeoJSON to serve")
        raise HTTPException(404, "no GeoJSON yet")
    features = []
    for path in files:
        fc = json.loads(path.read_text(encoding="utf-8"))
        for feat in fc.get("features", []):
            feat.setdefault("properties", {})["source_scene"] = path.name
            features.append(feat)
    return JSONResponse({"type": "FeatureCollection", "features": features,
                         "properties": {"job_id": job_id, "scenes": len(files),
                                        "accuracy": "NOT MEASURED - see summary.json"}})


@app.post("/jobs/{job_id}/review")
def submit_review(job_id: str, decision: str = Form(...), reason: str = Form(""),
                  _: bool = Depends(require_reviewer)) -> dict:
    """Record a human decision. The worker acts on the flag file on its next tick."""
    stage, job_dir = find_job(job_id)
    if stage != "review":
        raise HTTPException(409, f"job is in {stage!r}, nothing to review")
    decision = decision.lower().strip()
    if decision == "approved":
        (job_dir / "APPROVED").write_text(reason or "approved via API", encoding="utf-8")
    elif decision == "rejected":
        if not reason:
            raise HTTPException(400, "a rejection needs a reason")
        (job_dir / "REJECTED").write_text(reason, encoding="utf-8")
    else:
        raise HTTPException(400, "decision must be 'approved' or 'rejected'")
    return {"job_id": job_id, "decision": decision,
            "note": "the worker will move the job on its next tick"}
