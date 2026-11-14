"""The automation layer: a Drive-folder queue with a human review gate.

Folder states (a job is wherever its folder is - no database to get out of sync):

  jobs/inbox/<job_id>/      input/*.tif + job.json + READY   (READY written LAST)
  jobs/processing/<job_id>/ claimed by a worker
  jobs/review/<job_id>/     output/ ready, waiting for a person
  jobs/done/<job_id>/       a human created APPROVED
  jobs/failed/<job_id>/     crashed, or a human created REJECTED

Three things this worker will not do, by design:
  - It never emails, shares or uploads anything. Delivery is a human action
    after APPROVED, because an unreviewed label set reaching a customer is the
    one mistake that cannot be walked back.
  - It never acts on a job without READY, so a half-uploaded 2 GB GeoTIFF is
    not processed as if it were complete.
  - It never silently drops a job: a crash writes error.txt and moves the
    folder to failed/ with the traceback intact.

Colab sessions die. On startup `requeue_stale()` pushes jobs abandoned in
processing/ back to inbox, and because RunMemory keeps completed stages, the
retry resumes at the stage that was interrupted instead of re-detecting
everything.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Iterable

from .config import IMG_EXT, JobConfig
from .memory import now_iso
from .pipeline import run_job
from .report import write_report

STAGES = ("inbox", "processing", "review", "done", "failed")


# ------------------------------------------------------------------- plumbing
def ensure_layout(root: Path) -> Path:
    root = Path(root)
    for s in STAGES:
        (root / "jobs" / s).mkdir(parents=True, exist_ok=True)
    (root / "worker").mkdir(parents=True, exist_ok=True)
    return root


def log(job_dir: Path | None, msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    if job_dir is not None:
        try:
            with open(job_dir / "log.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass            # a log write must never take the job down


def move(job_dir: Path, root: Path, stage: str) -> Path:
    dst = root / "jobs" / stage / job_dir.name
    if dst.exists():
        dst = dst.with_name(f"{job_dir.name}__{int(time.time())}")
    shutil.move(str(job_dir), str(dst))
    return dst


def write_status(job_dir: Path, state: str, **extra) -> None:
    (job_dir / "status.json").write_text(
        json.dumps({"state": state, "updated": now_iso(), **extra}, indent=2), encoding="utf-8")


def heartbeat(root: Path, **extra) -> None:
    (root / "worker" / "heartbeat.json").write_text(
        json.dumps({"alive": now_iso(), **extra}, indent=2), encoding="utf-8")


def claimable(job: Path) -> bool:
    return job.is_dir() and (job / "READY").exists() and (job / "job.json").exists()


# --------------------------------------------------------------------- resume
def requeue_stale(root: Path, older_than_min: float = 0.0) -> list[str]:
    """Move abandoned processing/ jobs back to inbox so a new session retries them."""
    root = Path(root)
    moved = []
    cutoff = time.time() - older_than_min * 60
    for job in sorted((root / "jobs" / "processing").iterdir()):
        if not job.is_dir():
            continue
        if older_than_min and job.stat().st_mtime > cutoff:
            continue
        (job / "READY").touch()
        attempts = _bump_attempts(job)
        log(job, f"requeued after an interrupted session (attempt {attempts})")
        move(job, root, "inbox")
        moved.append(job.name)
    return moved


def _bump_attempts(job: Path) -> int:
    p = job / "attempts.json"
    n = 0
    if p.exists():
        try:
            n = int(json.loads(p.read_text(encoding="utf-8")).get("attempts", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            n = 0
    n += 1
    p.write_text(json.dumps({"attempts": n, "last": now_iso()}), encoding="utf-8")
    return n


def attempts_of(job: Path) -> int:
    p = job / "attempts.json"
    if not p.exists():
        return 0
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("attempts", 0))
    except (json.JSONDecodeError, OSError, ValueError):
        return 0


# ----------------------------------------------------------------- one pass
def tick(root: Path, device=None, max_attempts: int = 3, verbose: bool = True) -> dict:
    """One pass over the queue. Safe to call repeatedly; safe to interrupt."""
    root = ensure_layout(root)
    stats = {"processed": 0, "failed": 0, "approved": 0, "rejected": 0, "skipped": 0}

    for job in sorted((root / "jobs" / "inbox").iterdir()):
        if not claimable(job):
            stats["skipped"] += 1
            continue
        if attempts_of(job) >= max_attempts:
            log(job, f"giving up after {attempts_of(job)} attempts")
            write_status(job, "failed", error=f"exceeded max_attempts={max_attempts}")
            move(job, root, "failed")
            stats["failed"] += 1
            continue

        job = move(job, root, "processing")
        _bump_attempts(job)
        write_status(job, "processing", attempt=attempts_of(job))
        try:
            cfg = JobConfig.load(job / "job.json")
            if cfg.unknown_keys:
                log(job, f"job.json has unrecognised keys (ignored): {cfg.unknown_keys}")
            log(job, f"start {cfg.job_id}: {cfg.detector} + {cfg.segmenter}, "
                     f"tile {cfg.tile}px/{cfg.overlap} overlap, "
                     f"max_resident_experts={cfg.max_resident_experts}")
            summary = run_job(job, cfg=cfg, device=device, verbose=verbose)
            write_report(summary, job / "output")
            write_status(job, "awaiting_review",
                         detections=summary["totals"]["detections"],
                         polygons=summary["totals"]["polygons"],
                         stage_seconds=summary["stage_seconds"])
            log(job, f"done: {summary['totals']['detections']} detections, "
                     f"{summary['totals']['polygons']} polygons, "
                     f"{summary['totals']['wall_s']}s -> review")
            move(job, root, "review")
            stats["processed"] += 1
        except Exception as e:
            (job / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_status(job, "failed", error=f"{type(e).__name__}: {e}"[:400],
                         attempt=attempts_of(job))
            log(job, f"FAILED: {type(e).__name__}: {e}")
            move(job, root, "failed")
            stats["failed"] += 1

    # the human gate - the only thing that promotes a job to done/
    for job in sorted((root / "jobs" / "review").iterdir()):
        if not job.is_dir():
            continue
        if (job / "APPROVED").exists():
            write_status(job, "approved_ready_to_deliver",
                         note="Share output/ with the customer manually. The worker never sends.")
            log(job, "APPROVED by a human -> done (delivery is manual)")
            move(job, root, "done")
            stats["approved"] += 1
        elif (job / "REJECTED").exists():
            reason = (job / "REJECTED").read_text(encoding="utf-8", errors="replace")[:500]
            write_status(job, "rejected", reason=reason)
            log(job, f"REJECTED by a human: {reason[:120]}")
            move(job, root, "failed")
            stats["rejected"] += 1
    return stats


# ------------------------------------------------------------------- the loop
def run_forever(root: str | Path, poll_s: int = 30, max_hours: float = 11.0,
                device=None, requeue_on_start: bool = True, max_attempts: int = 3,
                verbose: bool = True) -> None:
    """Watch the queue until the STOP file appears or max_hours elapses.

    max_hours defaults to 11: a free Colab session is cut around 12, and
    stopping cleanly before that leaves the queue in a consistent state rather
    than mid-move.
    """
    root = ensure_layout(Path(root))
    stop_file = root / "worker" / "STOP"
    if requeue_on_start:
        stale = requeue_stale(root)
        if stale:
            log(None, f"requeued {len(stale)} interrupted job(s): {stale}")
    deadline = time.time() + max_hours * 3600
    log(None, f"worker up. watching {root / 'jobs' / 'inbox'}")
    log(None, f"stop cleanly by creating {stop_file}")
    while time.time() < deadline:
        if stop_file.exists():
            log(None, "STOP found, exiting")
            break
        try:
            stats = tick(root, device=device, max_attempts=max_attempts, verbose=verbose)
            if any(v for k, v in stats.items() if k != "skipped"):
                log(None, f"tick {stats}")
            heartbeat(root, **stats, deadline_utc=now_iso())
        except Exception as e:
            log(None, f"tick error (continuing): {type(e).__name__}: {e}")
        time.sleep(poll_s)
    else:
        log(None, f"max_hours={max_hours} reached, exiting cleanly")
    heartbeat(root, state="stopped")


# -------------------------------------------------------------------- intake
def create_job(root: str | Path, job_id: str, images: Iterable[str | Path], **cfg) -> Path:
    """Stage a job. READY is written last, so a worker never sees a partial upload."""
    root = ensure_layout(Path(root))
    job = root / "jobs" / "inbox" / job_id
    (job / "input").mkdir(parents=True, exist_ok=True)
    copied = []
    for image in images:
        src = Path(image)
        if src.suffix.lower() not in IMG_EXT:
            raise ValueError(f"{src.name}: not an accepted image type {sorted(IMG_EXT)}")
        shutil.copy2(src, job / "input" / src.name)
        copied.append(src.name)
    (job / "job.json").write_text(
        json.dumps({"job_id": job_id, **cfg}, indent=2), encoding="utf-8")
    (job / "READY").touch()
    log(job, f"staged {len(copied)} image(s): {copied}")
    return job


def queue_state(root: str | Path) -> dict:
    root = ensure_layout(Path(root))
    return {s: sorted(p.name for p in (root / "jobs" / s).iterdir() if p.is_dir())
            for s in STAGES}
