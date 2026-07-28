"""Minimal in-memory job registry -- a Phase-1 stand-in for the real
async execution layer (Phase 2's FastAPI /jobs + tier-2/3 dispatch,
engine/service/queue.py). `get_run_status` (task 2.5) needs something
real to query rather than a mock; no write-tool that WOULD populate
this (start_retrain) is in Phase 1's scope, so tests seed jobs directly
via `register_job`/`update_job`. Never silently invents a job's status
for an unknown id -- fails loud instead (asynchronous honesty, tool
contracts spec principle 5).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

_JOBS: dict[str, dict[str, Any]] = {}


class UnknownJobError(Exception):
    """Raised by get_job() for a job_id this registry has never seen."""


def register_job(job_id: str, kind: str) -> dict[str, Any]:
    job = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "progress": 0.0,
        "partial_results": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _JOBS[job_id] = job
    return dict(job)


def update_job(job_id: str, status: str | None = None, progress: float | None = None, partial_results: Any = None) -> dict[str, Any]:
    if job_id not in _JOBS:
        raise UnknownJobError(f"update_job: no job registered for job_id={job_id!r}")
    job = _JOBS[job_id]
    if status is not None:
        job["status"] = status
    if progress is not None:
        job["progress"] = progress
    if partial_results is not None:
        job["partial_results"] = partial_results
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    return dict(job)


def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in _JOBS:
        raise UnknownJobError(f"get_job: no job registered for job_id={job_id!r}")
    return dict(_JOBS[job_id])
