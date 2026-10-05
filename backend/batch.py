"""
batch.py — Background batch-processing queue.

A *batch job* applies one operation (an effect chain, a format export, or a
waveform edit) to many library files.  Items run **sequentially in a single
worker thread**, which keeps memory bounded even when the selection contains
multi-hundred-MB files (the DSP pipelines themselves stream chunk-by-chunk).

Guarantees
----------
* **Failure isolation** — an exception while processing one file is recorded on
  that item and the queue moves on; other files are unaffected.
* **No half products on cancel** — every output is rendered into a private
  temporary directory and only *registered into the library after it finishes
  successfully*.  Cancellation (or a failure) deletes the partial file; nothing
  unregistered ever lands in ``data/audio``.
* **Completed work survives cancellation** — items already registered stay in
  the library; only pending / in-flight items are dropped (the in-flight one is
  dropped without registering).

The manager is deliberately agnostic about DSP: the Flask layer supplies a
``runner`` (renders one output file, streaming, honouring cancel/progress
hooks) and a ``register`` callback (moves the finished file into the library
and returns the new entry).
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from . import storage
from .audio_io import ProcessingCancelled


# Item / job lifecycle states.
QUEUED = "queued"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL_ITEM = (SUCCESS, FAILED, CANCELLED)

# Runner signature: (item dict, out_path, on_progress, is_cancelled) -> None
Runner = Callable[[Dict[str, Any], str, Callable[[float], None],
                   Callable[[], bool]], None]
# Register signature: (item dict, out_path) -> new library entry dict
Register = Callable[[Dict[str, Any], str], Dict[str, Any]]

_MAX_JOBS = 30


class BatchManager:
    """Holds all batch jobs and runs one shared sequential worker."""

    def __init__(self, temp_root: str):
        self.temp_root = temp_root
        os.makedirs(temp_root, exist_ok=True)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._workers = 0
        self._purge_temp()
        # A single daemon worker drains every queue, one item at a time.
        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------ #

    def _purge_temp(self) -> None:
        """Remove render leftovers from a previous process run."""
        for name in os.listdir(self.temp_root):
            p = os.path.join(self.temp_root, name)
            if name.startswith("batch-"):
                shutil.rmtree(p, ignore_errors=True)

    def create_job(self, op: str, params: Dict[str, Any],
                   sources: List[Dict[str, Any]], runner: Runner,
                   register: Register, op_label: str = "") -> Dict[str, Any]:
        job_id = storage.new_id()
        now = time.time()
        items = [{
            "file_id": s["id"],
            "name": s.get("name", s["id"]),
            "status": QUEUED,
            "progress": 0.0,
            "message": "",
            "error": "",
            "result": None,
        } for s in sources]
        job = {
            "id": job_id,
            "op": op,
            "op_label": op_label,
            "params": copy.deepcopy(params),
            "status": "running",
            "created_at": now,
            "finished_at": None,
            "cancel_requested": False,
            "items": items,
            "_runner": runner,
            "_register": register,
            "_tmpdir": "",
        }
        with self._lock:
            self._jobs[job_id] = job
            self._trim_jobs_locked()
        self._wake.set()
        return self.snapshot(job_id)

    def _trim_jobs_locked(self) -> None:
        """Bound memory: drop the oldest dismissed/terminal jobs."""
        if len(self._jobs) <= _MAX_JOBS:
            return
        order = sorted(self._jobs.values(),
                       key=lambda j: j.get("finished_at") or j.get("created_at", 0))
        for j in order:
            if len(self._jobs) <= _MAX_JOBS:
                break
            if j["status"] in ("done", "cancelled"):
                self._jobs.pop(j["id"], None)

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = [self._snapshot_locked(j) for j in self._jobs.values()]
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return jobs

    def latest_job(self) -> Optional[Dict[str, Any]]:
        jobs = self.list_jobs()
        return jobs[0] if jobs else None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            return self._snapshot_locked(j) if j else None

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return None
            j["cancel_requested"] = True
            return self._snapshot_locked(j)

    def dismiss(self, job_id: str) -> bool:
        with self._lock:
            j = self._jobs.get(job_id)
            if not j or j["status"] not in ("done", "cancelled"):
                return False
            self._jobs.pop(job_id, None)
            return True

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    def _next_work(self) -> Optional[Dict[str, Any]]:
        """Find the oldest job that still has a queued item (single worker)."""
        with self._lock:
            jobs = sorted((j for j in self._jobs.values()
                           if j["status"] == "running"),
                          key=lambda j: j["created_at"])
            for job in jobs:
                if any(it["status"] == QUEUED for it in job["items"]):
                    return job
            return None

    def _worker_loop(self) -> None:
        while True:
            job = self._next_work()
            if job is None:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            self._run_job(job)

    def _run_job(self, job: Dict[str, Any]) -> None:
        job_id = job["id"]
        tmpdir = tempfile.mkdtemp(prefix=f"batch-{job_id}-", dir=self.temp_root)
        with self._lock:
            job["_tmpdir"] = tmpdir

        try:
            while True:
                with self._lock:
                    item = next((it for it in job["items"]
                                 if it["status"] == QUEUED), None)
                    if item is None:
                        break
                    if job["cancel_requested"]:
                        # A cancel arrived while the previous item was in flight:
                        # don't start another one; mark everything left.
                        self._cancel_rest_locked(job)
                        break
                    item["status"] = RUNNING
                self._run_item(job, item, tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            with self._lock:
                job["_tmpdir"] = ""
                self._finalize_locked(job)
                self._trim_jobs_locked()
            self._wake.set()

    def _cancel_rest_locked(self, job: Dict[str, Any]) -> None:
        for it in job["items"]:
            if it["status"] in (QUEUED, RUNNING):
                it["status"] = CANCELLED
                it["message"] = "已取消"

    def _run_item(self, job: Dict[str, Any], item: Dict[str, Any],
                  tmpdir: str) -> None:
        is_cancelled = lambda: job["cancel_requested"]  # noqa: E731

        def on_progress(frac: float) -> None:
            with self._lock:
                item["progress"] = max(0.0, min(1.0, float(frac)))

        out_path: Optional[str] = None
        try:
            suffix = "." + str(job["params"].get("format", "wav"))
            fd, out_path = tempfile.mkstemp(prefix="out-", suffix=suffix,
                                            dir=tmpdir)
            os.close(fd)
            os.unlink(out_path)  # renderers create the file themselves
            job["_runner"](item, out_path, on_progress, is_cancelled)

            if job["cancel_requested"]:
                # Render finished just as cancel landed — do not register it,
                # so the library never contains an item the user backed out of.
                raise _BatchCancel()

            entry = job["_register"](item, out_path)
            out_path = None  # register took ownership of the file
            with self._lock:
                item["status"] = SUCCESS
                item["progress"] = 1.0
                item["result"] = {
                    "id": entry.get("id"),
                    "name": entry.get("name"),
                }
                item["message"] = "完成"
        except (_BatchCancel, ProcessingCancelled):
            with self._lock:
                item["status"] = CANCELLED
                item["message"] = "已取消"
        except Exception as e:  # failure isolation: record and continue
            msg = str(e) or e.__class__.__name__
            with self._lock:
                item["status"] = FAILED
                item["error"] = msg[:500]
                item["message"] = "失败"
        finally:
            if out_path:
                try:
                    if os.path.isfile(out_path):
                        os.unlink(out_path)
                except OSError:
                    pass

    def _finalize_locked(self, job: Dict[str, Any]) -> None:
        if job["status"] != "running":
            return
        statuses = {it["status"] for it in job["items"]}
        if RUNNING in statuses or QUEUED in statuses:
            return
        job["finished_at"] = time.time()
        # A cancelled job may still contain completed items — they are kept.
        job["status"] = "cancelled" if job["cancel_requested"] else "done"

    # ------------------------------------------------------------------ #
    # Snapshots (JSON-safe copies handed to the API / UI)
    # ------------------------------------------------------------------ #

    def _snapshot_locked(self, job: Dict[str, Any]) -> Dict[str, Any]:
        items = []
        counts = {SUCCESS: 0, FAILED: 0, CANCELLED: 0,
                  RUNNING: 0, QUEUED: 0}
        progress_sum = 0.0
        for it in job["items"]:
            st = it["status"]
            counts[st] = counts.get(st, 0) + 1
            weight = 1.0 if st == SUCCESS else (it.get("progress", 0.0)
                                                if st == RUNNING else 0.0)
            progress_sum += weight
            items.append({
                "file_id": it["file_id"],
                "name": it["name"],
                "status": st,
                "progress": round(it.get("progress", 0.0), 4),
                "message": it.get("message", ""),
                "error": it.get("error", ""),
                "result": it.get("result"),
            })
        n = max(1, len(items))
        return {
            "id": job["id"],
            "op": job["op"],
            "op_label": job["op_label"],
            "params": copy.deepcopy(job["params"]),
            "status": job["status"],
            "cancel_requested": job["cancel_requested"],
            "created_at": job["created_at"],
            "finished_at": job.get("finished_at"),
            "total": len(items),
            "succeeded": counts[SUCCESS],
            "failed": counts[FAILED],
            "cancelled_count": counts[CANCELLED],
            "running": counts[RUNNING],
            "queued": counts[QUEUED],
            "progress": round(progress_sum / n, 4),
            "items": items,
        }

    def snapshot(self, job_id: str) -> Dict[str, Any]:
        return self.get_job(job_id)


class _BatchCancel(Exception):
    """Internal: render completed but cancellation was requested meanwhile."""
