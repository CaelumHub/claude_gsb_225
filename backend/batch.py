"""
batch.py — Batch processing queue for the audio library.

A *batch job* applies one operation (an effect chain, an export/format
conversion, or a single edit step) to many selected files.  Requirements this
module is built around:

* **Queued one-click processing.**  POSTing a job queues it; a small worker
  pool drains items sequentially within the job (one file at a time keeps
  memory bounded even when the selection contains multi-hundred-MB files).
* **Per-file progress and outcome.**  Every item has its own status
  (pending / running / success / failed / cancelled) and a 0..1 progress
  value; failures record the concrete error message and never abort the
  siblings.
* **Cancellation with no half products.**  Cancellation is cooperative: the
  in-flight item's streaming callbacks raise ``audio_io.CancelledError``, the
  not-yet-started items are marked ``cancelled``, and every output is rendered
  to a scratch temp file that is deleted unless the item finishes
  successfully.  Successful items have already been committed to the library
  and are kept.
* **Crash-safe commits.**  An item only enters the library registry once its
  render fully succeeds (``processing.register_*`` moves the finished file
  into place under the registry lock).

Job state lives in memory behind a lock; the library itself is the persistent
source of truth for results.  Finished jobs are retained (most recent first)
for the UI history panel and pruned automatically.
"""

from __future__ import annotations

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from . import audio_io, processing, storage
from .audio_io import CancelledError
from .storage import new_id, now_iso

# Item-level states
PENDING = "pending"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"
CANCELLED = "cancelled"

ACTIVE_STATES = (PENDING, RUNNING)
KEEP_JOBS = 30
MAX_ITEMS = 200

SUPPORTED_FORMATS = ("wav", "aiff", "raw", "mp3", "ogg", "flac", "m4a")
SUPPORTED_EDITS = ("gain", "normalize", "fade", "reverse")


class BatchError(ValueError):
    """Invalid job/operation specification (maps to HTTP 400)."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_operation(operation: str, params: Dict) -> None:
    if operation == "effects":
        chain = params.get("chain")
        if not isinstance(chain, list) or not chain:
            raise BatchError("效果链为空，请先选择预设或添加效果")
        # Fail fast on an unknown effect type rather than on file #1 of 40.
        from .effects import make_effect
        try:
            for spec in chain:
                if spec.get("enabled", True):
                    make_effect(spec, 44100)  # construct-only validation; sr irrelevant
        except ValueError as e:
            raise BatchError(str(e))
    elif operation == "export":
        fmt = params.get("format")
        if fmt not in SUPPORTED_FORMATS:
            raise BatchError(f"不支持的导出格式: {fmt!r}")
        if fmt in ("mp3", "ogg", "m4a", "aiff", "flac") and not audio_io.has_ffmpeg():
            raise BatchError(f"导出 {fmt.upper()} 需要 ffmpeg，但服务器未安装")
        if params.get("sample_rate") is not None:
            int(float(params["sample_rate"]))
        if params.get("channels") is not None and int(params["channels"]) not in (1, 2):
            raise BatchError("声道只支持 1（单声道）或 2（立体声）")
    elif operation == "edit":
        op = params.get("op")
        if op not in SUPPORTED_EDITS:
            raise BatchError(f"不支持的批量编辑: {op!r}")
        if op == "gain":
            float(params.get("edit_params", {}).get("gain_db", 0))
    else:
        raise BatchError(f"未知批量操作: {operation!r}")


# --------------------------------------------------------------------------- #
# Job manager
# --------------------------------------------------------------------------- #

class BatchManager:
    def __init__(self, store: storage.Storage):
        self.store = store
        self.scratch_dir = os.path.join(store.root, "batch_tmp")
        os.makedirs(self.scratch_dir, exist_ok=True)
        self._jobs: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        # Sequential jobs, one in-flight item each: bounded memory regardless
        # of how many large files are selected.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch")
        self._cleanup_stale_scratch()

    # ------------------------------------------------------------------ #

    def _cleanup_stale_scratch(self) -> None:
        """Remove scratch files left by a previous process (server restart)."""
        for name in os.listdir(self.scratch_dir):
            try:
                os.unlink(os.path.join(self.scratch_dir, name))
            except OSError:
                pass

    # ------------------------------------------------------------------ #

    def create_job(self, operation: str, params: Dict,
                   entries: List[Dict]) -> Dict:
        validate_operation(operation, params)
        if not entries:
            raise BatchError("未选择任何文件")
        # De-duplicate by id while preserving order.
        seen = set()
        items = []
        for e in entries:
            fid = e.get("id")
            if not fid or fid in seen:
                continue
            seen.add(fid)
            items.append({
                "id": new_id(),
                "file_id": fid,
                "name": e.get("name", fid),
                "status": PENDING,
                "progress": 0.0,
                "error": None,
                "result_id": None,
            })
        if not items:
            raise BatchError("未选择任何文件")
        if len(items) > MAX_ITEMS:
            raise BatchError(f"单次批量最多 {MAX_ITEMS} 个文件，当前 {len(items)} 个")

        job = {
            "id": new_id(),
            "operation": operation,
            "params": params,
            "label": params.get("label") or _default_label(operation, params),
            "status": "running",
            "cancel_requested": False,
            "cancel_event": threading.Event(),
            "created_at": now_iso(),
            "started_at": now_iso(),
            "finished_at": None,
            "items": items,
        }
        with self._lock:
            self._jobs[job["id"]] = job
        self._pool.submit(self._run, job["id"])
        return self.public_job(job)

    def cancel_job(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job["status"] == "running" and not job["cancel_requested"]:
                job["cancel_requested"] = True
                job["cancel_event"].set()
            return self.public_job(job)

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return self.public_job(job) if job else None

    def list_jobs(self, limit: int = KEEP_JOBS) -> List[Dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(),
                          key=lambda j: j.get("created_at", ""), reverse=True)
            return [self.public_job(j) for j in jobs[:limit]]

    def active_job(self) -> Optional[Dict]:
        with self._lock:
            for job in self._jobs.values():
                if job["status"] == "running":
                    return self.public_job(job)
        return None

    def public_job(self, job: Dict) -> Dict:
        """Deep-copy a job with internal-only fields stripped."""
        with self._lock:
            counts = {PENDING: 0, RUNNING: 0, SUCCESS: 0, FAILED: 0, CANCELLED: 0}
            items = []
            for it in job["items"]:
                counts[it["status"]] = counts.get(it["status"], 0) + 1
                items.append({
                    "id": it["id"],
                    "file_id": it["file_id"],
                    "name": it["name"],
                    "status": it["status"],
                    "progress": round(float(it["progress"]), 4),
                    "error": it["error"],
                    "result_id": it["result_id"],
                })
            return {
                "id": job["id"],
                "operation": job["operation"],
                "label": job["label"],
                "status": job["status"],
                "cancel_requested": job["cancel_requested"],
                "created_at": job["created_at"],
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "total": len(items),
                "counts": counts,
                "items": items,
            }

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return
        try:
            for it in job["items"]:
                if job["cancel_event"].is_set():
                    with self._lock:
                        if it["status"] == PENDING:
                            it["status"] = CANCELLED
                    continue
                self._run_item(job, it)
        finally:
            with self._lock:
                job["status"] = "cancelled" if job["cancel_event"].is_set() else "done"
                job["finished_at"] = now_iso()
            self._prune()

    def _run_item(self, job: Dict, it: Dict) -> None:
        cancel = job["cancel_event"]
        with self._lock:
            it["status"] = RUNNING

        entry = self.store.get_file(it["file_id"])
        if entry is None:
            self._fail(it, FileNotFoundError("文件已从音频库中删除"))
            return
        src_path = os.path.join(self.store.root, entry.get("path", ""))
        if not os.path.isfile(src_path):
            self._fail(it, FileNotFoundError("音频文件丢失（存储中找不到）"))
            return

        op = job["operation"]
        ext = "wav"
        if op == "export":
            ext = job["params"].get("format", "wav")
        fd, out_path = tempfile.mkstemp(
            prefix=f"{job['id']}_{it['id']}_", suffix="." + ext,
            dir=self.scratch_dir)
        os.close(fd)
        # mkstemp creates the file; writers expect to create it themselves.
        os.unlink(out_path)

        def progress(frac: float) -> None:
            with self._lock:
                it["progress"] = float(frac)

        try:
            if op == "effects":
                processing.apply_effects(
                    src_path, out_path, job["params"].get("chain", []),
                    on_progress=progress, should_cancel=cancel.is_set)
                name = job["params"].get("name") or f"fx-{entry['name']}"
                new_entry = processing.register_wav(
                    self.store, entry["id"], name, out_path,
                    {"effects": job["params"].get("chain", []),
                     "batch_job": job["id"]})
            elif op == "edit":
                ep = job["params"].get("edit_params", {})
                sub_op = job["params"]["op"]
                processing.apply_edit(
                    src_path, out_path, sub_op, ep,
                    on_progress=progress, should_cancel=cancel.is_set)
                name = job["params"].get("name") or f"{sub_op}-{entry['name']}"
                new_entry = processing.register_wav(
                    self.store, entry["id"], name, out_path,
                    {"op": sub_op, "batch_job": job["id"]})
            elif op == "export":
                p = job["params"]
                processing.export_file(
                    src_path, out_path, p.get("format", "wav"),
                    dst_sr=p.get("sample_rate"),
                    sample_width=p.get("bit_depth", "pcm16"),
                    channels=p.get("channels"),
                    bitrate=p.get("bitrate", "192k"),
                    on_progress=progress, should_cancel=cancel.is_set)
                name_base = entry["name"].rsplit(".", 1)[0]
                new_entry = processing.register_exported(
                    self.store, entry["id"], name_base, out_path,
                    p.get("format", "wav"))
                new_entry["batch_job"] = job["id"]
                self.store.update_file(new_entry["id"], {"batch_job": job["id"]})
            else:  # validated earlier; defensive
                raise BatchError(f"unknown operation {op!r}")

            with self._lock:
                it["status"] = SUCCESS
                it["progress"] = 1.0
                it["result_id"] = new_entry["id"]
        except CancelledError:
            self._discard(out_path)
            with self._lock:
                it["status"] = CANCELLED
                it["error"] = "已取消"
        except Exception as e:  # one file failing never affects the others
            self._discard(out_path)
            self._fail(it, e)

    # ------------------------------------------------------------------ #

    def _fail(self, it: Dict, e: Exception) -> None:
        msg = str(e).strip() or e.__class__.__name__
        if len(msg) > 300:
            msg = msg[:300] + "…"
        with self._lock:
            it["status"] = FAILED
            it["error"] = msg

    @staticmethod
    def _discard(path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass

    def _prune(self) -> None:
        """Keep only the newest KEEP_JOBS finished jobs in memory."""
        with self._lock:
            jobs = sorted(self._jobs.values(),
                          key=lambda j: j.get("created_at", ""), reverse=True)
            running = [j for j in jobs if j["status"] == "running"]
            keep_ids = {j["id"] for j in running}
            for j in jobs:
                if len(keep_ids) >= KEEP_JOBS:
                    break
                keep_ids.add(j["id"])
            for jid in [j["id"] for j in jobs if j["id"] not in keep_ids]:
                self._jobs.pop(jid, None)


def _default_label(operation: str, params: Dict) -> str:
    if operation == "effects":
        name = params.get("preset_name")
        n = len(params.get("chain", []))
        return f"效果链 · {name}" if name else f"效果链 · {n} 个效果"
    if operation == "export":
        return f"导出为 {str(params.get('format', 'wav')).upper()}"
    if operation == "edit":
        labels = {"gain": "增益调整", "normalize": "峰值归一化",
                  "fade": "淡入淡出", "reverse": "反转"}
        return "编辑 · " + labels.get(params.get("op", ""), params.get("op", ""))
    return operation
