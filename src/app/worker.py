"""
Background Video Worker & In-Memory Job Registry
Executes video processing asynchronously without blocking FastAPI event loop.
"""

import os
import json
import threading
import uuid
import time
from typing import Dict, Any, Optional

from src.app.schemas import JobStatus

JOBS_DIR = "outputs/jobs"
REPORTS_DIR = "outputs/reports"
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


class JobRecord:
    def __init__(self, job_id: str, input_path: str):
        self.job_id = job_id
        self.input_path = input_path
        self.status = JobStatus.PENDING
        self.progress_percentage = 0.0
        self.frames_processed = 0
        self.total_frames = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        status_val = self.status.value if hasattr(self.status, "value") else str(self.status)
        return {
            "job_id": self.job_id,
            "input_path": self.input_path,
            "status": status_val,
            "progress_percentage": self.progress_percentage,
            "frames_processed": self.frames_processed,
            "total_frames": self.total_frames,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
        rec = cls(data["job_id"], data.get("input_path", ""))
        status_str = data.get("status", "pending")
        try:
            rec.status = JobStatus(status_str)
        except ValueError:
            rec.status = JobStatus.PENDING
        rec.progress_percentage = float(data.get("progress_percentage", 0.0))
        rec.frames_processed = int(data.get("frames_processed", 0))
        rec.total_frames = int(data.get("total_frames", 0))
        rec.result = data.get("result")
        rec.error = data.get("error")
        rec.created_at = float(data.get("created_at", time.time()))
        rec.completed_at = data.get("completed_at")
        return rec


class JobManager:
    """Thread-safe and disk-persisted job registry."""

    def __init__(self):
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def _save_job_to_disk(self, record: JobRecord):
        try:
            path = os.path.join(JOBS_DIR, f"{record.job_id}.json")
            with open(path, "w") as f:
                json.dump(record.to_dict(), f, indent=2)
        except Exception:
            pass

    def create_job(self, input_path: str) -> JobRecord:
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id, input_path)
        with self._lock:
            self._jobs[job_id] = record
            self._save_job_to_disk(record)
        return record

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]

            # Try loading from disk (recovering from uvicorn reload/restart)
            disk_path = os.path.join(JOBS_DIR, f"{job_id}.json")
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, "r") as f:
                        data = json.load(f)
                    rec = JobRecord.from_dict(data)
                    self._jobs[job_id] = rec
                    return rec
                except Exception:
                    pass
            return None

    def update_progress(self, job_id: str, pct: float, current_frame: int, total_frames: int):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress_percentage = round(pct, 1)
                job.frames_processed = current_frame
                job.total_frames = total_frames
                job.status = JobStatus.PROCESSING

    def complete_job(self, job_id: str, result: Dict[str, Any]):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.progress_percentage = 100.0
                job.result = result
                job.completed_at = time.time()
                self._save_job_to_disk(job)

                # Persist completed report to outputs/reports/
                try:
                    rep_path = os.path.join(REPORTS_DIR, f"report_{int(time.time())}.json")
                    with open(rep_path, "w") as rf:
                        json.dump(result, rf, indent=2)
                except Exception:
                    pass

    def fail_job(self, job_id: str, error_msg: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error_msg
                job.completed_at = time.time()
                self._save_job_to_disk(job)


# Singleton instance
job_manager = JobManager()


def process_video_background(
    job_id: str,
    input_path: str,
    model_path: str,
    conf_threshold: float = 0.20,
    max_frames: Optional[int] = None
):
    """Worker function executed in background thread."""
    from src.detector import create_detector
    from src.video_processor import VideoSafetyProcessor

    if max_frames is not None and max_frames <= 0:
        max_frames = None

    job = job_manager.get_job(job_id)
    if not job:
        return

    try:
        job.status = JobStatus.PROCESSING
        detector = create_detector(model_path, device="cpu")
        processor = VideoSafetyProcessor(
            detector=detector,
            output_dir="outputs",
            conf_threshold=conf_threshold
        )

        def on_progress(pct: float, cur_frame: int, tot_frames: int):
            job_manager.update_progress(job_id, pct, cur_frame, tot_frames)

        result = processor.process_video(
            input_video_path=input_path,
            progress_callback=on_progress,
            max_frames=max_frames
        )

        job_manager.complete_job(job_id, result)

    except Exception as exc:
        job_manager.fail_job(job_id, str(exc))
