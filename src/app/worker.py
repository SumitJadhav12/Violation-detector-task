"""
Background Video Worker & In-Memory Job Registry
Executes video processing asynchronously without blocking FastAPI event loop.
"""

import os
import threading
import uuid
import time
from typing import Dict, Any, Optional

from src.app.schemas import JobStatus


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


class JobManager:
    """Thread-safe in-memory job registry."""

    def __init__(self):
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_job(self, input_path: str) -> JobRecord:
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id, input_path)
        with self._lock:
            self._jobs[job_id] = record
        return record

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

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

    def fail_job(self, job_id: str, error_msg: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error_msg
                job.completed_at = time.time()


# Singleton instance
job_manager = JobManager()


def process_video_background(
    job_id: str,
    input_path: str,
    model_path: str,
    conf_threshold: float = 0.35,
    max_frames: Optional[int] = None
):
    """Worker function executed in background thread."""
    from src.detector import create_detector
    from src.video_processor import VideoSafetyProcessor

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
