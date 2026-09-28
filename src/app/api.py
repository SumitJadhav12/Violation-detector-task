"""
FastAPI Router for Video Violation Detection Service
"""

import os
import shutil
import json
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Request

from src.app.schemas import (
    JobCreateResponse,
    JobStatusResponse,
    JobReportResponse,
    JobStatus,
    ViolationItem,
    VideoSummary
)
from src.app.worker import job_manager, process_video_background

router = APIRouter(prefix="/api/v1", tags=["Safety Detection"])

UPLOAD_DIR = "outputs/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_default_model_path() -> str:
    """Finds best available model (INT8 ONNX -> ONNX -> PyTorch)."""
    candidates = [
        "outputs/models/best_int8.onnx",
        "outputs/models/best.onnx",
        "outputs/models/best.pt",
        "yolov8n.pt"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "yolov8n.pt"


@router.post("/detect-video", response_model=JobCreateResponse, status_code=202)
async def upload_video_for_detection(
    background_tasks: BackgroundTasks,
    request: Request,
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    conf_threshold: float = Form(0.35),
    max_frames: Optional[int] = Form(None)
):
    """
    Submits a video for asynchronous helmet safety violation detection.
    Does NOT block the server; returns immediately with a Job ID.
    Accepts either an uploaded video file or a local file path.
    """
    if file is not None and file.filename:
        saved_filename = f"upload_{file.filename}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        target_video_path = saved_path
    elif video_path and os.path.exists(video_path):
        target_video_path = video_path
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either an uploaded video file or a valid local video_path"
        )

    # Register job
    job = job_manager.create_job(target_video_path)
    model_path = get_default_model_path()

    # Dispatch to background tasks
    background_tasks.add_task(
        process_video_background,
        job_id=job.job_id,
        input_path=target_video_path,
        model_path=model_path,
        conf_threshold=conf_threshold,
        max_frames=max_frames
    )

    base_url = str(request.base_url).rstrip("/")
    return JobCreateResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        message="Video processing initiated asynchronously.",
        poll_url=f"{base_url}/api/v1/jobs/{job.job_id}",
        report_url=f"{base_url}/api/v1/jobs/{job.job_id}/report"
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the status and progress of a video analysis job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_percentage=job.progress_percentage,
        frames_processed=job.frames_processed,
        total_frames=job.total_frames,
        error=job.error
    )


@router.get("/jobs/{job_id}/report", response_model=JobReportResponse)
async def get_job_report(job_id: str, request: Request):
    """
    Returns the final structured JSON report with total workers,
    unique violations, timestamps, and snapshot URLs.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")

    if job.status == JobStatus.FAILED:
        return JobReportResponse(
            job_id=job.job_id,
            status=job.status,
            error=job.error
        )

    if job.status != JobStatus.COMPLETED or not job.result:
        return JobReportResponse(
            job_id=job.job_id,
            status=job.status
        )

    base_url = str(request.base_url).rstrip("/")
    res = job.result

    summary_data = VideoSummary(**res["summary"])
    violations_data: List[ViolationItem] = []

    for v in res.get("violations", []):
        snapshot_url = f"{base_url}/static/snapshots/{v['snapshot_filename']}"
        violations_data.append(ViolationItem(
            track_id=v["track_id"],
            timestamp_sec=v["timestamp_sec"],
            formatted_timestamp=v["formatted_timestamp"],
            frame_index=v["frame_index"],
            confidence=v["confidence"],
            snapshot_filename=v["snapshot_filename"],
            snapshot_url=snapshot_url
        ))

    output_video_url = None
    if "output_video_name" in res:
        output_video_url = f"{base_url}/static/videos/{res['output_video_name']}"

    return JobReportResponse(
        job_id=job.job_id,
        status=job.status,
        summary=summary_data,
        violations=violations_data,
        output_video_url=output_video_url
    )


@router.get("/latest-report", response_model=JobReportResponse)
async def get_latest_report(request: Request):
    """Returns the most recent completed compliance report from outputs/reports/."""
    report_dir = "outputs/reports"
    if not os.path.exists(report_dir):
        raise HTTPException(status_code=404, detail="No reports available yet")
    files = [f for f in os.listdir(report_dir) if f.endswith(".json")]
    if not files:
        raise HTTPException(status_code=404, detail="No reports available yet")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(report_dir, x)), reverse=True)
    latest_file = os.path.join(report_dir, files[0])
    with open(latest_file, "r") as f:
        res = json.load(f)

    base_url = str(request.base_url).rstrip("/")
    summary_data = VideoSummary(**res.get("summary", {}))
    violations_data: List[ViolationItem] = []
    for v in res.get("violations", []):
        snapshot_url = f"{base_url}/static/snapshots/{v['snapshot_filename']}"
        violations_data.append(ViolationItem(
            track_id=v["track_id"],
            timestamp_sec=v["timestamp_sec"],
            formatted_timestamp=v["formatted_timestamp"],
            frame_index=v["frame_index"],
            confidence=v["confidence"],
            snapshot_filename=v["snapshot_filename"],
            snapshot_url=snapshot_url
        ))
    output_video_url = None
    if "output_video_name" in res:
        output_video_url = f"{base_url}/static/videos/{res['output_video_name']}"

    return JobReportResponse(
        job_id="latest-report",
        status=JobStatus.COMPLETED,
        summary=summary_data,
        violations=violations_data,
        output_video_url=output_video_url
    )
