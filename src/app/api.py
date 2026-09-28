"""
FastAPI Router for Video Violation Detection Service
"""

import os
import shutil
import json
import urllib.parse
from typing import Optional, List
import cv2
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

SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv')


def resolve_video_source(raw_path: str) -> tuple[str, str]:
    """
    Cleans and resolves user-supplied paths:
    - Strips surrounding quotes, whitespace, and shell prefixes
    - Expands user home directories (~) and environment variables
    - Handles file:// URIs
    - If path is a folder (e.g. C:\\Users\\sj165\\Downloads):
      locates all supported video files, prioritizes non-annotated raw source videos,
      and automatically selects the latest modified video.
    - If path is a direct file:
      validates existence and supported extension.
    """
    if not raw_path or not raw_path.strip():
        raise HTTPException(
            status_code=400,
            detail="Video path is empty. Please enter a valid video file or folder path."
        )

    cleaned = raw_path.strip()
    if cleaned.startswith("&"):
        cleaned = cleaned[1:].strip()

    cleaned = cleaned.strip('"').strip("'").strip()

    if cleaned.lower().startswith("file:///"):
        cleaned = urllib.parse.unquote(cleaned[8:])
    elif cleaned.lower().startswith("file://"):
        cleaned = urllib.parse.unquote(cleaned[7:])

    cleaned = cleaned.strip('"').strip("'").strip()
    expanded = os.path.expanduser(os.path.expandvars(cleaned))
    norm_path = os.path.normpath(expanded)

    if not os.path.exists(norm_path):
        raise HTTPException(
            status_code=400,
            detail=f"Path not found: '{cleaned}'. Please check the directory or file path and try again."
        )

    if os.path.isdir(norm_path):
        candidate_files: List[str] = []
        try:
            for fname in os.listdir(norm_path):
                fpath = os.path.join(norm_path, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                    candidate_files.append(fpath)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error accessing directory '{norm_path}': {str(e)}"
            )

        if not candidate_files:
            for root, _, files in os.walk(norm_path):
                for f in files:
                    if f.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                        candidate_files.append(os.path.join(root, f))
                if candidate_files:
                    break

        if not candidate_files:
            ext_str = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
            raise HTTPException(
                status_code=400,
                detail=f"No supported video files ({ext_str}) found in folder: '{norm_path}'. Please copy a video into this folder or specify the video file directly."
            )

        # Prioritize raw source videos over previously annotated output videos
        raw_videos = [f for f in candidate_files if not os.path.basename(f).lower().startswith("annotated_")]
        if raw_videos:
            chosen = max(raw_videos, key=os.path.getmtime)
        else:
            chosen = max(candidate_files, key=os.path.getmtime)

        resolved_path = os.path.abspath(chosen)
        message = (
            f"Folder detected: Automatically selected latest video '{os.path.basename(chosen)}' "
            f"from '{norm_path}'."
        )
        return resolved_path, message

    if os.path.isfile(norm_path):
        if not norm_path.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
            ext_str = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
            raise HTTPException(
                status_code=400,
                detail=f"File '{os.path.basename(norm_path)}' is not a supported video format ({ext_str})."
            )
        resolved_path = os.path.abspath(norm_path)
        message = f"Selected video file '{os.path.basename(resolved_path)}' for processing."
        return resolved_path, message

    raise HTTPException(
        status_code=400,
        detail=f"Invalid path type for '{cleaned}'. Please specify a file or folder."
    )


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
    conf_threshold: float = Form(0.20),
    max_frames: Optional[int] = Form(None)
):
    """
    Submits a video for asynchronous helmet safety violation detection.
    Does NOT block the server; returns immediately with a Job ID.
    Accepts:
      - Uploaded video file (multipart form data)
      - Direct local video file path (e.g. C:\\Users\\...\\video.mp4)
      - Local folder path (e.g. C:\\Users\\sj165\\Downloads) - automatically selects the latest video!
    """
    status_msg = "Video processing initiated asynchronously."
    if file is not None and file.filename:
        safe_filename = os.path.basename(file.filename)
        saved_filename = f"upload_{safe_filename}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        target_video_path, status_msg = resolve_video_source(saved_path)
    elif video_path:
        target_video_path, status_msg = resolve_video_source(video_path)
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either an uploaded video file or a valid local video_path or folder path"
        )

    # Validate that OpenCV can open the video stream before queueing
    cap = cv2.VideoCapture(target_video_path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read video file '{target_video_path}'. File may be corrupted or an invalid video stream."
        )
    cap.release()

    # Normalize max_frames <= 0 to None (all frames)
    effective_max_frames = max_frames if (max_frames is not None and max_frames > 0) else None

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
        max_frames=effective_max_frames
    )

    base_url = str(request.base_url).rstrip("/")
    return JobCreateResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        message=status_msg,
        poll_url=f"{base_url}/api/v1/jobs/{job.job_id}",
        report_url=f"{base_url}/api/v1/jobs/{job.job_id}/report",
        resolved_video_path=target_video_path
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the live status and progress percentage of a video analysis job."""
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
    If the video is still being processed in the background, returns
    the current progress percentage and frame count.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")

    if job.status == JobStatus.FAILED:
        return JobReportResponse(
            job_id=job.job_id,
            status=job.status,
            error=job.error,
            message=f"Job processing failed: {job.error}"
        )

    if job.status != JobStatus.COMPLETED or not job.result:
        pct = job.progress_percentage
        cur = job.frames_processed
        tot = job.total_frames or "?"
        return JobReportResponse(
            job_id=job.job_id,
            status=job.status,
            progress_percentage=pct,
            frames_processed=cur,
            total_frames=job.total_frames,
            message=f"Video analysis in progress ({pct}% completed: {cur}/{tot} frames). Please poll again or monitor GET /api/v1/jobs/{job.job_id}."
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
        progress_percentage=100.0,
        frames_processed=job.frames_processed,
        total_frames=job.total_frames,
        message="Video analysis completed successfully. Final structured report and HUD annotated video are available.",
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
        progress_percentage=100.0,
        frames_processed=summary_data.total_frames_processed,
        total_frames=summary_data.total_frames_processed,
        message="Loaded most recent completed video analysis report.",
        summary=summary_data,
        violations=violations_data,
        output_video_url=output_video_url
    )

