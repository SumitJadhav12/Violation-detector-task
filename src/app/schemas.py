"""
Pydantic Schemas for FastAPI Endpoints
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ViolationItem(BaseModel):
    track_id: int = Field(..., description="Unique persistent identifier of the worker")
    timestamp_sec: float = Field(..., description="Timestamp in seconds when violation occurred")
    formatted_timestamp: str = Field(..., description="Human-readable timestamp HH:MM:SS.mmm")
    frame_index: int = Field(..., description="Video frame number where violation was logged")
    confidence: float = Field(..., description="Detection confidence score")
    snapshot_filename: str = Field(..., description="Filename of the saved snapshot")
    snapshot_url: str = Field(..., description="URL endpoint to access the snapshot image")


class VideoSummary(BaseModel):
    total_frames_processed: int
    video_duration_sec: float
    total_workers_detected: int
    unique_violations: int
    processing_time_sec: float
    average_processing_fps: float


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    poll_url: str
    report_url: str
    resolved_video_path: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_percentage: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    error: Optional[str] = None


class JobReportResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    progress_percentage: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    summary: Optional[VideoSummary] = None
    violations: List[ViolationItem] = []
    output_video_url: Optional[str] = None
    error: Optional[str] = None
