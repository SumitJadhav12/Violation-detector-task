"""
Video Processing Pipeline
Decodes input video, executes detection & ByteTrack tracking,
extracts timestamped violation snapshots, draws an interactive HUD overlay,
and writes the final annotated video.
"""

import os
import time
from typing import Dict, Any, List, Optional, Callable
import cv2
import numpy as np

from src.detector import BaseDetector, create_detector
from src.tracker import SafetyTracker, TrackedWorker, WorkerState


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


class VideoSafetyProcessor:
    """End-to-end video analyzer for helmet violation detection and reporting."""

    def __init__(
        self,
        detector: BaseDetector,
        output_dir: str = "outputs",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ):
        self.detector = detector
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        self.videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

    def process_video(
        self,
        input_video_path: str,
        output_video_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process an input video file through detector and tracker.
        Returns a detailed JSON-serializable report.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video file not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames and max_frames < total_frames:
            total_frames = max_frames

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        if not output_video_name:
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            output_video_name = f"annotated_{base_name}_{int(time.time())}.mp4"

        output_video_path = os.path.join(self.videos_dir, output_video_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        tracker = SafetyTracker(iou_threshold=0.35, max_lost_frames=int(fps * 1.5))
        violations_record: List[Dict[str, Any]] = []

        frame_idx = 0
        start_time = time.time()
        fps_meter_time = start_time
        current_fps = 0.0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and frame_idx >= max_frames):
                    break

                timestamp_sec = frame_idx / fps
                frame_idx += 1

                # Step 1: Detect bounding boxes and classes
                detections = self.detector.detect(
                    frame,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold
                )

                # Step 2: Track & update violation state machine
                active_workers, new_violations = tracker.update(
                    detections, frame_idx, timestamp_sec
                )

                # Step 3: Handle snapshot extraction for new unique violations
                for violator in new_violations:
                    ms_stamp = int(timestamp_sec * 1000)
                    snapshot_filename = f"violation_track_{violator.track_id}_{ms_stamp}ms.jpg"
                    snapshot_path = os.path.join(self.snapshots_dir, snapshot_filename)

                    # Create annotated snapshot of the violation frame
                    snapshot_img = frame.copy()
                    x1, y1, x2, y2 = violator.xyxy.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # Highlight violator with red box
                    cv2.rectangle(snapshot_img, (x1, y1), (x2, y2), (0, 0, 240), 3)
                    label = f"VIOLATION: Track #{violator.track_id} ({violator.confidence:.2f})"
                    cv2.putText(
                        snapshot_img, label, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )

                    # Watermark timestamp
                    time_str = f"Time: {format_timestamp(timestamp_sec)} (Frame {frame_idx})"
                    cv2.putText(
                        snapshot_img, time_str, (20, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                    )

                    cv2.imwrite(snapshot_path, snapshot_img)
                    violator.snapshot_path = snapshot_path

                    violations_record.append({
                        "track_id": violator.track_id,
                        "timestamp_sec": round(timestamp_sec, 2),
                        "formatted_timestamp": format_timestamp(timestamp_sec),
                        "frame_index": frame_idx,
                        "confidence": round(float(violator.confidence), 3),
                        "snapshot_filename": snapshot_filename,
                        "snapshot_path": snapshot_path
                    })

                # Step 4: Render HUD & bounding boxes on current frame
                if frame_idx % 10 == 0:
                    now = time.time()
                    current_fps = 10.0 / max(0.001, (now - fps_meter_time))
                    fps_meter_time = now

                self._draw_overlay(
                    frame,
                    active_workers,
                    total_workers=tracker.all_time_workers_count,
                    unique_violations=tracker.unique_violations_count,
                    fps_val=current_fps,
                    timestamp_sec=timestamp_sec
                )

                writer.write(frame)

                if progress_callback and total_frames > 0:
                    progress_pct = (frame_idx / total_frames) * 100.0
                    progress_callback(progress_pct, frame_idx, total_frames)

        finally:
            cap.release()
            writer.release()

        elapsed_total = time.time() - start_time
        processing_fps = frame_idx / max(0.001, elapsed_total)

        return {
            "status": "success",
            "summary": {
                "total_frames_processed": frame_idx,
                "video_duration_sec": round(frame_idx / fps, 2),
                "total_workers_detected": tracker.all_time_workers_count,
                "unique_violations": tracker.unique_violations_count,
                "processing_time_sec": round(elapsed_total, 2),
                "average_processing_fps": round(processing_fps, 2)
            },
            "output_video_path": output_video_path,
            "output_video_name": output_video_name,
            "violations": violations_record
        }

    def _draw_overlay(
        self,
        frame: np.ndarray,
        active_workers: List[TrackedWorker],
        total_workers: int,
        unique_violations: int,
        fps_val: float,
        timestamp_sec: float
    ):
        """Renders bounding boxes and an information dashboard HUD onto the frame."""
        h, w = frame.shape[:2]

        # Draw worker bounding boxes
        for worker in active_workers:
            x1, y1, x2, y2 = worker.xyxy.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            is_violation = (worker.state == WorkerState.VIOLATION)
            color = (0, 0, 240) if is_violation else (0, 210, 0)
            status_text = "NO HELMET" if is_violation else "HELMET"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw badge background
            label = f"#{worker.track_id} {status_text} ({worker.confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            bg_y1 = max(0, y1 - label_h - 8)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + label_w + 6, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # Draw HUD Dashboard (top-left)
        hud_w, hud_h = 360, 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (25, 25, 25), -1)
        # Apply 75% alpha transparency
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border
        cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), (80, 80, 80), 1)

        # Header
        cv2.putText(
            frame, "HELMET SAFETY MONITOR", (30, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2
        )

        # Stats
        cv2.putText(
            frame, f"Total Workers Tracked: {total_workers}", (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1
        )

        viol_color = (0, 50, 255) if unique_violations > 0 else (0, 220, 0)
        cv2.putText(
            frame, f"Unique Violations: {unique_violations}", (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, viol_color, 2
        )

        time_fps_str = f"Time: {format_timestamp(timestamp_sec)} | {fps_val:.1f} FPS"
        cv2.putText(
            frame, time_fps_str, (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1
        )
