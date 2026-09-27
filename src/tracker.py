"""
Object Tracking & Temporal Violation State Machine
Implements ByteTrack association and a robust temporal debouncing state machine
to guarantee that unique violations are counted exactly once per worker,
even when a worker removes their helmet partway through the video.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque
import numpy as np

from src.detector import Detection


class WorkerState(str, Enum):
    UNKNOWN = "UNKNOWN"
    COMPLIANT = "COMPLIANT"       # Wearing helmet
    VIOLATION = "VIOLATION"       # Not wearing helmet


@dataclass
class TrackedWorker:
    """Maintains trajectory and compliance state for a single worker across time."""
    track_id: int
    xyxy: np.ndarray
    confidence: float
    state: WorkerState = WorkerState.UNKNOWN
    
    # Temporal history for debouncing
    history_window: Deque[int] = field(default_factory=lambda: deque(maxlen=12))
    
    # State flags
    has_violated: bool = False
    violation_timestamp_sec: Optional[float] = None
    violation_frame_idx: Optional[int] = None
    snapshot_path: Optional[str] = None
    
    # Liveness
    last_seen_frame: int = 0
    total_detections: int = 0
    consecutive_invisible_frames: int = 0

    def update_detection(
        self,
        xyxy: np.ndarray,
        class_id: int,
        conf: float,
        frame_idx: int,
        timestamp_sec: float
    ) -> bool:
        """
        Updates worker tracking state with new frame detection.
        Returns True if this detection marks a NEW UNIQUE VIOLATION event.
        """
        self.xyxy = xyxy
        self.confidence = conf
        self.last_seen_frame = frame_idx
        self.total_detections += 1
        self.consecutive_invisible_frames = 0

        # Append class: 0 = helmet, 1 = no_helmet
        self.history_window.append(class_id)

        # Debounce logic: calculate violation ratio in recent history
        no_helmet_count = sum(1 for c in self.history_window if c == 1)
        helmet_count = len(self.history_window) - no_helmet_count

        is_new_unique_violation = False

        # If majority of recent frames indicate no-helmet (with minimum evidence window)
        if len(self.history_window) >= 4 and no_helmet_count >= (len(self.history_window) * 0.6):
            if not self.has_violated:
                # First time violation triggered for this worker!
                self.has_violated = True
                self.state = WorkerState.VIOLATION
                self.violation_timestamp_sec = timestamp_sec
                self.violation_frame_idx = frame_idx
                is_new_unique_violation = True
            else:
                self.state = WorkerState.VIOLATION
        elif helmet_count >= (len(self.history_window) * 0.6):
            if not self.has_violated:
                self.state = WorkerState.COMPLIANT

        return is_new_unique_violation


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class SafetyTracker:
    """
    ByteTrack-inspired multi-object tracker for workers with compliance state management.
    Performs two-stage association (high-confidence matches followed by low-confidence matches).
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_lost_frames: int = 30,
        high_conf_thresh: float = 0.50
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.high_conf_thresh = high_conf_thresh

        self.next_track_id: int = 1
        self.tracked_workers: Dict[int, TrackedWorker] = {}
        self.unique_violations_count: int = 0
        self.all_time_workers_count: int = 0

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        timestamp_sec: float
    ) -> Tuple[List[TrackedWorker], List[TrackedWorker]]:
        """
        Process detections for current frame.
        Returns:
            - active_tracks: all actively tracked workers in this frame
            - new_violations: list of workers that newly triggered a unique violation
        """
        # Separate high-confidence and low-confidence detections
        high_dets = [d for d in detections if d.confidence >= self.high_conf_thresh]
        low_dets = [d for d in detections if d.confidence < self.high_conf_thresh]

        unmatched_tracks = list(self.tracked_workers.keys())
        matched_tracks: List[int] = []
        new_violations: List[TrackedWorker] = []

        # Stage 1: Associate existing tracks with high-confidence detections
        if high_dets and unmatched_tracks:
            matches, unmatched_high_dets, unmatched_tracks = self._associate(
                unmatched_tracks, high_dets
            )
            for track_id, det_idx in matches:
                det = high_dets[det_idx]
                worker = self.tracked_workers[track_id]
                is_new = worker.update_detection(
                    det.xyxy, det.class_id, det.confidence, frame_idx, timestamp_sec
                )
                if is_new:
                    self.unique_violations_count += 1
                    new_violations.append(worker)
                matched_tracks.append(track_id)
        else:
            unmatched_high_dets = list(range(len(high_dets)))

        # Stage 2: Associate remaining tracks with low-confidence detections
        if low_dets and unmatched_tracks:
            matches, _, unmatched_tracks = self._associate(
                unmatched_tracks, low_dets
            )
            for track_id, det_idx in matches:
                det = low_dets[det_idx]
                worker = self.tracked_workers[track_id]
                is_new = worker.update_detection(
                    det.xyxy, det.class_id, det.confidence, frame_idx, timestamp_sec
                )
                if is_new:
                    self.unique_violations_count += 1
                    new_violations.append(worker)
                matched_tracks.append(track_id)

        # Stage 3: Initiate new tracks for unmatched high-confidence detections
        for det_idx in unmatched_high_dets:
            det = high_dets[det_idx]
            track_id = self.next_track_id
            self.next_track_id += 1
            self.all_time_workers_count += 1

            new_worker = TrackedWorker(
                track_id=track_id,
                xyxy=det.xyxy,
                confidence=det.confidence
            )
            is_new = new_worker.update_detection(
                det.xyxy, det.class_id, det.confidence, frame_idx, timestamp_sec
            )
            if is_new:
                self.unique_violations_count += 1
                new_violations.append(new_worker)

            self.tracked_workers[track_id] = new_worker

        # Stage 4: Increment lost frames and prune stale tracks
        dead_tracks = []
        for track_id in unmatched_tracks:
            worker = self.tracked_workers[track_id]
            worker.consecutive_invisible_frames += 1
            if worker.consecutive_invisible_frames > self.max_lost_frames:
                dead_tracks.append(track_id)

        for track_id in dead_tracks:
            del self.tracked_workers[track_id]

        active_workers = [
            w for w in self.tracked_workers.values()
            if w.consecutive_invisible_frames == 0
        ]
        return active_workers, new_violations

    def _associate(
        self,
        track_ids: List[int],
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU bipartite matching between tracks and detections."""
        if not track_ids or not detections:
            return [], list(range(len(detections))), track_ids

        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for t_idx, t_id in enumerate(track_ids):
            track_box = self.tracked_workers[t_id].xyxy
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = calculate_iou(track_box, det.xyxy)

        matched_tracks = set()
        matched_dets = set()
        matches = []

        # Sort matches by highest IoU descending
        t_indices, d_indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
        for t_idx, d_idx in zip(t_indices, d_indices):
            iou_val = iou_matrix[t_idx, d_idx]
            if iou_val < self.iou_threshold:
                break
            if t_idx not in matched_tracks and d_idx not in matched_dets:
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                matches.append((track_ids[t_idx], d_idx))

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        unmatched_tracks = [track_ids[t] for t in range(len(track_ids)) if t not in matched_tracks]

        return matches, unmatched_dets, unmatched_tracks
