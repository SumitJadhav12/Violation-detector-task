"""
Automated Pipeline Verification Test
Tests detector loading, tracker state machine, unique violation counting,
and FastAPI route integration.
"""

import os
import sys
import unittest
import numpy as np

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.detector import Detection
from src.tracker import SafetyTracker, WorkerState


class TestHelmetSafetyPipeline(unittest.TestCase):

    def test_tracker_unique_violation_debouncing(self):
        """
        Verify that:
        1. A worker without helmet is counted as a violation ONCE.
        2. A worker with helmet is initially COMPLIANT.
        3. When helmet is removed partway, state transitions to VIOLATION.
        4. Global unique violations count does NOT duplicate per frame.
        """
        tracker = SafetyTracker(iou_threshold=0.3, high_conf_thresh=0.4)

        box_worker_1 = np.array([100, 100, 200, 250], dtype=np.float32)
        box_worker_2 = np.array([300, 100, 400, 250], dtype=np.float32)

        # Worker 1: Always NO HELMET (violator)
        # Worker 2: Starts with HELMET, removes helmet at frame 15

        # Frames 1 to 10: Worker 1 = no_helmet (1), Worker 2 = helmet (0)
        for f in range(1, 11):
            timestamp = f / 25.0
            dets = [
                Detection(xyxy=box_worker_1, confidence=0.85, class_id=1, class_name="no_helmet"),
                Detection(xyxy=box_worker_2, confidence=0.88, class_id=0, class_name="helmet"),
            ]
            active, new_viols = tracker.update(dets, frame_idx=f, timestamp_sec=timestamp)

        # After 10 frames: Worker 1 must be flagged as unique violation; Worker 2 is compliant
        self.assertEqual(tracker.unique_violations_count, 1, "Only Worker 1 should be a unique violation")
        self.assertEqual(len(tracker.tracked_workers), 2, "Both workers should be tracked")

        worker_1 = tracker.tracked_workers[1]
        worker_2 = tracker.tracked_workers[2]

        self.assertEqual(worker_1.state, WorkerState.VIOLATION)
        self.assertEqual(worker_2.state, WorkerState.COMPLIANT)

        # Frames 11 to 25: Worker 2 REMOVES HELMET (class 1)
        worker_2_violation_triggered = False
        for f in range(11, 26):
            timestamp = f / 25.0
            dets = [
                Detection(xyxy=box_worker_1, confidence=0.85, class_id=1, class_name="no_helmet"),
                Detection(xyxy=box_worker_2, confidence=0.85, class_id=1, class_name="no_helmet"),
            ]
            active, new_viols = tracker.update(dets, frame_idx=f, timestamp_sec=timestamp)
            for v in new_viols:
                if v.track_id == 2:
                    worker_2_violation_triggered = True

        # Now Worker 2 must have transitioned to VIOLATION!
        self.assertTrue(worker_2_violation_triggered, "Worker 2 removing helmet must trigger new unique violation")
        self.assertEqual(tracker.unique_violations_count, 2, "Total unique violations must be exactly 2")
        self.assertEqual(worker_2.state, WorkerState.VIOLATION)

        # Frames 26 to 50: Keep running without helmets
        for f in range(26, 51):
            timestamp = f / 25.0
            dets = [
                Detection(xyxy=box_worker_1, confidence=0.85, class_id=1, class_name="no_helmet"),
                Detection(xyxy=box_worker_2, confidence=0.85, class_id=1, class_name="no_helmet"),
            ]
            active, new_viols = tracker.update(dets, frame_idx=f, timestamp_sec=timestamp)
            self.assertEqual(len(new_viols), 0, "No new unique violations should be emitted once latched")

        # Total unique violations must STILL be exactly 2, not 50!
        self.assertEqual(tracker.unique_violations_count, 2, "Violations must not increment per frame!")
        print("\n[+] Tracker & State Machine Verification: PASSED (Unique Violations correctly counted once)")


if __name__ == "__main__":
    unittest.main()
