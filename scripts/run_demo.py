"""
CLI Video Demo Runner
Runs helmet safety violation detection on an input video,
draws bounding boxes and HUD overlay, saves timestamped snapshots for unique violations,
and prints the summary report.

Usage:
    python -m scripts.run_demo --video C:/Users/sj165/Downloads/task1.mp4 --model outputs/models/best.pt
"""

import argparse
import json
import os
import sys
import time

from src.detector import create_detector
from src.video_processor import VideoSafetyProcessor


def main():
    parser = argparse.ArgumentParser(description="Run Helmet Safety Violation Detection on Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4, .avi, etc.)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model (.pt or .onnx)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save snapshots and video")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit maximum frames to process (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[!] Error: Video file not found: {args.video}")
        sys.exit(1)

    print(f"============================================================")
    print(f" HELMET SAFETY VIOLATION DETECTION PIPELINE")
    print(f"============================================================")
    print(f"[*] Input Video: {args.video}")
    print(f"[*] Model Engine: {args.model}")
    print(f"[*] Confidence Threshold: {args.conf}")
    print(f"[*] Output Directory: {args.output_dir}")

    # Initialize detector
    print(f"[*] Loading detector...")
    detector = create_detector(args.model, device="cpu")

    # Initialize video processor
    processor = VideoSafetyProcessor(
        detector=detector,
        output_dir=args.output_dir,
        conf_threshold=args.conf
    )

    def print_progress(pct: float, current: int, total: int):
        bar_len = 30
        filled = int(bar_len * (pct / 100.0))
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\rProcessing: [{bar}] {pct:5.1f}% ({current}/{total} frames)")
        sys.stdout.flush()

    print(f"[*] Starting video processing...")
    result = processor.process_video(
        input_video_path=args.video,
        progress_callback=print_progress,
        max_frames=args.max_frames
    )
    print("\n[+] Video processing finished successfully!\n")

    # Display Summary
    summary = result["summary"]
    print("-------------------- SUMMARY REPORT --------------------")
    print(f"  Total Frames Processed:    {summary['total_frames_processed']}")
    print(f"  Video Duration:            {summary['video_duration_sec']} seconds")
    print(f"  Total Workers Detected:    {summary['total_workers_detected']}")
    print(f"  UNIQUE Violations Logged:  {summary['unique_violations']}")
    print(f"  Processing Time:           {summary['processing_time_sec']} seconds")
    print(f"  Average Processing Speed:  {summary['average_processing_fps']} FPS")
    print(f"  Output Annotated Video:    {result['output_video_path']}")
    print(f"  Snapshots Saved:           {len(result['violations'])} images in {processor.snapshots_dir}")
    print("--------------------------------------------------------")

    if result["violations"]:
        print("\nVIOLATION DETAILS:")
        for idx, v in enumerate(result["violations"], 1):
            print(f"  [{idx}] Track #{v['track_id']} at {v['formatted_timestamp']} (Conf: {v['confidence']}) -> {v['snapshot_filename']}")

    # Save JSON report to disk
    report_path = os.path.join(args.output_dir, "reports", f"report_{int(time.time())}.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[+] Full JSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
