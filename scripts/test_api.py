"""
FastAPI Integration Test
Tests:
  1. GET / (health check)
  2. POST /api/v1/detect-video (async submission returning 202 Accepted and job_id)
  3. GET /api/v1/jobs/{job_id} (job progress polling)
  4. GET /api/v1/jobs/{job_id}/report (structured JSON report with snapshots)
  5. GET /static/snapshots/{filename} (snapshot file serving)
"""

import os
import sys
import time
from starlette.testclient import TestClient

# Add workspace to path
sys.path.insert(0, os.path.abspath("."))

from src.app.main import app

client = TestClient(app)


def test_api_workflow():
    print("============================================================")
    print(" FASTAPI ASYNC ENDPOINT VERIFICATION")
    print("============================================================")

    # 1. Health check
    print("[*] Testing GET / ...")
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    print(f"    - Response: {resp.json()}")

    # 2. Async Video Detection Submission
    test_video = "outputs/videos/annotated_task1_1790482696.mp4"
    if not os.path.exists(test_video):
        # Fallback to sample if available
        test_video = r"C:\Users\sj165\Downloads\task1.mp4"

    print(f"[*] Submitting video asynchronously via POST /api/v1/detect-video ({test_video})...")
    post_resp = client.post(
        "/api/v1/detect-video",
        data={"video_path": test_video, "max_frames": 20, "conf_threshold": 0.35}
    )
    assert post_resp.status_code == 202, f"Expected 202, got {post_resp.status_code}: {post_resp.text}"
    job_data = post_resp.json()
    job_id = job_data["job_id"]
    print(f"    - Job created: {job_id}")
    print(f"    - Initial status: {job_data['status']}")
    print(f"    - Poll URL: {job_data['poll_url']}")

    # 3. Poll status
    print(f"[*] Polling job status via GET /api/v1/jobs/{job_id}...")
    completed = False
    for attempt in range(15):
        time.sleep(1.0)
        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        print(f"    - Attempt {attempt+1}: Status={status_data['status']}, Progress={status_data['progress_percentage']}%")
        if status_data["status"] == "completed":
            completed = True
            break
        elif status_data["status"] == "failed":
            raise RuntimeError(f"Job failed: {status_data.get('error')}")

    assert completed, "Job did not complete within timeout"

    # 4. Fetch JSON Report
    print(f"[*] Fetching final report via GET /api/v1/jobs/{job_id}/report...")
    report_resp = client.get(f"/api/v1/jobs/{job_id}/report")
    assert report_resp.status_code == 200
    report_data = report_resp.json()

    print(f"    - Final Report Status: {report_data['status']}")
    print(f"    - Total Workers: {report_data['summary']['total_workers_detected']}")
    print(f"    - Unique Violations: {report_data['summary']['unique_violations']}")
    print(f"    - Violations count in array: {len(report_data['violations'])}")

    # 5. Verify Static File Serving for snapshots
    if report_data["violations"]:
        sample_snap_name = report_data["violations"][0]["snapshot_filename"]
        snap_resp = client.get(f"/static/snapshots/{sample_snap_name}")
        assert snap_resp.status_code == 200, f"Expected 200 for snapshot, got {snap_resp.status_code}"
        print(f"    - Static Snapshot Serving: 200 OK ({len(snap_resp.content)} bytes)")

    print("\n[+] All FastAPI asynchronous endpoints verified successfully!\n")


if __name__ == "__main__":
    test_api_workflow()
