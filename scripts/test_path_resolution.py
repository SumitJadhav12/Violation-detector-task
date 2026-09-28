"""
Test Path Resolution in Video Detection Endpoint
Verifies:
1. Directory path e.g. C:\\Users\\sj165\\Downloads
2. Quoted directory path e.g. "C:\\Users\\sj165\\Downloads"
3. Quoted file path e.g. "C:\\Users\\sj165\\Downloads\\task1.mp4"
4. Non-existent path returns 400 with helpful error
5. Non-video file path returns 400
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from starlette.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def run_tests():
    print("============================================================")
    print(" TESTING VIDEO PATH RESOLUTION & DIRECTORY DETECTION")
    print("============================================================")

    # 1. Folder path: C:\Users\sj165\Downloads
    folder_path = r"C:\Users\sj165\Downloads"
    print(f"\n[*] Submitting folder path: {folder_path}")
    resp1 = client.post("/api/v1/detect-video", data={"video_path": folder_path, "max_frames": 5})
    print(f"    - Status: {resp1.status_code}")
    data1 = resp1.json()
    print(f"    - Message: {data1.get('message')}")
    print(f"    - Resolved: {data1.get('resolved_video_path')}")
    assert resp1.status_code == 202, f"Failed folder path test: {resp1.text}"
    assert data1.get("resolved_video_path") is not None
    assert os.path.exists(data1["resolved_video_path"])

    # 2. Quoted folder path: "C:\Users\sj165\Downloads"
    quoted_folder = f'"{folder_path}"'
    print(f"\n[*] Submitting quoted folder path: {quoted_folder}")
    resp2 = client.post("/api/v1/detect-video", data={"video_path": quoted_folder, "max_frames": 5})
    print(f"    - Status: {resp2.status_code}")
    data2 = resp2.json()
    print(f"    - Message: {data2.get('message')}")
    print(f"    - Resolved: {data2.get('resolved_video_path')}")
    assert resp2.status_code == 202, f"Failed quoted folder test: {resp2.text}"

    # 3. Quoted specific file path: "C:\Users\sj165\Downloads\task1.mp4"
    file_path = os.path.join(folder_path, "task1.mp4")
    if os.path.exists(file_path):
        quoted_file = f'"{file_path}"'
        print(f"\n[*] Submitting quoted file path: {quoted_file}")
        resp3 = client.post("/api/v1/detect-video", data={"video_path": quoted_file, "max_frames": 5})
        print(f"    - Status: {resp3.status_code}")
        data3 = resp3.json()
        print(f"    - Message: {data3.get('message')}")
        print(f"    - Resolved: {data3.get('resolved_video_path')}")
        assert resp3.status_code == 202, f"Failed quoted file test: {resp3.text}"

    # 4. Non-existent path
    fake_path = r"C:\Users\sj165\Downloads\non_existent_video_12345.mp4"
    print(f"\n[*] Submitting non-existent path: {fake_path}")
    resp4 = client.post("/api/v1/detect-video", data={"video_path": fake_path})
    print(f"    - Status: {resp4.status_code}")
    print(f"    - Detail: {resp4.json().get('detail')}")
    assert resp4.status_code == 400, f"Expected 400 for fake path, got {resp4.status_code}"

    # 5. Non-video file path
    non_video = "README.md"
    print(f"\n[*] Submitting non-video file: {non_video}")
    resp5 = client.post("/api/v1/detect-video", data={"video_path": non_video})
    print(f"    - Status: {resp5.status_code}")
    print(f"    - Detail: {resp5.json().get('detail')}")
    assert resp5.status_code == 400, f"Expected 400 for markdown file, got {resp5.status_code}"

    print("\n[+] ALL PATH RESOLUTION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
