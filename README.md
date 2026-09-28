# Helmet Safety Violation Detector: Train, Optimize, Deploy

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end, production-grade Computer Vision and Machine Learning engineering project built for industrial site safety monitoring. The pipeline detects construction workers, tracks individual identities across video streams with **ByteTrack**, and enforces a **temporal debouncing state machine** to ensure that safety violations (workers not wearing hard hats) are recorded **uniquely once per worker**—even when a worker removes their helmet partway through the video.

The service is packaged with an **optimized INT8 ONNX inference engine** tailored for CPU deployment, an **asynchronous FastAPI microservice**, and a **sleek dark-mode Web Dashboard** with real-time video playback and violation snapshot auditing.

---

## 📑 Table of Contents

1. [Key Features](#-key-features)
2. [System Architecture](#-system-architecture)
3. [Quickstart & Running Locally](#-quickstart--running-locally)
   - [One-Click Launch (`run.bat`)](#1-one-click-launch-recommended)
   - [VS Code PowerShell Terminal](#2-vscode-powershell-terminal)
   - [VS Code F5 Debugger](#3-vscode-f5-debug-runner)
4. [Interactive Web Dashboard](#-interactive-web-dashboard)
5. [Part 1: Training & Evaluation](#-part-1-training--evaluation)
   - [Held-Out Test Set Metrics](#held-out-test-set-metrics)
   - [Visual Failure Analysis (5 Cases)](#visual-failure-analysis-5-cases)
6. [Part 2: Model Optimization & CPU Benchmarking](#-part-2-model-optimization--cpu-benchmarking)
   - [CPU Benchmark Comparison](#cpu-benchmark-comparison)
   - [Quantization Trade-off Analysis](#quantization-trade-off-analysis)
7. [Part 3: Video Tracking & State Machine](#-part-3-video-tracking--state-machine)
   - [Unique Violation Counting Logic](#unique-violation-counting-logic)
   - [Dynamic Mid-Stream Helmet Removal](#dynamic-mid-stream-helmet-removal)
   - [Timestamped Snapshot Extraction](#timestamped-snapshot-extraction)
8. [Part 4: Asynchronous FastAPI Service](#-part-4-asynchronous-fastapi-service)
   - [API Endpoints Overview](#api-endpoints-overview)
   - [Sample Request & Response](#sample-request--response)
   - [Disk-Backed Job Persistence](#disk-backed-job-persistence)
9. [What Didn't Work & Future Improvements](#-what-didnt-work--future-improvements)
10. [Video Presentation Guide (3–5 min script)](#-video-presentation-guide)
11. [Author](#-author)

---

## 🚀 Key Features

* **Real-Time Detection & Spatial Tracking**: Fine-tuned YOLOv8 model coupled with ByteTrack bipartite spatial matching.
* **Temporal State Machine & Anti-Flicker Debouncing**: Prevents duplicate violation alerts caused by momentary occlusions or turning heads.
* **Dynamic Mid-Stream Detection**: Accurately flags compliant workers who take off their helmets mid-shift.
* **Automatic Snapshot Capture**: Crops and saves high-resolution, watermarked violation evidence with exact millisecond timestamps.
* **INT8 Quantized ONNX Engine**: Model size compressed from **12.26 MB down to 3.34 MB (72.7% reduction)** while retaining **98.5%** of baseline mAP@50.
* **Asynchronous Non-Blocking API**: Background task queuing prevents server lockups during long video inference runs.
* **Smart Path & Folder Auto-Discovery**: Paste direct paths (e.g. `C:\Users\sj165\Downloads`) or upload files; the system automatically detects, strips quotes, and processes the latest video.
* **Persistent Job Registry**: Job records survive server reloads and restarts via disk persistence (`outputs/jobs/`).

---

## 🏛 System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Video Source Ingestion                   │
                    │  (Web UI Upload / Local File / Folder / REST Endpoint) │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │            Asynchronous Background Worker              │
                    │          (FastAPI BackgroundTasks Queue)               │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          ▼                                           ▼
            ┌───────────────────────────┐               ┌───────────────────────────┐
            │   Optimized ONNX Engine   │               │   ByteTrack Association   │
            │    (Dynamic INT8 Quant)   │──────────────▶│    Multi-Object Tracker   │
            └───────────────────────────┘               └─────────────┬─────────────┘
                                                                      │
                                                                      ▼
                                                        ┌───────────────────────────┐
                                                        │  Temporal State Machine   │
                                                        │   (Anti-Flicker Debounce) │
                                                        └─────────────┬─────────────┘
                                                                      │
                                            ┌─────────────────────────┴─────────────────────────┐
                                            ▼                                                   ▼
                              ┌───────────────────────────┐                       ┌───────────────────────────┐
                              │   Unique Violation Event  │                       │   Compliant Track Event   │
                              │ (Captured ONCE per Worker)│                       │   (Hard Hat Worn in Scene)│
                              └─────────────┬─────────────┘                       └───────────────────────────┘
                                            │
                         ┌──────────────────┴──────────────────┐
                         ▼                                     ▼
           ┌───────────────────────────┐         ┌───────────────────────────┐
           │ Timestamped Snapshot JPEGs│         │   HUD-Annotated MP4 Video │
           │ (Watermarked Evidence)    │         │ (Live BBoxes + Dashboard) │
           └───────────────────────────┘         └───────────────────────────┘
```

---

## ⚡ Quickstart & Running Locally

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.7)
- Windows / macOS / Linux

### 1. Clone & Setup Environment
```powershell
git clone https://github.com/SumitJadhav12/ML-Internship-Task.git
cd ML-Internship-Task

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Application

#### Option A: One-Click Launch (Recommended)
Double-click [`run.bat`](file:///c:/Users/sj165/OneDrive/Desktop/kaddersTask/run.bat) or execute in PowerShell:
```powershell
.\run.bat
```
*This automatically activates `.venv`, opens `http://127.0.0.1:8000/` in your default browser, and launches Uvicorn.*

#### Option B: VS Code PowerShell Terminal
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Option C: VS Code F5 Debug Runner
- Press <kbd>F5</kbd> in VS Code. The configured debug configuration in `.vscode/launch.json` will launch the app with live hot-reload.

### Service URLs
| Component | URL |
| :--- | :--- |
| 🦺 **Live Web Dashboard** | **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** or **[http://localhost:8000/](http://localhost:8000/)** |
| 📖 **Interactive Swagger UI** | **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** |
| 📋 **OpenAPI Specification** | **[http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)** |

---

## 💻 Interactive Web Dashboard

The web dashboard is accessible at `http://127.0.0.1:8000/` and provides:
1. **Interactive Video Profile Tabs**:
   - 🏗️ **Construction Site** (`42926-434300944.mp4`): 19 workers tracked with ByteTrack, 100% compliant.
   - 👷 **Industrial Crew** (`41501-429661287_medium.mp4`): 9 workers tracked, 0 violations.
   - 🚨 **Violations Video** (`task1.mp4`): 2 workers, 2 unique violations logged once per worker.
   - 🎥 **Night Worker** (`39183-421020269.mp4`): Low-light cyclist/worker detection.
2. **Synchronized Evidence Viewer**: Clicking any snapshot in the violation gallery automatically jumps the video player to the exact timestamp of the violation.
3. **Smart Upload & Folder Resolution**:
   - Supports dragging and dropping or browsing video files.
   - Supports entering local folder paths (e.g., `C:\Users\sj165\Downloads`). The system auto-resolves and processes the latest video inside that folder.
   - Automatically cleans quotation marks (`"`, `'`) and whitespace.

---

## 📊 Part 1: Training & Evaluation

### Training Strategy
- **Base Architecture**: `yolov8n` (3.2M parameters) pre-trained on COCO and fine-tuned for PPE safety.
- **Image Resolution**: $640 \times 640$ pixels.
- **Classes**: `0: helmet`, `1: no_helmet`.
- **Augmentation Pipeline**: Mosaic ($p=1.0$), Mixup ($p=0.15$), HSV color-space jitter, and random horizontal flipping.

### Held-Out Test Set Metrics
Evaluated on an isolated test partition ($2,250$ test instances):

| Class | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **helmet** | 1,420 | **0.934** | **0.892** | **0.941** | **0.712** |
| **no_helmet** | 830 | **0.896** | **0.854** | **0.887** | **0.638** |
| **Overall (All Classes)** | 2,250 | **0.915** | **0.873** | **0.914** | **0.675** |

---

### Visual Failure Analysis (5 Cases)

#### 1. Small Scale & Distant Workers (< 20×20 px)
- **Manifestation**: Workers located far downrange in large construction yards are missed.
- **Root Cause**: Successive downsampling in the convolutional backbone (P3–P5 stride 8 to 32) compresses small heads into a single feature cell, washing out hard-hat textures.
- **Mitigation**: Implement **Sliced Aided Hyper Inference (SAHI)** during inference or train with a high-resolution P2 feature pyramid layer.

#### 2. Severe Foreground Occlusion
- **Manifestation**: Workers behind safety nets, scaffolding, or rebar meshes trigger false negatives.
- **Root Cause**: Bounding box IoU with the ground-truth head box drops when the central region of the helmet is obscured.
- **Mitigation**: Employ **CutMix** and **Random Erasing** augmentations to train the network to classify using partial visible helmet contours.

#### 3. Headwear Ambiguity (Caps & Hoodies)
- **Manifestation**: Workers wearing dark baseball caps or beanies may be falsely detected as wearing helmets.
- **Root Cause**: The dome contour and curved brim mimic standard hard-hat geometry under low-contrast lighting.
- **Mitigation**: **Hard-Negative Mining**: Inject annotated samples of workers wearing caps, beanies, and turbans labeled as `no_helmet`.

#### 4. High Dynamic Range & Deep Shadows
- **Manifestation**: Workers moving under concrete slabs or heavy machinery experience detection dropouts.
- **Root Cause**: Dynamic range clipping eliminates edge gradients and reflective strip cues.
- **Mitigation**: Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) preprocessing and expand brightness/exposure augmentation.

#### 5. Motion Blur from Rapid Camera Panning
- **Manifestation**: Quick PTZ camera slewing smears object boundaries, causing transient drops in confidence.
- **Root Cause**: High-frequency edge information is destroyed by shutter exposure during rapid motion.
- **Mitigation**: Train with directional motion blur kernels and leverage multi-frame temporal smoothing.

---

## ⚡ Part 2: Model Optimization & CPU Benchmarking

Benchmark conducted on:
- **Processor**: `AMD Ryzen 5 5500U` (6 cores, 12 threads, 2.10 GHz base up to 4.0 GHz boost)
- **Host RAM**: `16.0 GB`
- **Batch Size**: 1 (Single-frame real-time streaming mode)
- **Runs**: 150 benchmark passes following 20 warmup cycles

### CPU Benchmark Comparison

| Model Architecture / Format | Precision | Model Size | Mean Latency | P95 Latency | Throughput | mAP@50 | Accuracy Retention | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Baseline)** | FP32 | 6.25 MB | 113.2 ms | 134.7 ms | 8.8 FPS | 0.912 | 100.0% (Ref) | Research & Training |
| **ONNX Runtime (Standard)** | FP32 | 12.26 MB | 119.9 ms | 143.2 ms | 8.3 FPS | 0.912 | 100.0% | Standard Cloud CPU |
| **Quantized ONNX (Dynamic)** | INT8 | **3.34 MB** | 188.9 ms | 207.8 ms | 5.3 FPS* | **0.898** | **98.5%** | **Edge / Memory-Constrained** |

*\*Note on CPU Architecture*: Benchmarked on **AMD Ryzen 5 5500U (Zen 2)**. Zen 2 processors lack hardware `AVX-512 VNNI` (Vector Neural Network Instructions). As a result, 8-bit dynamic integer calculations are emulated via software unpack/repack cycles. On modern Intel or embedded edge hardware with native VNNI/DP4A silicon, INT8 achieves 2×–3× throughput speedups.

### Quantization Trade-off Analysis
1. **Memory Footprint**: Model size is reduced from **12.26 MB down to 3.34 MB (72.7% reduction)**. This enables model weights to reside inside fast L3 cache and fit on embedded memory-constrained edge boards.
2. **Accuracy Retention**: Quantization penalty is minimal: mAP@50 drops by only **0.014** ($0.912 \to 0.898$), retaining **98.5%** of baseline accuracy.

---

## 🎯 Part 3: Video Tracking & State Machine

### Unique Violation Counting Logic
Counting raw detections frame-by-frame leads to duplicate alerts (e.g. 30 alerts per second for one unhelmeted worker). To solve this:
1. **ByteTrack Multi-Object Association**: Every worker in the scene is assigned a persistent `track_id` tracked using spatial IoU bipartite matching and Kalman filtering.
2. **Temporal Hysteresis Debouncing**:
   - Each worker maintains a rolling history window of their last 12 frames.
   - A violation is confirmed only when $\ge 60\%$ of recent frames confirm `no_helmet`.
3. **Strict Once-Only Counting**:
   - Once a worker's state transitions to `VIOLATION`, `has_violated` is latched to `True`.
   - The global `unique_violations` counter increments **exactly once**.
   - Subsequent frames update bounding box trails but never duplicate violation records.

### Dynamic Mid-Stream Helmet Removal
When a worker enters wearing a helmet:
- Initially classified as `COMPLIANT` (`[HELMET]`).
- If the worker removes their helmet mid-video (e.g., at $T = 7.33\text{s}$ in `task1.mp4`):
  $$\text{State: } \text{COMPLIANT} \longrightarrow \text{VIOLATION}$$
- The state machine immediately latches the violation, logs the timestamp, extracts a snapshot, and paints the bounding box red.

### Timestamped Snapshot Extraction
For every unique violation event:
- An annotated full-resolution frame crop is saved to `outputs/snapshots/`.
- File naming convention: `violation_track_{track_id}_{timestamp_ms}ms.jpg`.
- Bounding box is highlighted in red, tagged with track ID, confidence, and human-readable time overlay (`HH:MM:SS.mmm`).

---

## 🌐 Part 4: Asynchronous FastAPI Service

### API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/detect-video` | Submits a video for asynchronous non-blocking violation detection. Returns `202 Accepted` with a `job_id`. |
| `GET` | `/api/v1/jobs/{job_id}` | Polls the live status, frames processed, and completion percentage. |
| `GET` | `/api/v1/jobs/{job_id}/report` | Retrieves the final structured compliance report with worker count, unique violations, snapshot URLs, and annotated video link. If still running, returns current progress. |
| `GET` | `/api/v1/latest-report` | Returns the most recent completed compliance report from disk. |
| `GET` | `/static/snapshots/{filename}` | Serves captured violation snapshot JPEGs. |
| `GET` | `/static/videos/{filename}` | Serves rendered HUD-annotated MP4 video streams. |
| `GET` | `/dashboard` | Interactive Web Dashboard. |
| `GET` | `/docs` | Swagger UI documentation with interactive testing. |

### Sample Request & Response

#### 1. Submit Video
```http
POST /api/v1/detect-video
Content-Type: multipart/form-data

file: [binary] OR video_path: "C:\Users\sj165\Downloads"
conf_threshold: 0.20
max_frames: 50
```

**Response (`202 Accepted`)**:
```json
{
  "job_id": "4b553b7f-e0c3-49c5-9900-da0d46f2ee05",
  "status": "pending",
  "message": "Folder detected: Automatically selected latest video '41501-429661287_medium.mp4' from 'C:\\Users\\sj165\\Downloads'.",
  "poll_url": "http://localhost:8000/api/v1/jobs/4b553b7f-e0c3-49c5-9900-da0d46f2ee05",
  "report_url": "http://localhost:8000/api/v1/jobs/4b553b7f-e0c3-49c5-9900-da0d46f2ee05/report",
  "resolved_video_path": "C:\\Users\\sj165\\Downloads\\41501-429661287_medium.mp4"
}
```

#### 2. Retrieve Compliance Report
```http
GET /api/v1/jobs/4b553b7f-e0c3-49c5-9900-da0d46f2ee05/report
```

**Response (`200 OK`)**:
```json
{
  "job_id": "4b553b7f-e0c3-49c5-9900-da0d46f2ee05",
  "status": "completed",
  "progress_percentage": 100.0,
  "frames_processed": 680,
  "total_frames": 680,
  "message": "Video analysis completed successfully. Final structured report and HUD annotated video are available.",
  "summary": {
    "total_frames_processed": 680,
    "video_duration_sec": 27.2,
    "total_workers_detected": 9,
    "unique_violations": 0,
    "processing_time_sec": 308.87,
    "average_processing_fps": 2.2
  },
  "violations": [],
  "output_video_url": "http://localhost:8000/static/videos/annotated_upload_41501-429661287_medium_1790592584.mp4"
}
```

### Disk-Backed Job Persistence
All submitted jobs are backed by JSON records in `outputs/jobs/{job_id}.json`. If the server reloads (e.g. during development or server restarts), job state and results are recovered seamlessly from disk.

---

## 🔍 What Didn't Work & Future Improvements

### What Didn't Work
1. **Static INT8 Quantization without Calibration**: Static post-training quantization caused accuracy degradation in high-contrast lighting. Switching to **Dynamic Quantization (`QUInt8`)** preserved **98.5%** of baseline mAP while shrinking model size by **72.7%**.
2. **Single-Frame Detection Logic**: Counting raw detections led to false positives when a worker briefly looked away from the camera. Introducing a **12-frame hysteresis window** eliminated transient flicker.

### Enterprise Improvements
- **YOLOv8-Pose Association**: Link helmet detections to human body keypoints to maintain identity through prolonged full-head occlusions.
- **SAHI Integration**: Integrate Sliced Aided Hyper Inference for detecting workers in 4K aerial drone streams.
- **Distributed Celery + Redis Cluster**: Move from local background tasks to distributed Celery workers with cloud object storage (AWS S3 / GCS) for enterprise scale.

---

## 🎙 Video Presentation Guide

*(A concise 3–5 minute walk-through script for recording)*

* **0:00 – 0:45 | Introduction**:
  > "Hello, I am Sumit Jadhav. This is my submission for the Helmet Safety Violation Detector assessment. The objective is to detect construction workers without helmets in video, ensure unique violations are counted strictly once per worker, optimize performance on CPU, and deploy as a production-grade FastAPI service."
* **0:45 – 1:45 | Architecture & Violation Logic**:
  > "The pipeline pairs fine-tuned YOLOv8 with ByteTrack for persistent multi-object tracking. To solve the problem of duplicate alerts and handle dynamic events—such as a worker removing their helmet mid-video—we implemented a temporal debouncing state machine. It verifies persistent non-compliance over a rolling window before latching a single unique violation and saving a timestamped snapshot."
* **1:45 – 2:45 | CPU Optimization**:
  > "We exported the model to ONNX and applied Dynamic INT8 Quantization. This reduced memory footprint by 72.7% from 12.26 MB down to 3.34 MB while retaining 98.5% of baseline accuracy, making it ideal for edge CPU deployment."
* **2:45 – 4:00 | Live Demonstration**:
  > "Here is our FastAPI service and dark-mode Web Dashboard. Users can upload video files or paste folder paths like `Downloads`. The system processes videos asynchronously in the background without blocking the UI, renders an annotated MP4 video stream with live bounding boxes and HUD stats, and provides a snapshot gallery for safety audits."
* **4:00 – 4:30 | Conclusion**:
  > "All test suites, benchmarks, and API documentation are fully verified and available in the repository. Thank you for your time and consideration."

---

## 👨‍💻 Author

**Sumit Jadhav**  
*AI / ML Engineer (Computer Vision)*  
*GitHub*: [@SumitJadhav12](https://github.com/SumitJadhav12)  
*Repository*: [ML-Internship-Task](https://github.com/SumitJadhav12/ML-Internship-Task)
