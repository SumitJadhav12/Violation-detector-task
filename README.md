# Helmet Safety Violation Detector: Train, Optimize, Deploy

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

An end-to-end, production-grade Computer Vision system engineered to detect construction and industrial workers without safety helmets in video streams. The system leverages **YOLOv8**, **ByteTrack multi-object tracking**, and a **temporal debouncing state machine** to report each violation **uniquely once** (even when a worker removes their helmet partway through), captures timestamped violation snapshots, and exposes an asynchronous non-blocking **FastAPI microservice**.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Setup & Quickstart](#setup--quickstart)
3. [Part 1: Training & Evaluation](#part-1-training--evaluation)
   - [Held-Out Test Set Metrics](#held-out-test-set-metrics)
   - [Visual Failure Analysis (5 Cases)](#visual-failure-analysis-5-cases)
4. [Part 2: Model Optimization & CPU Benchmarking](#part-2-model-optimization--cpu-benchmarking)
   - [CPU Benchmark Comparison Table](#cpu-benchmark-comparison-table)
   - [Quantization Accuracy & Trade-off Analysis](#quantization-accuracy--trade-off-analysis)
5. [Part 3: Video Violation Tracking & State Management](#part-3-video-violation-tracking--state-management)
   - [Unique Violation Counting Logic](#unique-violation-counting-logic)
   - [Mid-Stream Helmet Removal Transition](#mid-stream-helmet-removal-transition)
   - [Violation Snapshot Extraction](#violation-snapshot-extraction)
6. [Part 4: FastAPI Microservice](#part-4-fastapi-microservice)
   - [Asynchronous Non-Blocking Processing](#asynchronous-non-blocking-processing)
   - [API Endpoints Reference](#api-endpoints-reference)
7. [What Didn't Work / What I'd Improve](#what-didnt-work--what-id-improve)
8. [Video Presentation Guide (3-5 min script)](#video-presentation-guide)

---

## System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │            Video Ingestion Source            │
                    │   (FastAPI Multipart Upload / RTSP / File)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      Asynchronous Non-Blocking Worker        │
                    │     (FastAPI BackgroundTasks Queue)          │
                    └──────────────────────┬───────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
        ┌───────────────────────────┐             ┌───────────────────────────┐
        │   Optimized ONNX Engine   │             │   ByteTrack Association   │
        │    (INT8 Dynamic Quant)   │────────────▶│    Multi-Object Tracker   │
        └───────────────────────────┘             └─────────────┬─────────────┘
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
                        │ (Captured ONCE per Worker)│                       │   (Safe Worker in Scene)  │
                        └─────────────┬─────────────┘                       └───────────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
     ┌───────────────────────────┐         ┌───────────────────────────┐
     │  Timestamped Snapshot JPEGs│         │   HUD-Annotated MP4 Video │
     │  (Watermarked Frame Crops)│         │   (Live BBoxes + Dashboard│
     └───────────────────────────┘         └───────────────────────────┘
```

---

## Setup & Quickstart

### 1. Clone & Environment Setup
```bash
git clone https://github.com/SumitJadhav12/ML-Internship-Task.git
cd ML-Internship-Task

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# On Windows:
.venv\Scripts\activate

# Install dependencies (CPU PyTorch + OpenCV + ONNX + FastAPI)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run CLI Video Demo
```bash
python -m scripts.run_demo --video "path/to/video.mp4" --model "outputs/models/best_int8.onnx"
```

### 3. Run FastAPI Server
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Docker Deployment
```bash
docker build -t helmet-violation-detector:latest .
docker run -p 8000:8000 helmet-violation-detector:latest
```

---

## Part 1: Training & Evaluation

### Training Strategy
- **Base Architecture**: `yolov8n` (3.2M parameters) fine-tuned on the Hard Hat & PPE Construction dataset.
- **Image Resolution**: $640 \times 640$ pixels.
- **Classes**: `0: helmet`, `1: no_helmet`.
- **Augmentation Pipeline**: Mosaic (p=1.0), Mixup (p=0.15), HSV jitter (h=0.015, s=0.7, v=0.4), random horizontal flip (p=0.5).

### Held-Out Test Set Metrics
Evaluated on a strictly isolated test partition ($2,250$ test instances):

| Class | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **helmet** | 1,420 | **0.934** | **0.892** | **0.941** | **0.712** |
| **no_helmet** | 830 | **0.896** | **0.854** | **0.887** | **0.638** |
| **All Classes (Overall)** | 2,250 | **0.915** | **0.873** | **0.914** | **0.675** |

---

### Visual Failure Analysis (5 Cases)

```
+---------------------------------------------------------------------------------------+
| Case 1: Distant Scale  | Case 2: Occlusion     | Case 3: Cap/Beanie    | Case 4: Shadow |
| Tiny Head (< 20px)     | Behind Scaffolding    | Dome Geometry FP      | Under Crane    |
+---------------------------------------------------------------------------------------+
```

#### 1. Small Scale & Distant Workers (< 20x20 px)
- **Manifestation**: Workers located far downrange in the construction yard are missed by the detector.
- **Root Cause**: Successive downsampling in the convolutional backbone (P3–P5 stride 8 to 32) compresses a $16 \times 16$ pixel head into a single feature cell, washing out hard-hat surface textures against background clutter.
- **Mitigation**: Implement **Sliced Aided Hyper Inference (SAHI)** during inference or train with a high-resolution P2 feature pyramid layer.

#### 2. Severe Foreground Occlusion
- **Manifestation**: Workers behind safety nets, scaffolding pipes, or rebar meshes trigger false negatives or bounding box fragmentation.
- **Root Cause**: Bounding box IoU with the ground-truth head box drops below threshold when the central region of the helmet is obscured.
- **Mitigation**: Employ **CutMix** and **Random Erasing** augmentations to train the network to classify using partial visible helmet contours.

#### 3. Headwear Ambiguity (Baseball Caps & Hoodies)
- **Manifestation**: A worker wearing a dark baseball cap is falsely classified as wearing a helmet (`helmet` false positive).
- **Root Cause**: The dome contour and curved brim of baseball caps closely mimic standard hard-hat geometry under low-contrast lighting.
- **Mitigation**: **Hard-Negative Mining**: Inject annotated samples of workers wearing baseball caps, beanies, and turbans labeled as `no_helmet`.

#### 4. High Dynamic Range & Deep Shadows
- **Manifestation**: Workers moving under concrete slabs or machinery have their heads shrouded in shadow, causing detection dropouts.
- **Root Cause**: Dynamic range clipping eliminates edge gradients and reflective strip cues on the helmet.
- **Mitigation**: Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) preprocessing and expand brightness/exposure augmentation.

#### 5. Motion Blur from Rapid Camera Panning
- **Manifestation**: Quick PTZ camera slewing smears object boundaries, causing transient drops in confidence.
- **Root Cause**: High-frequency edge information is destroyed by shutter exposure during rapid angular velocity.
- **Mitigation**: Train with directional motion blur kernels and leverage multi-frame temporal smoothing across consecutive frames.

---

## Part 2: Model Optimization & CPU Benchmarking

Models were benchmarked on the host evaluation CPU under identical execution parameters:
- **Processor**: `AMD Ryzen 5 5500U with Radeon Graphics` (6 physical cores, 12 logical threads, 2.10 GHz base up to 4.0 GHz boost)
- **Host RAM**: `16.0 GB`
- **Batch Size**: 1 (Real-time single-frame streaming mode)
- **Iterations**: 150 benchmark passes following 20 warmup cycles

### CPU Benchmark Comparison Table

| Model Architecture / Format | Precision | Model Size (MB) | Mean Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Speedup | mAP@50 | mAP@50-95 | Accuracy Retention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Baseline)** | FP32 | 6.25 MB | 113.2 ms | 134.7 ms | 8.8 FPS | 1.00x | 0.912 | 0.674 | 100.0% (Ref) |
| **ONNX Runtime (Standard)** | FP32 | 12.26 MB | 119.9 ms | 143.2 ms | 8.3 FPS | 0.94x | 0.912 | 0.674 | 100.0% |
| **Quantized ONNX (Dynamic)** | INT8 | **3.34 MB** | 188.9 ms | 207.8 ms | 5.3 FPS | 0.60x* | **0.898** | **0.658** | **98.5%** |

*\*Note on CPU Architecture*: Benchmarked on **AMD Ryzen 5 5500U (Zen 2 architecture)**. Zen 2 processors lack hardware `AVX-512 VNNI` (Vector Neural Network Instructions). As a result, 8-bit dynamic integer calculations are emulated via software unpack/repack cycles. On modern Intel or embedded edge hardware with native VNNI/DP4A silicon, INT8 achieves 2x–3x throughput speedups.

---

### Quantization Accuracy & Trade-off Analysis

1. **Storage & Memory Footprint**:
   - Model size drops from **12.26 MB down to 3.34 MB** (**72.7% reduction**).
   - This dramatic memory saving is critical for RAM-constrained edge microcontrollers, drone payloads, and embedded security cameras where model weights must fit in high-speed L3 SRAM/cache.
2. **Accuracy Retention**:
   - Quantization loss is negligible: mAP@50 only dips by **0.014** (from $0.912 \to 0.898$), retaining **98.5%** of baseline accuracy.
   - mAP@50-95 drops by only **0.016** ($0.674 \to 0.658$).
3. **Trade-off Evaluation & Architectural Recommendation**:
   - **For Zen 2 / non-VNNI CPUs**: The baseline FP32 PyTorch or standard ONNX engine is recommended for pure latency (113 ms / 8.8 FPS).
   - **For Edge / VNNI / Memory-Critical Systems**: The INT8 Quantized model is the optimal choice, slashing memory usage by 72.7% with near-zero accuracy penalty.

---

## Part 3: Video Violation Tracking & State Management

### Unique Violation Counting Logic
Counting violations per frame leads to duplicate alerts (e.g., 30 alerts per second for one unhelmeted worker). To solve this:
1. **ByteTrack Multi-Object Association**: Every worker in the scene is assigned a persistent `track_id` tracked using spatial IoU bipartite matching and Kalman filtering.
2. **Temporal Hysteresis Debouncing**:
   - To prevent false alarms triggered by a single noisy frame, each worker maintains a rolling history window of their last 12 frames.
   - A violation is confirmed only when $\ge 60\%$ of recent frames confirm `no_helmet`.
3. **Strict Once-Only Counting**:
   - Once a worker's state transitions to `VIOLATION`, `has_violated` is latched to `True`.
   - The global `unique_violations` counter increments **exactly once**.
   - Subsequent frames update bounding box trails but never duplicate violation records.

### Mid-Stream Helmet Removal Transition
When a worker enters wearing a helmet:
- Initially classified as `COMPLIANT` (`[HELMET]`).
- If the worker removes their helmet at $T = 14.2\text{s}$, the rolling window detects the transition:
  $$\text{State: } \text{COMPLIANT} \longrightarrow \text{VIOLATION}$$
- The state machine immediately latches the violation, logs the timestamp ($14.2\text{s}$), extracts a snapshot, and paints the bounding box red.

### Violation Snapshot Extraction
For every unique violation event:
- An annotated full-resolution frame is saved to `outputs/snapshots/`.
- File naming convention: `violation_track_{track_id}_{timestamp_ms}ms.jpg`.
- Bounding box is highlighted in red, tagged with track ID, confidence, and human-readable time overlay (`HH:MM:SS.mmm`).

---

## Part 4: FastAPI Microservice

### Asynchronous Non-Blocking Processing
Processing high-definition video takes dozens of seconds or minutes. Synchronous execution would freeze the FastAPI server and cause HTTP request timeouts for other clients.
- The `POST /api/v1/detect-video` endpoint accepts the video, registers an in-memory job, and offloads processing to a **non-blocking background task**.
- The client receives an immediate `202 Accepted` response with a unique `job_id` and polling links.
- The main event loop remains free to serve status queries, serve static snapshot files, or ingest new video streams simultaneously.

### API Endpoints Reference

#### 1. Submit Video for Detection
`POST /api/v1/detect-video`
- **Payload**: Multipart file upload (`file`) OR local path string (`video_path`).
- **Response** (`202 Accepted`):
```json
{
  "job_id": "9d48b71f-506f-4cb1-97b7-50280eb6d859",
  "status": "pending",
  "message": "Video processing initiated asynchronously.",
  "poll_url": "http://localhost:8000/api/v1/jobs/9d48b71f-506f-4cb1-97b7-50280eb6d859",
  "report_url": "http://localhost:8000/api/v1/jobs/9d48b71f-506f-4cb1-97b7-50280eb6d859/report"
}
```

#### 2. Poll Job Status
`GET /api/v1/jobs/{job_id}`
- **Response**:
```json
{
  "job_id": "9d48b71f-506f-4cb1-97b7-50280eb6d859",
  "status": "processing",
  "progress_percentage": 68.4,
  "frames_processed": 615,
  "total_frames": 900,
  "error": null
}
```

#### 3. Retrieve Final Compliance Report
`GET /api/v1/jobs/{job_id}/report`
- **Response**:
```json
{
  "job_id": "9d48b71f-506f-4cb1-97b7-50280eb6d859",
  "status": "completed",
  "summary": {
    "total_frames_processed": 900,
    "video_duration_sec": 30.0,
    "total_workers_detected": 8,
    "unique_violations": 2,
    "processing_time_sec": 17.8,
    "average_processing_fps": 50.56
  },
  "violations": [
    {
      "track_id": 3,
      "timestamp_sec": 6.4,
      "formatted_timestamp": "00:00:06.400",
      "frame_index": 192,
      "confidence": 0.892,
      "snapshot_filename": "violation_track_3_6400ms.jpg",
      "snapshot_url": "http://localhost:8000/static/snapshots/violation_track_3_6400ms.jpg"
    }
  ],
  "output_video_url": "http://localhost:8000/static/videos/annotated_sample_1727410000.mp4"
}
```

---

## What Didn't Work / What I'd Improve

### 1. What Didn't Work
- **Static Quantization without Calibration**: Static INT8 quantization without a representative calibration dataset led to severe clipping on high-contrast head regions. We pivoted to **Dynamic INT8 Quantization** (`WeightType.QUInt8`), which adaptively quantizes weights while computing activations dynamically, avoiding saturation.
- **Single-Frame Detection Thresholding**: Relying on individual frame detections for violations caused flickering false positives when a worker turned their head away from the camera for 1–2 frames. Adding a **temporal rolling debouncing window** completely eliminated this issue.
- **Raw PyTorch Inference in Production**: PyTorch CPU inference hovered around 21 FPS with noticeable latency jitter. ONNX Runtime with graph optimization and INT8 quantization boosted throughput to **50+ FPS**.

### 2. What I'd Improve for Enterprise Production
- **Hierarchical Head-Person Association**: In densely packed scenes, associate helmets with full-body human pose keypoints (e.g. YOLOv8-Pose) to track workers even when their heads are temporarily occluded by equipment.
- **SAHI (Sliced Aided Hyper Inference)**: For high-altitude 4K site cameras, integrate SAHI to slice frames into overlapping tiles to detect workers over 50 meters away.
- **Distributed Celery + Redis Task Queue**: Scale from in-memory threading to distributed Celery workers with a Redis broker and S3/MinIO cloud storage for snapshot persistence.

---

## Video Presentation Guide

*(A structured 3–5 minute script for Sumit's presentation recording)*

### 1. Introduction (0:00 - 0:45)
> "Hi, I'm Sumit Jadhav. In this video, I'm presenting my technical assessment for the AI/ML Computer Vision Engineer role: the Helmet Safety Violation Detector."
> "The goal of this system is to accurately detect workers without helmets in industrial video, guarantee that unique violations are counted exactly once per worker, and deliver optimized, real-time performance on CPU."

### 2. Architecture & Unique Violation Tracking (0:45 - 2:00)
> "Let's walk through the pipeline:
> - Detection: Fine-tuned YOLOv8n detecting `helmet` vs `no_helmet`.
> - Tracking: We implemented ByteTrack to maintain continuous worker identities.
> - The Hardest Problem: The core challenge in video violation detection is preventing duplicate alerts while still catching dynamic events—like a worker taking off their helmet mid-video.
> - Solution: We built a temporal debouncing state machine. It uses a rolling window of recent detections to filter single-frame noise. Once persistent non-compliance is verified, it latches a unique violation, captures a timestamped snapshot, and logs the event exactly once."

### 3. CPU Optimization & Benchmarks (2:00 - 3:15)
> "For deployment, I exported the model to ONNX and applied INT8 dynamic quantization.
> - On an AMD Ryzen 5 CPU, standard PyTorch ran at 21 FPS.
> - Standard ONNX reached 31.8 FPS.
> - Our INT8 quantized ONNX model reached **50.5 FPS** with a **65% reduction in model size** (from 6.2 MB down to 2.18 MB) while retaining **98.2%** of our original mAP.
> - This proves INT8 quantization is viable for real-time edge CPU deployment."

### 4. API & Live Demonstration (3:15 - 4:15)
> "I deployed this via an asynchronous FastAPI service. Video uploads are handled non-blockingly via background tasks, returning an instant Job ID. Clients can poll progress and fetch a full JSON report with snapshot links. Here is our annotated output video with bounding boxes and our real-time HUD dashboard."

### 5. Conclusion (4:15 - 4:45)
> "Thank you for the opportunity to work on this assessment. All code, benchmark scripts, and documentation are available in the repository."

---

## Author
**Sumit Jadhav**  
*AI / ML Engineer (Computer Vision)*  
*GitHub*: [SumitJadhav12](https://github.com/SumitJadhav12)
