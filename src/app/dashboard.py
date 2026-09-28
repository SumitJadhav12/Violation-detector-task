"""
Interactive Web Dashboard for Helmet Safety Violation Detection
Sleek, dark-mode, glassmorphism UI for monitoring video streams,
viewing annotated footage, inspecting unique violation snapshots, and triggering detection jobs.
"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Helmet Safety Violation Detector | Live AI Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --surface: rgba(18, 24, 38, 0.75);
      --surface-border: rgba(255, 255, 255, 0.08);
      --surface-hover: rgba(26, 35, 55, 0.85);
      --primary: #3b82f6;
      --primary-glow: rgba(59, 130, 246, 0.35);
      --accent: #06b6d4;
      --danger: #ef4444;
      --danger-glow: rgba(239, 68, 68, 0.35);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.3);
      --warning: #f59e0b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(239, 68, 68, 0.05) 0px, transparent 50%);
      background-attachment: fixed;
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      line-height: 1.5;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }

    /* HEADER */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--surface);
      backdrop-filter: blur(16px);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-lg);
      margin-bottom: 28px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .logo-badge {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
      box-shadow: 0 0 20px var(--danger-glow);
    }

    .brand h1 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .brand p {
      font-size: 13px;
      color: var(--text-muted);
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 500;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.25);
      color: #34d399;
    }

    .pill-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(0.85); }
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 18px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
      font-family: inherit;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--primary) 0%, #2563eb 100%);
      color: #fff;
      box-shadow: 0 4px 14px var(--primary-glow);
    }

    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px var(--primary-glow);
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      color: var(--text);
      border: 1px solid var(--surface-border);
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
    }

    /* SELECTOR TABS */
    .tab-bar {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
    }

    .tab-btn {
      padding: 8px 16px;
      border-radius: var(--radius-sm);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
      transition: all 0.2s;
    }

    .tab-btn.active {
      background: rgba(59, 130, 246, 0.2);
      border-color: var(--primary);
      color: #60a5fa;
    }

    /* STATS GRID */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }

    .stat-card {
      background: var(--surface);
      backdrop-filter: blur(12px);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-md);
      padding: 20px;
      position: relative;
      overflow: hidden;
    }

    .stat-card::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--primary), var(--accent));
    }

    .stat-card.danger::after {
      background: linear-gradient(90deg, #ef4444, #f97316);
    }

    .stat-card.success::after {
      background: linear-gradient(90deg, #10b981, #06b6d4);
    }

    .stat-label {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .stat-val {
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .stat-sub {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 6px;
    }

    /* MAIN CONTENT SPLIT */
    .main-grid {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 24px;
      margin-bottom: 28px;
    }

    @media (max-width: 1024px) {
      .main-grid {
        grid-template-columns: 1fr;
      }
    }

    .card {
      background: var(--surface);
      backdrop-filter: blur(14px);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-lg);
      padding: 24px;
      position: relative;
    }

    .card-title {
      font-size: 17px;
      font-weight: 700;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .card-title span.tag {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 9999px;
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* VIDEO VIEWER */
    .video-wrapper {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: var(--radius-md);
      overflow: hidden;
      aspect-ratio: 16 / 9;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .video-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 14px;
      font-size: 13px;
      color: var(--text-muted);
    }

    /* VIOLATION GALLERY */
    .violations-list {
      display: flex;
      flex-direction: column;
      gap: 14px;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 6px;
    }

    .violations-list::-webkit-scrollbar {
      width: 6px;
    }

    .violations-list::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 3px;
    }

    .violation-item {
      display: flex;
      gap: 14px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(239, 68, 68, 0.25);
      border-radius: var(--radius-md);
      transition: all 0.2s ease;
      cursor: pointer;
    }

    .violation-item:hover {
      background: rgba(239, 68, 68, 0.08);
      border-color: rgba(239, 68, 68, 0.45);
      transform: translateX(3px);
    }

    .violation-thumb {
      width: 110px;
      height: 75px;
      border-radius: var(--radius-sm);
      object-fit: cover;
      background: #1e293b;
      flex-shrink: 0;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .violation-info {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }

    .viol-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 4px;
    }

    .viol-tag {
      font-size: 11px;
      font-weight: 700;
      color: #ef4444;
      background: rgba(239, 68, 68, 0.15);
      padding: 2px 8px;
      border-radius: 4px;
      letter-spacing: 0.04em;
    }

    .viol-time {
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      font-weight: 600;
      color: #fca5a5;
    }

    .viol-desc {
      font-size: 12px;
      color: var(--text-muted);
    }

    /* COMPLIANT BADGE */
    .compliant-box {
      padding: 32px 20px;
      text-align: center;
      background: rgba(16, 185, 129, 0.05);
      border: 1px dashed rgba(16, 185, 129, 0.3);
      border-radius: var(--radius-md);
    }

    .compliant-icon {
      font-size: 44px;
      margin-bottom: 10px;
    }

    /* RUN DETECTION FORM */
    .form-card {
      margin-bottom: 28px;
    }

    .form-grid {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr auto;
      gap: 16px;
      align-items: end;
    }

    @media (max-width: 900px) {
      .form-grid {
        grid-template-columns: 1fr;
      }
    }

    .form-group label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }

    .input-field {
      width: 100%;
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--surface-border);
      color: #fff;
      font-family: inherit;
      font-size: 14px;
      outline: none;
      transition: border-color 0.2s;
    }

    .input-field:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px var(--primary-glow);
    }

    .chips-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .chip-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      border-radius: 9999px;
      padding: 3px 10px;
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }

    .chip-btn:hover {
      background: rgba(59, 130, 246, 0.15);
      border-color: var(--primary);
      color: #93c5fd;
    }

    /* PROGRESS BAR */
    .progress-box {
      margin-top: 18px;
      display: none;
    }

    .progress-bar-bg {
      width: 100%;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 9999px;
      overflow: hidden;
    }

    .progress-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--primary), var(--accent));
      transition: width 0.3s ease;
    }

    .progress-status {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 6px;
      font-family: 'JetBrains Mono', monospace;
    }

    /* BENCHMARK TABLE */
    .bench-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }

    .bench-table th {
      text-align: left;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--surface-border);
    }

    .bench-table td {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .bench-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .badge-opt {
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
    }

    /* LIGHTBOX MODAL */
    .modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.85);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 20px;
    }

    .modal-content {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-lg);
      max-width: 900px;
      width: 100%;
      overflow: hidden;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
    }

    .modal-img {
      width: 100%;
      max-height: 550px;
      object-fit: contain;
      background: #000;
      display: block;
    }

    .modal-footer {
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- HEADER -->
    <header>
      <div class="brand">
        <div class="logo-badge">🦺</div>
        <div>
          <h1>Helmet Safety Violation Detector</h1>
          <p>Production AI Vision Pipeline • YOLOv8 + ByteTrack + INT8 ONNX Engine</p>
        </div>
      </div>
      <div class="nav-actions">
        <div class="pill">
          <div class="pill-dot"></div>
          <span>System Online: CPU Mode</span>
        </div>
        <a href="/docs" target="_blank" class="btn btn-secondary">Swagger API Docs</a>
        <a href="https://github.com/SumitJadhav12/ML-Internship-Task" target="_blank" class="btn btn-secondary">GitHub Repo</a>
      </div>
    </header>

    <!-- VIDEO SELECTION TABS -->
    <div class="tab-bar">
      <button class="tab-btn active" id="tab-construction" onclick="switchVideo('construction')">
        🏗️ Construction Site: 42926-434300944.mp4 (19 Workers)
      </button>
      <button class="tab-btn" id="tab-medium" onclick="switchVideo('medium')">
        👷 Industrial Crew: 41501-429661287.mp4 (9 Workers, Compliant)
      </button>
      <button class="tab-btn" id="tab-task1" onclick="switchVideo('task1')">
        🚨 Violations Video: task1.mp4 (2 Workers, 2 Violations)
      </button>
      <button class="tab-btn" id="tab-uploaded" onclick="switchVideo('uploaded')">
        🎥 Night Worker: 39183-421020269.mp4 (1 Worker)
      </button>
    </div>

    <!-- TOP KPI STATS -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Total Frames Processed <span>⏱️</span></div>
        <div class="stat-val" id="stat-frames">135</div>
        <div class="stat-sub" id="stat-duration">Full duration: 5.62s @ 24 FPS</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Workers Tracked <span>👷</span></div>
        <div class="stat-val" id="stat-workers">1</div>
        <div class="stat-sub">ByteTrack unique spatial identities</div>
      </div>
      <div class="stat-card danger">
        <div class="stat-label">Unique Violations <span>🚨</span></div>
        <div class="stat-val" id="stat-violations" style="color: #10b981;">0</div>
        <div class="stat-sub" id="stat-viol-sub">100% Compliant (All Helmets Worn)</div>
      </div>
      <div class="stat-card success">
        <div class="stat-label">Model Memory Size <span>⚡</span></div>
        <div class="stat-val" id="stat-size" style="color: #34d399;">3.34 MB</div>
        <div class="stat-sub">72.7% reduction via INT8 Quantization</div>
      </div>
    </div>

    <!-- MAIN GRID: VIDEO & VIOLATIONS -->
    <div class="main-grid">
      <!-- VIDEO VIEWER -->
      <div class="card">
        <div class="card-title">
          <span>HUD Annotated Video Stream</span>
          <span class="tag">ByteTrack Spatial Matching</span>
        </div>
        <div class="video-wrapper">
          <video id="annotated-video" controls preload="metadata">
            <source id="video-source" src="/static/videos/annotated_39183-421020269_1790571786.mp4" type="video/mp4">
            Your browser does not support HTML5 video.
          </video>
        </div>
        <div class="video-meta">
          <span id="video-filename">Source: annotated_39183-421020269_1790571786.mp4</span>
          <a id="video-download-btn" href="/static/videos/annotated_39183-421020269_1790571786.mp4" download class="btn btn-secondary" style="padding: 4px 12px; font-size: 12px;">Download Annotated Video</a>
        </div>
      </div>

      <!-- VIOLATION EVENT GALLERY -->
      <div class="card">
        <div class="card-title">
          <span>Captured Unique Violations</span>
          <span class="tag" style="background: rgba(239,68,68,0.15); color: #f87171; border-color: rgba(239,68,68,0.3);">Timestamped Crops</span>
        </div>
        <div class="violations-list" id="violations-container">
          <!-- Populated by JavaScript -->
        </div>
      </div>
    </div>

    <!-- ASYNC VIDEO SUBMISSION FORM -->
    <div class="card form-card">
      <div class="card-title">
        <span>Run Asynchronous Video Detection</span>
        <span class="tag">Non-Blocking FastAPI Worker</span>
      </div>
      <form id="detection-form" onsubmit="submitDetection(event)">
        <div class="form-grid">
          <div class="form-group">
            <label for="videoPath">Video Path / Folder or Browse File</label>
            <div style="display: flex; gap: 8px;">
              <input type="text" id="videoPath" class="input-field" value="C:\Users\sj165\Downloads" placeholder="e.g. C:\Users\sj165\Downloads or direct .mp4 path" oninput="onPathTyped()">
              <input type="file" id="filePicker" accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/*" style="display: none;" onchange="handleFilePicked(event)">
              <button type="button" class="btn btn-secondary" onclick="document.getElementById('filePicker').click()" style="white-space: nowrap; padding: 0 16px; font-size: 13px;" title="Browse video file from computer">
                📁 Browse
              </button>
            </div>
            <div class="chips-row">
              <span style="font-size: 11px; color: var(--text-muted); align-self: center;">Quick Select:</span>
              <button type="button" class="chip-btn" onclick="setPath('C:\\Users\\sj165\\Downloads')">📁 Downloads Folder</button>
              <button type="button" class="chip-btn" onclick="setPath('outputs/uploads/42926-434300944.mp4')">🏗️ Construction Site</button>
              <button type="button" class="chip-btn" onclick="setPath('C:\\Users\\sj165\\Downloads\\task1.mp4')">🚨 Violations (task1.mp4)</button>
              <button type="button" class="chip-btn" onclick="setPath('outputs/uploads/39183-421020269.mp4')">🌙 Night Worker</button>
            </div>
          </div>
          <div class="form-group">
            <label for="confThresh">Confidence Threshold</label>
            <input type="number" id="confThresh" class="input-field" step="0.05" min="0.05" max="0.95" value="0.20">
          </div>
          <div class="form-group">
            <label for="maxFrames">Max Frames (0 for all)</label>
            <input type="number" id="maxFrames" class="input-field" value="0" placeholder="e.g. 300 for quick test">
          </div>
          <div class="form-group">
            <button type="submit" id="submit-btn" class="btn btn-primary" style="height: 42px; width: 100%;">
              <span>▶ Run Detection</span>
            </button>
          </div>
        </div>
      </form>

      <!-- PROGRESS BAR -->
      <div class="progress-box" id="progress-box">
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="progress-fill"></div>
        </div>
        <div class="progress-status">
          <span id="progress-text">Initializing pipeline...</span>
          <span id="progress-pct">0%</span>
        </div>
      </div>
    </div>

    <!-- BENCHMARKS & ARCHITECTURE COMPARISON -->
    <div class="card" style="margin-bottom: 28px;">
      <div class="card-title">
        <span>CPU Optimization Benchmark (AMD Ryzen 5 5500U)</span>
        <span class="badge-opt">Quantization Trade-Off Analysis</span>
      </div>
      <div style="overflow-x: auto;">
        <table class="bench-table">
          <thead>
            <tr>
              <th>Architecture / Engine</th>
              <th>Precision</th>
              <th>Model Size</th>
              <th>Mean Latency</th>
              <th>Throughput</th>
              <th>mAP@50</th>
              <th>Accuracy Retention</th>
              <th>Recommended Use Case</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>PyTorch Baseline</strong></td>
              <td>FP32</td>
              <td>6.25 MB</td>
              <td>113.2 ms</td>
              <td>8.8 FPS</td>
              <td>0.912</td>
              <td>100.0% (Ref)</td>
              <td>Research & Training</td>
            </tr>
            <tr>
              <td><strong>ONNX Runtime</strong></td>
              <td>FP32</td>
              <td>12.26 MB</td>
              <td>119.9 ms</td>
              <td>8.3 FPS</td>
              <td>0.912</td>
              <td>100.0%</td>
              <td>Standard Cloud Inference</td>
            </tr>
            <tr style="background: rgba(16, 185, 129, 0.05);">
              <td><strong>Dynamic Quantized ONNX</strong></td>
              <td>INT8</td>
              <td><strong style="color: #34d399;">3.34 MB</strong> (-72.7%)</td>
              <td>188.9 ms</td>
              <td>5.3 FPS (50+ on VNNI)</td>
              <td><strong>0.898</strong></td>
              <td><strong style="color: #34d399;">98.5%</strong></td>
              <td><span class="badge-opt">Edge / Memory-Critical</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- LIGHTBOX MODAL -->
  <div class="modal-overlay" id="imageModal" onclick="closeModal()">
    <div class="modal-content" onclick="event.stopPropagation()">
      <img id="modalImg" src="" alt="Snapshot" class="modal-img">
      <div class="modal-footer">
        <div>
          <h4 id="modalTitle" style="font-size: 15px; font-weight: 700;">Violation Snapshot</h4>
          <p id="modalSub" style="font-size: 12px; color: var(--text-muted);">Timestamped Watermarked Evidence Crop</p>
        </div>
        <div style="display: flex; gap: 8px;">
          <a id="modalDownload" href="" download class="btn btn-primary" style="font-size: 12px; padding: 6px 14px;">Download Full Snapshot</a>
          <button onclick="closeModal()" class="btn btn-secondary" style="font-size: 12px; padding: 6px 14px;">Close</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    const videoElem = document.getElementById('annotated-video');

    const VIDEO_PROFILES = {
      construction: {
        frames: '490',
        duration: 'Full duration: 16.35s @ 30 FPS',
        workers: '19',
        violations: '0',
        violColor: '#10b981',
        violSub: '100% Compliant (19 Hard Hats Tracked)',
        videoUrl: '/static/videos/annotated_42926-434300944_1790573945.mp4',
        filename: 'Source: annotated_42926-434300944_1790573945.mp4',
        inputPath: 'outputs/uploads/42926-434300944.mp4',
        violationsList: []
      },
      task1: {
        frames: '2,690',
        duration: 'Full duration: 89.67s @ 30 FPS',
        workers: '2',
        violations: '2',
        violColor: '#ef4444',
        violSub: 'Counted once per worker (Debounced)',
        videoUrl: '/static/videos/annotated_task1_1790492077.mp4',
        filename: 'Source: annotated_task1_1790492077.mp4',
        inputPath: 'C:/Users/sj165/Downloads/task1.mp4',
        violationsList: [
          {
            track_id: 1,
            time: '00:00:00.100',
            timeSec: 0.1,
            frame: 4,
            conf: 56.0,
            img: '/static/snapshots/violation_track_1_100ms.jpg',
            desc: 'Initial entry without safety helmet'
          },
          {
            track_id: 2,
            time: '00:00:07.333',
            timeSec: 7.33,
            frame: 221,
            conf: 64.4,
            img: '/static/snapshots/violation_track_2_7333ms.jpg',
            desc: 'Worker moving across scene without helmet'
          }
        ]
      },
      uploaded: {
        frames: '135',
        duration: 'Full duration: 5.62s @ 24 FPS',
        workers: '1',
        violations: '0',
        violColor: '#10b981',
        violSub: '100% Compliant (All Helmets Worn)',
        videoUrl: '/static/videos/annotated_39183-421020269_1790571786.mp4',
        filename: 'Source: annotated_39183-421020269_1790571786.mp4',
        inputPath: 'outputs/uploads/39183-421020269.mp4',
        violationsList: []
      },
      medium: {
        frames: '680',
        duration: 'Full duration: 27.20s @ 25 FPS (2K / 1440p)',
        workers: '9',
        violations: '0',
        violColor: '#10b981',
        violSub: '100% Compliant (All 9 Hard Hats Tracked)',
        videoUrl: '/static/videos/annotated_upload_41501-429661287_medium_1790592584.mp4',
        filename: 'Source: annotated_upload_41501-429661287_medium_1790592584.mp4',
        inputPath: 'C:\\Users\\sj165\\Downloads\\41501-429661287_medium.mp4',
        violationsList: []
      }
    };

    function switchVideo(profileKey) {
      document.getElementById('tab-construction').classList.toggle('active', profileKey === 'construction');
      document.getElementById('tab-medium').classList.toggle('active', profileKey === 'medium');
      document.getElementById('tab-task1').classList.toggle('active', profileKey === 'task1');
      document.getElementById('tab-uploaded').classList.toggle('active', profileKey === 'uploaded');

      const p = VIDEO_PROFILES[profileKey];
      document.getElementById('stat-frames').innerText = p.frames;
      document.getElementById('stat-duration').innerText = p.duration;
      document.getElementById('stat-workers').innerText = p.workers;
      document.getElementById('stat-violations').innerText = p.violations;
      document.getElementById('stat-violations').style.color = p.violColor;
      document.getElementById('stat-viol-sub').innerText = p.violSub;

      videoElem.src = p.videoUrl;
      videoElem.load();
      document.getElementById('video-filename').innerText = p.filename;
      document.getElementById('video-download-btn').href = p.videoUrl;
      document.getElementById('videoPath').value = p.inputPath;

      renderViolations(p.violationsList);
    }

    function renderViolations(list) {
      const container = document.getElementById('violations-container');
      container.innerHTML = '';

      if (!list || list.length === 0) {
        container.innerHTML = `
          <div class="compliant-box">
            <div class="compliant-icon">✅</div>
            <h4 style="font-size: 15px; font-weight: 700; color: #34d399; margin-bottom: 6px;">100% Safety Compliance</h4>
            <p style="font-size: 13px; color: var(--text-muted);">All detected workers in this stream are properly wearing safety helmets with zero violations logged.</p>
          </div>
        `;
        return;
      }

      list.forEach(v => {
        const item = document.createElement('div');
        item.className = 'violation-item';
        item.onclick = () => openModal(v.img, `Track #${v.track_id}: ${v.time} (Conf: ${v.conf}%)`, v.timeSec);
        item.innerHTML = `
          <img src="${v.img}" alt="Violation" class="violation-thumb">
          <div class="violation-info">
            <div class="viol-header">
              <span class="viol-tag">TRACK #${v.track_id}</span>
              <span class="viol-time">${v.time}</span>
            </div>
            <div class="viol-desc">Frame #${v.frame} • Conf: ${v.conf}% • ${v.desc}</div>
            <div style="font-size: 11px; color: #60a5fa; margin-top: 4px;">▶ Click to jump video & view</div>
          </div>
        `;
        container.appendChild(item);
      });
    }

    function openModal(imgSrc, title, timeSec) {
      document.getElementById('modalImg').src = imgSrc;
      document.getElementById('modalTitle').innerText = title;
      document.getElementById('modalDownload').href = imgSrc;
      document.getElementById('imageModal').style.display = 'flex';

      if (timeSec !== undefined && videoElem) {
        videoElem.currentTime = Math.max(0, timeSec - 0.5);
        videoElem.play();
      }
    }

    function closeModal() {
      document.getElementById('imageModal').style.display = 'none';
    }

    // Initialize with construction video profile
    switchVideo('construction');

    // File picker and shortcut helpers
    let selectedUploadFile = null;

    function handleFilePicked(event) {
      const file = event.target.files[0];
      if (file) {
        selectedUploadFile = file;
        document.getElementById('videoPath').value = `[Browse Upload]: ${file.name}`;
      }
    }

    function onPathTyped() {
      selectedUploadFile = null;
    }

    function setPath(p) {
      selectedUploadFile = null;
      document.getElementById('videoPath').value = p;
    }

    // Async Detection Trigger & Polling
    let pollInterval = null;

    async function submitDetection(e) {
      e.preventDefault();
      const rawPath = document.getElementById('videoPath').value.trim();
      const conf = parseFloat(document.getElementById('confThresh').value);
      const maxFramesVal = parseInt(document.getElementById('maxFrames').value);

      if (!selectedUploadFile && !rawPath) {
        alert('Please enter a video or folder path, or browse to select a file.');
        return;
      }

      const submitBtn = document.getElementById('submit-btn');
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span>⏳ Queuing Job...</span>';

      const progressBox = document.getElementById('progress-box');
      const progressFill = document.getElementById('progress-fill');
      const progressText = document.getElementById('progress-text');
      const progressPct = document.getElementById('progress-pct');

      progressBox.style.display = 'block';
      progressFill.style.width = '0%';
      progressText.innerText = 'Submitting video to background worker...';
      progressPct.innerText = '0%';

      try {
        const formData = new FormData();
        if (selectedUploadFile) {
          formData.append('file', selectedUploadFile);
        } else {
          formData.append('video_path', rawPath);
        }
        formData.append('conf_threshold', conf);
        if (maxFramesVal > 0) {
          formData.append('max_frames', maxFramesVal);
        }

        const res = await fetch('/api/v1/detect-video', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || 'Failed to submit detection job');
        }

        const data = await res.json();
        const jobId = data.job_id;

        if (data.resolved_video_path) {
          document.getElementById('videoPath').value = data.resolved_video_path;
        }

        progressText.innerText = data.message || ('Processing job ' + jobId.substring(0, 8) + '...');

        // Poll every 1.5s
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
          try {
            const statusRes = await fetch(`/api/v1/jobs/${jobId}`);
            if (!statusRes.ok) return;
            const statusData = await statusRes.json();

            const pct = Math.round(statusData.progress_percentage || 0);
            progressFill.style.width = pct + '%';
            progressPct.innerText = pct + '%';
            progressText.innerText = `Processing: ${statusData.frames_processed} / ${statusData.total_frames || '?'} frames (${pct}%)`;

            if (statusData.status === 'completed') {
              clearInterval(pollInterval);
              progressText.innerText = 'Detection completed successfully! Updating dashboard...';
              submitBtn.disabled = false;
              submitBtn.innerHTML = '<span>▶ Run Detection</span>';
              fetchFinalReport(jobId);
            } else if (statusData.status === 'failed') {
              clearInterval(pollInterval);
              progressText.innerText = 'Error: ' + (statusData.error || 'Job failed');
              submitBtn.disabled = false;
              submitBtn.innerHTML = '<span>▶ Run Detection</span>';
            }
          } catch (pErr) {
            console.error('Polling error', pErr);
          }
        }, 1500);

      } catch (err) {
        alert('Submission error: ' + err.message);
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>▶ Run Detection</span>';
        progressBox.style.display = 'none';
      }
    }

    async function fetchFinalReport(jobId) {
      try {
        const repRes = await fetch(`/api/v1/jobs/${jobId}/report`);
        if (!repRes.ok) return;
        const rep = await repRes.json();

        if (rep.summary) {
          document.getElementById('stat-frames').innerText = Number(rep.summary.total_frames_processed).toLocaleString();
          document.getElementById('stat-workers').innerText = rep.summary.total_workers_detected;
          document.getElementById('stat-violations').innerText = rep.summary.unique_violations;
        }

        if (rep.output_video_url) {
          videoElem.src = rep.output_video_url;
          videoElem.load();
          document.getElementById('video-filename').innerText = 'Source: ' + rep.output_video_url.split('/').pop();
          document.getElementById('video-download-btn').href = rep.output_video_url;
        }

        if (rep.violations) {
          const list = rep.violations.map(v => ({
            track_id: v.track_id,
            time: v.formatted_timestamp,
            timeSec: v.timestamp_sec,
            frame: v.frame_index,
            conf: (v.confidence * 100).toFixed(1),
            img: v.snapshot_url,
            desc: 'Safety violation detected'
          }));
          renderViolations(list);
        }
      } catch (e) {
        console.error('Report fetch error:', e);
      }
    }
  </script>
</body>
</html>
"""
