"""
FastAPI Server Entrypoint
Helmet Safety Violation Detection Microservice
"""

import os
import socket
import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.app.api import router
from src.app.dashboard import DASHBOARD_HTML


def start_ipv6_bridge(port: int = 8000):
    """
    Binds to IPv6 ::1:8000 and forwards to IPv4 127.0.0.1:8000
    so that both 'http://localhost:8000' and 'http://127.0.0.1:8000'
    resolve instantly without connection errors on Windows.
    """
    def bridge_runner():
        try:
            s_v6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            s_v6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s_v6.bind(('::1', port))
            s_v6.listen(64)
            while True:
                client_v6, _ = s_v6.accept()
                def handle_pair(c6):
                    try:
                        c4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        c4.connect(('127.0.0.1', port))
                        def pipe(src, dst):
                            try:
                                while True:
                                    buf = src.recv(32768)
                                    if not buf:
                                        break
                                    dst.sendall(buf)
                            except Exception:
                                pass
                            finally:
                                try: src.close()
                                except: pass
                                try: dst.close()
                                except: pass
                        threading.Thread(target=pipe, args=(c6, c4), daemon=True).start()
                        threading.Thread(target=pipe, args=(c4, c6), daemon=True).start()
                    except Exception:
                        try: c6.close()
                        except: pass
                threading.Thread(target=handle_pair, args=(client_v6,), daemon=True).start()
        except Exception:
            pass

    threading.Thread(target=bridge_runner, daemon=True).start()


start_ipv6_bridge(8000)

app = FastAPI(
    title="Helmet Safety Violation Detection API",
    description="""
    ## Production Computer Vision Service
    Detects workers without helmets in industrial and construction video feeds.
    Features:
    * Asynchronous, non-blocking video processing.
    * ByteTrack object tracking to count UNIQUE violations once per worker.
    * Handles dynamic events: workers removing helmets partway through.
    * Automatic timestamped snapshot extraction for compliance audits.
    * CPU-optimized with ONNX Runtime & INT8 quantization.
    """,
    version="1.0.0"
)

# Enable CORS for cross-origin frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file directories for snapshots and videos
os.makedirs("outputs/snapshots", exist_ok=True)
os.makedirs("outputs/videos", exist_ok=True)

app.mount("/static/snapshots", StaticFiles(directory="outputs/snapshots"), name="snapshots")
app.mount("/static/videos", StaticFiles(directory="outputs/videos"), name="videos")

# Include API Router
app.include_router(router)


@app.get("/", tags=["Dashboard & Health"])
async def root(request: Request):
    """
    Returns the rich Web Dashboard for browser users,
    or the JSON health check for API clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return HTMLResponse(content=DASHBOARD_HTML)
    return {
        "service": "Helmet Safety Violation Detector API",
        "status": "online",
        "dashboard": "/dashboard",
        "documentation": "/docs",
        "version": "1.0.0"
    }


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard & Health"])
async def dashboard():
    """Direct URL for the Helmet Safety Violation Detector Dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)
