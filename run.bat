@echo off
title Helmet Safety Violation Detector
echo ============================================================
echo  HELMET SAFETY VIOLATION DETECTOR: LIVE SERVICE
echo ============================================================
echo.
echo [1/3] Activating Python virtual environment (.venv)...
call .venv\Scripts\activate.bat

echo [2/3] Launching Web Dashboard in default browser...
start http://127.0.0.1:8000/

echo [3/3] Starting FastAPI Uvicorn Server on port 8000...
echo.
echo  * Web Dashboard:  http://127.0.0.1:8000/
echo  * Swagger Docs:   http://127.0.0.1:8000/docs
echo.
echo (Press CTRL+C anytime to stop the server)
echo.
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
