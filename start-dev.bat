@echo off
REM RadiantAI Development Server Startup Script (Windows)
REM This script starts both backend and frontend servers

echo.
echo ================================
echo Starting RadiantAI Development Environment...
echo ================================
echo.

REM Check if .env exists
if not exist .env (
    echo ERROR: .env file not found. Please create one with your API keys.
    exit /b 1
)

REM Check if virtual environment exists
if not exist .venv (
    echo Creating Python virtual environment...
    python -m venv .venv
    echo Virtual environment created.
    echo.
)

REM Activate virtual environment and install dependencies
echo Installing/updating backend dependencies...
call .venv\Scripts\activate.bat
pip install -q --upgrade pip
pip install -q -r backend\requirements.txt
echo Backend dependencies ready.
echo.

REM Install frontend dependencies if needed
if not exist frontend\node_modules (
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
    echo Frontend dependencies installed.
) else (
    echo Frontend dependencies already installed.
)
echo.

REM Create logs directory
if not exist logs mkdir logs

REM Start backend server
echo Starting Backend Server (FastAPI on port 8000)...
cd backend
start "RadiantAI Backend" cmd /k "..\\.venv\\Scripts\\activate.bat && uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"
cd ..
echo Backend server started.
echo.

REM Wait for backend to initialize
timeout /t 3 /nobreak > nul

REM Start frontend server
echo Starting Frontend Server (Vite on port 5173)...
cd frontend
start "RadiantAI Frontend" cmd /k "npm run dev"
cd ..
echo Frontend server started.
echo.

REM Wait for servers to be ready
echo Waiting for servers to be ready...
timeout /t 5 /nobreak > nul
echo.

echo ================================
echo RadiantAI is now running!
echo ================================
echo.
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to open the application in your browser...
pause > nul

REM Open browser
start http://localhost:5173

echo.
echo To stop the servers, close the command windows labeled:
echo - "RadiantAI Backend"
echo - "RadiantAI Frontend"
echo.
pause
