#!/bin/bash

# RadiantAI Development Server Startup Script
# This script starts both backend and frontend servers concurrently

set -e  # Exit on error

echo "🚀 Starting RadiantAI Development Environment..."
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :"$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js.${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm not found. Please install npm.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites found${NC}"
echo ""

# Check if ports are available
if port_in_use 8000; then
    echo -e "${RED}❌ Port 8000 is already in use. Please stop the process using it.${NC}"
    exit 1
fi

if port_in_use 5173; then
    echo -e "${RED}❌ Port 5173 is already in use. Please stop the process using it.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Ports 8000 and 5173 are available${NC}"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found. Please create one with your API keys.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment file found${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}📦 Creating Python virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment and install backend dependencies
echo -e "${BLUE}📦 Installing/updating backend dependencies...${NC}"
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r backend/requirements.txt 2>/dev/null || true
echo -e "${GREEN}✓ Backend dependencies ready${NC}"
echo ""

# Install frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
    cd frontend
    npm install
    cd ..
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
fi
echo ""

# Create log directory
mkdir -p logs

# Start backend server in background
echo -e "${BLUE}🔧 Starting Backend Server (FastAPI on port 8000)...${NC}"
source .venv/bin/activate
cd backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..
echo -e "${GREEN}✓ Backend server started (PID: $BACKEND_PID)${NC}"
echo ""

# Wait a moment for backend to initialize
sleep 2

# Start frontend server in background
echo -e "${BLUE}🎨 Starting Frontend Server (Vite on port 5173)...${NC}"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}✓ Frontend server started (PID: $FRONTEND_PID)${NC}"
echo ""

# Wait for servers to be ready
echo "⏳ Waiting for servers to be ready..."
sleep 3

# Check if servers are running
if ! port_in_use 8000; then
    echo -e "${RED}❌ Backend failed to start. Check logs/backend.log${NC}"
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

if ! port_in_use 5173; then
    echo -e "${RED}❌ Frontend failed to start. Check logs/frontend.log${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Success message
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ RadiantAI is now running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}🌐 Frontend:${NC} http://localhost:5173"
echo -e "${BLUE}🔧 Backend:${NC}  http://localhost:8000"
echo -e "${BLUE}📚 API Docs:${NC} http://localhost:8000/docs"
echo ""
echo -e "${BLUE}📋 Process IDs:${NC}"
echo "   Backend PID:  $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo -e "${BLUE}📝 Logs:${NC}"
echo "   Backend:  tail -f logs/backend.log"
echo "   Frontend: tail -f logs/frontend.log"
echo ""
echo -e "${RED}⚠️  Press Ctrl+C to stop both servers${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${BLUE}🛑 Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Servers stopped${NC}"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Keep script running and tail logs
echo -e "${BLUE}📊 Live Backend Logs (Ctrl+C to stop):${NC}"
echo ""
tail -f logs/backend.log
