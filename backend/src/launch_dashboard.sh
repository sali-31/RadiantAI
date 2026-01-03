#!/bin/bash

echo "🔬 RadiantAI Dashboard Launcher"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source .venv/bin/activate

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not installed!"
    echo "Run: pip install streamlit"
    exit 1
fi

echo "✓ Starting dashboard..."
echo ""
echo "📍 Dashboard will open at: http://localhost:8501"
echo "⌨️  Press Ctrl+C to stop"
echo ""

# Launch Streamlit
streamlit run app/streamlit_app.py
