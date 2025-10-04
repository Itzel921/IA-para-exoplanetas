#!/bin/bash
# NASA Space Apps Challenge 2025 - Exoplanet Detection System
# Startup script for Unix/Linux/MacOS

echo "🚀 NASA Space Apps Challenge 2025 - Exoplanet Detection System"
echo "================================================================"

# Check Python version
echo "📋 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found. Please install Python 3.8+"; exit 1; }

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Start backend server
echo "🚀 Starting FastAPI backend server..."
echo "Backend will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Frontend: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd src/backend
python main.py