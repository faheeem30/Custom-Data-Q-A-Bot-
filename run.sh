#!/bin/bash
# RAG Q&A Bot - Setup & Run Script

set -e

echo "========================================"
echo "   Custom Data Q&A Bot - RAG Setup"
echo "========================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install deps
echo "📦 Installing dependencies..."
pip install -r requirements.txt --quiet

# Check ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  ANTHROPIC_API_KEY not set!"
    echo "   Export it before running:"
    echo "   export ANTHROPIC_API_KEY=sk-ant-..."
    echo ""
    read -p "Enter your Anthropic API key: " key
    export ANTHROPIC_API_KEY="$key"
fi

echo ""
echo "🚀 Starting server at http://localhost:8000"
echo "   Open your browser and go to: http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
