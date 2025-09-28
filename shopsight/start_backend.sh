#!/bin/bash

# ShopSight Backend Startup Script

echo "🚀 Starting ShopSight Backend..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env and add your OpenAI API key:"
    echo "   OPENAI_API_KEY=your_api_key_here"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Start the backend server
echo "🌟 Starting FastAPI server..."
cd backend
python main.py
