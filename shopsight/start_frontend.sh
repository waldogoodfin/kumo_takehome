#!/bin/bash

# ShopSight Frontend Startup Script

echo "🚀 Starting ShopSight Frontend..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the development server
echo "🌟 Starting React development server..."
npm start
