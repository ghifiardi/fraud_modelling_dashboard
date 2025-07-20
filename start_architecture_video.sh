#!/bin/bash

echo "🏗️ Fraud Detection System Architecture Video"
echo "============================================="
echo ""

# Check if architecture video file exists
if [ ! -f "architecture_video_script.py" ]; then
    echo "❌ Error: architecture_video_script.py not found!"
    echo "Please ensure the architecture video file is in the current directory."
    exit 1
fi

echo "✅ Architecture video file found"
echo ""

# Check if port 8503 is available
if lsof -Pi :8503 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Warning: Port 8503 is already in use"
    echo "   The architecture video will use port 8504 instead"
    VIDEO_PORT=8504
else
    VIDEO_PORT=8503
fi

echo "🎬 Starting architecture video on port $VIDEO_PORT..."
echo ""

# Start the architecture video
echo "📋 Architecture Video Instructions:"
echo "1. Wait for video to load (http://localhost:$VIDEO_PORT)"
echo "2. Open QuickTime Player (File → New Screen Recording)"
echo "3. Select the browser window with the architecture video"
echo "4. Click record and follow the video script:"
echo "   - Click '🎬 Start Architecture Video'"
echo "   - Watch each step appear automatically"
echo "   - Show the complete system flow"
echo "   - Highlight key features and technology stack"
echo "5. Keep recording under 15 seconds"
echo "6. Export as MP4 or GIF"
echo ""

echo "🎯 Architecture Video URL: http://localhost:$VIDEO_PORT"
echo "⏱️  Target Duration: 10-15 seconds"
echo "📱 Format: MP4 or GIF"
echo ""

echo "Starting Streamlit architecture video..."
echo "Press Ctrl+C to stop the video"
echo ""

# Start the architecture video
python3 -m streamlit run architecture_video_script.py --server.port $VIDEO_PORT --server.headless true 