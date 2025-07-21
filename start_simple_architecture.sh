#!/bin/bash

echo "🏗️ Simple Architecture Video - Fraud Detection System"
echo "====================================================="
echo ""

# Check if simple architecture video file exists
if [ ! -f "simple_architecture_video.py" ]; then
    echo "❌ Error: simple_architecture_video.py not found!"
    echo "Please ensure the architecture video file is in the current directory."
    exit 1
fi

echo "✅ Simple architecture video file found"
echo ""

# Check if port 8504 is available
if lsof -Pi :8504 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Warning: Port 8504 is already in use"
    echo "   The architecture video will use port 8505 instead"
    VIDEO_PORT=8505
else
    VIDEO_PORT=8504
fi

echo "🎬 Starting simple architecture video on port $VIDEO_PORT..."
echo ""

# Start the simple architecture video
echo "📋 Simple Architecture Video Instructions:"
echo "1. Wait for video to load (http://localhost:$VIDEO_PORT)"
echo "2. Open QuickTime Player (File → New Screen Recording)"
echo "3. Select the browser window with the architecture video"
echo "4. Click record and follow the video script:"
echo "   - Show the complete architecture flow diagram"
echo "   - Highlight key features (Speed, Accuracy, Transparency, Security)"
echo "   - Show technology stack components"
echo "   - Keep recording under 15 seconds"
echo "5. Export as MP4 or GIF"
echo ""

echo "🎯 Simple Architecture Video URL: http://localhost:$VIDEO_PORT"
echo "⏱️  Target Duration: 10-15 seconds"
echo "📱 Format: MP4 or GIF"
echo ""

echo "Starting Streamlit simple architecture video..."
echo "Press Ctrl+C to stop the video"
echo ""

# Start the simple architecture video
python3 -m streamlit run simple_architecture_video.py --server.port $VIDEO_PORT --server.headless true 