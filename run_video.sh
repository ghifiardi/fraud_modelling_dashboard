#!/bin/bash

echo "🎬 Fraud Detection Architecture Video"
echo "====================================="
echo ""

# Check if video is already running
if curl -s http://localhost:8504 > /dev/null 2>&1; then
    echo "✅ Architecture video is already running!"
    echo "🌐 Open your browser and go to: http://localhost:8504"
    echo ""
    echo "📱 Or click this link to open automatically:"
    echo "   http://localhost:8504"
    echo ""
    echo "🎬 To record the video:"
    echo "   1. Open QuickTime Player"
    echo "   2. File → New Screen Recording"
    echo "   3. Select the browser window"
    echo "   4. Record for 10-15 seconds"
    echo "   5. Save as MP4"
    echo ""
    open http://localhost:8504
    exit 0
fi

# Start the video if not running
echo "🚀 Starting architecture video..."
echo ""

if [ -f "simple_architecture_video.py" ]; then
    echo "✅ Found architecture video file"
    echo "🌐 Starting on port 8504..."
    echo ""
    echo "📋 Instructions:"
    echo "   1. Wait for browser to open"
    echo "   2. Open QuickTime Player"
    echo "   3. File → New Screen Recording"
    echo "   4. Select the browser window"
    echo "   5. Record for 10-15 seconds"
    echo "   6. Save as MP4"
    echo ""
    
    # Start the video
    python3 -m streamlit run simple_architecture_video.py --server.port 8504 --server.headless true &
    
    # Wait a moment and open browser
    sleep 3
    open http://localhost:8504
    
    echo "🎬 Video is now running!"
    echo "🌐 URL: http://localhost:8504"
    echo "⏱️  Record for 10-15 seconds"
    echo "📱 Save as MP4 or GIF"
    
else
    echo "❌ Error: simple_architecture_video.py not found!"
    echo "Please make sure you're in the correct directory."
    exit 1
fi 