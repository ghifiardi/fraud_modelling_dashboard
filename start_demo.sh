#!/bin/bash

echo "🎬 Blockchain Fraud Detection Dashboard Demo"
echo "============================================="
echo ""

# Check if demo dashboard file exists
if [ ! -f "create_dashboard_demo.py" ]; then
    echo "❌ Error: create_dashboard_demo.py not found!"
    echo "Please ensure the demo file is in the current directory."
    exit 1
fi

echo "✅ Demo dashboard file found"
echo ""

# Check if port 8502 is available
if lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Warning: Port 8502 is already in use"
    echo "   The demo will use port 8503 instead"
    DEMO_PORT=8503
else
    DEMO_PORT=8502
fi

echo "🚀 Starting demo dashboard on port $DEMO_PORT..."
echo ""

# Start the demo dashboard
echo "📋 Recording Instructions:"
echo "1. Wait for dashboard to load (http://localhost:$DEMO_PORT)"
echo "2. Open QuickTime Player (File → New Screen Recording)"
echo "3. Select the browser window with the dashboard"
echo "4. Click record and follow the demo script:"
echo "   - Click '🚀 Start Demo'"
echo "   - Click '⚡ Generate Transaction'"
echo "   - Click '⛏️ Mine Block'"
echo "   - Show fraud alerts and success metrics"
echo "5. Keep recording under 10 seconds"
echo "6. Export as MP4 or GIF"
echo ""

echo "🎯 Demo URL: http://localhost:$DEMO_PORT"
echo "⏱️  Target Duration: 5-10 seconds"
echo "📱 Format: MP4 or GIF"
echo ""

echo "Starting Streamlit demo dashboard..."
echo "Press Ctrl+C to stop the demo"
echo ""

# Start the demo dashboard
python3 -m streamlit run create_dashboard_demo.py --server.port $DEMO_PORT --server.headless true 