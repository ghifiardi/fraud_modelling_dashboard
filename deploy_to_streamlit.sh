#!/bin/bash

echo "🚀 Streamlit Cloud Deployment Verification"
echo "=========================================="

# Check if required files exist
echo "📁 Checking required files..."

if [ -f "streamlit_cloud_app.py" ]; then
    echo "✅ streamlit_cloud_app.py - Found"
else
    echo "❌ streamlit_cloud_app.py - Missing"
    exit 1
fi

if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt - Found"
else
    echo "❌ requirements.txt - Missing"
    exit 1
fi

if [ -f ".streamlit/config.toml" ]; then
    echo "✅ .streamlit/config.toml - Found"
else
    echo "❌ .streamlit/config.toml - Missing"
    exit 1
fi

# Check git status
echo ""
echo "🔍 Checking git status..."
if git status --porcelain | grep -q .; then
    echo "⚠️  You have uncommitted changes. Please commit them first."
    git status
    exit 1
else
    echo "✅ All changes committed"
fi

# Check if remote is set
echo ""
echo "🌐 Checking remote repository..."
if git remote -v | grep -q origin; then
    echo "✅ Remote repository configured"
    git remote -v
else
    echo "❌ No remote repository configured"
    exit 1
fi

# Check if pushed
echo ""
echo "📤 Checking if code is pushed to GitHub..."
if git status -uno | grep -q "Your branch is up to date"; then
    echo "✅ Code is pushed to GitHub"
else
    echo "⚠️  Code may not be pushed. Run: git push origin main"
fi

echo ""
echo "🎯 Deployment Instructions:"
echo "=========================="
echo "1. Go to: https://share.streamlit.io"
echo "2. Sign in with GitHub"
echo "3. Click 'New app'"
echo "4. Repository: ghifiardi/fraud_modelling_dashboard"
echo "5. Branch: main"
echo "6. Main file path: streamlit_cloud_app.py"
echo "7. Python version: 3.9+"
echo "8. Click 'Deploy!'"
echo ""
echo "🌐 Your app will be available at:"
echo "https://your-app-name-ghifiardi.streamlit.app"
echo ""
echo "✅ Ready for deployment!" 