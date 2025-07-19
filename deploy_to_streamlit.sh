#!/bin/bash

# 🚀 Streamlit Cloud Deployment Script
# This script helps you deploy your fraud detection dashboard to Streamlit Cloud

echo "🛡️ AI Fraud Detection Monitor - Streamlit Cloud Deployment"
echo "=========================================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Please run this script from the fraud_modelling_project directory"
    exit 1
fi

echo "✅ Project structure verified"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: AI Fraud Detection Monitor"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Check if remote exists
if ! git remote get-url origin &> /dev/null; then
    echo ""
    echo "🌐 GitHub Repository Setup"
    echo "=========================="
    echo "Please provide your GitHub repository URL:"
    echo "Format: https://github.com/YOUR_USERNAME/fraud_modelling_project.git"
    read -p "GitHub URL: " github_url
    
    if [ -n "$github_url" ]; then
        git remote add origin "$github_url"
        git branch -M main
        git push -u origin main
        echo "✅ Repository pushed to GitHub"
    else
        echo "⚠️  No GitHub URL provided. You'll need to set up the remote manually."
    fi
else
    echo "✅ GitHub remote already configured"
    echo "Current remote: $(git remote get-url origin)"
fi

echo ""
echo "🔧 Streamlit Cloud Deployment Steps"
echo "==================================="
echo ""
echo "1. 🌐 Go to Streamlit Cloud: https://share.streamlit.io"
echo "2. 🔐 Sign in with your GitHub account"
echo "3. ➕ Click 'New app'"
echo "4. 📁 Select your repository: fraud_modelling_project"
echo "5. 📄 Set main file path: streamlit_app.py"
echo "6. 🔑 Configure secrets (API keys):"
echo ""
echo "   Add this to Streamlit Cloud Secrets:"
echo "   ------------------------------------"
echo "   OPENAI_API_KEY = \"your-openai-api-key-here\""
echo ""
echo "7. 🚀 Click 'Deploy!'"
echo ""
echo "📚 Documentation:"
echo "   - Deployment Guide: DEPLOYMENT_GUIDE.md"
echo "   - Complete Documentation: PROJECT_DOCUMENTATION.md"
echo ""
echo "🎉 Your fraud detection dashboard will be live at:"
echo "   https://YOUR_APP_NAME-YOUR_USERNAME.streamlit.app"
echo ""
echo "🇮🇩 Indonesian fraud detection system ready for deployment! 🛡️" 