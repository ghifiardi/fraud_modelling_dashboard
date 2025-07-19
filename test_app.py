#!/usr/bin/env python3
"""
Simple test app for Streamlit Cloud debugging
"""

import streamlit as st
import sys
import os

st.set_page_config(
    page_title="Fraud Detection Test",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Fraud Detection Monitor - Test")
st.write("This is a test to verify Streamlit Cloud deployment is working.")

# Test basic functionality
st.header("System Check")
st.write("✅ Streamlit is working")
st.write("✅ Python imports are working")

# Test file structure
st.header("File Structure Check")
try:
    files = os.listdir('.')
    st.write("Files in root directory:")
    for file in files[:10]:  # Show first 10 files
        st.write(f"- {file}")
except Exception as e:
    st.error(f"Error reading directory: {e}")

# Test src directory
st.header("Source Directory Check")
try:
    if os.path.exists('src'):
        src_files = os.listdir('src')
        st.write("Files in src directory:")
        for file in src_files[:10]:
            st.write(f"- {file}")
    else:
        st.error("src directory not found")
except Exception as e:
    st.error(f"Error reading src directory: {e}")

# Test imports
st.header("Import Test")
try:
    import pandas as pd
    st.write("✅ pandas imported successfully")
except Exception as e:
    st.error(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    st.write("✅ numpy imported successfully")
except Exception as e:
    st.error(f"❌ numpy import failed: {e}")

try:
    import plotly
    st.write("✅ plotly imported successfully")
except Exception as e:
    st.error(f"❌ plotly import failed: {e}")

# Test dashboard import
st.header("Dashboard Import Test")
try:
    sys.path.append('src')
    from dashboard import FraudDetectionDashboard
    st.write("✅ Dashboard imported successfully")
    
    # Try to initialize
    dashboard = FraudDetectionDashboard()
    st.write("✅ Dashboard initialized successfully")
    
except Exception as e:
    st.error(f"❌ Dashboard import failed: {e}")
    st.info("This is likely the cause of the blank page.")

st.header("Next Steps")
st.write("If all tests pass, the main app should work. If not, check the error messages above.") 