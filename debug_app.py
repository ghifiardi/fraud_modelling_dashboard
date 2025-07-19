#!/usr/bin/env python3
"""
Debug version of AI Fraud Detection Monitor
This will help identify what's causing the blank page issue
"""

import streamlit as st
import sys
import os
import traceback

# Add src directory to path
sys.path.append('src')

st.set_page_config(
    page_title="AI Fraud Detection Monitor - Debug",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header
st.markdown('<h1 style="text-align: center; color: #667eea;">🛡️ AI Fraud Detection Monitor - Debug</h1>', unsafe_allow_html=True)

# Debug information
st.header("🔍 Debug Information")

# Check basic imports
st.subheader("1. Basic Imports")
try:
    import pandas as pd
    st.success("✅ pandas imported successfully")
except Exception as e:
    st.error(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    st.success("✅ numpy imported successfully")
except Exception as e:
    st.error(f"❌ numpy import failed: {e}")

try:
    import plotly.express as px
    st.success("✅ plotly.express imported successfully")
except Exception as e:
    st.error(f"❌ plotly.express import failed: {e}")

try:
    import plotly.graph_objects as go
    st.success("✅ plotly.graph_objects imported successfully")
except Exception as e:
    st.error(f"❌ plotly.graph_objects import failed: {e}")

# Check file structure
st.subheader("2. File Structure")
try:
    files = os.listdir('.')
    st.write("Files in root directory:")
    for file in sorted(files):
        st.write(f"  - {file}")
    st.success("✅ File listing successful")
except Exception as e:
    st.error(f"❌ File listing failed: {e}")

# Check src directory
st.subheader("3. Src Directory")
try:
    if os.path.exists('src'):
        src_files = os.listdir('src')
        st.write("Files in src directory:")
        for file in sorted(src_files):
            st.write(f"  - {file}")
        st.success("✅ Src directory listing successful")
    else:
        st.error("❌ Src directory not found")
except Exception as e:
    st.error(f"❌ Src directory listing failed: {e}")

# Check models directory
st.subheader("4. Models Directory")
try:
    if os.path.exists('models'):
        model_files = os.listdir('models')
        st.write("Files in models directory:")
        for file in sorted(model_files):
            st.write(f"  - {file}")
        st.success("✅ Models directory listing successful")
    else:
        st.warning("⚠️ Models directory not found")
except Exception as e:
    st.error(f"❌ Models directory listing failed: {e}")

# Try to import dashboard
st.subheader("5. Dashboard Import Test")
try:
    from dashboard import FraudDetectionDashboard
    st.success("✅ Dashboard imported successfully")
    
    # Try to create dashboard instance
    try:
        dashboard = FraudDetectionDashboard()
        st.success("✅ Dashboard instance created successfully")
        
        # Try to load model
        try:
            detector = dashboard.load_model()
            if detector:
                st.success("✅ Model loaded successfully")
            else:
                st.warning("⚠️ Model loading returned None")
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            st.code(traceback.format_exc())
            
    except Exception as e:
        st.error(f"❌ Dashboard instance creation failed: {e}")
        st.code(traceback.format_exc())
        
except Exception as e:
    st.error(f"❌ Dashboard import failed: {e}")
    st.code(traceback.format_exc())

# Try to run dashboard
st.subheader("6. Dashboard Run Test")
try:
    from dashboard import FraudDetectionDashboard
    dashboard = FraudDetectionDashboard()
    
    # Create a simple test page
    st.header("📊 Test Dashboard")
    
    # Test basic Streamlit components
    st.write("Testing basic Streamlit components...")
    
    # Test metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Metric 1", "100", "+5%")
    with col2:
        st.metric("Test Metric 2", "50", "-2%")
    with col3:
        st.metric("Test Metric 3", "75", "+10%")
    with col4:
        st.metric("Test Metric 4", "25", "+15%")
    
    # Test chart
    try:
        import plotly.express as px
        import pandas as pd
        
        # Create sample data
        data = {
            'Category': ['A', 'B', 'C', 'D'],
            'Value': [10, 20, 15, 25]
        }
        df = pd.DataFrame(data)
        
        fig = px.bar(df, x='Category', y='Value', title='Test Chart')
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Chart rendering successful")
        
    except Exception as e:
        st.error(f"❌ Chart rendering failed: {e}")
    
    st.success("✅ Basic dashboard test successful")
    
except Exception as e:
    st.error(f"❌ Dashboard run test failed: {e}")
    st.code(traceback.format_exc())

# Environment information
st.subheader("7. Environment Information")
st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")
st.write(f"Working directory: {os.getcwd()}")

# Memory usage
try:
    import psutil
    memory = psutil.virtual_memory()
    st.write(f"Memory usage: {memory.percent}%")
    st.write(f"Available memory: {memory.available / 1024 / 1024 / 1024:.2f} GB")
except:
    st.write("Memory information not available")

st.header("🎯 Next Steps")
st.write("""
1. If all imports are successful, the main app should work
2. If there are import errors, we need to fix them
3. If the dashboard loads but doesn't display, there might be a rendering issue
4. Check the Streamlit Cloud logs for any additional errors
""")

# Footer
st.markdown("---")
st.markdown("🔧 **Debug Mode** - This will help identify deployment issues") 