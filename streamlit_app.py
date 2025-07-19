#!/usr/bin/env python3
"""
AI Fraud Detection Monitor - Streamlit Cloud Deployment
Main application file for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os
import traceback

# Add src directory to path
sys.path.append('src')

def main():
    """Main Streamlit application entry point."""
    
    # Set page config first
    st.set_page_config(
        page_title="AI Fraud Detection Monitor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add status indicator
    st.success("✅ Application starting...")
    
    try:
        # Import dashboard components
        st.info("🔍 Loading dashboard components...")
        from dashboard import FraudDetectionDashboard
        
        # Initialize dashboard
        st.info("🔧 Initializing dashboard...")
        dashboard = FraudDetectionDashboard()
        
        # Run the dashboard
        st.info("🚀 Starting dashboard...")
        dashboard.run_dashboard()
        
    except ImportError as e:
        st.error(f"❌ Import Error: {e}")
        st.info("Please check that all required files are present in the repository.")
        st.code(traceback.format_exc())
        
    except Exception as e:
        st.error(f"❌ Error starting dashboard: {e}")
        st.info("Please check the Streamlit Cloud logs for more details.")
        st.code(traceback.format_exc())
        
        # Fallback to simple dashboard
        st.header("🛡️ AI Fraud Detection Monitor - Fallback Mode")
        st.warning("⚠️ Using fallback dashboard due to error in main dashboard.")
        
        # Show basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Fallback Mode", "⚠️")
        with col2:
            st.metric("Transactions", "1,247", "+12%")
        with col3:
            st.metric("Fraud Detected", "8", "-3%")
        with col4:
            st.metric("Success Rate", "99.4%", "+0.2%")
        
        st.info("The main dashboard encountered an error. Please check the logs above for details.")

if __name__ == "__main__":
    main() 