#!/usr/bin/env python3
"""
AI Fraud Detection Monitor - Streamlit Cloud Deployment
Main application file for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append('src')

def main():
    """Main Streamlit application entry point."""
    
    try:
        # Import dashboard components
        from dashboard import FraudDetectionDashboard
        
        # Initialize dashboard
        dashboard = FraudDetectionDashboard()
        
        # Run the dashboard
        dashboard.run_dashboard()
        
    except ImportError as e:
        st.error(f"Import Error: {e}")
        st.info("Please check that all required files are present in the repository.")
        
    except Exception as e:
        st.error(f"Error starting dashboard: {e}")
        st.info("Please check the Streamlit Cloud logs for more details.")

if __name__ == "__main__":
    main() 