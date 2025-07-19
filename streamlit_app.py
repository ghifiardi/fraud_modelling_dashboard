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

# Import dashboard components
from dashboard import FraudDetectionDashboard

def main():
    """Main Streamlit application entry point."""
    
    # Initialize dashboard
    dashboard = FraudDetectionDashboard()
    
    # Run the dashboard
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 