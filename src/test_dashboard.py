#!/usr/bin/env python3
"""
Minimal test dashboard to verify basic functionality
"""

import streamlit as st

def main():
    st.set_page_config(
        page_title="Test Dashboard",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Test Dashboard")
    st.write("This is a test dashboard to verify basic functionality.")
    
    # Test sidebar
    st.sidebar.title("Test Sidebar")
    st.sidebar.write("Sidebar is working!")
    
    # Test main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Column 1")
        st.write("This is column 1 content.")
        st.button("Test Button 1")
    
    with col2:
        st.subheader("Column 2")
        st.write("This is column 2 content.")
        st.button("Test Button 2")
    
    # Test metrics
    st.subheader("Test Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Metric 1", "100", "10")
    
    with col2:
        st.metric("Test Metric 2", "200", "-5")
    
    with col3:
        st.metric("Test Metric 3", "300", "0")
    
    st.success("✅ Test dashboard is working!")

if __name__ == "__main__":
    main() 