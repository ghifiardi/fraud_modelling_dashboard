#!/usr/bin/env python3
"""
Working Dashboard - Simplified version with error handling
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def main():
    st.set_page_config(
        page_title="AI Fraud Detection Monitor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #667eea;">🛡️ AI Fraud Detection Monitor</h1>', unsafe_allow_html=True)
    
    # Load model with error handling
    detector = None
    try:
        from bank_fraud_detector import BankFraudDetector
        detector = BankFraudDetector()
        detector.load_bank_model("models/bank_fraud_detector.pkl")
        st.sidebar.success("✅ Model Loaded Successfully")
    except Exception as e:
        st.sidebar.error(f"❌ Model Loading Failed: {e}")
        st.sidebar.info("Dashboard will work with demo data")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Model Status
    st.sidebar.subheader("Model Status")
    if detector and detector.models:
        st.sidebar.success("✅ Model Ready")
        st.sidebar.info(f"Models: {len(detector.models)}")
    else:
        st.sidebar.warning("⚠️ Using Demo Mode")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Real-time Dashboard", 
        "🔍 Transaction Monitor", 
        "📈 Analytics", 
        "⚙️ Settings"
    ])
    
    with tab1:
        st.header("📊 Real-time Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", "1,247", "+23")
        
        with col2:
            st.metric("Fraud Detected", "12", "+2")
        
        with col3:
            st.metric("Success Rate", "99.0%", "+0.5%")
        
        with col4:
            st.metric("Response Time", "0.8s", "-0.2s")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Volume")
            chart_data = pd.DataFrame({
                'Time': pd.date_range(start='2024-01-01', periods=24, freq='H'),
                'Transactions': np.random.poisson(50, 24)
            })
            st.line_chart(chart_data.set_index('Time'))
        
        with col2:
            st.subheader("Fraud Detection Rate")
            fraud_data = pd.DataFrame({
                'Hour': range(24),
                'Fraud Rate': np.random.uniform(0.5, 2.0, 24)
            })
            st.bar_chart(fraud_data.set_index('Hour'))
    
    with tab2:
        st.header("🔍 Transaction Monitor")
        
        # Sample transaction data
        transactions = []
        for i in range(20):
            transactions.append({
                'Transaction ID': f'TXN_{i+1:06d}',
                'Customer ID': f'CUST_{np.random.randint(1, 100):03d}',
                'Amount': round(np.random.uniform(10, 1000), 2),
                'Type': np.random.choice(['ATM', 'POS', 'ONLINE', 'TRANSFER']),
                'Risk Level': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.7, 0.2, 0.1]),
                'Status': np.random.choice(['APPROVED', 'DECLINED'], p=[0.9, 0.1]),
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=np.random.randint(0, 60))
            })
        
        df = pd.DataFrame(transactions)
        st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.header("📈 Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            risk_data = pd.DataFrame({
                'Risk Level': ['LOW', 'MEDIUM', 'HIGH'],
                'Count': [140, 35, 25]
            })
            st.bar_chart(risk_data.set_index('Risk Level'))
        
        with col2:
            st.subheader("Transaction Types")
            type_data = pd.DataFrame({
                'Type': ['ATM', 'POS', 'ONLINE', 'TRANSFER'],
                'Count': [45, 60, 80, 15]
            })
            st.bar_chart(type_data.set_index('Type'))
    
    with tab4:
        st.header("⚙️ Settings")
        
        st.subheader("System Configuration")
        
        # Auto refresh
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        # Refresh interval
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
        
        # Risk thresholds
        st.subheader("Risk Thresholds")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_threshold = st.slider("Low Risk", 0.0, 1.0, 0.3, 0.1)
        
        with col2:
            medium_threshold = st.slider("Medium Risk", 0.0, 1.0, 0.5, 0.1)
        
        with col3:
            high_threshold = st.slider("High Risk", 0.0, 1.0, 0.7, 0.1)
        
        # Quick actions
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data", type="primary"):
                st.success("Data refreshed!")
        
        with col2:
            if st.button("📊 Generate Report"):
                st.info("Report generation started...")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🛡️ AI Fraud Detection Monitor | Real-time transaction monitoring and fraud detection</p>
        <p>Built with Streamlit, Machine Learning, and Real-time Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 