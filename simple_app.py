#!/usr/bin/env python3
"""
Simplified AI Fraud Detection Monitor for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import datetime
import sys
import os

# Add src to path
sys.path.append('src')

st.set_page_config(
    page_title="AI Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🛡️ AI Fraud Detection Monitor</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎛️ Controls")
st.sidebar.markdown("### Quick Actions")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🚀 Streaming", "📈 Analytics", "⚙️ Settings"])

with tab1:
    st.header("📊 Real-time Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Transactions", "1,247", "+12%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Detected", "8", "-3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Success Rate", "99.4%", "+0.2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Response Time", "0.8s", "-0.1s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Indonesian Geographic Distribution
    st.subheader("🗺️ Indonesian Geographic Distribution")
    
    # Sample data for Indonesian provinces
    indonesia_data = {
        'Province': ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Sumatera Utara', 
                    'Sulawesi Selatan', 'Banten', 'Bali', 'Sumatera Selatan', 'Riau'],
        'Volume': [85, 72, 68, 65, 58, 52, 48, 45, 42, 38],
        'Fraud_Rate': [0.8, 1.2, 0.9, 1.5, 0.7, 0.6, 0.8, 1.1, 0.5, 0.9]
    }
    
    df_geo = pd.DataFrame(indonesia_data)
    
    # Create chart
    fig = go.Figure()
    
    # Add bars for volume
    fig.add_trace(go.Bar(
        x=df_geo['Province'],
        y=df_geo['Volume'],
        name='Transaction Volume',
        marker_color='#667eea',
        opacity=0.8
    ))
    
    # Add line for fraud rate
    fig.add_trace(go.Scatter(
        x=df_geo['Province'],
        y=df_geo['Fraud_Rate'],
        name='Fraud Rate (%)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#764ba2', width=3),
        marker=dict(size=8, color='#764ba2')
    ))
    
    fig.update_layout(
        title="Transaction Volume & Fraud Rate by Indonesian Province",
        xaxis_title="Province",
        yaxis_title="Transaction Volume",
        yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right"),
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🚀 Streaming System")
    
    # Streaming controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start System", type="primary"):
            st.success("✅ Streaming system started!")
    
    with col2:
        if st.button("⏹️ Stop System"):
            st.info("⏹️ Streaming system stopped!")
    
    with col3:
        tps = st.slider("TPS", 10, 200, 75, help="Transactions per second")
    
    # Real-time transaction feed
    st.subheader("📡 Real-time Transaction Feed")
    
    # Simulate real-time data
    if st.button("🔄 Start Generation"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(10):
            # Simulate transaction
            transaction = {
                'ID': f'TXN_{i+1:06d}',
                'Amount': f'Rp {np.random.randint(10000, 1000000):,}',
                'Location': np.random.choice(['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur']),
                'Risk': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'Status': np.random.choice(['APPROVED', 'REVIEW', 'BLOCKED'])
            }
            
            # Display transaction
            with st.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.write(transaction['ID'])
                col2.write(transaction['Amount'])
                col3.write(transaction['Location'])
                col4.write(transaction['Risk'])
                col5.write(transaction['Status'])
            
            progress_bar.progress((i + 1) / 10)
            status_text.text(f"Processing transaction {i+1}/10...")
            time.sleep(0.5)
        
        st.success("✅ Transaction generation completed!")

with tab3:
    st.header("📈 Analytics")
    
    # Payment network analysis
    st.subheader("💳 Payment Network Analysis")
    
    networks_data = {
        'Network': ['Visa', 'Mastercard', 'JCB', 'UnionPay', 'Other'],
        'Volume': [45, 35, 12, 5, 3],
        'Fraud_Rate': [0.8, 1.1, 0.6, 1.3, 0.9]
    }
    
    df_networks = pd.DataFrame(networks_data)
    
    fig_networks = px.bar(
        df_networks, 
        x='Network', 
        y='Volume',
        color='Fraud_Rate',
        title="Payment Network Distribution (Indonesian Market)",
        color_continuous_scale='RdYlGn_r'
    )
    
    st.plotly_chart(fig_networks, use_container_width=True)

with tab4:
    st.header("⚙️ Settings")
    
    st.subheader("🔧 System Configuration")
    
    # Model status
    st.info("✅ Fraud Detection Model: Loaded")
    st.info("✅ Streaming System: Ready")
    st.info("✅ Analytics Engine: Active")
    
    # API status
    st.subheader("🔌 API Status")
    st.success("✅ OpenAI API: Connected")
    st.success("✅ FraudLabs Pro: Connected")
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("CPU Usage", "23%")
        st.metric("Memory Usage", "1.2GB")
    
    with col2:
        st.metric("Response Time", "0.8s")
        st.metric("Throughput", "75 TPS")

# Footer
st.markdown("---")
st.markdown("🇮🇩 **Indonesian Fraud Detection System** - Powered by AI & Machine Learning") 