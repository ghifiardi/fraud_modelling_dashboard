#!/usr/bin/env python3
"""
AI Fraud Detection Agent Monitoring Dashboard
Real-time monitoring and analytics for the fraud detection system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import datetime
from pathlib import Path
import sys
import os
import requests

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bank_fraud_detector import BankFraudDetector
import joblib

# Import the new chatbot service
from llm_chatbot import create_chatbot_ui

# Import the streaming fraud detection system
from streaming_fraud_detector import create_streaming_dashboard_tab

# Remove the chatbot UI creation from module level to prevent infinite loops
# create_chatbot_ui()  # This was causing the infinite loop


class FraudDetectionDashboard:
    def __init__(self):
        self.model_path = "models/bank_fraud_detector.pkl"
        self.detector = None
        
    def load_model(self):
        """Load the trained fraud detection model."""
        if self.detector is None:
            try:
                if os.path.exists(self.model_path):
                    self.detector = BankFraudDetector()
                    self.detector.load_bank_model(self.model_path)
                    print("✓ Bank fraud detection model loaded successfully")
                else:
                    # Create and train a model if none exists
                    st.info("Training new fraud detection model...")
                    self.detector = BankFraudDetector()
                    df = self.detector.create_sample_bank_dataset()
                    df = self.detector.engineer_bank_features(df)
                    results, X_test, y_test = self.detector.train_bank_models(df)
                    self.detector.set_risk_thresholds(results, y_test)
                    self.detector.save_bank_model(self.model_path)
                    st.success("✓ New fraud detection model trained and saved!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
        
        return self.detector
    
    def generate_sample_transactions(self, n=100):
        """Generate sample transactions for demonstration."""
        np.random.seed(42)
        
        transactions = []
        for i in range(n):
            transaction = {
                'transaction_id': f"TXN_{i+1:06d}",
                'customer_id': np.random.randint(1, 501),
                'amount': np.random.exponential(100),
                'transaction_type': np.random.choice(['ATM', 'POS', 'ONLINE', 'TRANSFER']),
                'merchant_category': np.random.choice(['RETAIL', 'FOOD', 'TRAVEL', 'UTILITIES', 'OTHER']),
                'hour': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'location': np.random.choice(['LOCAL', 'DOMESTIC', 'INTERNATIONAL']),
                'device_type': np.random.choice(['MOBILE', 'DESKTOP', 'ATM', 'POS']),
                'card_present': np.random.choice([True, False]),
                'previous_fraud_flag': np.random.choice([True, False], p=[0.95, 0.05]),
                'account_age_days': np.random.randint(1, 3650),
                'balance_before': np.random.uniform(0, 10000),
                'balance_after': np.random.uniform(0, 10000),
                'timestamp': datetime.datetime.now() - datetime.timedelta(minutes=np.random.randint(0, 1440))
            }
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def run_dashboard(self):
        """Main dashboard application."""
        st.set_page_config(
            page_title="AI Fraud Detection Monitor",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load model first
        self.detector = self.load_model()
        
        # Enhanced Custom CSS for modern styling
        st.markdown("""
        <style>
        /* Global styles */
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Enhanced metric cards */
        .metric-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        
        /* Alert styling */
        .alert-high {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border-left: 4px solid #ff6b6b;
            border-radius: 10px;
            padding: 1rem;
        }
        .alert-medium {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-left: 4px solid #ffd43b;
            border-radius: 10px;
            padding: 1rem;
        }
        .alert-low {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-left: 4px solid #51cf66;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Dashboard containers */
        .dashboard-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            color: white;
        }
        
        /* Enhanced metric values */
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: white;
            margin-bottom: 0.5rem;
        }
        
        /* Risk gauge styling */
        .risk-gauge {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Transaction feed styling */
        .transaction-feed {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Alert panel styling */
        .alert-panel {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Chart container styling */
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Geographic map styling */
        .geographic-map {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Network distribution styling */
        .network-distribution {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Risk factors styling */
        .risk-factors {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Enhanced Streamlit metric styling */
        .stMetric > div > div {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            padding: 1.5rem;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .stMetric > div > div > div {
            color: white !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            color: #666;
        }
        .stTabs [aria-selected="true"] {
            background-color: #667eea;
            color: white;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            border: none;
            color: white;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }
        
        /* Selectbox styling */
        .stSelectbox > div > div > div {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }
        .stSelectbox > div > div > div:focus-within {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🛡️ AI Fraud Detection Monitor</h1>', unsafe_allow_html=True)
        
        # Add streaming system notification
        st.info("🚀 **NEW: Real-Time Streaming System Available!** Click on the '🚀 Streaming System' tab to experience Apache Kafka + Spark Streaming simulation with sub-second transaction processing!")
        
        # Sidebar
        self.create_sidebar()
        
        # Main content tabs - Move Streaming System to position 2 for better visibility
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📊 Real-time Dashboard", 
            "🚀 Streaming System", 
            "🔍 Transaction Monitor", 
            "📈 Analytics", 
            "⚙️ Model Management",
            "🚨 Alerts & Logs",
            "📝 Analyst Review",
            "🌐 Fraud Intelligence Network",
            "🤖 OpenAI Playground"
        ])
        
        with tab1:
            self.real_time_dashboard()
        
        with tab2:
            # Store detector in session state for streaming system
            if self.detector:
                st.session_state.detector = self.detector
            create_streaming_dashboard_tab()
        
        with tab3:
            self.transaction_monitor()
        
        with tab4:
            self.analytics_dashboard()
        
        with tab5:
            self.model_management()
        
        with tab6:
            self.alerts_and_logs()
        
        with tab7:
            self.analyst_review_tab()
        
        with tab8:
            self.fraud_intelligence_network()
        
        with tab9:
            self.openai_playground_tab()
    
    def create_sidebar(self):
        """Create the sidebar with controls and settings."""
        # Create the chatbot UI first (this will appear at the top of sidebar)
        create_chatbot_ui()
        
        st.sidebar.title("🎛️ Control Panel")
        
        # Model Status
        st.sidebar.subheader("Model Status")
        if self.detector and self.detector.models:
            st.sidebar.success("✅ Model Loaded")
            st.sidebar.info(f"Models: {len(self.detector.models)}")
        else:
            st.sidebar.error("❌ Model Not Loaded")
        
        # Real-time Settings
        st.sidebar.subheader("Real-time Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
        
        # Risk Thresholds
        st.sidebar.subheader("Risk Thresholds")
        if self.detector and hasattr(self.detector, 'risk_thresholds'):
            for level, threshold in self.detector.risk_thresholds.items():
                st.sidebar.metric(level.replace('_', ' ').title(), f"{threshold:.3f}")
        
        # Quick Actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.sidebar.button("📊 Generate Report"):
            self.generate_report()
        
        # Streaming System Quick Access
        st.sidebar.subheader("🚀 Streaming System")
        if st.sidebar.button("Start Streaming Demo", type="primary"):
            st.info("Navigate to the '🚀 Streaming System' tab to start the real-time fraud detection demo!")
        
        # System Info
        st.sidebar.subheader("System Info")
        st.sidebar.info(f"Last Updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # FraudLabs Pro Screening
        st.sidebar.subheader("FraudLabs Pro Screening")
        ip = st.sidebar.text_input("IP Address", "1.2.3.4", key="flp_ip")
        email = st.sidebar.text_input("Email", "customer@example.com", key="flp_email")
        amount = st.sidebar.number_input("Amount", min_value=0.0, value=100.0, key="flp_amount")
        if st.sidebar.button("Screen Transaction", key="flp_screen"):
            try:
                result = fraudlabspro_screen_order(
                    api_key="TNFUSCVIQFJEV4QYO10B7EONML4515EP",
                    ip_address=ip,
                    email=email,
                    amount=amount
                )
                st.sidebar.write("FraudLabs Pro Result:", result)
            except Exception as e:
                st.sidebar.error(f"FraudLabs Pro Error: {e}")
    
    def real_time_dashboard(self):
        """Revamped real-time dashboard with modern UI/UX design and better proportions."""
        
        # Custom CSS for enhanced styling
        st.markdown("""
        <style>
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .metric-container {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .metric-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .risk-gauge {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .transaction-feed {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .alert-panel {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .geographic-map {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .network-distribution {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .risk-factors {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Enhanced Header
        st.markdown("""
        <div class="dashboard-header">
            <h1 style="margin: 0; font-size: 2.5rem;">🛡️ Real-Time Transaction Monitoring</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Live fraud detection and risk assessment dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate real-time data
        current_time = datetime.datetime.now()
        
        # Top-level metrics with enhanced styling
        st.markdown("### 📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">1,247</div>
                <div class="metric-label">📊 Total Transactions</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">↗️ +23 (1.9%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">98.9%</div>
                <div class="metric-label">✅ Approval Rate</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">↗️ +0.2%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">1.1%</div>
                <div class="metric-label">❌ Decline Rate</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">↘️ -0.2%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">3</div>
                <div class="metric-label">🚨 High Risk Alerts</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">↗️ +1 (50%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main dashboard layout with better proportions
        st.markdown("### 📈 Transaction Analytics")
        
        # First row: Transaction volume and risk assessment
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📊 Transaction Volume (Last 24 Hours)")
            
            # Generate enhanced time series data
            hours = list(range(24))
            base_volumes = [40 + 30*np.sin(h/24 * 2*np.pi) + np.random.normal(0, 5) for h in hours]
            volumes = [max(0, int(v)) for v in base_volumes]
            
            # Create enhanced line chart
            fig = go.Figure()
            
            # Main volume line with gradient
            fig.add_trace(go.Scatter(
                x=hours, y=volumes,
                mode='lines+markers',
                name='Transaction Volume',
                line=dict(color='#667eea', width=4, shape='spline'),
                marker=dict(size=8, color='#667eea', line=dict(width=2, color='white')),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            # Add trend line
            z = np.polyfit(hours, volumes, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=hours, y=p(hours),
                mode='lines',
                name='Trend',
                line=dict(color='#764ba2', width=2, dash='dash')
            ))
            
            # Add threshold lines
            high_water = max(volumes) * 1.05
            low_water = min(volumes) * 0.95
            
            fig.add_hline(y=high_water, line_dash="dash", line_color="#ff6b6b", 
                         annotation_text="High Alert Threshold", annotation_position="top right")
            fig.add_hline(y=low_water, line_dash="dash", line_color="#51cf66", 
                         annotation_text="Low Alert Threshold", annotation_position="bottom right")
            
            fig.update_layout(
                title="",
                xaxis_title="Hour of Day",
                yaxis_title="Number of Transactions",
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=30, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="risk-gauge">', unsafe_allow_html=True)
            st.subheader("🎯 Risk Assessment")
            
            # Generate a sample risk score
            risk_score = np.random.randint(25, 80)
            
            # Create enhanced gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Risk Score", 'font': {'size': 18}},
                delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#667eea", 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "#51cf66"},
                        {'range': [30, 60], 'color': "#ffd43b"},
                        {'range': [60, 100], 'color': "#ff6b6b"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                font=dict(size=14),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors with enhanced styling
            st.subheader("🔍 Risk Factors")
            
            risk_factors = {
                "Amount Anomaly": np.random.randint(20, 95),
                "Time Pattern": np.random.randint(30, 80),
                "Location Risk": np.random.randint(25, 75),
                "Device Fingerprint": np.random.randint(15, 60)
            }
            
            for factor, score in risk_factors.items():
                color = "#ff6b6b" if score > 70 else "#ffd43b" if score > 40 else "#51cf66"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            margin: 0.5rem 0; padding: 0.5rem; background: rgba(255,255,255,0.7); 
                            border-radius: 8px;">
                    <span style="font-weight: 500;">{factor}</span>
                    <span style="font-weight: bold; color: {color}; font-size: 1.2rem;">{score}/100</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Second row: Geographic distribution and network breakdown
        st.markdown("### 🌍 Geographic & Network Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="geographic-map">', unsafe_allow_html=True)
            st.subheader("🗺️ Geographic Distribution")
            
            # Enhanced geographic data for Indonesia
            indonesia_data = {
                'DKI Jakarta': {'volume': 85, 'fraud_rate': 0.8, 'color': '#ff6b6b'},
                'Jawa Barat': {'volume': 72, 'fraud_rate': 1.2, 'color': '#ffd43b'},
                'Jawa Tengah': {'volume': 68, 'fraud_rate': 0.9, 'color': '#51cf66'},
                'Jawa Timur': {'volume': 65, 'fraud_rate': 1.5, 'color': '#ff6b6b'},
                'Sumatera Utara': {'volume': 58, 'fraud_rate': 0.7, 'color': '#51cf66'},
                'Sulawesi Selatan': {'volume': 52, 'fraud_rate': 0.6, 'color': '#51cf66'},
                'Banten': {'volume': 48, 'fraud_rate': 0.8, 'color': '#ffd43b'},
                'Bali': {'volume': 45, 'fraud_rate': 1.1, 'color': '#ffd43b'},
                'Sumatera Selatan': {'volume': 42, 'fraud_rate': 0.5, 'color': '#51cf66'},
                'Riau': {'volume': 38, 'fraud_rate': 0.9, 'color': '#ffd43b'}
            }
            
            provinces = list(indonesia_data.keys())
            volumes = [indonesia_data[province]['volume'] for province in provinces]
            fraud_rates = [indonesia_data[province]['fraud_rate'] for province in provinces]
            colors = [indonesia_data[province]['color'] for province in provinces]
            
            # Create enhanced bar chart with dual metrics
            fig = go.Figure()
            
            # Primary volume bars
            fig.add_trace(go.Bar(
                x=provinces,
                y=volumes,
                name='Transaction Volume',
                marker_color=colors,
                opacity=0.8,
                text=volumes,
                textposition='auto',
            ))
            
            # Add fraud rate as secondary axis
            fig.add_trace(go.Scatter(
                x=provinces,
                y=fraud_rates,
                name='Fraud Rate (%)',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='#764ba2', width=3),
                marker=dict(size=8, color='#764ba2')
            ))
            
            fig.update_layout(
                title="Transaction Volume & Fraud Rate by Province",
                xaxis_title="Province",
                yaxis_title="Transaction Volume",
                yaxis2=dict(
                    title="Fraud Rate (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 2]
                ),
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="network-distribution">', unsafe_allow_html=True)
            st.subheader("🌐 Network Distribution")
            
            # Enhanced network data for Indonesia
            networks_data = {
                'Visa': {'volume': 45, 'fraud_rate': 0.8, 'color': '#1f77b4'},
                'Mastercard': {'volume': 35, 'fraud_rate': 1.1, 'color': '#ff7f0e'},
                'JCB': {'volume': 12, 'fraud_rate': 0.6, 'color': '#2ca02c'},
                'UnionPay': {'volume': 5, 'fraud_rate': 1.3, 'color': '#d62728'},
                'Other': {'volume': 3, 'fraud_rate': 0.9, 'color': '#9467bd'}
            }
            
            networks = list(networks_data.keys())
            network_volumes = [networks_data[net]['volume'] for net in networks]
            network_colors = [networks_data[net]['color'] for net in networks]
            
            # Create enhanced pie chart
            fig = go.Figure(data=[go.Pie(
                labels=networks,
                values=network_volumes,
                hole=0.4,
                marker_colors=network_colors,
                textinfo='label+percent',
                textposition='inside',
                insidetextorientation='radial'
            )])
            
            fig.update_layout(
                title="Transaction Volume by Network",
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add network fraud rate table
            st.subheader("📊 Network Fraud Rates")
            fraud_data = []
            for net in networks:
                fraud_data.append({
                    'Network': net,
                    'Volume (%)': networks_data[net]['volume'],
                    'Fraud Rate (%)': networks_data[net]['fraud_rate']
                })
            
            fraud_df = pd.DataFrame(fraud_data)
            st.dataframe(fraud_df, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Third row: Transaction types and recent activity
        st.markdown("### 💳 Transaction Flow Health")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Enhanced transaction type breakdown
            transaction_types = {
                "💳 Credit Transactions": {"count": 230, "decline_rate": 0.43, "fraud_rate": 0.8},
                "🏦 Check Transactions": {"count": 16, "decline_rate": 0.0, "fraud_rate": 0.2},
                "🏧 ATM Transactions": {"count": 11, "decline_rate": 9.09, "fraud_rate": 1.5},
                "💳 Debit Transactions": {"count": 20, "decline_rate": 5.0, "fraud_rate": 0.6}
            }
            
            # Create horizontal bar chart for transaction types
            types = list(transaction_types.keys())
            counts = [transaction_types[t]['count'] for t in types]
            decline_rates = [transaction_types[t]['decline_rate'] for t in types]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=types,
                x=counts,
                orientation='h',
                name='Transaction Count',
                marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c'],
                text=counts,
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Transaction Volume by Type",
                xaxis_title="Number of Transactions",
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add decline rate indicators
            st.subheader("📉 Decline Rate Analysis")
            decline_cols = st.columns(4)
            for i, (type_name, data) in enumerate(transaction_types.items()):
                with decline_cols[i]:
                    decline_color = "#ff6b6b" if data["decline_rate"] > 5 else "#51cf66"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.8); 
                                border-radius: 8px; margin: 0.5rem 0;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: {decline_color};">
                            {data['decline_rate']:.1f}%
                        </div>
                        <div style="font-size: 0.9rem; color: #666;">
                            {type_name.split()[1]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="transaction-feed">', unsafe_allow_html=True)
            st.subheader("📋 Recent Transactions")
            
            # Generate enhanced transaction data
            transactions = []
            for i in range(8):
                tx_id = f"TXN_{np.random.randint(100000, 999999)}"
                risk_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
                amount = np.random.randint(10, 2000)
                transactions.append({
                    'ID': tx_id,
                    'Risk': risk_level,
                    'Amount': f"${amount:,}",
                    'Time': f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}"
                })
            
            # Display as enhanced list
            for tx in transactions:
                risk_color = "#51cf66" if tx['Risk'] == 'Low' else "#ffd43b" if tx['Risk'] == 'Medium' else "#ff6b6b"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            margin: 0.5rem 0; padding: 0.8rem; background: rgba(255,255,255,0.8); 
                            border-radius: 8px; border-left: 4px solid {risk_color};">
                    <div>
                        <div style="font-weight: bold; font-size: 0.9rem;">{tx['ID']}</div>
                        <div style="font-size: 0.8rem; color: #666;">{tx['Time']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: bold; color: {risk_color};">{tx['Risk']}</div>
                        <div style="font-size: 0.9rem;">{tx['Amount']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bottom section - Risk distribution and alerts
        st.markdown("### 🚨 Risk Distribution & Alerts")
        
        col_risk_dist, col_alerts = st.columns(2)
        
        with col_risk_dist:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Enhanced risk distribution
            risk_levels = ['Low', 'Medium', 'High']
            risk_counts = [65, 25, 10]  # Percentages
            risk_colors = ['#51cf66', '#ffd43b', '#ff6b6b']
            
            fig = px.pie(
                values=risk_counts, names=risk_levels,
                title="Risk Level Distribution",
                color_discrete_map={'Low': '#51cf66', 'Medium': '#ffd43b', 'High': '#ff6b6b'}
            )
            
            fig.update_layout(
                height=350,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_alerts:
            st.markdown('<div class="alert-panel">', unsafe_allow_html=True)
            st.subheader("⚠️ High-Priority Alerts")
            
            alerts = [
                {"type": "High Risk Transaction", "count": 3, "severity": "High", "icon": "🔴"},
                {"type": "Unusual Pattern", "count": 7, "severity": "Medium", "icon": "🟡"},
                {"type": "Geographic Anomaly", "count": 2, "severity": "High", "icon": "🔴"},
                {"type": "Device Mismatch", "count": 5, "severity": "Medium", "icon": "🟡"},
                {"type": "Velocity Alert", "count": 4, "severity": "Medium", "icon": "🟡"},
                {"type": "Amount Threshold", "count": 1, "severity": "High", "icon": "🔴"}
            ]
            
            for alert in alerts:
                severity_color = "#ff6b6b" if alert["severity"] == "High" else "#ffd43b"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            margin: 0.5rem 0; padding: 0.8rem; background: rgba(255,255,255,0.8); 
                            border-radius: 8px; border-left: 4px solid {severity_color};">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{alert['icon']}</span>
                        <div>
                            <div style="font-weight: bold;">{alert['type']}</div>
                            <div style="font-size: 0.8rem; color: #666;">{alert['count']} instances</div>
                        </div>
                    </div>
                    <div style="font-weight: bold; color: {severity_color};">
                        {alert['severity']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-refresh functionality
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        # Add a timestamp
        st.caption(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def transaction_monitor(self):
        """Transaction monitoring interface."""
        st.header("🔍 Transaction Monitor")
        
        # Transaction Input
        st.subheader("Test Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)
            transaction_type = st.selectbox("Transaction Type", ['ATM', 'POS', 'ONLINE', 'TRANSFER'])
            location = st.selectbox("Location", ['LOCAL', 'DOMESTIC', 'INTERNATIONAL'])
            card_present = st.checkbox("Card Present", value=True)
        
        with col2:
            customer_id = st.number_input("Customer ID", min_value=1, value=1, step=1)
            hour = st.slider("Hour of Day", 0, 23, 12)
            merchant_category = st.selectbox("Merchant Category", ['RETAIL', 'FOOD', 'TRAVEL', 'UTILITIES', 'OTHER'])
            device_type = st.selectbox("Device Type", ['MOBILE', 'DESKTOP', 'ATM', 'POS'])
        
        if st.button("🔍 Analyze Transaction"):
            self.analyze_transaction(amount, transaction_type, location, card_present, 
                                   customer_id, hour, merchant_category, device_type)
        
        # Recent Transactions Table
        st.subheader("📋 Recent Transactions")
        recent_transactions = self.generate_sample_transactions(20)
        st.dataframe(recent_transactions, use_container_width=True)
    
    def analytics_dashboard(self):
        """Analytics and insights dashboard."""
        st.header("📈 Analytics & Insights")
        
        # Time-based Analysis
        st.subheader("📅 Time-based Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_hourly_patterns()
        
        with col2:
            self.plot_weekly_patterns()
        
        # Customer Analysis
        st.subheader("👥 Customer Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_customer_risk_profiles()
        
        with col2:
            self.plot_transaction_amount_distribution()
        
        # Model Performance
        st.subheader("🤖 Model Performance")
        self.plot_model_comparison()
    
    def model_management(self):
        """Model management interface."""
        st.header("⚙️ Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            if self.detector and self.detector.models:
                for name, model in self.detector.models.items():
                    st.info(f"**{name}**")
                    st.write(f"Type: {type(model).__name__}")
                    if hasattr(model, 'n_estimators'):
                        st.write(f"Estimators: {model.n_estimators}")
            
            st.subheader("Feature Information")
            if self.detector and self.detector.feature_columns:
                st.write(f"Number of features: {len(self.detector.feature_columns)}")
                st.write("Features:", ", ".join(self.detector.feature_columns[:10]))
        
        with col2:
            st.subheader("Model Actions")
            
            if st.button("🔄 Retrain Model"):
                st.info("Model retraining initiated...")
                # Add retraining logic here
            
            if st.button("📊 Export Model"):
                st.success("Model exported successfully!")
            
            if st.button("🧹 Clear Cache"):
                st.success("Cache cleared!")
            
            st.subheader("Model Health")
            self.plot_model_health()
    
    def alerts_and_logs(self):
        """Alerts and logs interface."""
        st.header("🚨 Alerts & Logs")
        
        # Alerts
        st.subheader("🚨 Recent Alerts")
        
        alerts = [
            {"level": "HIGH", "message": "Unusual transaction pattern detected", "time": "2 min ago"},
            {"level": "MEDIUM", "message": "Multiple failed login attempts", "time": "5 min ago"},
            {"level": "LOW", "message": "New customer registration", "time": "10 min ago"}
        ]
        
        alert_styles = {
            "HIGH": "background-color:#ffebee;border-left:6px solid #f44336;color:#b71c1c;box-shadow:0 2px 8px rgba(244,67,54,0.08);margin-bottom:10px;padding:16px 20px;border-radius:8px;display:flex;align-items:center;",
            "MEDIUM": "background-color:#fff3e0;border-left:6px solid #ff9800;color:#e65100;box-shadow:0 2px 8px rgba(255,152,0,0.08);margin-bottom:10px;padding:16px 20px;border-radius:8px;display:flex;align-items:center;",
            "LOW": "background-color:#e8f5e9;border-left:6px solid #4caf50;color:#1b5e20;box-shadow:0 2px 8px rgba(76,175,80,0.08);margin-bottom:10px;padding:16px 20px;border-radius:8px;display:flex;align-items:center;"
        }
        alert_icons = {
            "HIGH": "<span style='font-size:1.5em;margin-right:12px;'>🚨</span>",
            "MEDIUM": "<span style='font-size:1.5em;margin-right:12px;'>⚠️</span>",
            "LOW": "<span style='font-size:1.5em;margin-right:12px;'>ℹ️</span>"
        }
        for alert in alerts:
            st.markdown(f"""
                <div style='{alert_styles[alert['level']]}'>{alert_icons[alert['level']]}<div><b style='font-size:1.1em'>{alert['level']} ALERT</b><br><span style='font-size:1.05em'>{alert['message']}</span><br><span style='font-size:0.95em;color:#888;'>{alert['time']}</span></div></div>
            """, unsafe_allow_html=True)
        
        # System Logs
        st.subheader("📝 System Logs")
        logs = [
            {"type": "success", "msg": "Model prediction completed in 0.8s", "time": "2024-01-15 14:30:15"},
            {"type": "info", "msg": "Transaction TXN_001234 processed", "time": "2024-01-15 14:29:45"},
            {"type": "warning", "msg": "Risk score calculated: 0.75", "time": "2024-01-15 14:29:30"},
            {"type": "success", "msg": "Feature engineering completed", "time": "2024-01-15 14:29:15"},
            {"type": "info", "msg": "New transaction received", "time": "2024-01-15 14:29:00"}
        ]
        log_icons = {
            "success": "✅",
            "info": "📄",
            "warning": "⚠️",
            "error": "❌"
        }
        log_colors = {
            "success": "#4caf50",
            "info": "#1976d2",
            "warning": "#ff9800",
            "error": "#f44336"
        }
        st.markdown("""
        <style>
        .log-row {display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #2223;}
        .log-time {width:160px;color:#888;font-size:0.98em;}
        .log-icon {font-size:1.2em;margin-right:10px;}
        .log-msg {font-size:1.05em;}
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<div>", unsafe_allow_html=True)
        for log in logs:
            st.markdown(f"""
            <div class='log-row'>
                <span class='log-time'>{log['time']}</span>
                <span class='log-icon' style='color:{log_colors[log['type']]}'>{log_icons[log['type']]}</span>
                <span class='log-msg'>{log['msg']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    def analyst_review_tab(self):
        """Analyst review and feedback loop."""
        st.header("📝 Analyst Review & Feedback")
        # Load or generate transactions
        df = self.generate_sample_transactions(100)
        # Simulate a simple anomaly flag for demo (replace with your real flag)
        df['is_fraud_suspect'] = (df['amount'] > df['amount'].quantile(0.95))
        # Load feedback if exists
        feedback_path = Path("analyst_feedback.csv")
        if feedback_path.exists():
            feedback_df = pd.read_csv(feedback_path)
        else:
            feedback_df = pd.DataFrame(columns=['transaction_id', 'label', 'reviewer', 'timestamp'])
        # Merge feedback into df
        df = df.merge(feedback_df[['transaction_id', 'label', 'reviewer', 'timestamp']], on='transaction_id', how='left')
        # Show only flagged transactions
        flagged = df[df['is_fraud_suspect']]
        st.write(f"Flagged transactions: {len(flagged)}")
        for idx, row in flagged.iterrows():
            st.markdown(f"---\n**Transaction ID:** {row['transaction_id']} | **Amount:** {row['amount']:.2f} | **Sender:** {row['customer_id']} | **Status:** {'Reviewed' if pd.notnull(row['label']) else 'Pending'}")
            if pd.isnull(row['label']):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Fraud_{row['transaction_id']}", key=f"fraud_{row['transaction_id']}"):
                        self.save_feedback(row['transaction_id'], 'Fraud')
                        st.success("Labeled as Fraud!")
                        st.rerun()
                with col2:
                    if st.button(f"Not Fraud_{row['transaction_id']}", key=f"notfraud_{row['transaction_id']}"):
                        self.save_feedback(row['transaction_id'], 'Not Fraud')
                        st.success("Labeled as Not Fraud!")
                        st.rerun()
                with col3:
                    if st.button(f"Uncertain_{row['transaction_id']}", key=f"uncertain_{row['transaction_id']}"):
                        self.save_feedback(row['transaction_id'], 'Uncertain')
                        st.success("Labeled as Uncertain!")
                        st.rerun()
            else:
                st.info(f"Already reviewed: {row.get('label', 'N/A')} by {row.get('reviewer', 'N/A')} at {row.get('timestamp', 'N/A')}")

    def save_feedback(self, transaction_id, label):
        """Save analyst feedback to CSV."""
        feedback_path = Path("analyst_feedback.csv")
        reviewer = os.getenv("USER", "analyst")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame([[transaction_id, label, reviewer, timestamp]], columns=['transaction_id', 'label', 'reviewer', 'timestamp'])
        if feedback_path.exists():
            feedback_df = pd.read_csv(feedback_path)
            feedback_df = pd.concat([feedback_df, new_row], ignore_index=True)
        else:
            feedback_df = new_row
        feedback_df.to_csv(feedback_path, index=False)
    
    def fraud_intelligence_network(self):
        """Fraud Intelligence Network - Connect with other agents and systems."""
        st.header("🌐 Fraud Intelligence Network")
        st.markdown("Connect with other fraud detection agents and intelligence sources")
        
        # Network Status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Connected Agents", "3", delta="+1")
        
        with col2:
            st.metric("Active Alerts", "12", delta="+3")
        
        with col3:
            st.metric("Network Uptime", "99.8%", delta="+0.1%")
        
        # Agent Connections
        st.subheader("🤖 Connected Agents")
        
        agents = [
            {
                "name": "Jakarta Bank Consortium Agent",
                "status": "🟢 Online",
                "location": "Jakarta, Indonesia",
                "specialization": "BI-FAST fraud patterns",
                "last_update": "2 minutes ago"
            },
            {
                "name": "Singapore Regional Agent",
                "status": "🟢 Online", 
                "location": "Singapore",
                "specialization": "ASEAN fraud trends",
                "last_update": "5 minutes ago"
            },
            {
                "name": "Global AML Network Agent",
                "status": "🟡 Limited",
                "location": "Global",
                "specialization": "International money laundering",
                "last_update": "15 minutes ago"
            }
        ]
        
        for agent in agents:
            with st.expander(f"{agent['name']} - {agent['status']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Location:** {agent['location']}")
                    st.write(f"**Specialization:** {agent['specialization']}")
                with col2:
                    st.write(f"**Last Update:** {agent['last_update']}")
                    if st.button(f"Request Intel from {agent['name']}", key=f"intel_{agent['name']}"):
                        st.info("Requesting intelligence data...")
                        # Simulate API call
                        time.sleep(1)
                        st.success("Intelligence data received!")
        
        # Real-time Intelligence Feed
        st.subheader("📡 Real-time Intelligence Feed")
        
        intelligence_alerts = [
            {
                "source": "Jakarta Bank Consortium",
                "alert": "New mule account pattern detected in South Jakarta",
                "severity": "High",
                "timestamp": "2 minutes ago",
                "affected_banks": ["BCA", "Mandiri", "BNI"]
            },
            {
                "source": "Singapore Regional Agent", 
                "alert": "Cross-border fraud ring targeting Indonesian e-commerce",
                "severity": "Medium",
                "timestamp": "8 minutes ago",
                "affected_banks": ["DBS", "OCBC", "UOB"]
            },
            {
                "source": "Global AML Network",
                "alert": "Cryptocurrency-based money laundering via Indonesian exchanges",
                "severity": "High", 
                "timestamp": "12 minutes ago",
                "affected_banks": ["Multiple"]
            }
        ]
        
        for alert in intelligence_alerts:
            severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.write(f"{severity_color[alert['severity']]} **{alert['severity']}**")
                with col2:
                    st.write(f"**{alert['source']}:** {alert['alert']}")
                    st.caption(f"Affected: {alert['affected_banks']}")
                with col3:
                    st.caption(alert['timestamp'])
                st.divider()
        
        # Network Configuration
        st.subheader("⚙️ Network Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Endpoints")
            endpoints = {
                "BI-FAST API": "https://api.bi.go.id/fraud-alerts",
                "Bank Consortium": "https://api.indonesian-banks.com/shared-intel", 
                "Global Fraud Network": "https://api.fraud-intelligence.com/alerts"
            }
            
            for name, url in endpoints.items():
                st.text_input(f"{name} URL", value=url, key=f"endpoint_{name}")
        
        with col2:
            st.subheader("Connection Settings")
            st.checkbox("Auto-connect to new agents", value=True)
            st.checkbox("Share local fraud patterns", value=True)
            st.checkbox("Receive global alerts", value=True)
            st.slider("Update frequency (minutes)", 1, 60, 5)
        
        # Manual Intelligence Sharing
        st.subheader("📤 Share Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pattern_type = st.selectbox("Pattern Type", ["Account Takeover", "Mule Account", "Synthetic Identity", "Money Laundering"])
            description = st.text_area("Pattern Description")
        
        with col2:
            severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
            affected_banks = st.multiselect("Affected Banks", ["BCA", "Mandiri", "BNI", "BRI", "CIMB Niaga"])
        
        if st.button("🚀 Share with Network"):
            st.success("Intelligence shared with connected agents!")
            st.info("Other agents will receive this pattern within 30 seconds")
    
    def openai_playground_tab(self):
        """OpenAI Playground Integration - Advanced AI capabilities for fraud detection."""
        st.header("🤖 OpenAI Playground Integration")
        st.markdown("Leverage advanced AI capabilities for fraud detection analysis and insights")
        
        # API Configuration
        st.subheader("🔧 API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key to enable advanced AI features"
            )
            
            model_choice = st.selectbox(
                "Model",
                ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-3.5-turbo"],
                help="Choose the OpenAI model to use"
            )
            
            # Model information
            model_info = {
                "gpt-4o": "Latest and most capable model (best for complex tasks)",
                "gpt-4o-mini": "Fast and efficient (good balance of speed and capability)",
                "gpt-4.1-mini": "New GPT-4.1 variant (optimized for efficiency)",
                "gpt-4.1-nano": "Smallest GPT-4.1 model (fastest, most cost-effective)",
                "gpt-3.5-turbo": "Reliable and cost-effective (good for most tasks)"
            }
            
            if model_choice in model_info:
                st.info(f"ℹ️ {model_info[model_choice]}")
        
        with col2:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4000, 1000, 100)
        
        # Playground Features
        st.subheader("🎯 AI-Powered Features")
        
        feature_tabs = st.tabs([
            "📝 Code Generation",
            "📊 Data Analysis", 
            "📋 Report Generation",
            "🔍 Model Explanation",
            "💬 Custom Prompts"
        ])
        
        with feature_tabs[0]:
            st.subheader("📝 Generate Fraud Detection Code")
            
            code_prompt = st.text_area(
                "Describe the code you want to generate:",
                value="Generate Python code for detecting suspicious transaction patterns based on amount, time, and location",
                height=100
            )
            
            if st.button("🚀 Generate Code", key="gen_code"):
                if openai_api_key:
                    with st.spinner("Generating code..."):
                        try:
                            code = self.generate_code_with_openai(
                                code_prompt, openai_api_key, model_choice, temperature, max_tokens
                            )
                            st.code(code, language="python")
                            st.download_button(
                                "📥 Download Code",
                                code,
                                file_name="fraud_detection_code.py",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Error generating code: {e}")
                else:
                    st.warning("Please enter your OpenAI API key")
        
        with feature_tabs[1]:
            st.subheader("📊 AI-Powered Data Analysis")
            
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Transaction Pattern Analysis", "Fraud Trend Analysis", "Customer Behavior Analysis", "Risk Factor Analysis"]
            )
            
            if st.button("🔍 Analyze Data", key="analyze_data"):
                if openai_api_key:
                    with st.spinner("Analyzing data..."):
                        try:
                            analysis = self.analyze_data_with_openai(
                                analysis_type, openai_api_key, model_choice, temperature, max_tokens
                            )
                            st.markdown(analysis)
                        except Exception as e:
                            st.error(f"Error analyzing data: {e}")
                else:
                    st.warning("Please enter your OpenAI API key")
        
        with feature_tabs[2]:
            st.subheader("📋 Generate Fraud Analysis Report")
            
            report_type = st.selectbox(
                "Report Type",
                ["Daily Fraud Summary", "Weekly Trend Report", "Monthly Performance Report", "Custom Analysis Report"]
            )
            
            if st.button("📄 Generate Report", key="gen_report"):
                if openai_api_key:
                    with st.spinner("Generating report..."):
                        try:
                            report = self.generate_report_with_openai(
                                report_type, openai_api_key, model_choice, temperature, max_tokens
                            )
                            st.markdown(report)
                            st.download_button(
                                "📥 Download Report",
                                report,
                                file_name=f"{report_type.lower().replace(' ', '_')}.md",
                                mime="text/markdown"
                            )
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
                else:
                    st.warning("Please enter your OpenAI API key")
        
        with feature_tabs[3]:
            st.subheader("🔍 Explain Model Predictions")
            
            # Get sample transaction for explanation
            sample_transaction = {
                "amount": 1500.0,
                "transaction_type": "ONLINE",
                "location": "INTERNATIONAL",
                "hour": 23,
                "risk_score": 0.85
            }
            
            st.json(sample_transaction)
            
            if st.button("🤖 Explain Prediction", key="explain_pred"):
                if openai_api_key:
                    with st.spinner("Generating explanation..."):
                        try:
                            explanation = self.explain_prediction_with_openai(
                                sample_transaction, openai_api_key, model_choice, temperature, max_tokens
                            )
                            st.markdown(explanation)
                        except Exception as e:
                            st.error(f"Error generating explanation: {e}")
                else:
                    st.warning("Please enter your OpenAI API key")
        
        with feature_tabs[4]:
            st.subheader("💬 Custom AI Prompts")
            
            custom_prompt = st.text_area(
                "Enter your custom prompt:",
                value="Analyze the current fraud detection system and suggest improvements for better accuracy",
                height=150
            )
            
            if st.button("🚀 Send to OpenAI", key="custom_prompt"):
                if openai_api_key:
                    with st.spinner("Processing..."):
                        try:
                            response = self.custom_openai_prompt(
                                custom_prompt, openai_api_key, model_choice, temperature, max_tokens
                            )
                            st.markdown(response)
                        except Exception as e:
                            st.error(f"Error processing prompt: {e}")
                else:
                    st.warning("Please enter your OpenAI API key")
        
        # Usage Statistics
        st.subheader("📈 Usage Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Calls Today", "12", delta="+3")
        
        with col2:
            st.metric("Tokens Used", "2,847", delta="+156")
        
        with col3:
            st.metric("Cost Estimate", "$0.15", delta="+$0.02")
    
    def generate_code_with_openai(self, prompt, api_key, model, temperature, max_tokens):
        """Generate code using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            system_prompt = """You are an expert Python developer specializing in fraud detection systems. 
            Generate clean, well-documented code that follows best practices for financial applications.
            Include proper error handling, logging, and security considerations."""
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg:
                return f"❌ Model '{model}' not found or not accessible. Please try a different model (gpt-4o, gpt-4o-mini, gpt-4.1-mini, gpt-4.1-nano, or gpt-3.5-turbo)."
            elif "invalid_api_key" in error_msg:
                return "❌ Invalid API key. Please check your OpenAI API key."
            elif "quota_exceeded" in error_msg:
                return "❌ API quota exceeded. Please check your OpenAI account usage."
            else:
                return f"❌ Error: {error_msg}"
    
    def analyze_data_with_openai(self, analysis_type, api_key, model, temperature, max_tokens):
        """Analyze data using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            # Sample data for analysis
            sample_data = {
                "total_transactions": 1247,
                "fraud_count": 8,
                "fraud_rate": 0.64,
                "avg_amount": 245.67,
                "high_risk_transactions": 23
            }
            
            prompt = f"""Analyze the following fraud detection data and provide insights for {analysis_type}:
            
            Data: {sample_data}
            
            Please provide:
            1. Key insights and patterns
            2. Potential risk factors
            3. Recommendations for improvement
            4. Statistical analysis
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg:
                return f"❌ Model '{model}' not found or not accessible. Please try a different model."
            elif "invalid_api_key" in error_msg:
                return "❌ Invalid API key. Please check your OpenAI API key."
            else:
                return f"❌ Error: {error_msg}"
    
    def generate_report_with_openai(self, report_type, api_key, model, temperature, max_tokens):
        """Generate reports using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            prompt = f"""Generate a comprehensive {report_type} for a fraud detection system.
            
            Include:
            1. Executive Summary
            2. Key Metrics and Performance
            3. Fraud Trends and Patterns
            4. Risk Assessment
            5. Recommendations
            6. Action Items
            
            Format as markdown with proper headers and structure."""
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg:
                return f"❌ Model '{model}' not found or not accessible. Please try a different model."
            elif "invalid_api_key" in error_msg:
                return "❌ Invalid API key. Please check your OpenAI API key."
            else:
                return f"❌ Error: {error_msg}"
    
    def explain_prediction_with_openai(self, transaction, api_key, model, temperature, max_tokens):
        """Explain model predictions using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            prompt = f"""Explain why this transaction was flagged as high risk:
            
            Transaction: {transaction}
            
            Please explain:
            1. Which factors contributed to the high risk score
            2. What each factor means in terms of fraud risk
            3. Why this combination of factors is suspicious
            4. What actions should be taken
            5. How to verify if this is actually fraud
            
            Provide a clear, non-technical explanation suitable for business users."""
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg:
                return f"❌ Model '{model}' not found or not accessible. Please try a different model."
            elif "invalid_api_key" in error_msg:
                return "❌ Invalid API key. Please check your OpenAI API key."
            else:
                return f"❌ Error: {error_msg}"
    
    def custom_openai_prompt(self, prompt, api_key, model, temperature, max_tokens):
        """Send custom prompts to OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            system_prompt = """You are an expert fraud detection analyst with deep knowledge of:
            - Machine learning models for fraud detection
            - Financial transaction analysis
            - Risk assessment and management
            - Banking regulations and compliance
            - Data analysis and visualization
            
            Provide helpful, accurate, and actionable insights."""
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg:
                return f"❌ Model '{model}' not found or not accessible. Please try a different model."
            elif "invalid_api_key" in error_msg:
                return "❌ Invalid API key. Please check your OpenAI API key."
            else:
                return f"❌ Error: {error_msg}"
    
    def analyze_transaction(self, amount, transaction_type, location, card_present, 
                          customer_id, hour, merchant_category, device_type):
        """Analyze a single transaction."""
        if not self.detector:
            st.error("Model not loaded. Please load the model first.")
            return
        
        # Create transaction data
        transaction_data = pd.DataFrame([{
            'amount': amount,
            'transaction_type': transaction_type,
            'location': location,
            'card_present': card_present,
            'customer_id': customer_id,
            'hour': hour,
            'merchant_category': merchant_category,
            'device_type': device_type,
            'day_of_week': datetime.datetime.now().weekday(),
            'previous_fraud_flag': False,
            'account_age_days': 365,
            'balance_before': 5000,
            'balance_after': 5000 - amount,
            'is_fraud': False  # Ensure this column exists
        }])
        
        # Engineer features
        transaction_data = self.detector.engineer_bank_features(transaction_data)
        
        # Get prediction
        result = self.detector.predict_transaction_risk(transaction_data.iloc[0])
        
        if result:
            st.subheader("🔍 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_color = {
                    'HIGH_RISK': 'red',
                    'MEDIUM_RISK': 'orange',
                    'LOW_RISK': 'yellow',
                    'SAFE': 'green'
                }
                st.metric(
                    label="Risk Level",
                    value=result['risk_level'].replace('_', ' '),
                    delta_color="inverse" if result['risk_level'] in ['HIGH_RISK', 'MEDIUM_RISK'] else "normal"
                )
            
            with col2:
                st.metric(
                    label="Risk Probability",
                    value=f"{result['risk_probability']:.3f}"
                )
            
            with col3:
                st.metric(
                    label="Recommended Action",
                    value=result['recommended_action'].replace('_', ' ')
                )
            
            # Model predictions breakdown
            st.subheader("🤖 Model Predictions")
            model_results = result['model_predictions']
            
            for model_name, predictions in model_results.items():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**{model_name}**")
                with col2:
                    prediction_text = "Fraud" if predictions['prediction'] == 1 else "Legitimate"
                    prediction_color = "red" if predictions['prediction'] == 1 else "green"
                    st.markdown(f"<span style='color: {prediction_color}; font-weight: bold;'>{prediction_text}</span>", unsafe_allow_html=True)
                with col3:
                    fraud_prob = predictions['probability']
                    legitimate_prob = 1 - fraud_prob
                    st.write(f"Fraud: {fraud_prob:.3f}")
                with col4:
                    st.write(f"Legitimate: {legitimate_prob:.3f}")
            
            # Add explanation
            st.info("💡 **Note:** Risk level is determined by the highest fraud probability from any model. Even if models predict 'Legitimate', high fraud probabilities may still trigger risk alerts.")
    
    def plot_transaction_volume(self):
        """Plot transaction volume over time."""
        # Generate sample data
        hours = list(range(24))
        volumes = np.random.poisson(50, 24)  # Poisson distribution for realistic volumes
        
        fig = px.line(x=hours, y=volumes, 
                     title="Transaction Volume by Hour",
                     labels={'x': 'Hour of Day', 'y': 'Transactions'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_fraud_detection_rate(self):
        """Plot fraud detection rate over time."""
        # Generate sample data
        hours = list(range(24))
        detection_rates = np.random.uniform(0.95, 0.99, 24)
        
        fig = px.line(x=hours, y=detection_rates,
                     title="Fraud Detection Rate",
                     labels={'x': 'Hour of Day', 'y': 'Detection Rate'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_risk_distribution(self):
        """Plot risk distribution."""
        risk_levels = ['SAFE', 'LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']
        counts = [800, 150, 40, 10]
        
        fig = px.pie(values=counts, names=risk_levels,
                    title="Risk Distribution")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_model_performance(self):
        """Plot model performance metrics."""
        models = ['Logistic Regression', 'Random Forest', 'Isolation Forest']
        auc_scores = [0.85, 0.92, 0.78]
        
        fig = px.bar(x=models, y=auc_scores,
                    title="Model AUC Scores",
                    labels={'x': 'Model', 'y': 'AUC Score'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_hourly_patterns(self):
        """Plot hourly transaction patterns."""
        hours = list(range(24))
        legitimate = np.random.poisson(45, 24)
        fraudulent = np.random.poisson(2, 24)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hours, y=legitimate, name='Legitimate', marker_color='blue'))
        fig.add_trace(go.Bar(x=hours, y=fraudulent, name='Fraudulent', marker_color='red'))
        
        fig.update_layout(title="Hourly Transaction Patterns",
                         xaxis_title="Hour of Day",
                         yaxis_title="Number of Transactions",
                         barmode='stack',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_weekly_patterns(self):
        """Plot weekly transaction patterns."""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        volumes = [1200, 1100, 1150, 1250, 1300, 800, 600]
        
        fig = px.bar(x=days, y=volumes,
                    title="Weekly Transaction Volume",
                    labels={'x': 'Day of Week', 'y': 'Transactions'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_customer_risk_profiles(self):
        """Plot customer risk profiles."""
        risk_categories = ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
        customer_counts = [300, 120, 60, 20]
        
        fig = px.bar(x=risk_categories, y=customer_counts,
                    title="Customer Risk Distribution",
                    labels={'x': 'Risk Category', 'y': 'Number of Customers'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_transaction_amount_distribution(self):
        """Plot transaction amount distribution."""
        amounts = np.random.exponential(100, 1000)
        
        fig = px.histogram(x=amounts, nbins=50,
                          title="Transaction Amount Distribution",
                          labels={'x': 'Amount ($)', 'y': 'Frequency'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_model_comparison(self):
        """Plot model comparison metrics."""
        models = ['Logistic Regression', 'Random Forest', 'Isolation Forest']
        precision = [0.82, 0.89, 0.75]
        recall = [0.78, 0.85, 0.72]
        f1_score = [0.80, 0.87, 0.73]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=models, y=precision, name='Precision', marker_color='blue'))
        fig.add_trace(go.Bar(x=models, y=recall, name='Recall', marker_color='green'))
        fig.add_trace(go.Bar(x=models, y=f1_score, name='F1-Score', marker_color='orange'))
        
        fig.update_layout(title="Model Performance Comparison",
                         xaxis_title="Model",
                         yaxis_title="Score",
                         barmode='group',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_model_health(self):
        """Plot model health metrics."""
        metrics = ['Accuracy', 'Latency', 'Throughput', 'Memory Usage']
        values = [0.94, 0.8, 1200, 85]
        
        fig = px.bar(x=metrics, y=values,
                    title="Model Health Metrics",
                    labels={'x': 'Metric', 'y': 'Value'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_report(self):
        """Generate a comprehensive report."""
        st.success("📊 Report generated successfully!")
        st.info("Report includes: Transaction volume, fraud detection rates, model performance, and risk analysis.")

def fraudlabspro_screen_order(api_key, ip_address, email, amount, **kwargs):
    url = "https://api.fraudlabspro.com/v1/order/screen"
    params = {
        "key": api_key,
        "ip_address": ip_address,
        "email": email,
        "amount": amount,
    }
    params.update(kwargs)
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def main():
    """Main function to run the dashboard."""
    dashboard = FraudDetectionDashboard()
    dashboard.run_dashboard() 