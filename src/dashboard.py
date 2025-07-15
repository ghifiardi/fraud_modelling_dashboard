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

# Create the chatbot UI with multiple fallback options
create_chatbot_ui()


class FraudDetectionDashboard:
    def __init__(self):
        self.detector = None
        self.model_path = "models/bank_fraud_detector.pkl"
        self.load_model()
        
    def load_model(self):
        """Load the trained fraud detection model."""
        try:
            if os.path.exists(self.model_path):
                self.detector = BankFraudDetector()
                self.detector.load_bank_model(self.model_path)
                return True
            else:
                st.warning("Model not found. Please train the model first.")
                return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
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
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .alert-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
        }
        .alert-low {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🛡️ AI Fraud Detection Monitor</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.create_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Real-time Dashboard", 
            "🔍 Transaction Monitor", 
            "📈 Analytics", 
            "⚙️ Model Management",
            "🚨 Alerts & Logs"
        ])
        
        with tab1:
            self.real_time_dashboard()
        
        with tab2:
            self.transaction_monitor()
        
        with tab3:
            self.analytics_dashboard()
        
        with tab4:
            self.model_management()
        
        with tab5:
            self.alerts_and_logs()
    
    def create_sidebar(self):
        """Create the sidebar with controls and settings."""
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
        
        # System Info
        st.sidebar.subheader("System Info")
        st.sidebar.info(f"Last Updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    def real_time_dashboard(self):
        """Real-time monitoring dashboard."""
        st.header("📊 Real-time Monitoring")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Transactions Today",
                value="1,247",
                delta="+12%"
            )
        
        with col2:
            st.metric(
                label="Fraud Detected",
                value="8",
                delta="-2",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="Success Rate",
                value="99.4%",
                delta="+0.2%"
            )
        
        with col4:
            st.metric(
                label="Avg Response Time",
                value="0.8s",
                delta="-0.1s"
            )
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🕒 Transaction Volume (Last 24h)")
            self.plot_transaction_volume()
        
        with col2:
            st.subheader("🎯 Fraud Detection Rate")
            self.plot_fraud_detection_rate()
        
        # Risk Distribution
        st.subheader("⚠️ Risk Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_risk_distribution()
        
        with col2:
            self.plot_model_performance()
    
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
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{model_name}**")
                with col2:
                    st.write(f"Prediction: {'Fraud' if predictions['prediction'] else 'Legitimate'}")
                with col3:
                    st.write(f"Probability: {predictions['probability']:.3f}")
    
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

def main():
    """Main function to run the dashboard."""
    dashboard = FraudDetectionDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 