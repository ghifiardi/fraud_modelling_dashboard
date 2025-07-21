#!/usr/bin/env python3
"""
Simple Architecture Video - Fraud Detection System
Professional architecture showcase without complex animations
"""

import streamlit as st
import time

def create_simple_architecture_video():
    """Create a simple but effective architecture showcase"""
    
    st.set_page_config(
        page_title="Fraud Detection Architecture",
        page_icon="🏗️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .arch-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .arch-flow {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
    }
    .arch-step {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    .feature-highlight {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="arch-header">
        <h1>🏗️ Fraud Detection System Architecture</h1>
        <p><strong>End-to-End Blockchain-Powered Fraud Detection</strong></p>
        <p>Real-time processing with AI, Blockchain, and Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture Flow Diagram
    st.markdown("""
    <div class="arch-flow">
        <h3>🔄 Complete System Architecture Flow</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 2rem 0; flex-wrap: wrap;">
            <div style="text-align: center; flex: 1; min-width: 120px; margin: 0.5rem;">
                <div style="font-size: 3rem;">📄</div>
                <strong>Transaksi</strong><br>
                <small>Transaction Input</small>
            </div>
            <div style="font-size: 2rem; margin: 0.5rem;">➡️</div>
            <div style="text-align: center; flex: 1; min-width: 120px; margin: 0.5rem;">
                <div style="font-size: 3rem;">⚡</div>
                <strong>Kafka & Spark</strong><br>
                <small>Stream Processing</small>
            </div>
            <div style="font-size: 2rem; margin: 0.5rem;">➡️</div>
            <div style="text-align: center; flex: 1; min-width: 120px; margin: 0.5rem;">
                <div style="font-size: 3rem;">🤖</div>
                <strong>Multi-model AI</strong><br>
                <small>Ensemble Learning</small>
            </div>
            <div style="font-size: 2rem; margin: 0.5rem;">➡️</div>
            <div style="text-align: center; flex: 1; min-width: 120px; margin: 0.5rem;">
                <div style="font-size: 3rem;">🔗</div>
                <strong>Blockchain</strong><br>
                <small>Validation</small>
            </div>
            <div style="font-size: 2rem; margin: 0.5rem;">➡️</div>
            <div style="text-align: center; flex: 1; min-width: 120px; margin: 0.5rem;">
                <div style="font-size: 3rem;">📊</div>
                <strong>Dashboard</strong><br>
                <small>Decision</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("""
    <div class="feature-highlight">
        <h3>🎯 Key System Features</h3>
        <div style="display: flex; justify-content: space-around; margin: 1rem 0; flex-wrap: wrap;">
            <div style="text-align: center; margin: 1rem;">
                <div style="font-size: 2.5rem;">⚡</div>
                <strong>Speed</strong><br>
                <small>500+ TPS</small>
            </div>
            <div style="text-align: center; margin: 1rem;">
                <div style="font-size: 2.5rem;">🎯</div>
                <strong>Accuracy</strong><br>
                <small>95%+</small>
            </div>
            <div style="text-align: center; margin: 1rem;">
                <div style="font-size: 2.5rem;">🔍</div>
                <strong>Transparency</strong><br>
                <small>XAI</small>
            </div>
            <div style="text-align: center; margin: 1rem;">
                <div style="font-size: 2.5rem;">🔒</div>
                <strong>Security</strong><br>
                <small>Blockchain</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture Steps
    st.subheader("🏗️ Architecture Components")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="arch-step">
            <h4>📄 Step 1: Transaction Input</h4>
            <p><strong>Real-time transaction data ingestion</strong></p>
            <ul>
                <li>Payment transactions from multiple sources</li>
                <li>Customer behavior data</li>
                <li>Merchant information</li>
                <li>Geographic and temporal patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="arch-step">
            <h4>⚡ Step 2: Kafka & Spark Processing</h4>
            <p><strong>High-speed stream processing for real-time analysis</strong></p>
            <ul>
                <li><strong>Kafka:</strong> Message queuing and event streaming</li>
                <li><strong>Spark:</strong> Distributed data processing</li>
                <li><strong>Speed:</strong> 500+ transactions per second</li>
                <li><strong>Scalability:</strong> Horizontal scaling capability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="arch-step">
            <h4>🤖 Step 3: Multi-model AI & Ensemble Learning</h4>
            <p><strong>Advanced machine learning for fraud detection</strong></p>
            <ul>
                <li><strong>Random Forest:</strong> Pattern recognition</li>
                <li><strong>Logistic Regression:</strong> Baseline classification</li>
                <li><strong>Isolation Forest:</strong> Anomaly detection</li>
                <li><strong>Ensemble:</strong> 95%+ accuracy through model combination</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="arch-step">
            <h4>🔗 Step 4: Blockchain Validation</h4>
            <p><strong>Immutable transaction validation and smart contracts</strong></p>
            <ul>
                <li><strong>Proof of Work:</strong> Cryptographic validation</li>
                <li><strong>Smart Contracts:</strong> Automated fraud rules</li>
                <li><strong>Merkle Trees:</strong> Efficient data verification</li>
                <li><strong>Immutability:</strong> Tamper-proof transaction records</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="arch-step">
            <h4>📊 Step 5: Dashboard & Decision Making</h4>
            <p><strong>Real-time monitoring and decision support</strong></p>
            <ul>
                <li><strong>Live Monitoring:</strong> Real-time transaction feed</li>
                <li><strong>Risk Assessment:</strong> Color-coded risk levels</li>
                <li><strong>Analytics:</strong> Performance metrics and trends</li>
                <li><strong>Alerts:</strong> Instant fraud notifications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="arch-step">
            <h4>🔍 Explainable AI (XAI)</h4>
            <p><strong>Transparency in decision making</strong></p>
            <ul>
                <li><strong>Audit Trail:</strong> Complete decision history</li>
                <li><strong>Feature Importance:</strong> Understand model decisions</li>
                <li><strong>Compliance:</strong> Regulatory requirements met</li>
                <li><strong>Trust:</strong> Stakeholder confidence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Technology Stack
    st.subheader("🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.info("**Data Processing**\n- Apache Kafka\n- Apache Spark\n- Real-time streaming")
    
    with tech_col2:
        st.info("**Machine Learning**\n- Scikit-learn\n- Ensemble models\n- Feature engineering")
    
    with tech_col3:
        st.info("**Blockchain**\n- Custom implementation\n- Smart contracts\n- Proof of work")
    
    with tech_col4:
        st.info("**Frontend**\n- Streamlit\n- Real-time updates\n- Interactive dashboards")
    
    # Video recording instructions
    st.markdown("---")
    st.markdown("""
    ### 🎬 Video Recording Instructions
    
    **For Architecture Video:**
    1. **Record the complete flow** from top to bottom
    2. **Highlight key features** and technology stack
    3. **Show the architecture diagram** prominently
    4. **Keep recording under 15 seconds** for maximum impact
    
    **Key Points to Emphasize:**
    - Real-time processing capabilities (500+ TPS)
    - Multi-model AI accuracy (95%+)
    - Blockchain security features
    - End-to-end transparency with XAI
    - Complete technology stack
    """)

if __name__ == "__main__":
    create_simple_architecture_video() 