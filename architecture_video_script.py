#!/usr/bin/env python3
"""
Architecture Video Creation Script
Creates a professional video showcasing the fraud detection system architecture
"""

import streamlit as st
import time
import random
from datetime import datetime

def create_architecture_video():
    """Create an architecture showcase video"""
    
    st.set_page_config(
        page_title="Fraud Detection System Architecture",
        page_icon="🏗️",
        layout="wide"
    )
    
    # Custom CSS for architecture video
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
    .arch-step {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .arch-step:hover {
        transform: translateX(10px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .arch-highlight {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .flow-diagram {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Architecture header
    st.markdown("""
    <div class="arch-header">
        <h1>🏗️ Fraud Detection System Architecture</h1>
        <p><strong>End-to-End Blockchain-Powered Fraud Detection</strong></p>
        <p>Real-time processing with AI, Blockchain, and Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'arch_step' not in st.session_state:
        st.session_state.arch_step = 0
    if 'show_animation' not in st.session_state:
        st.session_state.show_animation = False
    
    # Video controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎬 Start Architecture Video", key="start_video"):
            st.session_state.arch_step = 1
            st.session_state.show_animation = True
            st.rerun()
    
    with col2:
        if st.button("⏸️ Pause/Resume", key="pause_video"):
            st.session_state.show_animation = not st.session_state.show_animation
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset", key="reset_video"):
            st.session_state.arch_step = 0
            st.session_state.show_animation = False
            st.rerun()
    
    # Architecture flow diagram
    st.markdown("""
    <div class="flow-diagram">
        <h3>🔄 System Architecture Flow</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 2rem 0;">
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 2rem;">📄</div>
                <strong>Transaksi</strong><br>
                <small>Transaction Input</small>
            </div>
            <div style="font-size: 1.5rem;">➡️</div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 2rem;">⚡</div>
                <strong>Kafka</strong><br>
                <small>Stream Processing</small>
            </div>
            <div style="font-size: 1.5rem;">➡️</div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 2rem;">🤖</div>
                <strong>Multi-model AI</strong><br>
                <small>Ensemble Learning</small>
            </div>
            <div style="font-size: 1.5rem;">➡️</div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 2rem;">🔗</div>
                <strong>Blockchain</strong><br>
                <small>Validation</small>
            </div>
            <div style="font-size: 1.5rem;">➡️</div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 2rem;">📊</div>
                <strong>Dashboard</strong><br>
                <small>Decision</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture steps
    if st.session_state.arch_step >= 1:
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
    
    if st.session_state.arch_step >= 2:
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
    
    if st.session_state.arch_step >= 3:
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
    
    if st.session_state.arch_step >= 4:
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
    
    if st.session_state.arch_step >= 5:
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
    
    # Key features highlight
    if st.session_state.arch_step >= 5:
        st.markdown("""
        <div class="arch-highlight">
            <h3>🎯 Key System Features</h3>
            <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">⚡</div>
                    <strong>Speed</strong><br>
                    <small>500+ TPS</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">🎯</div>
                    <strong>Accuracy</strong><br>
                    <small>95%+</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">🔍</div>
                    <strong>Transparency</strong><br>
                    <small>XAI</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">🔒</div>
                    <strong>Security</strong><br>
                    <small>Blockchain</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Technology stack
    if st.session_state.arch_step >= 5:
        st.subheader("🛠️ Technology Stack")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info("**Data Processing**\n- Apache Kafka\n- Apache Spark\n- Real-time streaming")
        
        with col2:
            st.info("**Machine Learning**\n- Scikit-learn\n- Ensemble models\n- Feature engineering")
        
        with col3:
            st.info("**Blockchain**\n- Custom implementation\n- Smart contracts\n- Proof of work")
        
        with col4:
            st.info("**Frontend**\n- Streamlit\n- Real-time updates\n- Interactive dashboards")
    
    # Auto-advance for video effect
    if st.session_state.show_animation and st.session_state.arch_step < 5:
        # Use st.empty() to create a placeholder for auto-advance
        auto_advance_placeholder = st.empty()
        with auto_advance_placeholder:
            with st.spinner("⏳ Auto-advancing..."):
                time.sleep(2)
        st.session_state.arch_step += 1
        st.rerun()
    
    # Video recording instructions
    st.markdown("---")
    st.markdown("""
    ### 🎬 Video Recording Instructions
    
    **For Architecture Video:**
    1. **Start the video** by clicking "Start Architecture Video"
    2. **Record the flow** as each step appears automatically
    3. **Highlight key features** and technology stack
    4. **Show the complete flow** from transaction to decision
    5. **Keep recording under 15 seconds** for maximum impact
    
    **Key Points to Emphasize:**
    - Real-time processing capabilities
    - Multi-model AI accuracy
    - Blockchain security features
    - End-to-end transparency
    """)

if __name__ == "__main__":
    create_architecture_video() 