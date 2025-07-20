#!/usr/bin/env python3
"""
Dashboard Demo Automation Script
Creates automated actions for recording a professional demo video
"""

import streamlit as st
import time
import random
import webbrowser
from datetime import datetime

def create_demo_script():
    """Create a demo script with automated actions"""
    
    demo_script = """
# 🎬 Dashboard Demo Script
# Duration: 5-10 seconds
# Purpose: Show real-time fraud detection in action

## Demo Flow:
1. Open dashboard (http://localhost:8501)
2. Wait for page to load (2 seconds)
3. Click "Generate Transaction" button
4. Show real-time transaction feed
5. Click "Mine Block" button
6. Show blockchain updates
7. Highlight fraud alerts
8. End with success metrics

## Recording Instructions:
1. Use QuickTime Player (built into macOS)
2. File → New Screen Recording
3. Select the browser window with dashboard
4. Click record and follow the demo script
5. Keep recording under 10 seconds
6. Export as MP4 or GIF

## Automated Demo Actions:
"""
    
    return demo_script

def create_enhanced_demo_app():
    """Create an enhanced demo version with automatic actions"""
    
    st.set_page_config(
        page_title="Blockchain Fraud Detection - DEMO",
        page_icon="🎬",
        layout="wide"
    )
    
    # Custom CSS for demo mode
    st.markdown("""
    <style>
    .demo-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .demo-alert {
        background: #FFE66D;
        border-left: 5px solid #FF6B6B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .demo-success {
        background: #95E1D3;
        border-left: 5px solid #4ECDC4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Demo header
    st.markdown("""
    <div class="demo-header">
        <h1>🔗 Blockchain Fraud Detection Dashboard</h1>
        <p><strong>DEMO MODE</strong> - Real-time Fraud Detection in Action</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for demo
    if 'demo_step' not in st.session_state:
        st.session_state.demo_step = 0
    if 'demo_transactions' not in st.session_state:
        st.session_state.demo_transactions = []
    if 'demo_blocks' not in st.session_state:
        st.session_state.demo_blocks = []
    
    # Demo controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Demo", key="start_demo"):
            st.session_state.demo_step = 1
            st.rerun()
    
    with col2:
        if st.button("⚡ Generate Transaction", key="gen_tx"):
            # Generate demo transaction
            tx = {
                'id': f"TX{random.randint(1000, 9999)}",
                'customer': f"CUST{random.randint(100, 999)}",
                'amount': round(random.uniform(50, 5000), 2),
                'merchant': random.choice(['Amazon', 'Netflix', 'Uber', 'Starbucks']),
                'risk_score': round(random.uniform(0.1, 0.9), 2),
                'risk_level': random.choice(['SAFE', 'LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']),
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.demo_transactions.append(tx)
            st.session_state.demo_step = 2
            st.rerun()
    
    with col3:
        if st.button("⛏️ Mine Block", key="mine_block"):
            if st.session_state.demo_transactions:
                block = {
                    'index': len(st.session_state.demo_blocks) + 1,
                    'transactions': len(st.session_state.demo_transactions),
                    'hash': f"0000{random.randint(100000, 999999)}",
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.demo_blocks.append(block)
                st.session_state.demo_transactions = []
                st.session_state.demo_step = 3
                st.rerun()
    
    # Demo progress indicator
    if st.session_state.demo_step > 0:
        progress_bar = st.progress(0)
        progress_bar.progress(st.session_state.demo_step / 3)
        
        # Demo step indicators
        steps = ["🚀 Demo Started", "⚡ Transaction Generated", "⛏️ Block Mined"]
        for i, step in enumerate(steps):
            if i < st.session_state.demo_step:
                st.success(step)
            else:
                st.info(step)
    
    # Real-time transaction feed
    st.subheader("📊 Real-time Transaction Feed")
    
    if st.session_state.demo_transactions:
        for tx in st.session_state.demo_transactions[-5:]:  # Show last 5
            risk_color = {
                'SAFE': '🟢',
                'LOW_RISK': '🟡', 
                'MEDIUM_RISK': '🟠',
                'HIGH_RISK': '🔴'
            }
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**{tx['id']}**")
            with col2:
                st.write(f"${tx['amount']:,.2f}")
            with col3:
                st.write(f"{risk_color[tx['risk_level']]} {tx['risk_level']}")
            with col4:
                st.write(f"Risk: {tx['risk_score']:.1%}")
    
    # Blockchain status
    st.subheader("🔗 Blockchain Status")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chain Length", len(st.session_state.demo_blocks))
    with col2:
        st.metric("Pending TX", len(st.session_state.demo_transactions))
    with col3:
        st.metric("Total TX", sum(block.get('transactions', 0) for block in st.session_state.demo_blocks))
    with col4:
        st.metric("Mining Rate", "0.2 blocks/sec")
    
    # Recent blocks
    if st.session_state.demo_blocks:
        st.subheader("📦 Recent Blocks")
        for block in st.session_state.demo_blocks[-3:]:  # Show last 3
            st.info(f"Block #{block['index']} - {block['transactions']} transactions - Hash: {block['hash']}")
    
    # Fraud alerts
    st.subheader("🚨 Fraud Alerts")
    
    high_risk_txs = [tx for tx in st.session_state.demo_transactions if tx['risk_level'] in ['MEDIUM_RISK', 'HIGH_RISK']]
    
    if high_risk_txs:
        for tx in high_risk_txs:
            st.markdown(f"""
            <div class="demo-alert">
                <strong>🚨 HIGH RISK TRANSACTION DETECTED!</strong><br>
                Transaction {tx['id']} - ${tx['amount']:,.2f} at {tx['merchant']}<br>
                Risk Level: {tx['risk_level']} ({tx['risk_score']:.1%})
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="demo-success">
            <strong>✅ All transactions are safe!</strong><br>
            No fraud detected in recent transactions.
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-refresh for demo effect
    if st.session_state.demo_step > 0:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    create_enhanced_demo_app() 