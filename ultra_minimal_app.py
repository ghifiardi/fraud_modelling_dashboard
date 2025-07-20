#!/usr/bin/env python3
"""
Ultra-Minimal Blockchain Fraud Detection Dashboard
No external dependencies - should work on any Streamlit Cloud environment
"""

import streamlit as st
import time
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Blockchain Fraud Detection",
    page_icon="🔗",
    layout="wide"
)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []

if 'blocks' not in st.session_state:
    st.session_state.blocks = []

def main():
    st.title("🔗 Blockchain Fraud Detection Dashboard")
    st.markdown("**Status**: ✅ Ultra-Minimal Version - No External Dependencies")
    
    # Sidebar controls
    st.sidebar.title("🔧 Controls")
    
    if st.sidebar.button("🚀 Generate Transaction"):
        # Create a simple transaction
        tx = {
            'id': f"TX{random.randint(10000, 99999)}",
            'customer': f"CUST{random.randint(1000, 9999)}",
            'amount': round(random.uniform(10, 5000), 2),
            'merchant': random.choice(['amazon', 'netflix', 'uber', 'starbucks']),
            'location': random.choice(['New York', 'London', 'Tokyo', 'Singapore']),
            'risk_score': round(random.uniform(0, 1), 2),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        
        # Determine risk level
        if tx['risk_score'] < 0.3:
            tx['risk_level'] = '🟢 SAFE'
        elif tx['risk_score'] < 0.5:
            tx['risk_level'] = '🟡 LOW_RISK'
        elif tx['risk_score'] < 0.7:
            tx['risk_level'] = '🟠 MEDIUM_RISK'
        else:
            tx['risk_level'] = '🔴 HIGH_RISK'
        
        st.session_state.transactions.append(tx)
        st.sidebar.success(f"Transaction {tx['id']} added!")
    
    if st.sidebar.button("⛏️ Mine Block"):
        if st.session_state.transactions:
            # Create a block with pending transactions
            block = {
                'index': len(st.session_state.blocks),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'transactions': st.session_state.transactions.copy(),
                'hash': f"BLOCK_{random.randint(100000, 999999)}"
            }
            st.session_state.blocks.append(block)
            st.session_state.transactions = []  # Clear pending transactions
            st.sidebar.success(f"Block {block['index']} mined!")
        else:
            st.sidebar.warning("No pending transactions to mine")
    
    if st.sidebar.button("🔄 Reset"):
        st.session_state.transactions = []
        st.session_state.blocks = []
        st.sidebar.success("Blockchain reset!")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Blocks", len(st.session_state.blocks))
    
    with col2:
        st.metric("Pending TX", len(st.session_state.transactions))
    
    with col3:
        total_tx = sum(len(block['transactions']) for block in st.session_state.blocks)
        st.metric("Total TX", total_tx)
    
    with col4:
        if st.session_state.blocks:
            st.metric("Last Block", f"#{st.session_state.blocks[-1]['index']}")
        else:
            st.metric("Last Block", "None")
    
    # Blockchain status
    st.subheader("📊 Blockchain Status")
    st.write(f"**Chain Length:** {len(st.session_state.blocks)} blocks")
    st.write(f"**Pending Transactions:** {len(st.session_state.transactions)}")
    st.write(f"**Total Transactions:** {total_tx}")
    if st.session_state.blocks:
        st.write(f"**Last Block Hash:** `{st.session_state.blocks[-1]['hash']}`")
    
    # Recent transactions
    st.subheader("💳 Recent Transactions")
    if st.session_state.transactions:
        for tx in st.session_state.transactions[-5:]:  # Show last 5
            st.write(f"**{tx['id']}** - {tx['customer']} - ${tx['amount']} - {tx['merchant']} - {tx['location']} - {tx['risk_level']}")
            st.write(f"Time: {tx['timestamp']} | Risk Score: {tx['risk_score']}")
            st.divider()
    else:
        st.info("No pending transactions. Click 'Generate Transaction' to start!")
    
    # Block explorer
    st.subheader("🔍 Block Explorer")
    if st.session_state.blocks:
        for block in st.session_state.blocks:
            with st.expander(f"Block {block['index']} - {block['timestamp']}"):
                st.write(f"**Hash:** {block['hash']}")
                st.write(f"**Transactions:** {len(block['transactions'])}")
                if block['transactions']:
                    st.write("**Transaction Details:**")
                    for tx in block['transactions']:
                        st.write(f"- {tx['id']}: ${tx['amount']} ({tx['risk_level']})")
    else:
        st.info("No blocks mined yet. Generate transactions and mine blocks!")
    
    # Risk distribution
    if st.session_state.blocks:
        st.subheader("📈 Risk Distribution")
        risk_counts = {'🟢 SAFE': 0, '🟡 LOW_RISK': 0, '🟠 MEDIUM_RISK': 0, '🔴 HIGH_RISK': 0}
        
        for block in st.session_state.blocks:
            for tx in block['transactions']:
                risk_counts[tx['risk_level']] += 1
        
        # Display risk distribution
        cols = st.columns(4)
        for i, (risk, count) in enumerate(risk_counts.items()):
            with cols[i]:
                st.metric(risk, count)
    
    # Footer
    st.markdown("---")
    st.markdown("**Ultra-Minimal Version** - No external dependencies required!")
    st.markdown("Built with pure Python and Streamlit for maximum compatibility.")

if __name__ == "__main__":
    main() 