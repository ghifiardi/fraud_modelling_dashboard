#!/usr/bin/env python3
"""
Simple Blockchain Fraud Detection Dashboard
Optimized for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import random

# Page configuration
st.set_page_config(
    page_title="Blockchain Fraud Detection",
    page_icon="🔗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .blockchain-status {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enums and Data Classes
class RiskLevel(Enum):
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"

@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    amount: float
    merchant_id: str
    timestamp: float
    location: str
    payment_method: str
    risk_score: float
    risk_level: RiskLevel
    fraud_probability: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'transaction_id': self.transaction_id,
            'customer_id': self.customer_id,
            'amount': self.amount,
            'merchant_id': self.merchant_id,
            'timestamp': self.timestamp,
            'location': self.location,
            'payment_method': self.payment_method,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value,
            'fraud_probability': self.fraud_probability,
            'metadata': self.metadata
        }

class SimpleBlockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = 4
        self.create_genesis_block()
    
    def create_genesis_block(self):
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': [],
            'previous_hash': '0',
            'nonce': 0,
            'hash': 'genesis_hash'
        }
        self.chain.append(genesis_block)
    
    def add_transaction(self, transaction: Transaction):
        self.pending_transactions.append(transaction)
        return True
    
    def mine_block(self):
        if not self.pending_transactions:
            return None
        
        block = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'transactions': [tx.to_dict() for tx in self.pending_transactions],
            'previous_hash': self.chain[-1]['hash'],
            'nonce': 0
        }
        
        # Simple proof of work
        target = "0" * self.difficulty
        while block['hash'][:self.difficulty] != target:
            block['nonce'] += 1
            block_string = json.dumps(block, default=str, sort_keys=True)
            block['hash'] = hashlib.sha256(block_string.encode()).hexdigest()
        
        self.chain.append(block)
        self.pending_transactions = []
        return block
    
    def get_status(self):
        return {
            'chain_length': len(self.chain),
            'pending_transactions': len(self.pending_transactions),
            'total_transactions': sum(len(block.get('transactions', [])) for block in self.chain),
            'last_block_hash': self.chain[-1]['hash'] if self.chain else 'None'
        }

# Initialize blockchain
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = SimpleBlockchain()

if 'transactions' not in st.session_state:
    st.session_state.transactions = []

def generate_random_transaction():
    """Generate a random transaction for demo purposes"""
    locations = ['New York', 'London', 'Tokyo', 'Singapore', 'Sydney', 'Paris']
    payment_methods = ['credit_card', 'debit_card', 'mobile_payment', 'bank_transfer']
    merchants = ['amazon', 'netflix', 'uber', 'starbucks', 'walmart', 'target']
    
    risk_score = random.uniform(0, 1)
    if risk_score < 0.3:
        risk_level = RiskLevel.SAFE
    elif risk_score < 0.5:
        risk_level = RiskLevel.LOW_RISK
    elif risk_score < 0.7:
        risk_level = RiskLevel.MEDIUM_RISK
    else:
        risk_level = RiskLevel.HIGH_RISK
    
    return Transaction(
        transaction_id=f"TX{random.randint(10000, 99999)}",
        customer_id=f"CUST{random.randint(1000, 9999)}",
        amount=random.uniform(10, 5000),
        merchant_id=random.choice(merchants),
        timestamp=time.time(),
        location=random.choice(locations),
        payment_method=random.choice(payment_methods),
        risk_score=risk_score,
        risk_level=risk_level,
        fraud_probability=risk_score * 0.8,
        metadata={'device_id': f"DEV{random.randint(100, 999)}"}
    )

def main():
    st.markdown('<h1 class="main-header">🔗 Blockchain Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    if st.sidebar.button("🚀 Generate Transaction"):
        tx = generate_random_transaction()
        st.session_state.transactions.append(tx)
        st.session_state.blockchain.add_transaction(tx)
        st.sidebar.success(f"Transaction {tx.transaction_id} added!")
    
    if st.sidebar.button("⛏️ Mine Block"):
        if st.session_state.pending_transactions:
            block = st.session_state.blockchain.mine_block()
            if block:
                st.sidebar.success(f"Block {block['index']} mined!")
        else:
            st.sidebar.warning("No pending transactions to mine")
    
    if st.sidebar.button("🔄 Reset Blockchain"):
        st.session_state.blockchain = SimpleBlockchain()
        st.session_state.transactions = []
        st.sidebar.success("Blockchain reset!")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Chain Length", st.session_state.blockchain.get_status()['chain_length'])
    
    with col2:
        st.metric("Pending TX", st.session_state.blockchain.get_status()['pending_transactions'])
    
    with col3:
        st.metric("Total TX", st.session_state.blockchain.get_status()['total_transactions'])
    
    with col4:
        status = st.session_state.blockchain.get_status()
        st.metric("Last Block", status['last_block_hash'][:8] + "...")
    
    # Blockchain Status
    st.markdown('<div class="blockchain-status">', unsafe_allow_html=True)
    st.subheader("📊 Blockchain Status")
    status = st.session_state.blockchain.get_status()
    st.write(f"**Chain Length:** {status['chain_length']} blocks")
    st.write(f"**Pending Transactions:** {status['pending_transactions']}")
    st.write(f"**Total Transactions:** {status['total_transactions']}")
    st.write(f"**Last Block Hash:** `{status['last_block_hash']}`")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent Transactions
    st.subheader("💳 Recent Transactions")
    if st.session_state.transactions:
        # Create DataFrame for display
        tx_data = []
        for tx in st.session_state.transactions[-10:]:  # Show last 10
            tx_data.append({
                'ID': tx.transaction_id,
                'Customer': tx.customer_id,
                'Amount': f"${tx.amount:.2f}",
                'Merchant': tx.merchant_id,
                'Location': tx.location,
                'Risk Score': f"{tx.risk_score:.2%}",
                'Risk Level': tx.risk_level.value.upper(),
                'Time': datetime.fromtimestamp(tx.timestamp).strftime('%H:%M:%S')
            })
        
        df = pd.DataFrame(tx_data)
        st.dataframe(df, use_container_width=True)
        
        # Risk Level Distribution
        risk_counts = {}
        for tx in st.session_state.transactions:
            risk = tx.risk_level.value
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        if risk_counts:
            fig = px.pie(
                values=list(risk_counts.values()),
                names=list(risk_counts.keys()),
                title="Risk Level Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transactions yet. Click 'Generate Transaction' to start!")
    
    # Block Explorer
    st.subheader("🔍 Block Explorer")
    if len(st.session_state.blockchain.chain) > 1:  # More than genesis block
        for i, block in enumerate(st.session_state.blockchain.chain[1:], 1):  # Skip genesis
            with st.expander(f"Block {block['index']} - {datetime.fromtimestamp(block['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}"):
                st.json(block)
    else:
        st.info("No blocks mined yet. Generate transactions and mine blocks to see the blockchain!")
    
    # Auto-refresh
    if st.session_state.transactions:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main() 