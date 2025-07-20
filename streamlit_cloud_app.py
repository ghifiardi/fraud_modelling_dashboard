#!/usr/bin/env python3
"""
Blockchain Fraud Detection Dashboard - Streamlit Cloud Deployment
Standalone version that includes blockchain functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import threading
import queue
import random

# Page configuration
st.set_page_config(
    page_title="Blockchain Fraud Detection Dashboard",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .transaction-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-high { color: #FF6B6B; font-weight: bold; }
    .risk-medium { color: #FFA726; font-weight: bold; }
    .risk-low { color: #66BB6A; font-weight: bold; }
    .risk-safe { color: #42A5F5; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Enums and Data Classes
class TransactionStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    FRAUDULENT = "fraudulent"

class RiskLevel(Enum):
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"

@dataclass
class Transaction:
    """Represents a financial transaction with fraud detection data"""
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
    status: TransactionStatus
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

@dataclass
class Block:
    """Represents a block in the blockchain"""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int
    merkle_root: str
    block_hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate the hash of the block"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root
        }, default=str, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """Mine the block with proof of work"""
        target = "0" * difficulty
        while self.block_hash[:difficulty] != target:
            self.nonce += 1
            self.block_hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root,
            'block_hash': self.block_hash
        }

class SmartContract:
    """Smart contract for fraud detection rules"""
    
    def __init__(self):
        self.rules = {
            'high_amount_threshold': 10000.0,
            'velocity_threshold': 5,  # transactions per hour
            'location_mismatch_penalty': 0.3,
            'night_transaction_penalty': 0.2,
            'new_merchant_penalty': 0.15,
            'international_penalty': 0.25
        }
    
    def validate_transaction(self, transaction: Transaction, customer_history: List[Transaction]) -> Dict[str, Any]:
        """Validate transaction using smart contract rules"""
        validation_result = {
            'is_valid': True,
            'risk_factors': [],
            'risk_score_adjustment': 0.0,
            'recommendation': 'allow'
        }
        
        # Rule 1: High amount check
        if transaction.amount > self.rules['high_amount_threshold']:
            validation_result['risk_factors'].append('high_amount')
            validation_result['risk_score_adjustment'] += 0.3
        
        # Rule 2: Velocity check (transactions per hour)
        recent_transactions = [
            tx for tx in customer_history 
            if tx.timestamp > transaction.timestamp - 3600  # Last hour
        ]
        if len(recent_transactions) > self.rules['velocity_threshold']:
            validation_result['risk_factors'].append('high_velocity')
            validation_result['risk_score_adjustment'] += 0.4
        
        # Rule 3: Location mismatch
        if customer_history:
            last_location = customer_history[-1].location
            if last_location != transaction.location:
                validation_result['risk_factors'].append('location_mismatch')
                validation_result['risk_score_adjustment'] += self.rules['location_mismatch_penalty']
        
        # Rule 4: Night transaction check
        hour = datetime.fromtimestamp(transaction.timestamp).hour
        if hour < 6 or hour > 23:
            validation_result['risk_factors'].append('night_transaction')
            validation_result['risk_score_adjustment'] += self.rules['night_transaction_penalty']
        
        # Rule 5: New merchant check
        merchant_history = [tx.merchant_id for tx in customer_history]
        if transaction.merchant_id not in merchant_history:
            validation_result['risk_factors'].append('new_merchant')
            validation_result['risk_score_adjustment'] += self.rules['new_merchant_penalty']
        
        # Determine final recommendation
        adjusted_risk_score = transaction.risk_score + validation_result['risk_score_adjustment']
        if adjusted_risk_score > 0.7:
            validation_result['is_valid'] = False
            validation_result['recommendation'] = 'block'
        elif adjusted_risk_score > 0.5:
            validation_result['recommendation'] = 'review'
        
        return validation_result

class Blockchain:
    """Permissioned blockchain for fraud detection"""
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.smart_contract = SmartContract()
        self.customer_transactions: Dict[str, List[Transaction]] = {}
        self.transaction_counter = 0
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0",
            nonce=0,
            merkle_root=""
        )
        genesis_block.block_hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to the pending pool"""
        # Update customer history
        if transaction.customer_id not in self.customer_transactions:
            self.customer_transactions[transaction.customer_id] = []
        self.customer_transactions[transaction.customer_id].append(transaction)
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        return True
    
    def mine_pending_transactions(self) -> Optional[Block]:
        """Mine pending transactions into a new block"""
        if not self.pending_transactions:
            return None
        
        # Create new block
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.get_latest_block().block_hash,
            nonce=0,
            merkle_root=self.calculate_merkle_root(self.pending_transactions)
        )
        
        # Mine the block
        block.mine_block(self.difficulty)
        
        # Add to chain
        self.chain.append(block)
        
        # Clear pending transactions
        self.pending_transactions = []
        
        return block
    
    def calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """Calculate Merkle root for transactions"""
        if not transactions:
            return hashlib.sha256("".encode()).hexdigest()
        
        # Convert transactions to strings
        tx_strings = [tx.to_json() for tx in transactions]
        
        # Build Merkle tree
        while len(tx_strings) > 1:
            if len(tx_strings) % 2 == 1:
                tx_strings.append(tx_strings[-1])
            
            new_level = []
            for i in range(0, len(tx_strings), 2):
                combined = tx_strings[i] + tx_strings[i + 1]
                new_level.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_strings = new_level
        
        return tx_strings[0]
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block hash is valid
            if current_block.block_hash != current_block.calculate_hash():
                return False
            
            # Check if previous hash is correct
            if current_block.previous_hash != previous_block.block_hash:
                return False
        
        return True
    
    def get_fraud_statistics(self) -> Dict[str, Any]:
        """Get fraud detection statistics"""
        total_transactions = sum(len(block.transactions) for block in self.chain)
        total_transactions += len(self.pending_transactions)
        
        if total_transactions == 0:
            return {
                'total_transactions': 0,
                'fraudulent_transactions': 0,
                'blocked_transactions': 0,
                'fraud_rate': 0.0,
                'block_rate': 0.0
            }
        
        # Count fraudulent and blocked transactions
        fraudulent_count = 0
        blocked_count = 0
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.fraud_probability > 0.5:
                    fraudulent_count += 1
                if tx.risk_level == RiskLevel.HIGH_RISK:
                    blocked_count += 1
        
        for tx in self.pending_transactions:
            if tx.fraud_probability > 0.5:
                fraudulent_count += 1
            if tx.risk_level == RiskLevel.HIGH_RISK:
                blocked_count += 1
        
        return {
            'total_transactions': total_transactions,
            'fraudulent_transactions': fraudulent_count,
            'blocked_transactions': blocked_count,
            'fraud_rate': fraudulent_count / total_transactions if total_transactions > 0 else 0.0,
            'block_rate': blocked_count / total_transactions if total_transactions > 0 else 0.0
        }

class BlockchainManager:
    """Manages blockchain operations"""
    
    def __init__(self):
        self.blockchain = Blockchain()
        self.transaction_counter = 0
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on risk score"""
        if risk_score < 0.3:
            return RiskLevel.SAFE
        elif risk_score < 0.5:
            return RiskLevel.LOW_RISK
        elif risk_score < 0.7:
            return RiskLevel.MEDIUM_RISK
        else:
            return RiskLevel.HIGH_RISK
    
    def create_transaction(self, customer_id: str, amount: float, merchant_id: str,
                          location: str, payment_method: str, risk_score: float,
                          fraud_probability: float, metadata: Optional[Dict[str, Any]] = None) -> Transaction:
        """Create a new transaction"""
        self.transaction_counter += 1
        transaction_id = f"TX{self.transaction_counter:08d}"
        
        transaction = Transaction(
            transaction_id=transaction_id,
            customer_id=customer_id,
            amount=amount,
            merchant_id=merchant_id,
            timestamp=time.time(),
            location=location,
            payment_method=payment_method,
            risk_score=risk_score,
            risk_level=self._get_risk_level(risk_score),
            fraud_probability=fraud_probability,
            status=TransactionStatus.PENDING,
            metadata=metadata or {}
        )
        
        return transaction
    
    def process_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        """Process a transaction through the blockchain"""
        # Get customer history
        customer_history = self.blockchain.customer_transactions.get(transaction.customer_id, [])
        
        # Validate with smart contract
        validation_result = self.blockchain.smart_contract.validate_transaction(transaction, customer_history)
        
        # Add to blockchain
        self.blockchain.add_transaction(transaction)
        
        return {
            'success': True,
            'transaction_id': transaction.transaction_id,
            'validation_result': validation_result,
            'blockchain_status': 'pending'
        }
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get current blockchain status"""
        return {
            'chain': [block.to_dict() for block in self.blockchain.chain],
            'pending_transactions': [tx.to_dict() for tx in self.blockchain.pending_transactions],
            'difficulty': self.blockchain.difficulty,
            'is_valid': self.blockchain.is_chain_valid(),
            'statistics': self.blockchain.get_fraud_statistics()
        }

# Initialize session state
if 'blockchain_manager' not in st.session_state:
    st.session_state.blockchain_manager = BlockchainManager()

if 'transactions' not in st.session_state:
    st.session_state.transactions = []

if 'auto_generate' not in st.session_state:
    st.session_state.auto_generate = False

# Main dashboard
def main():
    st.markdown('<h1 class="main-header">🔗 Blockchain Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🔗 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard Overview", "Live Transaction Feed", "Block Explorer", "Fraud Analytics", "Smart Contracts", "Transaction Simulator"]
    )
    
    if page == "Dashboard Overview":
        show_dashboard_overview()
    elif page == "Live Transaction Feed":
        show_live_feed()
    elif page == "Block Explorer":
        show_block_explorer()
    elif page == "Fraud Analytics":
        show_fraud_analytics()
    elif page == "Smart Contracts":
        show_smart_contracts()
    elif page == "Transaction Simulator":
        show_transaction_simulator()

def show_dashboard_overview():
    """Show main dashboard overview"""
    st.header("📊 Dashboard Overview")
    
    # Get blockchain status
    status = st.session_state.blockchain_manager.get_blockchain_status()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Blocks",
            value=len(status['chain']),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Transactions",
            value=status['statistics']['total_transactions'],
            delta=None
        )
    
    with col3:
        st.metric(
            label="Fraud Rate",
            value=f"{status['statistics']['fraud_rate']:.2%}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Pending Transactions",
            value=len(status['pending_transactions']),
            delta=None
        )
    
    # Blockchain status
    st.markdown('<div class="blockchain-status">', unsafe_allow_html=True)
    st.subheader("🔗 Blockchain Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Chain Valid:** {'✅ Yes' if status['is_valid'] else '❌ No'}")
        st.write(f"**Mining Difficulty:** {status['difficulty']}")
        st.write(f"**Latest Block:** #{len(status['chain']) - 1}")
    
    with col2:
        st.write(f"**Blocked Transactions:** {status['statistics']['blocked_transactions']}")
        st.write(f"**Fraudulent Transactions:** {status['statistics']['fraudulent_transactions']}")
        st.write(f"**Block Rate:** {status['statistics']['block_rate']:.2%}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent transactions
    st.subheader("📋 Recent Transactions")
    if status['pending_transactions']:
        for tx in status['pending_transactions'][-5:]:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"**{tx['transaction_id']}** - {tx['customer_id']}")
                with col2:
                    st.write(f"${tx['amount']:,.2f}")
                with col3:
                    risk_class = f"risk-{tx['risk_level']}"
                    st.markdown(f'<span class="{risk_class}">{tx["risk_level"].replace("_", " ").title()}</span>', unsafe_allow_html=True)
                with col4:
                    st.write(tx['location'])
    else:
        st.info("No pending transactions")
    
    # Auto-generate transactions
    st.subheader("⚡ Auto-Generate Transactions")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Start Auto-Generation"):
            st.session_state.auto_generate = True
            st.success("Auto-generation started!")
        
        if st.button("Stop Auto-Generation"):
            st.session_state.auto_generate = False
            st.info("Auto-generation stopped!")
    
    with col2:
        if st.session_state.auto_generate:
            st.write("🔄 Generating transactions automatically...")
            # Auto-generate a transaction
            if st.button("Generate Transaction"):
                generate_random_transaction()

def show_live_feed():
    """Show live transaction feed"""
    st.header("📡 Live Transaction Feed")
    
    # Real-time transaction feed
    status = st.session_state.blockchain_manager.get_blockchain_status()
    
    # Create a real-time chart
    if status['pending_transactions']:
        # Prepare data for chart
        df = pd.DataFrame([
            {
                'timestamp': datetime.fromtimestamp(tx['timestamp']),
                'amount': tx['amount'],
                'risk_score': tx['risk_score'],
                'fraud_probability': tx['fraud_probability'],
                'risk_level': tx['risk_level']
            }
            for tx in status['pending_transactions']
        ])
        
        # Transaction amount over time
        fig = px.line(df, x='timestamp', y='amount', 
                     title='Transaction Amounts Over Time',
                     labels={'amount': 'Amount ($)', 'timestamp': 'Time'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk score distribution
        fig2 = px.histogram(df, x='risk_score', nbins=20,
                           title='Risk Score Distribution',
                           labels={'risk_score': 'Risk Score', 'count': 'Count'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Live transaction list
    st.subheader("🔄 Live Transactions")
    if status['pending_transactions']:
        for tx in reversed(status['pending_transactions']):
            with st.container():
                st.markdown('<div class="transaction-card">', unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"**{tx['transaction_id']}**")
                    st.write(f"Customer: {tx['customer_id']}")
                    st.write(f"Merchant: {tx['merchant_id']}")
                
                with col2:
                    st.write(f"**${tx['amount']:,.2f}**")
                
                with col3:
                    risk_class = f"risk-{tx['risk_level']}"
                    st.markdown(f'<span class="{risk_class}">{tx["risk_level"].replace("_", " ").title()}</span>', unsafe_allow_html=True)
                
                with col4:
                    st.write(f"{tx['fraud_probability']:.1%}")
                
                with col5:
                    st.write(tx['location'])
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No pending transactions")

def show_block_explorer():
    """Show block explorer"""
    st.header("🔍 Block Explorer")
    
    status = st.session_state.blockchain_manager.get_blockchain_status()
    
    # Block selection
    if status['chain']:
        block_index = st.selectbox(
            "Select Block",
            range(len(status['chain'])),
            format_func=lambda x: f"Block #{x}"
        )
        
        selected_block = status['chain'][block_index]
        
        # Block details
        st.subheader(f"Block #{selected_block['index']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Hash:** {selected_block['block_hash'][:20]}...")
            st.write(f"**Previous Hash:** {selected_block['previous_hash'][:20]}...")
            st.write(f"**Timestamp:** {datetime.fromtimestamp(selected_block['timestamp'])}")
        
        with col2:
            st.write(f"**Nonce:** {selected_block['nonce']}")
            st.write(f"**Merkle Root:** {selected_block['merkle_root'][:20]}...")
            st.write(f"**Transactions:** {len(selected_block['transactions'])}")
        
        # Block transactions
        if selected_block['transactions']:
            st.subheader("Transactions in Block")
            for tx in selected_block['transactions']:
                with st.expander(f"{tx['transaction_id']} - ${tx['amount']:,.2f}"):
                    st.json(tx)
        else:
            st.info("No transactions in this block")
    
    # Mine block button
    if st.button("⛏️ Mine Pending Transactions"):
        with st.spinner("Mining block..."):
            block = st.session_state.blockchain_manager.blockchain.mine_pending_transactions()
            if block:
                st.success(f"Block #{block.index} mined successfully!")
                st.rerun()
            else:
                st.warning("No pending transactions to mine")

def show_fraud_analytics():
    """Show fraud analytics"""
    st.header("📊 Fraud Analytics")
    
    status = st.session_state.blockchain_manager.get_blockchain_status()
    
    # Analytics metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", status['statistics']['total_transactions'])
    
    with col2:
        st.metric("Fraud Rate", f"{status['statistics']['fraud_rate']:.2%}")
    
    with col3:
        st.metric("Block Rate", f"{status['statistics']['block_rate']:.2%}")
    
    # Fraud analytics charts
    if status['pending_transactions']:
        df = pd.DataFrame([
            {
                'amount': tx['amount'],
                'risk_score': tx['risk_score'],
                'fraud_probability': tx['fraud_probability'],
                'risk_level': tx['risk_level'],
                'location': tx['location']
            }
            for tx in status['pending_transactions']
        ])
        
        # Amount vs Risk Score
        fig = px.scatter(df, x='amount', y='risk_score', 
                        color='risk_level', size='fraud_probability',
                        title='Transaction Amount vs Risk Score',
                        labels={'amount': 'Amount ($)', 'risk_score': 'Risk Score'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level distribution
        risk_counts = df['risk_level'].value_counts()
        fig2 = px.pie(values=risk_counts.values, names=risk_counts.index,
                     title='Risk Level Distribution')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Location-based fraud
        location_fraud = df.groupby('location')['fraud_probability'].mean().sort_values(ascending=False)
        fig3 = px.bar(x=location_fraud.index, y=location_fraud.values,
                     title='Average Fraud Probability by Location',
                     labels={'x': 'Location', 'y': 'Average Fraud Probability'})
        st.plotly_chart(fig3, use_container_width=True)

def show_smart_contracts():
    """Show smart contract management"""
    st.header("⚖️ Smart Contract Management")
    
    blockchain = st.session_state.blockchain_manager.blockchain
    
    # Current rules
    st.subheader("📋 Current Smart Contract Rules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**High Amount Threshold:** ${blockchain.smart_contract.rules['high_amount_threshold']:,.2f}")
        st.write(f"**Velocity Threshold:** {blockchain.smart_contract.rules['velocity_threshold']} transactions/hour")
        st.write(f"**Location Mismatch Penalty:** {blockchain.smart_contract.rules['location_mismatch_penalty']:.1%}")
    
    with col2:
        st.write(f"**Night Transaction Penalty:** {blockchain.smart_contract.rules['night_transaction_penalty']:.1%}")
        st.write(f"**New Merchant Penalty:** {blockchain.smart_contract.rules['new_merchant_penalty']:.1%}")
        st.write(f"**International Penalty:** {blockchain.smart_contract.rules['international_penalty']:.1%}")
    
    # Update rules
    st.subheader("⚙️ Update Smart Contract Rules")
    
    with st.form("update_rules"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_high_amount = st.number_input("High Amount Threshold ($)", 
                                            value=float(blockchain.smart_contract.rules['high_amount_threshold']),
                                            min_value=1000.0, max_value=50000.0, step=1000.0)
            new_velocity = st.number_input("Velocity Threshold (transactions/hour)",
                                         value=int(blockchain.smart_contract.rules['velocity_threshold']),
                                         min_value=1, max_value=20, step=1)
            new_location_penalty = st.slider("Location Mismatch Penalty",
                                           value=float(blockchain.smart_contract.rules['location_mismatch_penalty']),
                                           min_value=0.0, max_value=1.0, step=0.05)
        
        with col2:
            new_night_penalty = st.slider("Night Transaction Penalty",
                                        value=float(blockchain.smart_contract.rules['night_transaction_penalty']),
                                        min_value=0.0, max_value=1.0, step=0.05)
            new_merchant_penalty = st.slider("New Merchant Penalty",
                                           value=float(blockchain.smart_contract.rules['new_merchant_penalty']),
                                           min_value=0.0, max_value=1.0, step=0.05)
            new_international_penalty = st.slider("International Penalty",
                                                value=float(blockchain.smart_contract.rules['international_penalty']),
                                                min_value=0.0, max_value=1.0, step=0.05)
        
        if st.form_submit_button("Update Rules"):
            blockchain.smart_contract.rules.update({
                'high_amount_threshold': new_high_amount,
                'velocity_threshold': new_velocity,
                'location_mismatch_penalty': new_location_penalty,
                'night_transaction_penalty': new_night_penalty,
                'new_merchant_penalty': new_merchant_penalty,
                'international_penalty': new_international_penalty
            })
            st.success("Smart contract rules updated successfully!")

def show_transaction_simulator():
    """Show transaction simulator"""
    st.header("🎮 Transaction Simulator")
    
    # Transaction form
    with st.form("transaction_simulator"):
        st.subheader("Create New Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.text_input("Customer ID", value=f"CUST{random.randint(1000, 9999)}")
            amount = st.number_input("Amount ($)", min_value=1.0, max_value=50000.0, value=100.0, step=10.0)
            merchant_id = st.text_input("Merchant ID", value=f"MERCH{random.randint(100, 999)}")
            location = st.selectbox("Location", [
                "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
                "London, UK", "Paris, FR", "Tokyo, JP", "Sydney, AU"
            ])
        
        with col2:
            payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Digital Wallet"])
            risk_score = st.slider("Risk Score", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            fraud_probability = st.slider("Fraud Probability", min_value=0.0, max_value=1.0, value=0.05, step=0.05)
            metadata = st.text_area("Metadata (JSON)", value='{"device_id": "DEV123", "ip_address": "192.168.1.1"}')
        
        if st.form_submit_button("Create Transaction"):
            try:
                metadata_dict = json.loads(metadata) if metadata else {}
                
                # Create transaction
                transaction = st.session_state.blockchain_manager.create_transaction(
                    customer_id=customer_id,
                    amount=amount,
                    merchant_id=merchant_id,
                    timestamp=time.time(),
                    location=location,
                    payment_method=payment_method,
                    risk_score=risk_score,
                    fraud_probability=fraud_probability,
                    metadata=metadata_dict
                )
                
                # Process transaction
                result = st.session_state.blockchain_manager.process_transaction(transaction)
                
                st.success("Transaction created and processed successfully!")
                
                # Show validation result
                validation = result['validation_result']
                st.subheader("Smart Contract Validation Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Transaction ID:** {result['transaction_id']}")
                    st.write(f"**Valid:** {'✅ Yes' if validation['is_valid'] else '❌ No'}")
                    st.write(f"**Recommendation:** {validation['recommendation'].title()}")
                
                with col2:
                    st.write(f"**Risk Factors:** {', '.join(validation['risk_factors']) if validation['risk_factors'] else 'None'}")
                    st.write(f"**Risk Adjustment:** {validation['risk_score_adjustment']:.1%}")
                    st.write(f"**Blockchain Status:** {result['blockchain_status']}")
                
            except json.JSONDecodeError:
                st.error("Invalid JSON in metadata field")
            except Exception as e:
                st.error(f"Error creating transaction: {str(e)}")

def generate_random_transaction():
    """Generate a random transaction for demonstration"""
    customers = [f"CUST{i:04d}" for i in range(1, 1001)]
    merchants = [f"MERCH{i:03d}" for i in range(1, 501)]
    locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", 
                "London, UK", "Paris, FR", "Tokyo, JP", "Sydney, AU"]
    payment_methods = ["Credit Card", "Debit Card", "Digital Wallet"]
    
    # Create random transaction
    transaction = st.session_state.blockchain_manager.create_transaction(
        customer_id=random.choice(customers),
        amount=random.uniform(10.0, 5000.0),
        merchant_id=random.choice(merchants),
        location=random.choice(locations),
        payment_method=random.choice(payment_methods),
        risk_score=random.uniform(0.1, 0.9),
        fraud_probability=random.uniform(0.01, 0.5),
        metadata={"auto_generated": True, "timestamp": time.time()}
    )
    
    # Process transaction
    result = st.session_state.blockchain_manager.process_transaction(transaction)
    
    # Add to session state for display
    st.session_state.transactions.append({
        'transaction': transaction.to_dict(),
        'result': result,
        'timestamp': time.time()
    })

if __name__ == "__main__":
    main() 