#!/usr/bin/env python3
"""
Enhanced Blockchain Fraud Detection Dashboard
Advanced features with real-time analytics and smart contracts
"""

import streamlit as st
import time
import random
import json
import hashlib
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Enhanced Blockchain Fraud Detection",
    page_icon="🔗",
    layout="wide"
)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []

if 'blocks' not in st.session_state:
    st.session_state.blocks = []

if 'smart_contracts' not in st.session_state:
    st.session_state.smart_contracts = {
        'high_amount_threshold': 10000.0,
        'velocity_threshold': 5,
        'location_mismatch_penalty': 0.3,
        'night_transaction_penalty': 0.2,
        'new_merchant_penalty': 0.15
    }

if 'fraud_alerts' not in st.session_state:
    st.session_state.fraud_alerts = []

# Smart Contract Functions
def validate_transaction_smart_contract(tx, customer_history):
    """Validate transaction using smart contract rules"""
    risk_factors = []
    risk_adjustment = 0.0
    
    # Rule 1: High amount check
    if tx['amount'] > st.session_state.smart_contracts['high_amount_threshold']:
        risk_factors.append('High Amount')
        risk_adjustment += 0.3
    
    # Rule 2: Velocity check
    recent_tx = [t for t in customer_history if t['customer'] == tx['customer']]
    if len(recent_tx) > st.session_state.smart_contracts['velocity_threshold']:
        risk_factors.append('High Velocity')
        risk_adjustment += 0.4
    
    # Rule 3: Location mismatch
    if customer_history:
        last_location = customer_history[-1]['location']
        if last_location != tx['location']:
            risk_factors.append('Location Mismatch')
            risk_adjustment += st.session_state.smart_contracts['location_mismatch_penalty']
    
    # Rule 4: Night transaction
    hour = datetime.now().hour
    if hour < 6 or hour > 23:
        risk_factors.append('Night Transaction')
        risk_adjustment += st.session_state.smart_contracts['night_transaction_penalty']
    
    # Rule 5: New merchant
    merchant_history = [t['merchant'] for t in customer_history if t['customer'] == tx['customer']]
    if tx['merchant'] not in merchant_history:
        risk_factors.append('New Merchant')
        risk_adjustment += st.session_state.smart_contracts['new_merchant_penalty']
    
    return risk_factors, risk_adjustment

def main():
    st.title("🔗 Enhanced Blockchain Fraud Detection Dashboard")
    st.markdown("**Status**: ✅ Successfully Deployed on Streamlit Cloud")
    
    # Sidebar
    st.sidebar.title("🔧 Advanced Controls")
    
    # Smart Contract Management
    st.sidebar.subheader("📋 Smart Contract Rules")
    st.session_state.smart_contracts['high_amount_threshold'] = st.sidebar.number_input(
        "High Amount Threshold ($)", 
        value=float(st.session_state.smart_contracts['high_amount_threshold']),
        step=100.0
    )
    st.session_state.smart_contracts['velocity_threshold'] = st.sidebar.number_input(
        "Velocity Threshold (tx/hour)", 
        value=int(st.session_state.smart_contracts['velocity_threshold']),
        step=1
    )
    
    # Transaction Generation
    if st.sidebar.button("🚀 Generate Transaction"):
        # Create transaction with enhanced data
        tx = {
            'id': f"TX{random.randint(10000, 99999)}",
            'customer': f"CUST{random.randint(1000, 9999)}",
            'amount': round(random.uniform(10, 15000), 2),
            'merchant': random.choice(['amazon', 'netflix', 'uber', 'starbucks', 'walmart', 'target', 'ebay', 'paypal']),
            'location': random.choice(['New York', 'London', 'Tokyo', 'Singapore', 'Sydney', 'Paris', 'Berlin', 'Mumbai']),
            'risk_score': round(random.uniform(0, 1), 2),
            'timestamp': datetime.now(),
            'payment_method': random.choice(['credit_card', 'debit_card', 'mobile_payment', 'bank_transfer', 'crypto']),
            'device_id': f"DEV{random.randint(100, 999)}",
            'ip_address': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        }
        
        # Apply smart contract validation
        all_transactions = st.session_state.transactions + [t for block in st.session_state.blocks for t in block['transactions']]
        risk_factors, risk_adjustment = validate_transaction_smart_contract(tx, all_transactions)
        
        # Adjust risk score
        tx['risk_score'] = min(1.0, tx['risk_score'] + risk_adjustment)
        tx['risk_factors'] = risk_factors
        
        # Determine risk level
        if tx['risk_score'] < 0.3:
            tx['risk_level'] = '🟢 SAFE'
        elif tx['risk_score'] < 0.5:
            tx['risk_level'] = '🟡 LOW_RISK'
        elif tx['risk_score'] < 0.7:
            tx['risk_level'] = '🟠 MEDIUM_RISK'
        else:
            tx['risk_level'] = '🔴 HIGH_RISK'
            # Add to fraud alerts
            st.session_state.fraud_alerts.append({
                'transaction_id': tx['id'],
                'customer': tx['customer'],
                'amount': tx['amount'],
                'risk_factors': risk_factors,
                'timestamp': datetime.now()
            })
        
        st.session_state.transactions.append(tx)
        st.sidebar.success(f"Transaction {tx['id']} added!")
    
    if st.sidebar.button("⛏️ Mine Block"):
        if st.session_state.transactions:
            # Create block with proof of work
            block = {
                'index': len(st.session_state.blocks),
                'timestamp': datetime.now(),
                'transactions': st.session_state.transactions.copy(),
                'previous_hash': st.session_state.blocks[-1]['hash'] if st.session_state.blocks else '0',
                'nonce': 0,
                'difficulty': 4
            }
            
            # Simple proof of work
            target = "0" * block['difficulty']
            block_string = json.dumps(block, default=str, sort_keys=True)
            while block['hash'][:block['difficulty']] != target:
                block['nonce'] += 1
                block['hash'] = hashlib.sha256(block_string.encode()).hexdigest()
                block_string = json.dumps(block, default=str, sort_keys=True)
            
            st.session_state.blocks.append(block)
            st.session_state.transactions = []
            st.sidebar.success(f"Block {block['index']} mined with hash: {block['hash'][:8]}...")
        else:
            st.sidebar.warning("No pending transactions to mine")
    
    if st.sidebar.button("🔄 Reset Blockchain"):
        st.session_state.transactions = []
        st.session_state.blocks = []
        st.session_state.fraud_alerts = []
        st.sidebar.success("Blockchain reset!")
    
    # Main Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Blocks", len(st.session_state.blocks))
    
    with col2:
        st.metric("Pending TX", len(st.session_state.transactions))
    
    with col3:
        total_tx = sum(len(block['transactions']) for block in st.session_state.blocks)
        st.metric("Total TX", total_tx)
    
    with col4:
        st.metric("Fraud Alerts", len(st.session_state.fraud_alerts))
    
    # Fraud Alerts
    if st.session_state.fraud_alerts:
        st.subheader("🚨 Recent Fraud Alerts")
        for alert in st.session_state.fraud_alerts[-5:]:
            with st.container():
                st.error(f"**{alert['transaction_id']}** - Customer: {alert['customer']} - Amount: ${alert['amount']}")
                st.write(f"Risk Factors: {', '.join(alert['risk_factors'])}")
                st.write(f"Time: {alert['timestamp'].strftime('%H:%M:%S')}")
                st.divider()
    
    # Real-time Analytics
    st.subheader("📊 Real-time Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Distribution
        if st.session_state.blocks:
            risk_counts = {'🟢 SAFE': 0, '🟡 LOW_RISK': 0, '🟠 MEDIUM_RISK': 0, '🔴 HIGH_RISK': 0}
            
            for block in st.session_state.blocks:
                for tx in block['transactions']:
                    risk_counts[tx['risk_level']] += 1
            
            st.write("**Risk Distribution:**")
            for risk, count in risk_counts.items():
                st.write(f"{risk}: {count}")
    
    with col2:
        # Transaction Volume
        if st.session_state.blocks:
            total_volume = sum(tx['amount'] for block in st.session_state.blocks for tx in block['transactions'])
            avg_amount = total_volume / total_tx if total_tx > 0 else 0
            
            st.write("**Transaction Statistics:**")
            st.write(f"Total Volume: ${total_volume:,.2f}")
            st.write(f"Average Amount: ${avg_amount:.2f}")
            st.write(f"Total Transactions: {total_tx}")
    
    # Recent Transactions with Enhanced Details
    st.subheader("💳 Recent Transactions")
    if st.session_state.transactions:
        for tx in st.session_state.transactions[-5:]:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{tx['id']}** - {tx['customer']}")
                    st.write(f"${tx['amount']} - {tx['merchant']}")
                with col2:
                    st.write(f"{tx['location']} - {tx['payment_method']}")
                    st.write(f"Device: {tx['device_id']} - IP: {tx['ip_address']}")
                with col3:
                    st.write(tx['risk_level'])
                    if tx['risk_factors']:
                        st.write(f"Factors: {', '.join(tx['risk_factors'])}")
                st.divider()
    else:
        st.info("No pending transactions. Click 'Generate Transaction' to start!")
    
    # Enhanced Block Explorer
    st.subheader("🔍 Enhanced Block Explorer")
    if st.session_state.blocks:
        for block in st.session_state.blocks:
            with st.expander(f"Block {block['index']} - {block['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write(f"**Hash:** {block['hash']}")
                st.write(f"**Previous Hash:** {block['previous_hash']}")
                st.write(f"**Nonce:** {block['nonce']}")
                st.write(f"**Difficulty:** {block['difficulty']}")
                st.write(f"**Transactions:** {len(block['transactions'])}")
                
                if block['transactions']:
                    st.write("**Transaction Details:**")
                    for tx in block['transactions']:
                        st.write(f"- {tx['id']}: ${tx['amount']} ({tx['risk_level']}) - {tx['customer']}")
    else:
        st.info("No blocks mined yet. Generate transactions and mine blocks!")
    
    # Smart Contract Status
    st.subheader("📋 Smart Contract Status")
    st.write("**Active Rules:**")
    for rule, value in st.session_state.smart_contracts.items():
        if isinstance(value, float):
            st.write(f"- {rule.replace('_', ' ').title()}: ${value:,.2f}")
        else:
            st.write(f"- {rule.replace('_', ' ').title()}: {value}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Enhanced Blockchain Fraud Detection** - Advanced features with smart contracts and real-time analytics")
    st.markdown("Successfully deployed on Streamlit Cloud! 🚀")

if __name__ == "__main__":
    main() 