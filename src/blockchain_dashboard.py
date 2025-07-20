#!/usr/bin/env python3
"""
Blockchain Dashboard for Fraud Detection
Integrates blockchain technology with the fraud detection system
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Import blockchain components
from blockchain_core import BlockchainManager, Transaction, TransactionStatus, RiskLevel

class BlockchainDashboard:
    """Blockchain dashboard for fraud detection"""
    
    def __init__(self):
        self.blockchain_manager = BlockchainManager()
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="🔗 Blockchain Fraud Detection Dashboard",
            page_icon="🔗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔗 Blockchain Fraud Detection Dashboard")
        st.markdown("**Smart Contract Validation & Immutable Audit Trail**")
    
    def run(self):
        """Run the blockchain dashboard"""
        # Sidebar
        self.render_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_blockchain_status()
            self.render_transaction_feed()
        
        with col2:
            self.render_smart_contract_monitor()
            self.render_mining_status()
        
        # Bottom sections
        col3, col4 = st.columns(2)
        
        with col3:
            self.render_block_explorer()
        
        with col4:
            self.render_fraud_analytics()
        
        # Transaction simulation
        self.render_transaction_simulator()
    
    def render_sidebar(self):
        """Render sidebar with blockchain controls"""
        with st.sidebar:
            st.header("🔗 Blockchain Controls")
            
            # Blockchain status
            status = self.blockchain_manager.get_blockchain_status()
            st.success(f"✅ Chain Valid: {status['is_valid']}")
            st.info(f"📦 Blocks: {len(status['chain'])}")
            st.info(f"⏳ Pending TX: {len(status['pending_transactions'])}")
            st.info(f"🔧 Difficulty: {status['difficulty']}")
            
            st.markdown("---")
            
            # Smart contract rules
            st.header("📋 Smart Contract Rules")
            rules = self.blockchain_manager.blockchain.smart_contract.rules
            
            for rule, value in rules.items():
                if isinstance(value, float):
                    st.metric(rule.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    st.metric(rule.replace('_', ' ').title(), value)
            
            st.markdown("---")
            
            # Actions
            st.header("⚡ Actions")
            
            if st.button("🔄 Refresh Blockchain"):
                st.rerun()
            
            if st.button("⛏️ Force Mine Block"):
                self.blockchain_manager.blockchain.mine_pending_transactions()
                st.success("Block mined!")
                st.rerun()
            
            if st.button("🧹 Clear Pending Transactions"):
                self.blockchain_manager.blockchain.pending_transactions.clear()
                st.success("Pending transactions cleared!")
                st.rerun()
    
    def render_blockchain_status(self):
        """Render blockchain status overview"""
        st.header("📊 Blockchain Status")
        
        status = self.blockchain_manager.get_blockchain_status()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Blocks", len(status['chain']))
        
        with col2:
            st.metric("Pending Transactions", len(status['pending_transactions']))
        
        with col3:
            total_tx = sum(len(block['transactions']) for block in status['chain'])
            st.metric("Total Transactions", total_tx)
        
        with col4:
            chain_valid = "✅ Valid" if status['is_valid'] else "❌ Invalid"
            st.metric("Chain Status", chain_valid)
        
        # Blockchain visualization
        st.subheader("🔗 Blockchain Structure")
        
        if len(status['chain']) > 1:
            # Create blockchain visualization
            fig = self.create_blockchain_visualization(status['chain'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Genesis block only. Add transactions to see blockchain structure.")
    
    def create_blockchain_visualization(self, chain: List[Dict]) -> go.Figure:
        """Create blockchain visualization"""
        # Prepare data
        block_indices = [block['index'] for block in chain]
        block_hashes = [block['block_hash'][:8] + "..." for block in chain]
        transaction_counts = [len(block['transactions']) for block in chain]
        
        # Create figure
        fig = go.Figure()
        
        # Add blocks as nodes
        fig.add_trace(go.Scatter(
            x=block_indices,
            y=[0] * len(block_indices),
            mode='markers+text',
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            text=block_hashes,
            textposition="top center",
            name="Blocks",
            hovertemplate="<b>Block %{x}</b><br>Hash: %{text}<br>Transactions: %{customdata}<extra></extra>",
            customdata=transaction_counts
        ))
        
        # Add connections between blocks
        for i in range(len(block_indices) - 1):
            fig.add_trace(go.Scatter(
                x=[block_indices[i], block_indices[i + 1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title="Blockchain Structure",
            xaxis_title="Block Index",
            yaxis=dict(showticklabels=False, showgrid=False),
            height=300,
            showlegend=False
        )
        
        return fig
    
    def render_transaction_feed(self):
        """Render real-time transaction feed"""
        st.header("📡 Live Transaction Feed")
        
        # Get recent transactions
        status = self.blockchain_manager.get_blockchain_status()
        all_transactions = []
        
        for block in status['chain']:
            for tx in block['transactions']:
                all_transactions.append(tx)
        
        # Add pending transactions
        all_transactions.extend(status['pending_transactions'])
        
        if all_transactions:
            # Sort by timestamp (newest first)
            all_transactions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Display recent transactions
            for tx in all_transactions[:10]:  # Show last 10
                self.render_transaction_card(tx)
        else:
            st.info("No transactions yet. Use the transaction simulator below to add transactions.")
    
    def render_transaction_card(self, transaction: Dict):
        """Render individual transaction card"""
        # Determine color based on risk level
        risk_colors = {
            'safe': '🟢',
            'low_risk': '🟡',
            'medium_risk': '🟠',
            'high_risk': '🔴'
        }
        
        status_colors = {
            'pending': '⏳',
            'validated': '✅',
            'rejected': '❌',
            'fraudulent': '🚨'
        }
        
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{transaction['transaction_id']}**")
                st.markdown(f"Customer: {transaction['customer_id']} | Amount: ${transaction['amount']:,.2f}")
                st.markdown(f"Merchant: {transaction['merchant_id']} | Location: {transaction['location']}")
            
            with col2:
                risk_icon = risk_colors.get(transaction['risk_level'], '⚪')
                status_icon = status_colors.get(transaction['status'], '❓')
                st.markdown(f"{risk_icon} {transaction['risk_level'].replace('_', ' ').title()}")
                st.markdown(f"{status_icon} {transaction['status'].title()}")
            
            with col3:
                st.markdown(f"Risk: {transaction['risk_score']:.2%}")
                st.markdown(f"Fraud: {transaction['fraud_probability']:.2%}")
            
            st.markdown("---")
    
    def render_smart_contract_monitor(self):
        """Render smart contract monitoring"""
        st.header("📋 Smart Contract Monitor")
        
        # Contract rules
        rules = self.blockchain_manager.blockchain.smart_contract.rules
        
        st.subheader("Active Rules")
        for rule, value in rules.items():
            rule_name = rule.replace('_', ' ').title()
            if isinstance(value, float):
                st.metric(rule_name, f"{value:.2f}")
            else:
                st.metric(rule_name, value)
        
        # Recent validations
        st.subheader("Recent Validations")
        
        # Get recent transactions for validation history
        status = self.blockchain_manager.get_blockchain_status()
        recent_tx = []
        
        for block in status['chain'][-3:]:  # Last 3 blocks
            for tx in block['transactions']:
                recent_tx.append(tx)
        
        if recent_tx:
            for tx in recent_tx[-5:]:  # Last 5 transactions
                validation_result = self.blockchain_manager.blockchain.smart_contract.validate_transaction(
                    Transaction(**tx), []
                )
                
                with st.expander(f"TX {tx['transaction_id']} - {validation_result['recommendation'].title()}"):
                    st.markdown(f"**Risk Factors:** {', '.join(validation_result['risk_factors']) if validation_result['risk_factors'] else 'None'}")
                    st.markdown(f"**Risk Adjustment:** {validation_result['risk_score_adjustment']:.2%}")
                    st.markdown(f"**Final Recommendation:** {validation_result['recommendation'].title()}")
        else:
            st.info("No recent validations to show.")
    
    def render_mining_status(self):
        """Render mining status"""
        st.header("⛏️ Mining Status")
        
        blockchain = self.blockchain_manager.blockchain
        
        # Mining status
        mining_status = "🟢 Active" if blockchain.is_mining else "🔴 Stopped"
        st.metric("Mining Status", mining_status)
        
        # Mining metrics
        st.metric("Block Time", f"{blockchain.block_time}s")
        st.metric("Difficulty", blockchain.difficulty)
        st.metric("Queue Size", blockchain.mining_queue.qsize())
        
        # Mining controls
        st.subheader("Mining Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⏸️ Pause Mining"):
                blockchain.stop_mining()
                st.success("Mining paused!")
        
        with col2:
            if st.button("▶️ Resume Mining"):
                blockchain.start_mining()
                st.success("Mining resumed!")
    
    def render_block_explorer(self):
        """Render block explorer"""
        st.header("🔍 Block Explorer")
        
        status = self.blockchain_manager.get_blockchain_status()
        
        if len(status['chain']) > 1:
            # Block selector
            block_index = st.selectbox(
                "Select Block",
                range(len(status['chain'])),
                format_func=lambda x: f"Block {x} ({len(status['chain'][x]['transactions'])} transactions)"
            )
            
            selected_block = status['chain'][block_index]
            
            # Block details
            st.subheader(f"Block {selected_block['index']} Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Hash:** `{selected_block['block_hash']}`")
                st.markdown(f"**Previous Hash:** `{selected_block['previous_hash']}`")
                st.markdown(f"**Timestamp:** {datetime.fromtimestamp(selected_block['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.markdown(f"**Nonce:** {selected_block['nonce']}")
                st.markdown(f"**Merkle Root:** `{selected_block['merkle_root']}`")
                st.markdown(f"**Transactions:** {len(selected_block['transactions'])}")
            
            # Block transactions
            if selected_block['transactions']:
                st.subheader("Block Transactions")
                
                for tx in selected_block['transactions']:
                    with st.expander(f"TX {tx['transaction_id']} - ${tx['amount']:,.2f}"):
                        st.json(tx)
            else:
                st.info("This block contains no transactions (Genesis block).")
        else:
            st.info("Only genesis block exists. Add transactions to explore blocks.")
    
    def render_fraud_analytics(self):
        """Render fraud analytics with blockchain data"""
        st.header("📈 Fraud Analytics")
        
        analytics = self.blockchain_manager.get_fraud_analytics()
        
        # Key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Transactions", analytics['total_transactions'])
            st.metric("Fraud Rate", f"{analytics['fraud_rate']:.2%}")
        
        with col2:
            st.metric("Blocked Transactions", analytics['blocked_transactions'])
            st.metric("Block Rate", f"{analytics['block_rate']:.2%}")
        
        # Risk distribution chart
        if analytics['risk_distribution']:
            st.subheader("Risk Level Distribution")
            
            risk_data = analytics['risk_distribution']
            fig = px.pie(
                values=list(risk_data.values()),
                names=list(risk_data.keys()),
                title="Transaction Risk Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fraud trends over time
        st.subheader("Fraud Trends")
        
        # Simulate fraud trends (in real implementation, this would come from blockchain data)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        fraud_rates = [random.uniform(0.01, 0.05) for _ in range(len(dates))]
        
        trend_df = pd.DataFrame({
            'Date': dates,
            'Fraud Rate': fraud_rates
        })
        
        fig = px.line(trend_df, x='Date', y='Fraud Rate', title="Fraud Rate Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_transaction_simulator(self):
        """Render transaction simulator"""
        st.header("🎮 Transaction Simulator")
        
        with st.form("transaction_simulator"):
            col1, col2 = st.columns(2)
            
            with col1:
                customer_id = st.text_input("Customer ID", value=f"CUST{random.randint(1000, 9999)}")
                amount = st.number_input("Amount ($)", min_value=1.0, max_value=50000.0, value=random.uniform(10, 1000))
                merchant_id = st.text_input("Merchant ID", value=f"MERCH{random.randint(100, 999)}")
                location = st.selectbox("Location", ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Miami, FL", "London, UK", "Tokyo, JP"])
            
            with col2:
                payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Digital Wallet", "Bank Transfer"])
                risk_score = st.slider("Risk Score", 0.0, 1.0, value=random.uniform(0.1, 0.8), step=0.01)
                fraud_probability = st.slider("Fraud Probability", 0.0, 1.0, value=random.uniform(0.01, 0.3), step=0.01)
                metadata = st.text_area("Metadata (JSON)", value='{"device_id": "mobile", "ip_address": "192.168.1.1"}')
            
            submitted = st.form_submit_button("🚀 Submit Transaction")
            
            if submitted:
                try:
                    # Parse metadata
                    metadata_dict = json.loads(metadata) if metadata else {}
                    
                    # Create transaction
                    transaction = self.blockchain_manager.create_transaction(
                        customer_id=customer_id,
                        amount=amount,
                        merchant_id=merchant_id,
                        location=location,
                        payment_method=payment_method,
                        risk_score=risk_score,
                        fraud_probability=fraud_probability,
                        metadata=metadata_dict
                    )
                    
                    # Process transaction
                    result = self.blockchain_manager.process_transaction(transaction)
                    
                    if result['success']:
                        st.success(f"✅ Transaction {transaction.transaction_id} submitted successfully!")
                        st.json(result)
                        
                        # Auto-refresh after a short delay
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Transaction failed: {result['error']}")
                
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON in metadata field")
                except Exception as e:
                    st.error(f"❌ Error creating transaction: {str(e)}")

def main():
    """Main function to run the blockchain dashboard"""
    dashboard = BlockchainDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 