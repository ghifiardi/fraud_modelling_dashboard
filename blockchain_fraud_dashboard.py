#!/usr/bin/env python3
"""
Blockchain Fraud Detection Dashboard
Main application integrating blockchain technology with fraud detection
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import random
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import blockchain components
from src.blockchain_core import BlockchainManager, Transaction, TransactionStatus, RiskLevel

class BlockchainFraudDashboard:
    """Main blockchain fraud detection dashboard"""
    
    def __init__(self):
        self.blockchain_manager = BlockchainManager()
        self.setup_page()
        self.api_base_url = "http://localhost:5001/api/blockchain"
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="🔗 Blockchain Fraud Detection",
            page_icon="🔗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔗 Blockchain Fraud Detection Dashboard")
        st.markdown("**Smart Contract Validation & Immutable Audit Trail for Fraud Detection**")
        
        # Add blockchain info
        st.markdown("""
        ### 🚀 Key Features:
        - **Smart Contract Validation**: Automated fraud detection rules
        - **Immutable Audit Trail**: All transactions recorded on blockchain
        - **Real-time Monitoring**: Live transaction feed with risk assessment
        - **Block Explorer**: Detailed view of blockchain structure
        - **Fraud Analytics**: Advanced analytics with blockchain data
        """)
    
    def run(self):
        """Run the main dashboard"""
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["🏠 Dashboard", "📡 Live Feed", "🔍 Block Explorer", "📊 Analytics", "⚙️ Smart Contracts", "🎮 Simulator"]
        )
        
        if page == "🏠 Dashboard":
            self.render_main_dashboard()
        elif page == "📡 Live Feed":
            self.render_live_feed()
        elif page == "🔍 Block Explorer":
            self.render_block_explorer()
        elif page == "📊 Analytics":
            self.render_analytics()
        elif page == "⚙️ Smart Contracts":
            self.render_smart_contracts()
        elif page == "🎮 Simulator":
            self.render_simulator()
    
    def render_main_dashboard(self):
        """Render main dashboard overview"""
        st.header("🏠 Dashboard Overview")
        
        # Blockchain status
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
            fig = self.create_blockchain_visualization(status['chain'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Genesis block only. Add transactions to see blockchain structure.")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_recent_transactions()
        
        with col2:
            self.render_mining_status()
    
    def render_live_feed(self):
        """Render live transaction feed"""
        st.header("📡 Live Transaction Feed")
        
        # Auto-refresh
        if st.button("🔄 Refresh"):
            st.rerun()
        
        # Get recent transactions
        status = self.blockchain_manager.get_blockchain_status()
        all_transactions = []
        
        for block in status['chain']:
            for tx in block['transactions']:
                all_transactions.append(tx)
        
        all_transactions.extend(status['pending_transactions'])
        
        if all_transactions:
            all_transactions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_filter = st.selectbox("Risk Level", ["All", "Safe", "Low Risk", "Medium Risk", "High Risk"])
            
            with col2:
                status_filter = st.selectbox("Status", ["All", "Pending", "Validated", "Rejected", "Fraudulent"])
            
            with col3:
                search = st.text_input("Search Transaction ID")
            
            # Filter transactions
            filtered_transactions = all_transactions
            
            if risk_filter != "All":
                risk_map = {"Safe": "safe", "Low Risk": "low_risk", "Medium Risk": "medium_risk", "High Risk": "high_risk"}
                filtered_transactions = [tx for tx in filtered_transactions if tx['risk_level'] == risk_map[risk_filter]]
            
            if status_filter != "All":
                status_map = {"Pending": "pending", "Validated": "validated", "Rejected": "rejected", "Fraudulent": "fraudulent"}
                filtered_transactions = [tx for tx in filtered_transactions if tx['status'] == status_map[status_filter]]
            
            if search:
                filtered_transactions = [tx for tx in filtered_transactions if search.lower() in tx['transaction_id'].lower()]
            
            # Display transactions
            for tx in filtered_transactions[:20]:  # Show last 20
                self.render_transaction_card(tx)
        else:
            st.info("No transactions yet. Use the simulator to add transactions.")
    
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
    
    def render_analytics(self):
        """Render fraud analytics"""
        st.header("📊 Fraud Analytics")
        
        analytics = self.blockchain_manager.get_fraud_analytics()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", analytics['total_transactions'])
        
        with col2:
            st.metric("Fraud Rate", f"{analytics['fraud_rate']:.2%}")
        
        with col3:
            st.metric("Blocked Transactions", analytics['blocked_transactions'])
        
        with col4:
            st.metric("Block Rate", f"{analytics['block_rate']:.2%}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if analytics['risk_distribution']:
                st.subheader("Risk Level Distribution")
                
                risk_data = analytics['risk_distribution']
                fig = px.pie(
                    values=list(risk_data.values()),
                    names=list(risk_data.keys()),
                    title="Transaction Risk Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Fraud Trends")
            
            # Simulate fraud trends
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            fraud_rates = [random.uniform(0.01, 0.05) for _ in range(len(dates))]
            
            trend_df = pd.DataFrame({
                'Date': dates,
                'Fraud Rate': fraud_rates
            })
            
            fig = px.line(trend_df, x='Date', y='Fraud Rate', title="Fraud Rate Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction volume over time
        st.subheader("Transaction Volume")
        
        # Simulate transaction volume
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        volumes = [random.randint(100, 1000) for _ in range(len(dates))]
        
        volume_df = pd.DataFrame({
            'Date': dates,
            'Volume': volumes
        })
        
        fig = px.bar(volume_df, x='Date', y='Volume', title="Daily Transaction Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_smart_contracts(self):
        """Render smart contract management"""
        st.header("⚙️ Smart Contract Management")
        
        rules = self.blockchain_manager.blockchain.smart_contract.rules
        
        st.subheader("Active Rules")
        
        # Display current rules
        col1, col2 = st.columns(2)
        
        with col1:
            for i, (rule, value) in enumerate(rules.items()):
                if i % 2 == 0:
                    if isinstance(value, float):
                        st.metric(rule.replace('_', ' ').title(), f"{value:.2f}")
                    else:
                        st.metric(rule.replace('_', ' ').title(), value)
        
        with col2:
            for i, (rule, value) in enumerate(rules.items()):
                if i % 2 == 1:
                    if isinstance(value, float):
                        st.metric(rule.replace('_', ' ').title(), f"{value:.2f}")
                    else:
                        st.metric(rule.replace('_', ' ').title(), value)
        
        # Rule editor
        st.subheader("Edit Rules")
        
        with st.form("rule_editor"):
            col1, col2 = st.columns(2)
            
            with col1:
                high_amount = st.number_input("High Amount Threshold ($)", value=rules['high_amount_threshold'], step=100.0)
                velocity = st.number_input("Velocity Threshold (tx/hour)", value=rules['velocity_threshold'], step=1)
                location_penalty = st.slider("Location Mismatch Penalty", 0.0, 1.0, value=rules['location_mismatch_penalty'], step=0.05)
            
            with col2:
                night_penalty = st.slider("Night Transaction Penalty", 0.0, 1.0, value=rules['night_transaction_penalty'], step=0.05)
                merchant_penalty = st.slider("New Merchant Penalty", 0.0, 1.0, value=rules['new_merchant_penalty'], step=0.05)
                international_penalty = st.slider("International Penalty", 0.0, 1.0, value=rules['international_penalty'], step=0.05)
            
            if st.form_submit_button("💾 Update Rules"):
                # Update rules
                self.blockchain_manager.blockchain.smart_contract.rules.update({
                    'high_amount_threshold': high_amount,
                    'velocity_threshold': int(velocity),
                    'location_mismatch_penalty': location_penalty,
                    'night_transaction_penalty': night_penalty,
                    'new_merchant_penalty': merchant_penalty,
                    'international_penalty': international_penalty
                })
                st.success("✅ Rules updated successfully!")
                st.rerun()
        
        # Recent validations
        st.subheader("Recent Validations")
        
        status = self.blockchain_manager.get_blockchain_status()
        recent_tx = []
        
        for block in status['chain'][-3:]:
            for tx in block['transactions']:
                recent_tx.append(tx)
        
        if recent_tx:
            for tx in recent_tx[-5:]:
                validation_result = self.blockchain_manager.blockchain.smart_contract.validate_transaction(
                    Transaction(**tx), []
                )
                
                with st.expander(f"TX {tx['transaction_id']} - {validation_result['recommendation'].title()}"):
                    st.markdown(f"**Risk Factors:** {', '.join(validation_result['risk_factors']) if validation_result['risk_factors'] else 'None'}")
                    st.markdown(f"**Risk Adjustment:** {validation_result['risk_score_adjustment']:.2%}")
                    st.markdown(f"**Final Recommendation:** {validation_result['recommendation'].title()}")
        else:
            st.info("No recent validations to show.")
    
    def render_simulator(self):
        """Render transaction simulator"""
        st.header("🎮 Transaction Simulator")
        
        # Batch simulation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Single Transaction")
            
            with st.form("single_transaction"):
                customer_id = st.text_input("Customer ID", value=f"CUST{random.randint(1000, 9999)}")
                amount = st.number_input("Amount ($)", min_value=1.0, max_value=50000.0, value=random.uniform(10, 1000))
                merchant_id = st.text_input("Merchant ID", value=f"MERCH{random.randint(100, 999)}")
                location = st.selectbox("Location", ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Miami, FL", "London, UK", "Tokyo, JP"])
                payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Digital Wallet", "Bank Transfer"])
                risk_score = st.slider("Risk Score", 0.0, 1.0, value=random.uniform(0.1, 0.8), step=0.01)
                fraud_probability = st.slider("Fraud Probability", 0.0, 1.0, value=random.uniform(0.01, 0.3), step=0.01)
                
                if st.form_submit_button("🚀 Submit Transaction"):
                    try:
                        transaction = self.blockchain_manager.create_transaction(
                            customer_id=customer_id,
                            amount=amount,
                            merchant_id=merchant_id,
                            location=location,
                            payment_method=payment_method,
                            risk_score=risk_score,
                            fraud_probability=fraud_probability,
                            metadata={"simulator": True}
                        )
                        
                        result = self.blockchain_manager.process_transaction(transaction)
                        
                        if result['success']:
                            st.success(f"✅ Transaction {transaction.transaction_id} submitted!")
                            st.json(result)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Transaction failed: {result['error']}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            st.subheader("Batch Simulation")
            
            with st.form("batch_simulation"):
                num_transactions = st.number_input("Number of Transactions", min_value=1, max_value=100, value=10)
                fraud_rate = st.slider("Fraud Rate", 0.0, 1.0, value=0.1, step=0.01)
                
                if st.form_submit_button("🎲 Generate Batch"):
                    try:
                        created_count = 0
                        
                        for i in range(num_transactions):
                            # Randomize transaction parameters
                            is_fraudulent = random.random() < fraud_rate
                            
                            transaction = self.blockchain_manager.create_transaction(
                                customer_id=f"CUST{random.randint(1000, 9999)}",
                                amount=random.uniform(10, 5000),
                                merchant_id=f"MERCH{random.randint(100, 999)}",
                                location=random.choice(["New York, NY", "Los Angeles, CA", "Chicago, IL", "Miami, FL", "London, UK", "Tokyo, JP"]),
                                payment_method=random.choice(["Credit Card", "Debit Card", "Digital Wallet", "Bank Transfer"]),
                                risk_score=random.uniform(0.1, 0.9) if is_fraudulent else random.uniform(0.1, 0.5),
                                fraud_probability=random.uniform(0.3, 0.8) if is_fraudulent else random.uniform(0.01, 0.2),
                                metadata={"batch_simulator": True, "batch_id": i}
                            )
                            
                            result = self.blockchain_manager.process_transaction(transaction)
                            if result['success']:
                                created_count += 1
                        
                        st.success(f"✅ Created {created_count} transactions!")
                        time.sleep(1)
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    def render_recent_transactions(self):
        """Render recent transactions"""
        st.subheader("Recent Transactions")
        
        status = self.blockchain_manager.get_blockchain_status()
        recent_tx = []
        
        for block in status['chain'][-2:]:
            for tx in block['transactions']:
                recent_tx.append(tx)
        
        recent_tx.extend(status['pending_transactions'])
        recent_tx.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for tx in recent_tx[:5]:
            self.render_transaction_card(tx, compact=True)
    
    def render_mining_status(self):
        """Render mining status"""
        st.subheader("⛏️ Mining Status")
        
        blockchain = self.blockchain_manager.blockchain
        
        mining_status = "🟢 Active" if blockchain.is_mining else "🔴 Stopped"
        st.metric("Mining Status", mining_status)
        
        st.metric("Block Time", f"{blockchain.block_time}s")
        st.metric("Difficulty", blockchain.difficulty)
        st.metric("Queue Size", blockchain.mining_queue.qsize())
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⏸️ Pause Mining"):
                blockchain.stop_mining()
                st.success("Mining paused!")
        
        with col2:
            if st.button("▶️ Resume Mining"):
                blockchain.start_mining()
                st.success("Mining resumed!")
    
    def render_transaction_card(self, transaction: Dict, compact: bool = False):
        """Render transaction card"""
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
        
        if compact:
            st.markdown(f"**{transaction['transaction_id']}** - ${transaction['amount']:,.2f}")
            st.markdown(f"{risk_colors.get(transaction['risk_level'], '⚪')} {transaction['risk_level'].replace('_', ' ').title()}")
        else:
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
    
    def create_blockchain_visualization(self, chain: List[Dict]) -> go.Figure:
        """Create blockchain visualization"""
        block_indices = [block['index'] for block in chain]
        block_hashes = [block['block_hash'][:8] + "..." for block in chain]
        transaction_counts = [len(block['transactions']) for block in chain]
        
        fig = go.Figure()
        
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

def main():
    """Main function"""
    dashboard = BlockchainFraudDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 