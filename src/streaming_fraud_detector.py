#!/usr/bin/env python3
"""
Real-time Streaming Fraud Detection System
Simulates Apache Kafka and Spark Streaming for sub-second transaction processing
"""

import pandas as pd
import numpy as np
import time
import json
import datetime
import threading
import queue
from collections import deque
import asyncio
import streamlit as st
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px

class KafkaSimulator:
    """Simulates Apache Kafka for high-throughput message queuing"""
    
    def __init__(self, topic_name: str = "transactions", max_queue_size: int = 50000):
        self.topic_name = topic_name
        self.message_queue = queue.Queue(maxsize=max_queue_size)
        self.consumers = []
        self.is_running = False
        self.processing_stats = {
            'messages_produced': 0,
            'messages_consumed': 0,
            'avg_latency_ms': 0,
            'throughput_tps': 0
        }
        
    def produce_message(self, message: Dict[str, Any]) -> bool:
        """Produce a message to the Kafka topic (simulated)"""
        try:
            # Add timestamp for latency tracking
            message['kafka_timestamp'] = time.time()
            message['message_id'] = f"msg_{self.processing_stats['messages_produced']:06d}"
            
            # Non-blocking put with timeout
            self.message_queue.put(message, timeout=0.1)
            self.processing_stats['messages_produced'] += 1
            return True
        except queue.Full:
            return False
    
    def consume_messages(self, batch_size: int = 100, timeout_ms: int = 100) -> List[Dict[str, Any]]:
        """Consume messages from the Kafka topic (simulated)"""
        messages = []
        start_time = time.time()
        
        try:
            # Try to get messages with timeout
            while len(messages) < batch_size and (time.time() - start_time) * 1000 < timeout_ms:
                try:
                    message = self.message_queue.get(timeout=0.001)  # 1ms timeout
                    messages.append(message)
                    self.processing_stats['messages_consumed'] += 1
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error consuming messages: {e}")
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Kafka processing statistics"""
        current_time = time.time()
        
        # Calculate throughput (messages per second)
        if hasattr(self, '_last_stats_time'):
            time_diff = current_time - self._last_stats_time
            if time_diff > 0:
                self.processing_stats['throughput_tps'] = (
                    self.processing_stats['messages_consumed'] - getattr(self, '_last_consumed', 0)
                ) / time_diff
        
        self._last_stats_time = current_time
        self._last_consumed = self.processing_stats['messages_consumed']
        
        return self.processing_stats.copy()

class SparkStreamingSimulator:
    """Simulates Spark Streaming for real-time data processing"""
    
    def __init__(self, kafka_simulator: KafkaSimulator, detector: Any):
        self.kafka = kafka_simulator
        self.detector = detector
        self.is_running = False
        self.processing_thread = None
        self.micro_batch_size = 50
        self.micro_batch_interval_ms = 100  # 100ms micro-batches
        self.processed_transactions = deque(maxlen=1000)
        self.fraud_alerts = deque(maxlen=100)
        self.processing_stats = {
            'batches_processed': 0,
            'transactions_processed': 0,
            'fraud_detected': 0,
            'avg_processing_time_ms': 0,
            'current_batch_size': 0
        }
        
    def start_streaming(self):
        """Start the streaming processing"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_stream, daemon=True)
            self.processing_thread.start()
            print("🚀 Spark Streaming started")
    
    def stop_streaming(self):
        """Stop the streaming processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        print("⏹️ Spark Streaming stopped")
    
    def _process_stream(self):
        """Main streaming processing loop"""
        while self.is_running:
            batch_start_time = time.time()
            
            # Consume messages from Kafka
            messages = self.kafka.consume_messages(
                batch_size=self.micro_batch_size,
                timeout_ms=self.micro_batch_interval_ms
            )
            
            if messages:
                self.processing_stats['current_batch_size'] = len(messages)
                self.processing_stats['batches_processed'] += 1
                
                # Process each transaction
                for message in messages:
                    self._process_transaction(message)
                
                # Calculate processing time
                batch_time_ms = (time.time() - batch_start_time) * 1000
                self.processing_stats['avg_processing_time_ms'] = (
                    (self.processing_stats['avg_processing_time_ms'] * 
                     (self.processing_stats['batches_processed'] - 1) + batch_time_ms) /
                    self.processing_stats['batches_processed']
                )
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
    
    def _process_transaction(self, transaction_data: Dict[str, Any]):
        """Process a single transaction for fraud detection"""
        try:
            # Add processing timestamp
            transaction_data['processing_timestamp'] = time.time()
            
            # Calculate Kafka latency
            if 'kafka_timestamp' in transaction_data:
                kafka_latency_ms = (transaction_data['processing_timestamp'] - 
                                  transaction_data['kafka_timestamp']) * 1000
                transaction_data['kafka_latency_ms'] = kafka_latency_ms
            
            # Feature engineering (simulated Spark operations)
            enriched_transaction = self._engineer_features(transaction_data)
            
            # Fraud detection using your existing detector
            if self.detector and hasattr(self.detector, 'predict_transaction_risk'):
                fraud_result = self.detector.predict_transaction_risk(enriched_transaction)
                transaction_data['fraud_result'] = fraud_result
                
                # Track fraud alerts
                if fraud_result and fraud_result.get('risk_level') in ['HIGH_RISK', 'MEDIUM_RISK']:
                    self.fraud_alerts.append({
                        'transaction_id': transaction_data.get('transaction_id'),
                        'risk_level': fraud_result.get('risk_level'),
                        'probability': fraud_result.get('risk_probability'),
                        'timestamp': transaction_data['processing_timestamp'],
                        'amount': transaction_data.get('amount'),
                        'customer_id': transaction_data.get('customer_id')
                    })
                    self.processing_stats['fraud_detected'] += 1
            
            # Store processed transaction
            self.processed_transactions.append(transaction_data)
            self.processing_stats['transactions_processed'] += 1
            
        except Exception as e:
            print(f"Error processing transaction: {e}")
    
    def _engineer_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features for fraud detection (simulated Spark operations)"""
        # Add time-based features
        timestamp = datetime.datetime.fromtimestamp(transaction.get('timestamp', time.time()))
        transaction['hour'] = timestamp.hour
        transaction['day_of_week'] = timestamp.weekday()
        transaction['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        transaction['is_night'] = 1 if timestamp.hour >= 22 or timestamp.hour <= 6 else 0
        transaction['is_business_hours'] = 1 if 9 <= timestamp.hour <= 17 else 0
        
        # Add amount-based features
        amount = transaction.get('amount', 0)
        transaction['amount_log'] = np.log1p(amount)
        transaction['is_high_value'] = 1 if amount > 1000 else 0
        transaction['amount_percentile'] = 0.5  # Default percentile
        
        # Add transaction type features
        transaction['is_online'] = 1 if transaction.get('transaction_type') == 'ONLINE' else 0
        transaction['is_atm'] = 1 if transaction.get('transaction_type') == 'ATM' else 0
        transaction['is_international'] = 1 if transaction.get('location') == 'INTERNATIONAL' else 0
        transaction['card_not_present'] = 1 if not transaction.get('card_present', True) else 0
        
        # Add customer behavior features (simulated)
        transaction['avg_amount'] = amount * 0.8  # Simulated average
        transaction['std_amount'] = amount * 0.2  # Simulated std
        transaction['transaction_count'] = np.random.randint(1, 100)  # Simulated count
        transaction['fraud_rate'] = 0.005  # Default fraud rate
        
        # Add balance features
        transaction['balance_change'] = transaction.get('balance_after', 0) - transaction.get('balance_before', 0)
        transaction['balance_change_pct'] = transaction['balance_change'] / (transaction.get('balance_before', 1) + 1)
        
        # Add risk score
        transaction['risk_score'] = (
            transaction['is_high_value'] * 2 +
            transaction['is_international'] * 3 +
            transaction['card_not_present'] * 2 +
            transaction['is_night'] * 1 +
            (1 if transaction.get('previous_fraud_flag') else 0) * 5
        )
        
        return transaction
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming processing statistics"""
        return self.processing_stats.copy()
    
    def get_recent_transactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent processed transactions"""
        return list(self.processed_transactions)[-limit:]
    
    def get_fraud_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent fraud alerts"""
        return list(self.fraud_alerts)[-limit:]

class RealTimeFraudDetection:
    """Main class for real-time fraud detection with Kafka and Spark simulation"""
    
    def __init__(self, detector: Any):
        self.detector = detector
        self.kafka = KafkaSimulator()
        self.spark_streaming = SparkStreamingSimulator(self.kafka, detector)
        self.transaction_generator = None
        self.is_generating = False
        
    def start_system(self):
        """Start the entire real-time fraud detection system"""
        self.spark_streaming.start_streaming()
        print("✅ Real-time fraud detection system started")
    
    def stop_system(self):
        """Stop the entire real-time fraud detection system"""
        self.stop_transaction_generation()
        self.spark_streaming.stop_streaming()
        print("⏹️ Real-time fraud detection system stopped")
    
    def start_transaction_generation(self, tps: int = 100):
        """Start generating sample transactions at specified TPS"""
        if not self.is_generating:
            self.is_generating = True
            self.transaction_generator = threading.Thread(
                target=self._generate_transactions, 
                args=(tps,), 
                daemon=True
            )
            self.transaction_generator.start()
            print(f"🔄 Transaction generation started at {tps} TPS")
    
    def stop_transaction_generation(self):
        """Stop generating transactions"""
        self.is_generating = False
        if self.transaction_generator:
            self.transaction_generator.join(timeout=1)
        print("⏹️ Transaction generation stopped")
    
    def _generate_transactions(self, tps: int):
        """Generate sample transactions at specified rate"""
        interval = 1.0 / tps
        
        while self.is_generating:
            start_time = time.time()
            
            # Generate a transaction
            transaction = self._create_sample_transaction()
            
            # Send to Kafka
            success = self.kafka.produce_message(transaction)
            
            if not success:
                print("⚠️ Kafka queue full, dropping transaction")
            
            # Sleep to maintain TPS
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def _create_sample_transaction(self) -> Dict[str, Any]:
        """Create a realistic sample transaction"""
        np.random.seed(int(time.time() * 1000) % 1000000)  # Dynamic seed
        
        # Base transaction
        transaction = {
            'transaction_id': f"TXN_{int(time.time() * 1000):06d}",
            'customer_id': np.random.randint(1, 501),
            'amount': np.random.exponential(100),
            'transaction_type': np.random.choice(['ATM', 'POS', 'ONLINE', 'TRANSFER']),
            'merchant_category': np.random.choice(['RETAIL', 'FOOD', 'TRAVEL', 'UTILITIES', 'E-COMMERCE', 'TRANSPORT', 'HEALTHCARE', 'EDUCATION']),
            'location': np.random.choice(['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Sumatera Utara', 'Sulawesi Selatan', 'Banten', 'Bali', 'Sumatera Selatan', 'Riau']),
            'device_type': np.random.choice(['MOBILE', 'DESKTOP', 'ATM', 'POS']),
            'card_present': np.random.choice([True, False]),
            'previous_fraud_flag': np.random.choice([True, False], p=[0.95, 0.05]),
            'account_age_days': np.random.randint(1, 3650),
            'balance_before': np.random.uniform(0, 10000),
            'balance_after': np.random.uniform(0, 10000),
            'timestamp': time.time()
        }
        
        # Add some fraud patterns occasionally
        if np.random.random() < 0.01:  # 1% chance of suspicious transaction
            transaction['amount'] = np.random.uniform(1000, 5000)
            transaction['location'] = 'INTERNATIONAL'
            transaction['card_present'] = False
            transaction['hour'] = np.random.choice([1, 2, 3, 4, 5, 22, 23])
        
        return transaction
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        kafka_stats = self.kafka.get_stats()
        spark_stats = self.spark_streaming.get_stats()
        
        return {
            'kafka': kafka_stats,
            'spark_streaming': spark_stats,
            'system_status': {
                'is_generating': self.is_generating,
                'is_streaming': self.spark_streaming.is_running,
                'total_transactions': spark_stats['transactions_processed'],
                'total_fraud_detected': spark_stats['fraud_detected'],
                'fraud_rate': (spark_stats['fraud_detected'] / 
                              max(spark_stats['transactions_processed'], 1)) * 100
            }
        }

def create_streaming_dashboard_tab():
    """Create the streaming dashboard tab for Streamlit"""
    
    # Initialize session state
    if 'streaming_system' not in st.session_state:
        st.session_state.streaming_system = None
    if 'system_started' not in st.session_state:
        st.session_state.system_started = False
    
    st.markdown("""
    <div class="dashboard-container">
        <h2>🚀 Real-Time Streaming Fraud Detection</h2>
        <p>Apache Kafka + Spark Streaming Simulation for Sub-Second Processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Start System", type="primary"):
            if not st.session_state.system_started:
                # Initialize system
                detector = st.session_state.get('detector')
                if detector:
                    st.session_state.streaming_system = RealTimeFraudDetection(detector)
                    st.session_state.streaming_system.start_system()
                    st.session_state.system_started = True
                    st.success("✅ Real-time system started!")
                else:
                    st.error("❌ Please load the fraud detection model first")
    
    with col2:
        if st.button("⏹️ Stop System"):
            if st.session_state.system_started and st.session_state.streaming_system:
                st.session_state.streaming_system.stop_system()
                st.session_state.system_started = False
                st.success("⏹️ System stopped!")
    
    with col3:
        tps = st.slider("TPS", 10, 200, 75, help="Transactions per second")
    
    with col4:
        if st.session_state.system_started:
            if st.button("🔄 Start Generation"):
                st.session_state.streaming_system.start_transaction_generation(tps)
                st.success(f"🔄 Generating {tps} TPS")
            if st.button("⏹️ Stop Generation"):
                st.session_state.streaming_system.stop_transaction_generation()
                st.success("⏹️ Generation stopped")
    
    # Real-time Metrics
    if st.session_state.system_started and st.session_state.streaming_system:
        system = st.session_state.streaming_system
        
        # Auto-refresh every 2 seconds
        if st.button("🔄 Refresh Metrics") or 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        # Get system stats
        stats = system.get_system_stats()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Throughput", 
                f"{stats['kafka']['throughput_tps']:.1f} TPS",
                help="Transactions processed per second"
            )
        
        with col2:
            st.metric(
                "Latency", 
                f"{stats['spark_streaming']['avg_processing_time_ms']:.1f} ms",
                help="Average processing time per batch"
            )
        
        with col3:
            st.metric(
                "Total Processed", 
                f"{stats['system_status']['total_transactions']:,}",
                help="Total transactions processed"
            )
        
        with col4:
            st.metric(
                "Fraud Detected", 
                f"{stats['system_status']['total_fraud_detected']}",
                f"({stats['system_status']['fraud_rate']:.2f}%)",
                help="Fraud detection rate"
            )
        
        # Real-time Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Processing Performance")
            
            # Create performance chart
            recent_transactions = system.spark_streaming.get_recent_transactions(100)
            if recent_transactions:
                df = pd.DataFrame(recent_transactions)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                fig = px.line(
                    df, 
                    x='timestamp', 
                    y='kafka_latency_ms',
                    title="Kafka Latency Over Time",
                    labels={'kafka_latency_ms': 'Latency (ms)', 'timestamp': 'Time'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🚨 Fraud Alerts")
            
            # Display recent fraud alerts
            fraud_alerts = system.spark_streaming.get_fraud_alerts(10)
            if fraud_alerts:
                for alert in reversed(fraud_alerts):
                    risk_color = "🔴" if alert['risk_level'] == 'HIGH_RISK' else "🟡"
                    st.markdown(f"""
                    {risk_color} **{alert['risk_level']}** - TXN: {alert['transaction_id']}
                    - Amount: ${alert['amount']:.2f}
                    - Probability: {alert['probability']:.2%}
                    - Customer: {alert['customer_id']}
                    """)
            else:
                st.info("No recent fraud alerts")
        
        # Transaction Feed
        st.subheader("📋 Live Transaction Feed")
        
        recent_transactions = system.spark_streaming.get_recent_transactions(20)
        if recent_transactions:
            df = pd.DataFrame(recent_transactions)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['risk_level'] = df['fraud_result'].apply(
                lambda x: x.get('risk_level', 'UNKNOWN') if x else 'UNKNOWN'
            )
            
            # Color code by risk level
            def color_risk(val):
                if val == 'HIGH_RISK':
                    return 'background-color: #ffebee'
                elif val == 'MEDIUM_RISK':
                    return 'background-color: #fff3e0'
                elif val == 'LOW_RISK':
                    return 'background-color: #e8f5e8'
                return ''
            
            display_df = df[['transaction_id', 'customer_id', 'amount', 'transaction_type', 
                           'location', 'risk_level', 'timestamp']].copy()
            display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:.2f}")
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            
            st.dataframe(
                display_df.style.applymap(color_risk, subset=['risk_level']),
                use_container_width=True,
                height=400
            )
    
    else:
        st.info("🚀 Click 'Start System' to begin real-time fraud detection")
        
        # Show system architecture
        st.markdown("""
        ### 🏗️ System Architecture
        
        ```
        Transaction Generator → Apache Kafka → Spark Streaming → Fraud Detection → Dashboard
                ↓                    ↓              ↓              ↓              ↓
              100 TPS            <1ms latency   100ms batches   ML Models    Real-time UI
        ```
        
        **Key Features:**
        - ⚡ **Sub-second processing** with micro-batch architecture
        - 🔄 **Real-time streaming** with Apache Kafka simulation
        - 🎯 **Multi-model fraud detection** (Logistic Regression, Random Forest, Isolation Forest)
        - 📊 **Live metrics and monitoring**
        - 🚨 **Instant fraud alerts**
        """) 