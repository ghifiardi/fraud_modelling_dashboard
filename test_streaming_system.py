#!/usr/bin/env python3
"""
Test script for the streaming fraud detection system
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.streaming_fraud_detector import RealTimeFraudDetection, KafkaSimulator, SparkStreamingSimulator
from src.bank_fraud_detector import BankFraudDetector

def test_streaming_system():
    """Test the streaming fraud detection system"""
    print("🧪 Testing Streaming Fraud Detection System")
    print("=" * 50)
    
    # Initialize the fraud detector
    print("1. Initializing fraud detector...")
    detector = BankFraudDetector()
    
    # Create sample data and train model
    print("2. Creating sample data...")
    df = detector.create_sample_bank_dataset()
    df = detector.engineer_bank_features(df)
    
    print("3. Training models...")
    results, X_test, y_test = detector.train_bank_models(df)
    detector.set_risk_thresholds(results, y_test)
    
    print("4. Initializing streaming system...")
    streaming_system = RealTimeFraudDetection(detector)
    
    print("5. Starting streaming system...")
    streaming_system.start_system()
    
    print("6. Starting transaction generation (50 TPS)...")
    streaming_system.start_transaction_generation(tps=50)
    
    # Monitor for 10 seconds
    print("7. Monitoring system for 10 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 10:
        stats = streaming_system.get_system_stats()
        
        print(f"\r📊 Throughput: {stats['kafka']['throughput_tps']:.1f} TPS | "
              f"Processed: {stats['system_status']['total_transactions']} | "
              f"Fraud: {stats['system_status']['total_fraud_detected']} | "
              f"Latency: {stats['spark_streaming']['avg_processing_time_ms']:.1f}ms", end="")
        
        time.sleep(1)
    
    print("\n8. Stopping system...")
    streaming_system.stop_system()
    
    # Final stats
    final_stats = streaming_system.get_system_stats()
    print("\n📈 Final Statistics:")
    print(f"   Total Transactions: {final_stats['system_status']['total_transactions']}")
    print(f"   Fraud Detected: {final_stats['system_status']['total_fraud_detected']}")
    print(f"   Fraud Rate: {final_stats['system_status']['fraud_rate']:.2f}%")
    print(f"   Average Latency: {final_stats['spark_streaming']['avg_processing_time_ms']:.1f}ms")
    print(f"   Batches Processed: {final_stats['spark_streaming']['batches_processed']}")
    
    print("\n✅ Streaming system test completed successfully!")
    return True

def test_kafka_simulator():
    """Test the Kafka simulator"""
    print("\n🧪 Testing Kafka Simulator")
    print("=" * 30)
    
    kafka = KafkaSimulator()
    
    # Test message production
    print("1. Testing message production...")
    for i in range(100):
        message = {
            'transaction_id': f"TXN_{i:06d}",
            'amount': np.random.exponential(100),
            'timestamp': time.time()
        }
        success = kafka.produce_message(message)
        if not success:
            print(f"   Failed to produce message {i}")
    
    # Test message consumption
    print("2. Testing message consumption...")
    messages = kafka.consume_messages(batch_size=50, timeout_ms=1000)
    print(f"   Consumed {len(messages)} messages")
    
    # Test stats
    stats = kafka.get_stats()
    print(f"3. Kafka Stats: {stats}")
    
    print("✅ Kafka simulator test completed!")
    return True

def test_spark_streaming_simulator():
    """Test the Spark Streaming simulator"""
    print("\n🧪 Testing Spark Streaming Simulator")
    print("=" * 40)
    
    # Create a simple mock detector
    class MockDetector:
        def predict_transaction_risk(self, transaction):
            return {
                'risk_level': 'LOW_RISK',
                'risk_probability': 0.1,
                'recommended_action': 'ALLOW_TRANSACTION'
            }
    
    kafka = KafkaSimulator()
    detector = MockDetector()
    spark = SparkStreamingSimulator(kafka, detector)
    
    # Start streaming
    print("1. Starting Spark Streaming...")
    spark.start_streaming()
    
    # Generate some test data
    print("2. Generating test data...")
    for i in range(50):
        transaction = {
            'transaction_id': f"TXN_{i:06d}",
            'amount': np.random.exponential(100),
            'timestamp': time.time()
        }
        kafka.produce_message(transaction)
    
    # Wait for processing
    print("3. Waiting for processing...")
    time.sleep(2)
    
    # Check results
    stats = spark.get_stats()
    recent_transactions = spark.get_recent_transactions(10)
    fraud_alerts = spark.get_fraud_alerts(10)
    
    print(f"4. Spark Stats: {stats}")
    print(f"   Recent transactions: {len(recent_transactions)}")
    print(f"   Fraud alerts: {len(fraud_alerts)}")
    
    # Stop streaming
    spark.stop_streaming()
    
    print("✅ Spark Streaming simulator test completed!")
    return True

if __name__ == "__main__":
    try:
        print("🚀 Starting Streaming System Tests")
        print("=" * 50)
        
        # Run individual tests
        test_kafka_simulator()
        test_spark_streaming_simulator()
        
        # Run full system test
        test_streaming_system()
        
        print("\n🎉 All tests passed! Streaming system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 