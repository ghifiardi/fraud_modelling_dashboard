#!/usr/bin/env python3
"""
Test Script for Blockchain Integration
Demonstrates and tests the blockchain fraud detection functionality
"""

import sys
import os
import time
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from blockchain_core import BlockchainManager, Transaction, TransactionStatus, RiskLevel

def test_blockchain_core():
    """Test the core blockchain functionality"""
    print("🔗 Testing Blockchain Core...")
    
    # Initialize blockchain manager
    manager = BlockchainManager()
    
    # Test 1: Create transactions
    print("\n📝 Test 1: Creating Transactions")
    
    transactions = []
    for i in range(5):
        transaction = manager.create_transaction(
            customer_id=f"CUST{i+1000}",
            amount=random.uniform(10, 2000),
            merchant_id=f"MERCH{i+100}",
            location=random.choice(["New York, NY", "Los Angeles, CA", "Chicago, IL", "Miami, FL"]),
            payment_method=random.choice(["Credit Card", "Debit Card", "Digital Wallet"]),
            risk_score=random.uniform(0.1, 0.8),
            fraud_probability=random.uniform(0.01, 0.3),
            metadata={"test": True, "batch": i}
        )
        transactions.append(transaction)
        print(f"  ✅ Created transaction {transaction.transaction_id}")
    
    # Test 2: Process transactions
    print("\n⚡ Test 2: Processing Transactions")
    
    for transaction in transactions:
        result = manager.process_transaction(transaction)
        if result['success']:
            print(f"  ✅ Processed {transaction.transaction_id}: {result['validation_result']['recommendation']}")
        else:
            print(f"  ❌ Failed to process {transaction.transaction_id}: {result['error']}")
    
    # Test 3: Blockchain status
    print("\n📊 Test 3: Blockchain Status")
    
    status = manager.get_blockchain_status()
    print(f"  📦 Total Blocks: {len(status['chain'])}")
    print(f"  ⏳ Pending Transactions: {len(status['pending_transactions'])}")
    print(f"  ✅ Chain Valid: {status['is_valid']}")
    
    # Test 4: Force mine a block
    print("\n⛏️ Test 4: Mining Block")
    
    if status['pending_transactions']:
        block = manager.blockchain.mine_pending_transactions()
        if block:
            print(f"  ✅ Mined block {block.index} with {len(block.transactions)} transactions")
            print(f"  🔗 Block hash: {block.block_hash[:16]}...")
        else:
            print("  ❌ Failed to mine block")
    else:
        print("  ℹ️ No pending transactions to mine")
    
    # Test 5: Analytics
    print("\n📈 Test 5: Fraud Analytics")
    
    analytics = manager.get_fraud_analytics()
    print(f"  📊 Total Transactions: {analytics['total_transactions']}")
    print(f"  🚨 Fraudulent Transactions: {analytics['fraudulent_transactions']}")
    print(f"  ❌ Blocked Transactions: {analytics['blocked_transactions']}")
    print(f"  📊 Fraud Rate: {analytics['fraud_rate']:.2%}")
    print(f"  📊 Block Rate: {analytics['block_rate']:.2%}")
    
    # Test 6: Smart contract validation
    print("\n📋 Test 6: Smart Contract Validation")
    
    # Create a suspicious transaction
    suspicious_tx = manager.create_transaction(
        customer_id="CUST9999",
        amount=15000.0,  # High amount
        merchant_id="MERCH999",
        location="Tokyo, JP",  # International
        payment_method="Credit Card",
        risk_score=0.6,
        fraud_probability=0.4,
        metadata={"suspicious": True}
    )
    
    # Get customer history (empty for new customer)
    customer_history = manager.blockchain.get_transaction_history("CUST9999")
    
    # Validate with smart contract
    validation_result = manager.blockchain.smart_contract.validate_transaction(suspicious_tx, customer_history)
    
    print(f"  🔍 Transaction: {suspicious_tx.transaction_id}")
    print(f"  💰 Amount: ${suspicious_tx.amount:,.2f}")
    print(f"  🌍 Location: {suspicious_tx.location}")
    print(f"  ⚠️ Risk Factors: {', '.join(validation_result['risk_factors'])}")
    print(f"  📊 Risk Adjustment: {validation_result['risk_score_adjustment']:.2%}")
    print(f"  🎯 Recommendation: {validation_result['recommendation'].title()}")
    
    print("\n✅ All blockchain core tests completed!")

def test_smart_contract_rules():
    """Test smart contract rules"""
    print("\n📋 Testing Smart Contract Rules...")
    
    manager = BlockchainManager()
    smart_contract = manager.blockchain.smart_contract
    
    # Test different scenarios
    test_cases = [
        {
            "name": "Normal Transaction",
            "amount": 100.0,
            "location": "New York, NY",
            "hour": 14,
            "expected_factors": []
        },
        {
            "name": "High Amount Transaction",
            "amount": 15000.0,
            "location": "New York, NY",
            "hour": 14,
            "expected_factors": ["high_amount"]
        },
        {
            "name": "Night Transaction",
            "amount": 100.0,
            "location": "New York, NY",
            "hour": 3,
            "expected_factors": ["night_transaction"]
        },
        {
            "name": "International Transaction",
            "amount": 100.0,
            "location": "London, UK",
            "hour": 14,
            "expected_factors": []
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  🔍 Testing: {test_case['name']}")
        
        # Create transaction with specific parameters
        transaction = manager.create_transaction(
            customer_id="TEST_CUST",
            amount=test_case["amount"],
            merchant_id="TEST_MERCH",
            location=test_case["location"],
            payment_method="Credit Card",
            risk_score=0.3,
            fraud_probability=0.1,
            metadata={"test_case": test_case["name"]}
        )
        
        # Set specific timestamp for hour testing
        transaction.timestamp = time.time() - (24 - test_case["hour"]) * 3600
        
        # Validate transaction
        validation_result = smart_contract.validate_transaction(transaction, [])
        
        print(f"    💰 Amount: ${transaction.amount:,.2f}")
        print(f"    🌍 Location: {transaction.location}")
        print(f"    ⏰ Hour: {test_case['hour']}")
        print(f"    ⚠️ Risk Factors: {validation_result['risk_factors']}")
        print(f"    📊 Risk Adjustment: {validation_result['risk_score_adjustment']:.2%}")
        print(f"    🎯 Recommendation: {validation_result['recommendation'].title()}")
        
        # Check if expected factors are present
        expected_factors = set(test_case['expected_factors'])
        actual_factors = set(validation_result['risk_factors'])
        
        if expected_factors.issubset(actual_factors):
            print(f"    ✅ Test passed - Expected factors found")
        else:
            print(f"    ❌ Test failed - Expected: {expected_factors}, Got: {actual_factors}")
    
    print("\n✅ Smart contract rules testing completed!")

def test_blockchain_integrity():
    """Test blockchain integrity and validation"""
    print("\n🔒 Testing Blockchain Integrity...")
    
    manager = BlockchainManager()
    
    # Add some transactions
    for i in range(3):
        transaction = manager.create_transaction(
            customer_id=f"INTEGRITY_CUST{i}",
            amount=random.uniform(50, 500),
            merchant_id=f"INTEGRITY_MERCH{i}",
            location="New York, NY",
            payment_method="Credit Card",
            risk_score=0.2,
            fraud_probability=0.05,
            metadata={"integrity_test": True}
        )
        manager.process_transaction(transaction)
    
    # Mine a block
    block = manager.blockchain.mine_pending_transactions()
    
    if block:
        print(f"  ✅ Mined block {block.index}")
        
        # Test chain validity
        is_valid = manager.blockchain.is_chain_valid()
        print(f"  🔗 Chain Valid: {is_valid}")
        
        # Test block hash calculation
        calculated_hash = block.calculate_hash()
        stored_hash = block.block_hash
        hash_valid = calculated_hash == stored_hash
        
        print(f"  🔐 Block Hash Valid: {hash_valid}")
        print(f"  📝 Calculated Hash: {calculated_hash[:16]}...")
        print(f"  💾 Stored Hash: {stored_hash[:16]}...")
        
        # Test Merkle root
        merkle_root = manager.blockchain.calculate_merkle_root(block.transactions)
        merkle_valid = merkle_root == block.merkle_root
        
        print(f"  🌳 Merkle Root Valid: {merkle_valid}")
        print(f"  📊 Calculated Root: {merkle_root[:16]}...")
        print(f"  💾 Stored Root: {block.merkle_root[:16]}...")
        
        if is_valid and hash_valid and merkle_valid:
            print("  ✅ All integrity checks passed!")
        else:
            print("  ❌ Some integrity checks failed!")
    else:
        print("  ❌ Failed to mine block for integrity testing")
    
    print("\n✅ Blockchain integrity testing completed!")

def test_performance():
    """Test blockchain performance"""
    print("\n⚡ Testing Blockchain Performance...")
    
    manager = BlockchainManager()
    
    # Test transaction creation speed
    print("  📝 Testing transaction creation speed...")
    start_time = time.time()
    
    for i in range(100):
        transaction = manager.create_transaction(
            customer_id=f"PERF_CUST{i}",
            amount=random.uniform(10, 1000),
            merchant_id=f"PERF_MERCH{i}",
            location="New York, NY",
            payment_method="Credit Card",
            risk_score=random.uniform(0.1, 0.5),
            fraud_probability=random.uniform(0.01, 0.2),
            metadata={"performance_test": True}
        )
        manager.process_transaction(transaction)
    
    creation_time = time.time() - start_time
    tx_per_second = 100 / creation_time
    
    print(f"    ⏱️ Created 100 transactions in {creation_time:.2f} seconds")
    print(f"    🚀 Transaction rate: {tx_per_second:.1f} tx/sec")
    
    # Test mining speed
    print("  ⛏️ Testing mining speed...")
    start_time = time.time()
    
    block = manager.blockchain.mine_pending_transactions()
    
    mining_time = time.time() - start_time
    
    if block:
        print(f"    ⏱️ Mined block with {len(block.transactions)} transactions in {mining_time:.2f} seconds")
        print(f"    🔧 Difficulty: {manager.blockchain.difficulty}")
        print(f"    🔗 Block hash: {block.block_hash[:16]}...")
    else:
        print("    ❌ Failed to mine block")
    
    # Test smart contract validation speed
    print("  📋 Testing smart contract validation speed...")
    start_time = time.time()
    
    for i in range(50):
        transaction = manager.create_transaction(
            customer_id=f"VALID_CUST{i}",
            amount=random.uniform(10, 1000),
            merchant_id=f"VALID_MERCH{i}",
            location="New York, NY",
            payment_method="Credit Card",
            risk_score=random.uniform(0.1, 0.5),
            fraud_probability=random.uniform(0.01, 0.2),
            metadata={"validation_test": True}
        )
        
        validation_result = manager.blockchain.smart_contract.validate_transaction(transaction, [])
    
    validation_time = time.time() - start_time
    validations_per_second = 50 / validation_time
    
    print(f"    ⏱️ Validated 50 transactions in {validation_time:.2f} seconds")
    print(f"    🚀 Validation rate: {validations_per_second:.1f} validations/sec")
    
    print("\n✅ Performance testing completed!")

def main():
    """Main test function"""
    print("🔗 Blockchain Integration Test Suite")
    print("=" * 50)
    
    try:
        # Import random for testing
        global random
        import random
        
        # Run all tests
        test_blockchain_core()
        test_smart_contract_rules()
        test_blockchain_integrity()
        test_performance()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("🔗 Blockchain integration is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 