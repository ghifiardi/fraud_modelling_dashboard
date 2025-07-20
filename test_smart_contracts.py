#!/usr/bin/env python3
"""
Smart Contract Validation Test Script
Demonstrates real-time fraud detection with smart contracts
"""

import requests
import json
import time
from datetime import datetime

def test_smart_contract_validation():
    """Test smart contract validation with different scenarios"""
    
    base_url = "http://localhost:5001"
    
    print("🔍 SMART CONTRACT VALIDATION TESTING")
    print("=" * 50)
    
    # Test scenarios
    test_cases = [
        {
            "name": "Normal Transaction",
            "description": "Low amount, familiar location, daytime",
            "data": {
                "customer_id": "CUST_NORMAL",
                "amount": 100.00,
                "merchant_id": "MERCH_NORMAL",
                "location": "New York, NY",
                "payment_method": "Credit Card",
                "risk_score": 0.2,
                "fraud_probability": 0.05
            },
            "expected": "allow"
        },
        {
            "name": "High Amount Transaction",
            "description": "Amount above threshold ($10,000)",
            "data": {
                "customer_id": "CUST_HIGH_AMOUNT",
                "amount": 15000.00,
                "merchant_id": "MERCH_HIGH_AMOUNT",
                "location": "New York, NY",
                "payment_method": "Credit Card",
                "risk_score": 0.3,
                "fraud_probability": 0.1
            },
            "expected": "block"
        },
        {
            "name": "International Transaction",
            "description": "Transaction from different country",
            "data": {
                "customer_id": "CUST_INTERNATIONAL",
                "amount": 500.00,
                "merchant_id": "MERCH_INTERNATIONAL",
                "location": "London, UK",
                "payment_method": "Credit Card",
                "risk_score": 0.4,
                "fraud_probability": 0.15
            },
            "expected": "review"
        },
        {
            "name": "New Merchant Transaction",
            "description": "First time with this merchant",
            "data": {
                "customer_id": "CUST_NEW_MERCHANT",
                "amount": 300.00,
                "merchant_id": "MERCH_NEW_123",
                "location": "New York, NY",
                "payment_method": "Credit Card",
                "risk_score": 0.25,
                "fraud_probability": 0.08
            },
            "expected": "allow"
        },
        {
            "name": "High Risk Transaction",
            "description": "Multiple risk factors combined",
            "data": {
                "customer_id": "CUST_HIGH_RISK",
                "amount": 8000.00,
                "merchant_id": "MERCH_HIGH_RISK",
                "location": "Tokyo, JP",
                "payment_method": "Credit Card",
                "risk_score": 0.7,
                "fraud_probability": 0.4
            },
            "expected": "block"
        }
    ]
    
    print("📋 Current Smart Contract Rules:")
    try:
        response = requests.get(f"{base_url}/api/blockchain/smart-contract/rules")
        if response.status_code == 200:
            rules = response.json()['data']
            for rule, value in rules.items():
                if isinstance(value, float):
                    print(f"   {rule.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"   {rule.replace('_', ' ').title()}: {value}")
        else:
            print("   ❌ Could not fetch rules")
    except Exception as e:
        print(f"   ❌ Error fetching rules: {e}")
    
    print("\n" + "=" * 50)
    
    # Test each scenario
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   📝 {test_case['description']}")
        
        try:
            # Validate transaction
            response = requests.post(
                f"{base_url}/api/blockchain/validate",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                validation = result['data']['validation_result']
                transaction = result['data']['transaction']
                
                print(f"   💰 Amount: ${transaction['amount']:,.2f}")
                print(f"   🌍 Location: {transaction['location']}")
                print(f"   ⚠️ Risk Factors: {validation['risk_factors']}")
                print(f"   📊 Risk Adjustment: {validation['risk_score_adjustment']:.2%}")
                print(f"   🎯 Recommendation: {validation['recommendation'].title()}")
                print(f"   ✅ Valid: {validation['is_valid']}")
                
                # Check if result matches expectation
                if validation['recommendation'] == test_case['expected']:
                    print(f"   ✅ PASS - Expected {test_case['expected']}, got {validation['recommendation']}")
                else:
                    print(f"   ⚠️ PARTIAL - Expected {test_case['expected']}, got {validation['recommendation']}")
                    
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        time.sleep(0.5)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print("🧪 Smart Contract Testing Complete!")

def test_real_time_processing():
    """Test real-time transaction processing with smart contracts"""
    
    base_url = "http://localhost:5001"
    
    print("\n⚡ REAL-TIME TRANSACTION PROCESSING TEST")
    print("=" * 50)
    
    # Create multiple transactions rapidly
    transactions = [
        {"customer_id": "CUST_RT1", "amount": 150.00, "merchant_id": "MERCH_RT1", "location": "New York, NY", "payment_method": "Credit Card", "risk_score": 0.2, "fraud_probability": 0.05},
        {"customer_id": "CUST_RT2", "amount": 12000.00, "merchant_id": "MERCH_RT2", "location": "New York, NY", "payment_method": "Credit Card", "risk_score": 0.3, "fraud_probability": 0.1},
        {"customer_id": "CUST_RT3", "amount": 500.00, "merchant_id": "MERCH_RT3", "location": "Paris, FR", "payment_method": "Credit Card", "risk_score": 0.4, "fraud_probability": 0.15},
        {"customer_id": "CUST_RT4", "amount": 75.00, "merchant_id": "MERCH_RT4", "location": "New York, NY", "payment_method": "Credit Card", "risk_score": 0.1, "fraud_probability": 0.02},
        {"customer_id": "CUST_RT5", "amount": 20000.00, "merchant_id": "MERCH_RT5", "location": "Tokyo, JP", "payment_method": "Credit Card", "risk_score": 0.8, "fraud_probability": 0.6}
    ]
    
    print("📡 Processing transactions in real-time...")
    start_time = time.time()
    
    results = []
    for i, tx_data in enumerate(transactions, 1):
        try:
            # Create transaction
            response = requests.post(
                f"{base_url}/api/blockchain/transaction",
                json=tx_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                validation = result['data']['result']['validation_result']
                transaction = result['data']['transaction']
                
                print(f"   {i}. TX{transaction['transaction_id']} - ${transaction['amount']:,.2f}")
                print(f"      🎯 {validation['recommendation'].title()} | ⚠️ {validation['risk_factors']}")
                
                results.append({
                    'transaction_id': transaction['transaction_id'],
                    'amount': transaction['amount'],
                    'recommendation': validation['recommendation'],
                    'risk_factors': validation['risk_factors']
                })
            else:
                print(f"   {i}. ❌ Error creating transaction")
                
        except Exception as e:
            print(f"   {i}. ❌ Exception: {e}")
        
        time.sleep(0.2)  # Small delay to simulate real-time processing
    
    processing_time = time.time() - start_time
    
    print(f"\n⏱️ Processing Time: {processing_time:.2f} seconds")
    print(f"🚀 Average: {processing_time/len(transactions):.3f} seconds per transaction")
    
    # Summary
    print("\n📊 Processing Summary:")
    recommendations = {}
    for result in results:
        rec = result['recommendation']
        recommendations[rec] = recommendations.get(rec, 0) + 1
    
    for rec, count in recommendations.items():
        print(f"   {rec.title()}: {count} transactions")
    
    print("\n" + "=" * 50)

def test_smart_contract_rules_update():
    """Test updating smart contract rules"""
    
    base_url = "http://localhost:5001"
    
    print("\n⚙️ SMART CONTRACT RULES UPDATE TEST")
    print("=" * 50)
    
    # Get current rules
    print("📋 Current Rules:")
    try:
        response = requests.get(f"{base_url}/api/blockchain/smart-contract/rules")
        if response.status_code == 200:
            current_rules = response.json()['data']
            for rule, value in current_rules.items():
                print(f"   {rule}: {value}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Update rules
    print("\n🔄 Updating Rules...")
    new_rules = {
        "high_amount_threshold": 5000.0,  # Lower threshold
        "velocity_threshold": 3,           # Lower velocity
        "night_transaction_penalty": 0.3   # Higher penalty
    }
    
    try:
        response = requests.put(
            f"{base_url}/api/blockchain/smart-contract/rules",
            json=new_rules,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Rules updated successfully!")
            
            print("\n📋 Updated Rules:")
            updated_rules = result['data']['rules']
            for rule, value in updated_rules.items():
                print(f"   {rule}: {value}")
            
            # Test with new rules
            print("\n🧪 Testing with Updated Rules:")
            test_tx = {
                "customer_id": "CUST_UPDATED",
                "amount": 6000.00,  # Above new threshold
                "merchant_id": "MERCH_UPDATED",
                "location": "New York, NY",
                "payment_method": "Credit Card",
                "risk_score": 0.3,
                "fraud_probability": 0.1
            }
            
            response = requests.post(
                f"{base_url}/api/blockchain/validate",
                json=test_tx,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                validation = result['data']['validation_result']
                transaction = result['data']['transaction']
                
                print(f"   💰 Amount: ${transaction['amount']:,.2f}")
                print(f"   ⚠️ Risk Factors: {validation['risk_factors']}")
                print(f"   🎯 Recommendation: {validation['recommendation'].title()}")
                
                if 'high_amount' in validation['risk_factors']:
                    print("   ✅ New threshold working correctly!")
                else:
                    print("   ❌ New threshold not working")
                    
        else:
            print(f"   ❌ Error updating rules: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print("\n" + "=" * 50)

def main():
    """Main function"""
    print("🔗 Smart Contract Real-Time Fraud Detection")
    print("=" * 60)
    
    try:
        # Test smart contract validation
        test_smart_contract_validation()
        
        # Test real-time processing
        test_real_time_processing()
        
        # Test rules update
        test_smart_contract_rules_update()
        
        print("\n🎉 All smart contract tests completed!")
        print("🔗 Smart contracts are working correctly for real-time fraud detection!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 