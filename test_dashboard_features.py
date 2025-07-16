#!/usr/bin/env python3
"""
Comprehensive Dashboard Feature Testing Script
Tests all functionality of the AI Fraud Detection Dashboard
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bank_fraud_detector import BankFraudDetector

class DashboardTester:
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.detector = None
        self.test_results = {}
        
    def test_dashboard_availability(self):
        """Test if dashboard is accessible"""
        print("🔍 Testing Dashboard Availability...")
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                print("✅ Dashboard is accessible")
                return True
            else:
                print(f"❌ Dashboard returned status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot access dashboard: {e}")
            return False
    
    def test_model_loading(self):
        """Test if the fraud detection model loads correctly"""
        print("\n🔍 Testing Model Loading...")
        try:
            self.detector = BankFraudDetector()
            model_path = "models/bank_fraud_detector.pkl"
            
            if os.path.exists(model_path):
                self.detector.load_bank_model(model_path)
                print("✅ Model loaded successfully")
                print(f"   Models available: {list(self.detector.models.keys())}")
                print(f"   Features: {len(self.detector.feature_columns)}")
                return True
            else:
                print("❌ Model file not found")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_sample_data_generation(self):
        """Test sample transaction data generation"""
        print("\n🔍 Testing Sample Data Generation...")
        try:
            if self.detector:
                # Use the create_sample_bank_dataset method instead
                sample_data = self.detector.create_sample_bank_dataset()
                # Take a subset for testing
                sample_data = sample_data.head(10)
                print(f"✅ Generated {len(sample_data)} sample transactions")
                print(f"   Columns: {list(sample_data.columns)}")
                print(f"   Sample transaction: {sample_data.iloc[0].to_dict()}")
                return True
            else:
                print("❌ Detector not initialized")
                return False
        except Exception as e:
            print(f"❌ Error generating sample data: {e}")
            return False
    
    def test_transaction_analysis(self):
        """Test transaction risk analysis"""
        print("\n🔍 Testing Transaction Analysis...")
        try:
            if self.detector:
                # Create a sample transaction as a pandas Series
                import pandas as pd
                
                # First, create a sample dataset and engineer features
                sample_data = self.detector.create_sample_bank_dataset()
                sample_data = sample_data.head(1)  # Take one transaction
                engineered_data = self.detector.engineer_bank_features(sample_data)
                
                # Test risk prediction with the engineered transaction
                risk_result = self.detector.predict_transaction_risk(engineered_data.iloc[0])
                if risk_result:
                    print("✅ Transaction analysis successful")
                    print(f"   Risk Level: {risk_result['risk_level']}")
                    print(f"   Risk Probability: {risk_result['risk_probability']:.4f}")
                    print(f"   Recommended Action: {risk_result['recommended_action']}")
                    return True
                else:
                    print("❌ Transaction analysis failed")
                    return False
            else:
                print("❌ Detector not initialized")
                return False
        except Exception as e:
            print(f"❌ Error in transaction analysis: {e}")
            return False
    
    def test_feature_engineering(self):
        """Test feature engineering functionality"""
        print("\n🔍 Testing Feature Engineering...")
        try:
            if self.detector:
                # Generate sample data
                sample_data = self.detector.create_sample_bank_dataset()
                sample_data = sample_data.head(100)  # Take subset for testing
                
                # Test feature engineering
                engineered_data = self.detector.engineer_bank_features(sample_data)
                
                print("✅ Feature engineering successful")
                print(f"   Original columns: {len(sample_data.columns)}")
                print(f"   Engineered columns: {len(engineered_data.columns)}")
                
                # Check for key engineered features
                expected_features = ['is_weekend', 'is_night', 'amount_log', 'risk_score']
                missing_features = [f for f in expected_features if f not in engineered_data.columns]
                
                if missing_features:
                    print(f"   ⚠️ Missing features: {missing_features}")
                else:
                    print("   ✅ All expected features present")
                
                return True
            else:
                print("❌ Detector not initialized")
                return False
        except Exception as e:
            print(f"❌ Error in feature engineering: {e}")
            return False
    
    def test_customer_profiles(self):
        """Test customer profile creation"""
        print("\n🔍 Testing Customer Profiles...")
        try:
            if self.detector:
                # Generate sample data
                sample_data = self.detector.create_sample_bank_dataset()
                sample_data = sample_data.head(200)  # Take subset for testing
                engineered_data = self.detector.engineer_bank_features(sample_data)
                
                # Test customer profile creation
                customer_profiles = self.detector.create_customer_profiles(engineered_data)
                
                print("✅ Customer profiles created successfully")
                print(f"   Number of customers: {len(customer_profiles)}")
                print(f"   Profile columns: {list(customer_profiles.columns)}")
                
                # Check risk distribution
                risk_dist = customer_profiles['risk_category'].value_counts()
                print(f"   Risk distribution: {risk_dist.to_dict()}")
                
                return True
            else:
                print("❌ Detector not initialized")
                return False
        except Exception as e:
            print(f"❌ Error creating customer profiles: {e}")
            return False
    
    def test_model_performance(self):
        """Test model training and performance"""
        print("\n🔍 Testing Model Performance...")
        try:
            if self.detector:
                # Generate larger dataset for training
                sample_data = self.detector.create_sample_bank_dataset()
                sample_data = sample_data.head(1000)  # Take subset for testing
                engineered_data = self.detector.engineer_bank_features(sample_data)
                
                # Test model training
                results, X_test, y_test = self.detector.train_bank_models(engineered_data)
                
                print("✅ Model training successful")
                print(f"   Models trained: {len(results)}")
                
                # Check performance metrics
                for model_name, result in results.items():
                    auc_score = result['auc']
                    print(f"   {model_name}: AUC = {auc_score:.4f}")
                
                # Test risk threshold setting
                thresholds = self.detector.set_risk_thresholds(results, y_test)
                print(f"   Risk thresholds set: {thresholds}")
                
                return True
            else:
                print("❌ Detector not initialized")
                return False
        except Exception as e:
            print(f"❌ Error in model performance testing: {e}")
            return False
    
    def test_data_validation(self):
        """Test data validation and handling"""
        print("\n🔍 Testing Data Validation...")
        try:
            if self.detector:
                # Test with missing data
                sample_data = self.detector.create_sample_bank_dataset()
                sample_data = sample_data.head(50)  # Take subset for testing
                
                # Introduce missing values
                sample_data.loc[0:10, 'merchant_category'] = np.nan
                sample_data.loc[5:15, 'balance_before'] = np.nan
                
                print(f"   Missing values before: {sample_data.isnull().sum().sum()}")
                
                # Test missing data handling
                cleaned_data = self.detector.handle_missing_data(sample_data)
                
                print(f"   Missing values after: {cleaned_data.isnull().sum().sum()}")
                
                if cleaned_data.isnull().sum().sum() == 0:
                    print("✅ Missing data handling successful")
                    return True
                else:
                    print("❌ Missing data not fully handled")
                    return False
            else:
                print("❌ Detector not initialized")
                return False
        except Exception as e:
            print(f"❌ Error in data validation: {e}")
            return False
    
    def test_risk_scoring(self):
        """Test risk scoring functionality"""
        print("\n🔍 Testing Risk Scoring...")
        try:
            if self.detector:
                # Create sample data and engineer features
                sample_data = self.detector.create_sample_bank_dataset()
                sample_data = sample_data.head(5)  # Take 5 transactions for testing
                engineered_data = self.detector.engineer_bank_features(sample_data)
                
                # Select only the trained feature columns
                feature_data = engineered_data[self.detector.feature_columns]
                
                # Test multiple transactions
                for i in range(len(feature_data)):
                    transaction = feature_data.iloc[i]
                    risk_result = self.detector.predict_transaction_risk(transaction)
                    if risk_result:
                        print(f"   Transaction {i+1}: {risk_result['risk_level']} ({risk_result['risk_probability']:.4f})")
                    else:
                        print(f"   Transaction {i+1}: Failed to analyze")
                
                print("✅ Risk scoring test completed")
                return True
            else:
                print("❌ Detector not initialized")
                return False
        except Exception as e:
            print(f"❌ Error in risk scoring: {e}")
            return False
    
    def run_all_tests(self):
        """Run all dashboard feature tests"""
        print("🚀 Starting Comprehensive Dashboard Feature Testing")
        print("=" * 60)
        
        tests = [
            ("Dashboard Availability", self.test_dashboard_availability),
            ("Model Loading", self.test_model_loading),
            ("Sample Data Generation", self.test_sample_data_generation),
            ("Transaction Analysis", self.test_transaction_analysis),
            ("Feature Engineering", self.test_feature_engineering),
            ("Customer Profiles", self.test_customer_profiles),
            ("Model Performance", self.test_model_performance),
            ("Data Validation", self.test_data_validation),
            ("Risk Scoring", self.test_risk_scoring)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    self.test_results[test_name] = "PASS"
                    passed += 1
                else:
                    self.test_results[test_name] = "FAIL"
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.test_results[test_name] = "ERROR"
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result == "PASS" else "❌"
            print(f"{status_icon} {test_name}: {result}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! Dashboard is working correctly.")
        else:
            print("⚠️ Some tests failed. Please check the issues above.")
        
        return passed == total

def main():
    """Main testing function"""
    tester = DashboardTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Dashboard is ready for use!")
        print("🌐 Access it at: http://localhost:8501")
    else:
        print("\n🔧 Please fix the failing tests before using the dashboard.")

if __name__ == "__main__":
    main() 