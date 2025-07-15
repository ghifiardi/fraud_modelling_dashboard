#!/usr/bin/env python3
"""
Test script for the AI Fraud Detection Dashboard
Verifies that all components are working correctly
"""

import requests
import time
import json
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bank_fraud_detector import BankFraudDetector

class DashboardTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.detector = None
        
    def test_model_loading(self):
        """Test if the fraud detection model can be loaded."""
        print("🧪 Testing model loading...")
        try:
            self.detector = BankFraudDetector()
            # Create sample data and train a simple model
            df = self.detector.create_sample_bank_dataset()
            df = self.detector.engineer_bank_features(df)
            results, _, _ = self.detector.train_bank_models(df)
            self.detector.save_bank_model("models/bank_fraud_detector.pkl")
            print("✅ Model loaded and saved successfully")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def test_fastapi_server(self):
        """Test if the FastAPI server is running and responding."""
        print("🧪 Testing FastAPI server...")
        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ FastAPI server is healthy: {health_data['status']}")
                return True
            else:
                print(f"❌ FastAPI server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ FastAPI server is not running")
            return False
        except Exception as e:
            print(f"❌ FastAPI server test failed: {e}")
            return False
    
    def test_transaction_prediction(self):
        """Test transaction prediction endpoint."""
        print("🧪 Testing transaction prediction...")
        try:
            transaction_data = {
                "transaction_id": "TEST_001",
                "customer_id": 1,
                "amount": 150.0,
                "transaction_type": "ONLINE",
                "merchant_category": "RETAIL",
                "hour": 14,
                "day_of_week": 2,
                "location": "DOMESTIC",
                "device_type": "MOBILE",
                "card_present": False,
                "previous_fraud_flag": False,
                "account_age_days": 365,
                "balance_before": 5000.0,
                "balance_after": 4850.0
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                json=transaction_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Transaction prediction successful: {result['risk_level']}")
                return True
            else:
                print(f"❌ Transaction prediction failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Transaction prediction test failed: {e}")
            return False
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        print("🧪 Testing metrics endpoint...")
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            if response.status_code == 200:
                metrics = response.json()
                print(f"✅ Metrics retrieved: {len(metrics)} metrics available")
                return True
            else:
                print(f"❌ Metrics endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Metrics test failed: {e}")
            return False
    
    def test_models_endpoint(self):
        """Test models endpoint."""
        print("🧪 Testing models endpoint...")
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                if "error" not in models:
                    print(f"✅ Models endpoint working: {len(models.get('models', {}))} models loaded")
                    return True
                else:
                    print(f"❌ Models endpoint error: {models['error']}")
                    return False
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Models test failed: {e}")
            return False
    
    def test_analytics_endpoints(self):
        """Test analytics endpoints."""
        print("🧪 Testing analytics endpoints...")
        try:
            # Test hourly analytics
            response = requests.get(f"{self.base_url}/analytics/hourly", timeout=5)
            if response.status_code == 200:
                print("✅ Hourly analytics endpoint working")
            else:
                print(f"❌ Hourly analytics failed: {response.status_code}")
                return False
            
            # Test daily analytics
            response = requests.get(f"{self.base_url}/analytics/daily", timeout=5)
            if response.status_code == 200:
                print("✅ Daily analytics endpoint working")
                return True
            else:
                print(f"❌ Daily analytics failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Analytics test failed: {e}")
            return False
    
    def test_alerts_endpoint(self):
        """Test alerts endpoint."""
        print("🧪 Testing alerts endpoint...")
        try:
            response = requests.get(f"{self.base_url}/alerts", timeout=5)
            if response.status_code == 200:
                alerts = response.json()
                print(f"✅ Alerts endpoint working: {len(alerts)} alerts available")
                return True
            else:
                print(f"❌ Alerts endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Alerts test failed: {e}")
            return False
    
    def test_streamlit_dashboard(self):
        """Test if Streamlit dashboard is accessible."""
        print("🧪 Testing Streamlit dashboard...")
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                print("✅ Streamlit dashboard is accessible")
                return True
            else:
                print(f"❌ Streamlit dashboard returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Streamlit dashboard is not running")
            return False
        except Exception as e:
            print(f"❌ Streamlit dashboard test failed: {e}")
            return False
    
    def test_react_frontend(self):
        """Test if React frontend is accessible."""
        print("🧪 Testing React frontend...")
        try:
            response = requests.get("http://localhost:3000", timeout=5)
            if response.status_code == 200:
                print("✅ React frontend is accessible")
                return True
            else:
                print(f"❌ React frontend returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ React frontend is not running")
            return False
        except Exception as e:
            print(f"❌ React frontend test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all dashboard tests."""
        print("🛡️ AI Fraud Detection Dashboard Test Suite")
        print("=" * 50)
        
        tests = [
            ("Model Loading", self.test_model_loading),
            ("FastAPI Server", self.test_fastapi_server),
            ("Transaction Prediction", self.test_transaction_prediction),
            ("Metrics Endpoint", self.test_metrics_endpoint),
            ("Models Endpoint", self.test_models_endpoint),
            ("Analytics Endpoints", self.test_analytics_endpoints),
            ("Alerts Endpoint", self.test_alerts_endpoint),
            ("Streamlit Dashboard", self.test_streamlit_dashboard),
            ("React Frontend", self.test_react_frontend),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                results.append((test_name, False))
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        print("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Dashboard is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
        
        return passed == total

def main():
    """Main function to run the test suite."""
    tester = DashboardTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Dashboard is ready to use!")
        print("\nAccess your dashboards at:")
        print("- FastAPI Server: http://localhost:8000")
        print("- API Documentation: http://localhost:8000/docs")
        print("- Streamlit Dashboard: http://localhost:8501")
        print("- React Frontend: http://localhost:3000")
    else:
        print("\n🔧 Please fix the failing tests before using the dashboard.")
        sys.exit(1)

if __name__ == "__main__":
    main() 