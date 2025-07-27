#!/usr/bin/env python3
"""
Kaggle Dataset Integration for Fraud Detection Platform
Downloads and processes real fraud detection datasets
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
import streamlit as st

class KaggleDataIntegrator:
    def __init__(self):
        self.data_dir = Path("data/kaggle")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_credit_card_fraud(self):
        """Download Credit Card Fraud Detection dataset"""
        print("📊 Downloading Credit Card Fraud Detection dataset...")
        
        # Dataset URL (you'll need to download manually from Kaggle)
        dataset_url = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        
        print(f"Please download the dataset from: {dataset_url}")
        print("1. Go to the URL above")
        print("2. Click 'Download'")
        print("3. Extract the CSV file to data/kaggle/")
        
        # Check if file exists
        csv_path = self.data_dir / "creditcard.csv"
        if csv_path.exists():
            print("✅ Credit card dataset found!")
            return self.process_credit_card_data(csv_path)
        else:
            print("❌ Dataset not found. Please download manually.")
            return None
    
    def process_credit_card_data(self, csv_path):
        """Process credit card fraud dataset"""
        print("🔄 Processing credit card fraud data...")
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df):,} transactions")
            
            # Basic statistics
            fraud_count = df['Class'].sum()
            total_count = len(df)
            fraud_rate = (fraud_count / total_count) * 100
            
            print(f"📊 Dataset Statistics:")
            print(f"  - Total transactions: {total_count:,}")
            print(f"  - Fraudulent transactions: {fraud_count:,}")
            print(f"  - Fraud rate: {fraud_rate:.3f}%")
            print(f"  - Features: {len(df.columns)}")
            
            # Save processed data
            processed_path = self.data_dir / "creditcard_processed.csv"
            df.to_csv(processed_path, index=False)
            print(f"✅ Processed data saved to: {processed_path}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing data: {e}")
            return None
    
    def create_sample_transactions(self, n=1000):
        """Create sample transactions for testing"""
        print("🔄 Creating sample transaction data...")
        
        np.random.seed(42)
        
        # Generate realistic transaction data
        transactions = []
        for i in range(n):
            # Determine if this is a fraudulent transaction
            is_fraud = np.random.random() < 0.01  # 1% fraud rate
            
            transaction = {
                'transaction_id': f'TXN_{i+1:06d}',
                'customer_id': f'CUST_{np.random.randint(1, 1001):04d}',
                'amount': np.random.exponential(100) if not is_fraud else np.random.uniform(500, 5000),
                'transaction_type': np.random.choice(['ATM', 'POS', 'ONLINE', 'TRANSFER']),
                'merchant_category': np.random.choice(['RETAIL', 'FOOD', 'TRAVEL', 'UTILITIES', 'E-COMMERCE']),
                'location': np.random.choice(['LOCAL', 'DOMESTIC', 'INTERNATIONAL']),
                'device_type': np.random.choice(['MOBILE', 'DESKTOP', 'ATM', 'POS']),
                'card_present': np.random.choice([True, False]),
                'hour': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'is_fraud': is_fraud,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=np.random.randint(0, 1440))
            }
            
            # Add fraud patterns
            if is_fraud:
                transaction['amount'] = np.random.uniform(1000, 5000)
                transaction['location'] = 'INTERNATIONAL'
                transaction['card_present'] = False
                transaction['hour'] = np.random.choice([1, 2, 3, 4, 5, 22, 23])
            
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Save sample data
        sample_path = self.data_dir / "sample_transactions.csv"
        df.to_csv(sample_path, index=False)
        
        print(f"✅ Created {len(df):,} sample transactions")
        print(f"  - Fraudulent: {df['is_fraud'].sum():,}")
        print(f"  - Fraud rate: {(df['is_fraud'].mean() * 100):.2f}%")
        print(f"  - Saved to: {sample_path}")
        
        return df
    
    def test_with_real_data(self):
        """Test the fraud detection platform with real data"""
        print("🧪 Testing fraud detection platform with real data...")
        
        # Try to load real dataset
        real_data = self.download_credit_card_fraud()
        
        if real_data is None:
            print("📝 Using sample data for testing...")
            test_data = self.create_sample_transactions(5000)
        else:
            print("📊 Using real Kaggle dataset for testing...")
            test_data = real_data
        
        return test_data

def main():
    """Main function to integrate Kaggle data"""
    print("🚀 Kaggle Dataset Integration for Fraud Detection Platform")
    print("=" * 60)
    
    integrator = KaggleDataIntegrator()
    
    # Test with real data
    test_data = integrator.test_with_real_data()
    
    if test_data is not None:
        print("\n✅ Data integration completed successfully!")
        print(f"📊 Ready to test with {len(test_data):,} transactions")
        
        # Show sample data
        print("\n📋 Sample Data Preview:")
        print(test_data.head())
        
        print("\n🎯 Next Steps:")
        print("1. Run your Streamlit dashboard")
        print("2. Navigate to 'Transaction Monitor' tab")
        print("3. Upload the processed data")
        print("4. Test fraud detection algorithms")
        
    else:
        print("❌ Data integration failed. Please check the dataset.")

if __name__ == "__main__":
    main() 