#!/usr/bin/env python3
"""
Test Bank-Level Fraud Detection System
Demonstrates real-world bank fraud detection with small datasets and domain-specific features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import bank fraud detector
import sys
sys.path.append('src')
from bank_fraud_detector import BankFraudDetector

def test_bank_data_creation():
    """Test realistic bank dataset creation."""
    print("=" * 60)
    print("1. REALISTIC BANK DATASET CREATION")
    print("=" * 60)
    
    detector = BankFraudDetector()
    df = detector.create_sample_bank_dataset()
    
    print(f"\nDataset Overview:")
    print(f"  Total transactions: {len(df):,}")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.4f}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
    # Show data quality issues
    print(f"\nData Quality Issues (Realistic for Banks):")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("  Missing values by column:")
        for col, missing in missing_data[missing_data > 0].items():
            print(f"    {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    # Show fraud patterns
    fraud_df = df[df['is_fraud'] == True]
    print(f"\nFraud Pattern Analysis:")
    high_value_fraud = (fraud_df['amount'] > 1000).sum()
    high_value_pct = (fraud_df['amount'] > 1000).mean() * 100
    print(f"  High-value fraud: {high_value_fraud} ({high_value_pct:.1f}%)")
    
    night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
    night_fraud = fraud_df['hour'].apply(lambda x: x in night_hours).sum()
    night_pct = fraud_df['hour'].apply(lambda x: x in night_hours).mean() * 100
    print(f"  Night-time fraud: {night_fraud} ({night_pct:.1f}%)")
    
    intl_fraud = (fraud_df['location'] == 'INTERNATIONAL').sum()
    intl_pct = (fraud_df['location'] == 'INTERNATIONAL').mean() * 100
    print(f"  International fraud: {intl_fraud} ({intl_pct:.1f}%)")
    
    cnp_fraud = (~fraud_df['card_present']).sum()
    cnp_pct = (~fraud_df['card_present']).mean() * 100
    print(f"  Card-not-present fraud: {cnp_fraud} ({cnp_pct:.1f}%)")
    
    return df, detector

def test_feature_engineering(df, detector):
    """Test bank-specific feature engineering."""
    print("\n" + "=" * 60)
    print("2. BANK-SPECIFIC FEATURE ENGINEERING")
    print("=" * 60)
    
    df_enhanced = detector.engineer_bank_features(df)
    
    print(f"\nEnhanced Dataset:")
    print(f"  Original features: {len(df.columns)}")
    print(f"  Enhanced features: {len(df_enhanced.columns)}")
    print(f"  New features added: {len(df_enhanced.columns) - len(df.columns)}")
    
    # Show new features
    original_cols = set(df.columns)
    new_features = [col for col in df_enhanced.columns if col not in original_cols]
    print(f"\nNew Features Created:")
    for i, feature in enumerate(new_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # Show customer behavior features
    customer_features = ['avg_amount', 'std_amount', 'transaction_count', 'fraud_rate']
    print(f"\nCustomer Behavior Features:")
    for feature in customer_features:
        if feature in df_enhanced.columns:
            print(f"  {feature}: {df_enhanced[feature].describe()['mean']:.2f} (mean)")
    
    return df_enhanced

def test_customer_profiling(df, detector):
    """Test customer risk profiling."""
    print("\n" + "=" * 60)
    print("3. CUSTOMER RISK PROFILING")
    print("=" * 60)
    
    customer_profiles = detector.create_customer_profiles(df)
    
    print(f"\nCustomer Risk Distribution:")
    risk_dist = customer_profiles['risk_category'].value_counts()
    for category, count in risk_dist.items():
        percentage = count / len(customer_profiles) * 100
        print(f"  {category}: {count} customers ({percentage:.1f}%)")
    
    # Show high-risk customers
    high_risk = customer_profiles[customer_profiles['risk_category'] == 'HIGH']
    very_high_risk = customer_profiles[customer_profiles['risk_category'] == 'VERY_HIGH']
    
    print(f"\nHigh-Risk Customer Analysis:")
    print(f"  High risk customers: {len(high_risk)}")
    print(f"  Very high risk customers: {len(very_high_risk)}")
    
    if len(high_risk) > 0:
        print(f"  Average fraud rate (high risk): {high_risk['fraud_rate'].mean():.4f}")
        print(f"  Average transaction count (high risk): {high_risk['transaction_count'].mean():.1f}")
    
    return customer_profiles

def test_model_training(df, detector):
    """Test bank fraud model training."""
    print("\n" + "=" * 60)
    print("4. BANK FRAUD MODEL TRAINING")
    print("=" * 60)
    
    results, X_test, y_test = detector.train_bank_models(df)
    
    print(f"\nModel Performance Summary:")
    for name, result in results.items():
        print(f"  {name}: AUC = {result['auc']:.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['auc'])
    print(f"\nBest performing model: {best_model}")
    print(f"Best AUC Score: {results[best_model]['auc']:.4f}")
    
    return results, X_test, y_test

def test_risk_thresholds(results, y_test, detector):
    """Test risk threshold setting."""
    print("\n" + "=" * 60)
    print("5. RISK THRESHOLD SETTING")
    print("=" * 60)
    
    thresholds = detector.set_risk_thresholds(results, y_test)
    
    print(f"\nRisk Thresholds:")
    for level, threshold in thresholds.items():
        print(f"  {level}: {threshold:.4f}")
    
    # Test threshold effectiveness
    best_model = max(results.keys(), key=lambda x: results[x]['auc'])
    probabilities = results[best_model]['probabilities']
    
    print(f"\nThreshold Effectiveness:")
    for level, threshold in thresholds.items():
        above_threshold = (probabilities >= threshold).sum()
        percentage = above_threshold / len(probabilities) * 100
        print(f"  {level}: {above_threshold} transactions ({percentage:.1f}%)")
    
    return thresholds

def test_real_time_prediction(detector):
    """Test real-time transaction risk prediction."""
    print("\n" + "=" * 60)
    print("6. REAL-TIME TRANSACTION RISK PREDICTION")
    print("=" * 60)
    
    # Create sample transactions for testing
    sample_transactions = [
        {
            'amount': 50, 'amount_log': 3.91, 'is_high_value': 0, 'amount_percentile': 0.3,
            'is_weekend': 0, 'is_night': 0, 'is_business_hours': 1,
            'is_online': 0, 'is_atm': 1, 'is_international': 0, 'card_not_present': 0,
            'avg_amount': 75, 'std_amount': 25, 'transaction_count': 15, 'fraud_rate': 0.0,
            'risk_score': 0, 'balance_change': -50, 'balance_change_pct': -0.05
        },
        {
            'amount': 2500, 'amount_log': 7.82, 'is_high_value': 1, 'amount_percentile': 0.95,
            'is_weekend': 1, 'is_night': 1, 'is_business_hours': 0,
            'is_online': 1, 'is_atm': 0, 'is_international': 1, 'card_not_present': 1,
            'avg_amount': 100, 'std_amount': 50, 'transaction_count': 8, 'fraud_rate': 0.0,
            'risk_score': 12, 'balance_change': -2500, 'balance_change_pct': -0.25
        },
        {
            'amount': 150, 'amount_log': 5.01, 'is_high_value': 0, 'amount_percentile': 0.6,
            'is_weekend': 0, 'is_night': 0, 'is_business_hours': 1,
            'is_online': 0, 'is_atm': 0, 'is_international': 0, 'card_not_present': 0,
            'avg_amount': 120, 'std_amount': 30, 'transaction_count': 25, 'fraud_rate': 0.0,
            'risk_score': 1, 'balance_change': -150, 'balance_change_pct': -0.12
        }
    ]
    
    transaction_types = ['Normal ATM Withdrawal', 'Suspicious Online Purchase', 'Regular POS Transaction']
    
    print(f"Testing Real-Time Predictions:")
    for i, (transaction, trans_type) in enumerate(zip(sample_transactions, transaction_types), 1):
        print(f"\nTransaction {i}: {trans_type}")
        print(f"  Amount: ${transaction['amount']}")
        print(f"  Type: {'Online' if transaction['is_online'] else 'ATM' if transaction['is_atm'] else 'POS'}")
        print(f"  Location: {'International' if transaction['is_international'] else 'Local'}")
        print(f"  Time: {'Night' if transaction['is_night'] else 'Day'}")
        
        # Predict risk
        risk_assessment = detector.predict_transaction_risk(pd.DataFrame([transaction]))
        
        if risk_assessment:
            print(f"  Risk Level: {risk_assessment['risk_level']}")
            print(f"  Risk Probability: {risk_assessment['risk_probability']:.4f}")
            print(f"  Recommended Action: {risk_assessment['recommended_action']}")
            
            # Show model predictions
            print(f"  Model Predictions:")
            for model_name, pred in risk_assessment['model_predictions'].items():
                print(f"    {model_name}: {pred['prediction']} (prob: {pred['probability']:.4f})")

def test_model_persistence(detector):
    """Test model saving and loading."""
    print("\n" + "=" * 60)
    print("7. MODEL PERSISTENCE")
    print("=" * 60)
    
    # Save model
    model_path = "models/bank_fraud_detector.pkl"
    detector.save_bank_model(model_path)
    
    # Load model
    new_detector = BankFraudDetector()
    new_detector.load_bank_model(model_path)
    
    print(f"✓ Model saved and loaded successfully")
    print(f"✓ Models available: {list(new_detector.models.keys())}")
    print(f"✓ Features: {len(new_detector.feature_columns)}")
    print(f"✓ Risk thresholds: {len(new_detector.risk_thresholds)}")

def create_visualizations(df, results, customer_profiles):
    """Create visualizations for bank fraud detection."""
    print("\n" + "=" * 60)
    print("8. CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Fraud distribution by transaction type
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    fraud_by_type = df.groupby('transaction_type')['is_fraud'].mean().sort_values(ascending=False)
    fraud_by_type.plot(kind='bar', color='red', alpha=0.7)
    plt.title('Fraud Rate by Transaction Type')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=45)
    
    # 2. Fraud distribution by location
    plt.subplot(2, 2, 2)
    fraud_by_location = df.groupby('location')['is_fraud'].mean().sort_values(ascending=False)
    fraud_by_location.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Fraud Rate by Location')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=45)
    
    # 3. Amount distribution for fraud vs legitimate
    plt.subplot(2, 2, 3)
    df[df['is_fraud'] == 0]['amount'].hist(bins=50, alpha=0.7, label='Legitimate', color='green')
    df[df['is_fraud'] == 1]['amount'].hist(bins=50, alpha=0.7, label='Fraudulent', color='red')
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.yscale('log')
    
    # 4. Customer risk distribution
    plt.subplot(2, 2, 4)
    customer_profiles['risk_category'].value_counts().plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Customer Risk Distribution')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('bank_fraud_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved bank fraud analysis visualization: bank_fraud_analysis.png")

def main():
    """Main testing function for bank fraud detection."""
    print("Bank-Level Fraud Detection System Testing")
    print("=" * 60)
    
    # Test 1: Data creation
    df, detector = test_bank_data_creation()
    
    # Test 2: Feature engineering
    df_enhanced = test_feature_engineering(df, detector)
    
    # Test 3: Customer profiling
    customer_profiles = test_customer_profiling(df_enhanced, detector)
    
    # Test 4: Model training
    results, X_test, y_test = test_model_training(df_enhanced, detector)
    
    # Test 5: Risk thresholds
    thresholds = test_risk_thresholds(results, y_test, detector)
    
    # Test 6: Real-time prediction
    test_real_time_prediction(detector)
    
    # Test 7: Model persistence
    test_model_persistence(detector)
    
    # Test 8: Visualizations
    create_visualizations(df_enhanced, results, customer_profiles)
    
    # Summary
    print("\n" + "=" * 60)
    print("BANK FRAUD DETECTION TESTING SUMMARY")
    print("=" * 60)
    print(f"✓ Dataset: {len(df):,} transactions, {df['customer_id'].nunique():,} customers")
    print(f"✓ Fraud rate: {df['is_fraud'].mean():.4f}")
    print(f"✓ Features engineered: {len(df_enhanced.columns) - len(df.columns)}")
    print(f"✓ Models trained: {len(results)}")
    print(f"✓ Best AUC: {max(results.values(), key=lambda x: x['auc'])['auc']:.4f}")
    print(f"✓ Customer profiles: {len(customer_profiles)}")
    print(f"✓ Risk thresholds: {len(thresholds)}")
    
    print(f"\n🎉 Bank fraud detection system tested successfully!")
    print(f"\nKey Advantages for Real Bank Use:")
    print(f"1. Handles small datasets (5K transactions)")
    print(f"2. Manages missing data (realistic for banks)")
    print(f"3. Domain-specific features (customer behavior, risk scores)")
    print(f"4. Real-time risk assessment with actionable recommendations")
    print(f"5. Customer profiling for risk management")
    print(f"6. Configurable risk thresholds for business needs")

if __name__ == "__main__":
    main() 