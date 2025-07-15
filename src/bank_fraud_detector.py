#!/usr/bin/env python3
"""
Bank-Level Fraud Detection System
Designed for real-world bank constraints with small datasets and domain-specific features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class BankFraudDetector:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.fraud_column = None
        self.customer_profiles = {}
        self.risk_thresholds = {}
        
    def load_bank_data(self, data_path, fraud_column='is_fraud'):
        """Load and validate bank transaction data."""
        print("Loading bank transaction data...")
        
        try:
            df = pd.read_csv(data_path)
            print(f"✓ Data loaded: {df.shape}")
            
            # Validate minimum requirements
            required_columns = ['transaction_id', 'amount', 'customer_id', fraud_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠ Missing required columns: {missing_columns}")
                print("Creating sample bank dataset...")
                df = self.create_sample_bank_dataset()
            
            self.fraud_column = fraud_column
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Creating sample bank dataset...")
            return self.create_sample_bank_dataset()
    
    def create_sample_bank_dataset(self):
        """Create a realistic sample bank dataset with real-world constraints."""
        print("Creating realistic bank transaction dataset...")
        
        np.random.seed(42)
        n_transactions = 5000  # Realistic small dataset
        n_customers = 500
        
        # Generate realistic bank data
        data = {
            'transaction_id': range(1, n_transactions + 1),
            'customer_id': np.random.randint(1, n_customers + 1, n_transactions),
            'amount': np.random.exponential(100, n_transactions),
            'transaction_type': np.random.choice(['ATM', 'POS', 'ONLINE', 'TRANSFER'], n_transactions),
            'merchant_category': np.random.choice(['RETAIL', 'FOOD', 'TRAVEL', 'UTILITIES', 'OTHER'], n_transactions),
            'hour': np.random.randint(0, 24, n_transactions),
            'day_of_week': np.random.randint(0, 7, n_transactions),
            'location': np.random.choice(['LOCAL', 'DOMESTIC', 'INTERNATIONAL'], n_transactions),
            'device_type': np.random.choice(['MOBILE', 'DESKTOP', 'ATM', 'POS'], n_transactions),
            'card_present': np.random.choice([True, False], n_transactions),
            'previous_fraud_flag': np.random.choice([True, False], n_transactions, p=[0.95, 0.05]),
            'account_age_days': np.random.randint(1, 3650, n_transactions),
            'balance_before': np.random.uniform(0, 10000, n_transactions),
            'balance_after': np.random.uniform(0, 10000, n_transactions),
        }
        
        df = pd.DataFrame(data)
        
        # Add realistic fraud patterns
        fraud_rate = 0.005  # 0.5% fraud rate (realistic for banks)
        fraud_indices = np.random.choice(n_transactions, int(n_transactions * fraud_rate), replace=False)
        
        # Create fraud column
        df['is_fraud'] = False
        df.loc[fraud_indices, 'is_fraud'] = True
        
        # Set fraud column name
        self.fraud_column = 'is_fraud'
        
        # Add realistic fraud patterns
        for idx in fraud_indices:
            # High-value transactions
            if np.random.random() < 0.3:
                df.loc[idx, 'amount'] = np.random.uniform(1000, 5000)
            
            # Unusual hours
            if np.random.random() < 0.4:
                df.loc[idx, 'hour'] = np.random.choice([1, 2, 3, 4, 5, 22, 23])
            
            # International transactions
            if np.random.random() < 0.3:
                df.loc[idx, 'location'] = 'INTERNATIONAL'
            
            # Card not present
            if np.random.random() < 0.6:
                df.loc[idx, 'card_present'] = False
        
        # Add missing values (realistic for bank data)
        missing_columns = ['merchant_category', 'device_type', 'balance_before', 'balance_after']
        for col in missing_columns:
            missing_indices = np.random.choice(len(df), int(len(df) * 0.1), replace=False)
            df.loc[missing_indices, col] = np.nan
        
        print(f"✓ Sample bank dataset created: {df.shape}")
        print(f"  Fraud rate: {df['is_fraud'].mean():.4f}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def engineer_bank_features(self, df):
        """Create bank-specific features for fraud detection."""
        print("Engineering bank-specific features...")
        
        # Time-based features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['is_high_value'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        # Transaction type features
        df['is_online'] = (df['transaction_type'] == 'ONLINE').astype(int)
        df['is_atm'] = (df['transaction_type'] == 'ATM').astype(int)
        df['is_international'] = (df['location'] == 'INTERNATIONAL').astype(int)
        df['card_not_present'] = (~df['card_present']).astype(int)
        
        # Customer behavior features
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'count'],
            'is_fraud': 'sum'
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'avg_amount', 'std_amount', 'transaction_count', 'fraud_count']
        customer_stats['fraud_rate'] = customer_stats['fraud_count'] / customer_stats['transaction_count']
        
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Risk indicators
        df['risk_score'] = (
            df['is_high_value'] * 2 +
            df['is_international'] * 3 +
            df['card_not_present'] * 2 +
            df['is_night'] * 1 +
            (df['previous_fraud_flag'] == True).astype(int) * 5
        )
        
        # Balance change
        df['balance_change'] = df['balance_after'] - df['balance_before']
        df['balance_change_pct'] = df['balance_change'] / (df['balance_before'] + 1)
        
        return df
    
    def handle_missing_data(self, df):
        """Handle missing values in bank data."""
        print("Handling missing data...")
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Remove target column
        if self.fraud_column in numerical_cols:
            numerical_cols.remove(self.fraud_column)
        
        # Impute numerical columns
        if numerical_cols:
            self.imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
        
        # Impute categorical columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'UNKNOWN')
        
        print(f"✓ Missing values handled: {df.isnull().sum().sum()} remaining")
        return df
    
    def create_customer_profiles(self, df):
        """Create customer risk profiles."""
        print("Creating customer risk profiles...")
        
        customer_profiles = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'min', 'max', 'count'],
            'is_fraud': ['sum', 'mean'],
            'transaction_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN',
            'location': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN',
            'hour': ['mean', 'std'],
            'risk_score': 'mean'
        }).reset_index()
        
        customer_profiles.columns = [
            'customer_id', 'avg_amount', 'std_amount', 'min_amount', 'max_amount', 'transaction_count',
            'fraud_count', 'fraud_rate', 'common_transaction_type', 'common_location',
            'avg_hour', 'std_hour', 'avg_risk_score'
        ]
        
        # Categorize customers by risk
        customer_profiles['risk_category'] = pd.cut(
            customer_profiles['avg_risk_score'],
            bins=[0, 2, 5, 10, 100],
            labels=['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
        )
        
        self.customer_profiles = customer_profiles.set_index('customer_id')
        
        print(f"✓ Customer profiles created for {len(customer_profiles)} customers")
        print(f"  Risk distribution: {customer_profiles['risk_category'].value_counts().to_dict()}")
        
        return customer_profiles
    
    def train_bank_models(self, df, test_size=0.2):
        """Train models specifically for bank fraud detection."""
        print("Training bank fraud detection models...")
        
        # Prepare features
        feature_cols = [
            'amount', 'amount_log', 'is_high_value', 'amount_percentile',
            'is_weekend', 'is_night', 'is_business_hours',
            'is_online', 'is_atm', 'is_international', 'card_not_present',
            'avg_amount', 'std_amount', 'transaction_count', 'fraud_rate',
            'risk_score', 'balance_change', 'balance_change_pct'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Prepare data
        X = df[available_features]
        y = df[self.fraud_column]
        
        # Handle missing values
        X = self.handle_missing_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Training fraud rate: {y_train.mean():.4f}")
        print(f"Test fraud rate: {y_test.mean():.4f}")
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Isolation Forest': IsolationForest(random_state=42, contamination=y_train.mean())
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Isolation Forest':
                # Isolation Forest predicts anomalies (-1) vs normal (1)
                model.fit(X_train_scaled)
                y_pred = model.predict(X_test_scaled)
                y_pred = (y_pred == -1).astype(int)  # Convert to fraud predictions
                y_pred_proba = model.decision_function(X_test_scaled)
                y_pred_proba = 1 - (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc': auc_score
            }
            
            print(f"  AUC Score: {auc_score:.4f}")
            
            # Classification report
            print(f"  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))
            
            self.models[name] = model
        
        return results, X_test_scaled, y_test
    
    def set_risk_thresholds(self, results, y_test):
        """Set risk thresholds for different alert levels."""
        print("Setting risk thresholds...")
        
        best_model = max(results.keys(), key=lambda x: results[x]['auc'])
        probabilities = results[best_model]['probabilities']
        
        # Calculate thresholds for different alert levels
        thresholds = {
            'high_risk': np.percentile(probabilities, 95),  # Top 5% risk
            'medium_risk': np.percentile(probabilities, 85),  # Top 15% risk
            'low_risk': np.percentile(probabilities, 70)   # Top 30% risk
        }
        
        self.risk_thresholds = thresholds
        
        print(f"Risk thresholds (using {best_model}):")
        for level, threshold in thresholds.items():
            print(f"  {level}: {threshold:.4f}")
        
        return thresholds
    
    def predict_transaction_risk(self, transaction_data):
        """Predict risk for a single transaction."""
        if not self.models or not self.feature_columns:
            print("❌ Models not trained. Please train models first.")
            return None
        
        # Prepare transaction data
        features = transaction_data[self.feature_columns]
        
        # Handle missing values
        if self.imputer:
            features = self.imputer.transform(features.values.reshape(1, -1))
        
        # Scale features
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            if name == 'Isolation Forest':
                pred = model.predict(features)[0]
                pred = (pred == -1).astype(int)
                prob = model.decision_function(features)[0]
                prob = 1 - (prob - model.decision_function(features).min()) / (model.decision_function(features).max() - model.decision_function(features).min())
            else:
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0, 1]
            
            predictions[name] = {'prediction': pred, 'probability': prob}
        
        # Determine risk level
        best_model = max(self.models.keys(), key=lambda x: predictions[x]['probability'])
        risk_prob = predictions[best_model]['probability']
        
        if risk_prob >= self.risk_thresholds['high_risk']:
            risk_level = 'HIGH_RISK'
        elif risk_prob >= self.risk_thresholds['medium_risk']:
            risk_level = 'MEDIUM_RISK'
        elif risk_prob >= self.risk_thresholds['low_risk']:
            risk_level = 'LOW_RISK'
        else:
            risk_level = 'SAFE'
        
        return {
            'risk_level': risk_level,
            'risk_probability': risk_prob,
            'model_predictions': predictions,
            'recommended_action': self.get_recommended_action(risk_level)
        }
    
    def get_recommended_action(self, risk_level):
        """Get recommended action based on risk level."""
        actions = {
            'HIGH_RISK': 'BLOCK_TRANSACTION',
            'MEDIUM_RISK': 'REQUIRE_ADDITIONAL_VERIFICATION',
            'LOW_RISK': 'MONITOR_CLOSELY',
            'SAFE': 'ALLOW_TRANSACTION'
        }
        return actions.get(risk_level, 'MONITOR_CLOSELY')
    
    def save_bank_model(self, filepath):
        """Save the trained bank fraud detection model."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'fraud_column': self.fraud_column,
            'customer_profiles': self.customer_profiles,
            'risk_thresholds': self.risk_thresholds
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Bank fraud detection model saved to {filepath}")
    
    def load_bank_model(self, filepath):
        """Load a trained bank fraud detection model."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.feature_columns = model_data['feature_columns']
        self.fraud_column = model_data['fraud_column']
        self.customer_profiles = model_data['customer_profiles']
        self.risk_thresholds = model_data['risk_thresholds']
        
        print(f"✓ Bank fraud detection model loaded from {filepath}")

def main():
    """Example usage of bank fraud detection system."""
    print("Bank-Level Fraud Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = BankFraudDetector()
    
    # Load or create bank data
    df = detector.load_bank_data("data/raw/bank_transactions.csv")
    
    # Engineer features
    df = detector.engineer_bank_features(df)
    
    # Create customer profiles
    customer_profiles = detector.create_customer_profiles(df)
    
    # Train models
    results, X_test, y_test = detector.train_bank_models(df)
    
    # Set risk thresholds
    thresholds = detector.set_risk_thresholds(results, y_test)
    
    # Save model
    detector.save_bank_model("models/bank_fraud_detector.pkl")
    
    print("\n🎉 Bank fraud detection system ready!")
    print("\nNext steps:")
    print("1. Use predict_transaction_risk() for real-time scoring")
    print("2. Monitor customer profiles for risk patterns")
    print("3. Adjust risk thresholds based on business needs")

if __name__ == "__main__":
    main() 