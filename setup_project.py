#!/usr/bin/env python3
"""
Setup script for the Fraud Detection Project
Creates necessary directories and sample dataset for immediate testing
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "notebooks",
        "src"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_sample_dataset():
    """Create a synthetic fraud dataset for testing."""
    print("\nCreating sample fraud dataset...")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic features similar to credit card fraud dataset
    features = {}
    
    # V1-V28 (anonymized features)
    for i in range(28):
        features[f'V{i+1}'] = np.random.normal(0, 1, n_samples)
    
    # Amount (exponential distribution)
    features['Amount'] = np.random.exponential(100, n_samples)
    
    # Time (uniform distribution)
    features['Time'] = np.random.uniform(0, 172792, n_samples)
    
    # Create fraud labels (imbalanced dataset - 0.17% fraud rate)
    fraud_rate = 0.0017
    features['Class'] = np.random.binomial(1, fraud_rate, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(features)
    
    # Save to file
    output_path = "data/raw/creditcard_sample.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Sample dataset created: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Fraud rate: {df['Class'].mean():.4f}")
    print(f"  Legitimate transactions: {df['Class'].value_counts()[0]:,}")
    print(f"  Fraudulent transactions: {df['Class'].value_counts()[1]:,}")
    
    return output_path

def test_imports():
    """Test if all required packages can be imported."""
    print("\nTesting package imports...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    optional_packages = [
        'xgboost',
        'lightgbm',
        'plotly',
        'imblearn',
        'optuna'
    ]
    
    print("Required packages:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            print(f"    Install with: pip install {package}")
    
    print("\nOptional packages:")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} - NOT FOUND (optional)")
            print(f"    Install with: pip install {package}")

def create_test_script():
    """Create a simple test script to verify the setup."""
    test_script = '''#!/usr/bin/env python3
"""
Quick test script to verify the fraud detection setup
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_data_loading():
    """Test if the sample dataset can be loaded."""
    try:
        df = pd.read_csv("data/raw/creditcard_sample.csv")
        print(f"✓ Data loaded successfully! Shape: {df.shape}")
        print(f"✓ Fraud rate: {df['Class'].mean():.4f}")
        return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def test_basic_analysis():
    """Test basic data analysis."""
    try:
        df = pd.read_csv("data/raw/creditcard_sample.csv")
        
        # Basic statistics
        print("\\nBasic Statistics:")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Features: {len(df.columns)}")
        print(f"  Fraud rate: {df['Class'].mean():.4f}")
        
        # Feature analysis
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features.remove('Class')
        
        correlations = df[numerical_features].corrwith(df['Class']).abs().sort_values(ascending=False)
        print(f"\\nTop 5 features by correlation with fraud:")
        for i, (feature, corr) in enumerate(correlations.head().items(), 1):
            print(f"  {i}. {feature}: {corr:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error in analysis: {e}")
        return False

if __name__ == "__main__":
    print("Testing Fraud Detection Setup")
    print("=" * 30)
    
    success = True
    success &= test_data_loading()
    success &= test_basic_analysis()
    
    if success:
        print("\\n🎉 All tests passed! Setup is complete.")
        print("\\nNext steps:")
        print("1. Run: python data_exploration.py")
        print("2. Run: python train_model.py")
        print("3. Or start Jupyter: jupyter notebook notebooks/fraud_detection_workflow.ipynb")
    else:
        print("\\n❌ Some tests failed. Please check the setup.")
'''
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    print("✓ Created test script: test_setup.py")

def main():
    """Main setup function."""
    print("Fraud Detection Project Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Create sample dataset
    create_sample_dataset()
    
    # Test imports
    test_imports()
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 40)
    print("SETUP COMPLETE!")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Test the setup: python test_setup.py")
    print("2. Explore data: python data_exploration.py")
    print("3. Train models: python train_model.py")
    print("4. Interactive analysis: jupyter notebook notebooks/fraud_detection_workflow.ipynb")
    print("\nFor real datasets:")
    print("- Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("- Or use: python src/data_downloader.py")
    print("\nHappy fraud detection! 🕵️‍♂️")

if __name__ == "__main__":
    main() 