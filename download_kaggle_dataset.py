#!/usr/bin/env python3
"""
Download Kaggle Fraud Detection Dataset
Provides multiple options for downloading the dataset
"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
import subprocess
import sys

def create_directories():
    """Create necessary directories."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    print("✓ Created data/raw directory")

def download_with_kaggle_api():
    """Download using Kaggle API if credentials are set up."""
    try:
        # Check if kaggle credentials exist
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            print("❌ Kaggle credentials not found.")
            print("To set up Kaggle API:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Scroll to 'API' section and click 'Create New API Token'")
            print("3. Download kaggle.json and place it in ~/.kaggle/")
            print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        print("✓ Kaggle credentials found. Downloading dataset...")
        
        # Download the credit card fraud dataset
        result = subprocess.run([
            sys.executable, "-m", "kaggle", "datasets", "download", 
            "mlg-ulb/creditcardfraud", "-p", "data/raw", "--unzip"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Dataset downloaded successfully!")
            return True
        else:
            print(f"❌ Error downloading: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error with Kaggle API: {e}")
        return False

def download_direct_link():
    """Download from direct link (if available)."""
    print("Attempting direct download...")
    
    # Note: This is a placeholder. Direct download links may not be available
    # due to Kaggle's terms of service
    print("Direct download not available due to Kaggle's terms of service.")
    print("Please use manual download method.")
    return False

def provide_manual_instructions():
    """Provide detailed manual download instructions."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\n1. Visit the Credit Card Fraud Detection dataset:")
    print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("\n2. Click 'Download' button (requires Kaggle account)")
    print("\n3. Extract the downloaded zip file")
    print("\n4. Copy 'creditcard.csv' to 'data/raw/' directory")
    print("\n5. Verify the file structure:")
    print("   data/raw/creditcard.csv")
    print("\n6. Run the fraud detection scripts:")
    print("   python3 data_exploration.py")
    print("   python3 train_model.py")
    
    # Check if file already exists
    if Path("data/raw/creditcard.csv").exists():
        print("\n✓ Found creditcard.csv in data/raw/")
        print("You can now run the fraud detection scripts!")
        return True
    else:
        print("\n❌ creditcard.csv not found in data/raw/")
        print("Please follow the manual download instructions above.")
        return False

def verify_dataset():
    """Verify that the dataset is properly downloaded and formatted."""
    csv_path = Path("data/raw/creditcard.csv")
    
    if not csv_path.exists():
        print("❌ creditcard.csv not found")
        return False
    
    try:
        # Load a small sample to verify format
        df = pd.read_csv(csv_path, nrows=1000)
        
        print("✓ Dataset verification:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Fraud column: {'Class' in df.columns}")
        print(f"  Fraud rate: {df['Class'].mean():.4f}")
        
        # Check for expected columns
        expected_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  ⚠ Missing columns: {missing_columns}")
        else:
            print("  ✓ All expected columns present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False

def main():
    """Main function to download the dataset."""
    print("Fraud Detection Dataset Downloader")
    print("=" * 40)
    
    create_directories()
    
    # Try Kaggle API first
    print("\n1. Trying Kaggle API download...")
    if download_with_kaggle_api():
        if verify_dataset():
            print("\n🎉 Dataset downloaded and verified successfully!")
            print("You can now run:")
            print("  python3 data_exploration.py")
            print("  python3 train_model.py")
            return
    
    # Try direct download
    print("\n2. Trying direct download...")
    if download_direct_link():
        if verify_dataset():
            print("\n🎉 Dataset downloaded successfully!")
            return
    
    # Provide manual instructions
    print("\n3. Manual download required...")
    provide_manual_instructions()
    
    # Final verification
    print("\n" + "=" * 40)
    print("FINAL VERIFICATION")
    print("=" * 40)
    
    if verify_dataset():
        print("\n🎉 Dataset is ready!")
        print("\nNext steps:")
        print("1. Explore data: python3 data_exploration.py")
        print("2. Train models: python3 train_model.py")
        print("3. Interactive analysis: jupyter notebook notebooks/fraud_detection_workflow.ipynb")
    else:
        print("\n❌ Dataset not ready. Please follow the manual download instructions.")

if __name__ == "__main__":
    main() 